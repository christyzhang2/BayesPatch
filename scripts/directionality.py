import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from bayespatch.data import get_image, load_aptos
from bayespatch.model import (
    DEFAULT_PROMPTS,
    encode_prompts,
    load_biomedclip,
)
from bayespatch.patching import (
    capture_block_activation,
    run_patched_margin,
)


SEED = 0

PAIRING_STRATEGIES = [
    "top10_extreme",
    "top5_extreme",
    "medium10",
]

CONFIGURATIONS = [
    {
        "config": "L11 CLS",
        "layer": 11,
        "region": "cls",
        "alpha": 1.0,
    },
    {
        "config": "L1 center",
        "layer": 1,
        "region": "center",
        "alpha": 1.0,
    },
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run BayesPatch directionality controls and "
            "donor-target selection robustness."
        )
    )

    parser.add_argument(
        "--margins",
        type=Path,
        default=Path("results/subset_margins.csv"),
    )

    parser.add_argument(
        "--pair-output",
        type=Path,
        default=Path(
            "results/donor_target_robustness_pair_deltas.csv"
        ),
    )

    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path(
            "results/donor_target_robustness_summary.csv"
        ),
    )

    parser.add_argument(
        "--strategy",
        choices=PAIRING_STRATEGIES,
        default=None,
        help=(
            "Run only one pairing strategy. "
            "If omitted, run all three strategies."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
    )

    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=3000,
    )

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def bootstrap_ci(values, n_boot=3000, seed=0):
    values = np.asarray(values, dtype=float)

    if len(values) == 0:
        raise ValueError("Cannot bootstrap an empty array")

    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)

    for index in range(n_boot):
        sample = rng.choice(
            values,
            size=len(values),
            replace=True,
        )
        means[index] = sample.mean()

    return (
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
    )


def select_pairing_groups(subset_df, strategy):
    dr_df = subset_df[
        subset_df["binary_dr"] == 1
    ].copy()

    no_dr_df = subset_df[
        subset_df["binary_dr"] == 0
    ].copy()

    dr_sorted_high = (
        dr_df
        .sort_values("margin", ascending=False)
        .reset_index(drop=True)
    )

    dr_sorted_mid = (
        dr_df
        .sort_values("margin", ascending=True)
        .reset_index(drop=True)
    )

    no_dr_sorted_low = (
        no_dr_df
        .sort_values("margin", ascending=True)
        .reset_index(drop=True)
    )

    no_dr_sorted_mid = (
        no_dr_df
        .sort_values("margin", ascending=True)
        .reset_index(drop=True)
    )

    if strategy == "top10_extreme":
        n = 10

        dr_donors = dr_sorted_high.head(n)
        no_dr_targets = no_dr_sorted_low.head(n)
        no_dr_donors = no_dr_sorted_low.iloc[n : 2 * n]
        dr_targets = dr_sorted_high.head(n)

    elif strategy == "top5_extreme":
        n = 5

        dr_donors = dr_sorted_high.head(n)
        no_dr_targets = no_dr_sorted_low.head(n)
        no_dr_donors = no_dr_sorted_low.iloc[n : 2 * n]
        dr_targets = dr_sorted_high.head(n)

    elif strategy == "medium10":
        n = 10

        dr_start = max(
            0,
            len(dr_sorted_mid) // 2 - n // 2,
        )

        no_dr_start = max(
            0,
            len(no_dr_sorted_mid) // 2 - n // 2,
        )

        dr_mid = dr_sorted_mid.iloc[
            dr_start : dr_start + n
        ]

        no_dr_mid = no_dr_sorted_mid.iloc[
            no_dr_start : no_dr_start + n
        ]

        dr_donors = dr_mid
        dr_targets = dr_mid
        no_dr_targets = no_dr_mid

        no_dr_donors = no_dr_sorted_mid.iloc[
            no_dr_start + n : no_dr_start + 2 * n
        ]

        if len(no_dr_donors) < n:
            no_dr_donors = no_dr_sorted_mid.iloc[
                no_dr_start - n : no_dr_start
            ]

    else:
        raise ValueError(
            f"Unknown pairing strategy: {strategy}"
        )

    groups = {
        "dr_donors": dr_donors.reset_index(drop=True),
        "no_dr_targets": no_dr_targets.reset_index(drop=True),
        "no_dr_donors": no_dr_donors.reset_index(drop=True),
        "dr_targets": dr_targets.reset_index(drop=True),
    }

    for group_name, group in groups.items():
        if len(group) != n:
            raise ValueError(
                f"{strategy} {group_name}: "
                f"expected {n} rows, found {len(group)}"
            )

    return groups


def prepare_rows(
    rows,
    dataset,
    preprocess,
    device,
    description,
):
    prepared = []

    for row in tqdm(
        rows.itertuples(index=False),
        total=len(rows),
        desc=description,
    ):
        image = get_image(
            dataset,
            row.dataset_index,
        )

        tensor = (
            preprocess(image)
            .unsqueeze(0)
            .to(device)
        )

        prepared.append(
            {
                "subset_index": int(row.subset_index),
                "dataset_index": int(row.dataset_index),
                "diagnosis": int(row.diagnosis),
                "binary_dr": int(row.binary_dr),
                "margin": float(row.margin),
                "tensor": tensor,
            }
        )

    return prepared


def evaluate_control(
    model,
    text_features,
    pairing_strategy,
    control,
    configuration,
    donors,
    targets,
    seed,
):
    rows = []

    layer = int(configuration["layer"])
    region = str(configuration["region"])
    alpha = float(configuration["alpha"])
    config_name = str(configuration["config"])

    description = (
        f"{pairing_strategy} | "
        f"{config_name} | "
        f"{control}"
    )

    for donor_position, donor in enumerate(
        tqdm(donors, desc=description)
    ):
        donor_activation = capture_block_activation(
            model=model,
            image_tensor=donor["tensor"],
            layer_idx=layer,
        )

        for target_position, target in enumerate(targets):
            pair_seed = (
                seed
                + donor_position * 100
                + target_position
            )

            patched_margin = run_patched_margin(
                model=model,
                target_tensor=target["tensor"],
                donor_activation=donor_activation,
                text_features=text_features,
                layer_idx=layer,
                region=region,
                alpha=alpha,
                random_seed=pair_seed,
            )

            delta = patched_margin - target["margin"]

            rows.append(
                {
                    "pairing_strategy": pairing_strategy,
                    "control": control,
                    "config": config_name,
                    "layer": layer,
                    "region": region,
                    "alpha": alpha,
                    "donor_position": donor_position,
                    "donor_index": donor["subset_index"],
                    "donor_dataset_index": donor["dataset_index"],
                    "donor_diagnosis": donor["diagnosis"],
                    "target_position": target_position,
                    "target_index": target["subset_index"],
                    "target_dataset_index": target["dataset_index"],
                    "target_diagnosis": target["diagnosis"],
                    "base_margin": target["margin"],
                    "patched_margin": patched_margin,
                    "delta": delta,
                }
            )

    return pd.DataFrame(rows)


def summarize_controls(
    pair_df,
    bootstrap_samples,
    seed,
):
    summary_rows = []

    group_columns = [
        "pairing_strategy",
        "config",
        "control",
        "layer",
        "region",
        "alpha",
    ]

    for group_number, (keys, group) in enumerate(
        pair_df.groupby(
            group_columns,
            sort=True,
        )
    ):
        values = group["delta"].to_numpy(dtype=float)

        ci_low, ci_high = bootstrap_ci(
            values=values,
            n_boot=bootstrap_samples,
            seed=seed + group_number,
        )

        summary = dict(
            zip(group_columns, keys)
        )

        summary.update(
            {
                "n_pairs": int(len(values)),
                "mean_delta": float(values.mean()),
                "std_delta": float(
                    values.std(ddof=1)
                ),
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )

        summary_rows.append(summary)

    return pd.DataFrame(summary_rows)


def save_csv(df, path):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    df.to_csv(
        temporary,
        index=False,
    )

    temporary.replace(path)


def main():
    args = parse_args()
    set_seed(args.seed)

    if not args.margins.exists():
        raise FileNotFoundError(
            f"Missing margins file: {args.margins}"
        )

    margins_df = pd.read_csv(args.margins)

    required_columns = {
        "subset_index",
        "dataset_index",
        "diagnosis",
        "binary_dr",
        "margin",
    }

    missing = (
        required_columns
        - set(margins_df.columns)
    )

    if missing:
        raise ValueError(
            f"Margins file is missing: {sorted(missing)}"
        )

    strategies = (
        [args.strategy]
        if args.strategy is not None
        else PAIRING_STRATEGIES
    )

    print("Strategies:", strategies)
    print("Configurations:", CONFIGURATIONS)

    dataset = load_aptos()

    model, preprocess, tokenizer, device = (
        load_biomedclip()
    )

    print("Device:", device)

    text_features = encode_prompts(
        model=model,
        tokenizer=tokenizer,
        prompts=DEFAULT_PROMPTS,
        device=device,
    )

    pair_frames = []

    for strategy in strategies:
        print()
        print("=" * 80)
        print("Pairing strategy:", strategy)
        print("=" * 80)

        groups = select_pairing_groups(
            margins_df,
            strategy,
        )

        for name, group in groups.items():
            print(
                name,
                "rows:",
                len(group),
                "mean margin:",
                float(group["margin"].mean()),
            )

        prepared = {}

        for group_name, group_df in groups.items():
            prepared[group_name] = prepare_rows(
                rows=group_df,
                dataset=dataset,
                preprocess=preprocess,
                device=device,
                description=(
                    f"{strategy}: preparing {group_name}"
                ),
            )

        scenarios = [
            {
                "control": "positive_dr_to_no_dr",
                "donors": prepared["dr_donors"],
                "targets": prepared["no_dr_targets"],
            },
            {
                "control": "control_no_dr_to_no_dr",
                "donors": prepared["no_dr_donors"],
                "targets": prepared["no_dr_targets"],
            },
            {
                "control": "reverse_no_dr_to_dr",
                "donors": prepared["no_dr_donors"],
                "targets": prepared["dr_targets"],
            },
        ]

        for configuration in CONFIGURATIONS:
            for scenario in scenarios:
                pair_frame = evaluate_control(
                    model=model,
                    text_features=text_features,
                    pairing_strategy=strategy,
                    control=scenario["control"],
                    configuration=configuration,
                    donors=scenario["donors"],
                    targets=scenario["targets"],
                    seed=args.seed,
                )

                pair_frames.append(pair_frame)

    pair_df = pd.concat(
        pair_frames,
        ignore_index=True,
    )

    expected_rows = (
        600
        if strategies == ["top10_extreme"]
        else 150
        if strategies == ["top5_extreme"]
        else 600
        if strategies == ["medium10"]
        else 1350
    )

    print()
    print("Pair rows:", len(pair_df))
    print("Expected pair rows:", expected_rows)

    if len(pair_df) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} pair rows, "
            f"found {len(pair_df)}"
        )

    summary_df = summarize_controls(
        pair_df=pair_df,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )

    save_csv(
        pair_df,
        args.pair_output,
    )

    save_csv(
        summary_df,
        args.summary_output,
    )

    print()
    print("Saved pair results:", args.pair_output)
    print("Saved summary:", args.summary_output)

    print()
    print("Directionality summary:")
    print(
        summary_df
        .sort_values(
            [
                "pairing_strategy",
                "config",
                "control",
            ]
        )
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()

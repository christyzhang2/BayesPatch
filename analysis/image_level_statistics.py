import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon


CONFIGURATIONS = [
    "L11 CLS",
    "L1 center",
]

CONDITIONS = [
    "positive_dr_to_no_dr",
    "control_no_dr_to_no_dr",
    "reverse_no_dr_to_dr",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate BayesPatch pair results by target and donor "
            "and run the camera-ready directionality tests."
        )
    )

    parser.add_argument(
        "--pairs",
        type=Path,
        default=Path(
            "results/"
            "donor_target_robustness_pair_deltas.csv"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
    )

    return parser.parse_args()


def validate_pair_table(df):
    required = {
        "pairing_strategy",
        "config",
        "control",
        "donor_index",
        "target_index",
        "delta",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Pair table is missing columns: {sorted(missing)}"
        )

    if df["delta"].isna().any():
        raise ValueError(
            "Pair table contains missing delta values"
        )


def select_main_experiment(df):
    main = df[
        (df["pairing_strategy"] == "top10_extreme")
        & (df["config"].isin(CONFIGURATIONS))
        & (df["control"].isin(CONDITIONS))
    ].copy()

    expected_rows = 600

    if len(main) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} top-10 pair rows, "
            f"found {len(main)}"
        )

    counts = (
        main.groupby(["config", "control"])
        .size()
    )

    if not counts.eq(100).all():
        raise ValueError(
            "Each configuration/condition must contain "
            "exactly 100 pair observations.\n"
            f"{counts}"
        )

    return main


def aggregate_by_target(main):
    return (
        main
        .groupby(
            [
                "config",
                "control",
                "target_index",
            ],
            as_index=False,
        )
        .agg(
            mean_delta=("delta", "mean"),
            std_delta_across_donors=("delta", "std"),
            n_donors=("donor_index", "nunique"),
        )
        .sort_values(
            [
                "config",
                "control",
                "target_index",
            ]
        )
        .reset_index(drop=True)
    )


def aggregate_by_donor(main):
    return (
        main
        .groupby(
            [
                "config",
                "control",
                "donor_index",
            ],
            as_index=False,
        )
        .agg(
            mean_delta=("delta", "mean"),
            std_delta_across_targets=("delta", "std"),
            n_targets=("target_index", "nunique"),
        )
        .sort_values(
            [
                "config",
                "control",
                "donor_index",
            ]
        )
        .reset_index(drop=True)
    )


def condition_values(
    aggregate_df,
    config,
    control,
):
    values = aggregate_df[
        (aggregate_df["config"] == config)
        & (aggregate_df["control"] == control)
    ]["mean_delta"].to_numpy(dtype=float)

    if len(values) != 10:
        raise ValueError(
            f"Expected 10 values for {config}, {control}; "
            f"found {len(values)}"
        )

    return values


def run_target_level_tests(target_level):
    rows = []

    for config in CONFIGURATIONS:
        positive = condition_values(
            target_level,
            config,
            "positive_dr_to_no_dr",
        )

        control = condition_values(
            target_level,
            config,
            "control_no_dr_to_no_dr",
        )

        reverse = condition_values(
            target_level,
            config,
            "reverse_no_dr_to_dr",
        )

        p_positive_gt_zero = wilcoxon(
            positive,
            alternative="greater",
        ).pvalue

        p_control_two_sided_zero = wilcoxon(
            control,
            alternative="two-sided",
        ).pvalue

        p_reverse_lt_zero = wilcoxon(
            reverse,
            alternative="less",
        ).pvalue

        positive_df = target_level[
            (target_level["config"] == config)
            & (
                target_level["control"]
                == "positive_dr_to_no_dr"
            )
        ][
            ["target_index", "mean_delta"]
        ].rename(
            columns={
                "mean_delta": "positive",
            }
        )

        control_df = target_level[
            (target_level["config"] == config)
            & (
                target_level["control"]
                == "control_no_dr_to_no_dr"
            )
        ][
            ["target_index", "mean_delta"]
        ].rename(
            columns={
                "mean_delta": "control",
            }
        )

        paired = positive_df.merge(
            control_df,
            on="target_index",
            how="inner",
            validate="one_to_one",
        )

        if len(paired) != 10:
            raise ValueError(
                f"Expected 10 paired no-DR targets for "
                f"{config}, found {len(paired)}"
            )

        p_positive_gt_control = wilcoxon(
            paired["positive"],
            paired["control"],
            alternative="greater",
        ).pvalue

        # Positive and reverse interventions use different
        # target classes, so these target-level samples are
        # treated as independent.
        p_positive_gt_reverse = mannwhitneyu(
            positive,
            reverse,
            alternative="greater",
        ).pvalue

        rows.append(
            {
                "config": config,
                "n_target_positive": len(positive),
                "n_target_control": len(control),
                "n_target_reverse": len(reverse),
                "positive_mean": float(positive.mean()),
                "control_mean": float(control.mean()),
                "reverse_mean": float(reverse.mean()),
                "p_positive_gt_zero": float(
                    p_positive_gt_zero
                ),
                "p_control_two_sided_zero": float(
                    p_control_two_sided_zero
                ),
                "p_reverse_lt_zero": float(
                    p_reverse_lt_zero
                ),
                "n_paired_positive_control": len(paired),
                "p_positive_gt_control": float(
                    p_positive_gt_control
                ),
                "p_positive_gt_reverse": float(
                    p_positive_gt_reverse
                ),
            }
        )

    return pd.DataFrame(rows)


def run_donor_level_tests(donor_level):
    alternatives = {
        "positive_dr_to_no_dr": "greater",
        "control_no_dr_to_no_dr": "two-sided",
        "reverse_no_dr_to_dr": "less",
    }

    rows = []

    for config in CONFIGURATIONS:
        for condition in CONDITIONS:
            values = condition_values(
                donor_level,
                config,
                condition,
            )

            test = wilcoxon(
                values,
                alternative=alternatives[condition],
            )

            rows.append(
                {
                    "config": config,
                    "condition": condition,
                    "n_donors": len(values),
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)),
                    "p_value": float(test.pvalue),
                }
            )

    return pd.DataFrame(rows)


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

    if not args.pairs.exists():
        raise FileNotFoundError(
            f"Missing pair table: {args.pairs}"
        )

    pair_df = pd.read_csv(args.pairs)

    print("Input:", args.pairs)
    print("Input shape:", pair_df.shape)

    validate_pair_table(pair_df)

    main_df = select_main_experiment(pair_df)

    print()
    print("Main top-10 pair summary:")
    print(
        main_df
        .groupby(["config", "control"])
        .agg(
            n_pairs=("delta", "size"),
            n_donors=("donor_index", "nunique"),
            n_targets=("target_index", "nunique"),
            mean_delta=("delta", "mean"),
            std_delta=("delta", "std"),
        )
        .to_string()
    )

    target_level = aggregate_by_target(
        main_df
    )

    donor_level = aggregate_by_donor(
        main_df
    )

    target_stats = run_target_level_tests(
        target_level
    )

    donor_stats = run_donor_level_tests(
        donor_level
    )

    target_aggregates_path = (
        args.output_dir
        / "directionality_target_level_aggregates.csv"
    )

    donor_aggregates_path = (
        args.output_dir
        / "directionality_donor_level_aggregates.csv"
    )

    target_tests_path = (
        args.output_dir
        / "directionality_target_level_tests.csv"
    )

    donor_tests_path = (
        args.output_dir
        / "directionality_donor_level_tests.csv"
    )

    save_csv(
        target_level,
        target_aggregates_path,
    )

    save_csv(
        donor_level,
        donor_aggregates_path,
    )

    save_csv(
        target_stats,
        target_tests_path,
    )

    save_csv(
        donor_stats,
        donor_tests_path,
    )

    print()
    print("Target-level statistics:")
    print(target_stats.to_string(index=False))

    print()
    print("Donor-level statistics:")
    print(donor_stats.to_string(index=False))

    print()
    print("Saved:")
    print(target_aggregates_path)
    print(donor_aggregates_path)
    print(target_tests_path)
    print(donor_tests_path)


if __name__ == "__main__":
    main()

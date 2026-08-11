import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Verify the BayesPatch activation-magnitude "
            "heuristic against the causal sparse objective."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "results/"
            "magnitude_heuristic_baseline.csv"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
    )

    return parser.parse_args()


def configuration_set(df):
    return {
        (
            int(row.layer),
            str(row.region),
            float(row.alpha),
        )
        for row in df.itertuples(index=False)
    }


def top_k_overlap(
    df,
    heuristic_column,
    causal_column,
    k,
):
    causal_top = (
        df.sort_values(
            causal_column,
            ascending=False,
        )
        .head(k)
    )

    heuristic_top = (
        df.sort_values(
            heuristic_column,
            ascending=False,
        )
        .head(k)
    )

    causal_set = configuration_set(
        causal_top
    )

    heuristic_set = configuration_set(
        heuristic_top
    )

    overlap_set = (
        causal_set
        & heuristic_set
    )

    return {
        "overlap_count": len(overlap_set),
        "overlap_fraction": (
            len(overlap_set) / k
        ),
        "overlap_configurations": sorted(
            overlap_set
        ),
        "causal_top": causal_top,
        "heuristic_top": heuristic_top,
    }


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

    if not args.input.exists():
        raise FileNotFoundError(
            f"Missing magnitude table: {args.input}"
        )

    df = pd.read_csv(args.input)

    required_columns = {
        "layer",
        "region",
        "alpha",
        "mean_shift",
        "causal_objective",
        "mean_activation_distance",
        "magnitude_score_raw",
        "magnitude_sparse_score",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"Magnitude table is missing: {sorted(missing)}"
        )

    valid = df.dropna(
        subset=[
            "layer",
            "region",
            "alpha",
            "causal_objective",
            "magnitude_score_raw",
            "magnitude_sparse_score",
        ]
    ).copy()

    valid["layer"] = (
        pd.to_numeric(
            valid["layer"],
            errors="raise",
        )
        .astype(int)
    )

    valid["region"] = (
        valid["region"]
        .astype(str)
        .str.lower()
    )

    valid["alpha"] = (
        pd.to_numeric(
            valid["alpha"],
            errors="raise",
        )
        .astype(float)
    )

    valid["causal_objective"] = (
        pd.to_numeric(
            valid["causal_objective"],
            errors="raise",
        )
        .astype(float)
    )

    valid["magnitude_score_raw"] = (
        pd.to_numeric(
            valid["magnitude_score_raw"],
            errors="raise",
        )
        .astype(float)
    )

    valid["magnitude_sparse_score"] = (
        pd.to_numeric(
            valid["magnitude_sparse_score"],
            errors="raise",
        )
        .astype(float)
    )

    duplicate_count = valid.duplicated(
        subset=[
            "layer",
            "region",
            "alpha",
        ]
    ).sum()

    if duplicate_count:
        raise ValueError(
            f"Found {duplicate_count} duplicate configurations"
        )

    print("Input rows:", len(df))
    print("Valid rows:", len(valid))
    print(
        "Unique configurations:",
        valid[
            ["layer", "region", "alpha"]
        ].drop_duplicates().shape[0],
    )

    rho_sparse, p_sparse = spearmanr(
        valid["magnitude_sparse_score"],
        valid["causal_objective"],
    )

    rho_raw, p_raw = spearmanr(
        valid["magnitude_score_raw"],
        valid["causal_objective"],
    )

    sparse_overlap = top_k_overlap(
        df=valid,
        heuristic_column="magnitude_sparse_score",
        causal_column="causal_objective",
        k=args.top_k,
    )

    raw_overlap = top_k_overlap(
        df=valid,
        heuristic_column="magnitude_score_raw",
        causal_column="causal_objective",
        k=args.top_k,
    )

    print()
    print("Spearman correlations:")
    print(
        "Sparsity-adjusted magnitude vs causal objective:",
        f"rho={rho_sparse:.12f}",
        f"p={p_sparse:.12g}",
    )
    print(
        "Raw magnitude vs causal objective:",
        f"rho={rho_raw:.12f}",
        f"p={p_raw:.12g}",
    )

    print()
    print(
        f"Top-{args.top_k} adjusted overlap:",
        sparse_overlap["overlap_count"],
    )
    print(
        f"Top-{args.top_k} raw overlap:",
        raw_overlap["overlap_count"],
    )

    public_columns = [
        "layer",
        "region",
        "alpha",
        "mean_shift",
        "causal_objective",
        "mean_activation_distance",
        "std_activation_distance",
        "magnitude_score_raw",
        "magnitude_score_norm",
        "magnitude_sparse_score",
    ]

    public_table = (
        valid[public_columns]
        .sort_values(
            ["layer", "region", "alpha"]
        )
        .reset_index(drop=True)
    )

    summary = {
        "n_configurations": int(
            len(valid)
        ),
        "spearman_magnitude_sparse_vs_causal": float(
            rho_sparse
        ),
        "p_magnitude_sparse_vs_causal": float(
            p_sparse
        ),
        "spearman_raw_magnitude_vs_causal": float(
            rho_raw
        ),
        "p_raw_magnitude_vs_causal": float(
            p_raw
        ),
        "top_k": int(args.top_k),
        "top_k_sparse_overlap": int(
            sparse_overlap["overlap_count"]
        ),
        "top_k_raw_overlap": int(
            raw_overlap["overlap_count"]
        ),
    }

    summary_table = pd.DataFrame(
        [
            {
                "baseline": (
                    "Sparsity-adjusted magnitude"
                ),
                "spearman_rho": float(
                    rho_sparse
                ),
                "p_value": float(
                    p_sparse
                ),
                f"top_{args.top_k}_overlap": int(
                    sparse_overlap[
                        "overlap_count"
                    ]
                ),
            },
            {
                "baseline": "Raw magnitude",
                "spearman_rho": float(
                    rho_raw
                ),
                "p_value": float(
                    p_raw
                ),
                f"top_{args.top_k}_overlap": int(
                    raw_overlap[
                        "overlap_count"
                    ]
                ),
            },
        ]
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    public_path = (
        args.output_dir
        / "magnitude_heuristic_public.csv"
    )

    summary_json_path = (
        args.output_dir
        / "magnitude_heuristic_summary.json"
    )

    summary_csv_path = (
        args.output_dir
        / "magnitude_heuristic_summary.csv"
    )

    adjusted_top_path = (
        args.output_dir
        / "magnitude_adjusted_top20.csv"
    )

    raw_top_path = (
        args.output_dir
        / "magnitude_raw_top20.csv"
    )

    causal_top_path = (
        args.output_dir
        / "causal_objective_top20.csv"
    )

    save_csv(
        public_table,
        public_path,
    )

    save_csv(
        summary_table,
        summary_csv_path,
    )

    save_csv(
        sparse_overlap["heuristic_top"],
        adjusted_top_path,
    )

    save_csv(
        raw_overlap["heuristic_top"],
        raw_top_path,
    )

    save_csv(
        sparse_overlap["causal_top"],
        causal_top_path,
    )

    with summary_json_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    print()
    print("Saved:")
    print(public_path)
    print(summary_json_path)
    print(summary_csv_path)
    print(adjusted_top_path)
    print(raw_top_path)
    print(causal_top_path)


if __name__ == "__main__":
    main()

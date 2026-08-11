import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PAPER_METHODS = [
    "random",
    "bayespatch_tpe_unique",
]

METHOD_LABELS = {
    "random": "Random search",
    "bayespatch_tpe_unique": "BayesPatch",
}

PAPER_BUDGETS = [
    10,
    20,
    30,
    50,
    100,
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce the BayesPatch budget-search table "
            "and figure from the historical per-seed curves."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "results/"
            "optimizer_budget_curves_historical.csv"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
    )

    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=Path("figures"),
    )

    return parser.parse_args()


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


def standardize_columns(df):
    work = df.copy()

    aliases = {
        "trial": "step",
        "iteration": "step",
        "budget": "step",
        "n_eval": "step",
        "objective": "best_objective",
        "best_so_far": "best_objective",
    }

    for source, destination in aliases.items():
        if (
            destination not in work.columns
            and source in work.columns
        ):
            work[destination] = work[source]

    required = {
        "method",
        "seed",
        "step",
        "best_objective",
    }

    missing = required - set(work.columns)

    if missing:
        raise ValueError(
            "Budget table is missing required columns: "
            f"{sorted(missing)}. "
            f"Available columns: {work.columns.tolist()}"
        )

    work["method"] = work["method"].astype(str)

    work["seed"] = pd.to_numeric(
        work["seed"],
        errors="raise",
    ).astype(int)

    work["step"] = pd.to_numeric(
        work["step"],
        errors="raise",
    ).astype(int)

    work["best_objective"] = pd.to_numeric(
        work["best_objective"],
        errors="raise",
    ).astype(float)

    return work


def determine_oracle(df):
    if "regret_to_oracle" in df.columns:
        regret = pd.to_numeric(
            df["regret_to_oracle"],
            errors="raise",
        ).astype(float)

        implied = (
            df["best_objective"]
            + regret
        )

        spread = (
            implied.max()
            - implied.min()
        )

        if spread > 1e-8:
            raise ValueError(
                "The implied oracle is not constant. "
                f"Range: {implied.min()} to {implied.max()}"
            )

        return float(implied.mean())

    for column in [
        "oracle_objective",
        "oracle",
    ]:
        if column in df.columns:
            values = pd.to_numeric(
                df[column],
                errors="raise",
            ).dropna()

            if values.empty:
                continue

            spread = (
                values.max()
                - values.min()
            )

            if spread > 1e-8:
                raise ValueError(
                    f"{column} is not constant"
                )

            return float(values.mean())

    raise ValueError(
        "Could not determine the budget-search oracle. "
        "Expected regret_to_oracle, oracle_objective, "
        "or oracle."
    )


def validate_trajectories(df):
    duplicate_count = df.duplicated(
        subset=[
            "method",
            "seed",
            "step",
        ]
    ).sum()

    if duplicate_count:
        raise ValueError(
            f"Found {duplicate_count} duplicate "
            "method/seed/step rows"
        )

    violations = []

    for (method, seed), group in df.groupby(
        ["method", "seed"]
    ):
        ordered = group.sort_values("step")

        differences = (
            ordered["best_objective"]
            .diff()
            .dropna()
        )

        if (differences < -1e-12).any():
            violations.append(
                (method, int(seed))
            )

    if violations:
        raise ValueError(
            "Best-objective trajectories decrease for: "
            f"{violations}"
        )


def summarize_paper_budgets(df, oracle):
    selected = df[
        df["step"].isin(PAPER_BUDGETS)
    ].copy()

    summary = (
        selected
        .groupby(
            ["method", "step"],
            as_index=False,
        )
        .agg(
            n_seeds=("seed", "nunique"),
            mean_best_objective=(
                "best_objective",
                "mean",
            ),
            std_best_objective=(
                "best_objective",
                "std",
            ),
        )
        .rename(
            columns={
                "step": "budget",
            }
        )
    )

    summary["mean_regret_to_oracle"] = (
        oracle
        - summary["mean_best_objective"]
    )

    summary["std_regret_to_oracle"] = (
        summary["std_best_objective"]
    )

    summary["method_label"] = (
        summary["method"]
        .map(METHOD_LABELS)
    )

    return (
        summary
        .sort_values(
            ["budget", "method"]
        )
        .reset_index(drop=True)
    )


def make_figure(curve_summary, oracle, path):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    colors = {
        "bayespatch_tpe_unique": "#1f77b4",
        "random": "#ff7f0e",
    }

    figure, axis = plt.subplots(
        figsize=(6.4, 4.0),
        dpi=180,
    )

    for method in [
        "bayespatch_tpe_unique",
        "random",
    ]:
        selected = (
            curve_summary[
                curve_summary["method"] == method
            ]
            .sort_values("step")
        )

        x = selected["step"].to_numpy()
        mean = selected[
            "mean_best_objective"
        ].to_numpy()

        std = selected[
            "std_best_objective"
        ].to_numpy()

        axis.plot(
            x,
            mean,
            linewidth=2,
            label=METHOD_LABELS[method],
            color=colors[method],
        )

        axis.fill_between(
            x,
            mean - std,
            mean + std,
            color=colors[method],
            alpha=0.15,
        )

    axis.axhline(
        oracle,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label="Fixed-table oracle",
    )

    axis.set_xlabel(
        "Evaluation budget"
    )

    axis.set_ylabel(
        "Best objective $J(c)$"
    )

    axis.set_title(
        "Budgeted search over the fixed intervention table"
    )

    axis.set_xlim(
        1,
        int(curve_summary["step"].max()),
    )

    axis.grid(
        alpha=0.2,
    )

    axis.legend(
        frameon=True,
    )

    figure.tight_layout()

    figure.savefig(
        path,
        bbox_inches="tight",
    )

    plt.close(figure)


def main():
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(
            f"Missing budget curves: {args.input}"
        )

    raw = pd.read_csv(args.input)
    work = standardize_columns(raw)

    print("Raw shape:", raw.shape)
    print("Methods:", sorted(work["method"].unique()))
    print("Seeds:", sorted(work["seed"].unique()))
    print(
        "Step range:",
        int(work["step"].min()),
        "to",
        int(work["step"].max()),
    )

    paper = work[
        work["method"].isin(PAPER_METHODS)
    ].copy()

    missing_methods = (
        set(PAPER_METHODS)
        - set(paper["method"])
    )

    if missing_methods:
        raise ValueError(
            "Missing paper methods: "
            f"{sorted(missing_methods)}"
        )

    oracle = determine_oracle(paper)

    print("Budget-search oracle:", oracle)

    if abs(oracle - 0.0890) > 0.001:
        raise ValueError(
            "Budget-search oracle differs materially "
            f"from the reported 0.0890: {oracle}"
        )

    validate_trajectories(paper)

    seed_counts = (
        paper.groupby("method")["seed"]
        .nunique()
    )

    print()
    print("Seed counts:")
    print(seed_counts)

    curve_summary = (
        paper
        .groupby(
            ["method", "step"],
            as_index=False,
        )
        .agg(
            n_seeds=("seed", "nunique"),
            mean_best_objective=(
                "best_objective",
                "mean",
            ),
            std_best_objective=(
                "best_objective",
                "std",
            ),
        )
        .sort_values(
            ["method", "step"]
        )
        .reset_index(drop=True)
    )

    budget_summary = summarize_paper_budgets(
        paper,
        oracle,
    )

    public_columns = [
        "method",
        "seed",
        "step",
        "best_objective",
    ]

    if "regret_to_oracle" in paper.columns:
        public_columns.append(
            "regret_to_oracle"
        )

    public_curves = (
        paper[public_columns]
        .sort_values(
            ["method", "seed", "step"]
        )
        .reset_index(drop=True)
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.figure_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    public_curves_path = (
        args.output_dir
        / "optimizer_budget_curves.csv"
    )

    curve_summary_path = (
        args.output_dir
        / "optimizer_budget_curve_summary.csv"
    )

    budget_summary_path = (
        args.output_dir
        / "optimizer_budget_summary.csv"
    )

    summary_json_path = (
        args.output_dir
        / "optimizer_budget_metadata.json"
    )

    figure_path = (
        args.figure_dir
        / "budget_search_curve.pdf"
    )

    save_csv(
        public_curves,
        public_curves_path,
    )

    save_csv(
        curve_summary,
        curve_summary_path,
    )

    save_csv(
        budget_summary,
        budget_summary_path,
    )

    metadata = {
        "oracle": oracle,
        "oracle_description": (
            "Fixed candidate-table oracle used by "
            "the historical budget-search experiment"
        ),
        "methods": PAPER_METHODS,
        "budgets": PAPER_BUDGETS,
        "n_seeds_by_method": {
            method: int(count)
            for method, count in seed_counts.items()
        },
        "historical_implementation_available": False,
        "note": (
            "The original Optuna implementation was not "
            "present in the exported notebook. These are "
            "the historical per-seed trajectories used by "
            "the camera-ready analysis."
        ),
    }

    with summary_json_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )

    make_figure(
        curve_summary=curve_summary,
        oracle=oracle,
        path=figure_path,
    )

    print()
    print("Paper budget summary:")
    print(
        budget_summary.to_string(
            index=False
        )
    )

    print()
    print("Saved:")
    print(public_curves_path)
    print(curve_summary_path)
    print(budget_summary_path)
    print(summary_json_path)
    print(figure_path)


if __name__ == "__main__":
    main()

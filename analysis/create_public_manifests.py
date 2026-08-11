from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

SOURCES = {
    "source_main_subset.csv": "main_subset_indices.csv",
    "source_seed1_subset.csv": "seed1_subset_indices.csv",
    "source_seed2_subset.csv": "seed2_subset_indices.csv",
}

PUBLIC_COLUMNS = [
    "subset_index",
    "dataset_index",
    "diagnosis",
    "binary_dr",
]

for source_name, output_name in SOURCES.items():
    source = DATA_DIR / source_name

    if not source.exists():
        print(f"Missing; skipped: {source}")
        continue

    df = pd.read_csv(source)

    missing = [c for c in PUBLIC_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{source_name} is missing columns: {missing}")

    public_df = df[PUBLIC_COLUMNS].copy()

    if "seed1" in source_name:
        public_df.insert(0, "subset_seed", 1)
    elif "seed2" in source_name:
        public_df.insert(0, "subset_seed", 2)
    else:
        public_df.insert(0, "subset_seed", 0)

    destination = DATA_DIR / output_name
    public_df.to_csv(destination, index=False)
    print(f"Wrote {destination}: {len(public_df)} rows")
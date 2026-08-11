from pathlib import Path

import pandas as pd
from datasets import load_dataset


DATASET_NAME = "bumbledeep/aptos"


def load_aptos():
    dataset_dict = load_dataset(DATASET_NAME)

    if "train" in dataset_dict:
        return dataset_dict["train"]

    return dataset_dict[next(iter(dataset_dict.keys()))]


def load_manifest(path):
    manifest = pd.read_csv(Path(path))

    required = {
        "subset_index",
        "dataset_index",
        "diagnosis",
        "binary_dr",
    }
    missing = required - set(manifest.columns)

    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")

    return manifest


def get_image(dataset, dataset_index):
    item = dataset[int(dataset_index)]
    image = item["image"]

    if image.mode != "RGB":
        image = image.convert("RGB")

    return image

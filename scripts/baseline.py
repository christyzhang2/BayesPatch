import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from tqdm.auto import tqdm


# Allow execution as: python scripts/baseline.py
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from bayespatch.data import get_image, load_aptos, load_manifest
from bayespatch.model import (
    DEFAULT_PROMPTS,
    encode_prompts,
    image_margin,
    load_biomedclip,
)


SEED = 0


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the frozen BiomedCLIP baseline on an APTOS manifest."
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/main_subset_indices.csv"),
        help="CSV containing the selected Hugging Face dataset indices.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory in which to save metrics and per-image margins.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
    )

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Manifest: {args.manifest}")
    manifest = load_manifest(args.manifest)

    print(f"Manifest rows: {len(manifest)}")
    print("Class counts:")
    print(manifest["binary_dr"].value_counts().sort_index())

    if len(manifest) != 100:
        print(
            f"WARNING: expected 100 examples for the main subset, "
            f"but found {len(manifest)}."
        )

    print("Loading APTOS from Hugging Face...")
    dataset = load_aptos()
    print(f"Dataset rows: {len(dataset)}")

    print("Loading BiomedCLIP...")
    model, preprocess, tokenizer, device = load_biomedclip()
    print(f"Device: {device}")

    text_features = encode_prompts(
        model=model,
        tokenizer=tokenizer,
        prompts=DEFAULT_PROMPTS,
        device=device,
    )

    result_rows = []

    for row in tqdm(
        manifest.itertuples(index=False),
        total=len(manifest),
        desc="Computing baseline margins",
    ):
        image = get_image(dataset, row.dataset_index)
        image_tensor = preprocess(image).unsqueeze(0).to(device)

        margin_tensor = image_margin(
            model=model,
            image_tensor=image_tensor,
            text_features=text_features,
        )

        margin = float(margin_tensor.item())
        prediction = int(margin > 0)

        result_rows.append(
            {
                "subset_seed": int(getattr(row, "subset_seed", 0)),
                "subset_index": int(row.subset_index),
                "dataset_index": int(row.dataset_index),
                "diagnosis": int(row.diagnosis),
                "binary_dr": int(row.binary_dr),
                "margin": margin,
                "prediction": prediction,
            }
        )

    results = pd.DataFrame(result_rows)

    accuracy = accuracy_score(
        results["binary_dr"],
        results["prediction"],
    )

    auroc = roc_auc_score(
        results["binary_dr"],
        results["margin"],
    )

    matrix = confusion_matrix(
        results["binary_dr"],
        results["prediction"],
        labels=[0, 1],
    )

    margins_path = args.output_dir / "subset_margins.csv"
    metrics_path = args.output_dir / "baseline_metrics.json"

    results.to_csv(margins_path, index=False)

    metrics = {
        "dataset": "bumbledeep/aptos",
        "model": (
            "microsoft/"
            "BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        ),
        "seed": args.seed,
        "n_images": int(len(results)),
        "accuracy": float(accuracy),
        "auroc": float(auroc),
        "confusion_matrix": matrix.tolist(),
        "no_dr_prompt": DEFAULT_PROMPTS[0],
        "dr_prompt": DEFAULT_PROMPTS[1],
    }

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print()
    print("Baseline complete")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUROC: {auroc:.4f}")
    print("Confusion matrix:")
    print(matrix)
    print()
    print(f"Saved margins: {margins_path}")
    print(f"Saved metrics: {metrics_path}")

    expected_matrix = [[49, 1], [15, 35]]

    if (
        abs(accuracy - 0.84) < 1e-6
        and abs(auroc - 0.9416) < 0.001
        and matrix.tolist() == expected_matrix
    ):
        print("CHECK PASSED: baseline matches the paper.")
    else:
        print("CHECK FAILED: baseline does not exactly match the paper.")
        print("Do not start the full grid until this is investigated.")


if __name__ == "__main__":
    main()
    
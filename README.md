# BayesPatch

**Sparsity-Constrained Activation-Patching Search for Medical Vision-Language Models**

Official research-code release for the MI4MedFM workshop at MICCAI 2026.

**Authors:** Ziyue Zhang, Indira Duisembayeva, Muhammad Bilal, and Yutong Xie  
**Affiliation:** Mohamed bin Zayed University of Artificial Intelligence, Abu Dhabi, United Arab Emirates

BayesPatch is a proof-of-concept framework for searching sparse, causal activation-level interventions in medical vision-language models. It treats activation patching as a finite optimization problem over the visual-transformer layer, token region, and patch strength, rewarding interventions that shift a disease-prompt margin while penalizing broad patches.

The camera-ready experiments apply BayesPatch to frozen BiomedCLIP representations on diabetic-retinopathy prompts and balanced subsets derived from the APTOS 2019 dataset.

> **Clinical-use disclaimer:** BayesPatch is a mechanistic-interpretability research prototype. It is not a diagnostic system, clinical decision-support tool, lesion-localization method, or demonstration of clinical validity or safety.

## Overview

For a donor image $x_d$, target image $x_t$, layer $\ell$, token region $r$, and patch strength $\alpha$, BayesPatch interpolates donor activations into the target forward pass:

```math
\widetilde{A}_{\ell,r}(x_t)
=
(1-\alpha)A_{\ell,r}(x_t)
+
\alpha A_{\ell,r}(x_d).
```

The model's diabetic-retinopathy prompt margin is

```math
m(x)
=
s(x,t_{\mathrm{DR}})
-
s(x,t_{\mathrm{NoDR}}),
```

where $s$ is cosine similarity between normalized image and text embeddings. The causal effect of a configuration $c=(\ell,r,\alpha)$ is

```math
\Delta m(c)
=
m_{\mathrm{patched}}(x_t;c)
-
m(x_t).
```

BayesPatch scores configurations using

```math
J(c)
=
\mathbb{E}_{(x_d,x_t)}[\Delta m(c)]
-
\lambda C(c),
```

with

```math
C(c)=C(r)\alpha.
```

The sparsity penalty favors compact and inspectable interventions. It is a hand-designed proxy for inspection burden, not a clinically validated measure of interpretability.

## Main findings

On the balanced 100-image APTOS-HF subset, frozen BiomedCLIP achieved:

| Metric | Result |
|---|---:|
| Accuracy | 0.8400 |
| AUROC | 0.9416 |
| Confusion matrix | `[[49, 1], [15, 35]]` |

At $\lambda=0.03$, the historical 528-configuration sparse grid identified:

| Configuration | Mean DR-margin shift | Sparse objective |
|---|---:|---:|
| L11 CLS, $\alpha=1.0$ | 0.1084 | 0.1069 |
| L1 center, $\alpha=1.0$ | 0.0877 | 0.0802 |

Directionality controls on the reproducible Hugging Face subset produced:

| Configuration | DR to no-DR | no-DR to no-DR | Reverse no-DR to DR |
|---|---:|---:|---:|
| L11 CLS, $\alpha=1.0$ | +0.1119 ± 0.0169 | +0.0111 ± 0.0074 | -0.1008 ± 0.0156 |
| L1 center, $\alpha=1.0$ | +0.0995 ± 0.0240 | +0.0134 ± 0.0096 | -0.0488 ± 0.0180 |

Because the 100 pair-level observations reuse 10 donors and 10 targets, inferential tests are performed after target-level and donor-level aggregation rather than treating all pairs as independent.

The non-causal magnitude heuristic correlated only weakly with causal patching:

| Comparison | Spearman $\rho$ |
|---|---:|
| Sparsity-adjusted magnitude vs. causal objective | 0.216 |
| Raw activation magnitude vs. causal objective | 0.249 |

The sparsity-adjusted magnitude ranking had zero overlap with the top 20 causal configurations.

In the historical fixed-table budget experiment, duplicate-free TPE achieved a higher mean best objective than random search at every reported budget:

| Budget | BayesPatch | Random search |
|---:|---:|---:|
| 10 | 0.0484 ± 0.0170 | 0.0427 ± 0.0133 |
| 20 | 0.0655 ± 0.0143 | 0.0545 ± 0.0162 |
| 30 | 0.0716 ± 0.0122 | 0.0564 ± 0.0142 |
| 50 | 0.0775 ± 0.0121 | 0.0589 ± 0.0133 |
| 100 | 0.0801 ± 0.0115 | 0.0735 ± 0.0136 |

The fixed-table budget-search oracle is $J^*=0.0890$. It is specific to the optimizer-evaluation table and is intentionally reported separately from the sparse-grid objective of 0.1069.

## Repository structure

```text
BayesPatch/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
│
├── bayespatch/
│   ├── __init__.py
│   ├── data.py
│   ├── model.py
│   ├── regions.py
│   ├── patching.py
│   └── objective.py
│
├── scripts/
│   ├── baseline.py
│   ├── directionality.py
│   └── run_grid.py
│
├── analysis/
│   ├── image_level_statistics.py
│   ├── magnitude_analysis.py
│   └── budget_analysis.py
│
├── data/
│   ├── main_subset_indices.csv
│   ├── seed1_subset_indices.csv
│   ├── seed2_subset_indices.csv
│   └── historical_selected_pairs.csv
│
├── results/
│   ├── baseline_metrics.json
│   ├── subset_margins.csv
│   ├── historical_sparse_grid_clean.csv
│   ├── donor_target_robustness_pair_deltas.csv
│   ├── donor_target_robustness_summary.csv
│   ├── directionality_target_level_aggregates.csv
│   ├── directionality_donor_level_aggregates.csv
│   ├── directionality_target_level_tests.csv
│   ├── directionality_donor_level_tests.csv
│   ├── magnitude_heuristic_public.csv
│   ├── magnitude_heuristic_summary.json
│   ├── optimizer_budget_curves.csv
│   ├── optimizer_budget_summary.csv
│   └── optimizer_budget_metadata.json
│
└── figures/
    └── budget_search_curve.pdf
```

Some result or robustness files may be omitted from a minimal release. The commands below describe the files included in the camera-ready repository.

## Model and dataset

### Model

The experiments use the frozen BiomedCLIP checkpoint:

```text
microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
```

The model is loaded through `open_clip_torch`. Both encoders remain frozen; no fine-tuning is performed.

### Dataset

The reproducible baseline and directionality experiments load the Hugging Face mirror:

```text
bumbledeep/aptos
```

The main subset contains:

- 50 no-DR images;
- 50 DR-present images;
- grade distribution for DR-present images: 8 grade-1, 27 grade-2, 4 grade-3, and 11 grade-4 examples;
- subset seed 0.

Two additional balanced subsets use seeds 1 and 2.

APTOS retinal images are **not** included in this repository. Public CSV manifests contain only dataset indices, subset indices, grades, binary labels, and reproducibility metadata.

### Prompts

No-DR prompt:

```text
normal retina without diabetic retinopathy
```

DR prompt:

```text
retina with diabetic retinopathy lesions
```

## Search space

The historical sparse grid contains:

- 12 visual-transformer layers: 0 through 11;
- 11 token regions;
- 4 patch strengths: 0.25, 0.50, 0.75, and 1.00;
- 528 configurations in total.

The regions are:

```text
cls
center
upper
lower
left
right
q1
q2
q3
q4
random
```

The four quadrants correspond to upper-left, upper-right, lower-left, and lower-right regions of the 14 by 14 patch-token grid. The random region contains 49 of the 196 spatial patch tokens.

Region costs are:

| Region | $C(r)$ |
|---|---:|
| CLS | 0.05 |
| Center | 0.25 |
| Each quadrant | 0.25 |
| Random subset | 0.30 |
| Upper/lower/left/right half | 0.50 |

The main experiments use $\lambda=0.03$.

## Installation

### Recommended: Google Colab

A GPU-enabled Google Colab runtime is recommended.

```bash
git clone https://github.com/christyzhang2/BayesPatch.git
cd BayesPatch
pip install -r requirements.txt
```

Colab already provides a CUDA-compatible PyTorch installation. `requirements.txt` therefore lists the additional packages required by this release without forcing Colab to replace its PyTorch build.

### Local environment

Create an isolated environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install a PyTorch build appropriate for your operating system and hardware, then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

A CUDA-capable GPU is strongly recommended for model-based experiments. Result-table analyses do not require a GPU.

## Reproducing the baseline

Run:

```bash
python scripts/baseline.py \
    --manifest data/main_subset_indices.csv \
    --output-dir results
```

Expected output:

```text
Accuracy: 0.8400
AUROC: 0.9416
Confusion matrix:
[[49  1]
 [15 35]]
```

The command writes:

```text
results/subset_margins.csv
results/baseline_metrics.json
```

`subset_margins.csv` intentionally excludes local image paths.

## Reproducing directionality controls

Run the main top-10 directionality experiment:

```bash
python scripts/directionality.py \
    --margins results/subset_margins.csv \
    --strategy top10_extreme \
    --pair-output results/directionality_top10_pairs.csv \
    --summary-output results/directionality_top10_summary.csv
```

Run all donor-target selection strategies:

```bash
python scripts/directionality.py \
    --margins results/subset_margins.csv \
    --pair-output results/donor_target_robustness_pair_deltas.csv \
    --summary-output results/donor_target_robustness_summary.csv
```

The full run evaluates:

- `top10_extreme`;
- `top5_extreme`;
- `medium10`;
- L11 CLS and L1 center;
- DR donor to no-DR target;
- no-DR donor to no-DR target;
- reverse no-DR donor to DR target.

## Image-aggregated statistical analysis

Run:

```bash
python analysis/image_level_statistics.py \
    --pairs results/donor_target_robustness_pair_deltas.csv \
    --output-dir results
```

This produces:

```text
results/directionality_target_level_aggregates.csv
results/directionality_donor_level_aggregates.csv
results/directionality_target_level_tests.csv
results/directionality_donor_level_tests.csv
```

The analysis uses:

- one-sided Wilcoxon signed-rank tests for positive and reverse effects against zero;
- a paired Wilcoxon signed-rank test for DR-to-no-DR versus the same-target no-DR control;
- a Mann-Whitney U test for positive versus reverse target-level effects, which use different target classes;
- donor-level aggregation as a complementary dependence-aware analysis.

## Historical sparse grid

The exact camera-ready grid table is:

```text
results/historical_sparse_grid_clean.csv
```

It contains 528 unique configurations and standardized public region names.

Historical names were mapped as follows:

```text
quadrant_ul          -> q1
quadrant_ur          -> q2
quadrant_ll          -> q3
quadrant_lr          -> q4
random_patch_tokens  -> random
```

The original exported Colab contained the resulting grid table and its analysis but not the original grid-execution loop. `scripts/run_grid.py` is therefore a reconstruction of the exhaustive runner using the released patching implementation and Hugging Face manifest.

Because the historical grid and cleaned Hugging Face experiments have distinct subset provenance, a newly reconstructed HF grid should be treated as a replication experiment rather than expected to reproduce every historical grid value exactly. The released historical table is the source for the camera-ready grid values.

## Magnitude heuristic

Run:

```bash
python analysis/magnitude_analysis.py \
    --input results/magnitude_heuristic_public.csv \
    --output-dir results \
    --top-k 20
```

The historical magnitude analysis contains 288 configurations. It compares mean donor-target activation-distance magnitude with the causal sparse objective and reports Spearman correlations and top-$k$ overlap.

The magnitude heuristic is non-causal: it measures activation differences without intervening on the target forward pass.

## Budget-search analysis

Run:

```bash
python analysis/budget_analysis.py \
    --input results/optimizer_budget_curves.csv \
    --output-dir results \
    --figure-dir figures
```

This regenerates:

```text
results/optimizer_budget_curve_summary.csv
results/optimizer_budget_summary.csv
results/optimizer_budget_metadata.json
figures/budget_search_curve.pdf
```

The released curves contain the per-seed trajectories used for the camera-ready budget table and figure.

The original exported Colab did not contain the historical Optuna execution implementation. Accordingly, this release does not claim that regenerated TPE trajectories will be bitwise identical. It releases the exact historical trajectories and a reproducible analysis pipeline. The method recorded in those trajectories is duplicate-free TPE, named `bayespatch_tpe_unique`, compared with random search without replacement.

## Reproducibility notes

### Historical grid versus Hugging Face experiments

During repository reconstruction, two provenance layers were preserved explicitly:

1. The exact camera-ready sparse-grid table and historical selected image IDs.
2. Cleaned Hugging Face subset manifests used for the reproducible baseline and directionality experiments.

This separation avoids presenting a new HF-subset grid run as though it were the original historical grid execution.

### Random token region

The random region contains 49 spatial tokens. Pair-level reconstructed experiments seed random-region selection deterministically.

### Frozen model

BiomedCLIP remains in evaluation mode and all model parameters have gradients disabled. BayesPatch modifies intermediate activations only during hooked forward passes.

### Seeds

The main subset and model-side random seed are 0. Additional balanced-subset manifests use seeds 1 and 2. Historical budget curves include 10 seeds.

## Limitations

BayesPatch is evaluated on:

- one medical vision-language model;
- one diabetic-retinopathy prompt-classification task;
- small balanced subsets;
- confidence-selected donor-target examples;
- a manually specified sparsity-cost schedule;
- a finite search space for which exhaustive evaluation remains feasible.

The intervention results identify candidate activation-level mechanisms only in the computational sense: changing these activations changes the model's disease-prompt margin. They do not establish biological mechanisms, lesion localization, clinical meaningfulness, diagnostic validity, fairness, robustness in deployment, or patient safety.

The center-region intervention may reflect correlated image characteristics such as illumination, contrast, field of view, or image quality. Clinical interpretation would require lesion annotations, nuisance-matched controls, and expert ophthalmological review.

## Data and privacy

This repository does not redistribute APTOS images.

Public artifacts must not contain:

- Google Drive paths;
- Colab document identifiers;
- Hugging Face access tokens;
- API keys or credentials;
- model checkpoints;
- private image paths.

Before publishing changes, run:

```bash
grep -RniE \
  '/content/drive|MyDrive|colab.research.google.com/drive|password|api[_-]?key|bearer|hf_token|secret' \
  . \
  --exclude-dir=.git
```

Also inspect staged files:

```bash
git diff --cached --name-only
```

## Citation

A proceedings citation will be added when the final workshop bibliographic record is available.

For now, please cite the paper as:

```bibtex
@inproceedings{zhang2026bayespatch,
  author      = {Zhang, Ziyue and Duisembayeva, Indira and Bilal, Muhammad and Xie, Yutong},
  title       = {BayesPatch: Sparsity-Constrained Activation-Patching Search for Medical Vision-Language Models},
  booktitle   = {Proceedings of the MI4MedFM Workshop at MICCAI},
  institution = {Mohamed bin Zayed University of Artificial Intelligence},
  year        = {2026}
}
```

## License

The source code is released under the MIT License. Dataset and model use remain subject to their respective licenses and terms.

## Acknowledgments

This work uses the APTOS 2019 Blindness Detection dataset, its Hugging Face mirror, BiomedCLIP, OpenCLIP, Hugging Face Datasets, Optuna, PyTorch, SciPy, and scikit-learn.

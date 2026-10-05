# Phase 2 Report: Automated Malaria Cell Detection

## 1. Introduction

Malaria is diagnosed by looking at Giemsa-stained blood smears under a microscope, where a trained technician checks each red blood cell for parasites. This works, but it is slow (each slide holds hundreds of cells), it depends on scarce expert skill (especially in the rural regions where malaria is common), and results vary between observers and labs.

Our goal is an automated deep-learning system that classifies single red-blood-cell images as **parasitized** or **uninfected**, accurately and consistently, and light enough to eventually run at the point of care.

Phase 1 surveyed 11 published studies. In Phase 2 (this report) we reproduce the strongest-documented one, Marques et al. (2022), as our baseline, then test whether its accuracy holds up once slide-level leakage is removed.

**Why Marques et al. (2022).** Of the 11 studies, we selected this one because it best fits the reproducibility criteria for baseline formation:

- **Most fully specified recipe.** It states the backbone (EfficientNet-B0, ImageNet-pretrained), optimiser (Adam, lr 1e-4), scheduler (ReduceLROnPlateau), 10-fold cross-validation and fold-averaging ensemble — enough detail to rebuild it, which is why our result lands within ~1 pp.
- **Public, free-GPU-friendly.** It uses the standard NIH dataset (~350 MB on Kaggle) and a small backbone that trains at 224×224 in minutes per fold on a free Kaggle/Colab GPU.
- **Clear, comparable metrics.** It reports accuracy, precision, recall, F1 and ROC-AUC with concrete target numbers to reproduce against, and uses the same NIH benchmark as the other 10 surveyed studies.
- **An improvable assumption.** It splits at the image level, so cells from one patient/slide can fall in both training and test. That potential leakage is exactly what motivates our single-variable hypotheses (H1, H2).

## 2. Baseline reproduction (Marques et al., 2022)

**Goal.** Rebuild the paper's model on the same data and check we land within about 1 percentage point of its reported results. This becomes our Phase 2 baseline (experiment A0).

**Method.**
- Data: NIH malaria cell images, 27,558 cells (13,779 parasitized, 13,779 uninfected), Kaggle mirror.
- Model: EfficientNet-B0 pretrained on ImageNet, 224 × 224 input, dropout 0.2.
- Training: Adam (lr 1e-4), ReduceLROnPlateau (patience 6, down to 1e-6), up to 15 epochs with early stopping (patience 5).
- Validation: 10 % untouched hold-out test set; the rest split by stratified 10-fold CV. We trained 3 of the 10 folds (free GPU budget) and averaged them into an ensemble.
- Hardware: Kaggle, 2× Tesla T4 (global batch 128).

**Results** (`results/A0_baseline/`):

| Metric | Paper | Ours | Δ (pp) |
|---|---|---|---|
| Mean single-fold accuracy | 97.70 % | 97.38 ± 0.74 % | −0.32 |
| Ensemble accuracy | 98.29 % | 97.57 % | −0.72 |
| Recall | 98.82 % | 96.81 % | −2.01 |
| Precision | 97.74 % | 98.31 % | +0.57 |
| F1 | 98.28 % | 97.55 % | −0.73 |
| ROC-AUC | 99.76 % | 99.76 % | 0.00 |
| MCC | — | 0.95 | — |

![Training curves, confusion matrix and ROC](../results/A0_baseline/A0_curves_cm_roc.png)

![Ours vs. paper](../results/A0_baseline/A0_vs_paper.png)

**Deviations from the paper.** We kept every setting the paper reports. Where it is silent or our compute was limited, we chose:

| Setting | Paper | Ours | Why |
|---|---|---|---|
| Augmentation | Albumentations (details not listed) | Flips + 90° rotations | Paper doesn't specify its pipeline |
| Epochs | Not reported | ≤ 15, early stopping (patience 5) | Fits a free GPU session |
| Folds trained | 10 | 3 of 10 | Free GPU budget (~21 min per fold) |
| Batch size | Not reported | 64 per GPU, 128 global (2× T4) | Kaggle provided two GPUs |
| Test set | Not described | 10 % hold-out, never used in training | Clean ensemble evaluation |

**Takeaway.** Both headline accuracies are within 1 pp of the paper, so the reproduction succeeds and A0 is our baseline. Recall is 2 pp lower: on the test set the model missed 44 infected cells and wrongly flagged 23 healthy ones. Training all 10 folds (as the paper did) may close part of this gap.

## 3. Hypotheses and planned experiments

With the baseline (A0) established, we test two single-variable hypotheses in the next milestone. Each experiment changes exactly one thing relative to the experiment it is compared against, so any change in the result is attributable.

- **H1 — Slide-level leakage inflates accuracy.** The paper's image-level split lets cells from the same patient/slide appear in both training and test. We predict that switching to a slide-grouped split (StratifiedGroupKFold on the slide ID), with nothing else changed, will *lower* accuracy versus A0 — because the model can no longer lean on memorised slide-specific staining and must generalise to unseen patients. Tested by **A1** (vs A0).
- **H2 — Stain normalisation recovers cross-slide generalisation.** We predict that adding YUV stain normalisation (+ histogram equalisation) on top of the leakage-free split will *partly recover* the accuracy lost in A1 — because it removes the colour/staining differences between slides that the model was using as a shortcut. Tested by **A2** (vs A1).

| ID | Split | Preprocessing | Tests | Status |
|---|---|---|---|---|
| A0 | image-level (paper) | none | baseline reproduction | done |
| A1 | slide-grouped | none | H1 (vs A0) | to run |
| A2 | slide-grouped | YUV + hist-eq | H2 (vs A1) | to run |
| A3 | image-level | YUV + hist-eq | control: is any gain specific to unseen slides? (vs A0) | to run |

Results (mean ± SD over folds, recall and MCC for each experiment) will be filled in here after the A1–A3 runs.

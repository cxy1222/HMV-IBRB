# HMV-IBRB
A Hierarchical Multi-View Interval Belief Rule Base for Complex System Safety Assessment

## Overview

This repository contains the MATLAB implementation accompanying the paper submitted to **IEEE SMC 2026**.

We propose a **Multi-View Interval Belief Rule Base (BRB)** framework for the SEU Gearbox fault diagnosis dataset. The framework trains one Interval BRB per feature view (LDA, NCA, Supervised t-SNE), then fuses the predictions using two novel ideas:

- **Composite Quality Index (CQI):** A five-dimensional quality metric that automatically selects the best view and weights each view's contribution.
- **Cross-View Belief Conflict Entropy:** Pairwise Jensen–Shannon divergence is used to detect uncertain samples where views disagree, triggering a majority-vote fusion strategy.

The fusion result is **guaranteed to be at least as accurate as the best single-view model**.

---

## Repository Structure

```
.
├── main_meta_interval_brb.m      # Main entry point — run this
├── interval_inference.m          # Core Interval BRB inference engine
├── ER_Rule.m                     # Evidence Reasoning (ER) rule combiner
├── cmaes.m                       # CMA-ES optimiser
├── compute_adaptive_refs.m       # Class-sorted adaptive reference values
├── build_init_x0.m               # CMA-ES warm-start initialisation
├── compute_CQI.m                 # Idea 1: Composite Quality Index
├── compute_conflict_entropy.m    # Idea 2: Cross-view belief conflict entropy
├── classify_output.m             # Snap continuous output to nearest label
├── stratified_split.m            # Stratified 80/20 train/test split
├── fun_interval.m                # Training MSE objective (called by CMA-ES)
├── funtest_interval.m            # Evaluation MSE + prediction storage
├── plot_and_save.m               # Publication-quality figure export
├── save_results_xlsx.m           # Excel results export (9 sheets)
└── data/
    ├── View1_LDA_20260327.txt    # LDA feature view
    ├── View4_NCA_20260327.txt    # NCA feature view
    └── View7_SupervisedTSNE_20260327.txt  # Supervised t-SNE feature view
```

---

## Dataset

**SEU Gearbox Fault Diagnosis Dataset** — 5 fault classes:

| Label | Class   | Fault Type       |
| ----- | ------- | ---------------- |
| 0.00  | Surface | Surface spalling |
| 0.25  | Root    | Root crack       |
| 0.50  | Miss    | Missing tooth    |
| 0.75  | Health  | Healthy          |
| 1.00  | Chipped | Chipped tooth    |

Each data file (`.txt`) contains three columns: **[Dim1, Dim2, Label]**, representing a 2D feature embedding and the corresponding fault class label. The three files correspond to LDA, NCA, and Supervised t-SNE projections of the original vibration signals.

> **Data placement:** Place all three `.txt` files inside the `data/` subfolder before running.

---

## Requirements

- **MATLAB** R2019b or later (R2021a+ recommended)
- No additional toolboxes are strictly required.  
  - `exportgraphics` (used in `plot_and_save.m`) requires R2020a+; older versions fall back to `print`.
  - `writetable` (used in `save_results_xlsx.m`) requires the base MATLAB installation.

---

## Quick Start

1. Clone this repository:

   ```bash
   git clone https://github.com/<your-username>/<repo-name>.git
   cd <repo-name>
   ```

2. Place the dataset files into the `data/` folder:

   ```
   data/View1_LDA_20260327.txt
   data/View4_NCA_20260327.txt
   data/View7_SupervisedTSNE_20260327.txt
   ```

3. Open MATLAB, navigate to the repository root, and run:

   ```matlab
   main_meta_interval_brb
   ```

4. Results are saved to `Results_v10/`:

   - **Figures** (PDF / EPS / SVG / EMF / .fig): `Fig1` – `Fig6`
   - **Excel report** (9 sheets): `MetaBRB_Gearbox_Results.xlsx`
   - **Workspace**: `workspace_gearbox_v10.mat`

---

## Key Configuration Parameters

All main parameters are set at the top of `main_meta_interval_brb.m`:

| Parameter        | Default | Description                                    |
| ---------------- | ------- | ---------------------------------------------- |
| `N_INT`          | 5       | Number of intervals per dimension              |
| `N`              | 5       | Number of output belief levels                 |
| `G`              | 200     | CMA-ES maximum generations                     |
| `TRAIN_RATIO`    | 0.80    | Training set proportion                        |
| `RANDOM_SEED`    | 42      | Random seed for reproducibility                |
| `FUSION_MIN_ACC` | 60      | Minimum accuracy (%) for a view to join fusion |

---

## Method Summary

```
For each view (LDA / NCA / Sup. t-SNE):
  1. Compute class-sorted adaptive reference values (compute_adaptive_refs)
  2. Initialise BRB parameters using empirical histograms (build_init_x0)
  3. Optimise parameters with CMA-ES (cmaes → fun_interval)
  4. Run inference on test set (interval_inference → ER_Rule)

Fusion:
  5. Compute CQI for each view; select best view (compute_CQI)
  6. Detect uncertain samples via conflict entropy (compute_conflict_entropy)
  7. Confident samples → use best-view (LDA) prediction directly
  8. Uncertain samples → majority-vote among qualified views
     (tie → fall back to best view; guarantee: accuracy ≥ best single view)
```

---

## Output Description

### Console Output

Training/test MSE and accuracy are printed for every view, followed by the CQI table, conflict entropy statistics, and the final fusion result with a detailed breakdown.

### Figures

| Figure | Content                                         |
| ------ | ----------------------------------------------- |
| Fig1   | (a) Accuracy bar chart  (b) MSE bar chart       |
| Fig2   | CQI composite score per view                    |
| Fig3   | CQI five-dimension breakdown                    |
| Fig4   | Cross-view belief conflict entropy distribution |
| Fig5   | Predicted vs. true scatter (per view + fused)   |
| Fig6   | Belief distribution heatmap (best view)         |

### Excel Sheets

| Sheet               | Content                                                 |
| ------------------- | ------------------------------------------------------- |
| Performance_Summary | Acc / Prec / Rec / F1 / AUC / MSE for all views + fused |
| PerClass_Metrics    | Per-class Prec / Rec / F1 / AUC                         |
| Confusion_Matrices  | Raw counts + row-normalised %                           |
| ROC_AUC             | Per-class AUC + Macro AUC                               |
| View1_LDA_Pred      | Sample-level predictions + belief degrees               |
| View4_NCA_Pred      | Sample-level predictions + belief degrees               |
| View7_TSNE_Pred     | Sample-level predictions + belief degrees               |
| Conflict_Entropy    | Per-sample conflict score and uncertainty flag          |
| CQI_Dimensions      | CQI five-dimension scores per view                      |

---

## License

This project is released for academic research purposes. Please contact the authors before any commercial use.

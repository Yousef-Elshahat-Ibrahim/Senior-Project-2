# Photometric Classification of Rare Transient Events (Tidal Disruption Events)

A multi-stage ensemble pipeline for detecting rare astronomical events in simulated LSST survey data, under **extreme class imbalance (~5% positive class)** and **sparse, irregular multi-band time-series data**.

While framed as an astrophysics problem, the core challenge — reliably detecting a rare, high-value event inside noisy, incomplete time-series data where false negatives are costly — is the same structural problem behind fraud detection, industrial defect detection, rare-disease screening, and churn prediction. This repo documents the full pipeline: data preprocessing, augmentation, model architecture, and evaluation.

📄 Full write-up with methodology and analysis: [thesis PDF](./main.pdf) *(add path once uploaded to repo)*

---

## Problem

Tidal Disruption Events (TDEs) — a star being torn apart by a supermassive black hole — are one of the rarest classes of transient in wide-field sky surveys. In the dataset used here (MALLORN, a realistic LSST-like simulation):

- TDEs make up **~5% of labelled events**
- Fewer than 150 confirmed examples exist to train on
- TDEs are **photometrically near-identical** to Active Galactic Nuclei (AGN), the dominant class in the data, and to several supernova types
- Light curves are **sparse and asynchronously sampled** across 6 filters — a real observation may have data in one band and a two-week gap in another

Any one of these alone is a standard ML challenge. Together, they rule out a simple "train one classifier" approach.

## Approach

A **4-stage ensemble**, where each model specializes in a sub-problem, rather than asking one network to solve everything at once:

| Stage | Model | Task | Why |
|---|---|---|---|
| 1 | Temporal Convolutional Network (TCN) | Remove AGN (dominant class, main source of confusion) | AGN recall = 0.986 — filters the majority contaminant before TDE detection even starts |
| 2 | CNN | Binary TDE detection (AGN-free) | Learns TDE signature in a cleaner, lower-noise setting |
| 3 | CNN (multiclass) | Distinguish TDEs from the specific classes most often confused with them (SN Ia, SN II, SN IIn, AGN) | Targets the exact false-positive sources identified from Stage 2's errors |
| 4 | CNN (baseline, sign-inverted input) | Standalone TDE detector | Provides a complementary feature representation + a fair baseline to benchmark the ensemble against |

The final model **freezes all four networks**, extracts their learned feature representations, and trains a lightweight dense head on the concatenated features — a transfer-learning ensemble. This keeps the trainable parameter count low relative to the small dataset, reducing overfitting risk.

**7 custom augmentation strategies** were built to combat data scarcity and simulate realistic survey conditions: Gaussian Process light-curve synthesis, noise injection, flux scaling, channel dropout, time masking, flux smoothing, and time shifting.

## Results

Direct comparison on identical validation data (full dataset, all classes):

| Model | F1 | Recall | Precision | Balanced Accuracy |
|---|---|---|---|---|
| Single-model baseline | 0.465 | 0.333 | 0.769 | 0.664 |
| **Ensemble (this pipeline)** | **0.526** | **0.500** | 0.556 | **0.740** |

**The ensemble recovers 50% more true positives than the single-model baseline** (recall: 0.333 → 0.500), at an acceptable precision trade-off. In this domain, a missed event is permanently lost, while a false positive just costs a follow-up check — so the ensemble is deliberately tuned toward sensitivity over precision, and the results reflect that design choice.

Full metrics, confusion matrices, and per-class misclassification analysis are in the thesis write-up.

## Repo Structure

```
├── 01_agn_classifier_tcn.ipynb       # Stage 1: TCN AGN filter
├── 02_tde_binary_classifier.ipynb    # Stage 2: Binary TDE detector (AGN-free)
├── 03_multiclass_classifier.ipynb    # Stage 3: Confusable-class discriminator
├── 04_ensemble_baseline.ipynb        # Stage 4 + baseline + final ensemble
├── xgboost_experiment.ipynb          # Tabular baseline experiment
├── augmentation.py                   # 7 augmentation strategies (importable module)
├── utils.py                          # Shared preprocessing utilities
├── train_test_split.py               # Stratified, leakage-safe splitting
├── Data/                             # (not tracked — see Setup)
├── Saved Models/                     # Trained model weights
├── Results/                          # Metrics, confusion matrices, plots
└── requirements.txt
```

*(Notebook filenames above reflect the intended renamed structure — update to match once renamed in the repo.)*

## Setup

```bash
git clone https://github.com/Yousef-Elshahat-Ibrahim/Senior-Project-2.git
cd Senior-Project-2
pip install -r requirements.txt
```

Run notebooks in order (01 → 04) to reproduce the full pipeline from raw data to final ensemble.

## Tech Stack

Python · TensorFlow/Keras · scikit-learn · XGBoost · NumPy · pandas · SciPy (Gaussian Process augmentation)

## Limitations & Future Work

- Trained on <150 confirmed positive examples — the dominant constraint on performance, not model architecture
- Single train/validation split (k-fold was computationally infeasible given per-fold class counts)
- Semi-supervised learning on the 7,000 unlabelled events in the dataset is a natural next step
- See the full write-up for a complete discussion of limitations and proposed extensions (transformer-based architectures, SHAP-based feature importance, real-survey validation)

## License

MIT — see [LICENSE](./LICENSE)

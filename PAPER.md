# Beyond Leaderboards: A Mechanistic Comparison of Tabular Foundation Models

**Authors:** [Your Name] · [Affiliation] · [Contact]
**Code & data:** https://github.com/ayuushmaan/tabfm-why · **Status:** draft v1.0 (October 2026)

---

## Abstract

Tabular foundation models (TFMs) such as TabPFN and TabICL top recent benchmarks, but leaderboard means conceal *why* they win, *where* they don't, and what that implies for practitioners. We conduct a reproducibility-first comparative study on CPU-only hardware: 6 models (TabPFN v2.5, TabICL, LightGBM, RandomForest, linear, MLP) evaluated on 7 TabArena datasets (6 classification, 1 regression; 3 seeds each) plus 8 targeted mechanistic probes — feature interactions, categorical/numerical ablations, missingness (MCAR 0–30%), calibration, permutation-attribution agreement, shortcut exploitation, subgroup recall, and label-noise robustness — with paired significance testing. TabPFN (0.872 mean accuracy) and TabICL (0.868) lead baselines, but the two FMs are statistically indistinguishable (paired p=0.23), and the FM-vs-RandomForest gap (+0.031, p≈0.07) is not significant: it is concentrated in a single dataset (vehicle, +11–21 pp on confusable classes) while forests match FMs elsewhere. Mechanistically, FMs win through (i) genuine interaction modeling (XOR ≈ 0.97–0.99 vs 0.51 linear control), (ii) superior categorical exploitation, (iii) flat missingness degradation (≤0.009 at 30% MCAR vs 0.043 LightGBM), (iv) best-in-class calibration (ECE ≈ 0.035 vs 0.13 LightGBM), (v) minimal shortcut reliance, and (vi) noise immunity (TabPFN +0.030 under 20% label noise vs −0.043 LightGBM). Baselines tie FMs on linear, data-rich, and separable tasks. We derive per-baseline prescriptions (recalibration, categorical-embedding architectures, interaction features, FM distillation) and document a practical deployment finding: current TabPFN distributions require API-gated authentication that breaks headless use. All code, raw results, and plots are released.

## 1. Introduction

Supervised learning on tables remains dominated by gradient-boosted trees, yet in-context tabular foundation models — pretrained on millions of synthetic tables and applied without gradient fitting — now top benchmarks such as TabArena (52 datasets; Hollmann et al., 2025; Ma et al., 2024). Rankings alone, however, cannot guide deployment: a model that wins by 0.5 pp on average but fails on missing data, miscalibrated probabilities, or shortcut features is a liability in medicine and finance. We ask not *which* model wins but *why*, through controlled probes with explicit controls that separate evidence from correlation.

**Contributions.** (1) A head-to-head evaluation of all TFMs accessible in a CPU-only environment (TabPFN v2.5, TabICL) against four baselines on 7 TabArena datasets. (2) Eight mechanistic probes with pre-registered-style controls (chance-level linear baseline on XOR; clean-vs-spurious; clean-vs-noisy). (3) Paired significance testing that overturns naive rank reading (FM-vs-RF is not significant). (4) Actionable, probe-grounded prescriptions per baseline. (5) Full reproducibility artifacts.

## 2. Related Work

**Tabular foundation models.** TabPFN (Hollmann et al., 2023, 2025) frames classification as in-context inference with a transformer pretrained on synthetic structural-causal-model tables; v2 extends to regression and larger scales. TabICL (Qu et al., 2025) adopts column-then-row attention with test-time context scaling. TabDPT, TABLET, UniPredict, and TabuLa-8B explore adjacent points (generative pretraining, language-model backbones) but offer no CPU-compatible open artifact and are excluded here — a limitation we state rather than hide.

**Benchmarks.** TabArena (2025) standardizes 52 datasets, nested CV, and Elo-style ranking; our protocol is a CPU-capped approximation (holdout, 3 seeds, ≤2000 train rows), so absolute numbers are not TabArena-comparable but relative claims within our harness are valid.

**Mechanistic studies.** Prior work reports aggregate wins; calibration of GBDT (Niculescu-Mizil & Caruana, 2005), shortcut learning (Geirhos et al., 2020), and synthetic interaction tests are established in vision/text but rarely applied jointly to TFMs. We port all three to tables.

## 3. Methodology

**Models.** TabPFN v2.5 (`tabpfn` 9.1.0, `n_estimators=4`), TabICL (default), LightGBM (300 trees, lr 0.05), RandomForest (300 trees), LogisticRegression/Ridge (C=1/α=1), MLP (128-64, early stopping). Preprocessing: median/mode imputation everywhere; ordinal encoding for trees/FMs (FMs additionally receive `category` dtype); standardization for linear/MLP.

**Datasets.** Six classification TabArena members — credit-g (finance, mixed), diabetes, breast-w, heart-statlog (medical), vehicle, segment — plus California housing regression; see REPORT.md §2 for shapes. 80/20 stratified split, seeds {0,1,2}.

**Metrics.** Accuracy, balanced accuracy, macro-F1, log-loss, OvR AUC, ECE (10 bins); RMSE/MAE/R² for regression.

**Mechanistic probes.** (P1) XOR-vs-linear synthetic (1200×6); (P2) credit-g full/num-only/cat-only ablations; (P3) MCAR 0/10/20/30% on diabetes; (P4) calibration at matched accuracy; (P5) permutation-importance Spearman agreement; (P6) spurious-column (0.9 train / 0.5 test correlation); (P7) vehicle per-class recall; (P8) 20% label-noise training.

**Statistics.** Paired t and Wilcoxon signed-rank over 18 dataset×seed cells per model pair; 95% CIs on mean differences. Regression (n=3) reported descriptively.

## 4. Results: Rankings

Classification: TabPFN 0.872, TabICL 0.868, RF 0.841, Linear 0.836, LightGBM 0.835, MLP 0.810. Regression RMSE: TabPFN 0.396, TabICL 0.422, LightGBM 0.496, RF 0.579, MLP 0.628, Ridge 0.734. Dataset-wise (Table 2, REPORT.md): FMs dominate vehicle (+11–21 pp) and California; tie elsewhere. **Significance (Appendix): FMs > LightGBM/Linear/MLP (p<0.05); TabPFN≈TabICL (p=0.23); FM-vs-RF n.s. (p≈0.07).**

## 5. Results: Mechanistic Analysis

**P1 Interactions.** XOR: LightGBM 1.00, TabPFN 0.99, RF 0.99, TabICL 0.97, MLP 0.92, Linear 0.51 (chance). FMs possess genuine interaction detectors.
**P2 Type sensitivity.** Removing categoricals costs TabPFN 0.076 vs RF 0.032 — FMs exploit categoricals better, consistent with native mixed-type pretraining.
**P3 Missingness.** 0→30% MCAR drops: TabPFN 0.004, RF/TabICL 0.009, LightGBM 0.043.
**P4 Calibration.** At ≈0.78 accuracy: ECE 0.034–0.035 (FMs) vs 0.130 (LightGBM); log-loss 0.44 vs 0.61.
**P5 Attribution.** FM–FM agreement ρ=0.79; FM–tree 0.44–0.73; FM–linear negative — FMs learn tree-like reliance.
**P6 Shortcuts.** Drops ≤0.030 for all; FMs smallest (0.004–0.017) — no evidence of elevated shortcut cheating.
**P7 Subgroups.** vehicle opel/saab recall: FMs 0.69–0.88 vs trees/MLP 0.50–0.61; bus/van trivial for all. The FM mean advantage is a confusable-class phenomenon.
**P8 Noise.** 20% flips: TabPFN +0.030, TabICL −0.009, LightGBM −0.043 — priors resist memorization.

## 6. Discussion: Why Winners Win

Architecture (row+column attention → mixed-type/interaction handling; in-context inference → no gradient fitting to noise/shortcuts; context ensembling → calibration) plus training (synthetic curricula with missingness, categoricals, diverse causal structures) jointly explain P1–P8. Dataset affinity follows: small-n, mixed-type, overlapping-class, smooth-nonlinear tasks favor FMs; large clean numeric (segment), linear (diabetes), separable (breast-w) tasks erase the gap. The TabPFN>TabICL edge is within noise and attributed only suggestively to broader pretraining.

## 7. Prescriptions for Weaker Models

LightGBM: post-hoc recalibration (mandatory), native-NaN handling, noise-robust losses. MLP: categorical embeddings + row-attention architectures (FT-Transformer/SAINT family), Mixup/label smoothing — expect +5 pp XOR, +15–20 pp hard-class recall. RandomForest: finer leaves, class balancing, FM-soft-label distillation. Linear: explicit interaction features (binding constraint per P1). Deployment rule: <2k-row mixed-type tasks → FM with calibrated-RF challenger; large clean numeric → LightGBM suffices.

## 8. Limitations

Three seeds (CIs honest but wide); CPU-capped sizes favor small-data methods; probes are single-dataset each; no embedding/attention internals available (behavioral proxies only); TabPFN API-gating breaks offline/headless use (documented §0); TabDPT/TabuLa-class models absent. Architecture/training attributions are best-supported explanations, not retraining ablations.

## 9. Reproducibility

`benchmark.py` + `benchmark_tabpfn_only.py` + `mechanistic.py` + `plots.py` + `sigtest.py`; raw CSVs (`results_full.csv`, `results_mech_*.csv`, `results_significance.csv`); 11 plots. Environment pinned in REPORT.md §0.

## References

Hollmann et al. TabPFN: A transformer that solves small tabular classification problems in a second. ICLR 2023.
Hollmann et al. Accurate predictions on small data with a tabular foundation model. Nature 2025.
Ma et al. TabArena. 2024/2025.
Qu et al. TabICL. 2025.
Niculescu-Mizil & Caruana. Predicting good probabilities with supervised learning. ICML 2005.
Geirhos et al. Shortcut learning in deep neural networks. Nature MI 2020.

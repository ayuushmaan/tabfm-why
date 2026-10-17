# RQ2 — what predicts the FM advantage (50 datasets: 45 cls + 5 reg)

## 50-set confirmation: NON-LOCALITY CONFIRMED (n=41, Bonferroni-surviving)

TabPFN-minus-RF gap (wins 33/41): vs lm_1nn ρ=−0.41 (p=0.008), vs tree3 ρ=−0.53 (p=0.0004),
vs logreg ρ=−0.51 (p=0.0006), vs smoothness ρ=−0.37 (p=0.017); tree-minus-linear ρ≈0 (n.s.).
TabPFN-minus-LightGBM (wins 39/41): vs 1NN ρ=−0.62 (p<0.0001), vs smoothness ρ=−0.57
(p=0.0001), vs tree3 ρ=−0.54 (p=0.0003). Strongest p-values survive Bonferroni over 12
comparisons. Claim upgraded: datasets where LOCAL neighborhood structure is weak
(low 1NN/smoothness) show the largest FM advantage — and it is NOT nonlinearity
(tree-minus-linear ≈ 0 twice). FMs combined win 37/43 sets (TabPFN 23, TabICL 14).
Linear LODO-R² still ≤0 (rank signal strong, linear generalization pending — needs
nonlinear meta-model or the custom interaction/axis probes).

## Domain replication: finance + health (16 sets, Oct 2026)

Same analysis on the domain benchmark. Landmarkers are computed by
`scripts/molab/rq2_domain.py` (`results/domain/landmarkers.csv`); the FM advantage is
`mean(FM AUC) - mean(XGBoost, AutoGluon AUC)` from the stored domain CSVs. Spearman:

- vs smoothness(macro): rho = **-0.52 (p=0.037)**; vs 1NN AUC: rho = -0.49 (p=0.057) [n=16].
- Excluding the degenerate `creditcard-fraud` point (XGBoost AUC 0.5 under ~0.2% positives):
  vs 1NN AUC rho = **-0.59 (p=0.020)**; vs smoothness(macro) rho = **-0.59 (p=0.021)** [n=15].
- Nonlinearity control PERSISTS: tree-minus-linear rho = -0.15 (n.s.) -> the signal is
  locality (1NN/smoothness), NOT nonlinearity.
- Reads the same as the main 50-set result (rho ~ -0.6): the FM advantage is largest where
  local neighborhood structure is weakest. The mechanism generalizes to real finance/health data.
- Caveat: n=16 (power limited); the creditcard-fraud outlier is reported, not hidden.

Artifacts: `results/domain/landmarkers{,_finance,_health}.csv`, `results/domain/rq2_domain_corr.csv`,
`figures/domain/rq2_locality.png`.

## Earlier interim (25 sets, superseded but consistent)

## Headline gaps (TabPFN minus baseline, mean over 3 seeds)

- vs RF: mean +0.037, range [−0.002, +0.189]. Biggest wins: balance-scale (+0.189), vehicle (+0.171), dresses-sales (+0.077). NEVER loses (worst −0.002).
- vs LightGBM: mean +0.032. Only loss: tic-tac-toe (−0.016, LightGBM 1.000).
- vs Linear: mean +0.069. Biggest: car (+0.311), tic-tac-toe (+0.309).
- Regression: TabPFN sweeps boston/cpu_act/puma8NH + california.
- FM wins 11/15 new cls sets; baselines take tic-tac-toe (LGBM), iris (Linear tie), mushroom (tie). **CORRECTION (Oct 2026 full roster):** the earlier "letter (RF, FMs absent at 26 classes)" line was wrong — TabPFN-3.5 (0.956) and TabICLv2 (0.941) both run letter cleanly; only LimiX/Mitra are class-capped. See `docs/MOLAB.md`.

## Win prediction: negative

- Binary LODO (beats-RF): 0.65 < majority 0.85. Gap Ridge LODO R² hugely negative (generic
  PyMFE stats, landmarks, and combined). At n=20 with 2–3 outlier gaps, no linear meta-model
  generalizes. RQ2 needs 50+ sets + custom probes (interaction strength, axis-alignment) —
 PyMFE generics are insufficient. This negative is logged, not hidden.

## Descriptive signal (n=20 interim, now CONFIRMED above at n=41 — kept for the record)

FM-minus-baseline gap correlated negatively with 1NN (ρ≈−0.56, p≈0.01) and smoothness
(ρ≈−0.55); tree-minus-linear ≈ 0. The 50-set run confirmed both with stronger p-values.

## TabDPT third-prior column (24 sets, MOLAB `/marimo/roster_TABDPt.csv`)

TabDPT ≈ TabICL tier almost everywhere (vehicle 0.888, balance 0.984, tic-tac-toe 0.984);
mushroom 0.999. **CORRECTION (Oct 2026):** TabDPT does NOT uniquely own letter — TabPFN-3.5
scores 0.956 (above TabDPT's 0.942) and TabICLv2 0.941 (tied) on the 26-class task.
No dataset where real-data prior beats both synthetic priors — the "messy real data" hypothesis
for TabDPT remains unconfirmed on this suite. Regression: boston 2.873 / cpu_act 2.196 /
puma8NH 2.576 (all trail TabPFN).

## Artifacts (MOLAB /marimo, to be pulled to repo)

- `bench25A.csv` (135 rows), `bench25B.csv` (129), `bench25PILOT.csv` (90)
- `metafeatures.csv` (24 sets × PyMFE general/statistical/info-theory)
- `landmarkers.csv` (1NN/stump/tree/logreg/tree-minus-linear/smoothness)
- **Domain track (IN REPO):** `results/domain/landmarkers.csv` + `landmarkers_finance.csv` +
  `landmarkers_health.csv` (16 sets × 3 seeds × 7 landmarkers); `results/domain/rq2_domain_corr.csv`;
  `figures/domain/rq2_locality.png`; runner `scripts/molab/rq2_domain.py`.

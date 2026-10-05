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

## Earlier interim (25 sets, superseded but consistent)

## Headline gaps (TabPFN minus baseline, mean over 3 seeds)

- vs RF: mean +0.037, range [−0.002, +0.189]. Biggest wins: balance-scale (+0.189), vehicle (+0.171), dresses-sales (+0.077). NEVER loses (worst −0.002).
- vs LightGBM: mean +0.032. Only loss: tic-tac-toe (−0.016, LightGBM 1.000).
- vs Linear: mean +0.069. Biggest: car (+0.311), tic-tac-toe (+0.309).
- Regression: TabPFN sweeps boston/cpu_act/puma8NH + california.
- FM wins 11/15 new cls sets; baselines take tic-tac-toe (LGBM), letter (RF, FMs absent at 26 classes), iris (Linear tie), mushroom (tie).

## Win prediction: negative

- Binary LODO (beats-RF): 0.65 < majority 0.85. Gap Ridge LODO R² hugely negative (generic
  PyMFE stats, landmarks, and combined). At n=20 with 2–3 outlier gaps, no linear meta-model
  generalizes. RQ2 needs 50+ sets + custom probes (interaction strength, axis-alignment) —
 PyMFE generics are insufficient. This negative is logged, not hidden.

## Descriptive signal (hypothesis-generating, NOT confirmatory)

FM-minus-baseline gap correlates NEGATIVELY with 1NN landmarker (ρ≈−0.56, p≈0.01) and 5NN
smoothness (ρ≈−0.55, p≈0.01): FMs win biggest where LOCAL neighborhood structure is weakest.
tree-minus-linear gap ≈ 0 (ρ≈+0.12, n.s.) — it is non-LOCALITY, not nonlinearity, that predicts
the advantage. Consistent with the vehicle card (diffuse-global TabPFN beats sharp-lookup TabICL
on confusable classes). Caveat: n=20, 18 comparisons; does not survive strict Bonferroni —
treat as the prior for the 50-set confirmation, correspondent to plan §6 Step 2.

## TabDPT third-prior column (24 sets, MOLAB `/marimo/roster_TABDPt.csv`)

TabDPT ≈ TabICL tier almost everywhere (vehicle 0.888, balance 0.984, tic-tac-toe 0.984);
unique value: letter 0.942 where TabPFN/TabICL cannot run (26 classes), mushroom 0.999.
No dataset where real-data prior beats both synthetic priors — the "messy real data" hypothesis
for TabDPT remains unconfirmed on this suite. Regression: boston 2.873 / cpu_act 2.196 /
puma8NH 2.576 (all trail TabPFN).

## Artifacts (MOLAB /marimo, to be pulled to repo)

- `bench25A.csv` (135 rows), `bench25B.csv` (129), `bench25PILOT.csv` (90)
- `metafeatures.csv` (24 sets × PyMFE general/statistical/info-theory)
- `landmarkers.csv` (1NN/stump/tree/logreg/tree-minus-linear/smoothness)

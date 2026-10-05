# RQ2 interim — what predicts the FM advantage (25 datasets: 21 cls + 4 reg)

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

## Artifacts (MOLAB /marimo, to be pulled to repo)

- `bench25A.csv` (135 rows), `bench25B.csv` (129), `bench25PILOT.csv` (90)
- `metafeatures.csv` (24 sets × PyMFE general/statistical/info-theory)
- `landmarkers.csv` (1NN/stump/tree/logreg/tree-minus-linear/smoothness)

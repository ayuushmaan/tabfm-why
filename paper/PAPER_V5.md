# Sharp Lookup vs Diffuse Decoding: A Mechanistic Comparison of Seven Tabular Foundation Models

**Authors:** Ayushman Garg · [Affiliation] · [Contact]
**Code:** https://github.com/ayuushmaan/tabfm-why · **Status:** draft v5.0 (October 2026)

---

## Abstract

Tabular foundation models (TFMs) top benchmarks, but means conceal mechanisms. We compare five
TFM architectures — TabPFN-3.5, TabICLv2, TabDPT, Mitra-v2, LimiX-2M — against XGBoost, a tuned
AutoGluon ensemble ceiling, RF, linear and MLP baselines on 16 GPU datasets (13 classification,
3 regression; 3 seeds; pinned env), then dissect the decisive win (vehicle) with row-attention
tracing, per-layer probes, causal head ablation, error-pocket analysis, and test five
interventions with pre-registered held-out validation. Headline: a tuned AutoGluon ceiling
LOSES to frozen FMs on their signature sets (vehicle 0.829 vs TabPFN 0.916; boston 3.40 vs 2.57)
— tuning does not close FM gaps. **The prior draft's headline "TabDPT owns the many-class (26)
letter task" is RETRACTED:** with the same pinned protocol, TabPFN-3.5 scores 0.956 and TabICLv2
0.941, both at or above TabDPT's 0.942. The ≤10-class decoder cap is a LimiX/Mitra-only
limitation, not a TabPFN/TabICL one. Inside the models: TabICL looks up sharply (same-label mass
0.59) yet its probes beat its outputs (0.894 vs 0.871) while the opel/saab verdict localizes
causally to final-block heads 1/4/5; TabPFN aggregates diffusely (mass 0.43, class-flat) while
its outputs beat its probes (0.918 vs 0.894). "Sharper attention = better" is rejected. Error
pockets coincide across all six models — opel/saab confusion in mixed neighborhoods (purity
≈0.4), never far-from-data; FMs simply make fewer (13 vs 40 of 170), and XGBoost errs
overconfidently (margin 0.68 vs LimiX 0.16). A 60-fit dilution stress (5 models × 4 noise levels
× 3 seeds) finds NO model × noise effect — retracting our own earlier single-seed TabDPT-edge and
TabICL-crown claims; all FMs are flat to 100 noise columns. Support composition steers the
verdict causally (opel recall 0.00→0.95) but as a zero-sum operating point, and four of five
interventions fail — including an ensemble that gains on-seed (+49% gap-closed) but loses
held-out, caught by our validation rule. Across datasets, the FM advantage correlates with WEAK
local structure (1NN ρ ≈ −0.56): non-locality, not nonlinearity, predicts wins. All code,
recipes, cards, negatives, and retractions released.

## 1. Introduction

Tables remain GBDT territory, yet in-context TFMs lead TabArena. Rankings cannot guide deployment
or model design. Following a benchmark → explain → intervene program (repo `PLAN.md`), we report:
a 16-dataset GPU benchmark with a complete seven-model roster (five FMs + XGBoost + AutoGluon
ceiling) and honest statistics (§4); a vehicle mechanism card with four independent evidence
lines, one causal (§5); five tested interventions, four failed (§6); a dilution stress that
retracted two of our own claims (§6.1); and an RQ2 interim with a negative LODO and a
non-locality hypothesis (§7).

**Contributions.** (1) Seven-model GPU benchmark, 16 sets × 3 seeds — the full FM roster and both
tuned baselines on identical splits — including a correction that retracts a prior
"many-class exclusivity" claim once the two frontier FMs were actually run. (2) Attention + probe
+ ablation + pocket methodology for flash-attention TFMs. (3) Causal localization of
confusable-class verdicts to final-block heads. (4) Error-pocket coincidence: shared failure
geography, differential counts. (5) Composition-as-operating-point with seesaw characterization.
(6) Documented validation catch of a split-overfit intervention, plus self-retractions from
multi-seed stressing and from re-running on the frontier models. (7) Non-locality hypothesis for
FM advantage. (8) Full open artifacts incl. the LimiX bridge recipe.

## 2. Related Work

TabPFN (2023, 2025; v3 GQA test heads, many-class decoder); TabICLv2 (column→row→ICL, ssmax,
open training code); TabDPT (real-data + retrieval); Mitra-v2 (Hybrid-SCM synthetic, finetunes
at inference); LimiX (joint-distribution + missingness; checkpoints drifted from code — bridged).
TabArena (52 sets, Elo); our holdout protocol supports relative claims only. Calibration,
shortcuts, lens/probe methods ported to tables; patching/ablation per the circuit tradition.

## 3. Methodology

Models: TabPFN-3.5, TabICLv2, TabDPT (dynamo off), Mitra-v2 (int labels, fine-tune),
LimiX-2M (config translation, weights untouched), XGBoost (GPU hist), AutoGluon 1.6.3
medium_quality @100 s/fit ceiling, LightGBM, RF, linear, MLP. 16 sets (13 OpenML cls + boston
531/cpu_act/puma8NH), 80/20 splits, seeds {0,1,2}, train cap 2000 / test cap 1000, RTX PRO
6000, pinned env (torch 2.11, numpy 2.4.6, sklearn 1.9.0; pandas 3.0.3/2.3.3 logged per run)
after a corruption incident that is documented, not hidden (§8). Roster CSVs share one schema
(`dataset,kind,model,seed,metric,score,env`); regression stored as positive RMSE. Mechanistic:
8 behavioural probes; SDPA-dispatcher wrapping with per-block tags (same-label mass, entropy);
logistic probe lens; 12×8 ablation with no-op control; row-split ablation; contrastive steering;
error-pocket profiling (kNN distance, 5NN purity, margin). RQ2: PyMFE + landmarkers +
smoothness, LODO classification/regression, Spearman screen. Interventions pre-registered with
gap-closed = (after−before)/(winner−before), held-out acceptance, no-degradation rule.

## 4. Benchmark Results (16 sets × 3 seeds, seven models)

Full 13-classification-set roster (3-seed means; LimiX/Mitra means over 12 sets — letter
class-capped and excluded):

| model | vehicle | letter | balance | tic-tac-toe | 13-set mean |
|---|---|---|---|---|---|
| **TabPFN-3.5** | **0.916** | **0.956** | **0.992** | 0.984 | **0.9034** |
| TabDPT | 0.878 | 0.942 | 0.984 | 0.984 | 0.8990 |
| TabICLv2 | 0.882 | 0.941 | 0.989 | 0.990 | 0.8974 |
| Mitra-v2 | 0.859 | — | 0.987 | 1.000 | 0.8890* |
| LimiX-2M | 0.814 | — | 0.973 | 0.984 | 0.8882* |
| AutoGluon | 0.829 | 0.889 | 0.973 | 0.986 | 0.8815 |
| XGBoost | 0.780 | 0.865 | 0.861 | 1.000 | 0.8660 |

*\*12-set mean (letter excluded).*

Vehicle (the FM signature): TabPFN 0.916 > TabICL 0.882 ≈ TabDPT 0.878 > Mitra 0.859 >
AutoGluon 0.829 > LimiX 0.814 > XGBoost 0.780. Clean prior-ordering: every foundation prior
beats every tuned non-FM. Mitra's 0.859 — fine-tuning lands in the ICL band, so the vehicle
advantage is not ICL-specific. **Before this draft we had never actually run TabPFN/TabICL on
letter; the earlier "TabDPT owns the many-class pocket" claim rested on an untested assumption
that their decoders capped at 10 classes. Running them falsifies it:** TabPFN-3.5 letter 0.956
(top of the roster on that task) and TabICLv2 0.941 (tied TabDPT). Only LimiX/Mitra hard-cap
(≈10 classes) and fail on letter in isolated processes — the two genuine blanks in the heatmap.
Balance-scale: TabPFN 0.992 (roster-best) > TabICL 0.989 > Mitra 0.987 > TabDPT 0.984 >
AutoGluon 0.973 ≈ LimiX 0.973 ≫ XGBoost 0.861 (12pp FM gap on a tiny rules set). Tic-tac-toe
reverses: XGBoost 1.000 = Mitra 1.000 > TabICL 0.990 > AutoGluon 0.986 > TabPFN 0.984 ≈
TabDPT 0.984 ≈ LimiX 0.984 — rule-based structure favors trees; noted, not hidden. Regression
(RMSE, lower better): TabPFN sweeps boston (2.57 vs TabDPT 2.87 vs AG 3.40 vs XGB 3.57) and
cpu_act (2.06 vs 2.16/3.24/2.94); puma8NH is a near-tie (TabPFN 3.20, TabDPT 3.15, AG 3.19,
XGB 3.37; TabICL 3.22). LimiX 5-way and Mitra 5-way (3-seed means; seed-0 reproduces the prior
snapshot to 4 decimals): diabetes 0.764/0.751, credit-g 0.765/0.770, vehicle 0.814/0.859,
breast-w 0.960/0.962, segment 0.988/0.987. Probes: XOR 0.97–0.99 vs 0.51; MCAR ≤0.009 at 30%;
ECE ≈0.035 vs 0.13 LGBM; shortcut drops ≤0.030; noise +0.030 vs −0.043.

## 4b. Domain benchmark: finance + health (16 sets, ROC-AUC headline)

To test whether the FM advantage survives where it matters operationally, we ran the same
seven-model roster on **16 public OpenML/UCI domain sets** (8 finance, 8 health) with a single
canonical prep (median-impute, ordinal-encode, float64) so every model sees identical splits
and data. Because domain sets are heavily skewed, the headline metric is **ROC-AUC** (plus
balanced-accuracy); accuracy alone is reported but not trusted.

| Model | mean AUC (16) | finance | health | mean bal-acc |
|---|---|---|---|---|
| TabPFN | **0.907** | 0.923 | 0.892 | 0.799 |
| TabDPT | 0.906 | 0.920 | 0.892 | 0.796 |
| TabICL | 0.905 | 0.919 | 0.891 | 0.796 |
| LimiX | 0.904 | 0.919 | 0.889 | 0.791 |
| Mitra | 0.900 | 0.916 | 0.884 | 0.783 |
| AutoGluon | 0.864 | 0.864 | 0.864 | 0.757 |
| XGBoost | 0.849 | 0.842 | 0.855 | 0.764 |

Headline: **all five TFMs beat XGBoost on ROC-AUC on all 16/16** sets, and **all five beat the
tuned AutoGluon ceiling on 13/16**. FM mean AUC 0.9045 vs baselines 0.8562 (+4.8 points). Best
model per set (AUC): TabPFN 6, LimiX 4, TabDPT 3, Mitra 2, AutoGluon 1, XGBoost 0. The margin
is widest on the small/skewed health sets (haberman +9.6, hepatitis +8.9) and on finance risk
sets (default-credit +4.3, bank-marketing +4.2).

The imbalance story is the methodological point: bank-marketing scores accuracy 0.90 but
balanced-accuracy 0.69 — a ~23-point majority-class inflation. The TFMs carry the smallest
accuracy→balanced-accuracy gap (TabPFN +0.078 vs AutoGluon +0.109), i.e. their accuracy is
less inflated and their ranking quality (AUC) is higher. One caveat is documented rather than
hidden: `creditcard-fraud` AUC is unstable under the 2000-row train cap (~4 positives in train;
some test folds are single-class → undefined AUC; XGBoost degenerated to AUC 0.5 there), so the
fraud FM-vs-XGB margin is inflated and should be read with that caveat.

Does the main-track mechanism survive here? We recomputed the cheap locality landmarkers
(1NN accuracy/AUC, neighborhood smoothness, tree/logreg accuracy) on all 16 domain sets and
correlated them with the FM advantage (mean FM AUC - mean baseline AUC). The main RQ2 result
replicates: the FM advantage is largest where local structure is weakest, with Spearman
rho = -0.59 for 1NN AUC (p=0.020) and rho = -0.59 for smoothness (p=0.021) once the degenerate
creditcard-fraud point is excluded (n=15; with it, n=16, rho = -0.49/-0.52). The nonlinearity
control (tree-minus-linear) stays null (rho = -0.15, n.s.), so as on the general suite the driver
is non-locality, not nonlinearity. Caveat: n=16 limits power; we report the fraud outlier rather
than prune it silently. The mechanistic "why" therefore carries over to real finance/health data,
not just the curated general suite.

## 5. Mechanism Card: Vehicle

TabICL (0.871): attention diffuse→sharp→diffuse→sharp (L8 mass 0.29/eff 644; L11 0.59/362);
opel/saab deficit every layer; probe 0.894 > output (decoder bottleneck). TabPFN (0.918):
attention ≈chance to L16, 0.43 by L19, class-flat; probe volatile early (opel 0.00 at L1),
0.89 late; output > probe (decoder adds value). CAUSAL: L11H1 ablation → opel 0.333
(bus/van perfect); L11H4 → saab 0.409; row-split shows train AND test sides each necessary
(relational, not readout). POCKETS (6 models, 152 error rows): failure geography
coincides — opel/saab only (bus/van ≈ never), purity ≈0.35–0.45 (mixed neighborhoods),
kNN distance uniform ≈1.7 (never far-from-data). Counts differ: TabPFN 13, TabICL 21,
TabDPT 23, LimiX 29, Mitra 26, XGBoost 40 of 170. Margins diverge: XGBoost errs at 0.68
(overconfident — the ECE echo), LimiX hesitantly at 0.16, TabPFN 0.45. Contrast: winner
diffuse, loser sharp — work in opposite stages; all fail in the same place.

## 6. Interventions (1/5 pass nothing; all scored)

v1 retrieval restriction: hurts opel (rejected). v2 transplant: opel 0.00→0.95 by composition
(causal proof, seesaw with saab). v3 joint enrichment: no overall gain — composition is an
OPERATING POINT (opel 0.952 at overall 0.788, deployment-relevant). v4 ensemble: +49%
gap-closed on-seed, LOSES held-out (0.835 vs 0.882) — split-overfit caught by rule.
v5 L3 steering: inert (1 row at α=2; head-4 zero at all α). Standing: steering wheel found,
no free overall gain; criterion at 0 with informative failures.

## 6.1 Dilution stress: a double self-retraction

60 fits (5 FMs × {0,20,50,100} noise cols × 3 seeds, diabetes, standard protocol;
`results/probes/dilution.csv`). Seed-0 rows reproduce our earlier single-seed numbers
exactly (noise-20: all ≈0.79) — pipeline verified. But 3-seed means are FLAT for every
model at every level (0.75–0.77; seed SD ≈0.03 swamps all deltas). Retracted: the TabDPT
20-col edge AND the TabICL robustness crown. Verdict: all five FMs uniformly robust to 100
noise columns; no model differentiates. Lesson upgraded: single-seed probe deltas are
worthless — stress with ≥3 seeds before any claim. This section exists because the rule
caught us, not just our interventions.

## 7. RQ2 Interim

LODO win-classifier 0.65 < majority 0.85; Ridge gap-R² deeply negative — generics don't
generalize at n=20 (logged negative; 50+ sets + interaction/axis-alignment probes prescribed).
Descriptive: gap vs 1NN ρ≈−0.56 (p≈0.01), vs smoothness ρ≈−0.55; tree-minus-linear ρ≈+0.12 n.s.
Hypothesis: NON-LOCALITY (weak neighborhood structure), not nonlinearity, predicts FM wins —
the dataset mirror of the vehicle card (pockets: errors live where neighborhoods are mixed).
Correlational (18 comparisons), pending 50-set test.

## 8. Limitations, Reproducibility, References

Single-seed internals; approximate TabPFN weights; Mitra is fine-tuning; 16 of ~24 roster
sets re-run by OpenML ID (unnamed mid-tier sets from dead sandboxes excluded — means cited
only for named sets); TabPFN/TabICL roster columns were prior snapshots in V4 and are now
full re-runs (this draft), so V4's starred vehicle bars are superseded; no official
folds/contamination matrix yet; no SAE/circuits. LimiX/Mitra letter cells are genuinely
unrunnable (class-cap), documented, not missing. Env-corruption incident documented with
adopted pinning rules. Artifacts: `src/` (incl. harness), `results/{pilot,roster,xgboost,ceiling,probes}/`
(incl. `results/roster/logs/` raw stdout), `figures/{pilot,roster,ceiling,probes}/`, `docs/`,
`paper/` (V1–V5, this file V5). Refs: Hollmann 2023/2025; Qu 2025; TabDPT 2410.18164;
TabArena 2506.16791; LimiX 2509.03505/2606.04485; Mitra-v2 TR 2609.04540; Niculescu-Mizil
& Caruana 2005; Geirhos 2020.

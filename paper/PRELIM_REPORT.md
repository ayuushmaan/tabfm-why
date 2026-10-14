# The Foundation Model Advantage in Tabular Learning: A Mechanistic Comparison

**Preliminary Analysis Report** — Ayushman Garg, October 2026
**Code & artifacts:** https://github.com/ayuushmaan/tabfm-why

## Abstract

Tabular foundation models (TFMs) top leaderboards, but aggregate means conceal mechanisms.
We compare five TFMs — TabPFN-3.5, TabICLv2, TabDPT, Mitra-v2, LimiX-2M — against XGBoost,
a tuned AutoGluon ensemble ceiling, RandomForest, LightGBM, linear and MLP baselines across
three studies: (1) a 7-dataset TabArena pilot; (2) a 16-dataset GPU roster (13 classification,
3 regression; 3 seeds; pinned environment); (3) a mechanistic dissection of the decisive
vehicle win plus controlled stress probes. The FM advantage is selective, not universal:
it concentrates on confusable class boundaries (vehicle: every FM beats every tuned
non-FM, 0.814–0.924 vs 0.780–0.829), many-class tasks (letter: TabDPT 0.942, unrunnable
for the other four FMs), and smooth regression manifolds, while collapsing to parity on
linear (diabetes), separable (breast-w), and rule-based (tic-tac-toe) tasks. A tuned
AutoGluon ceiling loses to frozen FMs on their signature sets. Internally, TabICL retrieves
sharply (same-label mass 0.59) but is bottlenecked by its readout (probe 0.894 > output
0.871), while TabPFN aggregates diffusely (mass 0.43) with a decoder that adds value
(output 0.918 > probe 0.894); the opel/saab verdict localizes causally to final-block
heads L11H1/L11H4. All six models fail in identical geometric pockets (mixed-neighborhood
opel/saab confusion); FMs simply fail less often (13 vs 40 of 170 errors) and more
calibratedly. A 60-fit dilution stress finds no model-by-noise interaction, retracting two
of our own earlier single-seed claims.

## 1. Introduction

We ask whether the accuracy edge of tabular foundation models over gradient-boosted
decision trees reflects a general property of deep tabular learning or an artifact of
specific non-local geometries and implicit lookup mechanisms. We examine empirical
behavior across three setups:

- **Small-scale benchmarks.** In-context FMs (TabPFN v2.5, TabICL) and tree baselines
  (LightGBM, RandomForest) on canonical TabArena datasets.
- **Full-roster scaling.** Five foundation architectures (TabPFN v3.5, TabICLv2, TabDPT,
  Mitra-v2, LimiX-2M), GPU XGBoost, and a tuned AutoGluon ceiling on 16 OpenML domains.
  Means are reported over 3 seeds with per-row environment pinning; paired significance
  tests cover the 7-set pilot (TabPFN ≈ TabICL, p = 0.23; FM–RF, p ≈ 0.07, n.s.).
- **Mechanistic and causal dissection.** Confusable-domain analysis (vehicle), controlled
  probes (XOR interactions, MCAR missingness, label noise, shortcuts, calibration), and
  internal tracing (SDPA attention, probe lenses, head ablations, error pockets).

Our goal is to characterize how TFMs predict from few examples: which attention heads
matter, how representations evolve across layers, and how class separation develops
layer by layer — comparing sharp lookup-style TabICLv2 against diffuse TabPFN-3.5,
synthetic-prior pretraining against real-data retrieval (TabDPT) and inference-time
fine-tuning (Mitra-v2), and all five against GBDTs and Random Forests.

## 2. Background

Tabular supervised learning has long been dominated by GBDTs, whose axis-aligned
partitions naturally match tabular structure. The "tree inductive bias" hypothesis
(Grinsztajn et al., 2022) holds that this dominance reflects neural networks'
difficulty with sparse features, non-smooth boundaries, and rotation sensitivity. We
operationalize this hypothesis through neighborhood geometry: we argue the emerging
TFM advantage comes from excelling where *local* neighborhood structure is weak —
overlapping class boundaries where GBDTs and nearest-neighbor methods struggle.

While TFMs lead benchmarks, the responsible mechanisms remain poorly understood. We ask
whether TabICLv2's sharp sample-level retrieval supports more effective in-context
reasoning than TabPFN-3.5's diffuse aggregation strategy, and whether synthetic
structural-causal pretraining yields more robust, better-calibrated representations
than empirical retrieval. We combine multi-faceted behavioral probing — synthetic XOR
interactions, MCAR missingness, label noise — with mechanistic interpretability:
wrapped-SDPA attention tracing, layer-wise probe lenses, and causal head ablations.

## 3. Study 1: Baseline (TabArena Pilot)

We benchmarked TabPFN v2.5 and TabICL against classical baselines on 7 TabArena
datasets (6 classification, 1 regression) over multiple seeds.

**Performance.** Both FMs topped aggregate means (TabPFN 0.872 accuracy / 0.396 RMSE;
TabICL 0.868 / 0.422; RF 0.842; LightGBM 0.835). A paired t-test gives p = 0.23:
we cannot reject the null of no difference between the two FMs. The overall FM–RF
gap is likewise non-significant (p ≈ 0.07).

**Inductive signature.** The FM advantage is not uniform: it concentrates on closely
related classes (vehicle, where classical models are near chance on opel/saab) and
collapses to parity on separable (breast-w), linear (diabetes), and data-rich
(segment) tasks.

**Consistency.** Both FMs prioritize near-identical features (Spearman ρ = 0.79) and
produce calibrated probabilities (ECE ≈ 0.035 vs 0.13 for LightGBM), avoiding the
characteristic GBDT overconfidence.

> **Figure 1** (`figures/pilot/plot_rank_accuracy.png`,
> `figures/pilot/plot_dataset_heatmap.png`): accuracy profiles and cross-domain
> heatmap. TabICL and TabPFN are statistically indistinguishable from classical
> baselines except at the confusable vehicle boundary.

## 4. Study 2: The Five-Model Roster (post final run)

We scaled to TabPFN v3.5, TabDPT, TabICL v2, Mitra v2, LimiX-2M, GPU XGBoost, and a
tuned AutoGluon ceiling (medium_quality, 100 s/fit) on 16 datasets × 3 seeds.

**Prior ordering on confusable boundaries.** On vehicle, every foundation prior beats
every tuned non-FM: TabPFN 0.924* > TabDPT 0.878 ≈ TabICL 0.876* > Mitra 0.859 >
AutoGluon 0.829 > LimiX 0.814 > XGBoost 0.780 (*prior snapshot; remainder this
session). Mitra's 0.859 is new: fine-tuning lands in the ICL band, so the advantage
traces to tabular priors, not in-context attention alone.

**Architectural divergence.** The five FMs cluster on outcomes but diverge internally:
TabICLv2 retrieves sharply (mass 0.59) yet is readout-bottlenecked (probe 0.894 >
output 0.871); TabPFN-3.5 aggregates diffusely (mass 0.43) with a value-adding
non-linear decoder (output 0.918 > probe 0.894); LimiX-2M's confirmed strengths are
mushroom (1.000), iris (0.989), tic-tac-toe (0.984) — it led the earlier seed-0
snapshot on diabetes/credit-g but trails on 3-seed means (0.764/0.765), a correction
to our interim report; TabDPT owns the many-class letter task (0.942), which is
unrunnable for the other four FMs (decoder caps ≤10 vs 26 classes) — an architectural
exclusivity, not just a win. Mitra takes balance-scale outright (0.987, the best cell
in the roster) and ties tic-tac-toe/mushroom at 1.000.

**The ceiling loses.** Tuned AutoGluon trails frozen FMs on their signature sets
(vehicle 0.829, letter 0.889, boston 3.40 vs 2.87, balance 0.973 vs 0.984).
Tic-tac-toe reverses (XGBoost and Mitra 1.000 > TabDPT 0.984): rule structure favors
trees — reported, not hidden. Dresses-sales equalizes everything (TabDPT = AutoGluon
= LimiX at 0.613).

> **Figure 2** (`figures/roster/vehicle_leaderboard.png`,
> `figures/roster/roster_heatmap.png`): above, the vehicle hierarchy — a dense FM
> upper cluster with XGBoost trailing; below, the complete 5-model × 13-set heatmap
> (63/65 cells; the two blanks are the provably unrunnable letter/LimiX and
> letter/Mitra cells).

## 5. Study 3: Mechanistic Dissection (Vehicle)

We traced layer-wise representations with linear probes, ablated individual attention
heads (12×8 sweep), mapped all 152 test errors in feature space, and stress-tested
noise tolerance (60 fits, up to 100 irrelevant features).

**No geometric safety.** All six models fail in identical pockets: exclusively
opel/saab confusion at low local purity (≈0.4), never far-from-data (kNN distance
uniform ≈1.7). Counts differ: TabPFN 13, TabICL 21, TabDPT 23, LimiX 29, Mitra 26,
XGBoost 40 of 170. Error margins diverge: XGBoost errs overconfidently (0.68),
LimiX hesitantly (0.16), TabPFN calibrated (0.45).

**Final-block causal bottleneck.** The opel/saab verdict depends entirely on
final-block heads (TabICLv2 L11H1 → opel 0.333 on ablation; L11H4 → saab 0.409);
early layers are causally redundant, and row-split ablations show train and test
sides each necessary — relational comparison, not unilateral readout.

**Dilution invariance.** All five FMs hold accuracy flat to 100 noise columns on
diabetes (0.75–0.77; seed SD ≈ 0.03 swamps all deltas). This 3-seed stress retracts
two of our own earlier single-seed claims (a TabDPT edge and a TabICL crown):
seed-0 rows reproduce the old numbers exactly, proving pipeline fidelity and
single-seed fragility simultaneously.

> **Figure 3** (`figures/probes/pocket_margins.png`,
> `figures/probes/dilution_lines.png`): above, error-margin boxplots (XGBoost
> overconfident, FMs calibrated); below, flat dilution lines with seed-SD bars.

## 6. Conclusion

The evidence supports a narrower claim than FM superiority: a specialized geometric
capability holding across architectures (ICL, retrieval, joint-distribution,
fine-tuning), scale, and tuning budgets. The tabular deep-learning gap is a
non-locality gap — better separation of overlapping, confusable boundaries where
classical models struggle — with documented parity everywhere else, a ceiling that
tuning cannot lift, and retractions where our own early evidence did not survive
stressing.

## Limitations

Single-seed internals; approximate TabPFN weights; Mitra is fine-tuning, not
zero-shot; 16 of ~24 historical roster sets re-run by OpenML ID; no official
folds or contamination matrix; no SAE/circuit analysis. TabPFN/TabICL roster
columns marked * are prior snapshots, not this session's fits.

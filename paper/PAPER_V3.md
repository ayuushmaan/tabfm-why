# Sharp Lookup vs Diffuse Decoding: A Mechanistic Comparison of Five Tabular Foundation Models

**Authors:** Ayushman Garg · [Affiliation] · [Contact]
**Code:** https://github.com/ayuushmaan/tabfm-why · **Status:** draft v3.0 (October 2026)

---

## Abstract

Tabular foundation models (TFMs) top benchmarks, but means conceal mechanisms. We compare five
TFMs — TabPFN-3.5, TabICLv2, TabDPT, Mitra-v2, LimiX-2M — against tree, linear and MLP baselines
on 25 datasets (21 classification, 4 regression), then dissect the decisive win (vehicle) with
row-attention tracing, per-layer probes, and causal head ablation, and test five interventions
with pre-registered held-out validation. TabPFN leads (mean +0.037 over RF, never loses) but
paired tests show TabPFN ≈ TabICL (p = 0.23) and FM–RF non-significant (p ≈ 0.07). Inside the
models: TabICL looks up sharply (same-label mass 0.59) yet its probes beat its outputs
(0.894 vs 0.871) while the opel/saab verdict localizes causally to final-block heads 1/4/5;
TabPFN aggregates diffusely (mass 0.43, class-flat) while its outputs beat its probes
(0.918 vs 0.894). "Sharper attention = better" is rejected. Support composition steers the
verdict causally (opel recall 0.00→0.95) but as a zero-sum operating point, and four of five
interventions fail — including an ensemble that gains on-seed (+49% gap-closed) but loses
held-out, caught by our validation rule. Across datasets, the FM advantage correlates with
WEAK local structure (1NN ρ ≈ −0.56): non-locality, not nonlinearity, predicts wins. All code,
recipes, cards, and negatives released.

## 1. Introduction

Tables remain GBDT territory, yet in-context TFMs lead TabArena. Rankings cannot guide deployment
or model design. Following a benchmark → explain → intervene program (repo `PLAN.md`), we report:
a 25-dataset five-model benchmark with honest statistics (§4); a vehicle mechanism card with
three independent evidence lines, one causal (§5); five tested interventions, four failed (§6);
and an RQ2 interim with a negative LODO and a non-locality hypothesis (§7).

**Contributions.** (1) Five-TFM GPU benchmark, 25 sets, paired tests overturning naive ranks.
(2) Attention + probe + ablation methodology for flash-attention TFMs. (3) Causal localization
of confusable-class verdicts to final-block heads. (4) Composition-as-operating-point with
seesaw characterization. (5) Documented validation catch of a split-overfit intervention.
(6) Non-locality hypothesis for FM advantage. (7) Full open artifacts incl. LimiX bridge recipe.

## 2. Related Work

TabPFN (2023, 2025; v3 GQA test heads, many-class decoder); TabICLv2 (column→row→ICL, ssmax,
open training code); TabDPT (real-data + retrieval); Mitra-v2 (Hybrid-SCM synthetic, finetunes
at inference); LimiX (joint-distribution + missingness; checkpoints drifted from code — bridged).
TabArena (52 sets, Elo); our holdout protocol supports relative claims only. Calibration,
shortcuts, lens/probe methods ported to tables; patching/ablation per the circuit tradition.

## 3. Methodology

Models: TabPFN-3.5 (n_est=4), TabICLv2, TabDPT (dynamo off), Mitra-v2 (int labels, fine-tune),
LimiX-2M (config translation, weights untouched), LightGBM, RF, linear, MLP. 25 sets (OpenML
+ California), 80/20 splits, seeds {0,1,2}, RTX PRO 6000, pinned env
(torch 2.11, numpy 2.4.6, pandas 3.0.3, sklearn 1.9.0, scipy 1.18.0) after a corruption
incident that is documented, not hidden (§8). Mechanistic: 8 behavioural probes; SDPA-dispatcher
wrapping with per-block tags (same-label mass, entropy); logistic probe lens; 12×8 ablation
with no-op control; row-split ablation; contrastive steering. RQ2: PyMFE + landmarkers +
smoothness, LODO classification/regression, Spearman screen. Interventions pre-registered with
gap-closed = (after−before)/(winner−before), held-out acceptance, no-degradation rule.

## 4. Benchmark Results (25 sets)

Means over 21 cls sets: TabPFN top, +0.037 vs RF (range −0.002/+0.189), +0.032 vs LightGBM
(only loss tic-tac-toe −0.016), +0.069 vs Linear (car +0.311). FM wins 11/15 new sets;
baselines take tic-tac-toe (LGBM 1.000), letter (RF; FMs absent at 26 classes), iris (tie).
Regression: TabPFN sweeps all 4. Five-way snapshot: LimiX-2M leads diabetes (0.818), trails
vehicle (0.829 vs 0.924); TabDPT ties TabICL; Mitra 0.77. Significance: TabPFN≈TabICL (0.23);
FM–RF n.s. (0.07); FMs > rest (<0.05). Probes: XOR 0.97–0.99 vs 0.51; MCAR ≤0.009 at 30%;
ECE ≈0.035 vs 0.13 LGBM; shortcut drops ≤0.030; noise +0.030 vs −0.043.

## 5. Mechanism Card: Vehicle

TabICL (0.871): attention diffuse→sharp→diffuse→sharp (L8 mass 0.29/eff 644; L11 0.59/362);
opel/saab deficit every layer; probe 0.894 > output (decoder bottleneck). TabPFN (0.918):
attention ≈chance to L16, 0.43 by L19, class-flat; probe volatile early (opel 0.00 at L1),
0.89 late; output > probe (decoder adds value). CAUSAL: L11H1 ablation → opel 0.333
(bus/van perfect); L11H4 → saab 0.409; row-split shows train AND test sides each necessary
(relational, not readout). Contrast: winner diffuse, loser sharp — work in opposite stages.

## 6. Interventions (1/5 pass nothing; all scored)

v1 retrieval restriction: hurts opel (rejected). v2 transplant: opel 0.00→0.95 by composition
(causal proof, seesaw with saab). v3 joint enrichment: no overall gain — composition is an
OPERATING POINT (opel 0.952 at overall 0.788, deployment-relevant). v4 ensemble: +49%
gap-closed on-seed, LOSES held-out (0.835 vs 0.882) — split-overfit caught by rule.
v5 L3 steering: inert (1 row at α=2; head-4 zero at all α). Standing: steering wheel found,
no free overall gain; criterion at 0 with informative failures.

## 7. RQ2 Interim

LODO win-classifier 0.65 < majority 0.85; Ridge gap-R² deeply negative — generics don't
generalize at n=20 (logged negative; 50+ sets + interaction/axis-alignment probes prescribed).
Descriptive: gap vs 1NN ρ≈−0.56 (p≈0.01), vs smoothness ρ≈−0.55; tree-minus-linear ρ≈+0.12 n.s.
Hypothesis: NON-LOCALITY (weak neighborhood structure), not nonlinearity, predicts FM wins —
the dataset mirror of the vehicle card. Correlational (18 comparisons), pending 50-set test.

## 8. Limitations, Reproducibility, References

Single-seed internals; approximate TabPFN weights; Mitra is fine-tuning; 7→25 sets but no
official folds/contamination matrix yet; no SAE/circuits. Env-corruption incident documented
with adopted pinning rules. Artifacts: harness (`src/harness/`), docs/REPORT.md, MOLAB.md, MECHANISM_CARDS.md,
RQ2.md, CSVs, plots, PAPER_V2.md (this file: PAPER_V3.md). Refs: Hollmann 2023/2025; Qu 2025;
TabDPT 2410.18164; TabArena 2506.16791; LimiX 2509.03505/2606.04485; Mitra-v2 TR 2609.04540;
Niculescu-Mizil & Caruana 2005; Geirhos 2020.

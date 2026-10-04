# Sharp Lookup vs Diffuse Decoding: A Mechanistic Comparison of Five Tabular Foundation Models

**Authors:** [Your Name] · [Affiliation] · [Contact]
**Code:** https://github.com/ayuushmaan/tabfm-why · **Status:** draft v2.0 (October 2026)

---

## Abstract

Tabular foundation models (TFMs) now top benchmarks, but leaderboard means conceal *how* they win and whether the how differs across models. We compare five TFMs — TabPFN-3.5, TabICLv2, TabDPT, Mitra-v2, LimiX-2M — against tree, linear and MLP baselines on seven TabArena datasets, then dissect the largest win (vehicle, 4-class) with row-attention tracing and per-layer linear probes in two models. Mean accuracy ranks TabPFN first (0.872), but paired tests show TabPFN ≈ TabICL (p = 0.23) and the FM–RandomForest gap non-significant (p ≈ 0.07): the advantage is concentrated in confusable classes (vehicle opel/saab: +27–37pp recall). Inside the models we find opposite strategies: TabICL sharpens same-class attention across blocks (final same-label mass 0.59) yet its linear probes beat its own outputs (0.894 vs 0.871) — representation-rich, decoder-poor; TabPFN attends near-uniformly (mass 0.43, flat across classes) while its outputs beat its probes (0.918 vs 0.894) — diffuse aggregation with a strong decoder. "Sharper attention = better model" is rejected on this pair. A dilution stress test further rejects an early hypothesis (TabDPT robustness) in favour of TabICL (flat to 100 noise columns). All code, weights-recipes, raw results and the first mechanism card are released.

## 1. Introduction

Supervised tables remain the domain of gradient-boosted trees, yet in-context TFMs — pretrained on synthetic or real tables and applied without gradient fitting — lead TabArena (Hollmann et al., 2025; Ma et al., 2024). Rankings cannot guide deployment or hiring decisions about these models: a 0.5pp mean win that comes from miscalibrated probabilities or shortcut features is a liability. We ask *which algorithm each TFM effectively runs* on a given dataset and *where it breaks*, following a benchmark → explain → intervene program (full plan in repo `PLAN.md`). This paper reports the completed first two loops: a five-model benchmark with honest significance, and a dual-method mechanism card for the decisive win.

**Contributions.** (1) Five-TFM GPU benchmark (TabPFN-3.5, TabICLv2, TabDPT, Mitra-v2, LimiX-2M) with paired testing that overturns naive rank reading. (2) Row-attention tracing in TabICL (12 blocks) and TabPFN-3.5 (20 blocks) on identical rows, with a hook methodology for flash-attention models. (3) Per-layer linear-probe trajectories for both models. (4) A rejected-hypothesis record (dilution, sharp-attention) demonstrating the program's falsifiability. (5) Open recipes for running all five models, including a LimiX-2M checkpoint/code-version bridge.

## 2. Related Work

**TFMs.** TabPFN (Hollmann et al., 2023, 2025) frames classification as in-context inference over synthetic structural-causal tables; v3 adds GQA test heads and a many-class decoder. TabICLv2 (Qu et al., 2025) uses column→row→ICL stages with scalable softmax and ships training code. TabDPT (Ma et al., 2024) trains on 123 real OpenML datasets with retrieval. Mitra-v2 (AutoGluon) scales synthetic priors with a Hybrid-SCM mix but fine-tunes at inference. LimiX (2025, ICML'26) models joint distributions with missingness, releasing 2M/16M checkpoints whose code drifted from the weights (we bridge it).

**Benchmarks & mechanisms.** TabArena standardizes 52 datasets and Elo ranking; our protocol is a holdout approximation (relative claims valid, absolutes not board-comparable). Calibration (Niculescu-Mizil & Caruana, 2005), shortcut analysis (Geirhos et al., 2020), logit-lens and probe methods are ported here to tables; attention in ICL models follows the induction-head/circuit tradition at behavioural depth (patching is scheduled, §7).

## 3. Methodology

**Models & protocol.** TabPFN-3.5 (`tabpfn` 9.1, n_estimators=4), TabICLv2 (`tabicl` 2.2), TabDPT (Layer6 repo, dynamo disabled), Mitra-v2 (AutoGluon `MitraClassifier`, int labels, fine-tuning budgeted), LimiX-2M (repo @ main + config translation, weights untouched), plus LightGBM, RandomForest, LogisticRegression/Ridge, MLP. Seven TabArena datasets (credit-g, diabetes, breast-w, heart-statlog, vehicle, segment, California housing), 80/20 stratified splits, seeds {0,1,2} (GPU: RTX PRO 6000 Blackwell, fits in 0.1–1.6s). Metrics: accuracy, log-loss, AUC, ECE; RMSE/R² for regression.

**Mechanistic methods.** (i) Eight behavioural probes from our pilot (XOR/linear, cat/num ablations, MCAR 0–30%, calibration, permutation agreement, spurious column, subgroup recall, 20% label noise). (ii) Row-attention tracing: exact post-RoPE/ssmax weights reconstructed by wrapping the SDPA dispatcher (TabICL) and `_batched_scaled_dot_product_attention` (TabPFN) with per-block tags; same-label mass and attention entropy per block/class. (iii) Probe lens: logistic probes on train representations per block, evaluated on test rows, per class. Statistics: paired t + Wilcoxon over 18 dataset×seed cells.

## 4. Benchmark Results

Classification means: TabPFN 0.872, TabICL 0.868, RF 0.841, Linear 0.836, LightGBM 0.835, MLP 0.810; regression RMSE: TabPFN 0.396, TabICL 0.422, LightGBM 0.496. Dataset-wise the FM edge is vehicle (+11–21pp) and California; ties elsewhere (full tables: repo `REPORT.md`). Five-way snapshot (seed 0): LimiX-2M leads diabetes (0.818) but trails vehicle (0.829 vs 0.924); TabDPT ties TabICL; Mitra 0.766–0.775. **Significance: TabPFN≈TabICL (p=0.23); FM–RF n.s. (p≈0.07, wins 8–9/18); FMs > LightGBM/Linear/MLP (p<0.05).** Behavioural probes (pilot, reproduced on GPU): FMs solve XOR (0.97–0.99 vs 0.51 linear), flattest MCAR curves (≤0.009 at 30%), best ECE (≈0.035 vs 0.13 LightGBM), minimal shortcut drops, noise immunity (TabPFN +0.030 at 20% flips vs −0.043 LightGBM).

## 5. Mechanism Card: Vehicle

**TabICL (acc 0.871).** Same-label mass oscillates (0.29–0.56) across 12 blocks — diffuse→sharp→diffuse→sharp, with layer 8 re-broadening (mass 0.29, eff. rows 644) before final sharpening (L11: mass 0.59, eff. 362). Class split persists at every layer: bus/van 0.65–0.68 vs opel/saab 0.48–0.55. Probe lens 0.812→0.894, exceeding model output (0.871): decoder-limited.

**TabPFN-3.5 (acc 0.918).** Test-query attention near-uniform (mass ≈0.25 ≈ chance) through block 16, mild sharpening to 0.43 by block 19, flat across classes (0.42–0.45 each). Probe lens volatile early (opel probe 0.00 at block 1, rebuilt by block 8), 0.87–0.89 late; model output (0.918) exceeds final probe (0.894): decoder adds value.

**Contrast.** The winner attends more diffusely and uniformly; the loser looks up sharply but reads out poorly. Classification work sits in opposite stages. "Sharper attention = better" rejected; the differentiator is downstream (value routing/decoder), motivating causal patching (§7).

## 6. Rejected Hypotheses

(i) *TabDPT is most dilution-robust* (from a 20-noise-column lead, 0.792): rejected at 50–100 columns where TabDPT drops most (−0.030) and TabICL stays flat (+0.009). (ii) *Sharper attention wins* (§5). Both rejections used pre-committed stress tests on held-out variants — the program's falsifiability criterion in action.

## 7. Limitations & Next

Single-seed internals (n=85 test rows); TabPFN weights reconstructed approximately; no patching/SAE/circuits yet; Mitra is fine-tuning (different cost class); 7 datasets for ranking; no contamination matrix or official TabArena folds yet. Immediate next: activation patching on opel/saab rows (causal confirmation), 25-dataset + meta-feature RQ2, then the intervention loop with the gap-closed metric (repo `PLAN.md` Phases 3–5).

## 8. Reproducibility

`https://github.com/ayuushmaan/tabfm-why`: harness (`harness/`), MOLAB recipes (`MOLAB.md`), mechanism cards (`MECHANISM_CARDS.md`), raw CSVs, plots, REPORT.md. LimiX bridge, TabDPT dynamo workaround and Mitra int-label requirement documented. Tokens never committed.

## References

Hollmann et al. TabPFN. ICLR 2023; Nature 2025. Qu et al. TabICLv2. 2025. Ma et al. TabDPT (arXiv 2410.18164); TabArena (arXiv 2506.16791). LimiX (arXiv 2509.03505; 2606.04485). Mitra-v2 Tech Report (arXiv 2609.04540). Niculescu-Mizil & Caruana, ICML 2005. Geirhos et al., Nature MI 2020.

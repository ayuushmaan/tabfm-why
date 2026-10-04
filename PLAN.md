# tabfm-why — Forward Plan (MOLAB GPU track)

## Confirmed hardware (MOLAB)

- **GPU:** NVIDIA RTX PRO 6000 Blackwell, **96 GB VRAM**, single-GPU jobs (no multi-GPU assumed).
- **Max job/session runtime: 6 hours.** All runners shardable + resumable via manifest
  (pending/done/failed `run_id`s); target ≤4h per shard to leave headroom.
- First MOLAB command verifies torch/CUDA sees the GPU (Blackwell needs a recent CUDA/torch
  build — upgrade before anything else if `torch.cuda.is_available()` is false).

Still to confirm: interactive vs scheduled workflow + container/runtime restrictions; gated HF
checkpoint downloads (TabPFN, TabDPT, LimiX) vs manual weight staging.

Starting point: `pilot-v1` is done and pushed (2 TFMs + 4 baselines, 7 datasets, 8 behavioral probes,
paired tests, REPORT.md, PAPER.md). This plan builds on it toward the full research plan (RQ1–RQ5,
intervention success criterion: ≥50% of losing pairs half-closed, validated on held-out data).

## Phase 0 — MOLAB setup + infra hardening (week 1)

- [CONFIRM] GPU model, memory, quota/hours, max job length, shared filesystem paths, container policy.
- Stand up: Python 3.11 env mirror, PyTorch CUDA, `tabicl`, `tabpfn`, `tabdpt` (Layer6 HF), Mitra
  (autogluon HF), LimiX (stableai-org HF), PyMFE, MLflow (or W&B) on shared storage.
- HF auth: `TABPFN_TOKEN` as secret (never in repo); accept Prior Labs + gated-repo licenses once per
  service account. TabuLa/EXAONE entries stay excluded until checkpoint+license verified.
- Refactor harness into adapters: `adapters/{tabpfn,tabicl,tabdpt,mitra,limix,gbdt,mlp}.py` exposing
  `fit/predict/predict_proba/get_activations`; run configs hashed into `run_id`.
  **Status: local core done** — `harness/core.py` (manifest + Parquet store + resume),
  `harness/adapters.py` (registry incl. XGBoost/CatBoost/TabDPT stubs), `harness/run_suite.py`
  (sharded CLI, per-sample predictions); validated on CPU (heart-statlog × Linear/RF + resume).
- Storage: per-sample predictions + metrics in Parquet keyed by `run_id`
  (`runs/predictions/metrics_fold/efficiency/robustness/explanations/datasets/contamination`);
  activations (probe set only: 50 datasets × 256 rows, fp16) in Zarr.
- Gate: one end-to-end run (1 dataset × 2 models) reproduces pilot-v1 numbers within seed noise.

## Phase 1 — Roster + baselines expansion (weeks 2–3)

Order by mechanism value, not Elo:
1. TabDPT (real-data + retrieval — the synthetic-prior contrast), 2. Mitra-v2 (prior-mix isolation),
   3. LimiX-2M (tiny → circuit tractability), 4. RealTabPFN-2.5 vs TabPFN-2.5 matched pair,
   5. TabPFN-3 / 3.5 checkpoints, TabICLv2 (upgrade current TabICL).
Baselines: add XGBoost, CatBoost immediately; then RealMLP, TabM, ModernNCA, EBM.
Log per-model pretraining ranges (rows/cols Modalities) — TabICLv2 300–48K × 2–100 is the template.
- Gate: all models run default-config on the 7 pilot datasets; efficiency (time/mem) logged.

## Phase 2 — Dataset scale + contamination (weeks 3–5)

- TabArena 51 via official `tabarena` package (official folds — board-comparable).
- Domain suites target ~8–15 sets each: industry (C-MAPSS, SECOM, APS), chemistry (ESOL, Tox21,
  MatBench slice), biology (splice, MIC), physics (Higgs/SUSY slices), open medicine (NHANES, Pima,
  Diabetes130), climate/energy, finance (small), security (phishing, jm1). Restricted track
  (MIMIC-IV/eICU/UKB) only under DUAs, aggregate outputs only.
- Contamination: overlap matrix (model × dataset via OpenML IDs + row hashes); every result reported
  all-data AND clean-subset; contaminated wins excluded from mechanism claims.
- Gate: ≥100 datasets benchmarked in default config; Elo + bootstrap CIs computed.

## Phase 3 — RQ1/RQ2: wins + meta-features (weeks 5–7)

- Controlled sweeps on a 20-dataset pilot slice: context rows {100,500,1K,5K,10K,50K,full},
  features {5,20,50,all}, preprocessing (native vs shared quantile+ordinal), TFM estimators {1,4,8,16,32}.
- ~150 meta-features/dataset (PyMFE + custom: H-interaction, axis-alignment, smoothness, noise
  estimates, landmarkers incl. linear-vs-tree gap).
- Pairwise win predictor per model pair (LODO) + metric-gap regression; SHAP → hypothesis table
  accept/reject; sample-level loss clustering (broad vs pocket losses).
- Gate: win predictor beats majority baseline OOD; ≥3 hypotheses accepted/rejected with synthetic
  confirmation (Phase 4 probes).

## Phase 4 — RQ3: internals (weeks 7–10)

Cheapest-first, gated by Phase 3: behavioral algorithm matching (kNN/GP/trees/GBDT agreement) →
logit/tuned lens (convergence layer) → row-attention (effective rows, distance/label correlation) →
column-attention vs permutation importance → linear probes with controls → CKA/rank geometry
(LimiX collapse check) → activation patching on win/loss context pairs → SAEs (LimiX-2M, TabICLv2
first) → end-to-end circuit on LimiX-2M for linear/XOR synthetic, checked on real wins.
Controlled synthetic suite (~1 knob at a time: nonlinearity, rotation, tails, cardinality,
missingness type, irrelevant share) — every mechanism claim must reproduce here.
Output: JSON mechanism cards per decisive (model, dataset) pair.
- Gate: cards for top-5 most frequent loss mechanisms; each backed by ≥2 independent methods
  (e.g. lens + patching, not attention alone).

## Phase 5 — RQ4: interventions (weeks 10–13)

Ladder order per card: L1 input (selection, retrieval context, transforms, encodings, GBDT-leaf
features) → L2 inference (calibration, ensembles) → L3 activations (steering/ablation) →
L4 weights (LoRA/embeddings, domain fine-tune, never test data) → L5 prior augmentation
(TabICL-v1/Mitra open training code).
Rules: pre-register target + datasets; gap-closed = (after−before)/(winner−before); accept only on
held-out same-mechanism datasets with no regression on already-won sets; re-run probes post-fix.
Success criterion: ≥50% of losing pairs half-closed, stratified by loss kind
(concentrated-class vs broad low-signal — expect the latter to resist).
- Gate: criterion met on at least one loss-kind stratum, or a written negative result with cause.

## Phase 6 — Write-up (weeks 13–15)

Paper v2 from PAPER.md skeleton; results dataset release (Parquet predictions + metrics);
domain suites + contamination flags release; intervention library; code release.
Roster freeze date declared; mid-study releases (e.g. post-3.5 models) go to appendix.

## Compute sketch (refine after Phase 0 pilot)

~17 models × ~150 datasets × folds × {default,tuned,ensembled} ≈ 100–400K runs; defaults are
seconds–minutes on GPU, tuning dominates → budget ~3–5K GPU-h benchmark + 1–2K interpretability.
Tune top-8 models only; Phase 2 gate re-estimates from measured per-run costs.

## Top risks

Gated checkpoints (Prior Labs research-only license → non-commercial track, publish predictions not
weights) · contamination from real-data pretraining (overlap matrix) · tuning overruns (cap budgets) ·
probe over-interpretation (controls + patching before claims) · restricted medical DUAs (separate track).

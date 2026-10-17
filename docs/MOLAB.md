# tabfm-why — MOLAB GPU track notes

Hardware: NVIDIA RTX PRO 6000 Blackwell Server Edition (~102 GB), torch 2.11+cu130
(venv later upgraded by autogluon install: torch 2.13, pandas 2.3.3). 6 h/job cap.

## Roster status (all inferencing on GPU)

| Model | Entry | Notes |
|---|---|---|
| TabPFN-3.x (`tabpfn` 9.1) | pip + `TABPFN_TOKEN` | 0.3–0.7 s/fit |
| TabICLv2 (`tabicl` 2.2) | pip, pulls `jingang/TabICL` ckpts | 0.3–0.8 s/fit |
| TabDPT (Layer6) | clone `layer6ai-labs/TabDPT`, `sys.path` to `src/` | needs `TORCHDYNAMO_DISABLE=1` (torch.compile/inductor broken in this env); numpy+int inputs only |
| Mitra-v2 | `autogluon.tabular[mitra]`, `TabularPredictor(hyperparameters={"MITRA": {}})` | numeric-only: pandas3 `StringDtype`/`fillna(downcast)` breaks AutoGluon on categorical frames — ordinal-encode first |
| LimiX-2M (`stable-ai/LimiX-2M`) | clone `limix-ldm/LimiX`, see recipe below | 0.1 s/predict |

## LimiX-2M recipe (weights untouched, schema translated)

Released `LimiX-2M.ckpt` config matches neither `V1.0.1` nor `main` code. Working combo:
repo @ `main`, ckpt config patched in a COPY (`/tmp/limix_cache/LimiX-2M-v101.ckpt` pattern):
`tf_mlp_layer_type="normal"`, `tf_mlp_activation_fuction="gelu"`, `tf_mlp_use_bias=False`,
`tf_attention_layer_type="original"` (False is rejected), `tf_attention_use_bias=False`,
`num_buckets=0`, `model_structure_config={"nlayers": N, "layers": [{"emsize","nhead","hid_dim","arch"} × N]}`,
`encoder_config_x.categorical_features_class_num=100` (vestigial, unsized),
`RBF_config` old→new key rename (`token_embed_dim`→`RBF_token_embed_dim`, etc.;
`RBF_exponent_digits=1`, `RBF_log_base=10`), `encoder_config_y.num_features=num_inputs`.
Config: `config/cls_default_noretrieval.json`. Env: `RANK/WORLD_SIZE/MASTER_ADDR/MASTER_PORT` set.
Gotchas: clear `__pycache__` after any `git checkout` (stale bytecode resurrects deleted modules);
purge `sys.modules` (`inference/model/utils/limix`) between code-version switches in one kernel.

## 5-way snapshot (seed 0, 80/20 split)

| dataset | TabPFN | TabICL | TabDPT | LimiX-2M | Mitra |
|---|---|---|---|---|---|
| diabetes | 0.805 | 0.792 | 0.799 | **0.818** | 0.773 |
| credit-g | **0.800** | 0.795 | 0.790 | 0.785 | (encode fix pending) |
| vehicle | **0.924** | 0.876 | 0.871 | 0.829 | (encode fix pending) |

Dilution probe (diabetes + 20 noise cols): TabDPT 0.792 > TabICL 0.775 > TabPFN 0.771 (borderline, n=231).
LimiX leads noisy-medical, trails on confusable classes — joint-distribution vs ICL prior contrast.

## Follow-up: dilution stress (50/100 noise cols) — TabDPT hypothesis REJECTED

| noise cols | TabPFN | TabICL | TabDPT | LimiX-2M |
|---|---|---|---|---|
| 0 | 0.775 | 0.779 | 0.775 | 0.792 |
| 50 | 0.771 | 0.784 | 0.779 | 0.775 |
| 100 | 0.771 | **0.788** | 0.745 | 0.766 |

The 20-col TabDPT edge did not survive stronger dilution — at 100 cols TabDPT drops most (−0.030)
while TabICL is flat (even +0.009). Verdict: dilution-robustness crown goes to TabICL, not TabDPT.
Lesson logged: borderline Δ≈0.02 effects on n≈231 must be stressed before becoming claims.

## Update: 3-seed dilution rerun (Oct 2026, `/results/probes/dilution.csv`) — STRONGER RETRACTION

60 fits (5 models × 4 noise levels × 3 seeds, standard 80/20 protocol). Seed-0 rows reproduce
the old single-seed pattern almost exactly (noise-20: all ≈0.79; TabDPT 0.792) — pipeline verified.
But 3-seed means are FLAT for every model at every noise level (all 0.75–0.77, seed SD ≈0.03
swamps all deltas). Retracted: not only the TabDPT edge but also the TabICL "crown" — with
proper seeds there is NO model × dilution effect at all on diabetes to 100 noise cols. All five
FMs are uniformly dilution-robust. Lesson upgraded: single-seed probe deltas are worthless;
stress with ≥3 seeds before any claim.

## Update: TabPFN-3.5 + TabICLv2 full 16-set roster (Oct 2026, box 3) — CLASS-CAP CLAIM PARTLY RETRACTED

96 fits (2 models × 16 sets × 3 seeds), **0 failures**, `results/roster/{tabpfn,tabicl}_roster.csv`
(48 rows each), raw stdout in `results/roster/logs/`. Env pinned and logged per row:
torch 2.11.0+cu130, numpy 2.4.6, pandas 3.0.3, sklearn 1.9.0, tabpfn 9.1.0, tabicl 2.2.0.

**Letter (26 classes) RAN on both** — the ≤10-class decoder cap is LimiX/Mitra-only, NOT
TabPFN/TabICL as previously assumed. 3-seed means:

| model | letter | vehicle | balance | 13-set cls mean |
|---|---|---|---|---|
| TabPFN-3.5 | **0.956** | 0.916 | 0.992 | **0.9034** |
| TabICLv2 | 0.941 | 0.882 | 0.989 | 0.8974 |
| TabDPT | 0.942 | 0.878 | 0.984 | 0.8990 |

TabPFN letter (0.956) **beats TabDPT (0.942)** — the "TabDPT owns many-class" claim is
retracted. Regression (RMSE, lower better): TabPFN boston 2.57 / cpu_act 2.06 / puma8NH 3.20;
TabICL 3.07 / 2.13 / 3.22; vs TabDPT 2.87 / 2.16 / 3.15. Both stored as positive RMSE to
match the roster schema (`dataset,kind,model,seed,metric,score,env`).

Heatmap now 7 models × 13 sets; blanks are **letter × LimiX** and **letter × Mitra** only
(class-cap, verified in isolated processes). Vehicle leaderboard un-starred (all real runs).


`TabularPredictor(hyperparameters={"MITRA": {}})` is flaky in this env (read-only array error
inside `sklearn_interface._train_ensemble`, pandas-3/AutoGluon friction). Reliable path: direct
`from autogluon.tabular.models.mitra.sklearn_interface import MitraClassifier`, NumPy inputs,
**int-encoded labels** (`np.bincount` requires ints). Note this is fine-tuning (~50 epochs,
early stopping, ~1–2 min/fit on GPU), not zero-shot ICL — budget accordingly.
Mitra diabetes acc: 0.766–0.775 (same tie band).

## Cross-box reproducibility incident — RESOLVED (env corruption, not torch sensitivity)

Same TabICL 2.2.0 + same ckpt + same vehicle split/seed scored 0.871 (box 1), 0.647 (box 2),
then 0.871 again (box 3, clean). Root cause: box 2 ran a half-applied torch upgrade
(uv reported torch==2.13.0 installed while the kernel still imported 2.11.0 binaries —
franken-env with mismatched binaries/metadata), which silently corrupted numerics.
NOT a torch-version effect: box 3 (torch 2.11.0, numpy 2.4.6, pandas 3.0.3, sklearn 1.9.0,
scipy 1.18.0) reproduces 0.8706 × 3 fits deterministically. The "22pp torch swing" hypothesis
is RETRACTED. Lab rules adopted: (1) pin full env versions at session start and log with every
result; (2) never swap torch mid-session — restart kernel instead; (3) any cross-session
discrepancy triggers an env-diff before any scientific interpretation.

## Domain benchmark — finance + health (Oct 2026, boxes A + B)

Second, domain-specific track: the same 7-model roster on **16 public OpenML/UCI sets**
(8 finance, 8 health) with a **canonical shared prep** (median-impute, ordinal-encode, float64)
so every model sees identical splits/data. New headline metric is **ROC-AUC + balanced-accuracy**
(accuracy alone is misleading on skewed domain sets).

| Model | Box | Entry |
|---|---|---|
| TabPFN / TabICL / XGBoost | B | already installed |
| Mitra / AutoGluon / TabDPT / LimiX | A | `pip install autogluon.tabular[mitra,lightgbm,catboost,fastai]`, clone TabDPT + LimiX, patch ckpt, `faiss-cpu` + `nvtx` + `kditransform` + `einops` |

**Results** (`results/domain/domain_<model>.csv`, 192 rows = 16 sets × 3 seeds × 4 metrics each,
`scripts/molab/domain_run.py` + `domain_sets.py`):

- **All 5 TFMs beat XGBoost on ROC-AUC across all 16/16** sets; **all 5 beat tuned AutoGluon on 13/16**. FM mean AUC 0.9045 vs baselines 0.8562 (+4.8 points).
- Best model per set (AUC): TabPFN 6, LimiX 4, TabDPT 3, Mitra 2, AutoGluon 1, XGBoost 0.
- Finance mean AUC (TabPFN 0.923) > health (0.892); baseline gap is largest on finance (XGB 0.842).
- **Imbalance is the story**: bank-marketing accuracy 0.90 but balanced-acc 0.69 — majority-class inflation of ~23 points. FMs hold the smallest accuracy→balanced-acc gap (TabPFN +0.078 vs AutoGluon +0.109).
- **Caveat (documented):** `creditcard-fraud` AUC is unstable — the 2000-row train cap leaves ~4 positives, so some test folds have a single class (undefined AUC; `tabpfn`/`xgb` each logged 1 NaN; XGB degenerated to AUC 0.5). The FM-vs-XGB fraud margin is inflated for this reason.

**Set swap:** `taiwan-bankruptcy` (OpenML 42395) served a mislabeled/corrupt ARFF
(md5 mismatch) → replaced with verified `banknote-authentication` (1462). The other 15 sets
are the same OpenML IDs used in the main roster where they overlap.

Figures: `figures/domain/{leaderboard,heatmap_auc,imbalance_gap,domain_split}.png`
(`scripts/plots_domain.py`). Logs: `results/domain/logs/`.

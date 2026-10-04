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

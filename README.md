# tabfm-why — Beyond Leaderboards: A Mechanistic Comparison of Tabular Foundation Models

TabPFN ≈ TabICL > baselines — but the gap lives in confusable classes and weak-locality datasets, not everywhere. Five TFMs (TabPFN-3.5, TabICLv2, TabDPT, Mitra-v2, LimiX-2M) vs baselines on 50 datasets, with attention tracing, probe lens, causal ablations, and scored interventions — all with pre-registered held-out validation.

## Key findings

- **Ranked (50 sets):** FMs combined win 37/43; TabPFN never loses to RF (worst −0.002); only loss anywhere is tic-tac-toe to LightGBM (−0.016). TabPFN≈TabICL (p=0.23).
- **Mechanisms:** TabICL looks up sharply but reads out poorly (probe 0.894 > output 0.871); TabPFN aggregates diffusely with a strong decoder (0.918 > 0.894 probe). Opel/saab verdicts localize causally to final-block heads 1/4/5.
- **RQ2:** FM advantage tracks WEAK local structure (vs 1NN ρ≈−0.62, Bonferroni-surviving) — non-locality, not nonlinearity.
- **Interventions:** composition steers verdicts (opel 0.00→0.95) but zero-sum; ensemble +49% on-seed, fails held-out (caught by rule); steering inert. Criterion at 0 with informative failures.
- Full analysis: [`REPORT.md`](REPORT.md) (CPU pilot) · Paper: [`PAPER_V3.md`](PAPER_V3.md) · Cards: [`MECHANISM_CARDS.md`](MECHANISM_CARDS.md) · RQ2: [`RQ2.md`](RQ2.md) · GPU notes: [`MOLAB.md`](MOLAB.md)

## Install

```bash
pip install tabpfn tabicl lightgbm scikit-learn pandas matplotlib scipy
```

TabPFN v2.5+ needs a free Prior Labs key (otherwise every `.fit()` opens a browser login tab):

```bash
# get a key at https://ux.priorlabs.ai/account (accept the license), then:
export TABPFN_TOKEN="<your-key>"   # Windows: $env:TABPFN_TOKEN="<your-key>"
```

## Reproduce

```bash
python benchmark.py                        # main benchmark -> results_main.csv
python benchmark_tabpfn_only.py <dataset>  # per-dataset TabPFN rerun (repeat per dataset)
python mechanistic.py part1                # interaction, cat/num, missingness, calibration
python mechanistic.py part3                # shortcut, subgroup (part2 = attribution)
python plots.py                            # 11 plots -> plot_*.png
python sigtest.py                          # paired tests -> results_significance.csv
```

CPU-only pilot, ~30–60 min total. Protocol: 7 TabArena datasets, 80/20 stratified splits, seeds {0,1,2}, train ≤2000 rows. GPU track (50 sets, 5 TFMs) runs on MOLAB — see [`MOLAB.md`](MOLAB.md) and [`PLAN.md`](PLAN.md).

## Repo layout

| File | What |
|---|---|
| `benchmark.py` / `benchmark_tabpfn_only.py` | CPU pilot harness + TabPFN rerun |
| `mechanistic.py` | 8 mechanistic probes (P1–P8, see REPORT.md §3) |
| `plots.py` / `sigtest.py` | Figures / paired significance tests |
| `harness/` | Resume-aware Parquet suite runner (Phase 0) |
| `results_full.csv`, `results_mech_*.csv`, `results_significance.csv` | Pilot raw results |
| `plot_*.png` | Pilot figures |
| `REPORT.md` / `PAPER.md`–`PAPER_V3.md` | Pilot report / paper drafts (V3 current) |
| `MOLAB.md` / `PLAN.md` / `RQ2.md` / `MECHANISM_CARDS.md` | GPU track: recipes, plan, RQ2, mechanism cards |

## Citation

```bibtex
@misc{tabfmwhy2026,
  title  = {Beyond Leaderboards: A Mechanistic Comparison of Tabular Foundation Models},
  year   = {2026},
  note   = {Code: https://github.com/ayuushmaan/tabfm-why}
}
```

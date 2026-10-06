"""New-result figures (local, from repo CSVs). Outputs to figures/roster|ceiling|probes/."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results")
F = os.path.join(ROOT, "figures")
for d in ["roster", "ceiling", "probes"]:
    os.makedirs(os.path.join(F, d), exist_ok=True)

plt.rcParams.update({"font.size": 9, "figure.dpi": 150})

# ---------- 1. vehicle leaderboard ----------
veh = {"TabPFN*": 0.924, "TabDPT": 0.8784, "TabICL*": 0.876, "Mitra": 0.8588,
       "AutoGluon": 0.8294, "LimiX": 0.8137, "XGBoost": 0.7804}
names = ["TabPFN*", "TabDPT", "TabICL*", "Mitra", "AutoGluon", "LimiX", "XGBoost"]
vals = [veh[n] for n in names]
cols = ["tab:gray" if n.endswith("*") else ("tab:blue" if n in ("TabDPT", "Mitra", "LimiX") else
        ("tab:red" if n == "AutoGluon" else "tab:green")) for n in names]
fig, ax = plt.subplots(figsize=(7, 3.6))
ax.barh(names, vals, color=cols, xerr=[0.03 if n.endswith("*") else 0.015 for n in names],
        capsize=3)
ax.set_xlim(0.7, 0.97)
ax.set_xlabel("accuracy (3-seed mean)")
ax.set_title("Vehicle: every foundation prior beats every tuned non-FM")
fig.text(0.5, 0.01, "*TabPFN/TabICL: prior 5-way snapshot (docs/MOLAB.md); rest: this session, pinned env",
         fontsize=7, color="gray", ha="center")
fig.tight_layout(rect=[0, 0.06, 1, 1])
fig.savefig(os.path.join(F, "roster", "vehicle_leaderboard.png"))
print("vehicle_leaderboard.png")

# ---------- 2. ceiling gaps ----------
ag = pd.read_csv(os.path.join(R, "ceiling", "ag_ceiling.csv"))
td = pd.read_csv(os.path.join(R, "roster", "tabdpt_roster.csv"))
xgb = pd.read_csv(os.path.join(R, "xgboost", "xgb_roster.csv"))
cls_sets = sorted(ag[ag["metric"] == "acc"]["dataset"].unique())
fm_best, ag_m, xgb_m = [], [], []
for s in cls_sets:
    f = td[(td["dataset"] == s)]["score"].mean()
    a = ag[(ag["dataset"] == s)]["score"].mean()
    x = xgb[(xgb["dataset"] == s)]["score"].mean()
    fm_best.append(f)
    ag_m.append(a)
    xgb_m.append(x)
x = np.arange(len(cls_sets))
w = 0.25
fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(x - w, fm_best, w, label="TabDPT (FM)", color="tab:blue")
ax.bar(x, ag_m, w, label="AutoGluon ceiling", color="tab:red")
ax.bar(x + w, xgb_m, w, label="XGBoost", color="tab:green")
ax.set_xticks(x)
ax.set_xticklabels(cls_sets, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("accuracy")
ax.set_title("Tuned ceiling vs FM on 13 classification sets (3-seed means)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(F, "ceiling", "ceiling_gaps.png"))
print("ceiling_gaps.png")

reg_sets = ["boston", "cpu_act", "puma8NH"]
fig, ax = plt.subplots(figsize=(7, 3.4))
xr = np.arange(len(reg_sets))
ax.bar(xr - w, [td[td["dataset"] == s]["score"].mean() for s in reg_sets], w,
       label="TabDPT (FM)", color="tab:blue")
ax.bar(xr, [ag[ag["dataset"] == s]["score"].mean() for s in reg_sets], w,
       label="AutoGluon ceiling", color="tab:red")
ax.bar(xr + w, [xgb[xgb["dataset"] == s]["score"].mean() for s in reg_sets], w,
       label="XGBoost", color="tab:green")
ax.set_xticks(xr)
ax.set_xticklabels(reg_sets)
ax.set_ylabel("RMSE (lower is better)")
ax.set_title("Regression: ceiling trails TabDPT everywhere")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(F, "ceiling", "ceiling_regression.png"))
print("ceiling_regression.png")

# ---------- 3. pocket margins ----------
pk = pd.read_csv(os.path.join(R, "probes", "error_pockets.csv"))
order = ["tabpfn", "tabicl", "tabdpt", "limix", "mitra", "xgboost"]
data = [pk[pk["model"] == m]["margin"].dropna().values for m in order]
fig, ax = plt.subplots(figsize=(7, 3.6))
ax.boxplot(data, tick_labels=order, showfliers=False)
ns = pk.groupby("model").size()
ax.set_title("Margin on errors: XGB overconfident, LimiX hesitant\n(n=" +
             ",".join(f"{m}:{ns[m]}" for m in order) + " of 170)")
ax.set_ylabel("predictive margin (top - runner-up)")
fig.tight_layout()
fig.savefig(os.path.join(F, "probes", "pocket_margins.png"))
print("pocket_margins.png")

# ---------- 4. dilution flat lines ----------
di = pd.read_csv(os.path.join(R, "probes", "dilution.csv"))
fig, ax = plt.subplots(figsize=(6.5, 3.6))
for m in ["tabpfn", "tabicl", "tabdpt", "limix", "mitra"]:
    g = di[di["model"] == m].groupby("noise")["score"]
    mu, sd = g.mean(), g.std()
    ax.errorbar(mu.index, mu.values, yerr=sd.values, marker="o", label=m, capsize=3)
ax.set_xlabel("noise columns added (diabetes)")
ax.set_ylabel("accuracy (3-seed mean ± SD)")
ax.set_title("Dilution stress: all FMs flat to 100 noise cols")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(F, "probes", "dilution_lines.png"))
# ---------- 5. full roster heatmap (all models x all cls sets) ----------
td = pd.read_csv(os.path.join(R, "roster", "tabdpt_roster.csv"))
xgb = pd.read_csv(os.path.join(R, "xgboost", "xgb_roster.csv"))
ag = pd.read_csv(os.path.join(R, "ceiling", "ag_ceiling.csv"))
lx = pd.read_csv(os.path.join(R, "roster", "limix_5way.csv"))
mi = pd.read_csv(os.path.join(R, "roster", "mitra_5way.csv"))
models = [("TabDPT", td), ("XGBoost", xgb), ("AutoGluon", ag), ("LimiX", lx), ("Mitra", mi)]
acc = td[td["metric"] == "acc"].groupby("dataset")["score"].mean()
sets = list(acc.index)
mat = np.full((len(sets), len(models)), np.nan)
for j, (_, df) in enumerate(models):
    g = df[df["metric"] == "acc"].groupby("dataset")["score"].mean()
    for i, s in enumerate(sets):
        if s in g.index:
            mat[i, j] = g[s]
fig, ax = plt.subplots(figsize=(7.5, 6.5))
im = ax.imshow(mat, aspect="auto", vmin=0.55, vmax=1.0, cmap="YlGn")
ax.set_xticks(range(len(models)))
ax.set_xticklabels([m for m, _ in models])
ax.set_yticks(range(len(sets)))
ax.set_yticklabels(sets, fontsize=8)
for i in range(len(sets)):
    for j in range(len(models)):
        if not np.isnan(mat[i, j]):
            ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", fontsize=7,
                    color="white" if mat[i, j] > 0.85 else "black")
fig.colorbar(im, ax=ax, label="accuracy (3-seed mean)")
ax.set_title("Roster heatmap: 5 models x 13 sets (blank = model never ran that set)")
fig.tight_layout()
fig.savefig(os.path.join(F, "roster", "roster_heatmap.png"))
print("roster_heatmap.png")
print("ALL FIGURES DONE")

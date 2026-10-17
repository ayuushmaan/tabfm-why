"""Domain (finance + health) benchmark figures. Reads results/domain/domain_*.csv.
Outputs to figures/domain/. Mirrors scripts/plots_new.py conventions.
"""
import glob
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RD = os.path.join(ROOT, "results", "domain")
FD = os.path.join(ROOT, "figures", "domain")
os.makedirs(FD, exist_ok=True)
plt.rcParams.update({"font.size": 9, "figure.dpi": 150})

DISP = {"tabpfn": "TabPFN", "tabicl": "TabICL", "tabdpt": "TabDPT", "mitra": "Mitra",
        "limix": "LimiX", "ag": "AutoGluon", "xgb": "XGBoost"}
ORDER = ["tabpfn", "tabicl", "tabdpt", "mitra", "limix", "ag", "xgb"]
FM = {"tabpfn", "tabicl", "tabdpt", "mitra", "limix"}


def cf(m):
    return "tab:red" if m == "ag" else ("tab:green" if m == "xgb" else "tab:blue")


df = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(RD, "domain_*.csv"))])
df = df[df["model"].isin(ORDER)]
domain_of = df.drop_duplicates("dataset").set_index("dataset")["domain"].to_dict()


def mean_metric(metric):
    d = df[df["metric"] == metric]
    return d.pivot_table(index="dataset", columns="model", values="score", aggfunc="mean")


acc = mean_metric("acc")
bal = mean_metric("bal_acc")
auc = mean_metric("auc")

# ---------- 1. leaderboard: mean AUC + accuracy ----------
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, mat, title in [(axes[0], auc, "Mean ROC-AUC (16 domain sets)"),
                       (axes[1], acc, "Mean accuracy (16 domain sets)")]:
    means = mat.mean().reindex(ORDER)
    stds = mat.std().reindex(ORDER)
    ax.barh([DISP[m] for m in ORDER], means.values, xerr=stds.values, capsize=3,
            color=[cf(m) for m in ORDER])
    ax.set_xlim(0.7, 1.0)
    ax.set_title(title, fontsize=9)
    for i, v in enumerate(means.values):
        ax.text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=7)
    ax.invert_yaxis()
fig.suptitle("Finance + Health domain benchmark: foundation models vs tuned baselines", fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(FD, "leaderboard.png"))
print("leaderboard.png")

# ---------- 2. heatmap (AUC) ----------
sets = list(acc.index)
mat = np.full((len(sets), len(ORDER)), np.nan)
for j, m in enumerate(ORDER):
    if m in auc.columns:
        for i, s in enumerate(sets):
            v = auc.loc[s, m]
            if pd.notna(v):
                mat[i, j] = v
fig, ax = plt.subplots(figsize=(8, 6.5))
im = ax.imshow(mat, aspect="auto", vmin=0.7, vmax=1.0, cmap="YlGn")
ax.set_xticks(range(len(ORDER)))
ax.set_xticklabels([DISP[m] for m in ORDER], rotation=30, ha="right")
ax.set_yticks(range(len(sets)))
ax.set_yticklabels([f"{s}  [{domain_of.get(s,'?')[:4]}]" for s in sets], fontsize=8)
for i in range(len(sets)):
    for j in range(len(ORDER)):
        if not np.isnan(mat[i, j]):
            ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", fontsize=7,
                    color="white" if mat[i, j] > 0.9 else "black")
fig.colorbar(im, ax=ax, label="ROC-AUC (3-seed mean)")
ax.set_title("Domain roster: 7 models x 16 sets (finance + health)")
fig.tight_layout()
fig.savefig(os.path.join(FD, "heatmap_auc.png"))
print("heatmap_auc.png")

# ---------- 3. imbalance gap: accuracy overstates on skewed sets ----------
gap = (acc - bal).mean()  # per model mean (acc - bal_acc)
fig, ax = plt.subplots(figsize=(8, 3.8))
dsgap = (acc - bal).mean(axis=1).sort_values(ascending=False)
cols = ["tab:red" if domain_of.get(s) == "finance" else "tab:purple" for s in dsgap.index]
ax.bar(range(len(dsgap)), dsgap.values, color=cols)
ax.set_xticks(range(len(dsgap)))
ax.set_xticklabels(dsgap.index, rotation=45, ha="right", fontsize=7)
ax.set_ylabel("mean accuracy - mean balanced accuracy")
ax.set_title("Where accuracy misleads: majority-class inflation on imbalanced domain sets\n"
             "(averaged over 7 models; red = finance, purple = health)")
fig.tight_layout()
fig.savefig(os.path.join(FD, "imbalance_gap.png"))
print("imbalance_gap.png")

# ---------- 4. finance vs health split ----------
fin = [s for s in sets if domain_of.get(s) == "finance"]
hea = [s for s in sets if domain_of.get(s) == "health"]
fig, ax = plt.subplots(figsize=(8, 3.8))
w = 0.38
x = np.arange(len(ORDER))
fm_auc = [auc.loc[[s for s in fin if s in auc.index], m].mean() if m in auc.columns else np.nan for m in ORDER]
hm_auc = [auc.loc[[s for s in hea if s in auc.index], m].mean() if m in auc.columns else np.nan for m in ORDER]
ax.bar(x - w / 2, fm_auc, w, label=f"finance (n={len(fin)})", color="tab:orange")
ax.bar(x + w / 2, hm_auc, w, label=f"health (n={len(hea)})", color="tab:cyan")
ax.set_xticks(x)
ax.set_xticklabels([DISP[m] for m in ORDER], rotation=30, ha="right")
ax.set_ylabel("mean ROC-AUC")
ax.set_ylim(0.7, 1.0)
ax.legend(fontsize=8)
ax.set_title("Domain split: mean AUC by vertical")
fig.tight_layout()
fig.savefig(os.path.join(FD, "domain_split.png"))
print("domain_split.png")
print("ALL DOMAIN FIGURES DONE")

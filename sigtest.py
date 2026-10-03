import pandas as pd, numpy as np, math
from scipy.stats import ttest_rel, wilcoxon, t as tdist
PY = r"C:\Users\ayush\OneDrive\Desktop\Theatre\Guido\Projects\TFM4.0"
f = pd.read_csv(PY + "\\results_full.csv")
cl = f[f.task == "class"].copy()
w = cl.pivot_table(index=["dataset", "seed"], columns="model", values="accuracy")
pairs = [("TabPFN","TabICL"),("TabPFN","RandomForest"),("TabPFN","LightGBM"),
         ("TabPFN","Linear"),("TabPFN","MLP"),("TabICL","RandomForest"),
         ("TabICL","LightGBM"),("TabICL","Linear")]
rows = []
print("pair | mean_diff | 95% CI | t_p | wilcoxon_p | wins")
for a, b in pairs:
    d = (w[a] - w[b]).dropna(); n = len(d); md = float(d.mean())
    se = float(d.std(ddof=1)) / math.sqrt(n)
    h = float(tdist.ppf(0.975, n - 1)) * se
    tp = float(ttest_rel(w[a], w[b]).pvalue)
    try:
        wp = float(wilcoxon(w[a], w[b]).pvalue)
    except Exception:
        wp = None
    wins = str(int((w[a] > w[b]).sum())) + "/" + str(n)
    print(a + "-vs-" + b, round(md,4), "[" + str(round(md-h,4)) + "," + str(round(md+h,4)) + "]",
          round(tp,4), wp, wins)
    rows.append({"pair": a + "-vs-" + b, "mean_diff": round(md,4),
                 "ci_lo": round(md-h,4), "ci_hi": round(md+h,4),
                 "t_p": round(tp,4), "wilcox_p": wp, "wins": wins})
pd.DataFrame(rows).to_csv(PY + "\\results_significance.csv", index=False)
print("saved results_significance.csv")
rg = f[f.dataset == "california"].pivot_table(index="seed", columns="model", values="rmse")
print(rg.round(4).to_string())

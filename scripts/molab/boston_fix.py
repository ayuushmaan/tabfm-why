"""Boston fix: re-run dataset 'boston' (OpenML 531) 3 seeds, replace wrong-ID rows."""
import os, sys, time, warnings
warnings.filterwarnings("ignore")
MODEL = sys.argv[1] if len(sys.argv) > 1 else "tabdpt"  # tabdpt | xgb
OUT = {"tabdpt": "/marimo/tabdpt_roster.csv", "xgb": "/marimo/xgb_roster.csv"}[MODEL]
if MODEL == "tabdpt":
    os.environ["TORCHDYNAMO_DISABLE"] = "1"
    sys.path.insert(0, "/marimo/tabdpt/src")

import numpy as np, pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

d = pd.read_csv(OUT)
d = d[d["dataset"] != "boston"]
d.to_csv(OUT, index=False)
print(f"dropped wrong boston rows, {len(d)} remain", flush=True)

b = fetch_openml(data_id=531, as_frame=True, parser="auto")
X = b.data.copy()
for c in X.columns:
    if X[c].dtype == bool:
        X[c] = X[c].astype(int)
    if str(X[c].dtype) in ("object", "category", "string") or X[c].dtype == object:
        from sklearn.preprocessing import OrdinalEncoder
        X[c] = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1).fit_transform(X[c].astype(str))
    else:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0)
Xn = X.to_numpy(dtype=np.float64)
yn = pd.to_numeric(pd.Series(b.target), errors="coerce").fillna(0).to_numpy(dtype=np.float64)

if MODEL == "tabdpt":
    from tabdpt import TabDPTRegressor
else:
    from xgboost import XGBRegressor

for seed in [0, 1, 2]:
    itr, ite = train_test_split(np.arange(len(Xn)), test_size=0.2, random_state=seed)
    rs = np.random.RandomState(seed)
    if len(itr) > 2000:
        itr = rs.choice(itr, 2000, replace=False)
    if len(ite) > 1000:
        ite = rs.choice(ite, 1000, replace=False)
    if MODEL == "tabdpt":
        m = TabDPTRegressor(device="cuda", compile=False, verbose=False)
        m.fit(Xn[itr], yn[itr])
        score = float(np.sqrt(mean_squared_error(yn[ite], m.predict(Xn[ite]))))
    else:
        m = XGBRegressor(device="cuda", n_estimators=1000, random_state=seed, n_jobs=8)
        m.fit(Xn[itr], yn[itr])
        score = float(np.sqrt(mean_squared_error(yn[ite], m.predict(Xn[ite]))))
    pd.DataFrame([{"dataset": "boston", "kind": "reg", "model": MODEL, "seed": seed,
                   "metric": "rmse", "score": score, "env": "fix531"}]).to_csv(OUT, mode="a", header=False, index=False)
    print(f"DONE boston/{seed} rmse={score:.4f}", flush=True)
print("BOSTON FIXED", flush=True)

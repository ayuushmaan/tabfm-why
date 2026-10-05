"""XGBoost roster (Box 2). Same protocol as TabDPT job: 80/20 stratified,
seeds 0,1,2, train cap 2000 / test cap 1000. GPU hist. Resume-aware CSV.
"""
import os, sys, time, warnings
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd
from sklearn.datasets import fetch_openml, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score, mean_squared_error
from xgboost import XGBClassifier, XGBRegressor

OUT = "/marimo/xgb_roster.csv"
SETS = [
    ("credit-g", 31, "cls"), ("diabetes", 37, "cls"), ("breast-w", 15, "cls"),
    ("vehicle", 54, "cls"), ("segment", 36, "cls"), ("heart-statlog", 53, "cls"),
    ("balance-scale", 11, "cls"), ("tic-tac-toe", 50, "cls"),
    ("letter", 6, "cls"), ("mushroom", 24, "cls"), ("iris", 61, "cls"),
    ("car", 40978, "cls"), ("dresses-sales", 23381, "cls"),
    ("boston", 455, "reg"), ("cpu_act", 197, "reg"), ("puma8NH", 225, "reg"),
]
SEEDS = [0, 1, 2]

def load(name, oid, kind):
    try:
        b = fetch_openml(data_id=oid, as_frame=True, parser="auto")
    except Exception as e:
        print(f"SKIP {name}: {e}", flush=True)
        return None, None
    return b.data, b.target

def prep(X, y, kind):
    X = X.copy()
    for c in X.columns:
        if X[c].dtype == bool:
            X[c] = X[c].astype(int)
    cat = [c for c in X.columns if str(X[c].dtype) in ("object", "category", "string") or X[c].dtype == object]
    num = [c for c in X.columns if c not in cat]
    for c in num:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0)
    for c in cat:
        X[c] = X[c].astype(str).fillna("MISSING")
    if cat:
        X[cat] = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1).fit_transform(X[cat])
    Xn = X.to_numpy(dtype=np.float64)
    if kind == "cls":
        y = pd.Series(y).astype(str)
        y = y.map({c: i for i, c in enumerate(sorted(y.unique()))}).to_numpy(dtype=np.int64)
    else:
        y = pd.to_numeric(pd.Series(y), errors="coerce").fillna(0).to_numpy(dtype=np.float64)
    return Xn, y

def main():
    import torch, sklearn, scipy, xgboost
    env = f"torch={torch.__version__} xgb={xgboost.__version__} sklearn={sklearn.__version__} pandas={pd.__version__}"
    print("ENV:", env, flush=True)
    done = set()
    try:
        d = pd.read_csv(OUT)
        done = set(zip(d["dataset"], d["seed"]))
    except FileNotFoundError:
        pass
    t0 = time.time()
    for name, oid, kind in SETS:
        X, y = load(name, oid, kind)
        if X is None:
            continue
        for seed in SEEDS:
            if (name, seed) in done:
                print(f"skip {name}/{seed}", flush=True)
                continue
            Xn, yn = prep(X, y, kind)
            strat = yn if kind == "cls" and len(np.unique(yn)) > 1 else None
            itr, ite = train_test_split(np.arange(len(Xn)), test_size=0.2, random_state=seed, stratify=strat)
            rs = np.random.RandomState(seed)
            if len(itr) > 2000:
                itr = rs.choice(itr, 2000, replace=False)
            if len(ite) > 1000:
                ite = rs.choice(ite, 1000, replace=False)
            Xtr, ytr, Xte, yte = Xn[itr], yn[itr], Xn[ite], yn[ite]
            try:
                if kind == "cls":
                    m = XGBClassifier(device="cuda", n_estimators=500, max_depth=6,
                                      learning_rate=0.05, subsample=0.8, random_state=seed, n_jobs=8)
                    m.fit(Xtr, ytr)
                    score, metric = accuracy_score(yte, m.predict(Xte)), "acc"
                else:
                    m = XGBRegressor(device="cuda", n_estimators=1000, max_depth=6,
                                     learning_rate=0.05, subsample=0.8, random_state=seed, n_jobs=8)
                    m.fit(Xtr, ytr)
                    score, metric = float(np.sqrt(mean_squared_error(yte, m.predict(Xte)))), "rmse"
            except Exception as e:
                print(f"FAIL {name}/{seed}: {type(e).__name__}: {str(e)[:200]}", flush=True)
                continue
            pd.DataFrame([{"dataset": name, "kind": kind, "model": "xgboost", "seed": seed,
                           "metric": metric, "score": score, "env": env}]).to_csv(
                OUT, mode="a", header=not os.path.exists(OUT) or os.path.getsize(OUT) == 0, index=False)
            print(f"DONE {name}/{seed} {metric}={score:.4f} [{time.time()-t0:.0f}s]", flush=True)
    print(f"ALL DONE [{time.time()-t0:.0f}s]", flush=True)

if __name__ == "__main__":
    main()

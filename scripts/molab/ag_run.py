"""AutoGluon ceiling (Box 2). Same splits as roster: 80/20 stratified,
seeds 0,1,2, train cap 2000 / test cap 1000. Raw frames (AutoGluon handles
categoricals natively). presets=medium_quality, time_limit=100/fit.
Resume-aware CSV. Expected ~80-100 min.
"""
import os, sys, time, warnings
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from autogluon.tabular import TabularPredictor

OUT = "/marimo/ag_ceiling.csv"
SETS = [
    ("credit-g", 31, "cls"), ("diabetes", 37, "cls"), ("breast-w", 15, "cls"),
    ("vehicle", 54, "cls"), ("segment", 36, "cls"), ("heart-statlog", 53, "cls"),
    ("balance-scale", 11, "cls"), ("tic-tac-toe", 50, "cls"),
    ("letter", 6, "cls"), ("mushroom", 24, "cls"), ("iris", 61, "cls"),
    ("car", 40978, "cls"), ("dresses-sales", 23381, "cls"),
    ("boston", 531, "reg"), ("cpu_act", 197, "reg"), ("puma8NH", 225, "reg"),
]
SEEDS = [0, 1, 2]

def main():
    import torch, sklearn, autogluon
    env = f"torch={torch.__version__} ag={autogluon.__version__} pandas={pd.__version__}"
    print("ENV:", env, flush=True)
    done = set()
    try:
        d = pd.read_csv(OUT)
        done = set(zip(d["dataset"], d["seed"]))
    except FileNotFoundError:
        pass
    t0 = time.time()
    for name, oid, kind in SETS:
        try:
            b = fetch_openml(data_id=oid, as_frame=True, parser="auto")
        except Exception as e:
            print(f"SKIP {name}: {e}", flush=True)
            continue
        df = b.data.copy()
        df["__y__"] = b.target.values
        for seed in SEEDS:
            if (name, seed) in done:
                print(f"skip {name}/{seed}", flush=True)
                continue
            strat = df["__y__"] if kind == "cls" else None
            itr, ite = train_test_split(np.arange(len(df)), test_size=0.2, random_state=seed, stratify=strat)
            rs = np.random.RandomState(seed)
            if len(itr) > 2000:
                itr = rs.choice(itr, 2000, replace=False)
            if len(ite) > 1000:
                ite = rs.choice(ite, 1000, replace=False)
            train = df.iloc[itr].rename(columns={"__y__": "label"})
            test = df.iloc[ite].rename(columns={"__y__": "label"})
            try:
                pred = TabularPredictor(label="label", path=f"/tmp/ag_{name}_{seed}",
                                        verbosity=0).fit(train, presets="medium_quality",
                                                         time_limit=100).predict(test.drop(columns=["label"]))
                if kind == "cls":
                    score, metric = accuracy_score(test["label"].astype(str), pred.astype(str)), "acc"
                else:
                    score, metric = float(np.sqrt(mean_squared_error(
                        pd.to_numeric(test["label"]), pd.to_numeric(pred)))), "rmse"
            except Exception as e:
                print(f"FAIL {name}/{seed}: {type(e).__name__}: {str(e)[:200]}", flush=True)
                continue
            finally:
                import shutil
                shutil.rmtree(f"/tmp/ag_{name}_{seed}", ignore_errors=True)
            pd.DataFrame([{"dataset": name, "kind": kind, "model": "autogluon", "seed": seed,
                           "metric": metric, "score": score, "env": env}]).to_csv(
                OUT, mode="a", header=not os.path.exists(OUT) or os.path.getsize(OUT) == 0, index=False)
            print(f"DONE {name}/{seed} {metric}={score:.4f} [{time.time()-t0:.0f}s]", flush=True)
    print(f"ALL DONE [{time.time()-t0:.0f}s]", flush=True)

if __name__ == "__main__":
    main()

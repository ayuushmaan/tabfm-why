"""Dilution stress rerun (Box 1): diabetes + {0,20,50,100} Gaussian noise cols.
Models: tabpfn, tabicl, tabdpt, limix (fast) + mitra (fine-tune). 3 seeds.
Resume-aware CSV /marimo/dilution.csv. Verifies: TabICL dilution crown,
TabDPT rejection at 100 cols.
"""
import os, sys, time, warnings
warnings.filterwarnings("ignore")
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "29513"
sys.path.insert(0, "/marimo/tabdpt/src")
sys.path.insert(0, "/marimo/limix")

import numpy as np, pandas as pd, torch
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

OUT = "/marimo/dilution.csv"
NOISE = [0, 20, 50, 100]
SEEDS = [0, 1, 2]

limix_clf = None

def get_limix():
    global limix_clf
    if limix_clf is None:
        from inference.predictor import LimiXPredictor
        limix_clf = LimiXPredictor(device=torch.device("cuda"),
                                   model_path="/tmp/limix_cache/LimiX-2M-v101.ckpt",
                                   inference_config="/marimo/limix/config/cls_default_noretrieval.json")
    return limix_clf

def run_model(model, Xtr, ytr, Xte, yte, seed):
    if model == "tabpfn":
        from tabpfn import TabPFNClassifier
        m = TabPFNClassifier(device="cuda")
        m.fit(Xtr, ytr)
        return accuracy_score(yte, m.predict(Xte))
    if model == "tabicl":
        from tabicl import TabICLClassifier
        m = TabICLClassifier(device="cuda")
        m.fit(Xtr, ytr)
        return accuracy_score(yte, m.predict(Xte))
    if model == "tabdpt":
        from tabdpt import TabDPTClassifier
        m = TabDPTClassifier(device="cuda", compile=False, verbose=False)
        m.fit(Xtr.astype(np.float64), ytr.astype(np.int64))
        return accuracy_score(yte, m.predict(Xte.astype(np.float64), seed=seed))
    if model == "limix":
        pred = get_limix().predict(Xtr.astype(np.float32), ytr, Xte.astype(np.float32),
                                   task_type="Classification")
        return accuracy_score(yte, np.argmax(np.asarray(pred), axis=1))
    if model == "mitra":
        from autogluon.tabular.models.mitra.sklearn_interface import MitraClassifier
        m = MitraClassifier()
        m.fit(Xtr.astype(np.float32), ytr)
        return accuracy_score(yte, m.predict(Xte.astype(np.float32)))
    raise ValueError(model)

def main():
    env = f"torch={torch.__version__} pandas={pd.__version__}"
    print("ENV:", env, flush=True)
    done = set()
    try:
        d = pd.read_csv(OUT)
        done = set(zip(d["model"], d["noise"], d["seed"]))
    except FileNotFoundError:
        pass
    b = fetch_openml(data_id=37, as_frame=True, parser="auto")
    X0 = b.data.to_numpy(dtype=np.float64)
    y = pd.Series(b.target).astype(str)
    y0 = y.map({c: i for i, c in enumerate(sorted(y.unique()))}).to_numpy(dtype=np.int64)
    t0 = time.time()
    for n in NOISE:
        for seed in SEEDS:
            rng = np.random.RandomState(1000 + seed)
            Xn = np.hstack([X0, rng.randn(len(X0), n)]) if n else X0
            itr, ite = train_test_split(np.arange(len(Xn)), test_size=0.2, random_state=seed,
                                        stratify=y0)
            Xtr, ytr, Xte, yte = Xn[itr], y0[itr], Xn[ite], y0[ite]
            for model in ["tabpfn", "tabicl", "tabdpt", "limix", "mitra"]:
                if (model, n, seed) in done:
                    print(f"skip {model}/{n}/{seed}", flush=True)
                    continue
                try:
                    score = run_model(model, Xtr, ytr, Xte, yte, seed)
                except Exception as e:
                    print(f"FAIL {model}/{n}/{seed}: {type(e).__name__}: {str(e)[:200]}", flush=True)
                    continue
                pd.DataFrame([{"model": model, "noise": n, "seed": seed,
                               "metric": "acc", "score": score, "env": env}]).to_csv(
                    OUT, mode="a", header=not os.path.exists(OUT) or os.path.getsize(OUT) == 0, index=False)
                print(f"DONE {model}/noise{n}/{seed} acc={score:.4f} [{time.time()-t0:.0f}s]", flush=True)
    print(f"ALL DONE [{time.time()-t0:.0f}s]", flush=True)

if __name__ == "__main__":
    main()

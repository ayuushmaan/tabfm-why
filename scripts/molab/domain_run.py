"""Domain benchmark runner (MOLAB GPU). One process per model for env isolation.

Usage: python domain_run.py <model>   # tabpfn|tabicl|tabdpt|limix|mitra|xgb|ag

Protocol mirrors the main roster: 80/20 stratified, seeds {0,1,2}, train cap 2000 /
test cap 1000. A single canonical prep (float64, median impute, ordinal-encode) is
shared by every model so all 7 see identical splits + data (fair comparison); models
that want float32 cast at fit time. AutoGluon (the tuned "ceiling") gets raw frames
with the SAME row indices, as in the main roster.

Metrics per (dataset, seed): acc, bal_acc, auc, f1_macro.
Output /marimo/domain_<model>.csv (schema: dataset,domain,kind,model,seed,metric,score,env)
"""
import os
import sys
import time
import warnings

MODEL = sys.argv[1] if len(sys.argv) > 1 else "xgb"

# per-model env + sys.path setup BEFORE heavy imports
if MODEL == "tabdpt":
    os.environ["TORCHDYNAMO_DISABLE"] = "1"
    sys.path.insert(0, "/marimo/tabdpt/src")
if MODEL == "limix":
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29512")
    sys.path.insert(0, "/marimo/limix")

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    f1_score,
)

sys.path.insert(0, "/marimo")
from domain_sets import ALL  # noqa: E402

OUT = f"/marimo/domain_{MODEL}.csv"
SEEDS = [0, 1, 2]
MAX_TRAIN, MAX_TEST = 2000, 1000


def canonical_prep(Xraw, yraw):
    """Identical preprocessing for every model -> fair comparison."""
    X = Xraw.copy()
    for c in X.columns:
        if X[c].dtype == bool:
            X[c] = X[c].astype(int)
    cat = [c for c in X.columns
           if str(X[c].dtype) in ("object", "category", "string") or X[c].dtype == object]
    num = [c for c in X.columns if c not in cat]
    for c in num:
        X[c] = pd.to_numeric(X[c], errors="coerce")
        m = X[c].median() if X[c].notna().any() else 0
        X[c] = X[c].fillna(m)
    for c in cat:
        X[c] = X[c].astype(str).fillna("MISSING")
    if cat:
        X[cat] = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1).fit_transform(X[cat])
    Xn = X.to_numpy(dtype=np.float64)
    ys = pd.Series(yraw).astype(str)
    classes = sorted(ys.unique())
    yn = ys.map({c: i for i, c in enumerate(classes)}).to_numpy(dtype=np.int64)
    return Xn, yn, classes


def split_indices(yn, seed):
    itr, ite = train_test_split(np.arange(len(yn)), test_size=0.2, random_state=seed, stratify=yn)
    rs = np.random.RandomState(seed)
    if len(itr) > MAX_TRAIN:
        itr = rs.choice(itr, MAX_TRAIN, replace=False)
    if len(ite) > MAX_TEST:
        ite = rs.choice(ite, MAX_TEST, replace=False)
    return itr, ite


def metrics_of(yte, probs, pred):
    out = {
        "acc": float(accuracy_score(yte, pred)),
        "bal_acc": float(balanced_accuracy_score(yte, pred)),
        "f1_macro": float(f1_score(yte, pred, average="macro")),
    }
    try:
        k = len(np.unique(yte))
        if k == 2:
            out["auc"] = float(roc_auc_score(yte, probs[:, 1]))
        else:
            out["auc"] = float(roc_auc_score(yte, probs, multi_class="ovr", average="macro"))
    except Exception:
        out["auc"] = float("nan")
    return out


def fit_predict(Xtr, ytr, Xte, seed):
    """Returns (probs, pred) for the requested MODEL."""
    if MODEL == "tabpfn":
        from tabpfn import TabPFNClassifier
        m = TabPFNClassifier(device="cuda")
        m.fit(Xtr, ytr)
        return np.asarray(m.predict_proba(Xte)), np.asarray(m.predict(Xte))
    if MODEL == "tabicl":
        from tabicl import TabICLClassifier
        m = TabICLClassifier(device="cuda")
        m.fit(Xtr, ytr)
        return np.asarray(m.predict_proba(Xte)), np.asarray(m.predict(Xte))
    if MODEL == "tabdpt":
        from tabdpt import TabDPTClassifier
        m = TabDPTClassifier(device="cuda", compile=False, verbose=False)
        m.fit(Xtr, ytr.astype(np.int64))
        pred = np.asarray(m.predict(Xte, seed=seed))
        try:
            probs = np.asarray(m.predict_proba(Xte, seed=seed))
        except Exception:
            k = len(np.unique(ytr))
            probs = np.eye(k)[pred]
        return probs, pred
    if MODEL == "mitra":
        from autogluon.tabular.models.mitra.sklearn_interface import MitraClassifier
        m = MitraClassifier()
        m.fit(Xtr.astype(np.float32), ytr)
        return np.asarray(m.predict_proba(Xte.astype(np.float32))), np.asarray(m.predict(Xte.astype(np.float32)))
    if MODEL == "xgb":
        from xgboost import XGBClassifier
        m = XGBClassifier(device="cuda", n_estimators=500, max_depth=6,
                          learning_rate=0.05, subsample=0.8, random_state=seed, n_jobs=8, verbosity=0)
        m.fit(Xtr, ytr)
        return np.asarray(m.predict_proba(Xte)), np.asarray(m.predict(Xte))
    if MODEL == "limix":
        from inference.predictor import LimiXPredictor
        if not hasattr(fit_predict, "_clf"):
            fit_predict._clf = LimiXPredictor(
                device=torch.device("cuda"),
                model_path="/tmp/limix_cache/LimiX-2M-v101.ckpt",
                inference_config="/marimo/limix/config/cls_default_noretrieval.json",
            )
        probs = np.asarray(fit_predict._clf.predict(
            Xtr.astype(np.float32), ytr, Xte.astype(np.float32), task_type="Classification"))
        return probs, probs.argmax(axis=1)
    raise ValueError(MODEL)


def run_ag(Xtr, ytr, Xte, seed):
    from autogluon.tabular import TabularPredictor
    import shutil
    tr = pd.DataFrame(Xtr)
    te = pd.DataFrame(Xte)
    tr["label"] = ytr
    path = f"/tmp/agdom_{seed}"
    try:
        pred = TabularPredictor(label="label", path=path, verbosity=0).fit(
            tr, presets="medium_quality", time_limit=100).predict(te)
        proba = TabularPredictor.load(path).predict_proba(te).to_numpy()
        return proba, np.asarray(pred)
    finally:
        shutil.rmtree(path, ignore_errors=True)


def env_string():
    import sklearn
    try:
        import importlib.metadata as md
        tp = md.version("tabpfn")
    except Exception:
        tp = "na"
    try:
        import importlib.metadata as md
        ti = md.version("tabicl")
    except Exception:
        ti = "na"
    dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    return (f"model={MODEL} torch={torch.__version__} numpy={np.__version__} "
            f"pandas={pd.__version__} sklearn={sklearn.__version__} tabpfn={tp} tabicl={ti} cuda={dev}")


def main():
    env = env_string()
    print("ENV:", env, flush=True)
    done = set()
    try:
        d = pd.read_csv(OUT)
        done = set(zip(d["dataset"], d["seed"]))
    except Exception:
        pass

    t0 = time.time()
    for name, oid, domain in ALL:
        if all((name, s) in done for s in SEEDS):
            print(f"skip {name} (all seeds done)", flush=True)
            continue
        try:
            b = fetch_openml(data_id=oid, as_frame=True, parser="auto")
            Xn, yn, classes = canonical_prep(b.data, b.target)
        except Exception as e:
            print(f"LOAD-FAIL {name}: {type(e).__name__}: {str(e)[:160]}", flush=True)
            continue
        print(f"{name}: n={len(yn)} f={Xn.shape[1]} k={len(classes)} domain={domain}", flush=True)

        for seed in SEEDS:
            if (name, seed) in done:
                continue
            itr, ite = split_indices(yn, seed)
            Xtr, ytr, Xte, yte = Xn[itr], yn[itr], Xn[ite], yn[ite]
            t1 = time.time()
            try:
                if MODEL == "ag":
                    probs, pred = run_ag(Xtr, ytr, Xte, seed)
                else:
                    probs, pred = fit_predict(Xtr, ytr, Xte, seed)
                mm = metrics_of(yte, probs, pred)
                err = ""
            except Exception as e:
                mm = {k: float("nan") for k in ("acc", "bal_acc", "auc", "f1_macro")}
                err = f"{type(e).__name__}: {str(e)[:160]}"
            dt = round(time.time() - t1, 2)
            rows = [dict(dataset=name, domain=domain, kind="cls", model=MODEL, seed=seed,
                         metric=metric, score=val, env=env) for metric, val in mm.items()]
            pd.DataFrame(rows).to_csv(OUT, mode="a", header=not os.path.exists(OUT), index=False)
            tag = "OK" if not err else f"FAIL {err}"
            print(f"  {name}/{seed} {tag} acc={mm['acc']:.4f} auc={mm['auc']:.4f} "
                  f"bal={mm['bal_acc']:.4f} t={dt}s", flush=True)
    print(f"ALL DONE [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()

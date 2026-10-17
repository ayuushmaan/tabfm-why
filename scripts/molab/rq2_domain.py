"""RQ2 locality landmarkers on the finance/health domain sets (MOLAB).

Usage: python rq2_domain.py <finance|health|all>

Computes, per dataset x seed, cheap sklearn landmarkers that proxy how LOCAL /
smooth the decision boundary is -- the RQ2 predictor of the FM advantage
(see docs/RQ2.md: FM gap vs 1NN accuracy rho = -0.62, p<0.0001):

  lm_1nn_acc         1-nearest-neighbor test accuracy
  lm_1nn_auc         1-nearest-neighbor ROC-AUC (imbalance-robust, k=1)
  smooth_micro       mean fraction of a test point's 5 nearest train neighbors
                     sharing its true label (majority-inflated on skewed sets)
  smooth_macro       same, averaged per class (imbalance-robust)
  lm_tree_acc        decision-tree test accuracy
  lm_logreg_acc      logistic-regression test accuracy
  tree_minus_linear  lm_tree_acc - lm_logreg_acc  (nonlinearity control)

Uses the SAME canonical prep, 80/20 stratified split and caps as domain_run.py,
so the landmarkers align with the stored model AUCs.  Output /marimo/landmarkers_<which>.csv
schema: dataset,domain,seed,metric,score,env
"""
import os
import sys
import time
import warnings

WHICH = sys.argv[1] if len(sys.argv) > 1 else "all"

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

sys.path.insert(0, "/marimo")
from domain_sets import FINANCE, HEALTH, ALL  # noqa: E402

SETS = {"finance": FINANCE, "health": HEALTH, "all": ALL}[WHICH]
OUT = f"/marimo/landmarkers_{WHICH}.csv"
SEEDS = [0, 1, 2]
MAX_TRAIN, MAX_TEST = 2000, 1000


def canonical_prep(Xraw, yraw):
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


def knn_auc(ytr, yte, proba):
    labels = np.unique(ytr)
    try:
        if len(labels) == 2:
            return float(roc_auc_score(yte, proba[:, 1]))
        return float(roc_auc_score(yte, proba, multi_class="ovr", average="macro", labels=labels))
    except Exception:
        return float("nan")


def smoothness(ytr, yte, neighbors):
    same = (ytr[neighbors] == yte[:, None]).mean(axis=1)
    micro = float(same.mean())
    macro_vals = [same[yte == c].mean() for c in np.unique(yte) if (yte == c).any()]
    macro = float(np.mean(macro_vals)) if macro_vals else float("nan")
    return micro, macro


def landmarkers(Xtr, ytr, Xte, yte, seed):
    sc = StandardScaler().fit(Xtr)
    Ztr, Zte = sc.transform(Xtr), sc.transform(Xte)
    out = {}

    nn1 = KNeighborsClassifier(n_neighbors=1).fit(Ztr, ytr)
    p1 = nn1.predict_proba(Zte)
    out["lm_1nn_acc"] = float(accuracy_score(yte, nn1.predict(Zte)))
    out["lm_1nn_auc"] = knn_auc(ytr, yte, p1)

    nnb = NearestNeighbors(n_neighbors=5).fit(Ztr)
    _, idx = nnb.kneighbors(Zte)
    smicro, smacro = smoothness(ytr, yte, idx)
    out["smooth_micro"] = smicro
    out["smooth_macro"] = smacro

    tree = DecisionTreeClassifier(random_state=seed).fit(Xtr, ytr)
    out["lm_tree_acc"] = float(accuracy_score(yte, tree.predict(Xte)))

    lr = LogisticRegression(max_iter=2000, random_state=seed)
    lr.fit(Ztr, ytr)
    out["lm_logreg_acc"] = float(accuracy_score(yte, lr.predict(Zte)))

    out["tree_minus_linear"] = out["lm_tree_acc"] - out["lm_logreg_acc"]
    return out


def env_string():
    import sklearn
    import scipy
    return (f"sklearn={sklearn.__version__} scipy={scipy.__version__} "
            f"numpy={np.__version__} pandas={pd.__version__}")


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
    for name, oid, domain in SETS:
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
            t1 = time.time()
            try:
                mm = landmarkers(Xn[itr], yn[itr], Xn[ite], yn[ite], seed)
                err = ""
            except Exception as e:
                mm = {k: float("nan") for k in
                      ("lm_1nn_acc", "lm_1nn_auc", "smooth_micro", "smooth_macro",
                       "lm_tree_acc", "lm_logreg_acc", "tree_minus_linear")}
                err = f"{type(e).__name__}: {str(e)[:160]}"
            rows = [dict(dataset=name, domain=domain, seed=seed, metric=m, score=v, env=env)
                    for m, v in mm.items()]
            pd.DataFrame(rows).to_csv(OUT, mode="a", header=not os.path.exists(OUT), index=False)
            tag = "OK" if not err else f"FAIL {err}"
            print(f"  {name}/{seed} {tag} 1nn_auc={mm['lm_1nn_auc']:.4f} "
                  f"smooth_macro={mm['smooth_macro']:.4f} tree={mm['lm_tree_acc']:.4f} "
                  f"t={time.time()-t1:.1f}s", flush=True)
    print(f"ALL DONE [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()

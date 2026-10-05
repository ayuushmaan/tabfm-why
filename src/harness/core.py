"""Run lineage + storage layer (Phase 0). Manifest-driven, resumable, Parquet-backed."""
import json, os, time, hashlib
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STORE = os.path.join(REPO_ROOT, "store")

def paths(store=STORE):
    return {
        "manifest": os.path.join(store, "manifest.json"),
        "pred": os.path.join(store, "predictions.parquet"),
        "metrics": os.path.join(store, "metrics.parquet"),
        "runs": os.path.join(store, "runs.parquet"),
    }

def make_run_id(model, dataset, seed, config):
    h = hashlib.sha1(json.dumps(config, sort_keys=True).encode()).hexdigest()[:8]
    return f"{model}__{dataset}__s{seed}__{h}"

class Manifest:
    def __init__(self, store=STORE):
        self.fp = paths(store)["manifest"]
        os.makedirs(store, exist_ok=True)
        self.state = json.load(open(self.fp)) if os.path.exists(self.fp) else {}

    def status(self, rid):
        return self.state.get(rid, {}).get("status", "pending")

    def mark(self, rid, status, **kw):
        d = self.state.get(rid, {})
        d.update({"status": status, "ts": time.time(), **kw})
        self.state[rid] = d
        json.dump(self.state, open(self.fp, "w"), indent=1)

    def append_parquet(self, fp, df):
        if os.path.exists(fp):
            df = pd.concat([pd.read_parquet(fp), df], ignore_index=True)
            df = df.drop_duplicates()
        df.to_parquet(fp, index=False)

    def store_run(self, rid, meta, pred_df, metrics, store=STORE):
        p = paths(store)
        pred_df = pred_df.copy(); pred_df["run_id"] = rid
        m = dict(metrics); m["run_id"] = rid
        r = dict(meta); r["run_id"] = rid
        self.append_parquet(p["pred"], pred_df)
        self.append_parquet(p["metrics"], pd.DataFrame([m]))
        self.append_parquet(p["runs"], pd.DataFrame([r]))
        self.mark(rid, "done")

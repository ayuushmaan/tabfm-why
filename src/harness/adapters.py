"""Model adapters: one factory per model, sklearn-style fit/predict (+proba).
get_activations is a stub for open-weight TFMs (Phase 4 wires hooks)."""
import webbrowser as _wb
_wb.open = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("browser blocked headless"))

def build(name, task="class", seed=0):
    if name == "LightGBM":
        import lightgbm as lgb
        cls = lgb.LGBMClassifier if task == "class" else lgb.LGBMRegressor
        return cls(verbosity=-1, n_estimators=300, learning_rate=0.05, random_state=seed)
    if name == "RandomForest":
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        cls = RandomForestClassifier if task == "class" else RandomForestRegressor
        return cls(n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=seed)
    if name == "Linear":
        from sklearn.linear_model import LogisticRegression, Ridge
        return LogisticRegression(max_iter=2000) if task == "class" else Ridge()
    if name == "MLP":
        from sklearn.neural_network import MLPClassifier, MLPRegressor
        kw = dict(hidden_layer_sizes=(128, 64), early_stopping=True, max_iter=500, random_state=seed)
        return MLPClassifier(**kw) if task == "class" else MLPRegressor(**kw)
    if name == "TabPFN":
        if task == "class":
            from tabpfn import TabPFNClassifier
            return TabPFNClassifier(n_estimators=4, random_state=seed)
        from tabpfn import TabPFNRegressor
        return TabPFNRegressor(n_estimators=4, random_state=seed)
    if name == "TabICL":
        if task == "class":
            from tabicl import TabICLClassifier
            return TabICLClassifier(random_state=seed)
        from tabicl import TabICLRegressor
        return TabICLRegressor(random_state=seed)
    # Phase 1 additions (lazy; fail loudly with install hint if missing)
    if name == "XGBoost":
        import xgboost as xgb
        cls = xgb.XGBClassifier if task == "class" else xgb.XGBRegressor
        return cls(n_estimators=300, learning_rate=0.05, tree_method="hist", random_state=seed)
    if name == "CatBoost":
        from catboost import CatBoostClassifier, CatBoostRegressor
        cls = CatBoostClassifier if task == "class" else CatBoostRegressor
        return cls(iterations=300, learning_rate=0.05, verbose=False, random_seed=seed)
    if name == "TabDPT":
        from tabdpt import TabDPTClassifier, TabDPTRegressor  # Phase 1: verify package/API
        return TabDPTClassifier() if task == "class" else TabDPTRegressor()
    raise KeyError(f"unknown model {name}")

def get_activations(model, X):
    raise NotImplementedError("activation hooks land in Phase 4; behavioral probes only until then")

REGISTRY = ["TabPFN", "TabICL", "LightGBM", "RandomForest", "Linear", "MLP",
            "XGBoost", "CatBoost", "TabDPT"]

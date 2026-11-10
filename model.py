"""Training, prediction and explanations. Plain scikit-learn; no external services."""
import os, time, threading
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, precision_score

from features import FEATURES, LABELS
from synthetic import generate

MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(os.path.dirname(__file__), "..", "models"))
MODEL_FILE = os.path.join(MODEL_DIR, "model.joblib")
MIN_REAL = int(os.environ.get("MIN_REAL_ROWS", "40"))
_lock = threading.Lock()
_state = {"bundle": None}


def _vec(rows):
    return np.array([[np.nan if r.get(f) is None else float(r.get(f)) for f in FEATURES] for r in rows], dtype=float)


def _candidates():
    pre = lambda: [("imp", SimpleImputer(strategy="median", keep_empty_features=True)), ("sc", StandardScaler())]
    return {
        "logistic": Pipeline(pre() + [("clf", LogisticRegression(C=0.15, class_weight="balanced", max_iter=1000))]),
        "forest": Pipeline([("imp", SimpleImputer(strategy="median", keep_empty_features=True)),
                            ("clf", RandomForestClassifier(n_estimators=150, min_samples_leaf=3, class_weight="balanced", random_state=1, n_jobs=1))]),
    }


def _fit(X, y, source):
    fast = source == "sample"
    pos, neg = int(y.sum()), int(len(y) - y.sum())
    k = max(2, min(5, pos, neg))
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=1)
    best = None
    report = {}
    for name, pipe in _candidates().items():
        if fast and name != "logistic":
            continue
        proba = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
        auc = float(roc_auc_score(y, proba))
        pred = (proba >= 0.5).astype(int)
        report[name] = {"auc": round(auc, 3), "accuracy": round(float(accuracy_score(y, pred)), 3),
                        "fail_recall": round(float(recall_score(1 - y, 1 - pred, zero_division=0)), 3),
                        "fail_precision": round(float(precision_score(1 - y, 1 - pred, zero_division=0)), 3)}
        if best is None or auc > best[0]:
            best = (auc, name)
    name = best[1]
    pipe = _candidates()[name].fit(X, y)
    med = np.nanmedian(X, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    return {"pipe": pipe, "name": name, "source": source, "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n": int(len(y)), "pass_rate": round(float(y.mean()), 3), "cv": report, "median": med}


def _save(bundle):
    try:
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(bundle, MODEL_FILE)
    except Exception:
        pass


def load_or_baseline():
    with _lock:
        if _state["bundle"] is not None:
            return _state["bundle"]
        if os.path.exists(MODEL_FILE):
            try:
                _state["bundle"] = joblib.load(MODEL_FILE)
                return _state["bundle"]
            except Exception:
                pass
        X, y = generate()
        _state["bundle"] = _fit(X, y, "sample")
        return _state["bundle"]


def train(rows):
    """rows: list of {features..., label: 0/1}. 1 = passed."""
    y = np.array([int(r["label"]) for r in rows])
    X = _vec(rows)
    if len(rows) < MIN_REAL:
        raise ValueError("Need at least %d finished student-module records, got %d." % (MIN_REAL, len(rows)))
    if y.sum() < 5 or (len(y) - y.sum()) < 5:
        raise ValueError("Need at least 5 passes and 5 fails in the data to learn the difference.")
    with _lock:
        b = _fit(X, y, "real")
        _state["bundle"] = b
        _save(b)
    return summary(b)


def summary(b=None):
    b = b or load_or_baseline()
    cv = b["cv"].get(b["name"], {})
    return {"source": b["source"], "algorithm": b["name"], "trained_at": b["trained_at"], "n_train": b["n"], "pass_rate": b["pass_rate"], "cv": cv, "all_cv": b["cv"]}


def _logit(p):
    p = min(max(float(p), 1e-4), 1 - 1e-4)
    return float(np.log(p / (1 - p)))


def _explain(b, x):
    """Model-agnostic local explanation: replace each feature by the training median and see how the odds of passing move.
    effect > 0 means this value raises the chance of passing, < 0 lowers it (log-odds units)."""
    pipe = b["pipe"]
    base = _logit(pipe.predict_proba(x.reshape(1, -1))[0, 1])
    out = []
    for i, f in enumerate(FEATURES):
        if np.isnan(x[i]):
            continue
        z = x.copy(); z[i] = b["median"][i]
        delta = base - _logit(pipe.predict_proba(z.reshape(1, -1))[0, 1])
        if abs(delta) >= 0.15:
            out.append({"feature": f, "label": LABELS[f], "value": float(x[i]), "effect": round(delta, 2)})
    out.sort(key=lambda d: -abs(d["effect"]))
    return out[:4]


def predict(rows):
    b = load_or_baseline()
    X = _vec(rows)
    res = []
    for i, r in enumerate(rows):
        p = min(0.99, max(0.01, float(b["pipe"].predict_proba(X[i].reshape(1, -1))[0, 1])))
        risk = "high" if p < 0.4 else ("medium" if p < 0.65 else "low")
        res.append({"key": r.get("key"), "pass_probability": round(p, 3), "risk": risk, "factors": _explain(b, X[i])})
    return res

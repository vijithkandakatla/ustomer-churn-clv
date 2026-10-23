"""Who will churn? Classification with cross-validation and business lift metrics."""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def get_models(pos_weight: float, seed=42) -> dict:
    return {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.5,
                                                                                  class_weight="balanced")),
        "XGBoost": XGBClassifier(n_estimators=400, learning_rate=0.03, max_depth=4, subsample=0.8,
                                 colsample_bytree=0.8, min_child_weight=3, scale_pos_weight=pos_weight,
                                 eval_metric="logloss", random_state=seed, n_jobs=-1),
    }


def lift_at(y, prob, top=0.10) -> tuple[float, float]:
    """Churn rate in the riskiest `top` share vs overall, and share of all churners captured."""
    y, prob = np.asarray(y), np.asarray(prob)
    k = int(np.ceil(top * len(y)))
    idx = np.argsort(-prob)[:k]
    return y[idx].mean() / y.mean(), y[idx].sum() / y.sum()


def cross_validate(X, y, n_splits=5, seed=42):
    """Out-of-fold probabilities for every customer and per-model summary metrics."""
    models = get_models(pos_weight=(1 - y.mean()) / y.mean())
    oof = {m: np.zeros(len(y)) for m in models}
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y):
        for name, model in models.items():
            oof[name][te] = clone(model).fit(X.iloc[tr], y.iloc[tr]).predict_proba(X.iloc[te])[:, 1]
    rows = []
    for name, p in oof.items():
        lift, capture = lift_at(y, p)
        rows.append({"Model": name, "ROC-AUC": roc_auc_score(y, p), "PR-AUC": average_precision_score(y, p),
                     "F1 (thr 0.5)": f1_score(y, p >= 0.5), "Top-10% lift": lift,
                     "Churners caught in top 10%": capture})
    return pd.DataFrame(rows), oof, models

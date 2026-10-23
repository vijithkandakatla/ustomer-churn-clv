"""When will customers churn, and what are they worth?

Survival analysis treats tenure as time and churn as the event. Customers still
active are *censored*: we only know they have lasted at least `tenure` months.
Ignoring censoring, as a plain classifier does, underestimates lifetimes.
"""
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test

from src.data import model_matrix

COX_FEATURES_EXCLUDE = ["tenure", "TotalCharges"]   # tenure is the time axis; TotalCharges ≈ tenure × monthly


def kaplan_meier_by(df: pd.DataFrame, group: str, ax=None):
    fitted = {}
    for level, part in df.groupby(group):
        kmf = KaplanMeierFitter().fit(part["tenure"], part["churn"], label=str(level))
        if ax is not None:
            kmf.plot_survival_function(ax=ax, ci_show=True)
        fitted[level] = kmf
    test = multivariate_logrank_test(df["tenure"], df[group], df["churn"])
    return fitted, test.p_value


def fit_cox(df: pd.DataFrame, penalizer: float = 0.05) -> tuple[CoxPHFitter, pd.DataFrame]:
    X = model_matrix(df, drop_first=True).drop(columns=COX_FEATURES_EXCLUDE)
    data = pd.concat([X, df[["tenure", "churn"]]], axis=1)
    data = data[data["tenure"] > 0]                      # tenure 0 = joined this month, no exposure yet
    cph = CoxPHFitter(penalizer=penalizer).fit(data, duration_col="tenure", event_col="churn")
    return cph, X


def expected_clv(cph: CoxPHFitter, X: pd.DataFrame, df: pd.DataFrame, horizon_months: int = 24,
                 margin: float = 0.30, annual_discount: float = 0.10) -> pd.DataFrame:
    """Expected discounted margin over the next `horizon_months` for *active* customers.

    CLV_i = Σ_k  MonthlyCharges_i × margin × S_i(tenure + k | survived to tenure) / (1 + d)^k
    """
    active = df["churn"] == 0
    Xa, da = X[active], df[active]
    months = np.arange(1, horizon_months + 1)
    surv = cph.predict_survival_function(Xa, times=months, conditional_after=da["tenure"].to_numpy())
    disc = (1 + annual_discount) ** (-months / 12)
    exp_months = surv.to_numpy().T @ disc                          # discounted expected paying months
    out = da[["customerID", "tenure", "Contract", "MonthlyCharges"]].copy()
    out["p_churn_12m"] = 1 - surv.loc[min(12, horizon_months)].to_numpy()
    out["expected_months_24m"] = surv.to_numpy().T.sum(axis=1)
    out["clv_24m"] = out["MonthlyCharges"] * margin * exp_months
    return out


def retention_segments(clv: pd.DataFrame) -> pd.DataFrame:
    """2×2 matrix: value split at the median CLV, risk = top quartile of 12-month churn probability."""
    hi_val = clv["clv_24m"] >= clv["clv_24m"].median()
    hi_risk = clv["p_churn_12m"] >= clv["p_churn_12m"].quantile(0.75)
    seg = np.select([hi_val & hi_risk, ~hi_val & hi_risk, hi_val & ~hi_risk],
                    ["1. High value, high risk → retain now", "2. Low value, high risk → low-cost nudge",
                     "3. High value, low risk → nurture / upsell"], "4. Low value, low risk → standard")
    clv = clv.assign(segment=seg)
    g = clv.groupby("segment").agg(customers=("customerID", "size"), avg_p_churn_12m=("p_churn_12m", "mean"),
                                   avg_clv=("clv_24m", "mean"), total_clv=("clv_24m", "sum"))
    g["clv_at_risk_12m"] = clv.assign(r=clv["clv_24m"] * clv["p_churn_12m"]).groupby("segment")["r"].sum()
    return g.reset_index(), clv

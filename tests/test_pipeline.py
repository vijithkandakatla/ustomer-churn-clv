import numpy as np
import pandas as pd

from src.churn_model import lift_at
from src.data import load, model_matrix


def fake_telco(tmp_path, n=400, seed=0):
    rng = np.random.default_rng(seed)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n)
    tenure = rng.integers(0, 72, n)
    churn_p = np.where(contract == "Month-to-month", 0.45, 0.08)
    df = pd.DataFrame({
        "customerID": [f"C{i:05d}" for i in range(n)], "gender": rng.choice(["Male", "Female"], n),
        "SeniorCitizen": rng.integers(0, 2, n), "Partner": rng.choice(["Yes", "No"], n),
        "Dependents": rng.choice(["Yes", "No"], n), "tenure": tenure, "PhoneService": "Yes",
        "MultipleLines": rng.choice(["Yes", "No"], n), "InternetService": rng.choice(["DSL", "Fiber optic", "No"], n),
        **{c: rng.choice(["Yes", "No"], n) for c in ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                                                    "TechSupport", "StreamingTV", "StreamingMovies"]},
        "Contract": contract, "PaperlessBilling": rng.choice(["Yes", "No"], n),
        "PaymentMethod": rng.choice(["Electronic check", "Mailed check", "Credit card (automatic)"], n),
        "MonthlyCharges": rng.uniform(20, 110, n).round(2),
    })
    df["TotalCharges"] = (df["MonthlyCharges"] * df["tenure"]).round(2).astype(str)
    df.loc[df["tenure"] == 0, "TotalCharges"] = " "
    df["Churn"] = np.where(rng.random(n) < churn_p, "Yes", "No")
    p = tmp_path / "telco.csv"
    df.to_csv(p, index=False)
    return p


def test_load_handles_blank_total_charges(tmp_path):
    df = load(fake_telco(tmp_path))
    assert df["TotalCharges"].dtype.kind == "f"
    assert set(df["churn"].unique()) <= {0, 1}


def test_model_matrix_is_numeric_without_target(tmp_path):
    X = model_matrix(load(fake_telco(tmp_path)))
    assert all(X.dtypes.map(lambda d: d.kind == "f"))
    assert "churn" not in X.columns and "customerID" not in X.columns


def test_lift():
    y = np.array([1] * 10 + [0] * 90)
    perfect = np.linspace(1, 0, 100)
    lift, capture = lift_at(y, perfect, top=0.10)
    assert lift == 10 and capture == 1

"""IBM Telco Customer Churn dataset: 7,043 subscription customers, 21 columns."""
from pathlib import Path
import urllib.request

import pandas as pd

URL = ("https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
       "master/data/Telco-Customer-Churn.csv")
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
LOCAL = DATA_DIR / "telco_churn.csv"

CATEGORICAL = ["gender", "Partner", "Dependents", "PhoneService", "MultipleLines", "InternetService",
               "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
               "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod"]
NUMERIC = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]


def load(path: Path | None = None) -> pd.DataFrame:
    if path is None:
        if not LOCAL.exists():
            DATA_DIR.mkdir(exist_ok=True)
            print(f"Downloading {URL}")
            urllib.request.urlretrieve(URL, LOCAL)
        path = LOCAL
    df = pd.read_csv(path)
    # TotalCharges is blank for brand-new customers (tenure 0) -> 0
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    df["churn"] = (df["Churn"].str.strip() == "Yes").astype(int)
    return df.drop(columns=["Churn"])


def model_matrix(df: pd.DataFrame, drop_first: bool = False) -> pd.DataFrame:
    """One-hot encoded features (drop_first=True avoids collinearity for the Cox model)."""
    X = pd.get_dummies(df[CATEGORICAL], drop_first=drop_first, dtype=float)
    X.columns = [c.replace(" ", "_").replace("(", "").replace(")", "") for c in X.columns]
    return pd.concat([df[NUMERIC].astype(float), X], axis=1)

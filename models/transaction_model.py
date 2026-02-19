import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from config import RANDOM_STATE


def run_transaction_model(path):
    """
    Supervised fraud detection on transactions.
    Trains on train split, predicts on test split.
    Returns per-transaction risk scores for test set (prediction scenario).
    """
    df = pd.read_csv(path)

    categorical_cols = [
        "country",
        "bin_country",
        "channel",
        "merchant_category"
    ]

    # Encode categorical columns that exist
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    # Handle datetime columns - convert to timestamp (numeric)
    datetime_cols = []
    for col in df.columns:
        if df[col].dtype == 'object':
            sample = df[col].dropna().head(10)
            if len(sample) > 0:
                try:
                    pd.to_datetime(sample.iloc[0])
                    if any('T' in str(val) or ':' in str(val) for val in sample.head(3)):
                        datetime_cols.append(col)
                except (ValueError, TypeError):
                    pass

    for col in datetime_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce', utc=True)
        if df[col].dt.tz is not None:
            df[col] = df[col].dt.tz_convert('UTC').dt.tz_localize(None)
        epoch = pd.Timestamp('1970-01-01')
        df[col] = (df[col] - epoch) // pd.Timedelta('1s')
        df[col] = df[col].fillna(0)

    X = df.drop(columns=["transaction_id", "is_fraud"], errors='ignore')
    y = df["is_fraud"]

    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X_numeric = X[numeric_cols]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_numeric)

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=RANDOM_STATE
        )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)

    # Per-transaction fraud probability on TEST set only (no data leakage)
    test_scores = model.predict_proba(X_test)[:, 1]
    return np.array(test_scores), np.array(y_test)


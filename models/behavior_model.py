import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

from config import RANDOM_STATE


def run_behavior_model(path):
    """
    Detects anomaly patterns in behavioral dataset.
    Returns per-row behavior risk scores (one per sample).
    """
    df = pd.read_csv(path)

    # Use last 3 columns as behavioral signals
    feature_cols = df.columns[-3:]
    features = df[feature_cols]

    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    model = IsolationForest(
        contamination=0.1,
        random_state=RANDOM_STATE
    )
    model.fit(X)

    # Per-row anomaly scores (higher = more anomalous/risky)
    scores = -model.decision_function(X)
    return np.array(scores)


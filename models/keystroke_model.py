import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

from config import RANDOM_STATE


def run_keystroke_model(path):
    """
    Detects biometric anomaly in keystroke dynamics.
    Returns per-row keystroke risk scores (one per session/sample).
    """
    df = pd.read_csv(path)
    df_numeric = df.drop(columns=["subject"], errors='ignore')

    scaler = StandardScaler()
    X = scaler.fit_transform(df_numeric)

    model = IsolationForest(
        contamination=0.1,
        random_state=RANDOM_STATE
    )
    model.fit(X)

    scores = -model.decision_function(X)
    return np.array(scores)

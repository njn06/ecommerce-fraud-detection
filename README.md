# E-commerce Fraud Detection System

Multi-modal fraud detection system using behavior, transaction, and keystroke biometrics.

## Phase 1: Multi-Modal Fraud Detection Engine

### Architecture

```
Raw Data → Preprocessing → Independent ML Models → Risk Score Generation → Score-Level Fusion → Final Fraud Risk Score
```

### Models

1. **Behavior Model** (IsolationForest)
   - Detects anomaly patterns in behavioral dataset
   - Outputs per-row behavior risk scores

2. **Transaction Model** (RandomForestClassifier)
   - Supervised fraud detection
   - Train/test split (80/20)
   - Outputs per-transaction fraud probability scores

3. **Keystroke Model** (IsolationForest)
   - Detects biometric anomaly in keystroke dynamics
   - Outputs per-row keystroke risk scores

4. **Risk Fusion**
   - Late fusion with weights: 0.4 (behavior) + 0.4 (transaction) + 0.2 (keystroke)
   - Produces unified per-user/per-row fraud risk scores

### Project Structure

```
phase-1/
├── data/
│   ├── raw/          # Input datasets
│   └── processed/    # Output risk scores
├── models/           # ML models
├── preprocessing/    # Feature engineering
├── fusion/           # Score fusion
├── evaluation/       # Metrics and evaluation
├── config.py         # Configuration
└── main.py           # Main pipeline
```

### Usage

```bash
pip install -r requirements.txt
python main.py
```

Output: Per-transaction risk scores saved to `data/processed/transaction_risk_scores.csv`

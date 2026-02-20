from models.behavior_model import run_behavior_model
from models.transaction_model import run_transaction_model
from models.keystroke_model import run_keystroke_model
from fusion.risk_fusion import fuse_scores
from evaluation.metrics import print_summary


def main():
    """
    Multi-modal fraud detection pipeline.
    Produces per-user/per-row risk scores (not dataset-level mean).
    """

    print("Running Behavior Model...")
    behavior_scores = run_behavior_model("data/raw/data.csv")

    print("Running Transaction Model (train on 80%, predict on 20% test)...")
    transaction_scores, y_test = run_transaction_model("data/raw/transactions.csv")

    print("Running Keystroke Model...")
    keystroke_scores = run_keystroke_model("data/raw/DSL-StrongPasswordData1.csv")

    # Late fusion: transaction test set has only transaction scores (datasets heterogeneous)
    # For transaction test predictions: fused = transaction_scores (full weight)
    final_scores = fuse_scores(
        behavior_scores=None,
        transaction_scores=transaction_scores,
        keystroke_scores=None,
    )

    print_summary(
        behavior_scores,
        transaction_scores,
        keystroke_scores,
        final_scores,
    )

    # Save per-transaction risk scores (test set) for downstream decisions
    import os
    import numpy as np
    os.makedirs("data/processed", exist_ok=True)
    np.savetxt(
        "data/processed/transaction_risk_scores.csv",
        np.column_stack([np.arange(len(transaction_scores)), transaction_scores, y_test]),
        delimiter=",",
        header="test_idx,risk_score,is_fraud_actual",
        comments="",
    )
    print("\nSaved per-transaction risk scores to data/processed/transaction_risk_scores.csv")


if __name__ == "__main__":
    main()


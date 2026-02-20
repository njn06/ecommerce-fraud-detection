import numpy as np


def print_summary(behavior_scores, transaction_scores, keystroke_scores, final_scores):
    """
    Print per-user/per-row risk score summary and distribution.
    Accepts numpy arrays (per-row scores).
    """
    def _stats(name, arr):
        arr = np.asarray(arr)
        if arr.size == 0:
            return
        print(f"  {name}: mean={arr.mean():.4f}, min={arr.min():.4f}, max={arr.max():.4f}, n={len(arr)}")

    print("\n---- Per-Row Risk Score Summary ----")
    if behavior_scores is not None:
        _stats("Behavior", behavior_scores)
    if transaction_scores is not None:
        _stats("Transaction", transaction_scores)
    if keystroke_scores is not None:
        _stats("Keystroke", keystroke_scores)

    print("\n---- Final Fused Risk Score (per user/row) ----")
    final_scores = np.asarray(final_scores)
    if final_scores.size > 0:
        _stats("Fused", final_scores)
        print(f"\n  Sample predictions (first 5): {final_scores[:5].round(4).tolist()}")


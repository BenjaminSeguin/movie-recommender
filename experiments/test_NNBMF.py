import numpy as np
import pandas as pd

import numpy as np
from scipy.optimize import lsq_linear

def nnbmf(R, k=6, num_iters=30, lambda_reg=0.01, max_binary_flips=10, init_B_density=0.1, verbose=True):
    """
    Nonnegative / Binary Matrix Factorization by alternating updates.
    R: (n_users, n_items) ratings matrix with np.nan for missing values.
    Returns: U (n_users x k), B (n_items x k), predictions U @ B.T
    """
    R = np.array(R, dtype=float)
    mask = ~np.isnan(R)
    R_filled = np.nan_to_num(R, nan=0.0)
    n_users, n_items = R.shape

    rng = np.random.RandomState(42)
    # Initialize U >= 0
    U = np.abs(0.1 * rng.randn(n_users, k))
    # Initialize binary B with low density
    B = (rng.rand(n_items, k) < init_B_density).astype(int)

    # Pre-allocate identity for regularization
    sqrt_lambda_I = np.sqrt(lambda_reg) * np.eye(k)

    for it in range(num_iters):
        if verbose:
            print(f"NNBMF iter {it+1}/{num_iters}")

        # --- Update U (users) via NNLS per user ---
        for i in range(n_users):
            item_idx = np.where(mask[i, :])[0]
            if item_idx.size == 0:
                continue
            B_obs = B[item_idx, :]           # (n_obs_items, k)
            r_i = R_filled[i, item_idx]     # (n_obs_items,)
            # Stack regularization rows: sqrt(lambda)*I and zeros target
            A = np.vstack([B_obs, sqrt_lambda_I])
            b = np.hstack([r_i, np.zeros(k)])
            res = lsq_linear(A, b, bounds=(0, np.inf), tol=1e-6, max_iter=100)
            U[i, :] = res.x

        # --- Update B (items) with greedy bit flips per item ---
        for j in range(n_items):
            user_idx = np.where(mask[:, j])[0]
            if user_idx.size == 0:
                continue
            U_obs = U[user_idx, :]           # (n_obs_users, k)
            r_j = R_filled[user_idx, j]      # (n_obs_users,)
            # Greedy binary optimization:
            b = B[j].copy().astype(int)
            current_pred = U_obs @ b
            current_err = np.sum((current_pred - r_j) ** 2)
            flips = 0
            improved = True
            while improved and flips < max_binary_flips:
                improved = False
                best_delta = 0.0
                best_idx = -1
                best_new_b = None
                for f in range(k):
                    b_trial = b.copy()
                    b_trial[f] ^= 1  # flip bit
                    pred = U_obs @ b_trial
                    err = np.sum((pred - r_j) ** 2)
                    delta = current_err - err
                    if delta > best_delta + 1e-9:
                        best_delta = delta
                        best_idx = f
                        best_new_b = b_trial
                if best_idx >= 0:
                    b = best_new_b
                    current_err -= best_delta
                    flips += 1
                    improved = True
            B[j, :] = b

    predictions = U @ B.T
    return U, B, np.clip(predictions, 0, 5)


if __name__ == '__main__':
    import os, sys
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.append(parent_dir)

    from functions.vanilla_als import evaluate_predictions

    print("Loading ratings...")
    R_train = np.load("data/ratings_train.npy")
    R_test = np.load("data/ratings_test.npy")

    print("Running NNBMF (this may take a few minutes)...")
    U, B, preds = nnbmf(R_train, k=1, num_iters=50, lambda_reg=0.01, max_binary_flips=14, init_B_density=0.2, verbose=True)

    # Save predictions
    np.save("nnbmf_preds.npy", preds)
    print("Saved predictions to nnbmf_preds.npy")

    # Evaluate on test set
    mask_test = ~np.isnan(R_test)
    rmse, acc = evaluate_predictions(R_test, preds, mask_test)
    print(f"NNBMF Test RMSE: {rmse:.4f}")
    print(f"NNBMF Exact match accuracy: {acc:.4f}")


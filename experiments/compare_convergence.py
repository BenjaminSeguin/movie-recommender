import os
import sys
import numpy as np
import matplotlib.pyplot as plt

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from userreg import UserReg
from scipy.optimize import lsq_linear

def bounded_als_track(R, k=2, num_iterations=40, lambda_reg=0.001, mu_reg=1.0, max_factor=None, verbose=False):
    n_users, n_movies = R.shape
    np.random.seed(42)
    movie_factors = np.random.normal(0, 0.1, (n_movies, k))
    user_factors = np.random.normal(0, 0.1, (n_users, k))

    mask = ~np.isnan(R)
    R_clean = np.nan_to_num(R, nan=0.0)

    if max_factor is None:
        max_factor = np.sqrt(5 / max(1, k))

    rmse_history = []
    rmse_train_history = []
    for iteration in range(num_iterations):
        # Update movies
        for movie_idx in range(n_movies):
            user_idx = np.where(mask[:, movie_idx])[0]
            if user_idx.size == 0:
                continue
            U_items = user_factors[user_idx, :]
            ratings_i = R_clean[user_idx, movie_idx]
            A = np.vstack([U_items, np.sqrt(lambda_reg) * np.eye(k)])
            b = np.hstack([ratings_i, np.zeros(k)])
            res = lsq_linear(A, b, bounds=(0, max_factor), method='bvls')
            movie_factors[movie_idx, :] = res.x

        # Update users
        for user_idx in range(n_users):
            movie_idx = np.where(mask[user_idx, :])[0]
            if movie_idx.size == 0:
                continue
            I_users = movie_factors[movie_idx, :]
            ratings_u = R_clean[user_idx, movie_idx]
            A = np.vstack([I_users, np.sqrt(mu_reg) * np.eye(k)])
            b = np.hstack([ratings_u, np.zeros(k)])
            res = lsq_linear(A, b, bounds=(0, max_factor), method='bvls')
            user_factors[user_idx, :] = res.x

        # End of iteration: evaluate on test set
        preds = user_factors @ movie_factors.T
        rmse_test = compute_rmse_on_mask(R_test_global, preds, ~np.isnan(R_test_global))
        rmse_train = compute_rmse_on_mask(R, preds, ~np.isnan(R))
        rmse_history.append(rmse_test)
        rmse_train_history.append(rmse_train)
        if verbose:
            print(f"bounded_als iter {iteration+1}/{num_iterations} test_rmse={rmse_test:.4f} train_rmse={rmse_train:.4f}")

    return user_factors, movie_factors, np.array(rmse_history), np.array(rmse_train_history)


def compute_rmse_on_mask(R_true, R_pred, mask):
    # mask: boolean array matching R_true/R_pred shape selecting observed entries
    if mask is None:
        mask = ~np.isnan(R_true)
    # ensure boolean
    mask = np.array(mask, dtype=bool)
    n = mask.sum()
    if n == 0:
        return float('nan')
    diff = R_pred[mask] - R_true[mask]
    # If any values in R_true are NaN at these positions, drop them
    valid = ~np.isnan(diff)
    if valid.sum() == 0:
        return float('nan')
    return np.sqrt(np.mean((diff[valid]) ** 2))


if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)

    # Load data
    R_train = np.load('data/ratings_train.npy')
    R_test = np.load('data/ratings_test.npy')

    # For convenience in bounded_als_track
    global R_test_global
    R_test_global = R_test

    num_iters = 40

    print('Running bounded ALS...')
    _, _, bounded_rmse, bounded_train_rmse = bounded_als_track(R_train, k=1, num_iterations=num_iters, lambda_reg=0.001, mu_reg=1.0, verbose=True)

    print('Running UserReg...')
    # Run UserReg but capture per-iteration RMSE by copying its loop here
    ur = UserReg(k=1, lr=0.01, lambda_reg=0.05, beta_reg=8.0, num_iterations=num_iters, verbose=False)
    # initialize parameters
    np.random.seed(42)
    n_users, n_items = R_train.shape
    ur.P = 0.1 * np.random.randn(n_users, ur.k)
    ur.Q = 0.1 * np.random.randn(n_items, ur.k)
    ur.bu = np.zeros(n_users)
    ur.bi = np.zeros(n_items)

    mask = ~np.isnan(R_train)
    R_filled = np.nan_to_num(R_train, nan=0.0)

    userreg_rmse = []
    userreg_train_rmse = []
    for it in range(num_iters):
        # single epoch over users (same updates as in userreg.fit)
        for u in range(n_users):
            Iu = np.where(mask[u, :])[0]
            if len(Iu) == 0:
                continue
            mean_Q = np.mean(ur.Q[Iu, :], axis=0)
            for i in Iu:
                e_ui = R_filled[u, i] - (ur.bu[u] + ur.bi[i] + np.dot(ur.P[u], ur.Q[i]))
                ur.bu[u] += ur.lr * (e_ui - ur.lambda_reg * ur.bu[u])
                ur.bi[i] += ur.lr * (e_ui - ur.lambda_reg * ur.bi[i])
                ur.P[u] += ur.lr * (e_ui * ur.Q[i] - ur.beta_reg * (ur.P[u] - mean_Q) - ur.lambda_reg * ur.P[u])
                ur.Q[i] += ur.lr * (e_ui * ur.P[u] - ur.lambda_reg * ur.Q[i])

        preds = ur.P @ ur.Q.T + ur.bu[:, None] + ur.bi[None, :]
        rmse_test = compute_rmse_on_mask(R_test, preds, ~np.isnan(R_test))
        rmse_train = compute_rmse_on_mask(R_train, preds, ~np.isnan(R_train))
        userreg_rmse.append(rmse_test)
        userreg_train_rmse.append(rmse_train)
        print(f'UserReg iter {it+1}/{num_iters} test_rmse={rmse_test:.4f} train_rmse={rmse_train:.4f}')

    # Plot convergence
    plt.figure(figsize=(8,5))
    it = np.arange(1, num_iters+1)
    # bounded ALS -> color C0, UserReg -> color C1; dashed=train, solid=test
    plt.plot(it, bounded_train_rmse, label='bounded ALS (train)', linestyle='--', color='C0')
    plt.plot(it, bounded_rmse, label='bounded ALS (test)', linestyle='--', color='C1')
    plt.plot(it, userreg_train_rmse, label='UserReg (train)', linestyle='-', color='C2')
    plt.plot(it, userreg_rmse, label='UserReg (test)', linestyle='-', color='C3')
    # Set upper bound to 1.5 as requested (zoom into the lower RMSE region)
    plt.ylim(None, 1.5)
    plt.xlabel('Iteration')
    plt.ylabel('RMSE')
    plt.title('Convergence comparison')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    outpath = os.path.join('results', 'convergence.png')
    plt.savefig(outpath)
    print(f'Saved convergence plot to {outpath}')
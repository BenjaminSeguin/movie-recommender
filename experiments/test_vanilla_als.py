import pandas as pd
import numpy as np
from scipy.optimize import nnls
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from functions.vanilla_als import evaluate_predictions

# Load data
table = np.load("data/ratings_train.npy")  
ratings_test = pd.DataFrame(np.load("data/ratings_test.npy")) 


def als_optimization(
    ratings_matrix, k=1, num_iterations=30, lambda_reg=0.01, mu_reg=0.01
):
    """
    ALS optimization for matrix factorization - No Transpose Version
    Works directly with users x movies format

    Matrix dimensions:
    - R: 610x4980 (users x movies) - input format
    - movie_factors: 4980xk (movies x k factors)
    - user_factors: 610xk (users x k factors)
    - Reconstruction: user_factors @ movie_factors.T = 610x4980 (users x movies)
    """
    R_original = ratings_matrix
    n_users, n_movies = R_original.shape

    np.random.seed(42)
    movie_factors = np.random.normal(0, 0.1, (n_movies, k))  # 4980×k
    user_factors = np.random.normal(0, 0.1, (n_users, k))  # 610×k

    # Create mask for observed ratings
    mask_original = ~np.isnan(R_original)  # (610, 4980) users × movies
    R_clean = np.nan_to_num(R_original, nan=0.0)

    for iteration in range(num_iterations):
        # For each movie, find users who rated it and update its factors
        for movie_idx in range(movie_factors.shape[0]):  # 0 to 4979
            user_indices = np.where(mask_original[:, movie_idx])[0]
            if len(user_indices) > 0:
                U_items = user_factors[user_indices, :]
                ratings_i = R_clean[user_indices, movie_idx]
                A = U_items.T @ U_items + lambda_reg * np.eye(U_items.shape[1])
                b = U_items.T @ ratings_i
                movie_factors[movie_idx, :] = np.linalg.solve(A, b)

        # For each user, find movies they rated and update their factors
        for user_idx in range(user_factors.shape[0]):  # 0 to 609
            movie_indices = np.where(mask_original[user_idx, :])[0]
            if len(movie_indices) > 0:
                I_users = movie_factors[movie_indices, :]
                ratings_u = R_clean[user_idx, movie_indices]
                A = I_users.T @ I_users + mu_reg * np.eye(I_users.shape[1])
                b = I_users.T @ ratings_u
                user_factors[user_idx, :] = np.linalg.solve(A, b)

    # Reconstruct: user_factors @ movie_factors.T gives (610×4980)
    return user_factors @ movie_factors.T


def constrain_ratings(predictions):
    """Constrain predictions to valid movie ratings: 0-5 in 0.5 increments"""
    clipped = np.clip(predictions, 0, 5)
    return np.round(clipped * 2) / 2


# Apply ALS matrix factorization
print("Applying ALS matrix factorization...")
predicted_ratings = als_optimization(
    table, k=1, num_iterations=30, lambda_reg=0.01, mu_reg=0.01
)
table = constrain_ratings(predicted_ratings)
print("ALS optimization completed.")

print(f"Final shape: {table.shape} (users x movies)")

# Evaluate on test data (only on observed test ratings)
print("\nEvaluating on test dataset...")
R_test = ratings_test.values
mask_test = ~np.isnan(R_test)

rmse_test, acc_test = evaluate_predictions(R_test, table, mask_test)

print(f"Test set performance:")
print(f"  RMSE: {rmse_test:.4f}")
print(f"  Exact match accuracy: {acc_test:.4f}")

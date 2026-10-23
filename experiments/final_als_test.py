import pandas as pd
import numpy as np
from scipy.optimize import lsq_linear
import sys
import os

# Add the directory containing 'functions' folder to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from functions.vanilla_als import evaluate_predictions

table = np.load("data/ratings_train.npy")
ratings_test = pd.DataFrame(np.load("data/ratings_test.npy"))


# ============================================================================
# METHOD 1: PER-OBSERVATION SCALING
# ============================================================================
def annls_per_observation_scaling(ratings_matrix, k=1, num_iterations=30, 
                                   lambda_base=0.01, mu_base=1.0):
    """
    ALS with per-observation scaling of regularization.
    
    Key idea: Scale regularization by number of ratings, making λ and μ
    comparable in magnitude. This normalizes the regularization strength
    relative to the amount of data available.
    
    Formula:
    - Movie update: λ_effective = λ_base * n_ratings_for_movie
    - User update: μ_effective = μ_base * n_ratings_for_user
    """
    R_original = ratings_matrix
    n_users, n_movies = R_original.shape
    
    np.random.seed(42)
    movie_factors = np.random.normal(0, 0.1, (n_movies, k))
    user_factors = np.random.normal(0, 0.1, (n_users, k))
    
    mask_original = ~np.isnan(R_original)
    R_clean = np.nan_to_num(R_original, nan=0.0)
    
    # Pre-compute rating counts for scaling
    movie_rating_counts = np.sum(mask_original, axis=0)  # Count ratings per movie
    user_rating_counts = np.sum(mask_original, axis=1)    # Count ratings per user
    
    max_factor = np.sqrt(5 / k)
    
    for iteration in range(num_iterations):
        # Update movie factors
        for movie_idx in range(n_movies):
            user_indices = np.where(mask_original[:, movie_idx])[0]
            if len(user_indices) > 0:
                n_ratings = len(user_indices)
                U_items = user_factors[user_indices, :]
                ratings_i = R_clean[user_indices, movie_idx]
                
                # Scale regularization by number of ratings
                lambda_scaled = lambda_base * n_ratings
                
                A = np.vstack([U_items, np.sqrt(lambda_scaled) * np.eye(k)])
                b = np.hstack([ratings_i, np.zeros(k)])
                
                result = lsq_linear(A, b, bounds=(0, max_factor), method='bvls')
                movie_factors[movie_idx, :] = result.x
        
        # Update user factors
        for user_idx in range(n_users):
            movie_indices = np.where(mask_original[user_idx, :])[0]
            if len(movie_indices) > 0:
                n_ratings = len(movie_indices)
                I_users = movie_factors[movie_indices, :]
                ratings_u = R_clean[user_idx, movie_indices]
                
                # Scale regularization by number of ratings
                mu_scaled = mu_base * n_ratings
                
                A = np.vstack([I_users, np.sqrt(mu_scaled) * np.eye(k)])
                b = np.hstack([ratings_u, np.zeros(k)])
                
                result = lsq_linear(A, b, bounds=(0, max_factor), method='bvls')
                user_factors[user_idx, :] = result.x
        
        if iteration % 10 == 0:
            predictions = user_factors @ movie_factors.T
            rmse = np.sqrt(np.mean((R_clean[mask_original] - predictions[mask_original])**2))
            print(f"Iteration {iteration}, RMSE: {rmse:.4f}")
    
    predictions = user_factors @ movie_factors.T
    return np.clip(predictions, 0, 5)


# ============================================================================
# METHOD 2: CONFIDENCE WEIGHTING
# ============================================================================
def annls_confidence_weighting(ratings_matrix, k=1, num_iterations=30,
                               lambda_reg=0.01, mu_reg=1.0, alpha=40):
    """
    ALS with confidence weighting.
    
    Key idea: Not all ratings are equally reliable. Weight ratings by confidence.
    For explicit feedback, extreme ratings (1 or 5) are more confident than
    middle ratings (3).
    
    Confidence formula: c_ij = 1 + α * |r_ij - r_mean|
    - Higher α → more emphasis on extreme ratings
    - Typical values: α ∈ [10, 100]
    """
    R_original = ratings_matrix
    n_users, n_movies = R_original.shape
    
    np.random.seed(42)
    movie_factors = np.random.normal(0, 0.1, (n_movies, k))
    user_factors = np.random.normal(0, 0.1, (n_users, k))
    
    mask_original = ~np.isnan(R_original)
    R_clean = np.nan_to_num(R_original, nan=0.0)
    
    # Calculate mean rating for confidence computation
    mean_rating = np.mean(R_clean[mask_original])
    
    # Compute confidence matrix: c_ij = 1 + α * |r_ij - mean|
    confidence = np.ones_like(R_clean)
    confidence[mask_original] = 1 + alpha * np.abs(R_clean[mask_original] - mean_rating)
    
    max_factor = np.sqrt(5 / k)
    
    for iteration in range(num_iterations):
        # Update movie factors with confidence weighting
        for movie_idx in range(n_movies):
            user_indices = np.where(mask_original[:, movie_idx])[0]
            if len(user_indices) > 0:
                U_items = user_factors[user_indices, :]
                ratings_i = R_clean[user_indices, movie_idx]
                weights_i = np.sqrt(confidence[user_indices, movie_idx])  # sqrt for weighted least squares
                
                # Weight the equations by confidence
                A_weighted = U_items * weights_i[:, np.newaxis]
                b_weighted = ratings_i * weights_i
                
                # Add regularization (not weighted)
                A = np.vstack([A_weighted, np.sqrt(lambda_reg) * np.eye(k)])
                b = np.hstack([b_weighted, np.zeros(k)])
                
                result = lsq_linear(A, b, bounds=(0, max_factor), method='bvls')
                movie_factors[movie_idx, :] = result.x
        
        # Update user factors with confidence weighting
        for user_idx in range(n_users):
            movie_indices = np.where(mask_original[user_idx, :])[0]
            if len(movie_indices) > 0:
                I_users = movie_factors[movie_indices, :]
                ratings_u = R_clean[user_idx, movie_indices]
                weights_u = np.sqrt(confidence[user_idx, movie_indices])
                
                # Weight the equations by confidence
                A_weighted = I_users * weights_u[:, np.newaxis]
                b_weighted = ratings_u * weights_u
                
                # Add regularization (not weighted)
                A = np.vstack([A_weighted, np.sqrt(mu_reg) * np.eye(k)])
                b = np.hstack([b_weighted, np.zeros(k)])
                
                result = lsq_linear(A, b, bounds=(0, max_factor), method='bvls')
                user_factors[user_idx, :] = result.x
        
        if iteration % 10 == 0:
            predictions = user_factors @ movie_factors.T
            rmse = np.sqrt(np.mean((R_clean[mask_original] - predictions[mask_original])**2))
            print(f"Iteration {iteration}, RMSE: {rmse:.4f}")
    
    predictions = user_factors @ movie_factors.T
    return np.clip(predictions, 0, 5)


# ============================================================================
# METHOD 3: ADAPTIVE REGULARIZATION (RECOMMENDED!)
# ============================================================================
def annls_adaptive_regularization(ratings_matrix, k=1, num_iterations=30,
                                  lambda_base=0.01, mu_base=1.0,
                                  min_reg=0.0001, max_reg=10.0):
    """
    ALS with adaptive regularization based on rating counts.
    
    Key idea: Adapt regularization strength to each user/movie based on
    how much data is available.
    
    Formula:
    - λ_j = λ_base * f(n_ratings_for_movie_j)
    - μ_i = μ_base * f(n_ratings_for_user_i)
    
    where f() is a scaling function. Common choices:
    1. Linear: f(n) = n / avg_n
    2. Square root: f(n) = sqrt(n / avg_n)
    3. Log: f(n) = log(1 + n) / log(1 + avg_n)
    
    More ratings → stronger regularization (prevent overfitting)
    Fewer ratings → weaker regularization (maintain flexibility)
    """
    R_original = ratings_matrix
    n_users, n_movies = R_original.shape
    
    np.random.seed(42)
    movie_factors = np.random.normal(0, 0.1, (n_movies, k))
    user_factors = np.random.normal(0, 0.1, (n_users, k))
    
    mask_original = ~np.isnan(R_original)
    R_clean = np.nan_to_num(R_original, nan=0.0)
    
    # Compute rating counts
    movie_rating_counts = np.sum(mask_original, axis=0)  # Per movie
    user_rating_counts = np.sum(mask_original, axis=1)    # Per user
    
    # Compute average counts
    avg_movie_ratings = np.mean(movie_rating_counts[movie_rating_counts > 0])
    avg_user_ratings = np.mean(user_rating_counts[user_rating_counts > 0])
    
    print(f"Average ratings per movie: {avg_movie_ratings:.2f}")
    print(f"Average ratings per user: {avg_user_ratings:.2f}")
    
    # Compute adaptive regularizers for each entity
    # Using square root scaling (balances between linear and constant)
    lambda_adaptive = np.zeros(n_movies)
    for j in range(n_movies):
        if movie_rating_counts[j] > 0:
            scale_factor = np.sqrt(movie_rating_counts[j] / avg_movie_ratings)
            lambda_adaptive[j] = lambda_base * scale_factor
            lambda_adaptive[j] = np.clip(lambda_adaptive[j], min_reg, max_reg)
        else:
            lambda_adaptive[j] = min_reg
    
    mu_adaptive = np.zeros(n_users)
    for i in range(n_users):
        if user_rating_counts[i] > 0:
            scale_factor = np.sqrt(user_rating_counts[i] / avg_user_ratings)
            mu_adaptive[i] = mu_base * scale_factor
            mu_adaptive[i] = np.clip(mu_adaptive[i], min_reg, max_reg)
        else:
            mu_adaptive[i] = min_reg
    
    print(f"λ range: [{np.min(lambda_adaptive):.4f}, {np.max(lambda_adaptive):.4f}]")
    print(f"μ range: [{np.min(mu_adaptive):.4f}, {np.max(mu_adaptive):.4f}]")
    
    max_factor = np.sqrt(5 / k)
    
    for iteration in range(num_iterations):
        # Update movie factors with adaptive regularization
        for movie_idx in range(n_movies):
            user_indices = np.where(mask_original[:, movie_idx])[0]
            if len(user_indices) > 0:
                U_items = user_factors[user_indices, :]
                ratings_i = R_clean[user_indices, movie_idx]
                
                # Use movie-specific regularization
                lambda_j = lambda_adaptive[movie_idx]
                
                A = np.vstack([U_items, np.sqrt(lambda_j) * np.eye(k)])
                b = np.hstack([ratings_i, np.zeros(k)])
                
                result = lsq_linear(A, b, bounds=(0, max_factor), method='bvls')
                movie_factors[movie_idx, :] = result.x
        
        # Update user factors with adaptive regularization
        for user_idx in range(n_users):
            movie_indices = np.where(mask_original[user_idx, :])[0]
            if len(movie_indices) > 0:
                I_users = movie_factors[movie_indices, :]
                ratings_u = R_clean[user_idx, movie_indices]
                
                # Use user-specific regularization
                mu_i = mu_adaptive[user_idx]
                
                A = np.vstack([I_users, np.sqrt(mu_i) * np.eye(k)])
                b = np.hstack([ratings_u, np.zeros(k)])
                
                result = lsq_linear(A, b, bounds=(0, max_factor), method='bvls')
                user_factors[user_idx, :] = result.x
        
        if iteration % 10 == 0:
            predictions = user_factors @ movie_factors.T
            rmse = np.sqrt(np.mean((R_clean[mask_original] - predictions[mask_original])**2))
            print(f"Iteration {iteration}, RMSE: {rmse:.4f}")
    
    predictions = user_factors @ movie_factors.T
    return np.clip(predictions, 0, 5)


# ============================================================================
# METHOD 4: COMBINED APPROACH (Best of all worlds!)
# ============================================================================
def annls_combined(ratings_matrix, k=1, num_iterations=30,
                   lambda_base=0.01, mu_base=1.0,
                   alpha=40, use_confidence=True,
                   min_reg=0.0001, max_reg=10.0):
    """
    Combined approach: Adaptive regularization + Confidence weighting
    
    This gives the best results by:
    1. Adapting regularization to data availability
    2. Weighting ratings by confidence
    """
    R_original = ratings_matrix
    n_users, n_movies = R_original.shape
    
    np.random.seed(42)
    movie_factors = np.random.normal(0, 0.1, (n_movies, k))
    user_factors = np.random.normal(0, 0.1, (n_users, k))
    
    mask_original = ~np.isnan(R_original)
    R_clean = np.nan_to_num(R_original, nan=0.0)
    
    # Compute adaptive regularizers
    movie_rating_counts = np.sum(mask_original, axis=0)
    user_rating_counts = np.sum(mask_original, axis=1)
    
    avg_movie_ratings = np.mean(movie_rating_counts[movie_rating_counts > 0])
    avg_user_ratings = np.mean(user_rating_counts[user_rating_counts > 0])
    
    lambda_adaptive = np.zeros(n_movies)
    for j in range(n_movies):
        if movie_rating_counts[j] > 0:
            scale_factor = np.sqrt(movie_rating_counts[j] / avg_movie_ratings)
            lambda_adaptive[j] = np.clip(lambda_base * scale_factor, min_reg, max_reg)
        else:
            lambda_adaptive[j] = min_reg
    
    mu_adaptive = np.zeros(n_users)
    for i in range(n_users):
        if user_rating_counts[i] > 0:
            scale_factor = np.sqrt(user_rating_counts[i] / avg_user_ratings)
            mu_adaptive[i] = np.clip(mu_base * scale_factor, min_reg, max_reg)
        else:
            mu_adaptive[i] = min_reg
    
    # Compute confidence weights
    confidence = np.ones_like(R_clean)
    if use_confidence:
        mean_rating = np.mean(R_clean[mask_original])
        confidence[mask_original] = 1 + alpha * np.abs(R_clean[mask_original] - mean_rating)
    
    max_factor = np.sqrt(5 / k)
    
    for iteration in range(num_iterations):
        # Update movie factors
        for movie_idx in range(n_movies):
            user_indices = np.where(mask_original[:, movie_idx])[0]
            if len(user_indices) > 0:
                U_items = user_factors[user_indices, :]
                ratings_i = R_clean[user_indices, movie_idx]
                
                if use_confidence:
                    weights_i = np.sqrt(confidence[user_indices, movie_idx])
                    A_weighted = U_items * weights_i[:, np.newaxis]
                    b_weighted = ratings_i * weights_i
                else:
                    A_weighted = U_items
                    b_weighted = ratings_i
                
                lambda_j = lambda_adaptive[movie_idx]
                A = np.vstack([A_weighted, np.sqrt(lambda_j) * np.eye(k)])
                b = np.hstack([b_weighted, np.zeros(k)])
                
                result = lsq_linear(A, b, bounds=(0, max_factor), method='bvls')
                movie_factors[movie_idx, :] = result.x
        
        # Update user factors
        for user_idx in range(n_users):
            movie_indices = np.where(mask_original[user_idx, :])[0]
            if len(movie_indices) > 0:
                I_users = movie_factors[movie_indices, :]
                ratings_u = R_clean[user_idx, movie_indices]
                
                if use_confidence:
                    weights_u = np.sqrt(confidence[user_idx, movie_indices])
                    A_weighted = I_users * weights_u[:, np.newaxis]
                    b_weighted = ratings_u * weights_u
                else:
                    A_weighted = I_users
                    b_weighted = ratings_u
                
                mu_i = mu_adaptive[user_idx]
                A = np.vstack([A_weighted, np.sqrt(mu_i) * np.eye(k)])
                b = np.hstack([b_weighted, np.zeros(k)])
                
                result = lsq_linear(A, b, bounds=(0, max_factor), method='bvls')
                user_factors[user_idx, :] = result.x
        
        if iteration % 10 == 0:
            predictions = user_factors @ movie_factors.T
            rmse = np.sqrt(np.mean((R_clean[mask_original] - predictions[mask_original])**2))
            print(f"Iteration {iteration}, RMSE: {rmse:.4f}")
    
    predictions = user_factors @ movie_factors.T
    return np.clip(predictions, 0, 5)


def constrain_ratings(predictions):
    """Constrain predictions to valid movie ratings: 0-5 in 0.5 increments"""
    return np.round(predictions * 2) / 2


# ============================================================================
# MAIN EXECUTION - Choose your method!
# ============================================================================
if __name__ == "__main__":
    k = 1
    
    # Uncomment the method you want to try:
    
    # METHOD 1: Per-observation scaling
    # print("\n" + "="*70)
    # print("METHOD 1: PER-OBSERVATION SCALING")
    # print("="*70)
    # predicted_ratings = annls_per_observation_scaling(
    #     table, k=k, num_iterations=30,
    #     lambda_base=0.00001, mu_base=0.001  # Note: much smaller values!
    # )
    
    # METHOD 2: Confidence weighting
    # print("\n" + "="*70)
    # print("METHOD 2: CONFIDENCE WEIGHTING")
    # print("="*70)
    # predicted_ratings = annls_confidence_weighting(
    #     table, k=k, num_iterations=30,
    #     lambda_reg=0.01, mu_reg=1.0, alpha=40
    # )
    
    # METHOD 3: Adaptive regularization (RECOMMENDED!)
    print("\n" + "="*70)
    print("METHOD 3: ADAPTIVE REGULARIZATION")
    print("="*70)
    predicted_ratings = annls_adaptive_regularization(
        table, k=k, num_iterations=30,
        lambda_base=0.01, mu_base=1.0,
        min_reg=0.0001, max_reg=10.0
    )
    
    # METHOD 4: Combined approach (Best!)
    # print("\n" + "="*70)
    # print("METHOD 4: ADAPTIVE + CONFIDENCE (COMBINED)")
    # print("="*70)
    # predicted_ratings = annls_combined(
    #     table, k=k, num_iterations=30,
    #     lambda_base=0.01, mu_base=1.0,
    #     alpha=40, use_confidence=True,
    #     min_reg=0.0001, max_reg=10.0
    # )
    
    table_final = constrain_ratings(predicted_ratings)
    print(f"\nFinal shape: {table_final.shape} (users x movies)")
    
    # Evaluate on test data
    print("\n" + "="*70)
    print("EVALUATION ON TEST SET")
    print("="*70)
    R_test = ratings_test.values
    mask_test = ~np.isnan(R_test)
    
    rmse_test, acc_test = evaluate_predictions(R_test, table_final, mask_test)
    
    print(f"Test set performance:")
    print(f"  RMSE: {rmse_test:.4f}")
    print(f"  Exact match accuracy: {acc_test:.4f}")
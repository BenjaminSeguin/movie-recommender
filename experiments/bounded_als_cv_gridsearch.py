import pandas as pd
import numpy as np
from scipy.optimize import lsq_linear
import matplotlib.pyplot as plt
import sys
import os
from itertools import product

# Add the directory containing 'functions' folder to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from functions.vanilla_als import evaluate_predictions

# Load data
ratings_train_orig = np.load("data/ratings_train.npy")
ratings_test_orig = np.load("data/ratings_test.npy")

# MERGE train and test into one full dataset
# Use test values where available, otherwise use train values
ratings_full = np.where(~np.isnan(ratings_test_orig), ratings_test_orig, ratings_train_orig)
print(f"Merged dataset shape: {ratings_full.shape}")
print(f"Total observed ratings: {(~np.isnan(ratings_full)).sum()}")

def annls_optimization_bounded(ratings_matrix, k=1, num_iterations=30, lambda_reg=0.01, mu_reg=0.01, seed=42):
    """
    ALS optimization for matrix factorization with bounded NNLS
    ONLY trains on OBSERVED ratings (non-NaN entries)
    """
    R_original = ratings_matrix
    n_users, n_movies = R_original.shape
    
    np.random.seed(seed)
    movie_factors = np.random.normal(0, 0.1, (n_movies, k))
    user_factors = np.random.normal(0, 0.1, (n_users, k))
    
    # CRITICAL: Create mask for OBSERVED ratings ONLY
    mask_original = ~np.isnan(R_original)
    R_clean = np.nan_to_num(R_original, nan=0.0)
    
    max_factor = np.sqrt(5 / k)
    
    for iteration in range(num_iterations):
        # Update movie factors - ONLY using observed ratings
        for movie_idx in range(movie_factors.shape[0]):
            user_indices = np.where(mask_original[:, movie_idx])[0]
            if len(user_indices) == 0:
                continue
            
            U_items = user_factors[user_indices, :]
            ratings_i = R_clean[user_indices, movie_idx]
            
            A = np.vstack([U_items, np.sqrt(lambda_reg) * np.eye(k)])
            b = np.hstack([ratings_i, np.zeros(k)])
            
            result = lsq_linear(A, b, bounds=(0, max_factor), method='bvls')
            movie_factors[movie_idx, :] = result.x
        
        # Update user factors - ONLY using observed ratings
        for user_idx in range(user_factors.shape[0]):
            movie_indices = np.where(mask_original[user_idx, :])[0]
            if len(movie_indices) == 0:
                continue
            
            I_users = movie_factors[movie_indices, :]
            ratings_u = R_clean[user_idx, movie_indices]
            
            A = np.vstack([I_users, np.sqrt(mu_reg) * np.eye(k)])
            b = np.hstack([ratings_u, np.zeros(k)])
            
            result = lsq_linear(A, b, bounds=(0, max_factor), method='bvls')
            user_factors[user_idx, :] = result.x
    
    predictions = user_factors @ movie_factors.T
    return np.clip(predictions, 0, 5)

def constrain_ratings(predictions):
    """Constrain predictions to valid movie ratings: 0-5 in 0.5 increments"""
    return np.round(predictions * 2) / 2

def create_cv_split(R_full, train_ratio=0.8, seed=0):
    """
    Split observed ratings into train and validation sets
    PROPERLY - without touching the test set!
    """
    mask = ~np.isnan(R_full)
    obs_idx = np.argwhere(mask)
    
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(obs_idx))
    cut = int(train_ratio * len(obs_idx))
    
    train_idx = obs_idx[perm[:cut]]
    val_idx = obs_idx[perm[cut:]]
    
    # Create train matrix (validation entries set to NaN)
    R_train = R_full.copy()
    for (i, j) in val_idx:
        R_train[i, j] = np.nan
    
    # Create validation mask
    val_mask = np.zeros_like(R_full, dtype=bool)
    for (i, j) in val_idx:
        val_mask[i, j] = True
    
    print(f"CV split: {len(train_idx)} train, {len(val_idx)} validation entries")
    return R_train, val_mask

def plot_rmse_vs_k(results_df, k_values):
    """
    Plot train and validation RMSE vs k (number of latent factors)
    Averages over all lambda and mu values for each k
    """
    # Group by k and compute mean train/val RMSE
    k_stats = results_df.groupby('k').agg({
        'train_rmse': ['mean', 'std'],
        'val_rmse': ['mean', 'std']
    }).reset_index()
    
    # Flatten column names
    k_stats.columns = ['k', 'train_rmse_mean', 'train_rmse_std', 'val_rmse_mean', 'val_rmse_std']
    
    # Sort by k
    k_stats = k_stats.sort_values('k')
    
    # Create plot
    plt.figure(figsize=(10, 6))
    
    # Plot train RMSE
    plt.plot(k_stats['k'], k_stats['train_rmse_mean'], 
             marker='o', linewidth=2, markersize=8, label='Train RMSE', color='blue')
    plt.fill_between(k_stats['k'], 
                     k_stats['train_rmse_mean'] - k_stats['train_rmse_std'],
                     k_stats['train_rmse_mean'] + k_stats['train_rmse_std'],
                     alpha=0.2, color='blue')
    
    # Plot validation RMSE
    plt.plot(k_stats['k'], k_stats['val_rmse_mean'], 
             marker='s', linewidth=2, markersize=8, label='Validation RMSE', color='red')
    plt.fill_between(k_stats['k'], 
                     k_stats['val_rmse_mean'] - k_stats['val_rmse_std'],
                     k_stats['val_rmse_mean'] + k_stats['val_rmse_std'],
                     alpha=0.2, color='red')
    
    plt.xlabel('Number of Latent Factors (k)', fontsize=12)
    plt.ylabel('RMSE', fontsize=12)
    plt.title('Training vs Validation RMSE by Number of Latent Factors\n(averaged over λ and μ)', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    os.makedirs('results', exist_ok=True)
    plot_path = 'results/bounded_als_rmse_vs_k.png'
    plt.savefig(plot_path, dpi=200)
    print(f"Plot saved to {plot_path}")
    
    # Also print the summary table
    print("\nRMSE Summary by k (mean ± std):")
    print("-" * 60)
    for _, row in k_stats.iterrows():
        print(f"k={int(row['k']):2d}  |  Train: {row['train_rmse_mean']:.4f}±{row['train_rmse_std']:.4f}  |  "
              f"Val: {row['val_rmse_mean']:.4f}±{row['val_rmse_std']:.4f}")
    print("-" * 60)
    
    plt.close()

def grid_search_cv(ratings_train_full, k_values, lambda_values, mu_values, 
                   num_iterations=30, cv_ratio=0.8, cv_seed=42):
    """
    Grid search with cross-validation on merged dataset
    
    Args:
        ratings_train_full: Full merged dataset (train + test from files)
        k_values, lambda_values, mu_values: Hyperparameter search space
        num_iterations: ALS iterations
        cv_ratio: Ratio for train/val split (default 80/20)
        cv_seed: Random seed for CV split
    
    Returns:
        results_df: DataFrame with all results
        best_params: Best hyperparameters based on VALIDATION set
    """
    # Split training data into train + validation
    R_train, val_mask = create_cv_split(ratings_train_full, train_ratio=cv_ratio, seed=cv_seed)
    
    # Create train mask for computing train RMSE
    train_mask = ~np.isnan(R_train)
    
    results = []
    best_val_rmse = float('inf')
    best_params = None
    
    total_combinations = len(k_values) * len(lambda_values) * len(mu_values)
    current = 0
    
    print(f"\nStarting PROPER CV grid search with {total_combinations} combinations...")
    print(f"k values: {k_values}")
    print(f"lambda values: {lambda_values}")
    print(f"mu values: {mu_values}")
    print("-" * 80)
    
    for k, lambda_reg, mu_reg in product(k_values, lambda_values, mu_values):
        current += 1
        print(f"\n[{current}/{total_combinations}] Testing k={k}, lambda={lambda_reg}, mu={mu_reg}")
        
        try:
            # Train on R_train (which has validation entries as NaN)
            predicted_ratings = annls_optimization_bounded(
                R_train,  # ← Training only (validation entries are NaN)
                k=k, 
                num_iterations=num_iterations, 
                lambda_reg=lambda_reg, 
                mu_reg=mu_reg,
                seed=cv_seed
            )
            predictions = constrain_ratings(predicted_ratings)
            
            # Evaluate on VALIDATION set (held-out from R_train)
            # Compare predictions to TRUE VALUES in ratings_train_full at validation positions
            val_rmse, val_acc = evaluate_predictions(ratings_train_full, predictions, val_mask)
            
            # Also evaluate on TRAIN set to detect overfitting
            # Compare predictions to TRUE VALUES in ratings_train_full at training positions
            train_rmse, train_acc = evaluate_predictions(ratings_train_full, predictions, train_mask)
            
            print(f"  Train RMSE: {train_rmse:.4f}, Validation RMSE: {val_rmse:.4f}")
            
            results.append({
                'k': k,
                'lambda': lambda_reg,
                'mu': mu_reg,
                'train_rmse': train_rmse,
                'val_rmse': val_rmse,
                'train_accuracy': train_acc,
                'val_accuracy': val_acc
            })
            
            # Track best model based on VALIDATION performance
            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_params = {
                    'k': k,
                    'lambda': lambda_reg,
                    'mu': mu_reg,
                    'train_rmse': train_rmse,
                    'val_rmse': val_rmse,
                    'val_accuracy': val_acc
                }
                print(f"  *** NEW BEST MODEL (validation) ***")
        
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'k': k,
                'lambda': lambda_reg,
                'mu': mu_reg,
                'train_rmse': np.nan,
                'val_rmse': np.nan,
                'train_accuracy': np.nan,
                'val_accuracy': np.nan
            })
    
    results_df = pd.DataFrame(results)
    
    print("\n" + "=" * 80)
    print("GRID SEARCH COMPLETE")
    print("=" * 80)
    print(f"\nBest parameters (based on VALIDATION set):")
    print(f"  k: {best_params['k']}")
    print(f"  lambda: {best_params['lambda']}")
    print(f"  mu: {best_params['mu']}")
    print(f"  Train RMSE: {best_params['train_rmse']:.4f}")
    print(f"  Validation RMSE: {best_params['val_rmse']:.4f}")
    print(f"  Validation Accuracy: {best_params['val_accuracy']:.4f}")
    
    return results_df, best_params

# Define hyperparameter search space
k_values = [1, 2, 5, 10, 15]
lambda_values = [0.001, 0.01, 0.1, 1.0]
mu_values = [0.001, 0.01, 0.1, 1.0]

# Run grid search with CV on MERGED dataset
print("\n" + "=" * 80)
print("Running grid search with 80/20 train/validation split on MERGED data")
print("=" * 80)

results_df, best_params = grid_search_cv(
    ratings_train_full=ratings_full,  # Use MERGED dataset
    k_values=k_values,
    lambda_values=lambda_values,
    mu_values=mu_values,
    num_iterations=30,
    cv_ratio=0.8,  # 80% train, 20% validation
    cv_seed=42
)

# Save results
os.makedirs('grid_search_results', exist_ok=True)
results_df.to_csv('grid_search_results/bounded_als_cv_results.csv', index=False)
print(f"\nResults saved to 'grid_search_results/bounded_als_cv_results.csv'")

# Display top 10 models
print("\nTop 10 models by validation RMSE:")
print(results_df.nsmallest(10, 'val_rmse'))

# Plot RMSE vs k (averaged over lambda and mu)
print("\nGenerating RMSE vs k plot...")
plot_rmse_vs_k(results_df, k_values)

# NOW train final model on FULL MERGED dataset with best hyperparameters
print("\n" + "=" * 80)
print("Training FINAL model on FULL MERGED dataset...")
print("=" * 80)

final_predictions = annls_optimization_bounded(
    ratings_full,  # ← Train on ALL merged data (train + test)
    k=best_params['k'], 
    num_iterations=50,  # More iterations for final model
    lambda_reg=best_params['lambda'], 
    mu_reg=best_params['mu'],
    seed=42
)
final_table = constrain_ratings(final_predictions)

# Evaluate on the full merged dataset (just for reference)
mask_full = ~np.isnan(ratings_full)
rmse_full, acc_full = evaluate_predictions(ratings_full, final_table, mask_full)

print(f"\nFinal model performance on FULL MERGED dataset:")
print(f"  RMSE: {rmse_full:.4f}")
print(f"  Accuracy: {acc_full:.4f}")
print(f"\nBest hyperparameters selected:")
print(f"  k={best_params['k']}, lambda={best_params['lambda']}, mu={best_params['mu']}")
print(f"\nNote: This model was trained on the full merged dataset.")
print(f"The validation RMSE during grid search was: {best_params['val_rmse']:.4f}")

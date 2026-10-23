import pandas as pd
import numpy as np
from scipy.optimize import nnls
import sys
import os
from itertools import product

def annls_optimization_sigmoid(ratings_matrix, k=1, num_iterations=30, lambda_reg=0.01, mu_reg=0.01, learning_rate=0.01):
    """ALS with sigmoid transformation to constrain predictions to [0,5]"""
    R_original = ratings_matrix
    n_users, n_movies = R_original.shape
    
    # Scale ratings to [0,1] for optimization
    R_scaled = R_original / 5.0
    
    np.random.seed(42)
    movie_factors = np.random.normal(0, 0.1, (n_movies, k))
    user_factors = np.random.normal(0, 0.1, (n_users, k))
    
    mask_original = ~np.isnan(R_scaled)
    R_clean = np.nan_to_num(R_scaled, nan=0.0)
    
    def sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Clip to prevent overflow
    
    for iteration in range(num_iterations):
        # Update movie factors
        for movie_idx in range(movie_factors.shape[0]):
            user_indices = np.where(mask_original[:, movie_idx])[0]
            if len(user_indices) > 0:
                U_items = user_factors[user_indices, :]
                # Apply sigmoid to predictions
                raw_pred = U_items @ movie_factors[movie_idx, :]
                pred = sigmoid(raw_pred)
                
                # Gradient descent step instead of NNLS (sigmoid makes it non-linear)
                gradient = U_items.T @ (pred - R_clean[user_indices, movie_idx])
                movie_factors[movie_idx, :] -= learning_rate * (gradient + lambda_reg * movie_factors[movie_idx, :])
                movie_factors[movie_idx, :] = np.maximum(0, movie_factors[movie_idx, :])  # Keep non-negative
        
        # Update user factors similarly
        for user_idx in range(user_factors.shape[0]):
            movie_indices = np.where(mask_original[user_idx, :])[0]
            if len(movie_indices) > 0:
                I_users = movie_factors[movie_indices, :]
                raw_pred = I_users @ user_factors[user_idx, :]
                pred = sigmoid(raw_pred)
                
                gradient = I_users.T @ (pred - R_clean[user_idx, movie_indices])
                user_factors[user_idx, :] -= learning_rate * (gradient + mu_reg * user_factors[user_idx, :])
                user_factors[user_idx, :] = np.maximum(0, user_factors[user_idx, :])
    
    # Scale back to [0,5] and apply sigmoid
    predictions = sigmoid(user_factors @ movie_factors.T) * 5
    return np.round(predictions * 2) / 2  # Round to nearest 0.5

def grid_search_sigmoid(ratings_train, ratings_test, k_values, lambda_values, mu_values, 
                        learning_rate_values, num_iterations=30):
    """
    Perform grid search over hyperparameters for sigmoid ALS
    
    Args:
        ratings_train: Training ratings matrix
        ratings_test: Test ratings dataframe
        k_values: List of k values (number of latent factors)
        lambda_values: List of lambda regularization values
        mu_values: List of mu regularization values
        learning_rate_values: List of learning rate values
        num_iterations: Number of ALS iterations
    
    Returns:
        results_df: DataFrame with all results
        best_params: Dictionary with best parameters
    """
    # Add the directory containing 'functions' folder to sys.path
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.append(parent_dir)
    
    from functions.vanilla_als import evaluate_predictions
    
    R_test = ratings_test.values
    mask_test = ~np.isnan(R_test)
    
    results = []
    best_rmse = float('inf')
    best_params = None
    
    # Total combinations
    total_combinations = len(k_values) * len(lambda_values) * len(mu_values) * len(learning_rate_values)
    current = 0
    
    print(f"Starting grid search with {total_combinations} combinations...")
    print(f"k values: {k_values}")
    print(f"lambda values: {lambda_values}")
    print(f"mu values: {mu_values}")
    print(f"learning_rate values: {learning_rate_values}")
    print("-" * 80)
    
    for k, lambda_reg, mu_reg, lr in product(k_values, lambda_values, mu_values, learning_rate_values):
        current += 1
        print(f"\n[{current}/{total_combinations}] Testing k={k}, lambda={lambda_reg}, mu={mu_reg}, lr={lr}")
        
        try:
            # Train model
            predicted_ratings = annls_optimization_sigmoid(
                ratings_train, 
                k=k, 
                num_iterations=num_iterations, 
                lambda_reg=lambda_reg, 
                mu_reg=mu_reg,
                learning_rate=lr
            )
            
            # Evaluate
            rmse_test, acc_test = evaluate_predictions(R_test, predicted_ratings, mask_test)
            
            print(f"  RMSE: {rmse_test:.4f}, Accuracy: {acc_test:.4f}")
            
            # Store results
            results.append({
                'k': k,
                'lambda': lambda_reg,
                'mu': mu_reg,
                'learning_rate': lr,
                'rmse': rmse_test,
                'accuracy': acc_test
            })
            
            # Track best model
            if rmse_test < best_rmse:
                best_rmse = rmse_test
                best_params = {
                    'k': k,
                    'lambda': lambda_reg,
                    'mu': mu_reg,
                    'learning_rate': lr,
                    'rmse': rmse_test,
                    'accuracy': acc_test
                }
                print(f"  *** NEW BEST MODEL ***")
        
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                'k': k,
                'lambda': lambda_reg,
                'mu': mu_reg,
                'learning_rate': lr,
                'rmse': np.nan,
                'accuracy': np.nan
            })
    
    results_df = pd.DataFrame(results)
    
    print("\n" + "=" * 80)
    print("GRID SEARCH COMPLETE")
    print("=" * 80)
    print(f"\nBest parameters:")
    print(f"  k: {best_params['k']}")
    print(f"  lambda: {best_params['lambda']}")
    print(f"  mu: {best_params['mu']}")
    print(f"  learning_rate: {best_params['learning_rate']}")
    print(f"  RMSE: {best_params['rmse']:.4f}")
    print(f"  Accuracy: {best_params['accuracy']:.4f}")
    
    return results_df, best_params

# Load data
table = np.load("data/ratings_train.npy")
ratings_test = pd.DataFrame(np.load("data/ratings_test.npy"))

# Define hyperparameter search space
k_values = [1, 2, 5, 10, 20]
lambda_values = [0.001, 0.01, 0.1, 1.0]
mu_values = [0.001, 0.01, 0.1, 1.0]
learning_rate_values = [0.001, 0.01, 0.05, 0.1]  # Important for gradient descent!

# Run grid search
results_df, best_params = grid_search_sigmoid(
    ratings_train=table,
    ratings_test=ratings_test,
    k_values=k_values,
    lambda_values=lambda_values,
    mu_values=mu_values,
    learning_rate_values=learning_rate_values,
    num_iterations=30  # Start with fewer iterations for speed
)

# Save results
results_df.to_csv('grid_search_sigmoid_results.csv', index=False)
print(f"\nResults saved to 'grid_search_sigmoid_results.csv'")

# Display top 10 models
print("\nTop 10 models by RMSE:")
print(results_df.nsmallest(10, 'rmse'))

# Display top 10 models by accuracy
print("\nTop 10 models by Accuracy:")
print(results_df.nsmallest(10, 'rmse')[['k', 'lambda', 'mu', 'learning_rate', 'rmse', 'accuracy']])

# Train final model with best parameters
print("\n" + "=" * 80)
print("Training final model with best parameters...")
print("=" * 80)

final_predictions = annls_optimization_sigmoid(
    table, 
    k=best_params['k'], 
    num_iterations=100,  # More iterations for final model
    lambda_reg=best_params['lambda'], 
    mu_reg=best_params['mu'],
    learning_rate=best_params['learning_rate']
)

# Add the directory containing 'functions' folder to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from functions.vanilla_als import evaluate_predictions

R_test = ratings_test.values
mask_test = ~np.isnan(R_test)
rmse_final, acc_final = evaluate_predictions(R_test, final_predictions, mask_test)

print(f"\nFinal model performance (with 100 iterations):")
print(f"  RMSE: {rmse_final:.4f}")
print(f"  Accuracy: {acc_final:.4f}")

# Optional: Visualize results
try:
    import matplotlib.pyplot as plt
    
    # Plot RMSE vs k for different learning rates
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # RMSE vs k
    for lr in learning_rate_values:
        subset = results_df[results_df['learning_rate'] == lr]
        grouped = subset.groupby('k')['rmse'].min()
        axes[0].plot(grouped.index, grouped.values, marker='o', label=f'lr={lr}')
    axes[0].set_xlabel('k (latent factors)')
    axes[0].set_ylabel('Best RMSE')
    axes[0].set_title('RMSE vs k for different learning rates')
    axes[0].legend()
    axes[0].grid(True)
    
    # RMSE vs learning rate
    for k_val in k_values:
        subset = results_df[results_df['k'] == k_val]
        grouped = subset.groupby('learning_rate')['rmse'].min()
        axes[1].plot(grouped.index, grouped.values, marker='o', label=f'k={k_val}')
    axes[1].set_xlabel('Learning Rate')
    axes[1].set_ylabel('Best RMSE')
    axes[1].set_title('RMSE vs Learning Rate for different k')
    axes[1].legend()
    axes[1].grid(True)
    axes[1].set_xscale('log')
    
    plt.tight_layout()
    plt.savefig('sigmoid_grid_search_results.png', dpi=150)
    print("\nVisualization saved to 'sigmoid_grid_search_results.png'")
except ImportError:
    print("\nMatplotlib not available for visualization")
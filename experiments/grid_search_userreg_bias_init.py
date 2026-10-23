import numpy as np
import pandas as pd
import sys
import os
from itertools import product

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from userreg import UserReg


def grid_search_userreg_bias_init(
    ratings_train,
    ratings_test,
    k_values,
    learning_rate_values,
    lambda_reg_val,
    beta_reg_val,
    bias_init_type,
    num_iterations,
    random_state,
):
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
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.append(parent_dir)

    R_test = ratings_test.values
    mask_test = ~np.isnan(R_test)

    results = []
    best_rmse = float("inf")
    best_params = None

    # Total combinations
    total_combinations = (
        len(k_values)
        * len(learning_rate_values)
        * len(lambda_reg_val)
        * len(beta_reg_val)
        * len(bias_init_type)
    )
    current = 0

    print(f"Starting grid search with {total_combinations} combinations...")
    print(f"k values: {k_values}")
    print(f"learning_rate values: {learning_rate_values}")
    print(f"lambda values: {lambda_reg_val}")
    print(f"beta values: {beta_reg_val}")
    print(f"bias_init types: {bias_init_type}")

    for k, lr, lambd, beta, bias in product(
        k_values, learning_rate_values, lambda_reg_val, beta_reg_val, bias_init_type
    ):
        current += 1
        print(
            f"\n[{current}/{total_combinations}] Testing k={k}, lr={lr}, lambda={lambd}, beta={beta}, bias_init={bias}"
        )

        try:
            # Initialize and train model
            model = UserReg(
                k,
                lr,
                lambd,
                beta,
                bias,
                num_iterations,
                random_state=random_state,
                verbose=False,
            )
            model.fit(ratings_train)

            # Get predictions
            preds = model.predict()

            # Evaluate on test set
            mask_test = ~np.isnan(R_test)
            true_values = R_test[mask_test]
            pred_values = preds[mask_test]

            # Calculate RMSE
            rmse = np.sqrt(np.mean((true_values - pred_values) ** 2))

            print(f"RMSE: {rmse:.4f}")

            # Store results
            results.append(
                {
                    "k": k,
                    "lr": lr,
                    "lambda": lambd,
                    "beta": beta,
                    "bias_init": bias,
                    "rmse": rmse,
                }
            )

            if rmse < best_rmse:
                best_rmse = rmse
                best_params = {
                    "k": k,
                    "lr": lr,
                    "lambda": lambd,
                    "beta": beta,
                    "bias_init": bias,
                    "rmse": rmse,
                }
                print(f"  *** NEW BEST MODEL ***")

        except Exception as e:
            print(
                f"Error with parameters {{"
                f"k: {k},"
                f"lr: {lr},"
                f"lambda: {lambd},"
                f"beta: {beta},"
                f"bias_init: '{bias}',"
                f"rmse: {np.nan}"
                f"}} : {str(e)}"
            )
            results.append(
                {
                    "k": k,
                    "lr": lr,
                    "lambda": lambd,
                    "beta": beta,
                    "bias_init": bias,
                    "rmse": np.nan,
                }
            )
            continue

    results_df = pd.DataFrame(results)

    print("\n" + "=" * 80)
    print("GRID SEARCH COMPLETE")
    print("=" * 80)
    print(f"\nBest parameters:")
    print(f"  k: {best_params['k']}")
    print(f"  lambda: {best_params['lambda']}")
    print(f"  mu: {best_params['beta']}")
    print(f"  learning_rate: {best_params['bias_init']}")
    print(f"  RMSE: {best_params['rmse']:.4f}")

    return results_df, best_params


# Load data
table = np.load("data/ratings_train.npy")
ratings_test = pd.DataFrame(np.load("data/ratings_test.npy"))

# Define hyperparameter search space
k_values = [3, 5, 8]
learning_rate_values = [0.006, 0.008, 0.01]
lambda_reg_val = [0.01, 0.02, 0.03]
beta_reg_val = [8]
bias_init_type = [
    "global_mean",
    "none",
    "weak_adapted_mean",
    "medium_adapted_mean",
    "strong_adapted_mean",
]

# Run grid search
results_df, best_params = grid_search_userreg_bias_init(
    ratings_train=table,
    ratings_test=ratings_test,
    k_values=k_values,
    learning_rate_values=learning_rate_values,
    lambda_reg_val=lambda_reg_val,
    beta_reg_val=beta_reg_val,
    bias_init_type=bias_init_type,
    random_state=43,
    num_iterations=30,  # Start with fewer iterations for speed
)

# Save results
results_df.to_csv("grid_search_userreg_bias_init.csv", index=False)
print(f"\nResults saved to 'grid_search_userreg_bias_init.csv'")

# Display top 10 models
print("\nTop 10 models by RMSE:")
print(results_df.nsmallest(10, "rmse"))

"""
# Train final model with best parameters
print("\n" + "=" * 80)
print("Training final model with best parameters...")
print("=" * 80)

final_predictions = annls_optimization_sigmoid(
    table,
    k=best_params["k"],
    num_iterations=100,  # More iterations for final model
    lambda_reg=best_params["lambda"],
    mu_reg=best_params["mu"],
    learning_rate=best_params["learning_rate"],
)

# Add the directory containing 'functions' folder to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from functions.vanilla_als import evaluate_predictions

R_test = ratings_test.values
mask_test = ~np.isnan(R_test)
rmse_final, acc_final = evaluate_predictions(R_test, final_predictions, mask_test)

print(f"\nFinal model performance (with 100 iterations):")
print(f"  RMSE: {rmse_final:.4f}")
print(f"  Accuracy: {acc_final:.4f}")
"""

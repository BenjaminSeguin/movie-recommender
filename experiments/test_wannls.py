import pandas as pd
import numpy as np
from scipy.optimize import nnls
import sys
import os

# Add the directory containing 'functions' folder to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from functions.vanilla_als import evaluate_predictions


table = np.load("data/ratings_train.npy")  # Load the ratings table
ratings_test = pd.DataFrame(np.load("data/ratings_test.npy"))  # Load the test ratings

#Weighted ALS

def wannls_optimization(ratings_matrix, k=1, num_iterations=30, lambda_reg=0.01, mu_reg=0.01, alpha = 0.2, beta = 0.2):
        
        user_activity = np.nansum(~np.isnan(ratings_matrix), axis=1).astype(float)  # number of ratings given by each user
        movie_popularity = np.nansum(~np.isnan(ratings_matrix), axis=0).astype(float)  # number of ratings received by each movie
            
        user_weights = np.power(user_activity, -alpha)
        movie_weight = np.power(movie_popularity, -beta)

        R_original = ratings_matrix
        n_users, n_movies = R_original.shape
        
        np.random.seed(42)
        movie_factors = np.random.normal(0, 0.1, (n_movies, k))  # 4980×k
        user_factors = np.random.normal(0, 0.1, (n_users, k))    # 610×k
        
        # Create mask for observed ratings
        mask_original = ~np.isnan(R_original)  # (610, 4980) users × movies
        R_clean = np.nan_to_num(R_original, nan=0.0)
        
        for iteration in range(num_iterations):
            # For each movie, find users who rated it and update its factors
            for movie_idx in range(movie_factors.shape[0]):  # 0 to 4979
                user_indices = np.where(mask_original[:, movie_idx])[0]
                if len(user_indices) > 0:
                    U_items = user_factors[user_indices, :]

                    weights = user_weights[user_indices]

                    ratings_i = R_clean[user_indices, movie_idx] #scalar multiplication of weights and ratings_i

                    A = U_items.T @ np.diag(weights) @ U_items + lambda_reg * np.eye(U_items.shape[1])
                    b = U_items.T @ np.diag(weights) @ ratings_i
                    movie_factors[movie_idx, :], _ = nnls(A, b)
            
            # For each user, find movies they rated and update their factors
            for user_idx in range(user_factors.shape[0]):  # 0 to 609
                movie_indices = np.where(mask_original[user_idx, :])[0]
                if len(movie_indices) > 0:
                    I_users = movie_factors[movie_indices, :]

                    weights = movie_weight[movie_indices]

                    ratings_u = R_clean[user_idx, movie_indices]

                    A = I_users.T @ np.diag(weights) @ I_users + mu_reg * np.eye(I_users.shape[1])
                    b = I_users.T @ np.diag(weights) @ ratings_u
                    user_factors[user_idx, :], _ = nnls(A, b)
        
        # Reconstruct: user_factors @ movie_factors.T gives (610×4980)
        return user_factors @ movie_factors.T

def constrain_ratings(predictions):
    """Constrain predictions to valid movie ratings: 0-5 in 0.5 increments"""
    clipped = np.clip(predictions, 0, 5)
    return np.round(clipped * 2) / 2

results = []
best_rmse = float("inf")
best_params = None
iter = 0 



#GRID SEARCH k & weights parameters alpha, beta
for alpha in [0, 0.01, 0.1]:
     for beta in [0, 0.01, 0.1]:
          for k in [1,2]:
               iter += 1
               print(f"({iter}/80)")
               print(f"Applying WANNLS matrix factorization with alpha={alpha}, beta={beta}, k={k}...")
               predicted_ratings2 = wannls_optimization(table, k=k, num_iterations=50, lambda_reg=0.1, mu_reg=0.1, alpha=alpha, beta=beta)
               table2 = constrain_ratings(predicted_ratings2)
               print("WANNLS optimization completed.")

               print(f"Final shape: {table2.shape} (users x movies)")

               # Evaluate on test data (only on observed test ratings)
               print("\nEvaluating on test dataset...")
               R_test2 = ratings_test.values
               mask_test2 = ~np.isnan(R_test2)

               rmse_test2, acc_test2 = evaluate_predictions(R_test2, table2, mask_test2)

               print(f"Test set performance:")
               print(f"  RMSE: {rmse_test2:.4f}")
               print(f"  Exact match accuracy: {acc_test2:.4f}")

               results.append(
                    {
                        "k": k,
                        "alpha": alpha,
                        "beta": beta,
                        "rmse": rmse_test2,
                    }
                )
               if rmse_test2 < best_rmse:
                    best_rmse = rmse_test2
                    best_params = {
                        "k": k,
                        "alpha": alpha,
                        "beta": beta,
                        "rmse": rmse_test2,
                    }
                    print(f"  *** NEW BEST MODEL ***")
                
results_df = pd.DataFrame(results)
#results_df.to_csv("wannls_grid_search_results.csv", index=False)
#results_df.to_csv("wannls_grid_search_results.csv", mode='a', header=not os.path.exists("wannls_grid_search_results.csv"), index=False)
#print(f"\nResults saved to 'wannls_grid_search_results.csv'")


print("\nTop 10 models by RMSE:")
print(results_df.nsmallest(10, "rmse"))
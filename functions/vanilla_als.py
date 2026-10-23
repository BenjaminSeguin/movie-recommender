import numpy as np
import pandas as pd

# ALS optimization
def als_optimization(ratings_matrix, I, U, num_iterations=10, lambda_reg=0.1, mu_reg=0.1):
    """
    ALS optimization for matrix factorization
    
    Args:
        ratings_matrix: The ratings matrix (items x users)
        I: Item matrix (items x k factors)
        U: User matrix (users x k factors) 
        num_iterations: Number of ALS iterations
        lambda_reg: Regularization parameter for ITEM factors (λ in your equation)
        mu_reg: Regularization parameter for USER factors (μ in your equation)
    
    Returns:
        Optimized I and U matrices
    """
    
    # Convert to numpy for easier computation
    R = ratings_matrix.values
    I_mat = I.values.copy()
    U_mat = U.values.copy()
    
    # Create mask for observed ratings (non-NaN entries)
    mask = ~np.isnan(R)
    
    # Replace NaN values with 0 for computation (we know it's not optimal and biased since 0 can be a rating)
    R_clean = np.nan_to_num(R, nan=0.0)
    
    losses = []
    
    print(f"Data sparsity: {np.sum(mask) / mask.size * 100:.2f}% observed ratings")
    print(f"Item regularization (λ): {lambda_reg}")
    print(f"User regularization (μ): {mu_reg}")
    
    for iteration in range(num_iterations):
        # Update I matrix (fix U, optimize I)
        for i in range(I_mat.shape[0]):  # For each item
            # Find users who rated this item
            user_indices = np.where(mask[i, :])[0]
            if len(user_indices) > 0:
                U_users = U_mat[user_indices, :]  # Users who rated item i
                ratings_i = R_clean[i, user_indices]    # Ratings for item i
                
                # Solve I_i = argmin ||...||^2 + λ * ||I_i||^2
                A = U_users.T @ U_users + lambda_reg * np.eye(U_users.shape[1])
                b = U_users.T @ ratings_i
                
                # Use regularized least squares 
                I_mat[i, :] = np.linalg.solve(A, b)

        
        # Update U matrix (fix I, optimize U)  
        for u in range(U_mat.shape[0]):  # For each user
            # Find items rated by this user
            item_indices = np.where(mask[:, u])[0]
            if len(item_indices) > 0:
                I_items = I_mat[item_indices, :]  # Items rated by user u
                ratings_u = R_clean[item_indices, u]   # Ratings by user u
                
                # Solve U_u = argmin ||...||^2 + μ * ||U_u||^2
                A = I_items.T @ I_items + mu_reg * np.eye(I_items.shape[1])
                b = I_items.T @ ratings_u
                
                # Use regularized least squares
                U_mat[u, :] = np.linalg.solve(A, b)
        
        # Calculate reconstruction loss only on observed entries
        predicted = I_mat @ U_mat.T
        reconstruction_error = np.sum((mask * (R_clean - predicted)) ** 2)
        regularization_term = lambda_reg * np.sum(I_mat**2) + mu_reg * np.sum(U_mat**2)
        loss = reconstruction_error + regularization_term
        losses.append(loss)
            
        print(f"Iteration {iteration}, Loss: {loss:.6f}, RMSE: {np.sqrt(reconstruction_error / np.sum(mask)):.6f}")
    
    return pd.DataFrame(I_mat), pd.DataFrame(U_mat), losses

# Function to constrain ratings to valid movie rating format
def constrain_ratings(predictions):
    """
    Constrain predictions to valid movie ratings:
    - Between 0 and 5
    - In 0.5 increments
    """
    # First clip to [0, 5] range
    clipped = np.clip(predictions, 0, 5)
    
    # Round to nearest 0.5 increment
    rounded = np.round(clipped * 2) / 2
    
    return rounded

# Evaluation functions
def evaluate_predictions(true_ratings, predictions, mask):
    """
    Evaluate prediction quality on observed ratings
    """
    # Get only the observed ratings
    true_observed = true_ratings[mask]
    pred_observed = predictions[mask]
    
    # Remove any NaN values
    valid_mask = ~(np.isnan(true_observed) | np.isnan(pred_observed))
    true_clean = true_observed[valid_mask]
    pred_clean = pred_observed[valid_mask]
    
    if len(true_clean) == 0:
        return None, None, None
    
    # Calculate metrics
    rmse = np.sqrt(np.mean((true_clean - pred_clean) ** 2))
    
    # Calculate percentage of exact matches (for discrete ratings)
    exact_matches = np.sum(true_clean == pred_clean)
    accuracy = exact_matches / len(true_clean)
    
    return rmse, accuracy
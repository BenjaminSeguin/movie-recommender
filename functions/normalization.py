import numpy as np

#Normalize ratings toward mean 0 and variance 1
def normalize_ratings(R):
    print("Starting global rating normalization on dense matrix...")

    # Find all rating values that are not zero
    mask_original = ~np.isnan(R)  # (610, 4980) users × movies

    all_rated_items = R[mask_original]

    # If there are no ratings, do nothing
    if len(all_rated_items) == 0:
        print("No ratings found to normalize.")
        return R, (0.0, 1.0)

    # Calculate the global mean and standard deviation of all rated items
    global_mean = np.mean(all_rated_items)
    global_std_dev = np.std(all_rated_items)

    global_stats = (global_mean, global_std_dev)
    
    # Create a copy to store the normalized ratings
    normalized_matrix = R.copy()

    # Handle the case where overall standard deviation is 0
    if global_std_dev == 0:
        print("Warning: Global standard deviation is 0. Cannot normalize.")
        # Set all rated items to 0 (since rating - mean is 0)
        normalized_matrix[mask_original] = 0.0
        global_stats = (global_mean, 1.0) # Use 1.0 for std_dev to avoid issues
    else:
        # Apply normalization to all non-zero elements
        normalized_matrix[mask_original] = (all_rated_items - global_mean) / global_std_dev
        global_stats = (global_mean, global_std_dev)

    print(f"Global Mean: {global_mean:.2f}, Global Std Dev: {global_std_dev:.2f}")
    print("Normalization complete.")
    return normalized_matrix, global_stats

#Denormalize back to original scale
def denormalize_predictions(predicted_matrix_normalized, global_stats):
    predicted_matrix_normalized = np.array(predicted_matrix_normalized, dtype=float)

    global_mean, global_std_dev = global_stats
    
    print("\nStarting global denormalization...")

    # Apply the reverse transformation to the entire matrix of predictions
    denormalized_matrix = (predicted_matrix_normalized * global_std_dev) + global_mean

    print("Denormalization complete.")
    return denormalized_matrix
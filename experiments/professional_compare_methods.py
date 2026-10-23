import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import lsq_linear
import pickle
import torch

# allow imports from repository root
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from userreg import UserReg
from HERec import HECrec


def load_embeddings(path_dir, name):
    """Load pre-computed HIN embeddings from pickle files."""
    with open(os.path.join(path_dir, f"{name}_embeddings.pkl"), "rb") as f:
        emb = pickle.load(f)
    return emb


def compute_rmse_on_mask(R_true, R_pred, mask):
    mask = np.array(mask, dtype=bool)
    if mask.sum() == 0:
        return np.nan
    diff = R_pred[mask] - R_true[mask]
    return np.sqrt(np.mean(diff ** 2))


def stratified_per_user_masks(R, holdout_ratio=0.2, seed=0):
    """Stratified per-user holdout (hold out proportion from each user)"""
    n_users, n_items = R.shape
    train_mask = np.zeros_like(R, dtype=bool)
    test_mask = np.zeros_like(R, dtype=bool)
    rng = np.random.RandomState(seed)
    for u in range(n_users):
        items = np.where(~np.isnan(R[u, :]))[0]
        if len(items) == 0:
            continue
        perm = rng.permutation(len(items))
        cut = int((1 - holdout_ratio) * len(items))
        train_items = items[perm[:cut]]
        test_items = items[perm[cut:]]
        train_mask[u, train_items] = True
        test_mask[u, test_items] = True
    return train_mask, test_mask


def annls_optimization_bounded(ratings_matrix, k=1, num_iterations=30, lambda_reg=0.01, mu_reg=1, seed=42):
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


def run_bounded_als(R_full, R_train, test_mask, k=1, lambda_reg=0.01, mu_reg=1, num_iters=50, seed=0):
    """Run bounded ALS with proper regularization."""
    # Train the model once with all iterations
    preds = annls_optimization_bounded(
        R_train, 
        k=k, 
        num_iterations=num_iters,
        lambda_reg=lambda_reg, 
        mu_reg=mu_reg, 
        seed=seed
    )
    # Compute final RMSE
    rmse = compute_rmse_on_mask(R_full, preds, test_mask)
    # Return as list for compatibility with averaging code that expects hist[-5:]
    return [rmse] * 5  # Repeat 5 times so averaging last 5 works


def run_userreg(R_full, R_train, test_mask, k=5, num_iters=50, seed=0):
    """Run UserReg training."""
    n_users, n_items = R_train.shape
    ur = UserReg(k=k, lr=0.008, lambda_reg=0.02, beta_reg=8.0, num_iterations=num_iters, verbose=False)
    np.random.seed(seed)
    ur.P = 0.1 * np.random.randn(n_users, ur.k)
    ur.Q = 0.1 * np.random.randn(n_items, ur.k)
    ur.bu = np.zeros(n_users)
    ur.bi = np.zeros(n_items)
    mask_train = ~np.isnan(R_train)
    R_train_vals = np.nan_to_num(R_train, nan=0.0)
    rmse_hist = []
    for it in range(num_iters):
        for u in range(n_users):
            Iu = np.where(mask_train[u, :])[0]
            if len(Iu) == 0:
                continue
            mean_Q = np.mean(ur.Q[Iu, :], axis=0)
            for i in Iu:
                e_ui = R_train_vals[u, i] - (ur.bu[u] + ur.bi[i] + np.dot(ur.P[u], ur.Q[i]))
                ur.bu[u] += ur.lr * (e_ui - ur.lambda_reg * ur.bu[u])
                ur.bi[i] += ur.lr * (e_ui - ur.lambda_reg * ur.bi[i])
                ur.P[u] += ur.lr * (e_ui * ur.Q[i] - ur.beta_reg * (ur.P[u] - mean_Q) - ur.lambda_reg * ur.P[u])
                ur.Q[i] += ur.lr * (e_ui * ur.P[u] - ur.lambda_reg * ur.Q[i])
        preds = ur.P @ ur.Q.T + ur.bu[:, None] + ur.bi[None, :]
        rmse_hist.append(compute_rmse_on_mask(R_full, preds, test_mask))
    return rmse_hist


def run_herec(R_full, R_train, test_mask, 
              user_embeds_list, item_embeds_list,
              user_ids, item_ids,
              D_mf=10, D_emb=64, alpha=1, beta=1.5,
              num_iters=50, lr=0.001, seed=0):
    """
    Run HERec (Heterogeneous Embedding-based Collaborative Filtering).
    
    Parameters:
    - user_embeds_list: list of dicts with user embeddings from different meta-paths
    - item_embeds_list: list of dicts with item embeddings from different meta-paths
    - D_mf: latent dimension for matrix factorization
    - D_emb: dimension of fused embeddings
    - alpha, beta: weighting parameters for embedding terms
    """
    model = HECrec(
        user_embeds_list=user_embeds_list,
        item_embeds_list=item_embeds_list,
        user_ids=user_ids,
        item_ids=item_ids,
        D_mf=D_mf,
        D_emb=D_emb,
        alpha=alpha,
        beta=beta,
        seed=seed,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Train the model
    model.fit(
        R_train, 
        lr=lr, 
        epochs=num_iters,
        batch_size=4096,
        lambda_xy=1e-3,
        lambda_gamma_u=1e-3,
        lambda_gamma_i=1e-3,
        lambda_theta_u=1e-3,
        lambda_theta_i=1e-3,
        verbose=False
    )
    
    # Get predictions
    R_pred = model.predict()
    
    # Compute RMSE on test set
    rmse = compute_rmse_on_mask(R_full, R_pred, test_mask)
    
    # Return as list for compatibility
    return [rmse] * 5


def run_from_masks(R, train_mask, test_mask, k=1, lambda_reg=0.01, mu_reg=1, 
                   num_iters=50, seed=0, model='bounded',
                   herec_config=None):
    """Run specified model with given masks."""
    R_train = np.where(train_mask, R, np.nan)
    
    if model == 'bounded':
        return run_bounded_als(R, R_train, test_mask, k=k, lambda_reg=lambda_reg, 
                              mu_reg=mu_reg, num_iters=num_iters, seed=seed)
    elif model == 'userreg':
        return run_userreg(R, R_train, test_mask, k=k, num_iters=num_iters, seed=seed)
    elif model == 'herec':
        if herec_config is None:
            raise ValueError("herec_config must be provided for HERec model")
        return run_herec(R, R_train, test_mask, 
                        user_embeds_list=herec_config['user_embeds_list'],
                        item_embeds_list=herec_config['item_embeds_list'],
                        user_ids=herec_config['user_ids'],
                        item_ids=herec_config['item_ids'],
                        D_mf=herec_config.get('D_mf', 10),
                        D_emb=herec_config.get('D_emb', 64),
                        alpha=herec_config.get('alpha', 1),
                        beta=herec_config.get('beta', 1.5),
                        num_iters=num_iters,
                        lr=herec_config.get('lr', 0.001),
                        seed=seed)
    else:
        raise ValueError(f"Unknown model: {model}")


def run_stratified_per_user_comparison(ratios=None, seeds=None, num_iters=50, 
                                      k_bounded=1, k_userreg=5,
                                      lambda_reg=0.01, mu_reg=1,
                                      herec_config=None):
    """
    Run stratified per-user CV and compare all 3 methods:
    - Bounded ALS
    - UserReg
    - HERec (Heterogeneous Embedding-based CF)
    
    Parameters:
    - k_bounded: latent factors for Bounded ALS
    - k_userreg: latent factors for UserReg
    - herec_config: dict with HERec configuration (embeddings, ids, hyperparams)
    """
    if ratios is None:
        ratios = [0.2, 0.4, 0.6, 0.8]
    if seeds is None:
        seeds = [0, 1, 2]

    # Load data
    train = np.load('data/ratings_train.npy')
    test = np.load('data/ratings_test.npy')
    R = np.where(~np.isnan(test), test, train)

    print(f"Dataset shape: {R.shape}")
    print(f"Total observed ratings: {(~np.isnan(R)).sum()}")
    print(f"Sparsity: {(~np.isnan(R)).sum() / R.size:.4f}")
    print(f"\nModel configurations:")
    print(f"  Bounded ALS: k={k_bounded}, λ={lambda_reg}, μ={mu_reg}")
    print(f"  UserReg: k={k_userreg}")
    if herec_config:
        print(f"  HERec: D_mf={herec_config.get('D_mf', 10)}, D_emb={herec_config.get('D_emb', 64)}, "
              f"α={herec_config.get('alpha', 1)}, β={herec_config.get('beta', 1.5)}")
    print()

    os.makedirs('results', exist_ok=True)

    models = {
        'Bounded ALS': {'key': 'bounded', 'k': k_bounded},
        'UserReg': {'key': 'userreg', 'k': k_userreg},
        'HERec': {'key': 'herec', 'k': None}
    }
    
    results = {name: {'means': [], 'stds': []} for name in models.keys()}

    print("=" * 80)
    print("Running Stratified Per-User Cross-Validation")
    print("=" * 80)

    for ratio in ratios:
        print(f"\n--- Training ratio: {ratio:.1%} ---")
        
        for model_name, model_config in models.items():
            per_seed_rmse = []
            model_key = model_config['key']
            model_k = model_config['k']
            
            for seed in seeds:
                train_mask, test_mask = stratified_per_user_masks(
                    R, holdout_ratio=1-ratio, seed=seed
                )
                
                n_train = train_mask.sum()
                n_test = test_mask.sum()
                k_display = f"k={model_k}" if model_k is not None else "HIN-based"
                print(f"  {model_name} [seed={seed}, {k_display}]: train={n_train}, test={n_test}", end=" ")
                
                hist = run_from_masks(
                    R, train_mask, test_mask, 
                    k=model_k if model_k is not None else 1,
                    lambda_reg=lambda_reg, mu_reg=mu_reg,
                    num_iters=num_iters, seed=seed, model=model_key,
                    herec_config=herec_config
                )
                
                # Average last 5 iterations
                final_rmse = np.nanmean(hist[-5:]) if len(hist) >= 5 else np.nanmean(hist)
                per_seed_rmse.append(final_rmse)
                print(f"→ RMSE: {final_rmse:.4f}")
            
            results[model_name]['means'].append(np.nanmean(per_seed_rmse))
            results[model_name]['stds'].append(np.nanstd(per_seed_rmse))

    # Debug: print std values
    for model_name in results.keys():
        stds = results[model_name]['stds']
        print(f"\n{model_name} standard deviations:")
        for i, (ratio, std) in enumerate(zip(ratios, stds)):
            print(f"  {int(100*ratio)}%: std = {std:.6f}")
    
    # Create multiple visualization options
    from visualization_options import create_all_visualizations
    create_all_visualizations(results, ratios, k_bounded, k_userreg, 
                             base_path='results/cv_stratified_per_user_original')
    
    # Print summary table
    print("\nSummary Table (mean ± std RMSE):")
    print("-" * 90)
    print(f"{'Training %':<12} | {'Bounded ALS (k=' + str(k_bounded) + ')':<24} | {'UserReg (k=' + str(k_userreg) + ')':<24} | {'HERec':<20}")
    print("-" * 90)
    for i, ratio in enumerate(ratios):
        row = f"{int(100*ratio):>10}%  |"
        for model_name in models.keys():
            mean = results[model_name]['means'][i]
            std = results[model_name]['stds'][i]
            row += f" {mean:>6.4f} ± {std:<6.4f}    |"
        print(row)
    print("-" * 90)
    
    plt.close()


if __name__ == '__main__':
    print("Running Stratified Per-User Cross-Validation with 3 Models\n")
    
    # Load HIN embeddings for HERec
    print("Loading HIN embeddings...")
    base_dir_1 = "hin_embeddings_runs/run_20251019_120918"
    model_UMU = load_embeddings(os.path.join(base_dir_1, "UMU"), "UMU")
    model_MGM = load_embeddings(os.path.join(base_dir_1, "MGM"), "MGM")
    model_MYM = load_embeddings(os.path.join(base_dir_1, "MYM"), "MYM")
    
    base_dir_2 = "hin_embeddings_runs/run_20251021_110033"
    model_UMYMU = load_embeddings(os.path.join(base_dir_2, "UMYMU"), "UMYMU")
    model_UMGMU = load_embeddings(os.path.join(base_dir_2, "UMGMU"), "UMGMU")
    model_MGMYM = load_embeddings(os.path.join(base_dir_2, "MGMYM"), "MGMYM")
    
    # Configure HERec
    user_embeds_list = [model_UMU, model_UMYMU, model_UMGMU]
    item_embeds_list = [model_MGM, model_MYM, model_MGMYM]
    user_ids = sorted(model_UMU.keys(), key=lambda x: int(x[1:]))
    item_ids = sorted(model_MGM.keys(), key=lambda x: int(x[1:]))
    
    herec_config = {
        'user_embeds_list': user_embeds_list,
        'item_embeds_list': item_embeds_list,
        'user_ids': user_ids,
        'item_ids': item_ids,
        'D_mf': 10,       # Best from your tuning
        'D_emb': 64,
        'alpha': 1,   # Best from your tuning
        'beta': 1.5,    # Best from your tuning
        'lr': 0.001
    }
    
    print("Embeddings loaded successfully!\n")
    
    # Run comparison
    run_stratified_per_user_comparison(
        ratios=[0.2, 0.4, 0.6, 0.8],
        seeds=[0, 1, 2],
        num_iters=50,
        k_bounded=1,
        k_userreg=5,
        lambda_reg=0.01,
        mu_reg=1,
        herec_config=herec_config
    )
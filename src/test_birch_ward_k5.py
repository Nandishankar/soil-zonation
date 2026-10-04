import pandas as pd
import numpy as np
import joblib
from sklearn.cluster import MiniBatchKMeans
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import pairwise_distances_argmin

def run_two_phase_ward(X_scaled, features, seed, k_cut):
    mbk = MiniBatchKMeans(n_clusters=2000, random_state=seed, batch_size=2000, n_init=3)
    mbk_labels = mbk.fit_predict(X_scaled)
    
    centroids = mbk.cluster_centers_
    weights = np.bincount(mbk_labels)
    
    expanded_centroids = []
    for i in range(2000):
        repeats = max(1, int(np.round(weights[i] / 40.0)))
        for _ in range(repeats):
            expanded_centroids.append(centroids[i])
            
    expanded_centroids = np.array(expanded_centroids)
    
    Z = linkage(expanded_centroids, method='ward')
    expanded_labels = fcluster(Z, k_cut, criterion='maxclust')
    
    final_centroids = []
    for c_id in range(1, k_cut + 1):
        final_centroids.append(expanded_centroids[expanded_labels == c_id].mean(axis=0))
    final_centroids = np.array(final_centroids)
    
    # Unscale centroids
    scaler = joblib.load('data/processed/scaler.joblib')
    centroids_unscaled = scaler.inverse_transform(final_centroids)
    
    return centroids_unscaled

def main():
    print("Loading data...")
    df = pd.read_csv('data/processed/shc_zoned.csv')
    scaler = joblib.load('data/processed/scaler.joblib')
    
    features = [
        'n_index', 'p_index', 'k_index', 'oc_index',
        'b_pct_sufficient', 'cu_pct_sufficient', 'fe_pct_sufficient', 'mn_pct_sufficient',
        's_pct_sufficient', 'zn_pct_sufficient',
        'ph_pct_acidic', 'ph_pct_alkaline', 'ec_pct_non_saline'
    ]
    
    X_scaled = scaler.transform(df[features])
    
    for k in [5, 6]:
        print(f"\n--- Testing K={k} across 3 Seeds ---")
        for seed in [42, 7, 123]:
            centroids_unscaled = run_two_phase_ward(X_scaled, features, seed, k)
            
            # Check if Saline is present (ec_pct_non_saline < 0.7)
            ec_idx = features.index('ec_pct_non_saline')
            saline_clusters = (centroids_unscaled[:, ec_idx] < 0.7).sum()
            print(f"Seed {seed} at K={k}: Found {saline_clusters} Saline clusters. EC non-saline values: {np.round(centroids_unscaled[:, ec_idx], 2)}")

if __name__ == '__main__':
    main()

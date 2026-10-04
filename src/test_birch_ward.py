import pandas as pd
import numpy as np
import joblib
import time
import os
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.metrics import pairwise_distances_argmin

def get_unscaled_mean(df_wide, labels, features):
    df_wide['temp_label'] = labels
    return df_wide.groupby('temp_label')[features].mean()

def run_two_phase_ward(X_scaled, df_wide, features, seed):
    mbk = MiniBatchKMeans(n_clusters=2000, random_state=seed, batch_size=2000, n_init=3)
    mbk_labels = mbk.fit_predict(X_scaled)
    
    centroids = mbk.cluster_centers_
    weights = np.bincount(mbk_labels)
    
    scaler = joblib.load('data/processed/scaler.joblib')
    centroids_unscaled = scaler.inverse_transform(centroids)
    
    ec_idx = features.index('ec_pct_non_saline')
    saline_mask = centroids_unscaled[:, ec_idx] < 0.7
    saline_micro_clusters = saline_mask.sum()
    saline_weight = weights[saline_mask].sum()
    
    print(f"\n[Seed {seed}] Phase 1 Sanity Check:")
    print(f"  Micro-clusters with ec_pct_non_saline < 0.7: {saline_micro_clusters} / 2000")
    print(f"  Total weight of these Saline micro-clusters: {saline_weight} villages")
    
    expanded_centroids = []
    for i in range(2000):
        repeats = max(1, int(np.round(weights[i] / 40.0)))
        for _ in range(repeats):
            expanded_centroids.append(centroids[i])
            
    expanded_centroids = np.array(expanded_centroids)
    print(f"  Expanded centroids shape for Linkage: {expanded_centroids.shape}")
    
    Z = linkage(expanded_centroids, method='ward')
    
    if seed == 42:
        os.makedirs('outputs/figures', exist_ok=True)
        plt.figure(figsize=(10, 6))
        dendrogram(Z, truncate_mode='level', p=5)
        plt.title(f'Weighted Ward Dendrogram (Seed {seed})')
        plt.savefig('outputs/figures/ward_dendrogram.png')
        plt.close()
        print("  Saved dendrogram to outputs/figures/ward_dendrogram.png")
        
    expanded_labels = fcluster(Z, 4, criterion='maxclust')
    
    final_centroids = []
    for c_id in range(1, 5):
        final_centroids.append(expanded_centroids[expanded_labels == c_id].mean(axis=0))
    final_centroids = np.array(final_centroids)
    
    final_labels = pairwise_distances_argmin(X_scaled, final_centroids)
    
    means = get_unscaled_mean(df_wide, final_labels, features)
    
    saline_idx = means['ec_pct_non_saline'].idxmin()
    acidic_idx = means['ph_pct_acidic'].idxmax()
    
    # We will pick deficient carefully. To avoid picking saline as deficient by accident,
    # we take the minimum fe_pct_sufficient among the remaining two.
    remaining = set([0, 1, 2, 3]) - {saline_idx, acidic_idx}
    deficient_idx = min(remaining, key=lambda idx: means.loc[idx, 'fe_pct_sufficient'])
    
    remaining = remaining - {deficient_idx}
    fertile_idx = list(remaining)[0]
    
    zone_names = {
        deficient_idx: 'Micronutrient-Deficient',
        acidic_idx: 'Acidic/High-N/OC',
        saline_idx: 'Saline',
        fertile_idx: 'Generally Fertile'
    }
    
    named_labels = pd.Series(final_labels).map(zone_names)
    dist = named_labels.value_counts(normalize=True) * 100
    dist_counts = named_labels.value_counts()
    
    return dist, dist_counts

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
    
    # Make sure we don't have NA from the old labels
    X_unscaled = df[features].values
    X_scaled = scaler.transform(df[features])
    
    results = {}
    for seed in [42, 7, 123]:
        dist, dist_counts = run_two_phase_ward(X_scaled, df.copy(), features, seed)
        results[seed] = dist
        
    print("\n--- STABILITY VALIDATION ACROSS 3 SEEDS ---")
    df_res = pd.DataFrame(results).fillna(0).round(1)
    df_res.columns = [f"Seed_{c}" for c in df_res.columns]
    print(df_res)
    
if __name__ == '__main__':
    main()

import pandas as pd
import numpy as np
import joblib
from sklearn.cluster import MiniBatchKMeans
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import pairwise_distances_argmin

def get_unscaled_mean(df_wide, labels, features):
    df_wide['temp_label'] = labels
    return df_wide.groupby('temp_label')[features].mean()

def validate_linkage(X_scaled, df, features, seed, method):
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
    
    Z = linkage(expanded_centroids, method=method)
    
    labels_5 = fcluster(Z, 5, criterion='maxclust')
    labels_4 = fcluster(Z, 4, criterion='maxclust')
    labels_3 = fcluster(Z, 3, criterion='maxclust')
    
    merge_5_to_4 = []
    reverse_map_4 = {}
    for l5, l4 in zip(labels_5, labels_4):
        reverse_map_4.setdefault(l4, set()).add(l5)
    for l4, l5_set in reverse_map_4.items():
        if len(l5_set) > 1:
            merge_5_to_4 = list(l5_set)
            break
            
    merge_4_to_3 = []
    reverse_map_3 = {}
    for l4, l3 in zip(labels_4, labels_3):
        reverse_map_3.setdefault(l3, set()).add(l4)
    for l3, l4_set in reverse_map_3.items():
        if len(l4_set) > 1:
            merge_4_to_3 = list(l4_set)
            break
            
    print(f"\n--- {method.upper()} LINKAGE | SEED {seed} ---")
    
    final_centroids_scaled = []
    for c_id in range(1, 5):
        final_centroids_scaled.append(expanded_centroids[labels_4 == c_id].mean(axis=0))
    final_centroids_scaled = np.array(final_centroids_scaled)
    
    final_labels = pairwise_distances_argmin(X_scaled, final_centroids_scaled) + 1
    means = get_unscaled_mean(df.copy(), final_labels, features)
    
    saline_idx = means['ec_pct_non_saline'].idxmin()
    acidic_idx = means['ph_pct_acidic'].idxmax()
    remaining = set([1, 2, 3, 4]) - {saline_idx, acidic_idx}
    deficient_idx = min(remaining, key=lambda idx: means.loc[idx, 'fe_pct_sufficient'])
    remaining = remaining - {deficient_idx}
    fertile_idx = list(remaining)[0]
    
    role_map = {
        saline_idx: 'Saline',
        acidic_idx: 'Acidic',
        deficient_idx: 'Deficient',
        fertile_idx: 'Fertile'
    }
    
    merged_roles = [role_map.get(idx, "Unknown") for idx in merge_4_to_3]
    if not merged_roles: merged_roles = ["Unknown", "Unknown"]
    
    print(f"Merge K=5 to K=4: K=5 Clusters {merge_5_to_4}")
    if merge_4_to_3:
        print(f"Merge K=4 to K=3: K=4 Clusters {merge_4_to_3[0]} ({merged_roles[0]}) and {merge_4_to_3[1]} ({merged_roles[1]}) merge.")
    else:
        print("Merge K=4 to K=3: Could not be resolved properly.")
    
    dist_counts = pd.Series(final_labels).value_counts(normalize=True) * 100
    
    means.index = means.index.map(role_map)
    print("\nFull Dataset Centroids (Unscaled) at K=4:")
    print(means.round(3).to_string())
    
    print("\nFull Dataset Distribution:")
    for role in ['Saline', 'Acidic', 'Deficient', 'Fertile']:
        idx = [k for k, v in role_map.items() if v == role][0]
        size = dist_counts.get(idx, 0)
        print(f"  {role}: {size:.1f}%")

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
    
    for method in ['average', 'complete']:
        for seed in [42, 7, 123]:
            validate_linkage(X_scaled, df, features, seed, method)

if __name__ == '__main__':
    main()

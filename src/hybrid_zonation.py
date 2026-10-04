import pandas as pd
import numpy as np
import joblib
import time
import os
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import pairwise_distances_argmin

def get_unscaled_mean(df_wide, labels, features):
    df_wide['temp_label'] = labels
    return df_wide.groupby('temp_label')[features].mean()

def run_residual_ward(X_res_scaled, df_res, features, seed):
    mbk = MiniBatchKMeans(n_clusters=2000, random_state=seed, batch_size=2000, n_init=3)
    mbk_labels = mbk.fit_predict(X_res_scaled)
    
    centroids = mbk.cluster_centers_
    weights = np.bincount(mbk_labels)
    
    expanded_centroids = []
    for i in range(2000):
        repeats = max(1, int(np.round(weights[i] / 40.0)))
        for _ in range(repeats):
            expanded_centroids.append(centroids[i])
            
    expanded_centroids = np.array(expanded_centroids)
    
    Z = linkage(expanded_centroids, method='ward')
    expanded_labels = fcluster(Z, 3, criterion='maxclust')
    
    final_centroids = []
    for c_id in range(1, 4):
        final_centroids.append(expanded_centroids[expanded_labels == c_id].mean(axis=0))
    final_centroids = np.array(final_centroids)
    
    final_labels = pairwise_distances_argmin(X_res_scaled, final_centroids)
    
    means = get_unscaled_mean(df_res.copy(), final_labels, features)
    
    acidic_idx = means['ph_pct_acidic'].idxmax()
    deficient_idx = means['fe_pct_sufficient'].idxmin()
    
    remaining = set([0, 1, 2]) - {acidic_idx, deficient_idx}
    fertile_idx = list(remaining)[0]
    
    zone_names = {
        deficient_idx: 'Micronutrient-Deficient',
        acidic_idx: 'Acidic/High-N/OC',
        fertile_idx: 'Generally Fertile'
    }
    
    named_labels = pd.Series(final_labels).map(zone_names)
    dist = named_labels.value_counts(normalize=True) * 100
    
    return dist, named_labels.values

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
    
    print("\n--- STAGE 1: MACRO DENSITY PRE-SEGMENTATION ---")
    min_samples = 100
    print(f"Computing k-distance (k={min_samples})...")
    neigh = NearestNeighbors(n_neighbors=min_samples, algorithm='kd_tree', n_jobs=-1)
    neigh.fit(X_scaled)
    distances, _ = neigh.kneighbors(X_scaled)
    k_dist = distances[:, -1]
    k_dist_sorted = np.sort(k_dist)
    
    chosen_eps = np.percentile(k_dist_sorted, 95)
    print(f"Chosen EPS (95th percentile): {chosen_eps:.3f}")
    
    print("Running DBSCAN...")
    db = DBSCAN(eps=chosen_eps, min_samples=min_samples, algorithm='kd_tree', n_jobs=-1)
    db_labels = db.fit_predict(X_scaled)
    
    df['dbscan_macro_label'] = db_labels
    
    print("\nDBSCAN Macro Clusters found (excluding noise -1):")
    unique_labels = [l for l in np.unique(db_labels) if l != -1]
    
    cluster_info = []
    
    for l in unique_labels:
        mask = db_labels == l
        size = mask.sum()
        pct = size / len(df) * 100
        mean_feats = df.loc[mask, features].mean()
        cluster_info.append({'label': l, 'size': size, 'pct': pct, 'means': mean_feats})
        print(f"  Cluster {l}: {size} villages ({pct:.2f}%)")
        print(f"    EC non-saline: {mean_feats['ec_pct_non_saline']:.3f}, pH acidic: {mean_feats['ph_pct_acidic']:.3f}, Fe suff: {mean_feats['fe_pct_sufficient']:.3f}")
        
    print("\nEvaluating for Macro Density Zones (>= 1%)...")
    density_zones = {}
    for info in cluster_info:
        if info['pct'] >= 1.0:
            mean_feats = info['means']
            if mean_feats['ec_pct_non_saline'] < 0.7:
                name = 'Saline'
            elif mean_feats['ph_pct_acidic'] > 0.5:
                name = 'Acidic/High-N/OC'
            elif mean_feats['fe_pct_sufficient'] < 0.6:
                name = 'Micronutrient-Deficient'
            else:
                name = f'DensityZone_{info["label"]}'
                
            density_zones[info['label']] = name
            print(f"  -> Kept Cluster {info['label']} as '{name}' ({info['size']} villages)")
        else:
            print(f"  -> Dropped Cluster {info['label']} (<1% size)")
            
    df['zone_method'] = 'ward_residual'
    df['final_zone_name'] = None
    
    for l, name in density_zones.items():
        mask = df['dbscan_macro_label'] == l
        df.loc[mask, 'final_zone_name'] = name
        df.loc[mask, 'zone_method'] = 'density_presegmentation'
        
    # -----------------------------------------------------------------
    # STAGE 2: RESIDUAL WARD CLUSTERING
    # -----------------------------------------------------------------
    print("\n--- STAGE 2: RESIDUAL WARD CLUSTERING ---")
    residual_mask = df['final_zone_name'].isna()
    df_res = df[residual_mask].copy()
    X_res_scaled = X_scaled[residual_mask]
    
    print(f"Residual population size: {len(df_res)} villages ({(len(df_res)/len(df)*100):.1f}%)")
    
    results = {}
    saved_labels = None
    for seed in [42, 7, 123]:
        dist, labels = run_residual_ward(X_res_scaled, df_res, features, seed)
        results[seed] = dist
        if seed == 42:
            saved_labels = labels
            
    print("\nResidual Ward Stability (3 Seeds):")
    df_res_stability = pd.DataFrame(results).fillna(0).round(1)
    df_res_stability.columns = [f"Seed_{c}" for c in df_res_stability.columns]
    print(df_res_stability)
    
    df.loc[residual_mask, 'final_zone_name'] = saved_labels
    
    print("\n--- STAGE 3: FINAL COMBINED ZONE DISTRIBUTION ---")
    dist = df['final_zone_name'].value_counts()
    for name, count in dist.items():
        print(f"  {name}: {count} ({count/len(df)*100:.1f}%)")
        
    out_cols = [c for c in df.columns if c not in ['soil_zone', 'soil_zone_name', 'dbscan_macro_label', 'final_zone_name', 'temp_label']]
    out_cols.extend(['final_zone_name', 'zone_method'])
    df_zoned = df[out_cols].rename(columns={'final_zone_name': 'soil_zone_name'})
    df_zoned.to_csv('data/processed/shc_zoned.csv', index=False)
    print("\nSaved final updated shc_zoned.csv")

if __name__ == '__main__':
    main()

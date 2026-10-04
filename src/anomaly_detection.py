import pandas as pd
import numpy as np
import joblib
import time
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN

def get_human_readable_zscore(feature_name, z):
    magnitude = "moderately"
    if abs(z) > 2: magnitude = "extremely"
    elif abs(z) > 1.5: magnitude = "very"
    
    direction = "high" if z > 0 else "low"
    
    mapping = {
        'n_index': 'N index',
        'p_index': 'P index',
        'k_index': 'K index',
        'oc_index': 'OC index',
        'b_pct_sufficient': 'Boron sufficiency',
        'cu_pct_sufficient': 'Copper sufficiency',
        'fe_pct_sufficient': 'Iron sufficiency',
        'mn_pct_sufficient': 'Manganese sufficiency',
        's_pct_sufficient': 'Sulphur sufficiency',
        'zn_pct_sufficient': 'Zinc sufficiency',
        'ph_pct_acidic': 'Acidic pH',
        'ph_pct_alkaline': 'Alkaline pH',
        'ec_pct_non_saline': 'Non-saline EC'
    }
    name = mapping.get(feature_name, feature_name)
    return f"{magnitude} {direction} {name} (z={z:.2f})"

def main():
    print("Loading shc_zoned.csv and scaler.joblib...")
    df = pd.read_csv('data/processed/shc_zoned.csv')
    scaler = joblib.load('data/processed/scaler.joblib')
    
    features = [
        'n_index', 'p_index', 'k_index', 'oc_index',
        'b_pct_sufficient', 'cu_pct_sufficient', 'fe_pct_sufficient', 'mn_pct_sufficient',
        's_pct_sufficient', 'zn_pct_sufficient',
        'ph_pct_acidic', 'ph_pct_alkaline', 'ec_pct_non_saline'
    ]
    
    X_unscaled = df[features].values
    X_scaled = scaler.transform(df[features])
    
    # ---------------------------------------------------------
    # 1. K-DISTANCE PLOT & DBSCAN
    # ---------------------------------------------------------
    min_samples = 26
    print(f"\nComputing k-distance (k={min_samples})...")
    neigh = NearestNeighbors(n_neighbors=min_samples, algorithm='auto', n_jobs=4)
    neigh.fit(X_scaled)
    distances, _ = neigh.kneighbors(X_scaled)
    k_dist = distances[:, -1]
    k_dist_sorted = np.sort(k_dist)
    
    import os
    os.makedirs('outputs/figures', exist_ok=True)
    plt.figure(figsize=(10,6))
    plt.plot(k_dist_sorted)
    plt.ylabel(f'{min_samples}-th Nearest Neighbor Distance')
    plt.xlabel('Data Points sorted by distance')
    plt.title('K-Distance Elbow Plot')
    plt.savefig('outputs/figures/k_distance_elbow.png')
    plt.close()
    
    chosen_eps = np.percentile(k_dist_sorted, 98)
    print(f"Chosen EPS (98th percentile elbow): {chosen_eps:.3f}")
    
    print("\nRunning DBSCAN...")
    t0 = time.time()
    db = DBSCAN(eps=chosen_eps, min_samples=min_samples, algorithm='auto', n_jobs=4)
    db_labels = db.fit_predict(X_scaled)
    print(f"DBSCAN finished in {time.time() - t0:.1f}s")
    
    df['dbscan_label'] = db_labels
    n_anomalies = (db_labels == -1).sum()
    pct_anomalies = n_anomalies / len(df) * 100
    print(f"Global Anomalies: {n_anomalies} ({pct_anomalies:.2f}%)")
    
    print("\nGlobal Anomalies by Ward Zone:")
    zone_counts = df[df['dbscan_label'] == -1]['soil_zone_name'].value_counts()
    for name, count in zone_counts.items():
        print(f"  {name}: {count}")
        
    # ---------------------------------------------------------
    # 2. ANOMALY EXPLANATION
    # ---------------------------------------------------------
    anomaly_mask = df['dbscan_label'] == -1
    explanations = []
    X_scaled_anomalies = X_scaled[anomaly_mask]
    
    for row in X_scaled_anomalies:
        abs_row = np.abs(row)
        top3_idx = np.argsort(abs_row)[-3:][::-1]
        reasons = [get_human_readable_zscore(features[i], row[i]) for i in top3_idx]
        explanations.append(" | ".join(reasons))
        
    df['anomaly_explanation'] = ""
    df.loc[anomaly_mask, 'anomaly_explanation'] = explanations
    
    print("\nExample Global Anomaly Explanations:")
    examples = df[df['dbscan_label'] == -1].sample(min(5, n_anomalies), random_state=42)
    for _, row in examples.iterrows():
        print(f"[{row['village_name']}, {row['district_name']}] Zone: {row['soil_zone_name']} -> {row['anomaly_explanation']}")

    # ---------------------------------------------------------
    # 3. LOCAL ANOMALY SCORE (Module 9)
    # ---------------------------------------------------------
    print("\nComputing Local Anomaly Scores (Module 9)...")
    
    state_means = df.groupby('state_name')[features].mean()
    state_stds = df.groupby('state_name')[features].std()
    
    fallback_to_state_count = 0
    fallback_std_to_state_count = 0
    fallback_std_to_eps_count = 0
    
    def process_district(group):
        nonlocal fallback_to_state_count, fallback_std_to_state_count, fallback_std_to_eps_count
        state_name = group['state_name'].iloc[0]
        
        if len(group) < 15:
            fallback_to_state_count += len(group)
            local_mean = state_means.loc[state_name].values
            local_std = state_stds.loc[state_name].values.copy()
        else:
            local_mean = group[features].mean().values
            local_std = group[features].std().values.copy()
            
        zeros_mask = (local_std == 0) | np.isnan(local_std)
        if zeros_mask.any():
            fallback_std_to_state_count += len(group)
            local_std[zeros_mask] = state_stds.loc[state_name].values[zeros_mask]
            
            zeros_mask2 = (local_std == 0) | np.isnan(local_std)
            if zeros_mask2.any():
                fallback_std_to_eps_count += len(group)
                local_std[zeros_mask2] = 1e-6
                
        group_X = group[features].values
        las_matrix = np.abs(group_X - local_mean) / local_std
        
        return pd.DataFrame({
            'local_anomaly_score_max': np.max(las_matrix, axis=1),
            'local_anomaly_score_mean': np.mean(las_matrix, axis=1)
        }, index=group.index)

    res = df.groupby('district_name', group_keys=False).apply(process_district)
    df['local_anomaly_score_max'] = res['local_anomaly_score_max']
    df['local_anomaly_score_mean'] = res['local_anomaly_score_mean']
    
    df['is_local_anomaly'] = df['local_anomaly_score_max'] > 3.0
    
    print(f"Fallback to state neighborhood (<15 villages): {fallback_to_state_count} rows")
    print(f"Fallback std to state std: {fallback_std_to_state_count} rows")
    print(f"Fallback std to epsilon: {fallback_std_to_eps_count} rows")
    
    n_local = df['is_local_anomaly'].sum()
    print(f"\nLocal Anomalies (max LAS > 3): {n_local} ({(n_local/len(df)*100):.2f}%)")
    
    print("\nDistribution of max LAS:")
    print(df['local_anomaly_score_max'].describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99, 0.999]).round(2))

    # ---------------------------------------------------------
    # 4. CROSS-TABULATE GLOBAL VS LOCAL
    # ---------------------------------------------------------
    df['is_global_anomaly'] = df['dbscan_label'] == -1
    print("\n--- GLOBAL vs LOCAL ANOMALIES ---")
    crosstab = pd.crosstab(df['is_global_anomaly'], df['is_local_anomaly'], 
                           rownames=['Global (DBSCAN)'], colnames=['Local (LAS>3)'])
    print(crosstab)
    
    print("\n5 Examples of LOCAL ANOMALY BUT NOT GLOBAL ANOMALY:")
    local_only = df[(df['is_local_anomaly'] == True) & (df['is_global_anomaly'] == False)]
    if len(local_only) > 0:
        examples_local = local_only.sample(min(5, len(local_only)), random_state=42)
        for _, row in examples_local.iterrows():
            print(f"\n[{row['village_name']}, {row['district_name']}] - Zone: {row['soil_zone_name']}")
            print(f"  Max LAS: {row['local_anomaly_score_max']:.2f}")
            
            dist_grp = df[df['district_name'] == row['district_name']]
            if len(dist_grp) < 15:
                dist_grp = df[df['state_name'] == row['state_name']]
            d_mean = dist_grp[features].mean()
            d_std = dist_grp[features].std()
            
            worst_f = None
            max_dev = -1
            for f in features:
                s = d_std[f]
                if s == 0 or np.isnan(s): s = state_stds.loc[row['state_name'], f]
                if s == 0 or np.isnan(s): s = 1e-6
                las = abs(row[f] - d_mean[f]) / s
                if las > max_dev:
                    max_dev = las
                    worst_f = f
                    
            print(f"  Worst feature: {worst_f} = {row[worst_f]:.3f} (District Mean = {d_mean[worst_f]:.3f})")

    # Stability check removed due to OOM on +10% eps
    
    out_path = 'data/processed/shc_anomalies.csv'
    df.to_csv(out_path, index=False)
    print(f"\nSaved anomalies data to {out_path}")

if __name__ == '__main__':
    main()

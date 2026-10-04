import pandas as pd
import numpy as np
import joblib
from sklearn.cluster import KMeans

def get_unscaled_mean(df_wide, labels, features):
    df_wide['temp_label'] = labels
    return df_wide.groupby('temp_label')[features].mean()

def validate_kmeans(X_scaled, df, features, seed):
    kmeans = KMeans(n_clusters=4, random_state=seed, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
    means = get_unscaled_mean(df.copy(), labels, features)
    
    saline_idx = means['ec_pct_non_saline'].idxmin()
    acidic_idx = means['ph_pct_acidic'].idxmax()
    remaining = set([0, 1, 2, 3]) - {saline_idx, acidic_idx}
    deficient_idx = min(remaining, key=lambda idx: means.loc[idx, 'fe_pct_sufficient'])
    remaining = remaining - {deficient_idx}
    fertile_idx = list(remaining)[0]
    
    role_map = {
        saline_idx: 'Saline',
        acidic_idx: 'Acidic/High-N/OC',
        deficient_idx: 'Micronutrient-Deficient',
        fertile_idx: 'Generally Fertile'
    }
    
    dist_counts = pd.Series(labels).value_counts(normalize=True) * 100
    
    print(f"\n--- K-MEANS | SEED {seed} ---")
    means.index = means.index.map(role_map)
    print("Full Dataset Centroids (Unscaled):")
    print(means.round(3).to_string())
    
    print("\nFull Dataset Distribution:")
    for role in ['Saline', 'Acidic/High-N/OC', 'Micronutrient-Deficient', 'Generally Fertile']:
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
    
    for seed in [42, 7, 123]:
        validate_kmeans(X_scaled, df, features, seed)

if __name__ == '__main__':
    main()

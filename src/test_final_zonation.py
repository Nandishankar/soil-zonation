import pandas as pd
import numpy as np
import joblib
from sklearn.cluster import KMeans

def get_unscaled_mean(df_wide, labels, features):
    df_wide['temp_label'] = labels
    return df_wide.groupby('temp_label')[features].mean()

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
    
    print("\n--- 1. SALINE THRESHOLD SENSITIVITY ---")
    for t in [0.5, 0.6, 0.7]:
        mask = df['ec_pct_non_saline'] < t
        count = mask.sum()
        pct = count / len(df) * 100
        print(f"Threshold < {t}: {count} villages ({pct:.2f}%)")
        
    counts = {t: (df['ec_pct_non_saline'] < t).sum() for t in [0.5, 0.6, 0.7]}
    chosen_t = min(counts.keys(), key=lambda t: abs(counts[t]/len(df)*100 - 4.6))
    
    print(f"\nProceeding with chosen threshold < {chosen_t} (closest to earlier 4.5-4.8% figure) for K-Means K=3 stability check...")
    
    saline_mask = df['ec_pct_non_saline'] < chosen_t
    df_res = df[~saline_mask].copy()
    X_res_scaled = X_scaled[~saline_mask]
    
    print("\n--- 2. K=3 RESIDUAL STABILITY CHECK ---")
    for seed in [42, 7, 123]:
        kmeans = KMeans(n_clusters=3, random_state=seed, n_init=10)
        res_labels = kmeans.fit_predict(X_res_scaled)
        
        means = get_unscaled_mean(df_res.copy(), res_labels, features)
        
        deficient_idx = means['fe_pct_sufficient'].idxmin()
        acidic_idx = means['ph_pct_acidic'].idxmax()
        remaining = set([0, 1, 2]) - {deficient_idx, acidic_idx}
        fertile_idx = list(remaining)[0]
        
        role_map = {
            deficient_idx: 'Micronutrient-Deficient',
            acidic_idx: 'Acidic/High-N/OC',
            fertile_idx: 'Generally Fertile'
        }
        
        means.index = means.index.map(role_map)
        dist_counts = pd.Series(res_labels).value_counts(normalize=True) * 100
        
        print(f"\n[SEED {seed}] Residual Distribution:")
        for role in ['Acidic/High-N/OC', 'Micronutrient-Deficient', 'Generally Fertile']:
            idx = [k for k, v in role_map.items() if v == role][0]
            size = dist_counts.get(idx, 0)
            print(f"  {role}: {size:.1f}%")
            
        print(f"[SEED {seed}] Centroids:")
        print(means.round(3).to_string())

if __name__ == '__main__':
    main()

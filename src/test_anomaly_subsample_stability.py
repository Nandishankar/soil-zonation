import pandas as pd
import numpy as np
import joblib
from sklearn.cluster import DBSCAN

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
    
    # Sample 50,000
    df_sample = df.sample(n=50000, random_state=42)
    X_scaled = scaler.transform(df_sample[features])
    
    eps_base = 2.533
    eps_low = eps_base * 0.9
    eps_high = eps_base * 1.1
    min_samples = 26
    
    print(f"\n--- SUBSAMPLED STABILITY CHECK (50,000 villages) ---")
    
    print(f"Running DBSCAN with base eps={eps_base:.3f}...")
    db_base = DBSCAN(eps=eps_base, min_samples=min_samples, algorithm='auto', n_jobs=4)
    labels_base = db_base.fit_predict(X_scaled)
    
    print(f"Running DBSCAN with eps_low={eps_low:.3f} (-10%)...")
    db_low = DBSCAN(eps=eps_low, min_samples=min_samples, algorithm='auto', n_jobs=4)
    labels_low = db_low.fit_predict(X_scaled)
    
    print(f"Running DBSCAN with eps_high={eps_high:.3f} (+10%)...")
    db_high = DBSCAN(eps=eps_high, min_samples=min_samples, algorithm='auto', n_jobs=4)
    labels_high = db_high.fit_predict(X_scaled)
    
    set_base = set(np.where(labels_base == -1)[0])
    set_low = set(np.where(labels_low == -1)[0])
    set_high = set(np.where(labels_high == -1)[0])
    
    print(f"Anomalies at eps_base={eps_base:.3f}: {len(set_base)} ({len(set_base)/50000*100:.2f}%)")
    print(f"Anomalies at eps_low={eps_low:.3f}: {len(set_low)} ({len(set_low)/50000*100:.2f}%)")
    print(f"Anomalies at eps_high={eps_high:.3f}: {len(set_high)} ({len(set_high)/50000*100:.2f}%)")
    
    def jaccard(s1, s2):
        if len(s1.union(s2)) == 0: return 1.0
        return len(s1.intersection(s2)) / len(s1.union(s2))
        
    print(f"\nJaccard overlap (baseline vs -10%): {jaccard(set_base, set_low):.4f}")
    print(f"Jaccard overlap (baseline vs +10%): {jaccard(set_base, set_high):.4f}")
    print(f"Jaccard overlap (-10% vs +10%): {jaccard(set_low, set_high):.4f}")

if __name__ == '__main__':
    main()

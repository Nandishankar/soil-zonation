import pandas as pd
import numpy as np

def get_human_readable_feature(feature_name):
    mapping = {
        'n_index': 'Nitrogen',
        'p_index': 'Phosphorus',
        'k_index': 'Potassium',
        'oc_index': 'Organic Carbon',
        'b_pct_sufficient': 'Boron sufficiency',
        'cu_pct_sufficient': 'Copper sufficiency',
        'fe_pct_sufficient': 'Iron sufficiency',
        'mn_pct_sufficient': 'Manganese sufficiency',
        's_pct_sufficient': 'Sulphur sufficiency',
        'zn_pct_sufficient': 'Zinc sufficiency',
        'ph_pct_acidic': 'Acidity',
        'ph_pct_alkaline': 'Alkalinity',
        'ec_pct_non_saline': 'Non-Salinity'
    }
    return mapping.get(feature_name, feature_name)

def get_human_readable_zscore(feature_name, z_val):
    hr_name = get_human_readable_feature(feature_name)
    direction = "high" if z_val > 0 else "low"
    magnitude = "extremely" if abs(z_val) > 2 else "unusually"
    return f"{magnitude} {direction} {hr_name}"

def main():
    print("Loading anomalies data...")
    df = pd.read_csv('data/processed/shc_anomalies.csv')
    
    features = [
        'n_index', 'p_index', 'k_index', 'oc_index',
        'b_pct_sufficient', 'cu_pct_sufficient', 'fe_pct_sufficient', 'mn_pct_sufficient',
        's_pct_sufficient', 'zn_pct_sufficient',
        'ph_pct_acidic', 'ph_pct_alkaline', 'ec_pct_non_saline'
    ]
    
    print("Computing zone centroids and standard deviations...")
    zone_means = df.groupby('soil_zone_name')[features].mean()
    zone_stds = df.groupby('soil_zone_name')[features].std()
    
    print("Generating explanations...")
    explanations = []
    
    for _, row in df.iterrows():
        zone = row['soil_zone_name']
        
        # 1. ZONE EXPLANATION
        z_scores = []
        for f in features:
            s = zone_stds.loc[zone, f]
            if s == 0 or np.isnan(s): s = 1e-6
            z = (row[f] - zone_means.loc[zone, f]) / s
            z_scores.append((f, z))
            
        z_scores.sort(key=lambda x: abs(x[1]), reverse=True)
        top_3_z = z_scores[:3]
        
        reasons = [get_human_readable_zscore(f, z) for f, z in top_3_z]
        zone_text = f"This village is typical of the {zone} zone, standing out even within that zone for {reasons[0]}, {reasons[1]}, and {reasons[2]}."
        
        # 2. ANOMALY EXPLANATION
        anomaly_text = ""
        if row['dbscan_label'] == -1:
            anomaly_text = f" It is flagged as a global anomaly due to: {row['anomaly_explanation']}."
            
        # 3. LOCAL CONTEXT
        local_text = ""
        if row['is_local_anomaly']:
            local_text = f" However, it shows significant deviation (max local score {row['local_anomaly_score_max']:.1f}) relative to its immediate neighbors in {row['district_name']}, and may warrant individual soil testing."
            
        full_text = zone_text + anomaly_text + local_text
        explanations.append(full_text)
        
    df['explanation_text'] = explanations
    
    out_path = 'data/processed/shc_explained.csv'
    df.to_csv(out_path, index=False)
    print(f"Saved explained data to {out_path}")
    
    print("\n--- Example Explanations ---")
    examples = pd.concat([
        df[(df['is_local_anomaly'] == True) & (df['dbscan_label'] != -1)].head(3),
        df[(df['dbscan_label'] == -1)].head(3),
        df[(df['is_local_anomaly'] == False) & (df['dbscan_label'] != -1)].head(4)
    ])
    for _, row in examples.iterrows():
        print(f"\n[{row['village_name']}, {row['district_name']}] - Zone: {row['soil_zone_name']}")
        print(f"Explanation: {row['explanation_text']}")

if __name__ == '__main__':
    main()

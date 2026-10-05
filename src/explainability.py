import pandas as pd
import numpy as np

def get_human_readable_feature(feature_name):
    mapping = {
        'n_index': 'Nitrogen level',
        'p_index': 'Phosphorus level',
        'k_index': 'Potassium level',
        'oc_index': 'Organic Carbon level',
        'b_pct_sufficient': 'Boron sufficiency',
        'cu_pct_sufficient': 'Copper sufficiency',
        'fe_pct_sufficient': 'Iron sufficiency',
        'mn_pct_sufficient': 'Manganese sufficiency',
        's_pct_sufficient': 'Sulphur sufficiency',
        'zn_pct_sufficient': 'Zinc sufficiency',
        'ph_pct_acidic': 'Acidity',
        'ph_pct_alkaline': 'Alkalinity',
        'ec_pct_non_saline': 'Non-Saline EC'
    }
    return mapping.get(feature_name, feature_name)

def get_hr_zscore(feature_name, z_val):
    hr_name = get_human_readable_feature(feature_name)
    magnitude = "extremely" if abs(z_val) > 2.5 else "unusually"
    if abs(z_val) < 1.0: magnitude = "slightly"
    
    # Directionality logic based on feature type
    if feature_name in ['b_pct_sufficient', 'cu_pct_sufficient', 'fe_pct_sufficient', 'mn_pct_sufficient', 's_pct_sufficient', 'zn_pct_sufficient']:
        if z_val > 0: return f"{magnitude} high {hr_name} (a positive outlier, not a concern)"
        else: return f"{magnitude} low {hr_name} (a deficiency concern)"
    elif feature_name == 'ec_pct_non_saline':
        if z_val > 0: return f"{magnitude} high {hr_name} (a positive outlier, indicating excellent non-saline soil)"
        else: return f"{magnitude} low {hr_name} (a salinity concern)"
    elif feature_name in ['ph_pct_acidic', 'ph_pct_alkaline']:
        if z_val > 0: return f"{magnitude} high {hr_name} (a pH imbalance concern)"
        else: return f"{magnitude} low {hr_name} (a positive outlier, indicating neutral soil)"
    elif feature_name in ['n_index', 'p_index', 'k_index', 'oc_index']:
        if z_val < 0: return f"{magnitude} low {hr_name} (a deficiency concern)"
        else: return f"{magnitude} high {hr_name} (an excess/toxicity concern)"
    
    direction = "higher" if z_val > 0 else "lower"
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
    
    print("Computing means and stds...")
    zone_means = df.groupby('soil_zone_name')[features].mean()
    zone_stds = df.groupby('soil_zone_name')[features].std()
    
    global_means = df[features].mean()
    global_stds = df[features].std()
    
    dist_means = df.groupby('district_name')[features].mean()
    
    print("Generating explanations...")
    explanations = []
    
    for idx, row in df.iterrows():
        zone = row['soil_zone_name']
        
        # 1. ZONE CONTEXT
        z_scores_zone = []
        for f in features:
            s = zone_stds.loc[zone, f]
            if s == 0 or pd.isna(s): s = 1e-6
            z = (row[f] - zone_means.loc[zone, f]) / s
            z_scores_zone.append((f, z))
        z_scores_zone.sort(key=lambda x: abs(x[1]), reverse=True)
        top_2 = z_scores_zone[:2]
        
        zone_str = f"{zone} village, "
        if abs(top_2[0][1]) > 1.5:
            zone_str += f"standing out within its zone for {get_hr_zscore(top_2[0][0], top_2[0][1])} and {get_hr_zscore(top_2[1][0], top_2[1][1])}."
        else:
            zone_str += f"typical of its zone with no extreme internal deviations."
            
        # 2. GLOBAL ANOMALY STATUS
        is_global = (row['dbscan_label'] == -1)
        global_str = ""
        if is_global:
            z_scores_global = []
            for f in features:
                s = global_stds[f]
                if s == 0 or pd.isna(s): s = 1e-6
                z = (row[f] - global_means[f]) / s
                z_scores_global.append((f, z))
            z_scores_global.sort(key=lambda x: abs(x[1]), reverse=True)
            top_3_g = z_scores_global[:3]
            reasons = [get_hr_zscore(f, z) for f, z in top_3_g]
            global_str = f" Nationally, it is flagged as a severe global anomaly characterized by {reasons[0]}, {reasons[1]}, and {reasons[2]} compared to the national average."
            
        # 3. LOCAL ANOMALY STATUS
        is_local = row['is_local_anomaly']
        local_str = ""
        if is_local:
            d_name = row['district_name']
            if d_name in dist_means.index:
                d_mean = dist_means.loc[d_name]
            else:
                d_mean = global_means # Fallback
                
            worst_f = None
            max_dev = -1
            
            for f in features:
                s = global_stds[f]
                if s == 0: s = 1e-6
                dev = abs(row[f] - d_mean[f]) / s
                if dev > max_dev:
                    max_dev = dev
                    worst_f = f
                    
            hr_f = get_human_readable_feature(worst_f)
            val = row[worst_f]
            dist_val = d_mean[worst_f]
            z_local = (val - dist_val) / global_stds[worst_f]
            
            if not is_global:
                local_str = f" Despite appearing typical nationally, this village stands out locally for {get_hr_zscore(worst_f, z_local)} compared to its district average (district avg: {dist_val:.2f}, this village: {val:.2f})."
            else:
                local_str = f" Furthermore, it stands out locally for {get_hr_zscore(worst_f, z_local)} compared to its district average (district avg: {dist_val:.2f}, this village: {val:.2f})."
                
        full_text = zone_str + global_str + local_str
        explanations.append(full_text)
        
    df['explanation_text'] = explanations
    
    out_path = 'data/processed/shc_explained.csv'
    df.to_csv(out_path, index=False)
    print(f"Saved explained data to {out_path}")
    
    print("\n--- 10 Example Cards ---")
    
    print("\n[2 Plain Fertile-zone villages (No flags)]")
    samp = df[(df['soil_zone_name'] == 'Generally Fertile') & (~df['is_local_anomaly']) & (df['dbscan_label'] != -1)].sample(n=2, random_state=42)
    for _, r in samp.iterrows(): print(f"[{r['village_name']}, {r['district_name']}]: {r['explanation_text']}")
    
    print("\n[2 Deficient-zone with global anomaly]")
    samp = df[(df['soil_zone_name'] == 'Micronutrient-Deficient') & (df['dbscan_label'] == -1)].sample(n=2, random_state=42)
    for _, r in samp.iterrows(): print(f"[{r['village_name']}, {r['district_name']}]: {r['explanation_text']}")
        
    print("\n[2 Saline-zone with global anomaly (Compound catastrophe)]")
    samp = df[(df['soil_zone_name'] == 'Saline') & (df['dbscan_label'] == -1)].sample(n=2, random_state=42)
    for _, r in samp.iterrows(): print(f"[{r['village_name']}, {r['district_name']}]: {r['explanation_text']}")
        
    print("\n[2 Local-only anomalies (Normal nationally, unusual for district)]")
    samp = df[(df['is_local_anomaly']) & (df['dbscan_label'] != -1)].sample(n=2, random_state=42)
    for _, r in samp.iterrows(): print(f"[{r['village_name']}, {r['district_name']}]: {r['explanation_text']}")
        
    print("\n[2 Global + Local anomalies]")
    samp = df[(df['is_local_anomaly']) & (df['dbscan_label'] == -1)].sample(n=2, random_state=42)
    for _, r in samp.iterrows(): print(f"[{r['village_name']}, {r['district_name']}]: {r['explanation_text']}")

    print("\n[Verification of Specific Villages]")
    samp = df[df['village_name'].isin(['Govinda Nagar', 'Big Lapati'])]
    for _, r in samp.iterrows(): print(f"[{r['village_name']}, {r['district_name']}]: {r['explanation_text']}")

if __name__ == '__main__':
    main()

import pandas as pd
import numpy as np
import joblib
import os
import time
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

def print_step(msg):
    print(f"\n[{time.strftime('%H:%M:%S')}] {msg}")

def main():
    os.makedirs('data/processed', exist_ok=True)
    
    # ---------------------------------------------------------
    # 1. LOAD & FILTER
    # ---------------------------------------------------------
    print_step("STAGE 1: LOAD & FILTER")
    df_raw = pd.read_csv('data/raw/soil-nutrient-analysis.csv')
    print(f"Raw data shape: {df_raw.shape}")
    
    df = df_raw[df_raw['year'] == '2024-25'].copy()
    print(f"Filtered to 2024-25 shape: {df.shape}")
    
    # ---------------------------------------------------------
    # 2. PIVOT LONG -> WIDE
    # ---------------------------------------------------------
    print_step("STAGE 2: PIVOT TO WIDE FORMAT")
    keys = ['state_name', 'district_name', 'block_name', 'village_name', 'year']
    
    df['nutrient_name'] = df['nutrient_name'].str.strip()
    df['nutrient_level'] = df['nutrient_level'].str.strip()
    
    pivot_df = df.groupby(keys + ['nutrient_name', 'nutrient_level'])['value'].sum().unstack(level=['nutrient_name', 'nutrient_level']).fillna(0)
    
    wide = pd.DataFrame(index=pivot_df.index)
    
    expected_ordinal = {'Nitrogen': 'n_index', 'Phosphorus': 'p_index', 'Potassium': 'k_index', 'Organic Carbon': 'oc_index'}
    expected_binary = {'Boron': 'b_pct_sufficient', 'Copper': 'cu_pct_sufficient', 'Iron': 'fe_pct_sufficient', 'Manganese': 'mn_pct_sufficient', 'Sulphur': 's_pct_sufficient', 'Zinc': 'zn_pct_sufficient'}
    
    for nut, feature_name in expected_ordinal.items():
        if nut in pivot_df.columns.levels[0]:
            sub = pivot_df[nut]
            low = sub.get('Low', 0)
            med = sub.get('Medium', 0)
            high = sub.get('High', 0)
            tot = low + med + high
            wide[feature_name] = np.where(tot > 0, (low*1 + med*2 + high*3) / tot, np.nan)
            wide[f'n_samples_{feature_name.split("_")[0]}'] = tot
        else:
            wide[feature_name] = np.nan
            wide[f'n_samples_{feature_name.split("_")[0]}'] = 0
            
    for nut, feature_name in expected_binary.items():
        if nut in pivot_df.columns.levels[0]:
            sub = pivot_df[nut]
            suff = sub.get('Sufficient', 0)
            defi = sub.get('Deficient', 0)
            tot = suff + defi
            wide[feature_name] = np.where(tot > 0, suff / tot, np.nan)
            wide[f'n_samples_{feature_name.split("_")[0]}'] = tot
        else:
            wide[feature_name] = np.nan
            wide[f'n_samples_{feature_name.split("_")[0]}'] = 0

    if 'Soil Ph' in pivot_df.columns.levels[0]:
        sub = pivot_df['Soil Ph']
        acidic = sub.get('Acidic', 0)
        neutral = sub.get('Neutral', 0)
        alkaline = sub.get('Alkaline', 0)
        tot = acidic + neutral + alkaline
        wide['ph_pct_acidic'] = np.where(tot > 0, acidic / tot, np.nan)
        wide['ph_pct_neutral'] = np.where(tot > 0, neutral / tot, np.nan)
        wide['ph_pct_alkaline'] = np.where(tot > 0, alkaline / tot, np.nan)
        wide['n_samples_ph'] = tot
    else:
        wide['ph_pct_acidic'] = np.nan
        wide['ph_pct_neutral'] = np.nan
        wide['ph_pct_alkaline'] = np.nan
        wide['n_samples_ph'] = 0

    if 'Electrical Conductivity' in pivot_df.columns.levels[0]:
        sub = pivot_df['Electrical Conductivity']
        non_saline = sub.get('Non Saline', 0)
        saline = sub.get('Saline', 0)
        tot = non_saline + saline
        wide['ec_pct_non_saline'] = np.where(tot > 0, non_saline / tot, np.nan)
        wide['n_samples_ec'] = tot
    else:
        wide['ec_pct_non_saline'] = np.nan
        wide['n_samples_ec'] = 0

    wide = wide.reset_index()
    print(f"Wide format shape: {wide.shape}")
    
    # ---------------------------------------------------------
    # 3. RELIABILITY FILTERING
    # ---------------------------------------------------------
    print_step("STAGE 3: RELIABILITY FILTERING")
    
    features = [
        'n_index', 'p_index', 'k_index', 'oc_index',
        'b_pct_sufficient', 'cu_pct_sufficient', 'fe_pct_sufficient', 'mn_pct_sufficient',
        's_pct_sufficient', 'zn_pct_sufficient',
        'ph_pct_acidic', 'ph_pct_alkaline', 'ec_pct_non_saline'
    ]
    all_features = features + ['ph_pct_neutral']
    
    for f in all_features:
        prefix = f.split('_')[0]
        if f == 'ec_pct_non_saline': prefix = 'ec'
        count_col = f'n_samples_{prefix}'
        mask = wide[count_col] < 5
        wide.loc[mask, f] = np.nan
        
    nutrient_count_cols = [
        'n_samples_n', 'n_samples_p', 'n_samples_k', 'n_samples_oc',
        'n_samples_b', 'n_samples_cu', 'n_samples_fe', 'n_samples_mn',
        'n_samples_s', 'n_samples_zn', 'n_samples_ph', 'n_samples_ec'
    ]
    missing_nutrients = (wide[nutrient_count_cols] < 5).sum(axis=1)
    drop_mask = missing_nutrients >= 7
    wide_filtered = wide[~drop_mask].copy()
    wide_filtered = wide_filtered.sort_values(['state_name', 'district_name', 'block_name', 'village_name', 'year']).reset_index(drop=True)
    
    print(f"Dropped {drop_mask.sum()} rows with >=7 missing nutrients.")
    print(f"Remaining shape: {wide_filtered.shape}")
    
    # ---------------------------------------------------------
    # 4. IMPUTATION
    # ---------------------------------------------------------
    print_step("STAGE 4: IMPUTATION")
    
    flags = pd.DataFrame(0, index=wide_filtered.index, columns=all_features)
    
    dist_medians = wide_filtered.groupby('district_name')[all_features].median()
    state_medians = wide_filtered.groupby('state_name')[all_features].median()
    global_medians = wide_filtered[all_features].median()
    
    for f in all_features:
        missing = wide_filtered[f].isna()
        flags.loc[missing, f] = 1
        
        n_missing = missing.sum()
        if n_missing > 0:
            wide_filtered[f] = wide_filtered.apply(
                lambda row: dist_medians.loc[row['district_name'], f] if pd.isna(row[f]) else row[f], axis=1
            )
            missing2 = wide_filtered[f].isna()
            if missing2.sum() > 0:
                wide_filtered[f] = wide_filtered.apply(
                    lambda row: state_medians.loc[row['state_name'], f] if pd.isna(row[f]) else row[f], axis=1
                )
                missing3 = wide_filtered[f].isna()
                if missing3.sum() > 0:
                    wide_filtered.loc[missing3, f] = global_medians[f]
                    
    flags_path = 'data/processed/shc_wide_imputed_flags.csv'
    flags.to_csv(flags_path, index=False)
    
    # ---------------------------------------------------------
    # 5. FEATURE SCALING
    # ---------------------------------------------------------
    print_step("STAGE 5: FEATURE SCALING")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(wide_filtered[features])
    joblib.dump(scaler, 'data/processed/scaler.joblib')
    print("Saved scaler to data/processed/scaler.joblib")
    
    # ---------------------------------------------------------
    # 6. FINAL ZONATION (Domain-rule + Clustering Hybrid)
    # ---------------------------------------------------------
    print_step("STAGE 6: FINAL ZONATION (Saline rule + K-Means K=3)")
    
    wide_filtered['soil_zone_name'] = None
    wide_filtered['zone_method'] = None
    
    # a. Saline Zone (Domain rule)
    saline_mask = wide_filtered['ec_pct_non_saline'] < 0.7
    wide_filtered.loc[saline_mask, 'soil_zone_name'] = 'Saline'
    wide_filtered.loc[saline_mask, 'zone_method'] = 'domain_rule'
    print(f"Domain-rule Saline villages: {saline_mask.sum()}")
    
    # b. Residual Clustering
    residual_mask = ~saline_mask
    X_res_scaled = X_scaled[residual_mask]
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    res_labels = kmeans.fit_predict(X_res_scaled)
    
    df_res = wide_filtered[residual_mask].copy()
    df_res['temp_label'] = res_labels
    means = df_res.groupby('temp_label')[features].mean()
    
    deficient_idx = means['fe_pct_sufficient'].idxmin()
    acidic_idx = means['ph_pct_acidic'].idxmax()
    remaining = set([0, 1, 2]) - {deficient_idx, acidic_idx}
    fertile_idx = list(remaining)[0]
    
    zone_names = {
        deficient_idx: 'Micronutrient-Deficient',
        acidic_idx: 'Acidic/High-N/OC',
        fertile_idx: 'Generally Fertile'
    }
    
    res_zone_names = pd.Series(res_labels).map(zone_names).values
    wide_filtered.loc[residual_mask, 'soil_zone_name'] = res_zone_names
    wide_filtered.loc[residual_mask, 'zone_method'] = 'clustering'
    
    # ---------------------------------------------------------
    # 7. FINAL SUMMARY
    # ---------------------------------------------------------
    print_step("STAGE 7: FINAL SUMMARY")
    out_cols = ['state_name', 'district_name', 'block_name', 'village_name', 'year'] + \
               all_features + \
               ['soil_zone_name', 'zone_method']
               
    df_zoned = wide_filtered[out_cols]
    df_zoned.to_csv('data/processed/shc_zoned.csv', index=False)
    
    print(f"Final shc_zoned.csv shape: {df_zoned.shape}")
    print("\nZone Distribution:")
    counts = df_zoned['soil_zone_name'].value_counts()
    for name, count in counts.items():
        print(f"  {name}: {count} ({count/len(df_zoned)*100:.1f}%)")
        
    print("\nOutput Files Created:")
    for f in ['data/processed/shc_wide_imputed_flags.csv', 'data/processed/scaler.joblib', 'data/processed/shc_zoned.csv']:
        print(f" - {f}: {'Exists' if os.path.exists(f) else 'MISSING!'}")

if __name__ == '__main__':
    main()

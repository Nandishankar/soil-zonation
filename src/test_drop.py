import pandas as pd
import numpy as np

def main():
    df_raw = pd.read_csv('data/raw/soil-nutrient-analysis.csv')
    df = df_raw[df_raw['year'] == '2024-25'].copy()
    
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
            tot = sub.get('Low', 0) + sub.get('Medium', 0) + sub.get('High', 0)
            wide[f'n_samples_{feature_name.split("_")[0]}'] = tot
        else: wide[f'n_samples_{feature_name.split("_")[0]}'] = 0
            
    for nut, feature_name in expected_binary.items():
        if nut in pivot_df.columns.levels[0]:
            sub = pivot_df[nut]
            tot = sub.get('Sufficient', 0) + sub.get('Deficient', 0)
            wide[f'n_samples_{feature_name.split("_")[0]}'] = tot
        else: wide[f'n_samples_{feature_name.split("_")[0]}'] = 0

    if 'Soil Ph' in pivot_df.columns.levels[0]:
        sub = pivot_df['Soil Ph']
        wide['n_samples_ph'] = sub.get('Acidic', 0) + sub.get('Neutral', 0) + sub.get('Alkaline', 0)
    else: wide['n_samples_ph'] = 0

    if 'Electrical Conductivity' in pivot_df.columns.levels[0]:
        sub = pivot_df['Electrical Conductivity']
        wide['n_samples_ec'] = sub.get('Non Saline', 0) + sub.get('Saline', 0)
    else: wide['n_samples_ec'] = 0

    nutrient_count_cols = [
        'n_samples_n', 'n_samples_p', 'n_samples_k', 'n_samples_oc',
        'n_samples_b', 'n_samples_cu', 'n_samples_fe', 'n_samples_mn',
        'n_samples_s', 'n_samples_zn', 'n_samples_ph', 'n_samples_ec'
    ]
    
    missing_nutrients = (wide[nutrient_count_cols] < 5).sum(axis=1)
    drop_mask = missing_nutrients >= 7
    wide_filtered = wide[~drop_mask].copy()
    
    print(f"Original missing_nutrients >= 7 logic:")
    print(f"Dropped: {drop_mask.sum()}")
    print(f"Remaining: {len(wide_filtered)}")
    
if __name__ == '__main__':
    main()

import pandas as pd
import re

def get_base_recommendation(row):
    """
    Generate base recommendation using standard agronomic practices based on zone.
    """
    zone = row['soil_zone_name']
    
    if zone == 'Saline':
        return "Gypsum/soil amendment application, improved drainage, salt-tolerant crop varieties, periodic leaching with good-quality water."
    elif zone == 'Acidic/High-N/OC':
        return "Lime application to correct pH, careful nitrogen management to avoid over-application given already-high OC/N."
    elif zone == 'Micronutrient-Deficient':
        micros = {
            'Boron': row['b_pct_sufficient'],
            'Copper': row['cu_pct_sufficient'],
            'Iron': row['fe_pct_sufficient'],
            'Manganese': row['mn_pct_sufficient'],
            'Sulphur': row['s_pct_sufficient'],
            'Zinc': row['zn_pct_sufficient']
        }
        imputed_flags = {
            'Boron': row.get('b_pct_sufficient_is_imputed', False),
            'Copper': row.get('cu_pct_sufficient_is_imputed', False),
            'Iron': row.get('fe_pct_sufficient_is_imputed', False),
            'Manganese': row.get('mn_pct_sufficient_is_imputed', False),
            'Sulphur': row.get('s_pct_sufficient_is_imputed', False),
            'Zinc': row.get('zn_pct_sufficient_is_imputed', False)
        }
        
        # Rank by lowest sufficiency and alphabetical tie-breaker
        ranked = sorted(micros.items(), key=lambda x: (x[1], x[0]))
        bottom_2 = [ranked[0][0], ranked[1][0]]
        
        # Relative deviations from explanation text
        exp_text = str(row['explanation_text'])
        relative_needs = []
        for m in micros.keys():
            if f"{m} sufficiency (a deficiency concern)" in exp_text:
                if m not in bottom_2:
                    relative_needs.append(m)
        
        def fmt(m):
            return f"{m} (estimated from district median)" if imputed_flags.get(m) else m
            
        bottom_2_formatted = [fmt(m) for m in bottom_2]
        
        rec = f"Targeted micronutrient supplementation. Lowest sufficiency: {', '.join(bottom_2_formatted)}."
        if relative_needs:
            rel_formatted = [fmt(m) for m in relative_needs]
            rec += f" Also unusually low vs district/national: {', '.join(rel_formatted)}."
            
        return rec
    else:
        return "Maintenance-level balanced fertilization, routine re-testing on normal cycle."

def generate_recommendation(row):
    base_rec = get_base_recommendation(row)
    
    is_global = row['dbscan_label'] == -1
    is_local = row['is_local_anomaly']
    exp_text = str(row['explanation_text'])
    
    negative_concerns = ["deficiency concern", "imbalance concern", "salinity concern", "toxicity concern"]
    
    has_global_concerns = False
    if is_global and "Nationally," in exp_text:
        global_part = exp_text.split("Nationally,")[1]
        has_global_concerns = any(c in global_part for c in negative_concerns)
        
    has_local_concerns = False
    if is_local and "locally" in exp_text:
        local_part = exp_text.split("locally")[1]
        has_local_concerns = any(c in local_part for c in negative_concerns)
    
    addendum = ""
    if is_global and has_global_concerns:
        match = re.search(r"characterized by (.*?) compared to the national average", exp_text)
        factors = match.group(1) if match else "multiple extreme deviations"
        addendum = f" PRIORITY REVIEW: This village combines severe compounding factors ({factors}) — standard zone-level amendment alone is unlikely to be sufficient."
    elif is_local and not is_global and has_local_concerns:
        addendum = " PRIORITY REVIEW: Recommend individual soil testing — this village deviates significantly from its immediate district despite appearing typical nationally."
        
    return base_rec + addendum

def main():
    print("Loading explained data...")
    df = pd.read_csv('data/processed/shc_explained.csv')
    flags = pd.read_csv('data/processed/shc_wide_imputed_flags.csv')
    
    # Concatenate flags row-wise
    df = pd.concat([df, flags.add_suffix('_is_imputed')], axis=1)
    
    print("Loading raw data for sample counts...")
    raw = pd.read_csv('data/raw/soil-nutrient-analysis.csv', usecols=['state_name', 'district_name', 'village_name'])
    counts = raw.groupby(['state_name', 'district_name', 'village_name']).size().reset_index(name='n_samples')
    
    df = df.merge(counts, on=['state_name', 'district_name', 'village_name'], how='left')
    
    print("Generating recommendations...")
    df['recommendation'] = df.apply(generate_recommendation, axis=1)
    
    df.to_csv('data/processed/shc_final_output.csv', index=False)
    print("Saved final analytical deliverable to data/processed/shc_final_output.csv")
    
    # Assert across all villages that every nutrient in the explanation's deficiency drivers appears somewhere in the recommendation
    print("Validating Deficient-zone recommendations...")
    def_df = df[df['soil_zone_name'] == 'Micronutrient-Deficient']
    mismatches = 0
    imputed_count = 0
    
    for _, r in def_df.iterrows():
        exp_text = str(r['explanation_text'])
        rec_text = str(r['recommendation'])
        
        if "(estimated from district median)" in rec_text:
            imputed_count += 1
            
        micros = ['Boron', 'Copper', 'Iron', 'Manganese', 'Sulphur', 'Zinc']
        for m in micros:
            if f"{m} sufficiency (a deficiency concern)" in exp_text:
                if m not in rec_text:
                    mismatches += 1
                    
    print(f"Total Deficient-zone recommendations naming an imputed nutrient: {imputed_count}")
    print(f"Mismatch count (explanation deficiency not in recommendation): {mismatches}")
    
    print("\n--- Chokonggre Card ---")
    chok = df[df['village_name'] == 'Chokonggre'].iloc[0]
    print(f"[{chok['village_name']}, {chok['district_name']}, n={chok['n_samples']}]:")
    print(f"  EXP: {chok['explanation_text']}")
    print(f"  REC: {chok['recommendation']}")

if __name__ == '__main__':
    main()

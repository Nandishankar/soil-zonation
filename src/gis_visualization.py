import pandas as pd
import geopandas as gpd
import urllib.request
import os
import difflib

def main():
    print("1. ACQUIRING BOUNDARY DATA")
    geojson_path = 'data/raw/india_district.geojson'
    
    if not os.path.exists(geojson_path):
        print("Downloading India district GeoJSON...")
        url = "https://raw.githubusercontent.com/geohacker/india/master/district/india_district.geojson"
        urllib.request.urlretrieve(url, geojson_path)
    
    gdf = gpd.read_file(geojson_path)
    print(f"Loaded {len(gdf)} districts from boundary file.")
    print(f"Columns: {gdf.columns.tolist()}")
    
    # Typically 'NAME_2' is the district name in standard GADM-derived geojsons. 
    # Or 'dist_name', 'district', 'dtname'. Let's inspect a row.
    print(gdf.head(1))
    
    # Let's assume 'NAME_2' or 'dtname' based on typical files.
    if 'NAME_2' in gdf.columns:
        gdf_dist_col = 'NAME_2'
    elif 'dtname' in gdf.columns:
        gdf_dist_col = 'dtname'
    elif 'district' in gdf.columns:
        gdf_dist_col = 'district'
    elif 'DISTRICT' in gdf.columns:
        gdf_dist_col = 'DISTRICT'
    else:
        print("COULD NOT FIND DISTRICT COLUMN")
        return
        
    print(f"Using boundary district column: {gdf_dist_col}")
    
    # Capitalize for matching
    gdf['map_district_upper'] = gdf[gdf_dist_col].str.upper().str.strip()
    
    print("\n2. AGGREGATING SHC DATA TO DISTRICT LEVEL")
    shc_df = pd.read_csv('data/processed/shc_explained.csv')
    
    district_summary = []
    
    # We group by state and district
    grouped = shc_df.groupby(['state_name', 'district_name'])
    
    for (state, dist), group in grouped:
        total_villages = len(group)
        
        # Zone distribution
        zone_counts = group['soil_zone_name'].value_counts()
        dominant_zone = zone_counts.index[0] if not zone_counts.empty else None
        
        # Anomalies
        global_pct = (group['dbscan_label'] == -1).mean() * 100
        local_pct = group['is_local_anomaly'].mean() * 100
        avg_local_score = group['local_anomaly_score_max'].mean()
        
        district_summary.append({
            'state_name': state,
            'district_name': dist,
            'village_count': total_villages,
            'dominant_zone': dominant_zone,
            'global_anomaly_pct': global_pct,
            'local_anomaly_pct': local_pct,
            'avg_local_anomaly_score_mean': avg_local_score
        })
        
    dist_df = pd.DataFrame(district_summary)
    dist_df.to_csv('data/processed/district_summary.csv', index=False)
    print(f"Aggregated {len(dist_df)} districts. Saved to data/processed/district_summary.csv")
    
    print("\n3. NAME-MATCHING")
    
    dist_df['shc_district_upper'] = dist_df['district_name'].str.upper().str.strip()
    
    manual_aliases = {
        'PURBA BARDHAMAN': 'BARDDHAMAN',
        'PASCHIM MEDINIPUR': 'WEST MIDNAPORE',
        'HOOGHLY': 'HUGLI',
        'PURBA MEDINIPUR': 'EAST MIDNAPORE',
        'KAIMUR (BHABUA)': 'BHABUA',
        'CHHOTAUDEPUR': 'VADODARA',
        'KHARGONE (WEST NIMAR)': 'WEST NIMAR',
        'KENDUJHAR': 'KEONJHAR',
        'MAHISAGAR': 'PANCH MAHALS',
        'TIRUPATI': 'CHITTOOR',
        'ARVALLI': 'SABAR KANTHA',
        'JHARGRAM': 'WEST MIDNAPORE',
        'NARMADAPURAM': 'HOSHANGABAD',
        'ALLURI SITHARAMA RAJU': 'VISHAKHAPATNAM',
        'SRI POTTI SRIRAMULU NELLORE': 'NELLORE',
        'NARSIMHAPUR': 'NARSINGHPUR',
        'PARVATHIPURAM MANYAM': 'VIZIANAGARAM',
        'KHANDWA (EAST NIMAR)': 'EAST NIMAR',
        'AHILYANAGAR': 'AHMEDNAGAR',
        'HOWRAH': 'HAORA',
        'Y.S.R. KADAPA': 'CUDDAPAH',
        'ELURU': 'WEST GODAVARI',
        'TAPI': 'SURAT',
        'COOCH BEHAR': 'KOCHBIHAR',
        'EAST SINGHBUM': 'PURBA SINGHBHUM',
        'ANAKAPALLI': 'VISHAKHAPATNAM',
        'ANANTHAPURAMU': 'ANANTAPUR',
        'NANDYAL': 'KURNOOL',
        'RAIGAD': 'RAIGARH',
        'SRI SATHYA SAI': 'ANANTAPUR'
    }
    
    dist_df['shc_district_upper'] = dist_df['shc_district_upper'].replace(manual_aliases)
    
    map_districts = set(gdf['map_district_upper'].dropna())
    
    # Exact match
    dist_df['matched_exact'] = dist_df['shc_district_upper'].isin(map_districts)
    exact_match_villages = dist_df[dist_df['matched_exact']]['village_count'].sum()
    total_villages = dist_df['village_count'].sum()
    
    print(f"Exact match rate (by village count): {exact_match_villages/total_villages*100:.2f}%")
    
    unmatched = dist_df[~dist_df['matched_exact']].copy()
    
    if len(unmatched) > 0:
        print("\nAttempting Fuzzy Matching...")
        def fuzzy_match(name, choices):
            matches = difflib.get_close_matches(name, choices, n=1, cutoff=0.85)
            return matches[0] if matches else None
            
        unmatched['fuzzy_matched'] = unmatched['shc_district_upper'].apply(lambda x: fuzzy_match(x, map_districts))
        fuzzy_success = unmatched[unmatched['fuzzy_matched'].notnull()]
        
        fuzzy_match_villages = fuzzy_success['village_count'].sum()
        print(f"Fuzzy match rate (by village count): {fuzzy_match_villages/total_villages*100:.2f}%")
        
        dist_df.loc[fuzzy_success.index, 'shc_district_upper'] = fuzzy_success['fuzzy_matched']
        dist_df['matched_fuzzy'] = dist_df['shc_district_upper'].isin(map_districts)
        
        overall_match_villages = dist_df[dist_df['matched_fuzzy'] | dist_df['matched_exact']]['village_count'].sum()
        print(f"Overall match rate (exact + fuzzy) by village count: {overall_match_villages/total_villages*100:.2f}%")
        
        still_unmatched = dist_df[~(dist_df['matched_fuzzy'] | dist_df['matched_exact'])]
        if len(still_unmatched) > 0:
            print(f"\nTop 30 Unmatched Districts (Total {len(still_unmatched)} unmatched):")
            top_unmatched = still_unmatched.sort_values('village_count', ascending=False).head(30)
            for _, r in top_unmatched.iterrows():
                print(f"- {r['state_name']} | {r['district_name']} ({r['village_count']} villages)")
                
    print("\n4. BUILDING MAPS")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import folium
    
    os.makedirs('outputs/figures', exist_ok=True)
    
    gdf_mapped = gdf.merge(dist_df, left_on='map_district_upper', right_on='shc_district_upper', how='left')
    
    print("Generating Zone Map...")
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    
    zone_colors = {
        'Generally Fertile': '#2ca02c',
        'Acidic/High-N/OC': '#1f77b4',
        'Micronutrient-Deficient': '#ff7f0e',
        'Saline': '#d62728'
    }
    
    # We must plot manually because Geopandas color mapping can be tricky with missing categories
    for zone, color in zone_colors.items():
        gdf_mapped[gdf_mapped['dominant_zone'] == zone].plot(ax=ax, color=color, label=zone)
    gdf_mapped[gdf_mapped['dominant_zone'].isnull()].plot(ax=ax, color='#eeeeee')
    
    plt.title('Dominant Soil Zone by District')
    plt.axis('off')
    
    import matplotlib.patches as mpatches
    handles = [mpatches.Patch(color=c, label=z) for z, c in zone_colors.items()]
    handles.append(mpatches.Patch(color='#eeeeee', label='No Data'))
    ax.legend(handles=handles, loc='lower right')
    
    plt.savefig('outputs/figures/zone_map.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Generating Anomaly Density Map...")
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    gdf_mapped.plot(column='global_anomaly_pct', ax=ax, legend=True, 
                    cmap='Reds', missing_kwds={'color': '#eeeeee'},
                    legend_kwds={'label': "Global Anomaly %", 'orientation': "vertical"})
    plt.title('Global Anomaly Density by District (%)')
    plt.axis('off')
    plt.savefig('outputs/figures/anomaly_density_map.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Generating Interactive Folium Map...")
    # Simplify geometry for fast folium rendering
    gdf_mapped['geometry'] = gdf_mapped['geometry'].simplify(tolerance=0.01)
    
    gdf_mapped['dominant_zone'] = gdf_mapped['dominant_zone'].fillna('No Data')
    gdf_mapped['global_anomaly_pct'] = gdf_mapped['global_anomaly_pct'].fillna(0).round(2)
    gdf_mapped['local_anomaly_pct'] = gdf_mapped['local_anomaly_pct'].fillna(0).round(2)
    
    m = gdf_mapped.explore(
        column='dominant_zone',
        cmap=[zone_colors.get(z, '#eeeeee') for z in gdf_mapped['dominant_zone'].unique()],
        tooltip=['state_name', 'district_name', 'dominant_zone', 'global_anomaly_pct', 'local_anomaly_pct'],
        name='Zones'
    )
    m.save('outputs/figures/india_soil_map.html')
    print("Maps saved to outputs/figures/")

if __name__ == '__main__':
    main()

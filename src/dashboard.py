import streamlit as st
import pandas as pd
import streamlit.components.v1 as components
import os

st.set_page_config(page_title="India Soil Health Analysis", layout="wide")

@st.cache_data
def load_data():
    if os.path.exists('data/processed/shc_final_output.csv'):
        return pd.read_csv('data/processed/shc_final_output.csv'), False
    elif os.path.exists('data/sample/shc_final_sample.csv'):
        return pd.read_csv('data/sample/shc_final_sample.csv'), True
    else:
        st.error("Data files not found.")
        st.stop()

df, is_sample = load_data()

if is_sample:
    st.warning("⚠️ Running on 5k sample (full dataset not found).")

# 1. Sidebar filters
st.sidebar.title("Filters")
states = sorted(df['state_name'].dropna().unique())
selected_state = st.sidebar.selectbox("State", ["All"] + list(states))

if selected_state != "All":
    districts = sorted(df[df['state_name'] == selected_state]['district_name'].dropna().unique())
else:
    districts = sorted(df['district_name'].dropna().unique())
    
selected_district = st.sidebar.selectbox("District", ["All"] + list(districts))

if selected_district != "All":
    villages = sorted(df[(df['state_name'] == selected_state) & (df['district_name'] == selected_district)]['village_name'].dropna().unique())
    selected_village = st.sidebar.selectbox("Village", ["All"] + list(villages))
else:
    selected_village = "All"

soil_zones = sorted(df['soil_zone_name'].dropna().unique())
selected_zones = st.sidebar.multiselect("Soil Zone", soil_zones, default=soil_zones)

anomaly_status = st.sidebar.selectbox("Anomaly Status", ["All", "Global anomaly only", "Local anomaly only", "Both", "None"])

# Filter dataframe based on selections
filtered_df = df.copy()

if selected_state != "All":
    filtered_df = filtered_df[filtered_df['state_name'] == selected_state]
if selected_district != "All":
    filtered_df = filtered_df[filtered_df['district_name'] == selected_district]
if selected_village != "All":
    filtered_df = filtered_df[filtered_df['village_name'] == selected_village]
filtered_df = filtered_df[filtered_df['soil_zone_name'].isin(selected_zones)]

if anomaly_status == "Global anomaly only":
    filtered_df = filtered_df[(filtered_df['dbscan_label'] == -1) & (~filtered_df['is_local_anomaly'])]
elif anomaly_status == "Local anomaly only":
    filtered_df = filtered_df[(filtered_df['dbscan_label'] != -1) & (filtered_df['is_local_anomaly'])]
elif anomaly_status == "Both":
    filtered_df = filtered_df[(filtered_df['dbscan_label'] == -1) & (filtered_df['is_local_anomaly'])]
elif anomaly_status == "None":
    filtered_df = filtered_df[(filtered_df['dbscan_label'] != -1) & (~filtered_df['is_local_anomaly'])]

tab1, tab2, tab3 = st.tabs(["National Overview", "Village Lookup", "Methodology"])

with tab1:
    st.header("National Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Villages", f"{len(filtered_df):,}")
    
    global_anomalies = len(filtered_df[filtered_df['dbscan_label'] == -1])
    local_anomalies = len(filtered_df[filtered_df['is_local_anomaly']])
    
    col2.metric("Global Anomalies", f"{global_anomalies:,}")
    col3.metric("Local Anomalies", f"{local_anomalies:,}")
    col4.metric("Filtered %", f"{(len(filtered_df) / len(df) * 100):.1f}%")
    
    st.subheader("Interactive Soil Map")
    map_path = "outputs/figures/india_soil_map.html"
    if os.path.exists(map_path):
        with open(map_path, 'r', encoding='utf-8') as f:
            html_data = f.read()
        components.html(html_data, height=600)
        st.caption("Map shows all villages; sidebar filters apply to the cards and charts only.")
    else:
        st.warning("Map file not found.")
        
    colA, colB = st.columns(2)
    with colA:
        st.subheader("Zone Distribution")
        zone_counts = filtered_df['soil_zone_name'].value_counts()
        st.bar_chart(zone_counts)
        
    with colB:
        st.subheader("Top 10 Anomaly-Hotspot Districts (n >= 30)")
        # Calculate anomaly percentage per district
        dist_stats = filtered_df.groupby(['state_name', 'district_name']).agg(
            total_villages=('village_name', 'count'),
            global_anomalies=('dbscan_label', lambda x: (x == -1).sum())
        ).reset_index()
        
        dist_stats = dist_stats[dist_stats['total_villages'] >= 30]
        dist_stats['anomaly_pct'] = (dist_stats['global_anomalies'] / dist_stats['total_villages']) * 100
        
        top_hotspots = dist_stats.sort_values('anomaly_pct', ascending=False).head(10)
        # Prepare for st.bar_chart by setting index
        top_hotspots['district_label'] = top_hotspots['district_name'] + " (" + top_hotspots['state_name'] + ")"
        hotspot_chart_data = top_hotspots.set_index('district_label')[['anomaly_pct']]
        st.bar_chart(hotspot_chart_data)

with tab2:
    st.header("Village Lookup")
    
    search_query = st.text_input("Search for a village by name (optional)...")
    
    search_results = filtered_df.copy()
    if search_query:
        search_results = search_results[search_results['village_name'].str.contains(search_query, case=False, na=False)]
        
    if len(search_results) == 0:
        st.info("No villages found matching the current filters.")
    else:
        if len(search_results) > 50:
            st.warning(f"Found {len(search_results):,} villages matching filters. Showing the first 50. Use the search bar or sidebar to narrow down.")
            search_results = search_results.head(50)
                
            for _, r in search_results.iterrows():
                n_val = int(r['n_samples']) if pd.notna(r['n_samples']) else "Unknown"
                with st.expander(f"{r['village_name']}, {r['district_name']} ({r['state_name']}) [n={n_val}]", expanded=len(search_results)==1):
                    st.markdown(f"**Soil Zone:** {r['soil_zone_name']}")
                    
                    badges = []
                    if r['dbscan_label'] == -1: badges.append("🚨 **Global Anomaly**")
                    if r['is_local_anomaly']: badges.append("⚠️ **Local Anomaly**")
                    if badges:
                        st.markdown(" ".join(badges))
                    else:
                        st.markdown("✅ **Typical Profile**")
                        
                    st.markdown("### Explanation")
                    st.info(r['explanation_text'])
                    
                    st.markdown("### Management Recommendation")
                    st.success(r['recommendation'])

with tab3:
    st.header("Methodology Notes")
    notes_path = "methodology_notes.md"
    if os.path.exists(notes_path):
        with open(notes_path, 'r', encoding='utf-8') as f:
            st.markdown(f.read())
    else:
        st.warning("Methodology notes file not found.")

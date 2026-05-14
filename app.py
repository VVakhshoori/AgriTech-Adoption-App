import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import re
import plotly.express as px
import io 

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Agri-Tech Policy Simulator", layout="wide")

st.markdown("""
    <style>
    .block-container { 
        padding-top: 1rem !important; 
        padding-bottom: 0rem !important; 
        margin-top: 10px !important;
    }
    header { visibility: hidden; height: 0px !important; }
    h1 { 
        margin-top: -20px !important; 
        padding-top: 0px !important; 
    }
    .result-card { padding: 20px; border-radius: 10px; text-align: center; font-weight: bold; font-size: 18px; color: white; margin-bottom: 10px; }
    .bg-green { background-color: #28a745; }
    .bg-red { background-color: #dc3545; }
    .bg-blue { background-color: #007bff; }
    .bg-black { background-color: #000000; }
    .bg-yellow { background-color: #ffc107; color: black !important; }
    </style>
    """, unsafe_allow_html=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "Ensemble_Committee_Models")

SCENARIOS = ["Decrease", "Increase"]
POLICIES = ["Financial", "Education", "R&D", "Market", "Infrastructure", "Risk Management", "Technical", "Data Regulations"]

ALL_FEATURES_ORDER = [
    'Age', 'Total_area', 'Employee_number', 'Total_revenue_enc', 
    'Assets_value_enc', 'Hours_on_farm_enc', 'Hours_off_farm_enc',
    'Education_enc', 'Gender_enc', 'Immigration_enc', 'Indigenous_enc'
]

MAPPINGS = {
    'Hours_on_farm_enc': {'label': 'Hours on Farm', 'options': {'Fewer than 20 hours': 1, '20-29 hours': 2, '30-40 hours': 3, 'More than 40 hours': 4}},
    'Hours_off_farm_enc': {'label': 'Hours OFF Farm', 'options': {'None': 0, 'Fewer than 20 hours': 1, '20-29 hours': 2, '30-40 hours': 3, 'More than 40 hours': 4}},
    'Total_revenue_enc': {'label': 'Farm Revenue', 'options': {'Under $10k': 1, '$10k-$25k': 2, '$25k-$50k': 3, '$50k-$100k': 4, '$100k-$250k': 5, '$250k-$500k': 6, '$500k-$1M': 7, '$1M-$2M': 8, 'Over $2M': 9}},
    'Assets_value_enc': {'label': 'Assets Value', 'options': {'Under $100k': 1, '$100k-$200k': 2, '$200k-$350k': 3, '$350k-$500k': 4, '$500k-$1M': 5, '$1M-$1.5M': 6, '$1.5M-$2M': 7, '$2M-$3.5M': 8, 'Over $3.5M': 9}},
    'Education_enc': {'label': 'Education', 'options': {'No certificate': 1, 'High school': 2, 'Trade cert': 3, 'College diploma': 4, 'Uni (below bach)': 5, "Bachelor+": 6}},
    'Gender_enc': {'label': 'Gender', 'options': {'Men': 1, 'Women': 2, 'Other': 3, 'No Answer': 0}},
    'Immigration_enc': {'label': 'Immigration', 'options': {'No': 0, 'Yes': 1, 'No Answer': 2}},
    'Indigenous_enc': {'label': 'Indigenous', 'options': {'No': 0, 'Yes': 1, 'No Answer': 2}}
}

# --- 2. HELPER FUNCTIONS ---

def get_target_folder_name(tool, scenario, policy):
    return f"{scenario}_{tool.replace(' ', '_')}_{policy.replace(' ', '_')}"

def load_required_features(target_name):
    path = os.path.join(MODELS_DIR, target_name, "features.list")
    try: return joblib.load(path)
    except: return None

def map_raw_to_enc(df):
    def revenue_map(val):
        try:
            if pd.isna(val): return 1
            v = float(val)
            if v < 15: return v 
            if v < 10000: return 1
            elif v < 25000: return 2
            elif v < 50000: return 3
            elif v < 100000: return 4
            elif v < 250000: return 5
            elif v < 500000: return 6
            elif v < 1000000: return 7
            elif v < 2000000: return 8
            else: return 9
        except: return 1
    
    def assets_map(val):
        try:
            if pd.isna(val): return 1
            v = float(val)
            if v < 15: return v
            if v < 100000: return 1
            elif v < 200000: return 2
            elif v < 350000: return 3
            elif v < 500000: return 4
            elif v < 1000000: return 5
            elif v < 1500000: return 6
            elif v < 2000000: return 7
            elif v < 3500000: return 8
            else: return 9
        except: return 1

    def hours_map(val, is_off_farm=False):
        try:
            if pd.isna(val) or str(val).lower() == 'none' or val == 0: return 0
            v = float(val)
            if v <= 4 and v > 0: return v
            if v < 20: return 1
            elif v <= 29: return 2
            elif v <= 40: return 3
            else: return 4
        except: return 0

    df = df.copy()
    if 'Total_revenue_enc' in df.columns: df['Total_revenue_enc'] = df['Total_revenue_enc'].apply(revenue_map)
    if 'Assets_value_enc' in df.columns: df['Assets_value_enc'] = df['Assets_value_enc'].apply(assets_map)
    if 'Hours_on_farm_enc' in df.columns: df['Hours_on_farm_enc'] = df['Hours_on_farm_enc'].apply(lambda x: hours_map(x, False))
    if 'Hours_off_farm_enc' in df.columns: df['Hours_off_farm_enc'] = df['Hours_off_farm_enc'].apply(lambda x: hours_map(x, True))
    return df

def get_prediction_and_auc(target_name, input_df):
    target_path = os.path.join(MODELS_DIR, target_name)
    model_files = [f for f in os.listdir(target_path) if f.endswith(".pkl")]
    probs, aucs = [], []
    for f in model_files:
        model = joblib.load(os.path.join(target_path, f))
        probs.append(model.predict_proba(input_df)[0][1])
        match = re.search(r"AUC_(\d+\.\d+)", f)
        if match: aucs.append(float(match.group(1)))
    return np.mean(probs), probs, np.mean(aucs) if aucs else 0.0

def batch_predict(target_name, input_df):
    target_path = os.path.join(MODELS_DIR, target_name)
    if not os.path.exists(target_path): return None
    model_files = [f for f in os.listdir(target_path) if f.endswith(".pkl")]
    if not model_files: return None
    all_probs_matrix = np.zeros((len(input_df), len(model_files)))
    aucs = []
    for i, f in enumerate(model_files):
        model = joblib.load(os.path.join(target_path, f))
        all_probs_matrix[:, i] = model.predict_proba(input_df)[:, 1]
        match = re.search(r"AUC_(\d+\.\d+)", f)
        if match: aucs.append(float(match.group(1)))
    return np.mean(all_probs_matrix, axis=1), np.min(all_probs_matrix, axis=1), np.max(all_probs_matrix, axis=1), np.mean(aucs)

# --- 3. INTERFACE ---
st.title("Agri-Tech Policy Adoption Simulator")

with st.sidebar:
    st.header("1. Configuration")
    sel_tool = st.selectbox("Agri-Technology", ["Digital", "CEA", "Genomic"])
    sel_scenario = st.selectbox("Production Scenario", SCENARIOS)
    sel_policy = st.selectbox("Policy", POLICIES)
    target_name = get_target_folder_name(sel_tool, sel_scenario, sel_policy)
    active_features = load_required_features(target_name)
    model_found = active_features is not None

st.header("2. Socioeconomic Profile")
user_inputs = {}
c1, c2, c3 = st.columns(3)
for i, feat in enumerate(ALL_FEATURES_ORDER):
    is_active = feat in active_features if model_found else False
    with [c1, c2, c3][i % 3]:
        if feat in ['Age', 'Total_area', 'Employee_number']:
            val = st.number_input(f"{feat.replace('_', ' ')}", value=45 if feat == 'Age' else 2, disabled=not is_active, key=f"ui_{feat}")
            user_inputs[feat] = float(val)
        else:
            m = MAPPINGS[feat]
            txt = st.selectbox(f"{m['label']}", options=list(m['options'].keys()), disabled=not is_active, key=f"ui_{feat}")
            user_inputs[feat] = float(m['options'][txt])

if "manual_results" not in st.session_state:
    st.session_state.manual_results = None

if st.button("Run Ensemble Prediction", type="primary"):
    final_df = pd.DataFrame([user_inputs])[active_features]
    avg_p, all_p, avg_auc = get_prediction_and_auc(target_name, final_df)
    st.session_state.manual_results = {
        "avg_p": avg_p,
        "all_p": all_p,
        "avg_auc": avg_auc,
        "policy": sel_policy
    }

if st.session_state.manual_results:
    res = st.session_state.manual_results
    avg_p, all_p, avg_auc = res["avg_p"], res["all_p"], res["avg_auc"]
    st.header("3. Results")
    res_col1, res_col2, res_col3 = st.columns(3)
    color = "bg-green" if avg_p >= 0.7 else "bg-yellow" if avg_p >= 0.45 else "bg-red"
    label = "Strong Adoption" if avg_p >= 0.7 else "Moderate Adoption" if avg_p >= 0.45 else "Low Adoption"
    res_col1.markdown(f'<div class="result-card {color}">{label}: {avg_p:.1%}</div>', unsafe_allow_html=True)
    res_col2.markdown(f'<div class="result-card bg-blue">Adoption Range: {min(all_p):.1%} - {max(all_p):.1%}</div>', unsafe_allow_html=True)
    res_col3.markdown(f'<div class="result-card bg-black">Avg AUC: {avg_auc:.3f}</div>', unsafe_allow_html=True)
    chart_data = pd.DataFrame({'Top Models': [f"M{i+1}" for i in range(len(all_p))], 'Prob': all_p})
    fig = px.bar(chart_data, x='Top Models', y='Prob', text_auto='.1%', color='Prob', color_continuous_scale='YlGn')
    fig.update_layout(height=400, showlegend=False, margin=dict(t=30, b=30, l=30, r=30), paper_bgcolor="white", plot_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)

# --- 5. BATCH PROCESSING ---
st.markdown("---")

if "batch_excel_data" not in st.session_state:
    st.session_state.batch_excel_data = None
if "file_uploaded_flag" not in st.session_state:
    st.session_state.file_uploaded_flag = False

should_expand = (st.session_state.batch_excel_data is not None) or st.session_state.file_uploaded_flag

with st.expander(f"📂 Batch Processing (16 Scenarios for {sel_tool})", expanded=should_expand):
    st.markdown(f"**This will generate a single Excel file with one sheet containing all 16 results for {sel_tool}.**")
    uploaded_file = st.file_uploader("Upload Farmer Profiles", type=["csv", "xlsx"])
    
    if uploaded_file:
        st.session_state.file_uploaded_flag = True
        raw_df_original = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        
        if st.button("Run Batch Predictions"):
            all_results_list = [] # List to collect dataframes
            progress_bar = st.progress(0)
            status_text = st.empty()
            total_steps = len(SCENARIOS) * len(POLICIES)
            step = 0
            
            for sc in SCENARIOS:
                for pol in POLICIES:
                    step += 1
                    progress_bar.progress(step / total_steps)
                    status_text.text(f"Processing: {sc} - {pol}...")
                    
                    t_name = get_target_folder_name(sel_tool, sc, pol)
                    feat_list = load_required_features(t_name)
                    
                    if feat_list:
                        df_proc = raw_df_original.copy()
                        for feat, mapping in MAPPINGS.items():
                            if feat in df_proc.columns and not pd.api.types.is_numeric_dtype(df_proc[feat]):
                                clean_options = {str(k).strip().lower(): v for k, v in mapping['options'].items()}
                                df_proc[feat] = df_proc[feat].astype(str).str.strip().str.lower().map(clean_options)
                        
                        df_proc = map_raw_to_enc(df_proc)
                        input_data = df_proc[feat_list].fillna(0).astype(float)
                        res_tuple = batch_predict(t_name, input_data)
                        
                        if res_tuple:
                            avg_p, min_p, max_p, b_auc = res_tuple
                            sheet_df = raw_df_original.copy()
                            
                            # Add Scenario and Policy markers
                            sheet_df.insert(0, 'Scenario', sc)
                            sheet_df.insert(1, 'Policy', pol)
                            
                            sheet_df['Adoption_Probability'] = [f"{x:.1%}" for x in avg_p]
                            sheet_df['Adoption_Range'] = [f"{l:.1%} - {h:.1%}" for l, h in zip(min_p, max_p)]
                            sheet_df['Ensemble_AUC'] = round(b_auc, 3)
                            sheet_df['Adoption_Category'] = np.select([(avg_p >= 0.7), (avg_p >= 0.45)], ['Strong', 'Moderate'], default='Low')
                            
                            all_results_list.append(sheet_df)

            # Combine all results into a single DataFrame
            if all_results_list:
                final_combined_df = pd.concat(all_results_list, ignore_index=True)
                
                output_buffer = io.BytesIO()
                with pd.ExcelWriter(output_buffer, engine='xlsxwriter') as writer:
                    final_combined_df.to_excel(writer, sheet_name="Combined_Results", index=False)
                
                st.session_state.batch_excel_data = output_buffer.getvalue()
                st.success(f"16-Scenario Processing Complete for {sel_tool}!")
            
            progress_bar.empty()
            status_text.empty()

    if st.session_state.batch_excel_data:
        st.download_button(
            label=f"📥 Download Combined Results Excel ({sel_tool})",
            data=st.session_state.batch_excel_data,
            file_name=f"Combined_Batch_Results_{sel_tool}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

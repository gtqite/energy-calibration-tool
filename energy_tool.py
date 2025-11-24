import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import minimize
import time

# --- CONFIGURATION & PAGE SETUP ---
st.set_page_config(page_title="Auto-M&V Tool", layout="wide")

st.title("⚡ Energy Model Auto-Calibration Tool")
st.markdown("""
This tool allows you to upload monthly utility data, auto-calibrate a simulation model 
to match that data, and estimate savings from Energy Efficiency Measures (EEMs).
""")

# --- SIDEBAR: INPUTS ---
st.sidebar.header("1. Project Setup")
model_name = st.sidebar.text_input("Building Name", "Office_Building_01")
bldg_area = st.sidebar.number_input("Building Area (sq ft)", value=50000)

st.sidebar.header("2. Upload Utility Data")
uploaded_file = st.sidebar.file_uploader("Upload Monthly CSV (Columns: Month, kWh)", type=["csv"])

# --- CORE SIMULATION ENGINE (MOCK) ---
# In a real production app, this function would call the EnergyPlus CLI or OpenStudio SDK.
# Since we cannot run the .exe here, we simulate the physics response mathematically.

def run_energyplus_simulation(inputs):
    # Import the LBNL library
from energyplus_mcp import Sensitivity, Calibration

def run_real_calibration(uploaded_idf, utility_data):
    # 1. Setup the calibrator
    calibrator = Calibration.Calibrator(
        model_path=uploaded_idf, 
        weather_path="USA_CA_San.Francisco.epw"
    )

    # 2. Feed it the utility bill data (CSV)
    calibrator.load_observation_data(utility_data)

    # 3. Define what to vary (The "Knobs")
    # The library has built-in definitions for common variables
    calibrator.set_parameters([
        {'name': 'Lights', 'min': 0.5, 'max': 2.0},
        {'name': 'Cooling_Setpoint', 'min': 20, 'max': 26}
    ])

    # 4. Run the Bayesian Calibration
    # This takes time! (Minutes to Hours)
    results = calibrator.run()
    
    return results

# --- CALIBRATION LOGIC ---
def objective_function(params, actual_data):
    """
    The function the AI tries to minimize.
    Calculates the error (CV-RMSE) between the Model and the Bill.
    """
    modeled_data = run_energyplus_simulation(params)
    
    # Calculate RMSE
    mse = np.mean((modeled_data - actual_data) ** 2)
    rmse = np.sqrt(mse)
    
    # CV(RMSE)
    mean_actual = np.mean(actual_data)
    cv_rmse = (rmse / mean_actual) * 100
    return cv_rmse

# --- MAIN APP LOGIC ---

if uploaded_file is not None:
    # 1. Parse Data
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 1. Baseline Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(df.head(), use_container_width=True)
    with col2:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=df['Month'], y=df['kWh'], name='Utility Bill', marker_color='blue'))
        fig.update_layout(title="Uploaded Utility Data")
        st.plotly_chart(fig, use_container_width=True)
        
    actual_kwh = df['kWh'].values

    # 2. Calibration Control
    st.subheader("⚙️ 2. Auto-Calibration Engine")
    st.info("The system will now vary model parameters to match the utility bills.")
    
    if st.button("Start Calibration Loop"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Initial Guesses for the building [LPD, Setpoint, Heat_Eff]
        initial_guess = [1.5, 70.0, 0.8] 
        
        # Bounds (Physics Constraints)
        # LPD: 0.5 to 2.5 W/sf
        # Setpoint: 68 to 78 F
        # Eff: 0.6 to 0.95
        bounds = ((0.5, 2.5), (68.0, 78.0), (0.6, 0.95))
        
        status_text.text("Running optimization algorithms...")
        
        # Run the optimization (Minimize the error)
        result = minimize(
            objective_function, 
            initial_guess, 
            args=(actual_kwh,), 
            method='SLSQP', 
            bounds=bounds,
            tol=1e-3
        )
        
        progress_bar.progress(100)
        
        # 3. Results
        calibrated_params = result.x
        final_error = result.fun
        
        st.success(f"Calibration Converged! Final CV(RMSE): {final_error:.2f}%")
        
        # Display Parameter Changes
        res_col1, res_col2, res_col3 = st.columns(3)
        res_col1.metric("Calibrated Lighting (W/sf)", f"{calibrated_params[0]:.2f}")
        res_col2.metric("Calibrated Cooling Setpoint", f"{calibrated_params[1]:.1f} °F")
        res_col3.metric("Heating Efficiency", f"{calibrated_params[2]:.2f}")
        
        # visualize Before vs After
        calibrated_model = run_energyplus_simulation(calibrated_params)
        initial_model = run_energyplus_simulation(initial_guess)
        
        comp_fig = go.Figure()
        comp_fig.add_trace(go.Bar(x=df['Month'], y=actual_kwh, name='Actual Bill', marker_color='blue'))
        comp_fig.add_trace(go.Scatter(x=df['Month'], y=initial_model, name='Uncalibrated Model', line=dict(color='red', dash='dash')))
        comp_fig.add_trace(go.Scatter(x=df['Month'], y=calibrated_model, name='Calibrated Model', line=dict(color='green', width=3)))
        
        st.plotly_chart(comp_fig, use_container_width=True)
        
        # 4. Apply EEMs
        st.subheader("💡 3. Apply Energy Efficiency Measures")
        eem_choice = st.selectbox("Select Upgrade:", ["LED Lighting Upgrade", "Setback Thermostat"])
        
        if eem_choice == "LED Lighting Upgrade":
            # Reduce LPD by 50%
            new_params = calibrated_params.copy()
            new_params[0] = new_params[0] * 0.5 
            eem_model = run_energyplus_simulation(new_params)
            
            savings = np.sum(calibrated_model) - np.sum(eem_model)
            savings_pct = (savings / np.sum(calibrated_model)) * 100
            
            st.metric("Projected Annual Savings", f"{int(savings):,} kWh", f"{savings_pct:.1f}%")
            
            eem_fig = go.Figure()
            eem_fig.add_trace(go.Scatter(x=df['Month'], y=calibrated_model, name='Baseline (Calibrated)', fill='tozeroy'))
            eem_fig.add_trace(go.Scatter(x=df['Month'], y=eem_model, name='With LED Upgrade', fill='tozeroy'))
            st.plotly_chart(eem_fig, use_container_width=True)

else:
    st.warning("Please upload a CSV file to begin. (Format: 'Month', 'kWh')")
    # Create a sample CSV for the user to download
    sample_data = {
        "Month": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
        "kWh": [45000, 42000, 40000, 48000, 55000, 65000, 72000, 73000, 60000, 50000, 46000, 47000]
    }
    sample_df = pd.DataFrame(sample_data)

    st.download_button("Download Sample CSV", sample_df.to_csv(index=False).encode('utf-8'), "sample_utility_data.csv")

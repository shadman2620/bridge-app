# =====================================================
# GEC KHAGARIA: BRIDGE DIGITAL TWIN (INTEGRATED MASTER CODE)
# Developed by: Shehnai Gandhi, Babli Kumari, Shaili Kumari, Raj Kishor
# =====================================================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.ensemble import RandomForestRegressor

# ================= PAGE SETUP =================
st.set_page_config(page_title="GEC Khagaria Bridge Health Monitor", layout="wide")

# ================= USER GUIDE =================
with st.expander("📖 SYSTEM DOCUMENTATION & LOGIC"):
    st.markdown("""
    ### 🏗️ Integrated Features
    1. **Structural Physics:** Euler-Bernoulli theory for real-time deflection.
    2. **Dynamic Analysis:** Natural frequency calculation based on material stiffness.
    3. **AI Module:** Random Forest Regressor for life prediction (RUL).
    4. **Health Audit:** Automated reporting and stiffness degradation tracking.
    """)

# ================= MATERIAL DATA (IS 456:2000) =================
concrete_grades = {
    "M25": 25000, "M30": 27386, "M35": 29580, "M40": 31622, "M50": 35355
}

# ================= SIDEBAR =================
st.sidebar.header("🌉 Bridge Design Parameters")
grade = st.sidebar.selectbox("Select Concrete Grade", list(concrete_grades.keys()), index=1)
initial_E = float(concrete_grades[grade])

b = st.sidebar.number_input("Width b (m)", value=0.5)
h = st.sidebar.number_input("Depth h (m)", value=1.0)
L = st.sidebar.number_input("Span Length L (m)", value=20.0)

# Constants for Dynamic Analysis
mass_density = 2500 # kg/m3 for RCC
I_calc = (b * (h**3)) / 12
area = b * h
m_per_unit = mass_density * area

# ================= SESSION STATE =================
if 'e_current' not in st.session_state or st.sidebar.button("Reset Simulation"):
    st.session_state.e_current = initial_E
    st.session_state.history = []
    st.session_state.is_collapsed = False

# ================= STRUCTURAL CALCULATIONS =================
curr_e_pa = st.session_state.e_current * 1e6
limit_mm = (L * 1000) / 800

# Physics Logic: Natural Frequency (Fundamental Mode)
# Formula: f = (pi/2L^2) * sqrt(EI/m)
natural_freq = (np.pi / (2 * L**2)) * np.sqrt((curr_e_pa * I_calc) / m_per_unit)

p_perm = (limit_mm/1000 * 48 * curr_e_pa * I_calc) / (L**3) / 1000
p_ultimate = 1.5 * p_perm

# ================= MAIN DASHBOARD =================
st.title("🏗️ Bridge Digital Twin: AI Health Monitor")
st.subheader("GEC Khagaria | Physics + AI Integration")

# Live Metrics Row
m1, m2, m3, m4 = st.columns(4)
m1.metric("Current Stiffness", f"{st.session_state.e_current/1000:.2f} GPa")
m2.metric("Natural Frequency", f"{natural_freq:.2f} Hz")
m3.metric("Permissible Load", f"{p_perm:.1f} kN")
health_pct = (st.session_state.e_current / initial_E) * 100
m4.metric("Health Index", f"{health_pct:.1f}%")

st.markdown("---")

# ================= IMPACT ANALYSIS & VISUALIZATION =================
if not st.session_state.is_collapsed:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.write("### 🚀 Structural Impact Analysis")
        applied_p = st.number_input("Applied Load (kN)", value=100.0)

        if st.button("RUN IMPACT ANALYSIS"):
            if applied_p >= p_ultimate:
                st.session_state.is_collapsed = True
                st.session_state.e_current = 0
                st.error("💥 BRIDGE COLLAPSED: Load exceeded Ultimate Capacity!")
            else:
                # Damage Model: Miner's Rule Simulation
                load_ratio = applied_p / p_perm
                damage_factor = 0.015 + (load_ratio**3) * 0.12 # Non-linear decay
                
                delta = ((applied_p*1000*(L**3))/(48*curr_e_pa*I_calc))*1000

                st.session_state.history.append({
                    "Cycle": len(st.session_state.history)+1,
                    "Load_kN": applied_p,
                    "Deflection_mm": round(delta, 2),
                    "Frequency_Hz": round(natural_freq, 2),
                    "Remaining_Health_%": round(health_pct, 2)
                })

                st.session_state.e_current *= (1 - damage_factor)
                st.rerun()

    with col2:
        st.write("### 📊 Stress Heatmap")
        # Creating a dynamic heatmap based on health status
        fig_heat, ax_heat = plt.subplots(figsize=(8, 2))
        cmap = plt.cm.get_cmap('RdYlGn')
        # Map health index to color
        ax_heat.add_patch(plt.Rectangle((0, 0), L, h, color=cmap(health_pct/100)))
        ax_heat.set_xlim(0, L)
        ax_heat.set_ylim(0, h)
        ax_heat.set_yticks([])
        ax_heat.set_title(f"Structural Integrity State: {health_pct:.1f}%")
        st.pyplot(fig_heat)

# ================= AI MODULE & DATA LOGGING =================
st.markdown("---")
c1, c2 = st.columns(2)

with c1:
    st.subheader("🤖 AI Life Forecast")
    # AI Logic: Predict Remaining Useful Life based on Fatigue
    # Training a local RF model on synthetic fatigue data
    X_train = np.linspace(10, p_ultimate, 200).reshape(-1, 1)
    y_train = 10000 * np.exp(-0.008 * X_train.flatten()) # Inverse relation load vs life
    rf_model = RandomForestRegressor(n_estimators=100).fit(X_train, y_train)
    
    ai_load = st.number_input("Predict Life for Load (kN):", value=applied_p if 'applied_p' in locals() else 100.0)
    prediction = rf_model.predict([[ai_load]])[0]
    st.info(f"AI Prediction: Structure can sustain approx. **{int(prediction)}** more cycles at this load.")

with c2:
    st.subheader("📜 Structural Health Audit")
    if st.session_state.history:
        df_history = pd.DataFrame(st.session_state.history)
        st.dataframe(df_history.tail(5))
        # Export functionality
        csv = df_history.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Health Report", data=csv, file_name="bridge_health_audit.csv", mime='text/csv')
    else:
        st.write("No cycles recorded yet. Run an analysis to generate data.")

# ================= DYNAMIC MOVING LOAD =================
st.markdown("---")
st.subheader("🚗 Live Traffic Simulation")
if st.button("▶️ Start Moving Load Visualization"):
    plot_spot = st.empty()
    x_points = np.linspace(0, L, 100)
    
    for pos in np.linspace(0, L, 40):
        # Calculation for moving point load deflection
        y_def = []
        for xi in x_points:
            if xi <= pos:
                val = (applied_p * 1000 * (L-pos) * xi * (L**2 - (L-pos)**2 - xi**2)) / (6 * curr_e_pa * I_calc * L)
            else:
                val = (applied_p * 1000 * pos * (L-xi) * (L**2 - pos**2 - (L-xi)**2)) / (6 * curr_e_pa * I_calc * L)
            y_def.append(-val * 1000) # Negative for downward deflection

        fig_sim, ax_sim = plt.subplots(figsize=(10, 4))
        ax_sim.plot(x_points, y_def, color='darkblue', lw=2)
        ax_sim.axhline(0, color='black', lw=1)
        ax_sim.scatter([pos], [0], color='red', s=100, zorder=5, label="Vehicle")
        ax_sim.set_ylim(-limit_mm * 1.5, 5)
        ax_sim.set_title(f"Dynamic Elastic Curve (Vehicle at {pos:.1f}m)")
        ax_sim.legend()
        plot_spot.pyplot(fig_sim)
        plt.close(fig_sim)
        time.sleep(0.05)

if st.session_state.is_collapsed:
    st.error("🚨 SYSTEM HALTED: BRIDGE COLLAPSED DUE TO OVERLOAD.")

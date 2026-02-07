import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
from sklearn.ensemble import RandomForestRegressor
import io

# ================= PAGE SETUP =================
st.set_page_config(page_title="NIT Patna Bridge Digital Twin", layout="wide")

# ================= MATERIAL DATA (IS 456:2000) =================
concrete_grades = {"M25": 25000, "M30": 27386, "M35": 29580, "M40": 31622, "M50": 35355}

# ================= SIDEBAR =================
st.sidebar.header("🌉 Bridge Design Parameters")
grade = st.sidebar.selectbox("Select Concrete Grade", list(concrete_grades.keys()), index=1)
initial_E = float(concrete_grades[grade])

b = st.sidebar.number_input("Width b (m)", value=0.5)
h = st.sidebar.number_input("Depth h (m)", value=1.0)
L = st.sidebar.number_input("Span Length L (m)", value=20.0)
I_calc = (b * (h**3)) / 12

# ================= SESSION STATE =================
if 'e_current' not in st.session_state:
    st.session_state.e_current = initial_E
    st.session_state.history = []

def reset_sim():
    st.session_state.e_current = initial_E
    st.session_state.history = []

if st.sidebar.button("🔄 Reset Simulation"):
    reset_sim()
    st.rerun()

# ================= CALCULATIONS =================
limit_mm = (L * 1000) / 800
curr_e_pa = st.session_state.e_current * 1e6
p_perm = (limit_mm/1000 * 48 * curr_e_pa * I_calc) / (L**3) / 1000
p_ultimate = 1.5 * p_perm

# ================= HEADER =================
st.title("🏗️ NIT Patna Bridge Digital Twin")
st.write("**M.Tech Structural Engineering Research | Digital Twin & AI Framework**")

m1, m2, m3 = st.columns(3)
m1.metric("Current Stiffness (E)", f"{st.session_state.e_current/1000:.2f} GPa")
m2.metric("Ultimate Capacity (Pu)", f"{p_ultimate:.1f} kN")
m3.metric("Permissible Defl. (L/800)", f"{limit_mm:.2f} mm")

st.markdown("---")

# ================= AI FATIGUE PREDICTION =================
st.subheader("🤖 AI Fatigue & Life Prediction")
col_ai1, col_ai2 = st.columns([1, 2])

# Physics-based Synthetic Training for the Model
train_p = np.linspace(10, p_ultimate*1.2, 100).reshape(-1, 1)
train_y = (p_ultimate / (train_p + 1))**2.8 * 5000 # Fatigue life curve
rf_model = RandomForestRegressor(n_estimators=100).fit(train_p, train_y.ravel())

with col_ai1:
    test_p = st.number_input("Input Load for Prediction (kN)", value=200.0)
    if st.button("Calculate Remaining Life"):
        prediction = rf_model.predict([[test_p]])[0]
        st.info(f"Predicted Service Life: {int(prediction)} Cycles")

# ================= LIVE SIMULATION & ANIMATION =================
st.subheader("🚗 Live Moving Load & Impact Analysis")
sim_p = st.number_input("Vehicle Weight for Simulation (kN)", value=250.0)

if st.button("▶️ Start Digital Twin Animation"):
    x_range = np.linspace(0, L, 100)
    plot_spot = st.empty()
    
    # Structural Impact: Damage calculation for this cycle
    load_ratio = sim_p / p_perm
    damage = 0.01 + (load_ratio**3) * 0.05 if sim_p < p_ultimate else 1.0
    
    # Animation Loop
    for pos in np.linspace(0, L, 40):
        y_def = []
        for xi in x_range:
            if xi <= pos:
                b_dist = L - pos
                val = (sim_p * 1000 * b_dist * xi * (L**2 - b_dist**2 - xi**2)) / (6 * curr_e_pa * I_calc * L)
            else:
                a_dist = pos
                val = (sim_p * 1000 * a_dist * (L - xi) * (L**2 - a_dist**2 - (L - xi)**2)) / (6 * curr_e_pa * I_calc * L)
            y_def.append(-val * 1000)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_range, y=y_def, mode='lines', fill='tozeroy', name='Deflection', line=dict(color='royalblue', width=4)))
        fig.add_trace(go.Scatter(x=[pos], y=[5], mode='markers+text', marker=dict(symbol='triangle-down', size=25, color='red'), text=["VEHICLE"]))
        
        fig.update_layout(yaxis=dict(range=[-limit_mm*2.5, 15], title="Deflection (mm)"), xaxis=dict(title="Span (m)"), height=450, showlegend=False)
        plot_spot.plotly_chart(fig, use_container_width=True, key=f"f_{pos}")
        time.sleep(0.02)

    # After animation, update global stiffness
    st.session_state.e_current *= (1 - damage)
    st.session_state.history.append({
        "Cycle": len(st.session_state.history) + 1,
        "Load (kN)": sim_p,
        "Max Defl (mm)": round(abs(min(y_def)), 2),
        "Stiffness (GPa)": round(st.session_state.e_current/1000, 2)
    })
    st.rerun()

# ================= DATA LOG & EXPORT =================
if st.session_state.history:
    st.markdown("---")
    st.subheader("📜 Structural Health Log")
    df = pd.DataFrame(st.session_state.history)
    st.dataframe(df, use_container_width=True)
    
    # Excel Export
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Structural_Data')
    
    st.download_button(
        label="📥 Download Research Data (Excel)",
        data=buffer.getvalue(),
        file_name="bridge_health_data.xlsx",
        mime="application/vnd.ms-excel"
    )

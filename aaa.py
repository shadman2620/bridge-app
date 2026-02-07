import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
from sklearn.ensemble import RandomForestRegressor

# ================= PAGE SETUP =================
st.set_page_config(page_title="NIT Patna Bridge Digital Twin", layout="wide")

# ================= MATERIAL & DATA =================
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

if st.sidebar.button("🔄 Reset Simulation"):
    st.session_state.e_current = initial_E
    st.session_state.history = []
    st.rerun()

# ================= CALCULATIONS =================
limit_mm = (L * 1000) / 800
curr_e_pa = st.session_state.e_current * 1e6
p_perm = (limit_mm/1000 * 48 * curr_e_pa * I_calc) / (L**3) / 1000
p_ultimate = 1.5 * p_perm

# ================= HEADER =================
st.title("🏗️ NIT Patna Bridge Digital Twin")
st.write("**M.Tech Structural Engineering Research**")

m1, m2, m3 = st.columns(3)
m1.metric("Current Stiffness", f"{st.session_state.e_current/1000:.2f} GPa")
m2.metric("Max Capacity", f"{p_ultimate:.1f} kN")
m3.metric("Permissible Deflection", f"{limit_mm:.1f} mm")

# ================= AI PREDICTION MODULE =================
st.markdown("---")
st.subheader("🤖 AI Fatigue Prediction")

# Dummy Training Data for AI (Physics-based)
def get_life(p): return (p_ultimate/(p+1))**2.5 * 1000

train_p = np.linspace(10, p_ultimate, 100).reshape(-1, 1)
train_y = np.array([get_life(x) for x in train_p])
model = RandomForestRegressor(n_estimators=50).fit(train_p, train_y)

col_ai1, col_ai2 = st.columns([1, 2])
with col_ai1:
    test_load = st.number_input("Test Load for AI (kN)", value=200.0)
    if st.button("Predict Bridge Life"):
        pred = model.predict([[test_load]])[0]
        st.success(f"Estimated Life: {int(pred)} Cycles")

# ================= ANIMATION (STABLE VERSION) =================
st.markdown("---")
st.subheader("🚗 Live Moving Load Simulation")
sim_p = st.number_input("Simulation Vehicle Weight (kN)", value=200.0)

if st.button("▶️ Run Smooth Animation"):
    x_range = np.linspace(0, L, 100)
    plot_spot = st.empty()
    
    # Pre-calculating animation frames to avoid "blinking"
    positions = np.linspace(0, L, 30) 
    
    for pos in positions:
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
        # Bridge line
        fig.add_trace(go.Scatter(x=[0, L], y=[0, 0], mode='lines', line=dict(color='black', width=3)))
        # Deflection curve
        fig.add_trace(go.Scatter(x=x_range, y=y_def, mode='lines', fill='tozeroy', 
                                 line=dict(color='royalblue', width=5), name="Deflection"))
        # Load Marker
        fig.add_trace(go.Scatter(x=[pos], y=[5], mode='markers+text',
                                 marker=dict(symbol='triangle-down', size=25, color='red'),
                                 text=["VEHICLE"], textposition="top center"))

        fig.update_layout(
            yaxis=dict(range=[-limit_mm*2, 15], title="Deflection (mm)"),
            xaxis=dict(range=[0, L], title="Span (m)"),
            height=500, showlegend=False,
            title=f"Monitoring: Load at {pos:.1f}m | Max: {min(y_def):.2f}mm"
        )
        
        plot_spot.plotly_chart(fig, use_container_width=True, key=f"anim_{pos}")
        time.sleep(0.01)

st.info("💡 Agar blink kare, toh 'Reset Simulation' dabayein aur phir animation start karein.")

# =====================================================
# NIT PATNA: BRIDGE DIGITAL TWIN (FINAL MASTER CODE)
# M.Tech Structural Engineering Research Project
# =====================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import time
from sklearn.ensemble import RandomForestRegressor

# ================= PAGE CONFIG =================
st.set_page_config(page_title="NIT Patna Bridge Health Monitor", layout="wide")

# ================= MATERIAL DATA (IS 456) =================
concrete_grades = {
    "M25": 25000,
    "M30": 27386,
    "M35": 29580,
    "M40": 31623,
    "M50": 35355
}

# ================= SIDEBAR =================
st.sidebar.header("🌉 Bridge Design Parameters")
grade = st.sidebar.selectbox("Concrete Grade", list(concrete_grades.keys()))
initial_E = concrete_grades[grade] * 1e6  # MPa → Pa

b = st.sidebar.number_input("Width b (m)", value=0.5)
h = st.sidebar.number_input("Depth h (m)", value=1.0)
L = st.sidebar.number_input("Span Length L (m)", value=20.0)

I = (b * h**3) / 12  # Moment of inertia

# ================= SESSION STATE =================
if "E_current" not in st.session_state:
    st.session_state.E_current = initial_E
    st.session_state.history = []
    st.session_state.collapse = False

if st.sidebar.button("🔄 Reset Simulation"):
    st.session_state.E_current = initial_E
    st.session_state.history = []
    st.session_state.collapse = False

# ================= STRUCTURAL LIMITS =================
limit_mm = (L * 1000) / 800
E = st.session_state.E_current

# Permissible load from deflection criteria
P_perm = (limit_mm/1000 * 48 * E * I) / (L**3) / 1000  # kN
P_ult = 1.5 * P_perm

# ================= HEADER =================
st.title("🏗️ NIT Patna Bridge Health Monitor")
st.subheader("M.Tech Structural Engineering | Digital Twin + AI + Fatigue")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Current Stiffness (GPa)", f"{E/1e9:.2f}")
m2.metric("Safe Load (kN)", f"{0.75*P_perm:.1f}")
m3.metric("Permissible Load (kN)", f"{P_perm:.1f}")
m4.metric("Ultimate Load (kN)", f"{P_ult:.1f}")

st.markdown("---")

# ================= IMPACT ANALYSIS =================
col1, col2 = st.columns(2)

with col1:
    st.header("Structural Impact Analysis")
    P = st.number_input("Applied Load (kN)", value=100.0)

    if st.button("RUN IMPACT ANALYSIS"):
        if P >= P_ult:
            st.session_state.collapse = True
            st.error("💥 BRIDGE COLLAPSED")
        else:
            # Deflection (m)
            delta = (P*1000 * L**3) / (48 * E * I)
            delta_mm = delta * 1000

            load_ratio = P / P_perm
            damage = 0.01 + (load_ratio**3)*0.10  # fatigue degradation

            if delta_mm > limit_mm:
                st.error(f"🔴 Deflection = {delta_mm:.2f} mm")
            elif delta_mm > 0.75*limit_mm:
                st.warning(f"🟠 Deflection = {delta_mm:.2f} mm")
            else:
                st.success(f"🟢 Deflection = {delta_mm:.2f} mm SAFE")

            st.session_state.history.append([P, delta_mm, damage*100, E/1e9])
            st.session_state.E_current = E * (1 - damage)

with col2:
    health = (st.session_state.E_current / initial_E) * 100
    st.header(f"Health Index = {health:.2f} %")

    cmap = plt.get_cmap("RdYlGn")
    colors = cmap(np.linspace(0,1,200))
    fig, ax = plt.subplots(figsize=(6,1))
    ax.imshow([colors], extent=[0,100,0,1])
    ax.axvline(health, color="black", linewidth=3)
    ax.set_yticks([])
    ax.set_xlabel("Structural Health (%)")
    st.pyplot(fig)

# ================= FATIGUE + AI MODULE =================
st.markdown("---")
st.header("🤖 Fatigue Life Prediction (Physics + AI)")

sigma_u = P_ult
sigma_f = 0.9 * sigma_u
b_f = -0.09

def predict_cycles(load):
    if load >= sigma_u:
        return 1
    return (sigma_f/load)**(1/abs(b_f))

# Training AI Model
np.random.seed(0)
X = np.random.uniform(0.1*sigma_u, sigma_u, 300).reshape(-1,1)
Y = np.array([predict_cycles(x[0]) for x in X])
rf = RandomForestRegressor(n_estimators=200)
rf.fit(X, Y)

load_ai = st.number_input("Load for AI Prediction (kN)", value=100.0)
if st.button("AI Predict Life"):
    st.success(f"Physics Life = {int(predict_cycles(load_ai))} cycles")
    st.info(f"AI Predicted Life = {int(rf.predict([[load_ai]])[0])} cycles")

# ================= MOVING LOAD SIMULATION =================
st.markdown("---")
st.header("🚗 Live Moving Load Simulation")

sim_load = st.number_input("Vehicle Load (kN)", value=200.0)
if st.button("▶ Start Simulation"):
    x = np.linspace(0, L, 50)
    placeholder = st.empty()

    for pos in np.linspace(0, L, 40):
        a = pos
        b_dist = L - pos
        y = []

        for xi in x:
            if xi <= a:
                val = sim_load*1000 * b_dist * xi * (L**2 - b_dist**2 - xi**2) / (6*E*I*L)
            else:
                val = sim_load*1000 * a * (L-xi) * (L**2 - a**2 - (L-xi)**2) / (6*E*I*L)
            y.append(-val*1000)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0,L], y=[0,0], mode="lines", line=dict(color="black", width=3)))
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", fill="tozeroy"))
        fig.add_trace(go.Scatter(x=[pos], y=[0], mode="markers", marker=dict(size=12)))

        fig.update_layout(
            xaxis_title="Span (m)",
            yaxis_title="Deflection (mm)",
            yaxis=dict(range=[-limit_mm*1.5, 5]),
            height=400,
            showlegend=False
        )

        placeholder.plotly_chart(fig, use_container_width=True)
        time.sleep(0.05)

# ================= HISTORY TABLE =================
if st.session_state.history:
    st.markdown("---")
    st.header("📜 Structural History Log")
    df = pd.DataFrame(st.session_state.history, columns=["Load (kN)", "Deflection (mm)", "Damage %", "E (GPa)"])
    st.dataframe(df)

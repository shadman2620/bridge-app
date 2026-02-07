# =====================================================
# NIT PATNA: BRIDGE DIGITAL TWIN (FINAL MASTER CODE)
# =====================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import time
from sklearn.ensemble import RandomForestRegressor

# ================= RAINFLOW FUNCTION =================
def rainflow_cycles(signal):
    cycles, stack = [], []
    for x in signal:
        stack.append(x)
        while len(stack) >= 3:
            s0, s1, s2 = stack[-3], stack[-2], stack[-1]
            if abs(s1 - s0) <= abs(s2 - s1):
                break
            cycles.append(abs(s1 - s0))
            stack.pop(-2)
    return cycles

# ================= PAGE SETUP =================
st.set_page_config(page_title="NIT Patna Bridge Health Monitor", layout="wide")

# ================= MATERIAL DATA =================
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

I_calc = (b * h**3) / 12

# ================= SESSION STATE =================
if "e_current" not in st.session_state:
    st.session_state.e_current = initial_E
    st.session_state.history = []
    st.session_state.is_collapsed = False

if "anim_pos" not in st.session_state:
    st.session_state.anim_pos = None

# ================= STRUCTURAL CALC =================
limit_mm = (L * 1000) / 800
curr_e_pa = st.session_state.e_current * 1e6
p_perm = (limit_mm/1000 * 48 * curr_e_pa * I_calc) / (L**3) / 1000
p_ultimate = 1.5 * p_perm

# ================= HEADER =================
st.title("🏗️ NIT Patna Bridge Health Monitor")

# ================= HEALTH =================
health = (st.session_state.e_current / initial_E) * 100
fig, ax = plt.subplots(figsize=(6,1))
colors = plt.cm.get_cmap("RdYlGn")(np.linspace(0,1,200))
ax.imshow([colors], extent=[0,100,0,1])
ax.axvline(health, color="black", linewidth=6)
ax.set_yticks([])
st.pyplot(fig)

# ================= AI =================
sigma_u, sigma_f, b_f = p_ultimate, 0.9*p_ultimate, -0.09
def predict_cycles(load):
    if load >= sigma_u: return 1
    return (load/sigma_f)**(1/b_f)/2

rf = RandomForestRegressor().fit(
    np.random.uniform(0.1*sigma_u, sigma_u, 300).reshape(-1,1),
    np.random.uniform(1,1000,300)
)

# ================= LIVE MOVING LOAD =================
st.markdown("---")
st.subheader("🚗 Live Moving Load Simulation")

sim_load = st.number_input("Vehicle Weight (kN)", value=200.0)

if st.button("🚦 START LIVE MOVING LOAD SIMULATION", use_container_width=True):
    st.session_state.anim_pos = 0.1

if st.session_state.anim_pos is not None:

    pos = st.session_state.anim_pos
    x_points = np.linspace(0, L, 80)
    y_def = []

    a, b_dist = pos, L - pos
    for xi in x_points:
        if xi <= a:
            val = (sim_load*1000*b_dist*xi*(L**2-b_dist**2-xi**2))/(6*curr_e_pa*I_calc*L)
        else:
            val = (sim_load*1000*a*(L-xi)*(L**2-a**2-(L-xi)**2))/(6*curr_e_pa*I_calc*L)
        y_def.append(-val*1000)

    max_def = min(y_def)
    max_x = x_points[np.argmin(y_def)]

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=[0,L], y=[0,0],
        mode="lines", line=dict(color="black", width=4)))

    fig.add_trace(go.Scatter(x=x_points, y=y_def,
        mode="lines", fill="tozeroy",
        line=dict(color="blue", width=4)))

    fig.add_trace(go.Scatter(
        x=[pos], y=[0],
        mode="markers+text",
        marker=dict(color="red", size=16, symbol="square"),
        text=[f"{pos:.2f} m"],
        textposition="top center"
    ))

    fig.add_trace(go.Scatter(
        x=[max_x], y=[max_def],
        mode="markers+text",

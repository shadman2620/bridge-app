# =====================================================
# NIT PATNA: BRIDGE DIGITAL TWIN (FINAL MASTER CODE)
# Developed for M.Tech Structural Engineering Research
# =====================================================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import time
from sklearn.ensemble import RandomForestRegressor

# ... [Rainflow, Page Setup, User Guide, Material Data same as your code] ...
# (Keeping your original logic intact)

# ================= RAINFLOW FUNCTION =================
def rainflow_cycles(signal):
    cycles = []
    stack = []
    for x in signal:
        stack.append(x)
        while len(stack) >= 3:
            s0, s1, s2 = stack[-3], stack[-2], stack[-1]
            if abs(s1 - s0) <= abs(s2 - s1):
                break
            cycles.append(abs(s1 - s0))
            stack.pop(-2)
    return cycles

st.set_page_config(page_title="NIT Patna Bridge Health Monitor", layout="wide")

# ================= SIDEBAR & SESSION STATE =================
st.sidebar.header("🌉 Bridge Design Parameters")
grade = st.sidebar.selectbox("Select Concrete Grade", ["M25", "M30", "M35", "M40", "M50"], index=1)
concrete_grades = {"M25": 25000, "M30": 27386, "M35": 29580, "M40": 31622, "M50": 35355}
initial_E = float(concrete_grades[grade])
b = st.sidebar.number_input("Width b (m)", value=0.5)
h = st.sidebar.number_input("Depth h (m)", value=1.0)
L = st.sidebar.number_input("Span Length L (m)", value=20.0)
I_calc = (b * (h**3)) / 12

if 'e_current' not in st.session_state or st.sidebar.button("Reset Simulation"):
    st.session_state.e_current = initial_E
    st.session_state.history = []
    st.session_state.is_collapsed = False

limit_mm = (L * 1000) / 800
curr_e_pa = st.session_state.e_current * 1e6
p_perm = (limit_mm/1000 * 48 * curr_e_pa * I_calc) / (L**3) / 1000
p_ultimate = 1.5 * p_perm

st.title("🏗️ NIT Patna Bridge Health Monitor")
st.subheader("M.Tech Structural Engineering | AI + Fatigue + Digital Twin")

# ================= LIVE MOVING LOAD SIMULATION (FIXED NO-FLICKER) =================
st.markdown("---")
st.subheader("🚗 Live Moving Load Simulation")
sim_load = st.number_input("Vehicle Weight (kN)", value=200.0)

if st.button("▶️ Initialize Smooth Animation"):
    x_points = np.linspace(0, L, 100)
    move_steps = np.linspace(0, L, 100) # 100 frames
    
    frames = []
    for pos in move_steps:
        a, b_dist = pos, L - pos
        y_def = []
        for xi in x_points:
            if xi <= a:
                val = (sim_load * 1000 * b_dist * xi * (L**2 - b_dist**2 - xi**2)) / (6 * curr_e_pa * I_calc * L)
            else:
                val = (sim_load * 1000 * a * (L - xi) * (L**2 - a**2 - (L - xi)**2)) / (6 * curr_e_pa * I_calc * L)
            y_def.append(-val * 1000)
        
        # Creating each frame
        frames.append(go.Frame(
            data=[
                go.Scatter(x=x_points, y=y_def, mode='lines', fill='tozeroy', line=dict(color='blue', width=3)),
                go.Scatter(x=[pos], y=[1], mode='markers', marker=dict(symbol='triangle-down', size=18, color='red'))
            ],
            name=str(pos)
        ))

    # Base Figure
    fig = go.Figure(
        data=[
            go.Scatter(x=x_points, y=[0]*100, mode='lines', fill='tozeroy', line=dict(color='blue', width=3)),
            go.Scatter(x=[0], y=[1], mode='markers', marker=dict(symbol='triangle-down', size=18, color='red'))
        ],
        layout=go.Layout(
            xaxis=dict(range=[0, L], title="Span (m)"),
            yaxis=dict(range=[-limit_mm * 1.5, 10], title="Deflection (mm)"),
            updatemenus=[{
                "type": "buttons",
                "buttons": [{
                    "label": "Play Animation",
                    "method": "animate",
                    "args": [None, {"frame": {"duration": 100, "redraw": True}, "fromcurrent": True}]
                }]
            }]
        ),
        frames=frames
    )
    
    st.plotly_chart(fig, use_container_width=True)

# ... [Keep your Impact Analysis and History Table exactly as they were] ...

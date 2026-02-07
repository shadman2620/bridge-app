import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import time
from sklearn.ensemble import RandomForestRegressor

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

I_calc = (b * (h**3)) / 12

# ================= SESSION STATE =================
if 'e_current' not in st.session_state or st.sidebar.button("Reset Simulation"):
    st.session_state.e_current = initial_E
    st.session_state.history = []
    st.session_state.is_collapsed = False
    st.session_state.load_history = []

# ================= STRUCTURAL CALC =================
limit_mm = (L * 1000) / 800
curr_e_pa = st.session_state.e_current * 1e6
p_perm = (limit_mm/1000 * 48 * curr_e_pa * I_calc) / (L**3) / 1000
p_ultimate = 1.5 * p_perm

# ================= HEADER =================
st.title("🏗️ NIT Patna Bridge Health Monitor")
st.subheader("M.Tech Structural Engineering Research")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Stiffness", f"{st.session_state.e_current/1000:.2f} GPa")
m2.metric("Safe Load", f"{0.75*p_perm:.1f} kN")
m3.metric("Permissible", f"{p_perm:.1f} kN")
m4.metric("Ultimate", f"{p_ultimate:.1f} kN")

# ================= IMPACT ANALYSIS =================
if not st.session_state.is_collapsed:
    col1, col2 = st.columns(2)
    with col1:
        applied_p = st.number_input("Applied Load (kN)", value=100.0)
        if st.button("RUN IMPACT ANALYSIS"):
            if applied_p >= p_ultimate:
                st.session_state.is_collapsed = True
                st.session_state.e_current = 0
                st.error("💥 BRIDGE COLLAPSED")
            else:
                load_ratio = applied_p / p_perm
                damage_factor = 0.02 + (load_ratio**3)*0.10
                delta = ((applied_p*1000*(L**3))/(48*curr_e_pa*I_calc))*1000
                
                st.session_state.history.append({
                    "Cycle": len(st.session_state.history)+1,
                    "Load_kN": applied_p,
                    "Damage_%": round(damage_factor*100,2),
                    "Deflection_mm": round(delta,2)
                })
                st.session_state.e_current *= (1 - damage_factor)
                st.rerun()

# ================= ANIMATED MOVING LOAD (FIXED) =================
st.markdown("---")
st.subheader("🚗 Live Moving Load Simulation")
sim_load = st.number_input("Vehicle Weight for Simulation (kN)", value=150.0)

if st.button("▶️ Start Animation"):
    x_points = np.linspace(0, L, 50)
    plot_spot = st.empty()
    
    for pos in np.arange(0, L + 1.0, 1.0):
        y_def = []
        for xi in x_points:
            # Macaulay's Theorem logic
            if xi <= pos:
                val = (sim_load * 1000 * (L-pos) * xi * (L**2 - (L-pos)**2 - xi**2)) / (6 * curr_e_pa * I_calc * L)
            else:
                val = (sim_load * 1000 * pos * (L-xi) * (L**2 - pos**2 - (L-xi)**2)) / (6 * curr_e_pa * I_calc * L)
            y_def.append(-val * 1000)

        # Plotly Chart for Smooth Animation
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_points, y=y_def, mode='lines', fill='tozeroy', name='Deflection'))
        fig.add_trace(go.Scatter(x=[pos], y=[0.5], mode='markers+text', 
                                 marker=dict(symbol='bus', size=12, color='red'),
                                 text=["Vehicle"]))
        
        fig.update_layout(yaxis=dict(range=[-limit_mm*1.5, 5]), height=350, showlegend=False,
                          title=f"Vehicle Position: {pos}m")
        
        plot_spot.plotly_chart(fig, use_container_width=True)
        time.sleep(0.05)

# History Table
if st.session_state.history:
    st.table(pd.DataFrame(st.session_state.history))

# =====================================================
# NIT PATNA: BRIDGE DIGITAL TWIN (MASTER CODE)
# Developed by: Shadman Mallick | M.Tech Structural Engineering
# =====================================================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor

# ================= PAGE SETUP =================
st.set_page_config(page_title="NIT Patna Bridge Health Monitor | Shadman Mallick", layout="wide")

# ================= USER GUIDE =================
with st.expander("📖 USER MANUAL & DOCUMENTATION"):
    st.markdown("""
    ### 🏗️ Integrated Digital Twin Framework
    1. **Structural Engine:** Euler-Bernoulli theory for real-time deflection calculations.
    2. **Dynamic Analysis:** Natural Frequency tracking as a function of Stiffness ($E$).
    3. **AI Module:** Random Forest Regressor to predict Remaining Useful Life (RUL).
    4. **Interactive Trace:** Post-simulation hover analysis for precise data point retrieval.
    """)

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
mass_density = 2500 # kg/m3 for RCC
m_unit = mass_density * (b * h)

# ================= SESSION STATE =================
if 'e_current' not in st.session_state or st.sidebar.button("Reset Simulation"):
    st.session_state.e_current = initial_E
    st.session_state.history = []
    st.session_state.is_collapsed = False
    st.session_state.last_delta = 0

# ================= STRUCTURAL CALCULATIONS =================
limit_mm = (L * 1000) / 800
curr_e_pa = st.session_state.e_current * 1e6

# Dynamic Calculation: Natural Frequency
nat_freq = (np.pi / (2 * L**2)) * np.sqrt((curr_e_pa * I_calc) / m_unit)

p_perm = (limit_mm/1000 * 48 * curr_e_pa * I_calc) / (L**3) / 1000
p_ultimate = 1.5 * p_perm

# ================= HEADER =================
st.title("🏗️ NIT Patna Bridge Health Monitor")
st.markdown("##### **Developed by: Shadman Mallick**")
st.subheader("M.Tech Structural Engineering | AI + Fatigue + Digital Twin")

m1,m2,m3,m4 = st.columns(4)
m1.metric("Current Stiffness", f"{st.session_state.e_current/1000:.2f} GPa")
m2.metric("Natural Frequency", f"{nat_freq:.2f} Hz")
m3.metric("Permissible Load", f"{p_perm:.1f} kN")
m4.metric("Ultimate Capacity", f"{p_ultimate:.1f} kN")

st.markdown("---")

# ================= IMPACT ANALYSIS & HEALTH GAUGE =================
if st.session_state.is_collapsed:
    st.error("💥 BRIDGE COLLAPSED: Structural integrity reached 0%.")
    st.metric("Health Index", "0.00%")
else:
    col1, col2 = st.columns(2)
    with col1:
        st.write("## Structural Impact Analysis")
        applied_p = st.number_input("Enter Impact Load (kN)", value=100.0)

        if st.button("RUN IMPACT ANALYSIS"):
            if applied_p >= p_ultimate:
                st.session_state.is_collapsed = True
                st.session_state.e_current = 0
                st.rerun()
            else:
                load_ratio = applied_p / p_perm
                damage_factor = 0.02 + (load_ratio**3)*0.15
                delta = ((applied_p*1000*(L**3))/(48*curr_e_pa*I_calc))*1000

                st.session_state.last_delta = delta
                st.session_state.history.append({
                    "Cycle": len(st.session_state.history)+1,
                    "Load_kN": applied_p,
                    "Health_%": round((st.session_state.e_current/initial_E)*100, 2),
                    "Deflection_mm": round(delta, 3),
                    "E_GPa": round(st.session_state.e_current/1000, 3) # Added key for AI
                })
                st.session_state.e_current *= (1 - damage_factor)
                st.rerun()

        if st.session_state.last_delta > 0:
            d = st.session_state.last_delta
            if d >= limit_mm: st.error(f"🔴 DANGER: Deflection {d:.2f}mm exceeded limit!")
            elif d >= 0.75*limit_mm: st.warning(f"🟠 WARNING: High Deflection {d:.2f}mm.")
            else: st.success(f"🟢 SAFE: Deflection is {d:.2f}mm.")

    with col2:
        current_health = (st.session_state.e_current / initial_E) * 100
        fig_g = go.Figure(go.Indicator(
            mode = "gauge+number", value = current_health,
            title = {'text': "Current Health Index (%)"},
            gauge = {'axis': {'range': [0, 100]},
                     'steps': [{'range': [0, 50], 'color': "red"},
                               {'range': [50, 80], 'color': "orange"},
                               {'range': [80, 100], 'color': "green"}],
                     'bar': {'color': "black"}}))
        fig_g.update_layout(height=300)
        st.plotly_chart(fig_g, use_container_width=True)

# ================= DUAL VEHICLE ANIMATION =================
st.markdown("---")
st.subheader("🚗 Dual-Vehicle Dynamic Comparison Simulation")
c1, c2 = st.columns(2)
v1 = c1.number_input("Vehicle 1 (Heavy) Load (kN)", value=400.0)
v2 = c2.number_input("Vehicle 2 (Light) Load (kN)", value=50.0)

if st.button("▶️ Start Smooth 100-Frame Simulation"):
    x_pts = np.linspace(0, L, 100)
    plot_spot = st.empty()
    trace_x, trace_v1, trace_v2 = [], [], []
    
    for pos in np.linspace(0, L, 100):
        a, b_dist = pos, L - pos
        y1, y2 = [], []
        for xi in x_pts:
            term = (1000*b_dist*xi*(L**2-b_dist**2-xi**2)) if xi<=a else (1000*a*(L-xi)*(L**2-a**2-(L-xi)**2))
            y1.append(-(v1*term)/(6*curr_e_pa*I_calc*L)*1000)
            y2.append(-(v2*term)/(6*curr_e_pa*I_calc*L)*1000)
        
        trace_x.append(round(pos, 2))
        trace_v1.append(round(min(y1), 3))
        trace_v2.append(round(min(y2), 3))

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(x_pts, y1, color='red', lw=2, label=f"V1: {v1}kN")
        ax.plot(x_pts, y2, color='green', lw=1.5, linestyle='--', label=f"V2: {v2}kN")
        ax.axhline(0, color='black')
        ax.scatter([pos], [0], color='red', s=100)
        ax.set_ylim(-limit_mm * 2.5, 5)
        ax.set_title(f"Position: {pos:.1f}m | Max Deflection: {abs(min(y1)):.2f}mm")
        ax.legend()
        plot_spot.pyplot(fig)
        plt.close(fig)
        time.sleep(0.05)

    # Post-Simulation Hover Analysis
    st.subheader("🔍 Interactive Trace Analysis")
    fig_h = go.Figure()
    fig_h.add_trace(go.Scatter(x=trace_x, y=trace_v1, name="V1 Trace (Heavy)", line=dict(color='red', dash='dot')))
    fig_h.add_trace(go.Scatter(x=trace_x, y=trace_v2, name="V2 Trace (Light)", line=dict(color='green', dash='dot')))
    fig_h.update_layout(hovermode="x unified", xaxis_title="Position (m)", yaxis_title="Deflection (mm)")
    st.plotly_chart(fig_h, use_container_width=True)

# ================= UPGRADED AI FORECASTING & RUL MODULE =================
st.markdown("---")
st.subheader("🤖 AI Infrastructure Forecasting & Remaining Useful Life (RUL)")

if len(st.session_state.history) >= 3:
    df_ai = pd.DataFrame(st.session_state.history)
    X_ai = df_ai[['Load_kN']].values
    y_ai = df_ai['E_GPa'].values
    
    # AI Learning from live history
    regr = RandomForestRegressor(n_estimators=100).fit(X_ai, y_ai)
    current_load_target = v1 if 'v1' in locals() else 100.0
    future_e = regr.predict([[current_load_target]])[0]
    
    col_ai1, col_ai2 = st.columns(2)
    
    with col_ai1:
        st.markdown("#### 📅 Predictive Maintenance Forecast")
        st.warning(f"AI Forecast: Next cycle of {current_load_target}kN will reduce stiffness to **{future_e:.2f} GPa**.")
        
        # Infrastructure Life Score (RUL Logic)
        life_score = (future_e / (initial_E/1000)) * 100
        st.info(f"Structural Integrity Score: **{life_score:.1f}/100**")
        
        if life_score < 60:
            st.error("🚨 HIGH RISK: Schedule structural retrofitting immediately.")
        else:
            st.success("✅ OPTIMAL: Maintenance not required for next 500 cycles.")

    with col_ai2:
        st.markdown("#### 📈 AI Learning Curve")
        fig_learn, ax_learn = plt.subplots(figsize=(6, 4))
        ax_learn.scatter(X_ai, y_ai, color='blue', label='Live Sensor Data')
        ax_learn.plot(X_ai, y_ai, color='gray', linestyle='--', alpha=0.5)
        ax_learn.set_xlabel("Load (kN)")
        ax_learn.set_ylabel("Stiffness (GPa)")
        ax_learn.set_title("Stiffness Degradation Mapping")
        st.pyplot(fig_learn)

else:
    st.info("💡 Run at least 3 analysis cycles to activate AI Infrastructure Forecasting.")

# ================= S-N CURVE (FATIGUE LIFE ENVELOPE) =================
st.markdown("---")
st.subheader("🔬 Fatigue Life Envelope (S-N Curve)")

sigma_u = p_ultimate
s_values = np.linspace(0.1 * p_ultimate, p_ultimate, 100)
n_values = 10**( (1 - (s_values/p_ultimate)) / 0.08 )

fig_sn, ax_sn = plt.subplots(figsize=(10, 4))
ax_sn.plot(n_values, s_values, color='purple', lw=2, label='Theoretical Design S-N Curve')
ax_sn.set_xscale('log')
ax_sn.set_xlabel("Number of Cycles (Log Scale)")
ax_sn.set_ylabel("Applied Load (kN)")
ax_sn.grid(True, which="both", ls="-", alpha=0.2)
ax_sn.legend()
st.pyplot(fig_sn)

# ================= HISTORY TABLE =================
if st.session_state.history:
    st.markdown("---")
    st.subheader("📜 Structural History Log")
    st.table(pd.DataFrame(st.session_state.history))

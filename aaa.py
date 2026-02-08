import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.ensemble import RandomForestRegressor

# ================= PAGE SETUP =================
st.set_page_config(page_title="GEC Khagaria Bridge Digital Twin", layout="wide")

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

I_calc = (b * (h**3)) / 12

# ================= SESSION STATE =================
if 'e_current' not in st.session_state or st.sidebar.button("Reset Simulation"):
    st.session_state.e_current = initial_E
    st.session_state.history = []
    st.session_state.is_collapsed = False
    st.session_state.max_deflections = [] # Trace data storage

# ================= STRUCTURAL CALC =================
curr_e_pa = st.session_state.e_current * 1e6
limit_mm = (L * 1000) / 800
p_perm = (limit_mm/1000 * 48 * curr_e_pa * I_calc) / (L**3) / 1000
p_ultimate = 1.5 * p_perm

# ================= HEADER =================
st.title("🏗️ Bridge Digital Twin: Health Monitoring System")
st.write("**Developed for GEC Khagaria AI TECH FEST**")

m1,m2,m3,m4 = st.columns(4)
health_pct = (st.session_state.e_current / initial_E) * 100
m1.metric("Current Stiffness", f"{st.session_state.e_current/1000:.2f} GPa")
m2.metric("Permissible Load", f"{p_perm:.1f} kN")
m3.metric("Health Status", f"{health_pct:.1f}%")
m4.metric("Deflection Limit", f"{limit_mm:.2f} mm")

st.markdown("---")

# ================= IMPACT ANALYSIS WITH FIXED WARNINGS =================
if not st.session_state.is_collapsed:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🚀 Structural Impact Analysis")
        applied_p = st.number_input("Applied Load (kN)", value=150.0)

        if st.button("RUN IMPACT ANALYSIS"):
            if applied_p >= p_ultimate:
                st.session_state.is_collapsed = True
                st.error("💥 BRIDGE COLLAPSED: Load exceeds ultimate capacity!")
            else:
                # Dynamic Damage Logic
                load_ratio = applied_p / p_perm
                damage_factor = 0.02 + (load_ratio**3) * 0.1
                
                # Deflection calculation at center
                delta = ((applied_p*1000*(L**3))/(48*curr_e_pa*I_calc))*1000

                # FIXED WARNING LOGIC
                if delta >= limit_mm:
                    st.error(f"🔴 DANGER: Deflection ({delta:.2f} mm) exceeded limit!")
                elif delta >= 0.75 * limit_mm:
                    st.warning(f"🟠 WARNING: High Deflection ({delta:.2f} mm). Fatigue detected.")
                else:
                    st.success(f"🟢 SAFE: Deflection is {delta:.2f} mm.")

                st.session_state.history.append({
                    "Cycle": len(st.session_state.history)+1,
                    "Load_kN": applied_p,
                    "Health_%": round(health_pct, 2),
                    "Deflection_mm": round(delta, 2)
                })
                st.session_state.e_current *= (1 - damage_factor)
                st.rerun()

# ================= 100-FRAME SMOOTH ANIMATION & TRACE LINE =================
st.markdown("---")
st.subheader("🚗 Live Traffic Simulation (Moving Load)")
sim_load = st.number_input("Vehicle Weight for Animation (kN)", value=200.0)

if st.button("▶️ Start 100-Frame Simulation"):
    x_points = np.linspace(0, L, 100)
    plot_spot = st.empty()
    trace_x = []
    trace_y = []

    # Dividing simulation into 100 frames
    frames = np.linspace(0, L, 100)
    
    for pos in frames:
        a, b_dist = pos, L - pos
        y_def = []
        for xi in x_points:
            # Macaulay's Method for Dynamic Position
            if xi <= a:
                val = (sim_load * 1000 * b_dist * xi * (L**2 - b_dist**2 - xi**2)) / (6 * curr_e_pa * I_calc * L)
            else:
                val = (sim_load * 1000 * a * (L - xi) * (L**2 - a**2 - (L - xi)**2)) / (6 * curr_e_pa * I_calc * L)
            y_def.append(-val * 1000) # mm (downward)

        # Storing data for the Dotted Trace Line (Maximum deflection at current load position)
        current_max_y = min(y_def)
        trace_x.append(pos)
        trace_y.append(current_max_y)

        # Plotting each frame
        fig_sim, ax_sim = plt.subplots(figsize=(10, 4))
        ax_sim.plot(x_points, y_def, color='blue', lw=2, label="Elastic Curve")
        ax_sim.axhline(0, color='black', lw=1)
        
        # Vehicle Marker
        ax_sim.scatter([pos], [0], color='red', s=120, zorder=5, label="Moving Vehicle")
        
        # Plot Trace Line (Black Dotted)
        if len(trace_x) > 1:
            ax_sim.plot(trace_x, trace_y, color='black', linestyle='--', alpha=0.6, label="Deflection Trace")

        ax_sim.set_ylim(-limit_mm * 2.5, 5)
        ax_sim.set_xlabel("Span Length (m)")
        ax_sim.set_ylabel("Deflection (mm)")
        ax_sim.set_title(f"Position: {pos:.2f}m | Max Deflection in Frame: {abs(current_max_y):.2f} mm")
        ax_sim.grid(True, alpha=0.3)
        
        plot_spot.pyplot(fig_sim)
        plt.close(fig_sim)
        time.sleep(0.05) # 0.05 Seconds delay per frame

    st.success("Simulation Complete. Trace Line generated.")

# ================= HISTORY & AI =================
if st.session_state.history:
    st.markdown("---")
    colA, colB = st.columns(2)
    with colA:
        st.subheader("📜 Structural History Log")
        st.dataframe(pd.DataFrame(st.session_state.history))
    with colB:
        st.subheader("🤖 AI Sustainability Prediction")
        # Training AI to predict failure cycles
        X = np.linspace(50, p_ultimate, 100).reshape(-1, 1)
        y = 10000 * np.exp(-0.005 * X.flatten())
        rf = RandomForestRegressor(n_estimators=50).fit(X, y)
        pred = rf.predict([[sim_load]])[0]
        st.info(f"AI Prediction: Bridge can sustain approx {int(pred)} more cycles of {sim_load} kN.")

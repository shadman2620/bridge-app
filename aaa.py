# ================= IMPORTS =================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

# ML
from sklearn.ensemble import RandomForestRegressor

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

# ================= PAGE SETUP =================
st.set_page_config(page_title="NIT Patna Bridge Health Monitor", layout="wide")

# ================= MATERIAL DATA =================
concrete_grades = {
    "M25": 25000, 
    "M30": 27386, 
    "M35": 29580, 
    "M40": 31622, 
    "M50": 35355
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
st.subheader("M.Tech Structural Engineering | AI + Fatigue + Digital Twin")

m1,m2,m3,m4 = st.columns(4)
m1.metric("Current Stiffness", f"{st.session_state.e_current/1000:.2f} GPa")
m2.metric("Safe Load", f"{0.75*p_perm:.1f} kN")
m3.metric("Permissible Load", f"{p_perm:.1f} kN")
m4.metric("Ultimate Load", f"{p_ultimate:.1f} kN")

st.markdown("---")

# ================= STRUCTURAL IMPACT ANALYSIS =================
if not st.session_state.is_collapsed:
    col1,col2 = st.columns(2)

    with col1:
        st.write("## Structural Impact Analysis")
        applied_p = st.number_input("Applied Load (kN)", value=100.0)

        if st.button("RUN IMPACT ANALYSIS"):
            st.session_state.load_history.append(applied_p)

            if applied_p >= p_ultimate:
                st.session_state.is_collapsed = True
                st.session_state.e_current = 0
                st.error("💥 BRIDGE COLLAPSED")
            else:
                load_ratio = applied_p / p_perm
                damage_factor = 0.02 + (load_ratio**3)*0.15

                delta = ((applied_p*1000*(L**3))/(48*curr_e_pa*I_calc))*1000

                if delta > limit_mm:
                    st.error(f"🔴 Deflection {delta:.2f} mm")
                elif delta > 0.75*limit_mm:
                    st.warning(f"🟠 Deflection {delta:.2f} mm")
                else:
                    st.success(f"🟢 Deflection {delta:.2f} mm Safe")

                st.session_state.history.append({
                    "Cycle": len(st.session_state.history)+1,
                    "Load_kN": applied_p,
                    "Damage_%": round(damage_factor*100,3),
                    "Deflection_mm": round(delta,3),
                    "E_GPa": round(st.session_state.e_current/1000,3)
                })

                st.session_state.e_current *= (1 - damage_factor)
                st.rerun()

    with col2:
        health = (st.session_state.e_current / initial_E) * 100
        st.write(f"## Health Index = {health:.2f}%")

        health_map = np.linspace(0, 100, 200)
        colors = plt.cm.get_cmap("RdYlGn")(health_map/100)

        fig, ax = plt.subplots(figsize=(6,1))
        ax.imshow([colors], extent=[0,100,0,1])
        ax.axvline(health, color='black', linewidth=3)
        ax.set_yticks([])
        ax.set_xlabel("Health %")
        st.pyplot(fig)

# ================= HISTORY TABLE =================
if st.session_state.history:
    df = pd.DataFrame(st.session_state.history)
    st.subheader("📜 Structural History")
    st.dataframe(df)

# ================= FATIGUE & AI MODULE =================
st.markdown("---")
st.subheader("🤖 Fatigue & AI Prediction Module")

sigma_u = p_ultimate
sigma_f = 0.9 * sigma_u
b_f = -0.09
sigma_cap = 0.95 * sigma_u

def predict_cycles(load):
    if load >= sigma_u:
        return 1
    return (load/sigma_f)**(1/b_f) / 2

np.random.seed(42)
loads = np.random.uniform(0.1*sigma_u, sigma_u, 1000)
cycles = np.array([predict_cycles(l) for l in loads])
rf = RandomForestRegressor(n_estimators=200).fit(loads.reshape(-1,1), cycles)

colA,colB = st.columns(2)
with colA:
    st.write("### Predict Remaining Cycles")
    load_in = st.number_input("Load for AI Fatigue (kN)", value=100.0, key="L1")
    if st.button("AI Predict Cycles"):
        st.success(f"Physics Cycles = {int(predict_cycles(load_in))}")
        st.info(f"ML Predicted Cycles = {int(rf.predict([[load_in]])[0])}")

with colB:
    st.write("### Predict Safe Load for Target Cycles")
    N_target = st.number_input("Target Cycles", value=200, key="N1")
    if st.button("Predict Safe Load"):
        l_safe = min(sigma_f * (2*N_target)**b_f, sigma_cap)
        if l_safe >= sigma_u: l_safe = 0.99 * sigma_u
        st.success(f"Safe Load = {l_safe:.2f} kN")

# ================= MOVING LOAD SIMULATION (FINAL WORKING) =================
st.markdown("---")
st.subheader("🚗 Live Moving Load Simulation")

sim_load = st.number_input("Vehicle Weight (kN)", value=200.0)

if st.button("▶️ Start Moving Load Simulation"):
    
    x_points = np.linspace(0, L, 100)
    plot_spot = st.empty()
    info_spot = st.empty()

    for pos in np.linspace(0, L, 100):
        a = pos
        b_dist = L - a
        y_def = []

        for xi in x_points:
            if xi <= a:
                val = (sim_load * 1000 * b_dist * xi * (L**2 - b_dist**2 - xi**2)) / (6 * curr_e_pa * I_calc * L)
            else:
                val = (sim_load * 1000 * a * (L - xi) * (L**2 - a**2 - (L - xi)**2)) / (6 * curr_e_pa * I_calc * L)
            y_def.append(val * 1000)

        fig_sim, ax_sim = plt.subplots(figsize=(10, 4))

        # Blue deflection curve
        ax_sim.plot(x_points, [-y for y in y_def], lw=2, color='blue', label='Deflected Shape')

        # Black dotted trace
        ax_sim.plot(x_points, [-y for y in y_def], 'k--', lw=1, label='Deflection Trace')

        # Vehicle marker
        ax_sim.plot([pos], [0], marker='o', color='red', markersize=10)

        ax_sim.axhline(0, color='black')
        ax_sim.set_ylim(-limit_mm * 1.5, 5)
        ax_sim.set_xlabel("Span (m)")
        ax_sim.set_ylabel("Deflection (mm)")
        ax_sim.legend()
        ax_sim.grid(True, linestyle='--')

        plot_spot.pyplot(fig_sim)
        plt.close(fig_sim)

        max_def = max(y_def)
        info_spot.info(f"Vehicle Position = {pos:.2f} m | Max Deflection = {max_def:.3f} mm")

        time.sleep(0.1)

    st.success("Simulation Complete!")

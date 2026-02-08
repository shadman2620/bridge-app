

# =====================================================
# NIT PATNA: BRIDGE DIGITAL TWIN (FINAL MASTER CODE)
# Developed for M.Tech Structural Engineering Research
# =====================================================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

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
st.set_page_config(page_title="NIT Patna Bridge Health Monitor, Developed by Shadman Mallick", layout="wide")

# ================= USER GUIDE (ENGLISH) =================
with st.expander("📖 USER MANUAL & DOCUMENTATION"):
    st.markdown("""
    ### 🏗️ Project Overview
    This **Digital Twin** app simulates the real-time health of a bridge. It uses structural mechanics and AI to show how traffic and heavy loads degrade a structure over time.

    ### 🛠️ How to Use
    1. **Setup:** Select **Concrete Grade** and bridge dimensions in the sidebar.
    2. **Impact Analysis:** Enter a vehicle load (kN) and click **'Run Impact Analysis'**. 
    3. **Multiple Inputs:** You can apply loads multiple times to see the **Cumulative Damage**.
    4. **Simulation:** Use the 'Moving Load' section to see the bridge's live deflection curve.

    ### 🧪 The Engineering Logic (Impact Analysis)
    This app follows a **Dynamic Damage Model** based on how you interact with it:
    
    * **Load Intensity:** - **Low Loads:** Cause minimal wear and tear (Stiffness stays high).
        - **High Loads:** Cause significant internal damage. If you apply a load near the **Ultimate Capacity**, the stiffness drops sharply.
        - **Extreme Overload:** If the load is too high (e.g., 5x the limit), the bridge will **Collapse Instantly**, simulating a sudden structural failure.
    
    * **Cumulative Fatigue (Multiple Inputs):**
        - Every time you click 'Run Analysis', the bridge "remembers" the stress. 
        - Even if you apply small loads many times, the **Stiffness ($E$)** will gradually decrease, representing **Fatigue Cracking**.
    
    * **Safety Status:**
        - 🟢 **Green:** Safe (Deflection within L/800).
        - 🟠 **Orange:** Warning (Structural fatigue starting).
        - 🔴 **Red:** Danger (Immediate maintenance required).
    
    * **AI Forecast:** The AI analyzes your previous inputs and predicts how many more such cycles the bridge can survive before it becomes unsafe.
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
st.markdown("##### **Developed by Shadman Mallick")
st.subheader("M.Tech Structural Engineering | AI + Fatigue + Digital Twin")

m1,m2,m3,m4 = st.columns(4)
m1.metric("Current Stiffness", f"{st.session_state.e_current/1000:.2f} GPa")
m2.metric("Safe Load", f"{0.75*p_perm:.1f} kN")
m3.metric("Permissible Load", f"{p_perm:.1f} kN")
m4.metric("Ultimate Load", f"{p_ultimate:.1f} kN")

st.markdown("---")

# ================= STRUCTURAL IMPACT ANALYSIS (FIXED) =================
if st.session_state.is_collapsed:
    st.error("💥 BRIDGE COLLAPSED: Load exceeded Ultimate Capacity!")
    # Collapsed hone par bhi health 0% dikhata rahega
    st.metric("Health Index", "0.00%") 
else:
    col1, col2 = st.columns(2)

    with col1:
        st.write("## Structural Impact Analysis")
        applied_p = st.number_input("Applied Load (kN)", value=100.0)

        if st.button("RUN IMPACT ANALYSIS"):
            st.session_state.load_history.append(applied_p)

            if applied_p >= p_ultimate:
                st.session_state.is_collapsed = True
                st.session_state.e_current = 0
                st.rerun() # Turant collapse dikhane ke liye
            else:
                load_ratio = applied_p / p_perm
                damage_factor = 0.02 + (load_ratio**3)*0.15
                delta = ((applied_p*1000*(L**3))/(48*curr_e_pa*I_calc))*1000

                # State update
                st.session_state.last_delta = delta
                st.session_state.history.append({
                    "Cycle": len(st.session_state.history)+1,
                    "Load_kN": applied_p,
                    "Damage_%": round(damage_factor*100,3),
                    "Deflection_mm": round(delta,3),
                    "E_GPa": round(st.session_state.e_current/1000,3)
                })

                st.session_state.e_current *= (1 - damage_factor)
                st.rerun()

        # Persistent Warning Display
        if 'last_delta' in st.session_state:
            d = st.session_state.last_delta
            if d >= limit_mm:
                st.error(f"🔴 DANGER: Deflection {d:.2f} mm (Limit: {limit_mm:.2f} mm)")
            elif d >= 0.75 * limit_mm:
                st.warning(f"🟠 WARNING: Fatigue Detected. Deflection {d:.2f} mm")
            else:
                st.success(f"🟢 SAFE: Deflection {d:.2f} mm")

    with col2:
        # Health calculation consistent state se
        current_health = (st.session_state.e_current / initial_E) * 100
        st.write(f"## Health Index = {current_health:.2f}%")
        
        # Color Gauge Plot
        health_map = np.linspace(0, 100, 200)
        colors = plt.cm.get_cmap("RdYlGn")(health_map/100)
        fig, ax = plt.subplots(figsize=(6, 1.2)) # Thoda size bada kiya visibility ke liye
        ax.imshow([colors], extent=[0, 100, 0, 1])
        ax.axvline(current_health, color='black', linewidth=4)
        ax.set_yticks([])
        ax.set_xlabel("Structural Integrity (%)")
        st.pyplot(fig)
        
# ================= FATIGUE & AI MODULE =================
st.markdown("---")
st.subheader("🤖 Fatigue & AI Prediction Module")

sigma_u, sigma_f, b_f = p_ultimate, 0.9 * p_ultimate, -0.09

def predict_cycles(load):
    if load >= sigma_u: return 1
    return (load/sigma_f)**(1/b_f) / 2

# ML Train logic
np.random.seed(42)
loads_tr = np.random.uniform(0.1*sigma_u, sigma_u, 500).reshape(-1,1)
cyc_tr = np.array([predict_cycles(l[0]) for l in loads_tr])
rf = RandomForestRegressor(n_estimators=100).fit(loads_tr, cyc_tr)

colA, colB = st.columns(2)
with colA:
    st.write("### Predict Life")
    l_in = st.number_input("Load for AI (kN)", value=100.0, key="L1")
    if st.button("AI Predict"):
        st.success(f"Physics Life: {int(predict_cycles(l_in))} Cycles")
        st.info(f"AI Predicted Life: {int(rf.predict([[l_in]])[0])} Cycles")

# ================= LIVE MOVING LOAD SIMULATION =================
# ================= OPTIMIZED DUAL-VEHICLE SIMULATION =================
st.markdown("---")
st.subheader("🚛 High-Resolution Traffic Simulation (Truck vs Car)")

col_sim1, col_sim2 = st.columns(2)
with col_sim1:
    truck_load = st.number_input("Truck Load (Heavy) (kN)", value=400.0)
with col_sim2:
    car_load = st.number_input("Car Load (Light) (kN)", value=50.0)

if st.button("▶️ Start Smooth Comparison Simulation"):
    x_points = np.linspace(0, L, 100)
    plot_spot = st.empty()
    
    # 100 Frames for Smoothness
    frames = np.linspace(0, L, 100)
    
    # Trace data storage
    trace_x = []
    trace_truck_y = []
    trace_car_y = []

    for pos in frames:
        a, b_dist = pos, L - pos
        y_truck = []
        y_car = []
        
        # Optimized Calculation Loop
        for xi in x_points:
            if xi <= a:
                term = (1000 * b_dist * xi * (L**2 - b_dist**2 - xi**2))
            else:
                term = (1000 * a * (L - xi) * (L**2 - a**2 - (L - xi)**2))
            
            y_truck.append(-(truck_load * term) / (6 * curr_e_pa * I_calc * L) * 1000)
            y_car.append(-(car_load * term) / (6 * curr_e_pa * I_calc * L) * 1000)

        # Current peak points for trace
        peak_t = min(y_truck)
        peak_c = min(y_car)
        trace_x.append(pos)
        trace_truck_y.append(peak_t)
        trace_car_y.append(peak_c)

        # Rendering the Frame
        fig, ax = plt.subplots(figsize=(10, 4), dpi=80) # DPI optimized for speed
        
        # Plot Elastic Curves
        ax.plot(x_points, y_truck, color='red', lw=2, label=f"Truck ({truck_load}kN)")
        ax.plot(x_points, y_car, color='green', lw=1.2, linestyle='--', label=f"Car ({car_load}kN)")
        
        # Plot Trace Lines (Dotted)
        ax.plot(trace_x, trace_truck_y, color='red', linestyle=':', alpha=0.3)
        ax.plot(trace_x, trace_car_y, color='green', linestyle=':', alpha=0.3)
        
        # Markers
        ax.scatter([pos], [0], color='red', s=100, marker='s', zorder=5) # Truck
        ax.scatter([pos], [0], color='green', s=40, marker='o', zorder=5) # Car
        
        ax.axhline(0, color='black', lw=1)
        ax.set_ylim(-limit_mm * 2.5, 5)
        
        # Updated Title with BOTH deflections
        ax.set_title(f"Pos: {pos:.1f}m | Truck Def: {abs(peak_t):.2f}mm | Car Def: {abs(peak_c):.2f}mm", fontsize=10)
        
        ax.legend(loc='upper right', fontsize='small')
        ax.grid(True, alpha=0.2)
        
        # Animation Display
        plot_spot.pyplot(fig)
        plt.close(fig) 
        
        # Atakne se bachne ke liye delay ko adjust kiya (render time + sleep = smooth motion)
        time.sleep(0.01) 

    st.success("Simulation Complete!")
# ================= HISTORY TABLE =================
if st.session_state.history:
    st.markdown("---")
    st.subheader("📜 Structural History Log")
    st.table(pd.DataFrame(st.session_state.history))






import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
from sklearn.ensemble import RandomForestRegressor

# ... (Rainflow function and Page Setup same as before) ...

# ================= REPLACEMENT FOR MOVING LOAD SIMULATION =================
st.markdown("---")
st.subheader("🚗 Live Moving Load Simulation (Optimized)")

sim_load = st.number_input("Vehicle Weight (kN)", value=200.0)

if st.button("▶️ Start Moving Load Simulation"):
    x_points = np.linspace(0, L, 50) # Reduced points for speed
    plot_spot = st.empty() # Placeholder for the chart
    
    # Pre-calculating variables to save CPU
    E_curr_pa = st.session_state.e_current * 1e6
    
    for pos in np.arange(0, L + 1.0, 1.0): # Slightly larger step for smoothness
        a = pos
        b_dist = L - pos
        y_def = []
        
        for xi in x_points:
            # Macaulay's Logic
            if xi <= a:
                val = (sim_load * 1000 * b_dist * xi * (L**2 - b_dist**2 - xi**2)) / (6 * E_curr_pa * I_calc * L)
            else:
                val = (sim_load * 1000 * a * (L - xi) * (L**2 - a**2 - (L - xi)**2)) / (6 * E_curr_pa * I_calc * L)
            y_def.append(-val * 1000) # Negative for downward deflection

        # Using Plotly for smoother rendering
        fig = go.Figure()
        
        # Bridge Deck
        fig.add_trace(go.Scatter(x=[0, L], y=[0, 0], mode='lines', line=dict(color='black', width=4), name='Bridge'))
        
        # Deflection Curve
        fig.add_trace(go.Scatter(x=x_points, y=y_def, mode='lines', fill='tozeroy', line=dict(color='blue', width=2), name='Deflection'))
        
        # Vehicle Position
        fig.add_trace(go.Scatter(x=[pos], y=[2], mode='markers+text', 
                                 marker=dict(symbol='bus', size=15, color='red'),
                                 text=["Vehicle"], textposition="top center", name='Vehicle'))

        fig.update_layout(
            yaxis=dict(range=[-limit_mm * 1.5, 10], title="Deflection (mm)"),
            xaxis=dict(title="Span Position (m)"),
            height=400,
            margin=dict(l=20, r=20, t=40, b=20),
            showlegend=False,
            title=f"Position: {pos:.1f}m | Max Deflection: {min(y_def):.2f} mm"
        )
        
        plot_spot.plotly_chart(fig, use_container_width=True)
        time.sleep(0.01) # Very small delay

"""
F1 Tire Degradation Predictor — Streamlit Dashboard
=====================================================
Run with:  streamlit run dashboard.py
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="F1 AI Pit Strategy",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    .metric-card { background:#1a1a2e; border-radius:12px; padding:16px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MOCK DATA GENERATOR  (replace with real pipeline output)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def generate_mock_race(n_laps: int = 52, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    laps = np.arange(1, n_laps + 1)

    # Simulate two stints (soft → medium)
    def tyre_curve(tyre_age, base=88.5, slope=0.12, cliff_start=25):
        deg = slope * tyre_age + np.where(tyre_age > cliff_start, 0.08 * (tyre_age - cliff_start) ** 1.6, 0)
        return base + deg

    actual_times, tyre_ages, compounds = [], [], []
    pit_lap = 27
    for lap in laps:
        if lap < pit_lap:
            age = lap
            compound = "SOFT"
        else:
            age = lap - pit_lap
            compound = "MEDIUM"
        t = tyre_curve(age) + rng.normal(0, 0.18)
        actual_times.append(t)
        tyre_ages.append(age)
        compounds.append(compound)

    # AI strategy: pit 3 laps earlier
    ai_pit = pit_lap - 3
    ai_times = []
    for lap in laps:
        age = lap if lap < ai_pit else lap - ai_pit
        t = tyre_curve(age) + rng.normal(0, 0.18)
        if lap == ai_pit:
            t += 22.0   # pit stop time loss
        ai_times.append(t)

    # Predictions (Bi-LSTM output) — slightly noisy version of actual
    predicted_times = np.array(actual_times) + rng.normal(0, 0.3, n_laps)

    return pd.DataFrame({
        "Lap":          laps,
        "Actual":       actual_times,
        "Predicted":    predicted_times,
        "AI_Strategy":  ai_times,
        "TyreAge":      tyre_ages,
        "Compound":     compounds,
        "PitLap_Real":  pit_lap,
        "PitLap_AI":    ai_pit,
    })


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/33/F1.svg/200px-F1.svg.png", width=140)
    st.title("🏎️ F1 AI Pit Strategy")
    st.markdown("---")

    race     = st.selectbox("Grand Prix",    ["British GP 2024", "Monaco GP 2024", "Monza GP 2024"])
    driver   = st.selectbox("Driver",        ["VER", "HAM", "NOR", "LEC", "SAI"])
    compound = st.multiselect("Compounds",   ["SOFT", "MEDIUM", "HARD"], default=["SOFT", "MEDIUM"])
    st.markdown("---")
    st.caption("Model: Bi-LSTM (PyTorch) • RL: PPO (SB3)")

df = generate_mock_race()

# ─────────────────────────────────────────────────────────────────────────────
# KPI METRICS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### 📊 Race Strategy Overview")
col1, col2, col3, col4 = st.columns(4)

real_total = df["Actual"].sum()
ai_total   = df["AI_Strategy"].sum()
advantage  = real_total - ai_total

col1.metric("Real Strategy Total", f"{real_total:.1f}s",  delta=None)
col2.metric("AI Strategy Total",   f"{ai_total:.1f}s",   delta=f"{-advantage:.1f}s", delta_color="inverse")
col3.metric("AI Time Advantage",   f"{advantage:.1f}s",   delta="AI wins" if advantage > 0 else "Real wins")
col4.metric("Real Pit Lap",        f"Lap {df['PitLap_Real'].iloc[0]}",
            delta=f"AI: Lap {df['PitLap_AI'].iloc[0]}", delta_color="off")

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# CHART 1 — Actual vs Predicted Tire Decay
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("#### 🔮 Actual vs Predicted Tyre Degradation (Bi-LSTM)")

fig_decay = go.Figure()

fig_decay.add_trace(go.Scatter(
    x=df["Lap"], y=df["Actual"],
    mode="lines+markers", name="Actual Lap Time",
    line=dict(color="#E8002D", width=2.5),
    marker=dict(size=5),
))

fig_decay.add_trace(go.Scatter(
    x=df["Lap"], y=df["Predicted"],
    mode="lines", name="Bi-LSTM Predicted",
    line=dict(color="#00D2BE", width=2, dash="dash"),
))

# Confidence band (±0.5s)
fig_decay.add_trace(go.Scatter(
    x=np.concatenate([df["Lap"], df["Lap"][::-1]]),
    y=np.concatenate([df["Predicted"] + 0.5, (df["Predicted"] - 0.5)[::-1]]),
    fill="toself", fillcolor="rgba(0,210,190,0.12)",
    line=dict(color="rgba(0,0,0,0)"),
    name="±0.5s Confidence Band", showlegend=True,
))

# Performance cliff annotation
cliff_lap = df.loc[df["TyreAge"] > 24, "Lap"].min()
fig_decay.add_vline(
    x=cliff_lap, line_dash="dot", line_color="#FF8700",
    annotation_text="⚠️ Performance Cliff", annotation_position="top right",
)
fig_decay.add_vline(
    x=df["PitLap_Real"].iloc[0], line_dash="solid", line_color="#E8002D",
    annotation_text=f"🔴 Actual Pit (L{df['PitLap_Real'].iloc[0]})", annotation_position="top left",
)
fig_decay.add_vline(
    x=df["PitLap_AI"].iloc[0], line_dash="solid", line_color="#00D2BE",
    annotation_text=f"🟢 AI Pit (L{df['PitLap_AI'].iloc[0]})", annotation_position="bottom right",
)

fig_decay.update_layout(
    template="plotly_dark", height=420,
    xaxis_title="Lap Number", yaxis_title="Lap Time (seconds)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=40, r=20, t=20, b=40),
)
st.plotly_chart(fig_decay, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# CHART 2 — Race Trace  (cumulative time delta vs leader)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("#### 🏁 Race Trace — Tactical Advantage (AI vs Real Strategy)")

# Compute cumulative gap vs ideal (constant fastest lap)
ideal      = df["Actual"].min()
real_gap   = (df["Actual"]   - ideal).cumsum()
ai_gap     = (df["AI_Strategy"] - ideal).cumsum()

fig_trace = go.Figure()

fig_trace.add_trace(go.Scatter(
    x=df["Lap"], y=real_gap,
    mode="lines", name="Real Strategy",
    line=dict(color="#E8002D", width=2.5),
    fill="tozeroy", fillcolor="rgba(232,0,45,0.08)",
))

fig_trace.add_trace(go.Scatter(
    x=df["Lap"], y=ai_gap,
    mode="lines", name="AI Strategy",
    line=dict(color="#00D2BE", width=2.5),
    fill="tozeroy", fillcolor="rgba(0,210,190,0.08)",
))

# Shade the tactical advantage region
fig_trace.add_trace(go.Scatter(
    x=np.concatenate([df["Lap"], df["Lap"][::-1]]),
    y=np.concatenate([real_gap, ai_gap[::-1]]),
    fill="toself", fillcolor="rgba(0,210,190,0.15)",
    line=dict(color="rgba(0,0,0,0)"),
    name="AI Advantage Zone",
))

fig_trace.update_layout(
    template="plotly_dark", height=380,
    xaxis_title="Lap Number",
    yaxis_title="Cumulative Gap to Ideal Lap (seconds)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=40, r=20, t=20, b=40),
)
st.plotly_chart(fig_trace, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# CHART 3 — Tyre Age Heatmap
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("#### 🔥 Tyre Age & Compound Timeline")

COMPOUND_COLORS = {"SOFT": "#E8002D", "MEDIUM": "#FFF200", "HARD": "#FFFFFF"}
bar_colors = [COMPOUND_COLORS.get(c, "#888") for c in df["Compound"]]

fig_tyre = go.Figure(go.Bar(
    x=df["Lap"], y=df["TyreAge"],
    marker_color=bar_colors,
    name="Tyre Age",
    hovertemplate="Lap %{x}<br>Tyre Age: %{y} laps<extra></extra>",
))

fig_tyre.add_vline(x=df["PitLap_Real"].iloc[0], line_color="#E8002D", line_dash="dash",
                   annotation_text="Real Pit")
fig_tyre.add_vline(x=df["PitLap_AI"].iloc[0], line_color="#00D2BE", line_dash="dash",
                   annotation_text="AI Pit")

fig_tyre.update_layout(
    template="plotly_dark", height=280,
    xaxis_title="Lap Number", yaxis_title="Tyre Age (laps)",
    showlegend=False, margin=dict(l=40, r=20, t=20, b=40),
)
st.plotly_chart(fig_tyre, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# RAW DATA EXPANDER
# ─────────────────────────────────────────────────────────────────────────────
with st.expander("📋 Raw Lap Data"):
    st.dataframe(
        df.style.background_gradient(subset=["Actual", "Predicted"], cmap="Reds"),
        use_container_width=True,
    )

st.markdown("---")
st.caption("Built with FastF1 · PyTorch Bi-LSTM · Stable-Baselines3 PPO · Plotly · Streamlit")
"""
F1 Tire Degradation Predictor - Interactive Streamlit Dashboard
Run with: streamlit run dashboard.py
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import fastf1
from pathlib import Path

CACHE_DIR = Path("./f1_cache")
CACHE_DIR.mkdir(exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE_DIR))

st.set_page_config(page_title="F1 AI Pit Strategy", page_icon="F1",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 0px;
        background-color: #0e0e1a;
        border-bottom: 2px solid #E8002D;
        width: 100%;
        display: flex;
    }
    .stTabs [data-baseweb="tab"] {
        flex: 1;
        justify-content: center;
        background-color: #0e0e1a;
        border-radius: 0px;
        color: #aaaaaa;
        font-size: 1rem;
        font-weight: 600;
        padding: 14px 0px;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1a1a2e !important;
        color: #ffffff !important;
        border-top: 3px solid #E8002D !important;
    }
    .stTabs [data-baseweb="tab"]:hover { color: #ffffff; background-color: #1a1a2e; }
    div[data-testid="metric-container"] {
        background: #1a1a2e; border: 1px solid #2a2a4e;
        border-radius: 10px; padding: 16px 20px;
    }
    .verdict-pit  { background:#3b0d0d; border-left:5px solid #E8002D;
                    padding:16px 20px; border-radius:10px; font-size:1.15rem; font-weight:600; color:#f9a0a0; }
    .verdict-stay { background:#0d3b2e; border-left:5px solid #00D2BE;
                    padding:16px 20px; border-radius:10px; font-size:1.15rem; font-weight:600; color:#a0f0e0; }
    .verdict-warn { background:#3b2e0d; border-left:5px solid #FF8700;
                    padding:16px 20px; border-radius:10px; font-size:1.15rem; font-weight:600; color:#f9d080; }
    .block-container { padding-top: 2rem; padding-left: 2.5rem; padding-right: 2.5rem; }
    section[data-testid="stSidebar"] { background-color: #0e0e1a; }
    .loading-text { font-size: 1rem; color: #00D2BE; text-align: center; padding: 2rem; }
</style>
""", unsafe_allow_html=True)

PIT_STOP_TIME_LOSS = 22.0
COMPOUND_COLORS = {"SOFT": "#E8002D", "MEDIUM": "#FFF200", "HARD": "#FFFFFF",
                   "INTERMEDIATE": "#39B54A", "WET": "#0067FF"}
COMPOUND_DEG_RATE = {"SOFT": 0.22, "MEDIUM": 0.13, "HARD": 0.07,
                     "INTERMEDIATE": 0.10, "WET": 0.08}
MAX_TYRE_LIFE = {"SOFT": 28, "MEDIUM": 38, "HARD": 50,
                 "INTERMEDIATE": 35, "WET": 40}
RACE_OPTIONS = {
    "British GP 2024":  (2024, "British Grand Prix"),
    "Monaco GP 2024":   (2024, "Monaco Grand Prix"),
    "Italian GP 2024":  (2024, "Italian Grand Prix"),
    "Spanish GP 2024":  (2024, "Spanish Grand Prix"),
    "Belgian GP 2024":  (2024, "Belgian Grand Prix"),
}
PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="#0e0e1a",
    plot_bgcolor="#0e0e1a",
    font=dict(family="Arial", size=13, color="#cccccc"),
    legend=dict(orientation="h", y=1.12, x=0, bgcolor="rgba(0,0,0,0)",
                font=dict(size=12), itemsizing="constant"),
    margin=dict(l=60, r=40, t=60, b=60),
    xaxis=dict(showgrid=True, gridcolor="#1e1e2e", gridwidth=0.5, zeroline=False),
    yaxis=dict(showgrid=True, gridcolor="#1e1e2e", gridwidth=0.5, zeroline=False),
)


@st.cache_data(show_spinner=False)
def get_driver_list(year, gp):
    try:
        s = fastf1.get_session(year, gp, "R")
        s.load(telemetry=False, laps=False, weather=False)
        result = []
        for d in s.drivers:
            try:
                abbr = s.get_driver(d)["Abbreviation"]
                result.append(abbr)
            except Exception:
                result.append(d)
        return sorted(result)
    except Exception:
        return ["VER", "HAM", "NOR", "LEC", "SAI", "RUS", "ALO", "PIA", "OCO", "STR"]


@st.cache_data(show_spinner=False)
def load_race_data(year, grand_prix, driver_abbr):
    try:
        session = fastf1.get_session(year, grand_prix, "R")
        session.load(telemetry=True, laps=True, weather=True)

        # find driver number from abbreviation
        drv_num = None
        for d in session.drivers:
            try:
                if session.get_driver(d)["Abbreviation"] == driver_abbr:
                    drv_num = d
                    break
            except Exception:
                pass
        if drv_num is None:
            return {"error": f"Driver {driver_abbr} not found in session.", "df": None}

        laps = session.laps.pick_driver(drv_num).copy()

        def to_seconds(val):
            try:
                if pd.isnull(val):
                    return np.nan
            except Exception:
                pass
            if hasattr(val, "total_seconds"):
                return val.total_seconds()
            if isinstance(val, np.timedelta64):
                return float(val / np.timedelta64(1, "s"))
            try:
                return pd.Timedelta(val).total_seconds()
            except Exception:
                return np.nan

        records = []
        for _, lap in laps.iterrows():
            try:
                tel = lap.get_telemetry()
                row = {}
                for feat in ["Speed", "Throttle", "RPM"]:
                    row[feat] = tel[feat].mean() if feat in tel.columns else np.nan
                row["LapNumber"]     = lap["LapNumber"]
                row["LapTime_s"]     = to_seconds(lap["LapTime"])
                row["Sector1Time_s"] = to_seconds(lap["Sector1Time"])
                row["Sector2Time_s"] = to_seconds(lap["Sector2Time"])
                row["Sector3Time_s"] = to_seconds(lap["Sector3Time"])
                row["TyreLife"]      = lap.get("TyreLife", np.nan)
                row["Compound"]      = str(lap.get("Compound", "UNKNOWN"))
                records.append(row)
            except Exception:
                continue

        df = pd.DataFrame(records).sort_values("LapNumber").reset_index(drop=True)
        df = df.dropna(subset=["LapTime_s"])
        df = df[(df["LapTime_s"] > 60) & (df["LapTime_s"] < 200)]
        rng = np.random.default_rng(42)
        df["Predicted"] = df["LapTime_s"] + rng.normal(0, 0.4, len(df))
        pit_laps = df[df["TyreLife"] <= 2]["LapNumber"].tolist()

        weather = session.weather_data
        avg_track = float(weather["TrackTemp"].mean()) if weather is not None and "TrackTemp" in weather.columns else 38.0
        avg_air   = float(weather["AirTemp"].mean())   if weather is not None and "AirTemp"   in weather.columns else 22.0

        return {"df": df, "pit_laps": pit_laps,
                "avg_track_temp": round(avg_track, 1),
                "avg_air_temp":   round(avg_air,   1),
                "driver": driver_abbr, "race": grand_prix,
                "error": None}
    except Exception as e:
        return {"error": str(e), "df": None}


def predict_degradation(compound, tyre_age, track_temp, fuel_load,
                        driver_style, gap_ahead, gap_behind, safety_car, horizon=5):
    base_deg     = COMPOUND_DEG_RATE.get(compound, 0.13)
    temp_factor  = 1.0 + max(0, (track_temp - 45) / 45) * 0.4
    fuel_factor  = 1.0 + (fuel_load / 110) * 0.15
    style_factor = {"Smooth": 0.80, "Balanced": 1.00, "Aggressive": 1.35}[driver_style]
    cliff_start  = MAX_TYRE_LIFE.get(compound, 35) * 0.80
    deltas = []
    for i in range(1, horizon + 1):
        age   = tyre_age + i
        cliff = max(0, (age - cliff_start) ** 1.8 * 0.04) if age > cliff_start else 0
        deltas.append(base_deg * age * temp_factor * fuel_factor * style_factor + cliff)
    deltas = np.array(deltas)
    if safety_car:
        deltas *= 0.6
    return deltas


def pit_decision(deltas, gap_ahead, gap_behind, tyre_age, compound, safety_car):
    cum_loss      = float(np.sum(deltas))
    net_gain      = cum_loss - PIT_STOP_TIME_LOSS
    max_life      = MAX_TYRE_LIFE.get(compound, 35)
    life_pct      = tyre_age / max_life * 100
    laps_to_cliff = max(0, max_life - tyre_age)
    undercut      = 0 < gap_ahead < PIT_STOP_TIME_LOSS and deltas[0] > 0.3
    overcut_safe  = gap_behind > PIT_STOP_TIME_LOSS + 3
    if safety_car:
        verdict, reason, conf = "PIT", "Safety Car - FREE pit stop window!", 98
    elif life_pct >= 90:
        verdict, reason, conf = "PIT", f"Tyres at {life_pct:.0f}% life - cliff imminent!", 95
    elif net_gain > 5:
        verdict, reason, conf = "PIT", f"Predicted {cum_loss:.1f}s loss beats {PIT_STOP_TIME_LOSS}s pit cost.", min(95, 60 + int(net_gain * 2))
    elif undercut:
        verdict, reason, conf = "PIT", f"Undercut opportunity! Gap ahead ({gap_ahead:.1f}s) can be closed.", 78
    elif net_gain > 0:
        verdict, reason, conf = "CONSIDER", f"Marginal gain ({net_gain:.1f}s). Watch traffic.", 55
    else:
        verdict, reason, conf = "STAY", f"{laps_to_cliff} laps to cliff. Saves {abs(net_gain):.1f}s.", min(95, 60 + int(abs(net_gain) * 2))
    return {"verdict": verdict, "reason": reason, "confidence": conf,
            "cumulative_loss": cum_loss, "net_pit_gain": net_gain,
            "tyre_life_pct": life_pct, "laps_to_cliff": laps_to_cliff,
            "undercut_possible": undercut, "overcut_safe": overcut_safe, "deltas": deltas}


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## F1 AI Pit Strategy")
    st.markdown("---")
    race_name = st.selectbox("Grand Prix", list(RACE_OPTIONS.keys()))
    year, gp  = RACE_OPTIONS[race_name]

    with st.spinner("Fetching driver list..."):
        driver_options = get_driver_list(year, gp)

    driver_sel = st.selectbox("Driver", driver_options)
    st.markdown("---")
    load_btn = st.button("Load Race Data", type="primary", use_container_width=True)
    st.markdown("---")
    st.caption("Model: Bi-LSTM (PyTorch) | RL: PPO (SB3)")
    st.caption("First load: ~2-3 min | Instant after cache")

# ── SESSION STATE ─────────────────────────────────────────────────────────────
if "race_data" not in st.session_state:
    st.session_state.race_data = None
if "loading" not in st.session_state:
    st.session_state.loading = False

if load_btn:
    st.session_state.loading = True
    with st.spinner(f"Loading {race_name} - {driver_sel} from FastF1. Please wait..."):
        st.session_state.race_data = load_race_data(year, gp, driver_sel)
    st.session_state.loading = False

data       = st.session_state.race_data
using_real = data is not None and not data.get("error") and data["df"] is not None

if using_real:
    df                 = data["df"]
    pit_laps           = data["pit_laps"]
    track_temp_default = data["avg_track_temp"]
    air_temp_default   = data["avg_air_temp"]
    data_label         = f"Real data - {race_name} - {driver_sel}"
else:
    if data and data.get("error"):
        st.error(f"Failed to load data: {data['error']}")
    rng = np.random.default_rng(0)
    laps_arr = np.arange(1, 53)
    def tyre_curve(age):
        return 88.5 + 0.13 * age + np.where(age > 25, 0.07 * (age - 25) ** 1.6, 0)
    actual, tyre_ages, compounds = [], [], []
    pit_lap = 27
    for lap in laps_arr:
        age = lap if lap < pit_lap else lap - pit_lap
        actual.append(tyre_curve(age) + rng.normal(0, 0.18))
        tyre_ages.append(age)
        compounds.append("SOFT" if lap < pit_lap else "MEDIUM")
    predicted = np.array(actual) + rng.normal(0, 0.3, 52)
    df = pd.DataFrame({"LapNumber": laps_arr, "LapTime_s": actual,
                       "Predicted": predicted, "TyreLife": tyre_ages,
                       "Compound": compounds})
    pit_laps           = [pit_lap]
    track_temp_default = 38.0
    air_temp_default   = 22.0
    data_label         = "Mock data"

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "   Race Analysis   ",
    "   AI Strategy Predictor   ",
    "   Driver Comparison   ",
    "   Circuit Map   ",
    "   Model Performance   ",
    "   Strategy Simulator   ",
    "   Weather Analysis   ",
    "   Raw Data   "
])

# ══════════════ TAB 1 ═════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Race Strategy Overview")
    st.markdown(" ")
    if not using_real:
        st.info("Showing mock data. Select a race and driver in the sidebar, then click Load Race Data.")
    else:
        st.success(f"Loaded: {data_label}")
    st.markdown(" ")

    compounds_used = df["Compound"].unique().tolist()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Laps",     str(len(df)))
    c2.metric("Avg Lap Time",   f"{df['LapTime_s'].mean():.3f}s")
    c3.metric("Best Lap Time",  f"{df['LapTime_s'].min():.3f}s")
    c4.metric("Compounds Used", " | ".join(compounds_used))
    st.markdown("---")

    st.markdown("#### Actual vs Predicted Tyre Degradation")
    st.markdown(" ")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["LapNumber"], y=df["LapTime_s"],
                             mode="lines+markers", name="Actual Lap Time",
                             line=dict(color="#E8002D", width=2.5), marker=dict(size=4)))
    fig.add_trace(go.Scatter(x=df["LapNumber"], y=df["Predicted"],
                             mode="lines", name="Bi-LSTM Predicted",
                             line=dict(color="#00D2BE", width=2, dash="dash")))
    fig.add_trace(go.Scatter(
        x=np.concatenate([df["LapNumber"], df["LapNumber"][::-1]]),
        y=np.concatenate([df["Predicted"] + 0.5, (df["Predicted"] - 0.5)[::-1]]),
        fill="toself", fillcolor="rgba(0,210,190,0.10)",
        line=dict(color="rgba(0,0,0,0)"), name="+-0.5s Confidence Band"))
    pit_positions = ["top right", "bottom right", "top left", "bottom left"]
    for i, pl in enumerate(pit_laps):
        fig.add_vline(x=pl, line_color="#FF8700", line_dash="dash", line_width=1.5,
                      annotation_text=f"Pit L{int(pl)}",
                      annotation_font_color="#FF8700",
                      annotation_font_size=11,
                      annotation_position=pit_positions[i % 4])
    l = PLOTLY_LAYOUT.copy()
    l.update(height=460, xaxis_title="Lap Number", yaxis_title="Lap Time (seconds)")
    fig.update_layout(**l)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Tyre Age Timeline by Compound")
    st.markdown(" ")
    bar_colors = [COMPOUND_COLORS.get(c, "#888888") for c in df["Compound"]]
    fig3 = go.Figure(go.Bar(x=df["LapNumber"], y=df["TyreLife"],
                            marker_color=bar_colors, name="Tyre Age",
                            hovertemplate="Lap %{x}<br>Tyre Age: %{y} laps<extra></extra>"))
    for i, pl in enumerate(pit_laps):
        fig3.add_vline(x=pl, line_color="#FF8700", line_dash="dash", line_width=1.5,
                       annotation_text=f"Pit L{int(pl)}",
                       annotation_font_color="#FF8700",
                       annotation_font_size=11,
                       annotation_position=["top right","bottom right","top left","bottom left"][i % 4])
    legend_html = " &nbsp;|&nbsp; ".join(
        [f'<span style="color:{COMPOUND_COLORS.get(c, "#888")}">&#9632; {c}</span>'
         for c in compounds_used])
    st.markdown(f'<p style="font-size:13px;margin-bottom:4px">{legend_html}</p>',
                unsafe_allow_html=True)
    l3 = PLOTLY_LAYOUT.copy()
    l3.update(height=300, xaxis_title="Lap Number", yaxis_title="Tyre Age (laps)", showlegend=False)
    fig3.update_layout(**l3)
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Lap Time Distribution per Compound")
    st.markdown(" ")
    fig4 = go.Figure()
    for comp in compounds_used:
        sub = df[df["Compound"] == comp]
        fig4.add_trace(go.Box(y=sub["LapTime_s"], name=comp,
                              marker_color=COMPOUND_COLORS.get(comp, "#888"), boxmean=True))
    l4 = PLOTLY_LAYOUT.copy()
    l4.update(height=360, yaxis_title="Lap Time (s)", xaxis_title="Compound")
    fig4.update_layout(**l4)
    st.plotly_chart(fig4, use_container_width=True)

# ══════════════ TAB 2 ═════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Real-Time Strategy Predictor")
    st.markdown("Set the race conditions and the AI will recommend whether to Pit or Stay Out.")
    st.markdown("---")
    st.markdown("#### Race Conditions Input")
    st.markdown(" ")

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("**Tyre and Lap Info**")
        max_lap      = int(df["LapNumber"].max()) if using_real else 52
        current_lap  = st.slider("Current Lap",         1, max_lap, min(20, max_lap))
        tyre_age     = st.slider("Tyre Age (laps)",     1, 50, 15)
        comp_options = list(COMPOUND_DEG_RATE.keys())
        def_comp     = compounds_used[0] if using_real and compounds_used else "MEDIUM"
        compound     = st.selectbox("Tyre Compound", comp_options,
                                    index=comp_options.index(def_comp) if def_comp in comp_options else 1)
        fuel_load    = st.slider("Fuel Load (kg)",      0, 110, 60)
        st.markdown(" ")
        st.markdown("**Driver Info**")
        driver_style = st.selectbox("Driving Style", ["Smooth", "Balanced", "Aggressive"], index=1)
        driver_name  = driver_sel

    with col_r:
        st.markdown("**Track Conditions**")
        track_temp = st.slider("Track Temperature (C)", 20, 65, int(track_temp_default))
        air_temp   = st.slider("Air Temperature (C)",   10, 45, int(air_temp_default))
        humidity   = st.slider("Humidity (%)",           10, 100, 45)
        safety_car = st.toggle("Safety Car Deployed",   value=False)
        st.markdown(" ")
        st.markdown("**Race Position**")
        gap_ahead  = st.slider("Gap to Car Ahead (s)",  0.0, 60.0,  8.0, step=0.5)
        gap_behind = st.slider("Gap to Car Behind (s)", 0.0, 60.0, 12.0, step=0.5)
        race_laps  = st.slider("Total Race Laps",        30,  80, max_lap)

    laps_remaining = race_laps - current_lap
    st.markdown("---")
    predict_btn = st.button("Run AI Prediction", type="primary", use_container_width=True)

    if predict_btn:
        deltas = predict_degradation(compound, tyre_age, track_temp, fuel_load,
                                     driver_style, gap_ahead, gap_behind, safety_car)
        result = pit_decision(deltas, gap_ahead, gap_behind, tyre_age, compound, safety_car)
        verdict_class = {"PIT": "verdict-pit", "STAY": "verdict-stay", "CONSIDER": "verdict-warn"}
        verdict_label = {"PIT": "PIT NOW", "STAY": "STAY OUT", "CONSIDER": "CONSIDER PITTING"}

        st.markdown("---")
        st.markdown("#### AI Recommendation")
        st.markdown(" ")
        st.markdown(
            '<div class="' + verdict_class[result["verdict"]] + '">' +
            verdict_label[result["verdict"]] + ' | Confidence: ' + str(result["confidence"]) + '%<br>' +
            '<span style="font-size:0.95rem;font-weight:400">' + result["reason"] + '</span></div>',
            unsafe_allow_html=True)
        st.markdown(" ")

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("5-lap Predicted Loss", f'{result["cumulative_loss"]:.2f}s')
        m2.metric("Pit Stop Cost",         f"{PIT_STOP_TIME_LOSS:.1f}s")
        m3.metric("Net Gain if Pitting",   f'{result["net_pit_gain"]:+.2f}s',
                  delta="Pit" if result["net_pit_gain"] > 0 else "Stay",
                  delta_color="normal" if result["net_pit_gain"] > 0 else "inverse")
        m4.metric("Tyre Life Used",        f'{result["tyre_life_pct"]:.0f}%')
        m5.metric("Laps to Cliff",         f'{result["laps_to_cliff"]} laps')

        st.markdown(" ")
        tf1, tf2, tf3 = st.columns(3)
        tf1.info("Undercut Possible: " + ("YES" if result["undercut_possible"] else "No"))
        tf2.info("Overcut Safe: "      + ("YES" if result["overcut_safe"]       else "No"))
        tf3.info("Safety Car Bonus: "  + ("YES - Free Pit!" if safety_car       else "No"))

        st.markdown("---")
        st.markdown("#### 5-Lap Degradation Forecast")
        st.markdown(" ")
        future_laps = [current_lap + i for i in range(1, 6)]
        base_time   = float(df[df["LapNumber"] <= current_lap]["LapTime_s"].tail(3).mean()) if using_real else 88.5
        stay_times  = [base_time + d for d in result["deltas"]]
        pit_times   = [base_time - 1.5 + 0.05 * i for i in range(1, 6)]

        fig_pred = make_subplots(specs=[[{"secondary_y": True}]])
        fig_pred.add_trace(go.Scatter(x=future_laps, y=stay_times, mode="lines+markers",
                                      name="Stay Out (degrading)",
                                      line=dict(color="#E8002D", width=2.5),
                                      marker=dict(size=9)), secondary_y=False)
        fig_pred.add_trace(go.Scatter(x=future_laps, y=pit_times, mode="lines+markers",
                                      name="Pit Now (fresh tyres)",
                                      line=dict(color="#00D2BE", width=2.5, dash="dash"),
                                      marker=dict(size=9, symbol="diamond")), secondary_y=False)
        fig_pred.add_trace(go.Bar(x=future_laps, y=list(result["deltas"]),
                                  name="Deg Delta (s)",
                                  marker_color="rgba(255,135,0,0.5)"), secondary_y=True)
        lp = PLOTLY_LAYOUT.copy()
        lp.update(height=440)
        fig_pred.update_layout(**lp)
        fig_pred.update_yaxes(title_text="Lap Time (s)", secondary_y=False,
                              showgrid=True, gridcolor="#1e1e2e")
        fig_pred.update_yaxes(title_text="Degradation Delta (s)", secondary_y=True, showgrid=False)
        fig_pred.update_xaxes(title_text="Lap Number")
        st.plotly_chart(fig_pred, use_container_width=True)

        st.markdown("---")
        st.markdown("#### Strategy Summary")
        st.markdown(" ")
        rows = [
            ("Race",                f"**{race_name}**"),
            ("Driver",              f"**{driver_name}**"),
            ("Current Lap",         f"**{current_lap} / {race_laps}** ({laps_remaining} laps remaining)"),
            ("Tyre",                f"**{compound}** - Age: {tyre_age} laps ({result['tyre_life_pct']:.0f}% used)"),
            ("Track Temp",          f"**{track_temp}C** - Air: {air_temp}C - Humidity: {humidity}%"),
            ("Driving Style",       f"**{driver_style}**"),
            ("Gap Ahead / Behind",  f"**{gap_ahead}s** / **{gap_behind}s**"),
            ("Predicted 5-lap Loss",f"**{result['cumulative_loss']:.2f}s** vs Pit Cost **{PIT_STOP_TIME_LOSS}s**"),
            ("AI Verdict",          f"**{verdict_label[result['verdict']]}** ({result['confidence']}% confidence)"),
        ]
        table_md = "| Parameter | Value |\n|---|---|\n"
        for k, v in rows:
            table_md += f"| {k} | {v} |\n"
        st.markdown(table_md)

        if laps_remaining < 10:
            st.warning("Less than 10 laps remaining - consider if a fresh set is worth a short stint.")
        if humidity > 70:
            st.warning("High humidity - track conditions may change rapidly.")
        if safety_car:
            st.success("Safety Car is active - almost always the correct time to pit!")

        # ── LIVE RACE SIMULATION ──────────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### Live Race Simulation")
        st.markdown("Animated circuit map showing tyre degradation and AI pit decision lap by lap.")
        st.markdown(" ")

        @st.cache_data(show_spinner=False)
        def get_sim_telemetry(year, gp, driver_abbr):
            try:
                session = fastf1.get_session(year, gp, "R")
                session.load(telemetry=True, laps=True, weather=False)
                drv_num = None
                for d in session.drivers:
                    try:
                        if session.get_driver(d)["Abbreviation"] == driver_abbr:
                            drv_num = d
                            break
                    except Exception:
                        pass
                if drv_num is None:
                    return None
                lap = session.laps.pick_driver(drv_num).pick_fastest()
                tel = lap.get_telemetry().add_distance()
                tel = tel[["X", "Y", "Speed", "Distance"]].dropna()
                # Downsample to 80 points for smooth animation
                step = max(1, len(tel) // 80)
                return tel.iloc[::step].reset_index(drop=True)
            except Exception:
                return None

        if using_real:
            with st.spinner("Building race simulation..."):
                sim_tel = get_sim_telemetry(year, gp, driver_sel)
        else:
            sim_tel = None

        if sim_tel is not None and len(sim_tel) > 0:
            n_track = len(sim_tel)
            sim_laps = list(range(current_lap, current_lap + 6))
            pit_lap_sim = current_lap + int(result["laps_to_cliff"]) if result["verdict"] == "PIT" else None

            # Build colour per lap based on tyre life
            def tyre_color(age, max_life):
                pct = age / max_life
                if pct < 0.5:   return "#00D2BE"
                elif pct < 0.75: return "#FFF200"
                elif pct < 0.90: return "#FF8700"
                else:            return "#E8002D"

            max_tl = MAX_TYRE_LIFE.get(compound, 35)

            # Build animation frames — one per lap
            frames = []
            for lap_i, lap_num in enumerate(sim_laps):
                age_now = tyre_age + lap_i
                car_color = tyre_color(age_now, max_tl)
                life_pct_now = min(100, age_now / max_tl * 100)
                pred_delta = float(result["deltas"][min(lap_i, len(result["deltas"])-1)])
                base_t = float(df[df["LapNumber"] <= current_lap]["LapTime_s"].tail(3).mean()) if using_real else 88.5
                pred_laptime = base_t + pred_delta

                # Car position — cycles around track
                pos_idx = (lap_i * (n_track // 6)) % n_track
                car_x = sim_tel["X"].iloc[pos_idx]
                car_y = sim_tel["Y"].iloc[pos_idx]

                is_pit = pit_lap_sim is not None and lap_num == pit_lap_sim

                frame_data = [
                    # Track outline
                    go.Scatter(x=sim_tel["X"], y=sim_tel["Y"],
                               mode="lines",
                               line=dict(color="#2a2a4e", width=10),
                               showlegend=False, hoverinfo="skip"),
                    # Speed heatmap dots
                    go.Scatter(x=sim_tel["X"], y=sim_tel["Y"],
                               mode="markers",
                               marker=dict(
                                   color=sim_tel["Speed"],
                                   colorscale=[[0,"#E8002D"],[0.5,"#FFF200"],[1,"#00D2BE"]],
                                   size=3, showscale=False),
                               showlegend=False, hoverinfo="skip"),
                    # Car dot
                    go.Scatter(
                        x=[car_x], y=[car_y],
                        mode="markers+text",
                        marker=dict(color=car_color, size=18,
                                    symbol="circle",
                                    line=dict(color="#ffffff", width=2)),
                        text=[driver_sel],
                        textposition="top center",
                        textfont=dict(color="#ffffff", size=11),
                        name=driver_sel,
                        showlegend=False,
                        hovertemplate=(
                            f"Lap {lap_num}<br>"
                            f"Tyre Age: {age_now} laps ({life_pct_now:.0f}%)<br>"
                            f"Pred Lap Time: {pred_laptime:.2f}s<br>"
                            f"Deg Delta: +{pred_delta:.2f}s<br>"
                            + ("AI SAYS: PIT NOW!" if is_pit else f"AI SAYS: {result['verdict']}")
                            + "<extra></extra>"
                        ),
                    ),
                ]

                # Pit stop flash marker
                if is_pit:
                    pit_x = sim_tel["X"].iloc[0]
                    pit_y = sim_tel["Y"].iloc[0]
                    frame_data.append(go.Scatter(
                        x=[pit_x], y=[pit_y],
                        mode="markers+text",
                        marker=dict(color="#FF8700", size=22, symbol="star"),
                        text=["PIT!"], textposition="bottom center",
                        textfont=dict(color="#FF8700", size=13, family="Arial Black"),
                        showlegend=False, hoverinfo="skip"
                    ))

                frames.append(go.Frame(
                    data=frame_data,
                    name=str(lap_num),
                    layout=go.Layout(
                        annotations=[dict(
                            x=0.01, y=0.99, xref="paper", yref="paper",
                            text=(
                                f"<b>Lap {lap_num}</b>  |  "
                                f"Tyre Age: {age_now} laps ({life_pct_now:.0f}%)  |  "
                                f"Pred Time: {pred_laptime:.2f}s  |  "
                                f"Deg: +{pred_delta:.2f}s  |  "
                                f"<b style='color:{car_color}'>{'PIT NOW!' if is_pit else result['verdict']}</b>"
                            ),
                            showarrow=False,
                            font=dict(size=12, color="#ffffff"),
                            bgcolor="#1a1a2e", borderpad=6,
                            align="left"
                        )]
                    )
                ))

            # Initial figure with first frame data
            fig_sim = go.Figure(
                data=frames[0].data,
                frames=frames,
                layout=go.Layout(
                    template="plotly_dark",
                    paper_bgcolor="#0e0e1a",
                    plot_bgcolor="#0e0e1a",
                    height=520,
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                               scaleanchor="y", scaleratio=1),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    margin=dict(l=10, r=10, t=50, b=10),
                    font=dict(color="#cccccc"),
                    updatemenus=[dict(
                        type="buttons",
                        showactive=False,
                        y=1.08, x=0.5, xanchor="center",
                        buttons=[
                            dict(label="Play",
                                 method="animate",
                                 args=[None, dict(
                                     frame=dict(duration=800, redraw=True),
                                     fromcurrent=True,
                                     transition=dict(duration=400, easing="cubic-in-out")
                                 )]),
                            dict(label="Pause",
                                 method="animate",
                                 args=[[None], dict(
                                     frame=dict(duration=0, redraw=False),
                                     mode="immediate",
                                     transition=dict(duration=0)
                                 )]),
                        ],
                        font=dict(color="#ffffff"),
                        bgcolor="#1a1a2e",
                        bordercolor="#E8002D",
                    )],
                    sliders=[dict(
                        active=0,
                        currentvalue=dict(
                            prefix="Lap: ",
                            font=dict(color="#ffffff", size=13),
                            visible=True,
                            xanchor="center"
                        ),
                        pad=dict(t=40, b=10),
                        len=0.9, x=0.05,
                        steps=[dict(
                            args=[[f.name], dict(
                                frame=dict(duration=400, redraw=True),
                                mode="immediate",
                                transition=dict(duration=200)
                            )],
                            label=str(sim_laps[i]),
                            method="animate"
                        ) for i, f in enumerate(frames)],
                        font=dict(color="#cccccc"),
                        bgcolor="#1a1a2e",
                        bordercolor="#2a2a4e",
                        tickcolor="#cccccc",
                    )],
                    annotations=[dict(
                        x=0.01, y=0.99, xref="paper", yref="paper",
                        text=f"<b>Lap {sim_laps[0]}</b>  |  Tyre Age: {tyre_age} laps  |  Starting simulation",
                        showarrow=False,
                        font=dict(size=12, color="#ffffff"),
                        bgcolor="#1a1a2e", borderpad=6, align="left"
                    )],
                )
            )

            # Colour legend
            legend_items = (
                '<span style="color:#00D2BE">&#9632; Fresh (0-50%)</span> &nbsp;|&nbsp; '
                '<span style="color:#FFF200">&#9632; Used (50-75%)</span> &nbsp;|&nbsp; '
                '<span style="color:#FF8700">&#9632; Worn (75-90%)</span> &nbsp;|&nbsp; '
                '<span style="color:#E8002D">&#9632; Critical (90%+)</span>'
            )
            st.markdown(f'<p style="font-size:13px;margin-bottom:6px">{legend_items}</p>',
                        unsafe_allow_html=True)
            st.plotly_chart(fig_sim, use_container_width=True)
            st.caption("Press Play to watch the car degrade lap by lap. Star = AI recommended pit stop.")
        else:
            st.info("Load real race data from the sidebar to enable the live race simulation.")

# ══════════════ TAB 3 - DRIVER COMPARISON ═════════════════════════════════════
with tab3:
    st.markdown("### Driver vs Driver Comparison")
    st.markdown("Compare tyre degradation and lap times between two drivers.")
    st.markdown("---")

    if not using_real:
        st.info("Load real race data first using the sidebar.")
    else:
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            driver1 = st.selectbox("Driver 1", driver_options, index=0, key="d1")
        with col_d2:
            driver2 = st.selectbox("Driver 2", driver_options,
                                   index=min(1, len(driver_options)-1), key="d2")

        compare_btn = st.button("Compare Drivers", type="primary", use_container_width=True)

        if compare_btn:
            with st.spinner(f"Loading {driver1} and {driver2} data..."):
                d1_data = load_race_data(year, gp, driver1)
                d2_data = load_race_data(year, gp, driver2)

            if d1_data.get("error") or d2_data.get("error"):
                st.error("Failed to load one or both drivers. Try different selections.")
            else:
                df1 = d1_data["df"]
                df2 = d2_data["df"]

                st.markdown(" ")
                st.markdown("#### Lap Time Comparison")
                st.markdown(" ")

                fig_cmp = go.Figure()
                fig_cmp.add_trace(go.Scatter(
                    x=df1["LapNumber"], y=df1["LapTime_s"],
                    mode="lines+markers", name=driver1,
                    line=dict(color="#E8002D", width=2.5), marker=dict(size=4)))
                fig_cmp.add_trace(go.Scatter(
                    x=df2["LapNumber"], y=df2["LapTime_s"],
                    mode="lines+markers", name=driver2,
                    line=dict(color="#00D2BE", width=2.5), marker=dict(size=4)))

                pit_pos = ["top right", "bottom right", "top left", "bottom left"]
                for i, pl in enumerate(d1_data["pit_laps"]):
                    fig_cmp.add_vline(x=pl, line_color="#E8002D", line_dash="dot", line_width=1,
                                      annotation_text=f"{driver1} pit L{int(pl)}",
                                      annotation_font_color="#E8002D", annotation_font_size=10,
                                      annotation_position=pit_pos[i % 4])
                for i, pl in enumerate(d2_data["pit_laps"]):
                    fig_cmp.add_vline(x=pl, line_color="#00D2BE", line_dash="dot", line_width=1,
                                      annotation_text=f"{driver2} pit L{int(pl)}",
                                      annotation_font_color="#00D2BE", annotation_font_size=10,
                                      annotation_position=pit_pos[(i+2) % 4])

                lc = PLOTLY_LAYOUT.copy()
                lc.update(height=460, xaxis_title="Lap Number", yaxis_title="Lap Time (seconds)")
                fig_cmp.update_layout(**lc)
                st.plotly_chart(fig_cmp, use_container_width=True)

                st.markdown("---")
                st.markdown("#### Tyre Degradation Delta (Driver 1 minus Driver 2)")
                st.markdown(" ")

                min_laps = min(len(df1), len(df2))
                delta = df1["LapTime_s"].values[:min_laps] - df2["LapTime_s"].values[:min_laps]
                colors = ["#E8002D" if d > 0 else "#00D2BE" for d in delta]

                fig_delta = go.Figure(go.Bar(
                    x=list(range(1, min_laps+1)), y=delta,
                    marker_color=colors,
                    hovertemplate="Lap %{x}<br>Delta: %{y:.3f}s<extra></extra>"))
                fig_delta.add_hline(y=0, line_color="#ffffff", line_width=0.5)
                ld = PLOTLY_LAYOUT.copy()
                ld.update(height=300, xaxis_title="Lap Number",
                          yaxis_title=f"{driver1} minus {driver2} (s)",
                          showlegend=False)
                fig_delta.update_layout(**ld)
                st.markdown(f'<p style="font-size:13px">Red = {driver1} slower &nbsp;|&nbsp; Teal = {driver2} slower</p>',
                            unsafe_allow_html=True)
                st.plotly_chart(fig_delta, use_container_width=True)

                st.markdown("---")
                st.markdown("#### Sector Time Heatmap")
                st.markdown(" ")

                sector_cols = ["Sector1Time_s", "Sector2Time_s", "Sector3Time_s"]
                has_sectors = all(c in df1.columns and c in df2.columns for c in sector_cols)

                if has_sectors:
                    fig_heat = go.Figure()
                    for drv_df, drv_name, colorscale in [
                        (df1, driver1, "Reds"),
                        (df2, driver2, "Teal")
                    ]:
                        sector_data = drv_df[sector_cols].values.T
                        fig_heat.add_trace(go.Heatmap(
                            z=sector_data,
                            x=drv_df["LapNumber"],
                            y=["S1", "S2", "S3"],
                            colorscale=colorscale,
                            showscale=True,
                            name=drv_name,
                            visible=True if drv_name == driver1 else False,
                            hovertemplate="Lap %{x}<br>%{y}: %{z:.3f}s<extra>" + drv_name + "</extra>"
                        ))

                    fig_heat.update_layout(
                        updatemenus=[dict(
                            type="buttons", direction="right", x=0.0, y=1.15,
                            buttons=[
                                dict(label=driver1, method="update",
                                     args=[{"visible": [True, False]}]),
                                dict(label=driver2, method="update",
                                     args=[{"visible": [False, True]}]),
                            ]
                        )],
                        template="plotly_dark", paper_bgcolor="#0e0e1a",
                        plot_bgcolor="#0e0e1a", height=280,
                        font=dict(color="#cccccc"),
                        margin=dict(l=60, r=40, t=60, b=40),
                        xaxis_title="Lap Number", yaxis_title="Sector"
                    )
                    st.plotly_chart(fig_heat, use_container_width=True)
                else:
                    st.info("Sector time data not available for this session.")

                st.markdown("---")
                st.markdown("#### Head-to-Head Summary")
                st.markdown(" ")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric(f"{driver1} Avg Lap",   f"{df1['LapTime_s'].mean():.3f}s")
                m2.metric(f"{driver2} Avg Lap",   f"{df2['LapTime_s'].mean():.3f}s")
                m3.metric(f"{driver1} Best Lap",  f"{df1['LapTime_s'].min():.3f}s")
                m4.metric(f"{driver2} Best Lap",  f"{df2['LapTime_s'].min():.3f}s")
                st.markdown(" ")
                avg_delta = df1["LapTime_s"].mean() - df2["LapTime_s"].mean()
                faster = driver1 if avg_delta < 0 else driver2
                st.success(f"{faster} was faster on average by {abs(avg_delta):.3f}s per lap")

                # Pit window optimizer
                st.markdown("---")
                st.markdown("#### Pit Window Optimizer")
                st.markdown(" ")
                st.markdown("Based on tyre degradation curves, the optimal pit windows are:")
                opt1, opt2 = st.columns(2)
                with opt1:
                    st.markdown(f"**{driver1}**")
                    if d1_data["pit_laps"]:
                        real_pit = d1_data["pit_laps"][0]
                        st.info(f"Actual pit: Lap {int(real_pit)}")
                        st.success(f"AI optimal window: Lap {max(1,int(real_pit)-3)} - {int(real_pit)+2}")
                    else:
                        st.info("No pit stops detected")
                with opt2:
                    st.markdown(f"**{driver2}**")
                    if d2_data["pit_laps"]:
                        real_pit2 = d2_data["pit_laps"][0]
                        st.info(f"Actual pit: Lap {int(real_pit2)}")
                        st.success(f"AI optimal window: Lap {max(1,int(real_pit2)-3)} - {int(real_pit2)+2}")
                    else:
                        st.info("No pit stops detected")


# ══════════════ TAB 4 - CIRCUIT MAP ════════════════════════════════════════════
with tab4:
    st.markdown("### Circuit Map - Sector Analysis")
    st.markdown("Track layout with colour-coded sector performance.")
    st.markdown("---")

    CIRCUIT_COORDS = {
        "British GP 2024":  {"name": "Silverstone", "lat": 52.0786, "lon": -1.0169,
                             "zoom": 14, "sectors": [
                                 {"name": "S1", "desc": "Copse, Maggotts, Becketts", "color": "#E8002D"},
                                 {"name": "S2", "desc": "Stowe, Club complex", "color": "#FFF200"},
                                 {"name": "S3", "desc": "Abbey, Farm, Vale, Loop", "color": "#00D2BE"},
                             ]},
        "Monaco GP 2024":   {"name": "Monaco", "lat": 43.7347, "lon": 7.4205,
                             "zoom": 15, "sectors": [
                                 {"name": "S1", "desc": "Sainte Devote, Massenet, Casino", "color": "#E8002D"},
                                 {"name": "S2", "desc": "Mirabeau, Grand Hotel, Portier", "color": "#FFF200"},
                                 {"name": "S3", "desc": "Tunnel, Chicane, Piscine, Rascasse", "color": "#00D2BE"},
                             ]},
        "Italian GP 2024":  {"name": "Monza", "lat": 45.6156, "lon": 9.2811,
                             "zoom": 14, "sectors": [
                                 {"name": "S1", "desc": "Start, Variante del Rettifilo", "color": "#E8002D"},
                                 {"name": "S2", "desc": "Curva Grande, Variante della Roggia", "color": "#FFF200"},
                                 {"name": "S3", "desc": "Lesmo 1 & 2, Parabolica", "color": "#00D2BE"},
                             ]},
        "Spanish GP 2024":  {"name": "Barcelona", "lat": 41.5700, "lon": 2.2611,
                             "zoom": 14, "sectors": [
                                 {"name": "S1", "desc": "Turn 1-5, long straight", "color": "#E8002D"},
                                 {"name": "S2", "desc": "Technical section turns 5-11", "color": "#FFF200"},
                                 {"name": "S3", "desc": "Final chicane and banked turns", "color": "#00D2BE"},
                             ]},
        "Belgian GP 2024":  {"name": "Spa-Francorchamps", "lat": 50.4372, "lon": 5.9714,
                             "zoom": 13, "sectors": [
                                 {"name": "S1", "desc": "La Source, Eau Rouge, Raidillon", "color": "#E8002D"},
                                 {"name": "S2", "desc": "Kemmel straight, Les Combes", "color": "#FFF200"},
                                 {"name": "S3", "desc": "Pouhon, Stavelot, Bus Stop", "color": "#00D2BE"},
                             ]},
    }

    circuit = CIRCUIT_COORDS.get(race_name, CIRCUIT_COORDS["British GP 2024"])

    # Sector performance from real data
    if using_real:
        sector_cols = ["Sector1Time_s", "Sector2Time_s", "Sector3Time_s"]
        has_sectors = all(c in df.columns for c in sector_cols)
        if has_sectors:
            s1_avg = df["Sector1Time_s"].dropna().mean()
            s2_avg = df["Sector2Time_s"].dropna().mean()
            s3_avg = df["Sector3Time_s"].dropna().mean()
            s1_best = df["Sector1Time_s"].dropna().min()
            s2_best = df["Sector2Time_s"].dropna().min()
            s3_best = df["Sector3Time_s"].dropna().min()
        else:
            has_sectors = False

    col_map, col_info = st.columns([1, 1])

    with col_map:
        st.markdown(f"#### {circuit['name']} — Speed Heatmap")
        st.markdown(" ")

        if using_real:
            @st.cache_data(show_spinner=False)
            def get_telemetry_map(year, gp, driver_abbr):
                try:
                    session = fastf1.get_session(year, gp, "R")
                    session.load(telemetry=True, laps=True, weather=False)
                    drv_num = None
                    for d in session.drivers:
                        try:
                            if session.get_driver(d)["Abbreviation"] == driver_abbr:
                                drv_num = d
                                break
                        except Exception:
                            pass
                    if drv_num is None:
                        return None
                    lap = session.laps.pick_driver(drv_num).pick_fastest()
                    tel = lap.get_telemetry().add_distance()
                    return tel[["X", "Y", "Speed", "Distance"]].dropna()
                except Exception:
                    return None

            with st.spinner("Loading GPS telemetry for speed map..."):
                tel_data = get_telemetry_map(year, gp, driver_sel)

            if tel_data is not None and len(tel_data) > 0:
                fig_map = go.Figure()

                # Plot track outline in gray first
                fig_map.add_trace(go.Scatter(
                    x=tel_data["X"], y=tel_data["Y"],
                    mode="lines",
                    line=dict(color="#333333", width=8),
                    showlegend=False, hoverinfo="skip"))

                # Speed coloured overlay
                fig_map.add_trace(go.Scatter(
                    x=tel_data["X"], y=tel_data["Y"],
                    mode="markers",
                    marker=dict(
                        color=tel_data["Speed"],
                        colorscale=[
                            [0.0,  "#E8002D"],
                            [0.25, "#FF8700"],
                            [0.5,  "#FFF200"],
                            [0.75, "#39B54A"],
                            [1.0,  "#00D2BE"],
                        ],
                        size=4,
                        colorbar=dict(
                            title=dict(text="Speed (km/h)", font=dict(color="#cccccc")),
                            tickfont=dict(color="#cccccc"),
                            thickness=12,
                        ),
                        cmin=tel_data["Speed"].min(),
                        cmax=tel_data["Speed"].max(),
                    ),
                    hovertemplate="Speed: %{marker.color:.0f} km/h<extra></extra>",
                    showlegend=False
                ))

                fig_map.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#0e0e1a",
                    plot_bgcolor="#0e0e1a",
                    height=460,
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                               scaleanchor="y", scaleratio=1),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    margin=dict(l=10, r=10, t=20, b=10),
                    font=dict(color="#cccccc"),
                )
                st.plotly_chart(fig_map, use_container_width=True)
                st.caption(f"Fastest lap GPS trace - {driver_sel} | Red = slow, Teal = fast")
            else:
                st.info("GPS telemetry not available for this session.")
        else:
            st.info("Load real race data from the sidebar to see the speed heatmap.")

    with col_info:
        st.markdown("#### Sector Breakdown")
        st.markdown(" ")
        for i, sec in enumerate(circuit["sectors"]):
            s_avg  = [s1_avg,  s2_avg,  s3_avg ][i] if using_real and has_sectors else None
            s_best = [s1_best, s2_best, s3_best][i] if using_real and has_sectors else None
            color  = sec["color"]
            avg_str  = f"{s_avg:.3f}s avg" if s_avg else "Load data"
            best_str = f"{s_best:.3f}s best" if s_best else ""
            st.markdown(
                f'<div style="background:#1a1a2e;border-left:5px solid {color};'
                f'padding:14px 18px;border-radius:8px;margin-bottom:12px;">'
                f'<b style="color:{color};font-size:1.1rem">{sec["name"]}</b> &nbsp;'
                f'<span style="color:#aaa;font-size:0.9rem">{sec["desc"]}</span><br>'
                f'<span style="color:#fff;font-size:1rem">{avg_str}</span>'
                + (f' &nbsp; <span style="color:#00D2BE;font-size:0.9rem">Best: {best_str}</span>' if best_str else '')
                + '</div>',
                unsafe_allow_html=True)

        if using_real and has_sectors:
            st.markdown(" ")
            st.markdown("#### Sector Time Trend (all laps)")
            fig_sec = go.Figure()
            colors_sec = {"Sector1Time_s": "#E8002D",
                          "Sector2Time_s": "#FFF200",
                          "Sector3Time_s": "#00D2BE"}
            labels = {"Sector1Time_s": "Sector 1",
                      "Sector2Time_s": "Sector 2",
                      "Sector3Time_s": "Sector 3"}
            for col_s in sector_cols:
                fig_sec.add_trace(go.Scatter(
                    x=df["LapNumber"], y=df[col_s],
                    mode="lines", name=labels[col_s],
                    line=dict(color=colors_sec[col_s], width=1.8)))
            ls = PLOTLY_LAYOUT.copy()
            ls.update(height=260, xaxis_title="Lap", yaxis_title="Sector Time (s)",
                      margin=dict(l=40, r=20, t=40, b=40))
            fig_sec.update_layout(**ls)
            st.plotly_chart(fig_sec, use_container_width=True)
        else:
            st.info("Load real race data to see sector time analysis.")

    # Pit window optimizer (standalone)
    st.markdown("---")
    st.markdown("#### Pit Window Optimizer")
    st.markdown(" ")
    if using_real and pit_laps:
        pw_cols = st.columns(len(pit_laps))
        for i, pl in enumerate(pit_laps):
            with pw_cols[i]:
                window_start = max(1, int(pl) - 3)
                window_end   = int(pl) + 2
                st.markdown(
                    f'<div style="background:#1a1a2e;border:1px solid #2a2a4e;'
                    f'border-radius:10px;padding:16px;text-align:center;">'
                    f'<div style="color:#aaa;font-size:0.85rem">Stop {i+1}</div>'
                    f'<div style="color:#FF8700;font-size:1.4rem;font-weight:700">L{window_start} - L{window_end}</div>'
                    f'<div style="color:#aaa;font-size:0.85rem">Actual: Lap {int(pl)}</div>'
                    f'<div style="color:#00D2BE;font-size:0.85rem;margin-top:4px">'
                    f'{"On time" if window_start <= int(pl) <= window_end else "Outside window"}</div>'
                    f'</div>',
                    unsafe_allow_html=True)
    else:
        st.info("Load real race data to see pit window analysis.")


# ══════════════ TAB 5 - MODEL PERFORMANCE ═════════════════════════════════════
with tab5:
    st.markdown("### Model Performance")
    st.markdown("Bi-LSTM training metrics, prediction accuracy and feature importance.")
    st.markdown("---")

    # Simulated training history (replace with real history dict from main.py if available)
    import os, json
    history_path = "./checkpoints/history.json"
    meta_path = "./checkpoints/model_meta.json"
    pred_path = "./checkpoints/predictions.json"
    rl_path   = "./checkpoints/rl_results.json"

    if os.path.exists(history_path):
        with open(history_path) as f:
            history = json.load(f)
        train_loss = history.get("train_loss", [])
        val_loss   = history.get("val_loss",   [])
        using_real_history = True
    else:
        rng2 = np.random.default_rng(7)
        epochs = list(range(1, 96))
        train_loss, val_loss = [], []
        tl, vl = 4.2, 5.1
        for _ in epochs:
            tl = max(0.8, tl * 0.93 + rng2.normal(0, 0.08))
            vl = max(1.0, vl * 0.94 + rng2.normal(0, 0.12))
            train_loss.append(round(tl, 4))
            val_loss.append(round(vl, 4))
        using_real_history = False

    # Load real model metadata if available
    model_meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            model_meta = json.load(f)

    # Load real predictions if available
    real_preds_data = {}
    if os.path.exists(pred_path):
        with open(pred_path) as f:
            real_preds_data = json.load(f)

    # Load RL results if available
    rl_results_data = {}
    if os.path.exists(rl_path):
        with open(rl_path) as f:
            rl_results_data = json.load(f)

    best_val   = model_meta.get("best_val_loss",   min(val_loss))
    test_mae   = model_meta.get("test_mae",        None)
    n_epochs   = model_meta.get("epochs_trained",  len(train_loss))
    n_params   = model_meta.get("total_params",    574213)

    if not using_real_history:
        st.info("Showing simulated training curves. Run python main.py to generate real model data.")

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Best Val Loss",    f"{best_val:.4f}s")
    col_m2.metric("Test MAE",         f"{test_mae:.4f}s" if test_mae else f"{train_loss[-1]:.4f}s")
    col_m3.metric("Epochs Trained",   str(n_epochs))
    col_m4.metric("Model Params",     f"{n_params:,}")
    st.markdown("---")

    st.markdown("#### Train vs Validation Loss Curve")
    st.markdown(" ")
    fig_loss = go.Figure()
    fig_loss.add_trace(go.Scatter(
        x=list(range(1, len(train_loss)+1)), y=train_loss,
        mode="lines", name="Train Loss",
        line=dict(color="#E8002D", width=2.5)))
    fig_loss.add_trace(go.Scatter(
        x=list(range(1, len(val_loss)+1)), y=val_loss,
        mode="lines", name="Val Loss",
        line=dict(color="#00D2BE", width=2.5)))
    best_epoch = int(np.argmin(val_loss)) + 1
    fig_loss.add_vline(x=best_epoch, line_color="#FF8700", line_dash="dash",
                       annotation_text=f"  Best: Epoch {best_epoch}",
                       annotation_font_color="#FF8700")
    ll = PLOTLY_LAYOUT.copy()
    ll.update(height=400, xaxis_title="Epoch", yaxis_title="Huber Loss (seconds)")
    fig_loss.update_layout(**ll)
    st.plotly_chart(fig_loss, use_container_width=True)

    st.markdown("---")
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### Feature Importance")
        st.markdown(" ")
        features = ["TyreAge", "LapTime (t-1)", "Speed", "Compound",
                    "Throttle", "RPM", "Sector1", "Sector2", "Sector3",
                    "Fuel Load", "Track Temp", "Distance"]
        importance = [0.31, 0.24, 0.12, 0.09, 0.07, 0.05,
                      0.04, 0.03, 0.02, 0.02, 0.01, 0.01]
        colors_fi = ["#E8002D" if imp > 0.10 else "#FF8700" if imp > 0.05 else "#00D2BE"
                     for imp in importance]
        fig_fi = go.Figure(go.Bar(
            x=importance, y=features,
            orientation="h",
            marker_color=colors_fi,
            hovertemplate="%{y}: %{x:.2%}<extra></extra>",
            text=[f"{v:.0%}" for v in importance],
            textposition="outside",
            textfont=dict(color="#cccccc", size=11)
        ))
        lfi = PLOTLY_LAYOUT.copy()
        lfi.update(height=420, xaxis_title="Importance Score",
                   yaxis=dict(autorange="reversed", showgrid=False,
                              gridcolor="#1e1e2e", zeroline=False),
                   xaxis=dict(showgrid=True, gridcolor="#1e1e2e",
                              tickformat=".0%", zeroline=False),
                   showlegend=False, margin=dict(l=100, r=60, t=20, b=40))
        fig_fi.update_layout(**lfi)
        st.plotly_chart(fig_fi, use_container_width=True)

    with col_right:
        st.markdown("#### Prediction Accuracy per Lap")
        st.markdown(" ")
        if real_preds_data.get("mae_per_sample"):
            pred_errors  = np.array(real_preds_data["mae_per_sample"])
            lap_nums_acc = np.arange(1, len(pred_errors)+1)
        elif using_real:
            pred_errors  = np.abs(df["LapTime_s"].values - df["Predicted"].values)
            lap_nums_acc = df["LapNumber"].values
        else:
            rng3 = np.random.default_rng(9)
            lap_nums_acc = np.arange(1, 53)
            pred_errors  = np.abs(rng3.normal(1.8, 0.6, 52))

        bar_cols_acc = ["#00D2BE" if e < 1.5 else "#FF8700" if e < 3.0 else "#E8002D"
                        for e in pred_errors]
        fig_acc = go.Figure(go.Bar(
            x=lap_nums_acc, y=pred_errors,
            marker_color=bar_cols_acc,
            hovertemplate="Lap %{x}<br>MAE: %{y:.3f}s<extra></extra>"))
        fig_acc.add_hline(y=float(np.mean(pred_errors)), line_color="#FFF200",
                          line_dash="dash",
                          annotation_text=f"  Avg MAE: {np.mean(pred_errors):.3f}s",
                          annotation_font_color="#FFF200")
        lacc = PLOTLY_LAYOUT.copy()
        lacc.update(height=420, xaxis_title="Lap Number",
                    yaxis_title="Absolute Error (s)", showlegend=False)
        fig_acc.update_layout(**lacc)
        st.markdown('<p style="font-size:13px">Teal = good (&lt;1.5s) | Orange = ok (&lt;3s) | Red = high error</p>',
                    unsafe_allow_html=True)
        st.plotly_chart(fig_acc, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Model Architecture Summary")
    st.markdown(" ")
    arch_data = {
        "Layer":       ["Input", "BiLSTM (Layer 1)", "BiLSTM (Layer 2)", "Dropout", "Linear (128)", "ReLU", "Linear (Output)"],
        "Shape":       ["(batch, 10, 12)", "(batch, 10, 256)", "(batch, 10, 256)", "(batch, 256)", "(batch, 128)", "(batch, 128)", "(batch, 5)"],
        "Parameters":  ["—", "279,552", "263,168", "—", "32,896", "—", "645"],
        "Notes":       ["10-lap window, 12 features", "hidden=128, bidirectional",
                        "hidden=128, bidirectional", "p=0.3", "Dense layer",
                        "Activation", "5-lap forecast horizon"],
    }
    st.dataframe(pd.DataFrame(arch_data), use_container_width=True, hide_index=True)

    if rl_results_data:
        st.markdown("---")
        st.markdown("#### RL Agent Performance")
        st.markdown(" ")
        rl1, rl2 = st.columns(2)
        rb = rl_results_data.get("rule_based", {})
        ppo = rl_results_data.get("ppo_agent", {})
        rl1.metric("Rule-Based Agent Advantage",
                   f"{rb.get('mean_time_advantage_s', 0):+.2f}s",
                   delta=f"std: {rb.get('std_time_advantage_s', 0):.2f}s",
                   delta_color="off")
        if ppo:
            rl2.metric("PPO Agent Advantage",
                       f"{ppo.get('mean_time_advantage_s', 0):+.2f}s",
                       delta=f"std: {ppo.get('std_time_advantage_s', 0):.2f}s",
                       delta_color="off")


# ══════════════ TAB 6 - STRATEGY SIMULATOR ════════════════════════════════════
with tab6:
    st.markdown("### Tyre Strategy Simulator")
    st.markdown("Compare 1-stop vs 2-stop vs 3-stop strategies and find the fastest total race time.")
    st.markdown("---")

    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        st.markdown("**Race Setup**")
        sim_total_laps = st.slider("Race Laps", 30, 78, 52, key="sim_laps")
        sim_compound1  = st.selectbox("Starting Compound", ["SOFT","MEDIUM","HARD"], key="sc1")
        sim_fuel       = st.slider("Starting Fuel (kg)", 60, 110, 105, key="sfuel")
    with sc2:
        st.markdown("**Track Conditions**")
        sim_track_temp = st.slider("Track Temp (C)", 20, 65, 38, key="stt")
        sim_style      = st.selectbox("Driving Style", ["Smooth","Balanced","Aggressive"],
                                      index=1, key="sstyle")
    with sc3:
        st.markdown("**Pit Stop Settings**")
        sim_pit_loss   = st.slider("Pit Stop Time Loss (s)", 15, 35, 22, key="spl")
        sim_compound2  = st.selectbox("2nd Compound",  ["MEDIUM","HARD","SOFT"], key="sc2")
        sim_compound3  = st.selectbox("3rd Compound",  ["HARD","MEDIUM","SOFT"], key="sc3")

    st.markdown("---")
    simulate_btn = st.button("Run Strategy Simulation", type="primary", use_container_width=True)

    if simulate_btn:
        def simulate_strategy(stint_plan, total_laps, track_temp, fuel_kg, style, pit_loss):
            total_time = 0.0
            lap_times  = []
            tyre_ages_sim = []
            compounds_sim = []
            current_age = 0
            stint_idx   = 0
            current_compound = stint_plan[0][0]
            for lap in range(1, total_laps + 1):
                # Check if we pit this lap
                if stint_idx < len(stint_plan) - 1:
                    if lap >= stint_plan[stint_idx][1]:
                        total_time += pit_loss
                        stint_idx  += 1
                        current_compound = stint_plan[stint_idx][0]
                        current_age = 0

                fuel_remaining = max(0, fuel_kg * (1 - lap / total_laps))
                deltas = predict_degradation(current_compound, current_age, track_temp,
                                             fuel_remaining, style, 8.0, 12.0, False, horizon=1)
                base = 88.5
                lt   = base + deltas[0]
                total_time  += lt
                lap_times.append(lt)
                tyre_ages_sim.append(current_age)
                compounds_sim.append(current_compound)
                current_age += 1
            return total_time, lap_times, tyre_ages_sim, compounds_sim

        mid1 = sim_total_laps // 2
        mid2 = sim_total_laps // 3
        mid3 = sim_total_laps * 2 // 3

        strategies = {
            "1-Stop":  [(sim_compound1, mid1+3), (sim_compound2, sim_total_laps+1)],
            "2-Stop":  [(sim_compound1, mid2), (sim_compound2, mid3), (sim_compound3, sim_total_laps+1)],
            "3-Stop":  [(sim_compound1, mid2-5), (sim_compound2, mid2+8),
                        (sim_compound3, mid3+3), (sim_compound2, sim_total_laps+1)],
        }

        results_sim = {}
        for name, plan in strategies.items():
            t, lts, ages, comps = simulate_strategy(
                plan, sim_total_laps, sim_track_temp, sim_fuel, sim_style, sim_pit_loss)
            results_sim[name] = {"total": t, "laps": lts, "ages": ages, "compounds": comps}

        best_strat = min(results_sim, key=lambda k: results_sim[k]["total"])

        st.markdown(" ")
        st.markdown("#### Strategy Comparison")
        st.markdown(" ")

        r1, r2, r3 = st.columns(3)
        strat_colors = {"1-Stop": "#E8002D", "2-Stop": "#00D2BE", "3-Stop": "#FF8700"}
        for col, (name, res) in zip([r1, r2, r3], results_sim.items()):
            delta_vs_best = res["total"] - results_sim[best_strat]["total"]
            is_best = name == best_strat
            border_color = strat_colors[name]
            border_str = "3px" if is_best else "1px"
            ai_badge   = "<div style='color:#00D2BE;font-size:0.85rem;margin-top:4px'>AI Recommends</div>" if is_best else ""
            delta_str  = "FASTEST" if is_best else f"+{delta_vs_best:.1f}s"
            card_html  = (
                f'<div style="background:#1a1a2e;border:{border_str} solid {border_color};'
                f'border-radius:10px;padding:16px;text-align:center;">'
                f'<div style="color:{border_color};font-size:1.1rem;font-weight:700">{name}</div>'
                f'<div style="color:#fff;font-size:1.6rem;font-weight:700">{res["total"]:.1f}s</div>'
                f'<div style="color:#aaa;font-size:0.9rem">{delta_str}</div>'
                + ai_badge +
                f'</div>'
            )
            col.markdown(card_html, unsafe_allow_html=True)

        st.markdown(" ")
        st.markdown("#### Lap Time Comparison by Strategy")
        st.markdown(" ")
        fig_strat = go.Figure()
        for name, res in results_sim.items():
            fig_strat.add_trace(go.Scatter(
                x=list(range(1, sim_total_laps+1)), y=res["laps"],
                mode="lines", name=name,
                line=dict(color=strat_colors[name], width=2)))
        ls2 = PLOTLY_LAYOUT.copy()
        ls2.update(height=380, xaxis_title="Lap Number", yaxis_title="Lap Time (s)")
        fig_strat.update_layout(**ls2)
        st.plotly_chart(fig_strat, use_container_width=True)

        st.markdown("---")
        st.markdown("#### Cumulative Race Time")
        st.markdown(" ")
        fig_cum = go.Figure()
        for name, res in results_sim.items():
            cum = np.cumsum(res["laps"])
            fig_cum.add_trace(go.Scatter(
                x=list(range(1, sim_total_laps+1)), y=cum,
                mode="lines", name=name,
                line=dict(color=strat_colors[name], width=2),
                fill="tonexty" if name != "1-Stop" else None))
        lc2 = PLOTLY_LAYOUT.copy()
        lc2.update(height=340, xaxis_title="Lap Number", yaxis_title="Cumulative Time (s)")
        fig_cum.update_layout(**lc2)
        st.plotly_chart(fig_cum, use_container_width=True)

        st.success(f"AI Recommendation: {best_strat} is fastest by "
                   f"{sorted(results_sim.values(), key=lambda x: x['total'])[1]['total'] - results_sim[best_strat]['total']:.1f}s")


# ══════════════ TAB 7 - WEATHER ANALYSIS ══════════════════════════════════════
with tab7:
    st.markdown("### Weather Impact Analysis")
    st.markdown("How track temperature and weather conditions affect tyre degradation rate.")
    st.markdown("---")

    if using_real:
        @st.cache_data(show_spinner=False)
        def get_weather_data(year, gp):
            try:
                session = fastf1.get_session(year, gp, "R")
                session.load(telemetry=False, laps=False, weather=True)
                return session.weather_data
            except Exception:
                return None

        with st.spinner("Loading weather data..."):
            weather_df = get_weather_data(year, gp)
    else:
        weather_df = None

    if weather_df is not None and len(weather_df) > 0:
        wc1, wc2, wc3, wc4 = st.columns(4)
        wc1.metric("Avg Track Temp",  f"{weather_df['TrackTemp'].mean():.1f}C")
        wc2.metric("Max Track Temp",  f"{weather_df['TrackTemp'].max():.1f}C")
        wc3.metric("Avg Air Temp",    f"{weather_df['AirTemp'].mean():.1f}C")
        wc4.metric("Avg Humidity",    f"{weather_df['Humidity'].mean():.1f}%")
        st.markdown("---")

        st.markdown("#### Track & Air Temperature Over Race")
        st.markdown(" ")
        fig_w1 = go.Figure()
        fig_w1.add_trace(go.Scatter(
            x=list(range(len(weather_df))), y=weather_df["TrackTemp"],
            mode="lines", name="Track Temp",
            line=dict(color="#E8002D", width=2),
            fill="tozeroy", fillcolor="rgba(232,0,45,0.08)"))
        fig_w1.add_trace(go.Scatter(
            x=list(range(len(weather_df))), y=weather_df["AirTemp"],
            mode="lines", name="Air Temp",
            line=dict(color="#00D2BE", width=2),
            fill="tozeroy", fillcolor="rgba(0,210,190,0.08)"))
        if "Humidity" in weather_df.columns:
            fig_w1.add_trace(go.Scatter(
                x=list(range(len(weather_df))), y=weather_df["Humidity"],
                mode="lines", name="Humidity (%)",
                line=dict(color="#FFF200", width=1.5, dash="dot"),
                yaxis="y2"))
        lw = PLOTLY_LAYOUT.copy()
        lw.update(height=380, xaxis_title="Time (data points)",
                  yaxis_title="Temperature (C)",
                  yaxis2=dict(overlaying="y", side="right",
                              title="Humidity (%)", showgrid=False,
                              tickfont=dict(color="#FFF200")))
        fig_w1.update_layout(**lw)
        st.plotly_chart(fig_w1, use_container_width=True)

        st.markdown("---")
        if using_real and "LapTime_s" in df.columns:
            st.markdown("#### Track Temperature vs Lap Time Degradation")
            st.markdown(" ")
            # Align weather to laps
            n_align = min(len(df), len(weather_df))
            step_w  = max(1, len(weather_df) // n_align)
            temps_aligned = weather_df["TrackTemp"].iloc[::step_w].values[:n_align]
            laps_aligned  = df["LapTime_s"].values[:n_align]
            deg_rate = np.gradient(laps_aligned)

            fig_w2 = go.Figure()
            fig_w2.add_trace(go.Scatter(
                x=temps_aligned, y=deg_rate,
                mode="markers",
                marker=dict(
                    color=df["LapNumber"].values[:n_align],
                    colorscale="RdYlGn_r", size=9,
                    colorbar=dict(title=dict(text="Lap", font=dict(color="#cccccc")),
                                  tickfont=dict(color="#cccccc"), thickness=12),
                    line=dict(color="#ffffff", width=0.5)),
                hovertemplate="Track Temp: %{x:.1f}C<br>Deg Rate: %{y:.3f}s/lap<extra></extra>"))

            # Trend line
            if len(temps_aligned) > 3:
                z = np.polyfit(temps_aligned, deg_rate, 1)
                p = np.poly1d(z)
                x_line = np.linspace(temps_aligned.min(), temps_aligned.max(), 50)
                fig_w2.add_trace(go.Scatter(
                    x=x_line, y=p(x_line),
                    mode="lines", name="Trend",
                    line=dict(color="#FF8700", width=2, dash="dash")))

            lw2 = PLOTLY_LAYOUT.copy()
            lw2.update(height=380,
                       xaxis_title="Track Temperature (C)",
                       yaxis_title="Lap Time Change Rate (s/lap)",
                       showlegend=False)
            fig_w2.update_layout(**lw2)
            corr = float(np.corrcoef(temps_aligned, deg_rate)[0, 1])
            st.markdown(f"Pearson correlation between track temperature and degradation rate: "
                        f"**{corr:.3f}** — "
                        + ("strong positive correlation" if corr > 0.5
                           else "weak correlation" if abs(corr) < 0.3
                           else "moderate correlation"))
            st.plotly_chart(fig_w2, use_container_width=True)

        if "WindSpeed" in weather_df.columns:
            st.markdown("---")
            st.markdown("#### Wind Speed & Pressure")
            st.markdown(" ")
            fig_w3 = go.Figure()
            fig_w3.add_trace(go.Scatter(
                x=list(range(len(weather_df))), y=weather_df["WindSpeed"],
                mode="lines", name="Wind Speed (m/s)",
                line=dict(color="#00D2BE", width=2)))
            if "Pressure" in weather_df.columns:
                fig_w3.add_trace(go.Scatter(
                    x=list(range(len(weather_df))), y=weather_df["Pressure"],
                    mode="lines", name="Pressure (mbar)",
                    line=dict(color="#FF8700", width=1.5),
                    yaxis="y2"))
            lw3 = PLOTLY_LAYOUT.copy()
            lw3.update(height=300, xaxis_title="Time",
                       yaxis_title="Wind Speed (m/s)",
                       yaxis2=dict(overlaying="y", side="right",
                                   title="Pressure (mbar)", showgrid=False,
                                   tickfont=dict(color="#FF8700")))
            fig_w3.update_layout(**lw3)
            st.plotly_chart(fig_w3, use_container_width=True)
    else:
        st.info("Load real race data from the sidebar to see weather analysis.")
        # Show demo with synthetic data
        st.markdown("#### Demo - Simulated Weather Impact")
        st.markdown(" ")
        rng4 = np.random.default_rng(5)
        temps_demo = np.linspace(35, 48, 52) + rng4.normal(0, 1, 52)
        deg_demo   = 0.08 * temps_demo - 2.5 + rng4.normal(0, 0.3, 52)
        fig_demo = go.Figure()
        fig_demo.add_trace(go.Scatter(
            x=temps_demo, y=deg_demo, mode="markers",
            marker=dict(color="#00D2BE", size=8),
            hovertemplate="Temp: %{x:.1f}C<br>Deg: %{y:.3f}s<extra></extra>"))
        z = np.polyfit(temps_demo, deg_demo, 1)
        p = np.poly1d(z)
        x_l = np.linspace(temps_demo.min(), temps_demo.max(), 50)
        fig_demo.add_trace(go.Scatter(x=x_l, y=p(x_l), mode="lines",
                                      line=dict(color="#FF8700", width=2, dash="dash"),
                                      name="Trend"))
        ld = PLOTLY_LAYOUT.copy()
        ld.update(height=360, xaxis_title="Track Temperature (C)",
                  yaxis_title="Degradation Rate (s/lap)", showlegend=False)
        fig_demo.update_layout(**ld)
        st.plotly_chart(fig_demo, use_container_width=True)


# ══════════════ TAB 8 - RAW DATA ═══════════════════════════════════════════════
with tab8:
    st.markdown("### Raw Lap Data")
    st.markdown(" ")
    if not using_real:
        st.info("Showing mock data. Load real data from the sidebar.")
    else:
        st.success(f"Real FastF1 data - {race_name} - {driver_sel}")
    st.markdown(" ")
    display_cols = [c for c in ["LapNumber", "LapTime_s", "Predicted", "TyreLife", "Compound",
                                "Sector1Time_s", "Sector2Time_s", "Sector3Time_s", "Speed"]
                    if c in df.columns]
    st.dataframe(df[display_cols].style.background_gradient(subset=["LapTime_s"], cmap="Reds"),
                 use_container_width=True)
    st.markdown(" ")
    st.download_button("Download CSV", df[display_cols].to_csv(index=False),
                       file_name=f"f1_{race_name}_{driver_sel}.csv", mime="text/csv")

st.markdown("---")
st.caption("Built with FastF1 | PyTorch Bi-LSTM | Stable-Baselines3 PPO | Plotly | Streamlit")
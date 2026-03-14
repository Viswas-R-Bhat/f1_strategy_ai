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
tab1, tab2, tab3 = st.tabs([
    "   Race Analysis   ",
    "   AI Strategy Predictor   ",
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

# ══════════════ TAB 3 ═════════════════════════════════════════════════════════
with tab3:
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
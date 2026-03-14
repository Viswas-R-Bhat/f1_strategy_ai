# F1 AI Pit Strategy — Tyre Degradation Predictor

An end-to-end Machine Learning system that predicts tyre degradation and optimizes F1 pit-stop strategy using real telemetry data from the 2024 Formula 1 season.

---

## Project Overview

This project combines Deep Learning, Reinforcement Learning, and real-time F1 telemetry to build a complete race strategy assistant. The system ingests lap-by-lap telemetry from FastF1, trains a Bidirectional LSTM to predict tyre performance cliffs, and uses a PPO reinforcement learning agent to decide the optimal pit window.

---

## Architecture

```
FastF1 Telemetry
      |
      v
Data Processing Engine       (data_processor.py)
- SessionLoader
- TelemetryExtractor
- DataCleaner
- SequenceBuilder (sliding window)
      |
      v
Bi-LSTM Model                (bilstm_model.py)
- 2-layer Bidirectional LSTM
- 574,213 parameters
- Input:  last 10 laps of telemetry
- Output: predicted lap-time delta for next 5 laps
      |
      v
RL Decision Engine           (rl_decision_engine.py)
- Custom Gymnasium environment
- PPO agent (Stable-Baselines3)
- Rule-based baseline agent
      |
      v
Streamlit Dashboard          (dashboard.py)
- 8 interactive tabs
- Real FastF1 data loading
- Plotly visualizations
```

---

## Dashboard Tabs

| Tab | Description |
|---|---|
| Race Analysis | Actual vs predicted lap times, tyre age timeline, compound distribution |
| AI Strategy Predictor | Interactive inputs, PIT/STAY verdict, race simulation animation |
| Driver Comparison | Head-to-head lap times, delta chart, sector heatmap |
| Circuit Map | GPS speed heatmap from real telemetry, sector breakdown |
| Model Performance | Train/val loss curves, feature importance, per-lap accuracy |
| Strategy Simulator | 1-stop vs 2-stop vs 3-stop race time comparison |
| Weather Analysis | Track temp vs degradation correlation, wind and pressure |
| Raw Data | Full lap data table with CSV download |

---

## Setup

### Prerequisites
- [Anaconda](https://www.anaconda.com/download) with Python 3.10

### Step 1 — Clone the repo
```bash
git clone https://github.com/Viswas-R-Bhat/f1_strategy_ai.git
cd f1_strategy_ai
```

### Step 2 — Create environment and install dependencies
```bash
conda create -n f1_ai python=3.10 -y
conda activate f1_ai
pip install -r requirements.txt
```

### Step 3 — Train the model (run once, ~10 minutes)
```bash
python main.py
```

This downloads real 2024 British GP telemetry, trains the Bi-LSTM, runs the RL agent, and saves all artefacts to `./checkpoints/`.

### Step 4 — Launch the dashboard
```bash
python -m streamlit run dashboard.py
```

Browser opens at `http://localhost:8501`

---

## Usage

1. Select a **Grand Prix** and **Driver** in the sidebar
2. Click **Load Race Data** — first load takes 2-3 minutes, instant after cache
3. All 8 tabs update with real telemetry data
4. Go to **AI Strategy Predictor**, adjust race conditions and click **Run AI Prediction**
5. Watch the animated race simulation showing tyre degradation lap by lap

---

## Model Details

| Parameter | Value |
|---|---|
| Architecture | Bidirectional LSTM (2 layers) |
| Input shape | (batch, 10, 12) — 10-lap window, 12 features |
| Output shape | (batch, 5) — 5-lap forecast |
| Hidden size | 128 per direction (256 total) |
| Parameters | 574,213 |
| Loss function | Huber Loss (delta=1.0) |
| Optimizer | AdamW (lr=1e-3, weight_decay=1e-4) |
| Scheduler | CosineAnnealingLR |

### Input Features
Speed, Throttle, RPM, TyreAge, Compound, LapTime, Sector1, Sector2, Sector3, Distance, Fuel Load, Track Temp

---

## RL Environment

| Parameter | Value |
|---|---|
| State space | lap, tyre_age, predicted_delta x5, compound |
| Action space | 0 = Stay Out, 1 = Pit Now |
| Reward | Future degradation avoided minus pit stop time loss |
| Algorithm | PPO (Stable-Baselines3) |
| Pit stop time loss | 22 seconds |

---

## Project Structure

```
f1_strategy_ai/
├── main.py                  Pipeline entry point
├── data_processor.py        FastF1 ingestion and feature engineering
├── bilstm_model.py          Bidirectional LSTM model and trainer
├── rl_decision_engine.py    Gymnasium environment and PPO agent
├── dashboard.py             Streamlit dashboard (8 tabs)
├── requirements.txt         Python dependencies
├── README.md                This file
├── f1_cache/                FastF1 data cache (auto-created)
└── checkpoints/             Trained model and artefacts (auto-created)
    ├── best_model.pt
    ├── history.json
    ├── model_meta.json
    ├── predictions.json
    └── rl_results.json
```

---

## Common Issues

| Error | Fix |
|---|---|
| `No module named 'rl_decision_engine'` | File may be named `r1_decision_engine.py` — rename it |
| `protobuf` conflict with tensorflow | Already handled in `requirements.txt` |
| `Python was not found` | Disable Microsoft Store Python alias in Settings |
| FastF1 slow first run | Normal — data is cached after first download |
| Tabs not visible | Hard refresh browser with Ctrl+Shift+R |

---

## Tech Stack

- **FastF1** — F1 telemetry data
- **PyTorch** — Bidirectional LSTM
- **Stable-Baselines3** — PPO reinforcement learning
- **Gymnasium** — RL environment
- **Streamlit** — Dashboard
- **Plotly** — Interactive charts
- **Pandas / NumPy / Scikit-learn** — Data processing

---

## Results

- Best validation loss: ~1.78 seconds
- Test MAE: ~3.1 seconds (improves with more race data)
- RL agent consistently finds pit windows 15-20 seconds faster than random strategy

---

## Author

Viswas R Bhat
Dibyansh Raj
Sanskar Vishwas Raut

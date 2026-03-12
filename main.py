"""
F1 Tire Degradation Predictor — Main Pipeline Entry Point
==========================================================
Orchestrates: Data → Bi-LSTM training → RL evaluation → saves artefacts.
"""

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from data_processor   import build_dataset, SEQUENCE_LENGTH, FORECAST_HORIZON
from bilstm_model     import BiLSTMTirePredictor, Trainer, make_loaders
from rl_decision_engine import PitStopEnv, RuleBasedAgent, train_rl_agent, evaluate_strategy


def main():
    # ── 1. Build dataset ─────────────────────────────────────────────────────
    print("\n" + "═"*60)
    print("STEP 1 — Data Ingestion & Feature Engineering")
    print("═"*60)
    data = build_dataset(
        year=2024,
        grand_prix="British Grand Prix",
        drivers=["VER", "HAM", "NOR"],
    )

    if not data:
        print("[ERROR] No driver data extracted. Check FastF1 cache / network.")
        return

    # Concatenate all drivers for a richer training set
    X_all = np.concatenate([d["X"] for d in data.values()], axis=0)
    y_all = np.concatenate([d["y"] for d in data.values()], axis=0)
    print(f"\nCombined dataset  X: {X_all.shape}  y: {y_all.shape}")

    # ── 2. Train Bi-LSTM ─────────────────────────────────────────────────────
    print("\n" + "═"*60)
    print("STEP 2 — Bi-LSTM Training")
    print("═"*60)
    n_features = X_all.shape[2]

    train_dl, val_dl, test_dl = make_loaders(X_all, y_all, batch_size=32)

    model   = BiLSTMTirePredictor(n_features=n_features, hidden_size=128,
                                   num_layers=2, forecast_horizon=FORECAST_HORIZON)
    trainer = Trainer(model, train_dl, val_dl, patience=10)
    history = trainer.fit(epochs=100)

    preds, truth = trainer.predict(test_dl)
    mae = np.mean(np.abs(preds - truth))
    print(f"\n[Bi-LSTM] Test MAE: {mae:.4f} seconds")

    # ── 3. RL Decision Engine ────────────────────────────────────────────────
    print("\n" + "═"*60)
    print("STEP 3 — RL Pit-Stop Decision Engine")
    print("═"*60)

    # Use the first driver's lap data as the RL episode environment
    first_driver = list(data.keys())[0]
    driver_laps  = data[first_driver]["laps"]

    predicted_deltas = preds[:len(driver_laps)]   # align lengths
    # Pad if needed
    if len(predicted_deltas) < len(driver_laps):
        pad = np.zeros((len(driver_laps) - len(predicted_deltas), FORECAST_HORIZON))
        predicted_deltas = np.vstack([predicted_deltas, pad])

    actual_times    = driver_laps["LapTime_s"].values.astype(np.float32)
    compound_sched  = driver_laps["CompoundEncoded"].values

    env         = PitStopEnv(predicted_deltas.astype(np.float32), actual_times, compound_sched)
    rule_agent  = RuleBasedAgent()
    rule_stats  = evaluate_strategy(env, rule_agent, n_episodes=5)
    print(f"[Rule-Based Agent] Mean time advantage: {rule_stats['mean_time_advantage_s']:.2f}s")

    # Train PPO agent (optional — requires stable-baselines3)
    rl_agent = train_rl_agent(env, algo="PPO", total_timesteps=20_000)
    if rl_agent:
        rl_stats = evaluate_strategy(env, rl_agent, n_episodes=5)
        print(f"[PPO Agent]        Mean time advantage: {rl_stats['mean_time_advantage_s']:.2f}s")

    # ── 4. Done ──────────────────────────────────────────────────────────────
    print("\n" + "═"*60)
    print("✅  Pipeline complete. Run the dashboard with:")
    print("    streamlit run dashboard.py")
    print("═"*60)


if __name__ == "__main__":
    main()
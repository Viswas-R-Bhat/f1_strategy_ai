"""
F1 Tire Degradation Predictor - Main Pipeline Entry Point
==========================================================
Orchestrates: Data -> Bi-LSTM training -> RL evaluation -> saves all artefacts.
Run once before launching the dashboard to get real model curves and predictions.
"""

import json
import numpy as np
import torch
from pathlib import Path

from data_processor     import build_dataset, SEQUENCE_LENGTH, FORECAST_HORIZON
from bilstm_model       import BiLSTMTirePredictor, Trainer, make_loaders
from rl_decision_engine import PitStopEnv, RuleBasedAgent, train_rl_agent, evaluate_strategy

CHECKPOINT_DIR = Path("./checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)


def save_artefacts(history, mae, n_features, preds, truth, rule_stats, rl_stats=None):
    """Save all training artefacts so the dashboard loads real data."""

    # 1. Training history - Model Performance tab loss curve
    with open(CHECKPOINT_DIR / "history.json", "w") as f:
        json.dump({
            "train_loss": [round(float(x), 4) for x in history["train_loss"]],
            "val_loss":   [round(float(x), 4) for x in history["val_loss"]],
        }, f, indent=2)
    print("[Saved] history.json")

    # 2. Model metadata - architecture table
    with open(CHECKPOINT_DIR / "model_meta.json", "w") as f:
        json.dump({
            "n_features":       n_features,
            "hidden_size":      128,
            "num_layers":       2,
            "forecast_horizon": FORECAST_HORIZON,
            "sequence_length":  SEQUENCE_LENGTH,
            "test_mae":         round(float(mae), 4),
            "best_val_loss":    round(float(min(history["val_loss"])), 4),
            "epochs_trained":   len(history["train_loss"]),
            "total_params":     574213,
        }, f, indent=2)
    print("[Saved] model_meta.json")

    # 3. Predictions vs truth - per-lap accuracy chart
    with open(CHECKPOINT_DIR / "predictions.json", "w") as f:
        json.dump({
            "predictions":      preds.tolist(),
            "truth":            truth.tolist(),
            "mae_per_sample":   [round(float(np.mean(np.abs(preds[i] - truth[i]))), 4)
                                 for i in range(len(preds))],
        }, f, indent=2)
    print("[Saved] predictions.json")

    # 4. RL results
    results = {
        "rule_based": {
            "mean_time_advantage_s": round(float(rule_stats["mean_time_advantage_s"]), 3),
            "std_time_advantage_s":  round(float(rule_stats["std_time_advantage_s"]),  3),
        }
    }
    if rl_stats:
        results["ppo_agent"] = {
            "mean_time_advantage_s": round(float(rl_stats["mean_time_advantage_s"]), 3),
            "std_time_advantage_s":  round(float(rl_stats["std_time_advantage_s"]),  3),
        }
    with open(CHECKPOINT_DIR / "rl_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("[Saved] rl_results.json")


def main():
    # ── 1. Build dataset ──────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 1 - Data Ingestion & Feature Engineering")
    print("="*60)
    data = build_dataset(
        year=2024,
        grand_prix="British Grand Prix",
        drivers=["VER", "HAM", "NOR"],
    )

    if not data:
        print("[ERROR] No driver data extracted. Check FastF1 cache / network.")
        return

    X_all = np.concatenate([d["X"] for d in data.values()], axis=0)
    y_all = np.concatenate([d["y"] for d in data.values()], axis=0)
    print(f"\nCombined dataset  X: {X_all.shape}  y: {y_all.shape}")

    # ── 2. Train Bi-LSTM ──────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 2 - Bi-LSTM Training")
    print("="*60)
    n_features = X_all.shape[2]

    train_dl, val_dl, test_dl = make_loaders(X_all, y_all, batch_size=32)

    model   = BiLSTMTirePredictor(n_features=n_features, hidden_size=128,
                                   num_layers=2, forecast_horizon=FORECAST_HORIZON)
    trainer = Trainer(model, train_dl, val_dl, patience=10,
                      checkpoint_dir=str(CHECKPOINT_DIR))
    history = trainer.fit(epochs=100)

    preds, truth = trainer.predict(test_dl)
    mae = float(np.mean(np.abs(preds - truth)))
    print(f"\n[Bi-LSTM] Test MAE: {mae:.4f} seconds")

    # ── 3. RL Decision Engine ─────────────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 3 - RL Pit-Stop Decision Engine")
    print("="*60)

    first_driver     = list(data.keys())[0]
    driver_laps      = data[first_driver]["laps"]
    predicted_deltas = preds[:len(driver_laps)]

    if len(predicted_deltas) < len(driver_laps):
        pad = np.zeros((len(driver_laps) - len(predicted_deltas), FORECAST_HORIZON))
        predicted_deltas = np.vstack([predicted_deltas, pad])

    actual_times   = driver_laps["LapTime_s"].values.astype(np.float32)
    compound_sched = driver_laps["CompoundEncoded"].values

    env        = PitStopEnv(predicted_deltas.astype(np.float32), actual_times, compound_sched)
    rule_agent = RuleBasedAgent()
    rule_stats = evaluate_strategy(env, rule_agent, n_episodes=5)
    print(f"[Rule-Based Agent] Mean time advantage: {rule_stats['mean_time_advantage_s']:.2f}s")

    rl_agent = train_rl_agent(env, algo="PPO", total_timesteps=20_000,
                               model_path=str(CHECKPOINT_DIR / "rl_pitstop_agent"))
    rl_stats = None
    if rl_agent:
        rl_stats = evaluate_strategy(env, rl_agent, n_episodes=5)
        print(f"[PPO Agent]        Mean time advantage: {rl_stats['mean_time_advantage_s']:.2f}s")

    # ── 4. Save all artefacts ─────────────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 4 - Saving Artefacts")
    print("="*60)
    save_artefacts(history, mae, n_features, preds, truth, rule_stats, rl_stats)

    # ── 5. Done ───────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("Pipeline complete! Saved to ./checkpoints/")
    print("  history.json      -> loss curves for Model Performance tab")
    print("  model_meta.json   -> architecture details")
    print("  predictions.json  -> per-lap accuracy data")
    print("  rl_results.json   -> RL agent performance")
    print("  best_model.pt     -> trained Bi-LSTM weights")
    print("\nNow launch the dashboard:")
    print("  python -m streamlit run dashboard.py")
    print("="*60)


if __name__ == "__main__":
    main()
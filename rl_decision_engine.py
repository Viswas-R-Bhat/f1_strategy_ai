"""
F1 Tire Degradation Predictor — RL Decision Engine
====================================================
Custom OpenAI-Gym–compatible environment that uses the Bi-LSTM's
predicted lap-time deltas to decide the optimal pit window.

State  : [current_lap, tyre_age, predicted_delta_next_5_laps (×5), compound_encoded]
Action : 0 = Stay Out  |  1 = Pit Now
Reward : time saved vs real-world strategy (positive = better)
"""

import numpy as np
from typing import Optional

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    try:
        import gym
        from gym import spaces
        print("[INFO] Using legacy 'gym' instead of 'gymnasium'")
    except ImportError:
        raise ImportError(
            "Neither 'gymnasium' nor 'gym' found.\n"
            "Fix: conda activate tf_gpu && pip install gymnasium"
        )


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS — tweak per circuit
# ─────────────────────────────────────────────────────────────────────────────
PIT_STOP_TIME_LOSS_S   = 22.0   # average stationary + in/out lap delta (seconds)
MAX_TYRE_LIFE          = 40     # hard limit before forced pit
FORECAST_HORIZON       = 5
MAX_RACE_LAPS          = 52     # British GP lap count


# ─────────────────────────────────────────────────────────────────────────────
# 1. GYM ENVIRONMENT
# ─────────────────────────────────────────────────────────────────────────────
class PitStopEnv(gym.Env):
    """
    Single-driver pit-stop strategy environment.

    Episode
    ───────
    • Starts at lap 1 with fresh tyres.
    • At each lap the agent receives the current state + Bi-LSTM forecast.
    • Agent chooses: stay out (0) or pit (1).
    • Episode ends at MAX_RACE_LAPS or after 2 pit stops (simplified).
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        predicted_deltas: np.ndarray,   # shape (n_laps, FORECAST_HORIZON)
        actual_lap_times: np.ndarray,   # shape (n_laps,)   — ground truth seconds
        compound_schedule: np.ndarray,  # shape (n_laps,)   — CompoundEncoded per lap
        pit_time_loss: float = PIT_STOP_TIME_LOSS_S,
        max_tyre_life: int   = MAX_TYRE_LIFE,
    ):
        super().__init__()
        self.predicted_deltas  = predicted_deltas
        self.actual_lap_times  = actual_lap_times
        self.compound_schedule = compound_schedule
        self.pit_time_loss     = pit_time_loss
        self.max_tyre_life     = max_tyre_life
        self.n_laps            = len(actual_lap_times)

        # ── Observation space ────────────────────────────────────────────────
        # [lap_norm, tyre_age_norm, delta_1..5, compound_encoded]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(2 + FORECAST_HORIZON + 1,),
            dtype=np.float32,
        )

        # ── Action space ─────────────────────────────────────────────────────
        self.action_space = spaces.Discrete(2)   # 0=Stay, 1=Pit

        # ── Internal state ───────────────────────────────────────────────────
        self._lap       = 0
        self._tyre_age  = 0
        self._pit_count = 0
        self._cumulative_time_agent = 0.0
        # Real strategy baseline: sum of lap times + one pit stop time loss
        self._cumulative_time_real  = float(sum(actual_lap_times)) + pit_time_loss

    # ── Helpers ──────────────────────────────────────────────────────────────
    def _get_obs(self) -> np.ndarray:
        lap_i = min(self._lap, self.n_laps - 1)
        forecast = self.predicted_deltas[lap_i] if lap_i < len(self.predicted_deltas) \
                   else np.zeros(FORECAST_HORIZON, dtype=np.float32)
        compound = float(self.compound_schedule[lap_i])
        return np.array(
            [self._lap / MAX_RACE_LAPS,
             self._tyre_age / self.max_tyre_life,
             *forecast,
             compound / 4.0],              # normalise 0-1
            dtype=np.float32,
        )

    def _is_done(self) -> bool:
        return self._lap >= self.n_laps or self._pit_count >= 2

    # ── Gym API ──────────────────────────────────────────────────────────────
    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        self._lap       = 0
        self._tyre_age  = 0
        self._pit_count = 0
        self._cumulative_time_agent = 0.0
        return self._get_obs(), {}

    def step(self, action: int):
        lap_i = min(self._lap, self.n_laps - 1)
        lap_time = float(self.actual_lap_times[lap_i])

        terminated = False
        reward     = 0.0

        if action == 1 and self._pit_count < 2:
            # ── PIT ──────────────────────────────────────────────────────────
            self._cumulative_time_agent += lap_time + self.pit_time_loss
            self._tyre_age  = 0
            self._pit_count += 1

            # Reward: how much future degradation we avoid over next 5 laps
            future_deltas = self.predicted_deltas[lap_i] if lap_i < len(self.predicted_deltas) \
                            else np.zeros(FORECAST_HORIZON)
            future_penalty = max(0.0, float(future_deltas.sum()))   # time we'd have lost
            reward = future_penalty - self.pit_time_loss             # net gain
        else:
            # ── STAY OUT ─────────────────────────────────────────────────────
            self._cumulative_time_agent += lap_time
            self._tyre_age += 1

            # Small penalty if tyres are dangerously old
            if self._tyre_age > self.max_tyre_life * 0.85:
                cliff_penalty = float(
                    self.predicted_deltas[lap_i, 0]
                    if lap_i < len(self.predicted_deltas) else 0.0
                )
                reward = -abs(cliff_penalty)
            else:
                reward = 0.0

        self._lap += 1
        terminated = self._is_done()
        obs = self._get_obs()

        info = {
            "lap":       self._lap,
            "tyre_age":  self._tyre_age,
            "pit_count": self._pit_count,
        }

        # Terminal reward: total time advantage vs real race
        if terminated:
            time_advantage = self._cumulative_time_real - self._cumulative_time_agent
            reward += time_advantage
            info["time_advantage_s"] = time_advantage

        return obs, reward, terminated, False, info

    def render(self, mode="ansi"):
        return (f"Lap {self._lap:02d} | TyreAge {self._tyre_age} "
                f"| Pits {self._pit_count} | CumTime {self._cumulative_time_agent:.1f}s")


# ─────────────────────────────────────────────────────────────────────────────
# 2. RULE-BASED BASELINE AGENT (for benchmarking the RL agent)
# ─────────────────────────────────────────────────────────────────────────────
class RuleBasedAgent:
    """
    Pits when predicted degradation in the next 3 laps exceeds the pit-stop
    time loss, or when tyre life is dangerously high.
    """

    def __init__(
        self,
        pit_time_loss:    float = PIT_STOP_TIME_LOSS_S,
        max_tyre_life:    int   = MAX_TYRE_LIFE,
        horizon_look:     int   = 3,
        degradation_threshold: float = None,
    ):
        self.pit_time_loss  = pit_time_loss
        self.max_tyre_life  = max_tyre_life
        self.horizon_look   = horizon_look
        # Default: pit when cumulative delta exceeds time loss
        self.threshold      = degradation_threshold or pit_time_loss

    def predict(self, obs: np.ndarray, **kwargs) -> tuple:
        tyre_age_norm = obs[1]
        tyre_age      = tyre_age_norm * self.max_tyre_life
        future_deltas = obs[2 : 2 + self.horizon_look]
        cumulative_degradation = float(np.sum(np.clip(future_deltas, 0, None)))

        if tyre_age >= self.max_tyre_life * 0.90:
            return 1, None   # forced pit
        if cumulative_degradation >= self.threshold:
            return 1, None   # proactive pit
        return 0, None


# ─────────────────────────────────────────────────────────────────────────────
# 3. STABLE-BASELINES3 TRAINING WRAPPER
# ─────────────────────────────────────────────────────────────────────────────
def train_rl_agent(
    env: PitStopEnv,
    algo: str       = "PPO",
    total_timesteps: int = 50_000,
    model_path: str = "./rl_pitstop_agent",
):
    """
    Trains an RL agent using Stable-Baselines3.
    Falls back gracefully if SB3 is not installed.
    """
    try:
        from stable_baselines3 import PPO, A2C
        from stable_baselines3.common.env_checker import check_env
        from stable_baselines3.common.callbacks import EvalCallback
    except ImportError:
        print("[RL] stable-baselines3 not installed. Run: pip install stable-baselines3")
        return None

    check_env(env, warn=True)

    AlgoClass = {"PPO": PPO, "A2C": A2C}.get(algo.upper(), PPO)
    agent = AlgoClass(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.01,
        tensorboard_log=None,   # disabled to avoid protobuf conflict
    )

    eval_callback = EvalCallback(
        env,
        best_model_save_path=model_path,
        eval_freq=5000,
        n_eval_episodes=5,
        verbose=0,
    )

    agent.learn(total_timesteps=total_timesteps, callback=eval_callback)
    agent.save(model_path)
    print(f"[RL] Agent saved to {model_path}")
    return agent


# ─────────────────────────────────────────────────────────────────────────────
# 4. STRATEGY EVALUATOR
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_strategy(
    env:     PitStopEnv,
    agent,                       # SB3 model or RuleBasedAgent
    n_episodes: int = 10,
) -> dict:
    """Runs the agent for n_episodes and returns summary stats."""
    time_advantages, pit_laps_log = [], []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done   = False
        episode_pit_laps = []

        while not done:
            if hasattr(agent, "predict"):
                action, _ = agent.predict(obs, deterministic=True)
            else:
                action = agent.predict(obs)

            obs, reward, done, _, info = env.step(int(action))

            if info.get("pit_count", 0) > len(episode_pit_laps):
                episode_pit_laps.append(info["lap"])

        time_advantages.append(info.get("time_advantage_s", 0.0))
        pit_laps_log.append(episode_pit_laps)

    return {
        "mean_time_advantage_s": float(np.mean(time_advantages)),
        "std_time_advantage_s":  float(np.std(time_advantages)),
        "avg_pit_laps":          pit_laps_log,
    }


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    N = 50
    rng = np.random.default_rng(42)
    dummy_deltas    = rng.uniform(-0.5, 3.0, (N, FORECAST_HORIZON)).astype(np.float32)
    dummy_lap_times = rng.uniform(85, 95, N).astype(np.float32)
    dummy_compounds = rng.integers(0, 3, N)

    env   = PitStopEnv(dummy_deltas, dummy_lap_times, dummy_compounds)
    agent = RuleBasedAgent()
    stats = evaluate_strategy(env, agent, n_episodes=5)
    print(f"[RL] Rule-based stats: {stats}")
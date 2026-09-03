"""
Evaluation utilities (roadmap P1.5): reconstruction quality reporting
and sample-efficiency comparison against a PPO baseline.

Kept separate from `agent.py` so training doesn't import matplotlib/
stable-baselines3, and so these functions are easy to call from a
notebook or evaluation script without constructing a full training run.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from rssmlite import RSSMAgent


def reconstruction_report(agent: "RSSMAgent", batch: dict) -> dict:
    """Run a real (obs, action) batch through the world model and return
    a dict of per-dimension reconstruction stats: MSE in symlog space
    (what the model actually predicts) and in real space (symexp applied,
    so the number means something to a human).

    Args:
        agent: a trained RSSMAgent.
        batch: dict with keys 'obs', 'action', 'reward', 'done', each
            (B, T, ...) tensors — as returned by ReplayBuffer.sample().

    Returns dict with:
        mse_symlog:  scalar — MSE between predicted and target in symlog space.
        mse_real:    scalar — same, in original observation units.
        per_dim_mse: (obs_dim,) tensor — per-feature reconstruction MSE (symlog).
    """
    from rssmlite.utils import symexp, symlog

    rssm = agent.rssm
    obs, action = batch["obs"], batch["action"]

    with torch.no_grad():
        rollout = rssm.observe(obs, action)
        feature = torch.cat([rollout["deter"], rollout["stoch"]], dim=-1)
        obs_pred_symlog = rssm.decoder(feature)
        obs_target_symlog = symlog(obs)

        per_dim_mse = ((obs_pred_symlog - obs_target_symlog) ** 2).mean(dim=(0, 1))
        mse_symlog = per_dim_mse.mean()
        mse_real = torch.nn.functional.mse_loss(symexp(obs_pred_symlog), obs)

    return {
        "mse_symlog": mse_symlog.item(),
        "mse_real": mse_real.item(),
        "per_dim_mse": per_dim_mse,
    }


def plot_reconstruction(agent: "RSSMAgent", batch: dict, feature_idx: int = 0):
    """Plot real vs. model-predicted observations for one feature dimension
    over a single trajectory. Useful for quickly eyeballing whether the
    world model is tracking the real dynamics or has drifted.

    Returns a matplotlib Figure. Raises ImportError if matplotlib is absent
    (pip install rssmlite[viz]).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("plot_reconstruction needs matplotlib: pip install rssmlite[viz]") from e

    from rssmlite.utils import symexp, symlog

    rssm = agent.rssm
    obs = batch["obs"][:1]   # first trajectory only
    action = batch["action"][:1]

    with torch.no_grad():
        rollout = rssm.observe(obs, action)
        feature = torch.cat([rollout["deter"], rollout["stoch"]], dim=-1)
        obs_pred = symexp(rssm.decoder(feature))[0, :, feature_idx].numpy()
        obs_real = obs[0, :, feature_idx].numpy()

    t = np.arange(len(obs_real))
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(t, obs_real, label="real", color="steelblue")
    ax.plot(t, obs_pred, label="predicted", color="tomato", linestyle="--")
    ax.set_xlabel("timestep")
    ax.set_ylabel(f"obs dim {feature_idx}")
    ax.set_title("World model reconstruction: real vs predicted")
    ax.legend()
    fig.tight_layout()
    return fig


def compare_sample_efficiency(
    env_id: str,
    agent_steps: list[int],
    agent_returns: list[float],
    baseline_steps: list[int],
    baseline_returns: list[float],
    baseline_label: str = "PPO baseline",
):
    """Plot a sample-efficiency comparison curve: rssmlite vs. a
    model-free baseline (typically PPO via Stable-Baselines3).

    The caller is responsible for running both and passing in the
    (steps, returns) lists — this function only does the plotting, so it
    works whether the baseline was run via SB3, a custom script, or
    loaded from a CSV of pre-computed results.

    Returns a matplotlib Figure.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("compare_sample_efficiency needs matplotlib: pip install rssmlite[viz]") from e

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(agent_steps, agent_returns, label="rssmlite (RSSM)", color="steelblue", linewidth=2)
    ax.plot(baseline_steps, baseline_returns, label=baseline_label, color="tomato",
            linewidth=2, linestyle="--")
    ax.set_xlabel("environment steps")
    ax.set_ylabel("episode return")
    ax.set_title(f"Sample efficiency: {env_id}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def run_evaluation_episodes(agent: "RSSMAgent", env, n_episodes: int = 10) -> dict:
    """Run `n_episodes` real evaluation episodes (no exploration noise,
    greedy policy) and return summary stats. Used to generate the
    'agent_returns' data for compare_sample_efficiency.

    Returns dict: mean_return, std_return, min_return, max_return,
    episode_lengths (list).
    """
    from rssmlite.env_utils import collect_episode
    from rssmlite.replay_buffer import ReplayBuffer

    buf = ReplayBuffer(capacity_episodes=n_episodes)
    returns, lengths = [], []
    for _ in range(n_episodes):
        length, total_reward = collect_episode(
            env, buf, policy=agent._make_acting_policy(), max_steps=10_000
        )
        returns.append(total_reward)
        lengths.append(length)

    returns = np.array(returns)
    return {
        "mean_return": float(returns.mean()),
        "std_return": float(returns.std()),
        "min_return": float(returns.min()),
        "max_return": float(returns.max()),
        "episode_lengths": lengths,
    }

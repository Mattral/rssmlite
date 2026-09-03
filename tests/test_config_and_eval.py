"""Tests for config validation (P1.4) and evaluation utilities (P1.5)."""

import pytest
import torch

gym = pytest.importorskip("gymnasium")

from rssmlite import RSSMAgent, reconstruction_report, run_evaluation_episodes
from rssmlite.agent import _validate_config
from rssmlite.evaluation import plot_reconstruction


def make_tiny_agent():
    env = gym.make("CartPole-v1")
    agent = RSSMAgent.from_env(
        env,
        rssm_kwargs=dict(deter_dim=16, embed_dim=16, num_categoricals=4, num_classes=4, hidden_dim=32),
        hidden_dim=32,
    )
    env.close()
    return agent


# ── Config validation ────────────────────────────────────────────────────────

def test_valid_config_passes():
    _validate_config({"env": {"id": "CartPole-v1"}, "agent": {"hidden_dim": 64}, "train": {"steps": 1000}})


def test_unknown_section_raises():
    with pytest.raises(ValueError, match="Unknown config section"):
        _validate_config({"typo_section": {}})


def test_unknown_agent_key_raises():
    with pytest.raises(ValueError, match="Unknown agent config key"):
        _validate_config({"agent": {"lr": 3e-4}})  # should be world_model_lr


def test_unknown_train_key_raises():
    with pytest.raises(ValueError, match="Unknown train config key"):
        _validate_config({"train": {"num_epochs": 10}})  # not a valid key


def test_from_config_all_four_envs():
    """from_config must parse all four shipped configs without error.
    LunarLander requires box2d (gymnasium[box2d]) — skipped if absent."""
    import yaml

    configs = ["configs/cartpole.yaml", "configs/acrobot.yaml",
               "configs/pendulum.yaml", "configs/lunarlander.yaml"]
    for path in configs:
        with open(f"/home/claude/rssmlite-repo/{path}") as f:
            cfg = yaml.safe_load(f)
        try:
            env = gym.make(cfg["env"]["id"])
        except Exception as e:
            if "Box2D" in str(e) or "box2d" in str(e).lower():
                pytest.skip(f"Skipping {path}: box2d not installed")
            raise
        agent = RSSMAgent.from_config(f"/home/claude/rssmlite-repo/{path}", env=env)
        assert agent.rssm is not None
        env.close()


# ── Evaluation utilities ─────────────────────────────────────────────────────

def test_reconstruction_report_keys_and_finiteness():
    agent = make_tiny_agent()
    # put a little data in the buffer
    env = gym.make("CartPole-v1")
    from rssmlite.env_utils import collect_episode
    for _ in range(3):
        collect_episode(env, agent.buffer, max_steps=30)
    env.close()

    batch = agent.buffer.sample(batch_size=4, seq_len=8)
    report = reconstruction_report(agent, batch)
    assert set(report.keys()) == {"mse_symlog", "mse_real", "per_dim_mse"}
    assert torch.isfinite(torch.tensor(report["mse_symlog"]))
    assert torch.isfinite(torch.tensor(report["mse_real"]))
    assert report["per_dim_mse"].shape == (4,)  # CartPole obs_dim=4


def test_plot_reconstruction_returns_figure():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    agent = make_tiny_agent()
    env = gym.make("CartPole-v1")
    from rssmlite.env_utils import collect_episode
    for _ in range(2):
        collect_episode(env, agent.buffer, max_steps=30)
    env.close()

    batch = agent.buffer.sample(batch_size=2, seq_len=8)
    fig = plot_reconstruction(agent, batch, feature_idx=0)
    assert fig is not None


def test_run_evaluation_episodes_keys():
    agent = make_tiny_agent()
    env = gym.make("CartPole-v1")
    result = run_evaluation_episodes(agent, env, n_episodes=3)
    env.close()
    assert set(result.keys()) == {"mean_return", "std_return", "min_return", "max_return", "episode_lengths"}
    assert len(result["episode_lengths"]) == 3

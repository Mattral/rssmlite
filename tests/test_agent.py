"""Smoke tests for RSSMAgent (roadmap P1.3): a tiny, fast end-to-end run
covering world-model + actor-critic training together, checkpoint
save/resume, and imagine_rollout(). Not a convergence test — dims and
`steps` are kept tiny so this runs in seconds on CPU; the actual "does
CartPole solve in under 30 min on a T4" check has to happen on real
hardware (see ROADMAP.md P1.3), not in this test suite.
"""

import numpy as np
import pytest
import torch

gym = pytest.importorskip("gymnasium")

from rssmlite import RSSMAgent


def make_tiny_agent(discrete: bool, obs_dim: int = 4, action_dim: int = 2) -> RSSMAgent:
    return RSSMAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        discrete=discrete,
        rssm_kwargs=dict(deter_dim=16, embed_dim=16, num_categoricals=4, num_classes=4, hidden_dim=32),
        hidden_dim=32,
    )


def test_train_runs_without_error_discrete():
    env = gym.make("CartPole-v1")
    agent = RSSMAgent.from_env(
        env,
        rssm_kwargs=dict(deter_dim=16, embed_dim=16, num_categoricals=4, num_classes=4, hidden_dim=32),
        hidden_dim=32,
    )
    agent.train(
        env,
        steps=60,
        seed_episodes=2,
        batch_size=4,
        seq_len=8,
        horizon=4,
        max_episode_steps=30,
        log_every=0,
    )
    env.close()
    assert agent._env_steps >= 60


def test_train_runs_without_error_continuous():
    """Pendulum-v1 exercises the continuous-action Actor branch (Normal +
    rsample), which CartPole's discrete branch never touches."""
    env = gym.make("Pendulum-v1")
    agent = RSSMAgent.from_env(
        env,
        rssm_kwargs=dict(deter_dim=16, embed_dim=16, num_categoricals=4, num_classes=4, hidden_dim=32),
        hidden_dim=32,
    )
    agent.train(
        env,
        steps=60,
        seed_episodes=2,
        batch_size=4,
        seq_len=8,
        horizon=4,
        max_episode_steps=30,
        log_every=0,
    )
    env.close()
    assert agent._env_steps >= 60


def test_checkpoint_save_and_resume(tmp_path):
    env = gym.make("CartPole-v1")
    agent = RSSMAgent.from_env(
        env,
        rssm_kwargs=dict(deter_dim=16, embed_dim=16, num_categoricals=4, num_classes=4, hidden_dim=32),
        hidden_dim=32,
    )
    agent.train(env, steps=40, seed_episodes=2, batch_size=4, seq_len=8, horizon=4, max_episode_steps=30, log_every=0)

    ckpt_path = tmp_path / "checkpoint.pt"
    agent.save_checkpoint(ckpt_path)
    assert ckpt_path.exists()

    resumed = RSSMAgent.load_checkpoint(ckpt_path)
    assert resumed._env_steps == agent._env_steps

    # Weights should match exactly after resume, not just shapes.
    for (name, original_param), (_, resumed_param) in zip(
        agent.rssm.named_parameters(), resumed.rssm.named_parameters()
    ):
        assert torch.equal(original_param, resumed_param), f"mismatch in {name}"

    # Resumed agent should be trainable, not just loadable.
    resumed.train(env, steps=resumed._env_steps + 20, seed_episodes=0, batch_size=4, seq_len=8, horizon=4, max_episode_steps=30, log_every=0)
    env.close()


def test_imagine_rollout_shapes_and_finiteness():
    agent = make_tiny_agent(discrete=True)
    rollout = agent.imagine_rollout(steps=10)
    assert rollout["deter"].shape == (1, 10, 16)
    assert rollout["obs_pred"].shape == (1, 10, 4)
    assert rollout["reward_pred"].shape == (1, 10)
    assert torch.isfinite(rollout["obs_pred"]).all()
    assert torch.isfinite(rollout["reward_pred"]).all()


def test_acting_policy_returns_env_native_actions():
    """The stateful policy from _make_acting_policy must return actions
    collect_episode/env.step can consume directly: an int for Discrete,
    an ndarray for Box."""
    discrete_agent = make_tiny_agent(discrete=True, obs_dim=4, action_dim=3)
    policy = discrete_agent._make_acting_policy()
    action = policy(np.zeros(4, dtype=np.float32))
    assert isinstance(action, int)
    assert 0 <= action < 3

    continuous_agent = make_tiny_agent(discrete=False, obs_dim=4, action_dim=2)
    policy = continuous_agent._make_acting_policy()
    action = policy(np.zeros(4, dtype=np.float32))
    assert isinstance(action, np.ndarray)
    assert action.shape == (2,)


def test_visualize_latent_space_returns_a_figure():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")  # headless backend, no display needed for CI

    agent = make_tiny_agent(discrete=True)
    fig = agent.visualize_latent_space()
    assert fig is not None
    assert len(fig.axes) == 2  # main scatter axes + colorbar axes

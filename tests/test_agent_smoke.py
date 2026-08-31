"""End-to-end smoke test (roadmap P1.2): collect ~100 steps of real CartPole
experience, sample a training batch, and run it through RSSM.loss() with no
errors. This is the first test that touches a real Gymnasium env — skipped
automatically if gymnasium isn't installed (it's an optional extra: `pip
install rssmlite[envs]`), so the core test suite doesn't hard-require it.
"""

import pytest
import torch

gym = pytest.importorskip("gymnasium")

from rssmlite import RSSM, ReplayBuffer
from rssmlite.env_utils import action_dim_of, collect_episode


def test_cartpole_100_steps_end_to_end():
    env = gym.make("CartPole-v1")
    buffer = ReplayBuffer(capacity_episodes=50)

    obs_dim = env.observation_space.shape[0]
    action_dim = action_dim_of(env.action_space)

    # Collect at least 100 total steps of real experience, exactly as a
    # real training loop would before starting to fit the world model.
    total_steps = 0
    while total_steps < 100:
        length, _reward = collect_episode(env, buffer, max_steps=500)
        total_steps += length
    env.close()

    assert total_steps >= 100
    assert len(buffer) > 0

    seq_len = 10
    assert buffer.can_sample(seq_len), (
        "Need at least one episode >= seq_len; CartPole episodes are "
        "usually much longer than this even with a random policy."
    )
    batch = buffer.sample(batch_size=8, seq_len=seq_len)

    rssm = RSSM(
        obs_dim=obs_dim,
        action_dim=action_dim,
        deter_dim=32,
        embed_dim=32,
        num_categoricals=8,
        num_classes=8,
        hidden_dim=64,
    )
    losses = rssm.loss(batch["obs"], batch["action"], batch["reward"], 1.0 - batch["done"])

    assert torch.isfinite(losses["total"])
    losses["total"].backward()  # confirms the full pipeline is differentiable

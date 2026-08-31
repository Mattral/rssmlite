"""Tests for ReplayBuffer (roadmap P1.2). No Gymnasium dependency here —
episodes are synthetic, so this runs even in environments without
gymnasium installed."""

import numpy as np
import pytest
import torch

from rssmlite import ReplayBuffer


def make_episode(length: int, obs_dim: int = 4, action_dim: int = 2):
    obs = np.random.randn(length, obs_dim).astype(np.float32)
    action = np.random.randn(length, action_dim).astype(np.float32)
    reward = np.random.randn(length).astype(np.float32)
    done = np.zeros(length, dtype=np.float32)
    done[-1] = 1.0
    return obs, action, reward, done


def test_add_and_len():
    buf = ReplayBuffer(capacity_episodes=10)
    assert len(buf) == 0
    buf.add_episode(*make_episode(20))
    assert len(buf) == 1


def test_capacity_drops_oldest():
    buf = ReplayBuffer(capacity_episodes=3)
    for _ in range(5):
        buf.add_episode(*make_episode(10))
    assert len(buf) == 3  # oldest 2 dropped, not an error


def test_add_episode_rejects_mismatched_lengths():
    obs, action, reward, done = make_episode(20)
    buf = ReplayBuffer()
    with pytest.raises(AssertionError):
        buf.add_episode(obs, action[:-1], reward, done)


def test_can_sample_respects_seq_len():
    buf = ReplayBuffer()
    buf.add_episode(*make_episode(10))
    assert buf.can_sample(10) is True
    assert buf.can_sample(11) is False


def test_sample_shapes_and_dtype():
    buf = ReplayBuffer()
    for _ in range(4):
        buf.add_episode(*make_episode(length=30, obs_dim=4, action_dim=2))

    batch = buf.sample(batch_size=8, seq_len=12)
    assert batch["obs"].shape == (8, 12, 4)
    assert batch["action"].shape == (8, 12, 2)
    assert batch["reward"].shape == (8, 12)
    assert batch["done"].shape == (8, 12)
    for tensor in batch.values():
        assert tensor.dtype == torch.float32


def test_sample_never_crosses_episode_boundary():
    """Every sampled sequence's obs values should trace back to a single
    stored episode's contiguous slice — checked here via exact value
    membership, since obs are random floats (no accidental collisions)."""
    buf = ReplayBuffer()
    episodes = [make_episode(length=15) for _ in range(3)]
    for ep in episodes:
        buf.add_episode(*ep)

    batch = buf.sample(batch_size=20, seq_len=5)
    for i in range(20):
        seq_obs = batch["obs"][i].numpy()
        # The sampled slice must appear as a contiguous block in exactly
        # one of the source episodes.
        found_in_any = any(
            any(
                np.allclose(ep_obs[start : start + 5], seq_obs)
                for start in range(len(ep_obs) - 5 + 1)
            )
            for ep_obs, *_ in episodes
        )
        assert found_in_any


def test_sample_raises_when_no_episode_long_enough():
    buf = ReplayBuffer()
    buf.add_episode(*make_episode(5))
    with pytest.raises(ValueError):
        buf.sample(batch_size=4, seq_len=10)

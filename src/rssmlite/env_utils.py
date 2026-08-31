"""
Environment interaction: run one episode against a real Gymnasium env and
hand the transitions to a ReplayBuffer. Kept separate from `replay_buffer.py`
so that file stays pure storage/sampling (Section 6 design principle #4:
no hidden state, no framework magic — this is the one place that actually
talks to `env.step()`).
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from rssmlite.replay_buffer import ReplayBuffer


def encode_action(action, action_space) -> np.ndarray:
    """Discrete actions -> one-hot vector; continuous actions -> passed
    through as-is. RSSM's dynamics backbone expects a fixed-size vector
    either way (its `action_dim` at construction time), regardless of
    whether the underlying env is discrete or continuous.
    """
    import gymnasium as gym

    if isinstance(action_space, gym.spaces.Discrete):
        one_hot = np.zeros(action_space.n, dtype=np.float32)
        one_hot[action] = 1.0
        return one_hot
    return np.asarray(action, dtype=np.float32)


def action_dim_of(action_space) -> int:
    """The vector size `encode_action` will produce for this action space —
    what to pass as `RSSM(action_dim=...)`."""
    import gymnasium as gym

    if isinstance(action_space, gym.spaces.Discrete):
        return action_space.n
    return int(np.prod(action_space.shape))


def collect_episode(
    env,
    buffer: ReplayBuffer,
    policy: Callable[[np.ndarray], object] | None = None,
    max_steps: int = 500,
) -> tuple[int, float]:
    """Run one episode, store it in `buffer`. `policy(obs) -> action` in the
    env's native action format (an int for Discrete, an array for Box); if
    omitted, samples uniformly from `env.action_space` (useful for the P1.2
    smoke test and for seeding the replay buffer before the agent exists).

    Returns (episode_length, total_reward).
    """
    if policy is None:
        policy = lambda _obs: env.action_space.sample()

    obs, _info = env.reset()
    obs_list, action_list, reward_list, done_list = [], [], [], []
    total_reward = 0.0

    for _ in range(max_steps):
        action = policy(obs)
        next_obs, reward, terminated, truncated, _info = env.step(action)
        done = terminated or truncated

        obs_list.append(obs)
        action_list.append(encode_action(action, env.action_space))
        reward_list.append(reward)
        done_list.append(done)

        total_reward += reward
        obs = next_obs
        if done:
            break

    buffer.add_episode(
        obs=np.stack(obs_list),
        action=np.stack(action_list),
        reward=np.array(reward_list, dtype=np.float32),
        done=np.array(done_list, dtype=np.float32),
    )
    return len(obs_list), total_reward

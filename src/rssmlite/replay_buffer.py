"""
ReplayBuffer: stores complete episodes of (obs, action, reward, done) and
samples fixed-length sequences for training the world model.

Sequences never cross an episode boundary — sampling a fixed length out of
the middle of a real episode gives a physically coherent transition history,
which is what `RSSM.observe()`/`RSSM.loss()` expect. Padding across episode
boundaries would hand the model a fake transition (e.g. the reset state
appearing mid-rollout) and quietly corrupt training.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import torch


class ReplayBuffer:
    """Args:
        capacity_episodes: oldest episodes are dropped once this many are
            stored. Sized in episodes, not transitions, since sampling is
            episode-aware.
    """

    def __init__(self, capacity_episodes: int = 500):
        self.episodes: deque[dict] = deque(maxlen=capacity_episodes)

    def add_episode(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
    ) -> None:
        """Each array is (episode_len, ...) — one full episode, in order.
        `done[t]` should be True only on the last transition, matching
        Gymnasium's terminated/truncated-collapsed-to-bool convention.
        """
        assert len(obs) == len(action) == len(reward) == len(done) > 0, (
            "obs/action/reward/done must be equal-length and non-empty"
        )
        self.episodes.append(
            {
                "obs": np.asarray(obs, dtype=np.float32),
                "action": np.asarray(action, dtype=np.float32),
                "reward": np.asarray(reward, dtype=np.float32),
                "done": np.asarray(done, dtype=np.float32),
            }
        )

    def __len__(self) -> int:
        return len(self.episodes)

    def can_sample(self, seq_len: int) -> bool:
        """Whether at least one stored episode is long enough to sample from."""
        return any(len(ep["obs"]) >= seq_len for ep in self.episodes)

    def sample(self, batch_size: int, seq_len: int) -> dict[str, torch.Tensor]:
        """Returns dict of (batch_size, seq_len, ...) tensors: obs, action,
        reward, done. Raises if no stored episode is long enough — check
        `can_sample()` first if that's possible in your training loop."""
        eligible = [ep for ep in self.episodes if len(ep["obs"]) >= seq_len]
        if not eligible:
            raise ValueError(
                f"No episode has length >= seq_len={seq_len}. "
                f"Longest stored episode: {max((len(e['obs']) for e in self.episodes), default=0)}."
            )

        batch = {"obs": [], "action": [], "reward": [], "done": []}
        for _ in range(batch_size):
            ep = eligible[np.random.randint(len(eligible))]
            start = np.random.randint(0, len(ep["obs"]) - seq_len + 1)
            end = start + seq_len
            for key in batch:
                batch[key].append(ep[key][start:end])

        return {key: torch.from_numpy(np.stack(values)) for key, values in batch.items()}

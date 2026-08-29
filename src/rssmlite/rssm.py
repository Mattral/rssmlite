"""
RSSM: the world model itself.

State at each timestep is a pair (deter, stoch):
  - deter: deterministic recurrent state h_t (from the dynamics backbone).
  - stoch: categorical stochastic latent z_t (32x32 one-hot by default).
`feature_t = concat(deter_t, flatten(stoch_t))` is what every head
(decoder, reward, continue) actually reads.

Two ways to run the model, both used during training (see roadmap P1.3):
  - `observe()`: teacher-forced on real observations. Used to fit the
    world model itself.
  - `imagine()`: rolls forward using only the *prior* (no observations,
    just actions from a policy). Used to train the actor-critic purely
    in imagination, with no environment calls.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from rssmlite.backbones.gru import GRUDynamics
from rssmlite.networks import ContinueHead, Decoder, Encoder, PosteriorNet, RewardHead
from rssmlite.utils import kl_balancing, straight_through_sample, symlog


class RSSM(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        deter_dim: int = 200,
        embed_dim: int = 200,
        num_categoricals: int = 32,
        num_classes: int = 32,
        hidden_dim: int = 200,
        dynamics: nn.Module | None = None,
        kl_alpha: float = 0.8,
        kl_free_bits: float = 1.0,
    ):
        super().__init__()
        self.deter_dim = deter_dim
        self.num_categoricals = num_categoricals
        self.num_classes = num_classes
        self.stoch_dim = num_categoricals * num_classes
        self.feature_dim = deter_dim + self.stoch_dim
        self.kl_alpha = kl_alpha
        self.kl_free_bits = kl_free_bits

        self.encoder = Encoder(obs_dim, embed_dim, hidden_dim)
        self.posterior_net = PosteriorNet(
            deter_dim, embed_dim, num_categoricals, num_classes, hidden_dim
        )
        # Backbone is injectable (Section 6, design principle #3): defaults
        # to GRU, but any module with the same forward()/prior_logits()
        # signature works (e.g. TransformerDynamics in P2).
        self.dynamics = dynamics or GRUDynamics(
            deter_dim, self.stoch_dim, action_dim, num_categoricals, num_classes, hidden_dim
        )
        self.decoder = Decoder(self.feature_dim, obs_dim, hidden_dim)
        self.reward_head = RewardHead(self.feature_dim, hidden_dim)
        self.continue_head = ContinueHead(self.feature_dim, hidden_dim)

    def initial_state(self, batch_size: int, device: torch.device) -> dict:
        return {
            "deter": torch.zeros(batch_size, self.deter_dim, device=device),
            "stoch": torch.zeros(batch_size, self.stoch_dim, device=device),
        }

    @staticmethod
    def feature(state: dict) -> torch.Tensor:
        return torch.cat([state["deter"], state["stoch"]], dim=-1)

    def observe(self, obs_seq: torch.Tensor, action_seq: torch.Tensor) -> dict:
        """Teacher-forced pass over a real (obs, action) sequence.

        Args:
            obs_seq: (B, T, obs_dim)
            action_seq: (B, T, action_dim) — action_seq[:, t] is the action
                taken *after* observing obs_seq[:, t].

        Returns dict of stacked (B, T, ...) tensors: deter, stoch,
        post_logits, prior_logits — everything needed by `loss()`.
        """
        batch_size, seq_len, _ = obs_seq.shape
        device = obs_seq.device
        state = self.initial_state(batch_size, device)
        embeds = self.encoder(obs_seq)

        deters, stochs, post_logits_list, prior_logits_list = [], [], [], []
        for t in range(seq_len):
            if t > 0:
                state["deter"] = self.dynamics(state["deter"], state["stoch"], action_seq[:, t - 1])
            prior_logits = self.dynamics.prior_logits(state["deter"])
            post_logits = self.posterior_net(state["deter"], embeds[:, t])
            state["stoch"] = straight_through_sample(post_logits).flatten(-2, -1)

            deters.append(state["deter"])
            stochs.append(state["stoch"])
            post_logits_list.append(post_logits)
            prior_logits_list.append(prior_logits)

        return {
            "deter": torch.stack(deters, dim=1),
            "stoch": torch.stack(stochs, dim=1),
            "post_logits": torch.stack(post_logits_list, dim=1),
            "prior_logits": torch.stack(prior_logits_list, dim=1),
        }

    def imagine(self, initial_state: dict, policy, horizon: int) -> dict:
        """Roll forward `horizon` steps using only the prior — no
        observations. `policy(feature) -> action` is called at each step
        (RSSMAgent's actor, once it exists — P1.3).
        """
        state = dict(initial_state)
        deters, stochs, actions = [], [], []
        for _ in range(horizon):
            action = policy(self.feature(state))
            state["deter"] = self.dynamics(state["deter"], state["stoch"], action)
            prior_logits = self.dynamics.prior_logits(state["deter"])
            state["stoch"] = straight_through_sample(prior_logits).flatten(-2, -1)

            deters.append(state["deter"])
            stochs.append(state["stoch"])
            actions.append(action)

        return {
            "deter": torch.stack(deters, dim=1),
            "stoch": torch.stack(stochs, dim=1),
            "action": torch.stack(actions, dim=1),
        }

    def loss(
        self,
        obs_seq: torch.Tensor,
        action_seq: torch.Tensor,
        reward_seq: torch.Tensor,
        continue_seq: torch.Tensor,
    ) -> dict:
        """World-model loss on a real sequence: reconstruction + reward +
        continue + balanced, free-bits-clipped KL. All in symlog space for
        recon/reward, matching Decoder/RewardHead's convention."""
        rollout = self.observe(obs_seq, action_seq)
        feature = torch.cat([rollout["deter"], rollout["stoch"]], dim=-1)

        obs_pred = self.decoder(feature)
        recon_loss = F.mse_loss(obs_pred, symlog(obs_seq))

        reward_pred = self.reward_head(feature)
        reward_loss = F.mse_loss(reward_pred, symlog(reward_seq))

        continue_logit = self.continue_head(feature)
        continue_loss = F.binary_cross_entropy_with_logits(continue_logit, continue_seq)

        kl_loss = kl_balancing(
            rollout["post_logits"], rollout["prior_logits"], self.kl_alpha, self.kl_free_bits
        )

        total = recon_loss + reward_loss + continue_loss + kl_loss
        return {
            "total": total,
            "recon": recon_loss,
            "reward": reward_loss,
            "continue": continue_loss,
            "kl": kl_loss,
        }

"""
GRU dynamics backbone: the default recurrent core for the RSSM.

`RSSM` treats this as swappable — it only relies on `forward()` and
`prior_logits()`. `TransformerDynamics` (P2, ablation study) implements the
same two methods so `RSSM` doesn't change when the backbone does.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GRUDynamics(nn.Module):
    """Advances the deterministic recurrent state h_t -> h_{t+1}, and
    predicts the prior over the next stochastic latent from h_{t+1} alone
    (i.e. without seeing the real observation — this is what lets the model
    imagine rollouts without an environment).

    Args:
        deter_dim: size of the deterministic hidden state h.
        stoch_dim: flattened size of the stochastic latent (num_categoricals
            * num_classes).
        action_dim: size of the (one-hot or continuous) action vector.
        num_categoricals: number of categorical groups in the latent.
        num_classes: classes per categorical group.
        hidden_dim: width of the MLP that produces GRU input and prior logits.
    """

    def __init__(
        self,
        deter_dim: int,
        stoch_dim: int,
        action_dim: int,
        num_categoricals: int,
        num_classes: int,
        hidden_dim: int = 200,
    ):
        super().__init__()
        self.num_categoricals = num_categoricals
        self.num_classes = num_classes

        # Maps (prev stochastic latent, action) -> a vector the GRU can
        # consume as its input at this step.
        self.pre_gru = nn.Sequential(
            nn.Linear(stoch_dim + action_dim, hidden_dim),
            nn.SiLU(),
        )
        self.gru_cell = nn.GRUCell(hidden_dim, deter_dim)

        # Prior net: h_{t+1} alone -> logits over the next stochastic latent.
        # This is the network that gets trained to imagine without a real
        # observation.
        self.prior_net = nn.Sequential(
            nn.Linear(deter_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_categoricals * num_classes),
        )

    def forward(
        self, deter: torch.Tensor, stoch: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """One recurrent step. Shapes: deter (B, deter_dim), stoch
        (B, stoch_dim), action (B, action_dim) -> new deter (B, deter_dim)."""
        gru_input = self.pre_gru(torch.cat([stoch, action], dim=-1))
        return self.gru_cell(gru_input, deter)

    def prior_logits(self, deter: torch.Tensor) -> torch.Tensor:
        """(B, deter_dim) -> (B, num_categoricals, num_classes) prior logits."""
        logits = self.prior_net(deter)
        return logits.view(*logits.shape[:-1], self.num_categoricals, self.num_classes)

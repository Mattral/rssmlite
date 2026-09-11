"""
RSSMAgent: wraps RSSM with an actor-critic trained on imagined rollouts.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from rssmlite.actor_critic import Actor, Critic, lambda_return
from rssmlite.env_utils import action_dim_of, collect_episode
from rssmlite.replay_buffer import ReplayBuffer
from rssmlite.rssm import RSSM
from rssmlite.utils import straight_through_sample, symexp


_KNOWN_CONFIG_SECTIONS = {"env", "agent", "train"}
_KNOWN_AGENT_KEYS = {
    "obs_dim", "action_dim", "discrete", "rssm_kwargs", "hidden_dim",
    "world_model_lr", "actor_lr", "critic_lr", "gamma", "lam",
    "entropy_coef", "replay_capacity_episodes", "critic_ema_tau",
}
_KNOWN_TRAIN_KEYS = {
    "steps", "seed_episodes", "batch_size", "seq_len", "horizon",
    "max_episode_steps", "checkpoint_every", "log_every", "imagine_ratio",
}


def _validate_config(config: dict) -> None:
    unknown_sections = set(config) - _KNOWN_CONFIG_SECTIONS
    if unknown_sections:
        raise ValueError(f"Unknown config section(s): {unknown_sections}")
    unknown_agent = set(config.get("agent", {})) - _KNOWN_AGENT_KEYS
    if unknown_agent:
        raise ValueError(f"Unknown agent config key(s): {unknown_agent}")
    unknown_train = set(config.get("train", {})) - _KNOWN_TRAIN_KEYS
    if unknown_train:
        raise ValueError(f"Unknown train config key(s): {unknown_train}")


class RSSMAgent:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        discrete: bool,
        rssm_kwargs: dict | None = None,
        hidden_dim: int = 200,
        world_model_lr: float = 3e-4,
        actor_lr: float = 8e-5,
        critic_lr: float = 8e-5,
        gamma: float = 0.99,
        lam: float = 0.95,
        entropy_coef: float = 3e-4,
        replay_capacity_episodes: int = 500,
        critic_ema_tau: float = 0.02,
    ):
        self.discrete = discrete
        self.action_dim = action_dim
        self.gamma = gamma
        self.lam = lam
        self.entropy_coef = entropy_coef

        self.rssm = RSSM(obs_dim=obs_dim, action_dim=action_dim, **(rssm_kwargs or {}))
        self.actor = Actor(self.rssm.feature_dim, action_dim, discrete, hidden_dim)
        self.critic = Critic(self.rssm.feature_dim, hidden_dim)
        # EMA target critic — slowly-tracking copy used only for bootstrap
        # targets in lambda_return. Prevents the critic loss from diverging
        # when target and prediction co-move (the issue seen in T4 run 2).
        self.critic_ema = self.critic.make_ema(tau=critic_ema_tau)

        self.world_model_optimizer = torch.optim.Adam(self.rssm.parameters(), lr=world_model_lr)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.buffer = ReplayBuffer(capacity_episodes=replay_capacity_episodes)
        self._env_steps = 0

        self._init_kwargs = dict(
            obs_dim=obs_dim, action_dim=action_dim, discrete=discrete,
            rssm_kwargs=rssm_kwargs, hidden_dim=hidden_dim,
            world_model_lr=world_model_lr, actor_lr=actor_lr, critic_lr=critic_lr,
            gamma=gamma, lam=lam, entropy_coef=entropy_coef,
            replay_capacity_episodes=replay_capacity_episodes,
            critic_ema_tau=critic_ema_tau,
        )

    # ------------------------------------------------------------------
    # Device handling
    # ------------------------------------------------------------------

    @property
    def device(self) -> torch.device:
        return next(self.rssm.parameters()).device

    def to(self, device) -> "RSSMAgent":
        """Move all networks to device in one call."""
        self.rssm.to(device)
        self.actor.to(device)
        self.critic.to(device)
        self.critic_ema.to(device)
        return self

    def _batch_to_device(self, batch: dict) -> dict:
        return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()}

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls, env, **kwargs) -> "RSSMAgent":
        import gymnasium as gym
        obs_dim = env.observation_space.shape[0]
        discrete = isinstance(env.action_space, gym.spaces.Discrete)
        return cls(obs_dim=obs_dim, action_dim=action_dim_of(env.action_space),
                   discrete=discrete, **kwargs)

    @classmethod
    def from_config(cls, path: str, env=None) -> "RSSMAgent":
        with open(path) as f:
            config = yaml.safe_load(f)
        _validate_config(config)
        agent_kwargs = dict(config.get("agent", {}))
        if env is not None and not {"obs_dim", "action_dim", "discrete"} <= agent_kwargs.keys():
            import gymnasium as gym
            agent_kwargs.setdefault("obs_dim", env.observation_space.shape[0])
            agent_kwargs.setdefault("action_dim", action_dim_of(env.action_space))
            agent_kwargs.setdefault("discrete", isinstance(env.action_space, gym.spaces.Discrete))
        return cls(**agent_kwargs)

    # ------------------------------------------------------------------
    # Acting
    # ------------------------------------------------------------------

    def _make_acting_policy(self):
        device = self.device
        state = self.rssm.initial_state(1, device)
        prev_action = torch.zeros(1, self.action_dim, device=device)
        step = {"t": 0}

        def policy(obs):
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                if step["t"] > 0:
                    state["deter"] = self.rssm.dynamics(
                        state["deter"], state["stoch"], prev_action
                    )
                embed = self.rssm.encoder(obs_t)
                post_logits = self.rssm.posterior_net(state["deter"], embed)
                state["stoch"] = straight_through_sample(post_logits).flatten(-2, -1)
                feature = self.rssm.feature(state)
                action, _ = self.actor(feature)
            prev_action.copy_(action)
            step["t"] += 1
            if self.discrete:
                return action.squeeze(0).argmax().item()
            return action.squeeze(0).cpu().numpy()

        return policy

    # ------------------------------------------------------------------
    # Training steps
    # ------------------------------------------------------------------

    def _world_model_step(self, batch: dict) -> dict:
        batch = self._batch_to_device(batch)
        losses = self.rssm.loss(
            batch["obs"], batch["action"], batch["reward"], 1.0 - batch["done"]
        )
        self.world_model_optimizer.zero_grad()
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(self.rssm.parameters(), 100.0)
        self.world_model_optimizer.step()
        return {f"wm/{k}": v.item() for k, v in losses.items()}

    def _actor_critic_step(self, batch: dict, horizon: int) -> dict:
        batch = self._batch_to_device(batch)

        with torch.no_grad():
            rollout = self.rssm.observe(batch["obs"], batch["action"])

        B, T = rollout["deter"].shape[:2]
        start_state = {
            "deter": rollout["deter"].reshape(B * T, -1).detach(),
            "stoch": rollout["stoch"].reshape(B * T, -1).detach(),
        }

        def policy(feature):
            action, _ = self.actor(feature)
            return action

        imagined = self.rssm.imagine(start_state, policy, horizon)
        imagined_feature = self.rssm.feature(
            {"deter": imagined["deter"], "stoch": imagined["stoch"]}
        )
        start_feature = self.rssm.feature(start_state).unsqueeze(1)
        all_features = torch.cat([start_feature, imagined_feature], dim=1)  # (N, H+1, feat)

        with torch.no_grad():
            rewards = symexp(self.rssm.reward_head(imagined_feature))
            continues = torch.sigmoid(self.rssm.continue_head(imagined_feature))
            # KEY FIX: use the slow-moving EMA target critic for bootstrap
            # values rather than the online critic. This breaks the co-moving
            # target/prediction loop that caused critic_loss to diverge to 40+.
            ema_values = self.critic_ema(all_features)

        returns = lambda_return(
            rewards, continues, ema_values, self.gamma, self.lam
        )

        # Return normalisation — keeps actor loss on a stable scale.
        returns_std = returns.std().clamp(min=1.0)
        returns_norm = returns / returns_std

        # Actor loss: maximise normalised returns + entropy bonus.
        # Features must NOT be detached here so gradients reach actor params.
        _, entropy = self.actor(all_features[:, :-1])
        actor_loss = -returns_norm.mean() - self.entropy_coef * entropy.mean()

        # Critic loss: online critic predicts real-scale returns.
        # Features detached — critic update must not affect RSSM or actor.
        critic_pred = self.critic(all_features[:, :-1].detach())
        critic_loss = F.mse_loss(critic_pred, returns.detach())

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 100.0)
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 100.0)
        self.critic_optimizer.step()

        # Soft-update the EMA target after every critic gradient step.
        self.critic_ema.soft_update(self.critic)

        return {
            "ac/actor_loss": actor_loss.item(),
            "ac/critic_loss": critic_loss.item(),
            "ac/returns_mean": returns.mean().item(),
            "ac/returns_std": returns_std.item(),
        }

    def train(
        self,
        env,
        steps: int,
        seed_episodes: int = 5,
        batch_size: int = 16,
        seq_len: int = 50,
        horizon: int = 15,
        imagine_ratio: int = 1,
        max_episode_steps: int = 500,
        checkpoint_dir: str | None = None,
        checkpoint_every: int = 10_000,
        log_every: int = 1,
        log_fn=print,
    ) -> None:
        for _ in range(seed_episodes):
            collect_episode(env, self.buffer, policy=None, max_steps=max_episode_steps)
            self._env_steps += len(self.buffer.episodes[-1]["obs"])

        iteration = 0
        while self._env_steps < steps:
            length, episode_reward = collect_episode(
                env, self.buffer,
                policy=self._make_acting_policy(),
                max_steps=max_episode_steps,
            )
            self._env_steps += length

            if self.buffer.can_sample(seq_len):
                batch = self.buffer.sample(batch_size, seq_len)
                wm_logs = self._world_model_step(batch)

                ac_logs = {}
                for _ in range(imagine_ratio):
                    batch = self.buffer.sample(batch_size, seq_len)
                    ac_logs = self._actor_critic_step(batch, horizon)

                iteration += 1
                if log_every and iteration % log_every == 0:
                    log_fn(
                        f"[step {self._env_steps}] episode_reward={episode_reward:.1f} "
                        + " ".join(f"{k}={v:.4f}" for k, v in {**wm_logs, **ac_logs}.items())
                    )

            if checkpoint_dir and self._env_steps % checkpoint_every < length:
                self.save_checkpoint(
                    Path(checkpoint_dir) / f"checkpoint_{self._env_steps}.pt"
                )

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def imagine_rollout(self, steps: int = 15) -> dict:
        state = self.rssm.initial_state(1, self.device)

        def policy(feature):
            action, _ = self.actor(feature)
            return action

        with torch.no_grad():
            rollout = self.rssm.imagine(state, policy, steps)
            feature = self.rssm.feature(
                {"deter": rollout["deter"], "stoch": rollout["stoch"]}
            )
            rollout["obs_pred"] = symexp(self.rssm.decoder(feature))
            rollout["reward_pred"] = symexp(self.rssm.reward_head(feature))
        return rollout

    def visualize_latent_space(self, rollout: dict | None = None):
        try:
            import matplotlib.pyplot as plt
        except ImportError as e:
            raise ImportError("needs matplotlib: pip install rssmlite[viz]") from e

        if rollout is None:
            rollout = self.imagine_rollout(steps=50)
        stoch = rollout["stoch"].squeeze(0).cpu()
        _u, _s, v = torch.pca_lowrank(stoch, q=2)
        projected = (stoch @ v).detach().numpy()

        fig, ax = plt.subplots()
        scatter = ax.scatter(projected[:, 0], projected[:, 1],
                             c=range(len(projected)), cmap="viridis")
        fig.colorbar(scatter, ax=ax, label="imagined timestep")
        ax.set_title("RSSM stochastic latent space (2D PCA)")
        ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2")
        return fig

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "init_kwargs": self._init_kwargs,
            "env_steps": self._env_steps,
            "rssm": self.rssm.state_dict(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_ema": self.critic_ema._target.state_dict(),
            "world_model_optimizer": self.world_model_optimizer.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
        }, path)

    @classmethod
    def load_checkpoint(cls, path: str) -> "RSSMAgent":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        agent = cls(**ckpt["init_kwargs"])
        agent._env_steps = ckpt["env_steps"]
        agent.rssm.load_state_dict(ckpt["rssm"])
        agent.actor.load_state_dict(ckpt["actor"])
        agent.critic.load_state_dict(ckpt["critic"])
        if "critic_ema" in ckpt:
            agent.critic_ema._target.load_state_dict(ckpt["critic_ema"])
        agent.world_model_optimizer.load_state_dict(ckpt["world_model_optimizer"])
        agent.actor_optimizer.load_state_dict(ckpt["actor_optimizer"])
        agent.critic_optimizer.load_state_dict(ckpt["critic_optimizer"])
        return agent

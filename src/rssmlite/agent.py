"""
RSSMAgent: the thing a user actually calls. Wraps an RSSM (world model)
with an Actor/Critic and owns the full training loop — collecting real
experience, fitting the world model on it, then training the actor-critic
purely on rollouts imagined by that world model (Section 9's Persona A
usage: `agent.train(env, steps=200_000, checkpoint_dir=...)`).

Kept at the "orchestration" level: the actual math lives in `rssm.py`
(world model) and `actor_critic.py` (policy/value + lambda-returns). If
this file starts explaining *how* a loss is computed rather than *when*,
that logic belongs in one of those two files instead — see Section 6's
10-minute-readability rule.
"""

from __future__ import annotations

from pathlib import Path

import torch
import yaml

from rssmlite.actor_critic import Actor, Critic, lambda_return
from rssmlite.env_utils import action_dim_of, collect_episode
from rssmlite.replay_buffer import ReplayBuffer
from rssmlite.rssm import RSSM
from rssmlite.utils import straight_through_sample, symexp


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
    ):
        self.discrete = discrete
        self.action_dim = action_dim
        self.gamma = gamma
        self.lam = lam
        self.entropy_coef = entropy_coef

        self.rssm = RSSM(obs_dim=obs_dim, action_dim=action_dim, **(rssm_kwargs or {}))
        self.actor = Actor(self.rssm.feature_dim, action_dim, discrete, hidden_dim)
        self.critic = Critic(self.rssm.feature_dim, hidden_dim)

        self.world_model_optimizer = torch.optim.Adam(self.rssm.parameters(), lr=world_model_lr)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.buffer = ReplayBuffer(capacity_episodes=replay_capacity_episodes)
        self._env_steps = 0

        # Saved so `save_checkpoint`/`from_config` can fully reconstruct
        # this agent without the caller needing to remember every kwarg.
        self._init_kwargs = dict(
            obs_dim=obs_dim,
            action_dim=action_dim,
            discrete=discrete,
            rssm_kwargs=rssm_kwargs,
            hidden_dim=hidden_dim,
            world_model_lr=world_model_lr,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            gamma=gamma,
            lam=lam,
            entropy_coef=entropy_coef,
            replay_capacity_episodes=replay_capacity_episodes,
        )

    # ------------------------------------------------------------------
    # Construction from a Gymnasium env / YAML config
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls, env, **kwargs) -> "RSSMAgent":
        """Infer obs_dim/action_dim/discrete straight from a Gymnasium env
        — the common case, so callers don't have to know RSSM internals."""
        import gymnasium as gym

        obs_dim = env.observation_space.shape[0]
        discrete = isinstance(env.action_space, gym.spaces.Discrete)
        return cls(obs_dim=obs_dim, action_dim=action_dim_of(env.action_space), discrete=discrete, **kwargs)

    @classmethod
    def from_config(cls, path: str, env=None) -> "RSSMAgent":
        """Build an agent from a YAML config (roadmap P1.4 formalizes this
        further across all four target envs; this is the minimal version
        needed to unblock P1.3's `train.py --config configs/cartpole.yaml`).

        If the config doesn't specify obs_dim/action_dim/discrete directly,
        pass `env` and they're inferred from it.
        """
        with open(path) as f:
            config = yaml.safe_load(f)

        agent_kwargs = config.get("agent", {})
        if env is not None and not {"obs_dim", "action_dim", "discrete"} <= agent_kwargs.keys():
            import gymnasium as gym

            agent_kwargs.setdefault("obs_dim", env.observation_space.shape[0])
            agent_kwargs.setdefault("action_dim", action_dim_of(env.action_space))
            agent_kwargs.setdefault("discrete", isinstance(env.action_space, gym.spaces.Discrete))
        return cls(**agent_kwargs)

    # ------------------------------------------------------------------
    # Acting in the real environment (posterior filtering, not imagination)
    # ------------------------------------------------------------------

    def _make_acting_policy(self):
        """A fresh, stateful closure for one episode of real interaction.
        Unlike `RSSM.imagine()` (which never sees real observations), this
        updates the belief state from the real obs at every step — exactly
        `RSSM.observe()`'s recurrence, just one step at a time instead of a
        whole batch, and with the actor choosing the action instead of it
        being given.
        """
        state = self.rssm.initial_state(1, next(self.rssm.parameters()).device)
        prev_action = torch.zeros(1, self.action_dim)
        step = {"t": 0}

        def policy(obs):
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                if step["t"] > 0:
                    state["deter"] = self.rssm.dynamics(state["deter"], state["stoch"], prev_action)
                embed = self.rssm.encoder(obs_t)
                post_logits = self.rssm.posterior_net(state["deter"], embed)
                state["stoch"] = straight_through_sample(post_logits).flatten(-2, -1)
                feature = self.rssm.feature(state)
                action, _entropy = self.actor(feature)

            prev_action.copy_(action)
            step["t"] += 1
            return action.squeeze(0).argmax().item() if self.discrete else action.squeeze(0).numpy()

        return policy

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _world_model_step(self, batch: dict) -> dict:
        losses = self.rssm.loss(batch["obs"], batch["action"], batch["reward"], 1.0 - batch["done"])
        self.world_model_optimizer.zero_grad()
        losses["total"].backward()
        self.world_model_optimizer.step()
        return {f"wm/{k}": v.item() for k, v in losses.items()}

    def _actor_critic_step(self, batch: dict, horizon: int) -> dict:
        # Start imagination from every real (posterior) state seen in this
        # batch, flattening (batch, time) into one big batch of start states
        # — cheap way to get many diverse imagination starting points.
        with torch.no_grad():
            rollout = self.rssm.observe(batch["obs"], batch["action"])
        batch_size, seq_len = rollout["deter"].shape[:2]
        start_state = {
            "deter": rollout["deter"].reshape(batch_size * seq_len, -1).detach(),
            "stoch": rollout["stoch"].reshape(batch_size * seq_len, -1).detach(),
        }

        def policy(feature):
            action, _entropy = self.actor(feature)
            return action

        imagined = self.rssm.imagine(start_state, policy, horizon)
        imagined_feature = self.rssm.feature(
            {"deter": imagined["deter"], "stoch": imagined["stoch"]}
        )
        start_feature = self.rssm.feature(start_state).unsqueeze(1)
        all_features = torch.cat([start_feature, imagined_feature], dim=1)  # (N, H+1, feat)

        rewards = symexp(self.rssm.reward_head(imagined_feature))
        continues = torch.sigmoid(self.rssm.continue_head(imagined_feature))
        values = self.critic(all_features)

        returns = lambda_return(rewards.detach(), continues.detach(), values.detach(), self.gamma, self.lam)

        _actions, entropy = self.actor(all_features[:, :-1].detach())
        actor_loss = -(returns.mean()) - self.entropy_coef * entropy.mean()

        critic_pred = self.critic(all_features[:, :-1].detach())
        critic_loss = torch.nn.functional.mse_loss(critic_pred, returns.detach())

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return {"ac/actor_loss": actor_loss.item(), "ac/critic_loss": critic_loss.item()}

    def train(
        self,
        env,
        steps: int,
        seed_episodes: int = 5,
        batch_size: int = 16,
        seq_len: int = 50,
        horizon: int = 15,
        max_episode_steps: int = 500,
        checkpoint_dir: str | None = None,
        checkpoint_every: int = 10_000,
        log_every: int = 1,
        log_fn=print,
    ) -> None:
        """Main loop: collect real experience, fit the world model on it,
        train the actor-critic purely in imagination. Repeats until
        `steps` real environment steps have been collected.

        `checkpoint_dir` can be a Google Drive path once Drive is mounted
        in Colab (e.g. `/content/drive/MyDrive/rssmlite-ckpts`) — no
        special handling needed, it's just a filesystem path.
        """
        for _ in range(seed_episodes):
            collect_episode(env, self.buffer, policy=None, max_steps=max_episode_steps)
            self._env_steps += len(self.buffer.episodes[-1]["obs"])

        iteration = 0
        while self._env_steps < steps:
            length, episode_reward = collect_episode(
                env, self.buffer, policy=self._make_acting_policy(), max_steps=max_episode_steps
            )
            self._env_steps += length

            if self.buffer.can_sample(seq_len):
                batch = self.buffer.sample(batch_size, seq_len)
                wm_logs = self._world_model_step(batch)
                ac_logs = self._actor_critic_step(batch, horizon)

                iteration += 1
                if log_every and iteration % log_every == 0:
                    log_fn(
                        f"[step {self._env_steps}] episode_reward={episode_reward:.1f} "
                        + " ".join(f"{k}={v:.4f}" for k, v in {**wm_logs, **ac_logs}.items())
                    )

            if checkpoint_dir and self._env_steps % checkpoint_every < length:
                self.save_checkpoint(Path(checkpoint_dir) / f"checkpoint_{self._env_steps}.pt")

    # ------------------------------------------------------------------
    # Inspection / visualization (Section 9's Persona A methods)
    # ------------------------------------------------------------------

    def imagine_rollout(self, steps: int = 15) -> dict:
        """Roll the current policy forward purely in imagination from a
        fresh initial state, decoding predicted observations back to real
        units. Useful for sanity-checking what the world model "believes"
        will happen, independent of the real environment."""
        state = self.rssm.initial_state(1, next(self.rssm.parameters()).device)

        def policy(feature):
            action, _entropy = self.actor(feature)
            return action

        with torch.no_grad():
            rollout = self.rssm.imagine(state, policy, steps)
            feature = self.rssm.feature({"deter": rollout["deter"], "stoch": rollout["stoch"]})
            rollout["obs_pred"] = symexp(self.rssm.decoder(feature))
            rollout["reward_pred"] = symexp(self.rssm.reward_head(feature))
        return rollout

    def visualize_latent_space(self, rollout: dict | None = None):
        """2D PCA projection of stochastic latents from a rollout, colored
        by imagined timestep. Requires matplotlib (`pip install
        rssmlite[viz]`); imported lazily so it's not a hard dependency for
        training. Returns the Figure so the caller can `.show()` or save it.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as e:
            raise ImportError(
                "visualize_latent_space() needs matplotlib: pip install rssmlite[viz]"
            ) from e

        if rollout is None:
            rollout = self.imagine_rollout(steps=50)
        stoch = rollout["stoch"].squeeze(0)  # (T, stoch_dim), batch size 1
        _u, _s, v = torch.pca_lowrank(stoch, q=2)
        projected = (stoch @ v).detach().numpy()

        fig, ax = plt.subplots()
        scatter = ax.scatter(projected[:, 0], projected[:, 1], c=range(len(projected)), cmap="viridis")
        fig.colorbar(scatter, ax=ax, label="imagined timestep")
        ax.set_title("RSSM stochastic latent space (2D PCA)")
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        return fig

    # ------------------------------------------------------------------
    # Checkpointing (Section 12: required, Colab sessions disconnect)
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "init_kwargs": self._init_kwargs,
                "env_steps": self._env_steps,
                "rssm": self.rssm.state_dict(),
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "world_model_optimizer": self.world_model_optimizer.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
            },
            path,
        )

    @classmethod
    def load_checkpoint(cls, path: str) -> "RSSMAgent":
        """Fully reconstructs the agent — including optimizer state, so
        training can resume exactly where it left off after a Colab
        disconnect, not just from the model weights."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        agent = cls(**checkpoint["init_kwargs"])
        agent._env_steps = checkpoint["env_steps"]
        agent.rssm.load_state_dict(checkpoint["rssm"])
        agent.actor.load_state_dict(checkpoint["actor"])
        agent.critic.load_state_dict(checkpoint["critic"])
        agent.world_model_optimizer.load_state_dict(checkpoint["world_model_optimizer"])
        agent.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        agent.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        return agent

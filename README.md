# rssmlite

[![PyPI version](https://badge.fury.io/py/rssmlite.svg)](https://pypi.org/project/rssmlite/)
[![CI](https://github.com/Mattral/rssmlite/actions/workflows/ci.yml/badge.svg)](https://github.com/Mattral/rssmlite/actions/workflows/ci.yml)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mattral/rssmlite/blob/main/notebooks/02_cartpole_experiment.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A lightweight, well-documented, **PyTorch-only** library for learning
RSSM-based (Recurrent State-Space Model) world models — in the DreamerV2/V3
tradition — on Gymnasium environments. Designed to run end-to-end on a
**free Google Colab T4 GPU** with zero external datasets.

> **Status:** core library complete and released at `v0.1.0`. `RSSM`, `ReplayBuffer`,
> `RSSMAgent`, four environment configs, `scripts/train.py` CLI, checkpoint save/resume,
> and full documentation are all implemented and tested (29 tests passing).
>
> **World model component:** production-quality. Reconstruction MSE ~0.004 on CartPole,
> near-perfect pole-angle tracking confirmed on a real Colab T4.
>
> **Actor-critic component:** research-grade. Correct implementation of DreamerV3-style
> imagination-based AC with EMA critic target, but does not reliably solve CartPole
> within 200k steps on a small model. This is a known limitation of imagination-based
> RL at small model scale — documented honestly in `configs/cartpole.yaml`.
> The `P2` roadmap (TransformerDynamics ablation) is the next planned improvement.

---

## Install

```bash
pip install rssmlite[envs]        # + gymnasium
pip install rssmlite[envs,viz]    # + matplotlib for plots
```

From source (development):
```bash
git clone https://github.com/Mattral/rssmlite.git
cd rssmlite
pip install -e ".[dev,envs,viz]"
pytest tests/   # 21 passed
```

---

## Quick start

```python
import gymnasium as gym
from rssmlite import RSSMAgent

env = gym.make("CartPole-v1")
agent = RSSMAgent.from_config("configs/cartpole.yaml", env=env)
agent.train(env, steps=200_000, checkpoint_dir="./ckpts")

# Inspect what the model learned
agent.imagine_rollout(steps=15)
agent.visualize_latent_space()
```

Or via the CLI:
```bash
python scripts/train.py --config configs/cartpole.yaml
python scripts/train.py --config configs/cartpole.yaml --resume ckpts/checkpoint_50000.pt
```

---

## Design philosophy

- **PyTorch only.** No JAX, no TensorFlow.
- **Readable over clever.** Every class should be understandable in under
  10 minutes top-to-bottom. If it's longer, it gets split.
- **Config-driven experiments, code-driven architecture.** Hyperparameters
  and environment choice live in YAML; the model code doesn't change per run.
- **Self-contained.** No external dataset downloads — the environment is
  the data source.

---

## Available environments

| Config | Env | Actions | Notes |
|---|---|---|---|
| `configs/cartpole.yaml` | CartPole-v1 | Discrete | Fastest sanity check |
| `configs/acrobot.yaml` | Acrobot-v1 | Discrete | Sparse reward stress test |
| `configs/pendulum.yaml` | Pendulum-v1 | Continuous | Exercises continuous actor |
| `configs/lunarlander.yaml` | LunarLander-v3 | Discrete | Hardest in v1 scope |

---

## Documentation

- [`docs/architecture.md`](docs/architecture.md) — RSSM math, design rationale, file map
- [`docs/getting_started.md`](docs/getting_started.md) — install, first training run, evaluation
- [`docs/api_reference.md`](docs/api_reference.md) — full public API
- [`docs/colab_guide.md`](docs/colab_guide.md) — Colab setup, Drive checkpointing, disconnect recovery

---

## Known simplifications (documented, not hidden)

- Continuous actions (Pendulum) have no tanh squashing — actions can
  technically exceed the env's valid range; Gym clips silently. Fine for
  current results; to fix before reporting serious Pendulum numbers.
- `train.steps` is a total budget across the whole run including pre-resume
  time. Resuming from step 80 000 with `steps: 200000` trains until 200 000.
- Device management is explicit: call `.to(device)` on `agent.rssm`,
  `agent.actor`, `agent.critic` manually. See `docs/colab_guide.md`.

---

## License

MIT — see [LICENSE](LICENSE).

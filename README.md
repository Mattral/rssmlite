# rssmlite

A lightweight, well-documented, PyTorch-only library for learning RSSM-based
(Recurrent State-Space Model) world models — in the DreamerV2/V3 tradition —
on Gymnasium environments. Designed to run end-to-end on a free Google Colab
T4 GPU with zero external datasets.

> **Status:** active development. `RSSM` (world model core), `ReplayBuffer`,
> and `RSSMAgent` (actor-critic trained in imagination) are implemented and
> tested — a `python scripts/train.py --config configs/cartpole.yaml` run
> works end-to-end. Not yet verified: full convergence within 30 min on a
> real Colab T4 (only tiny/fast smoke runs have been checked so far — see
> ROADMAP.md P1.3). `configs/` beyond CartPole and `notebooks/` are next.
>
> The `0.0.1` release on PyPI is a placeholder that reserved this name;
> the real API ships starting at `0.1.0`.

## Design philosophy

- **PyTorch only.** No JAX, no TensorFlow.
- **Readable over clever.** Every class should be understandable by someone
  who's read the DreamerV3 paper once, in under 10 minutes per file.
- **Config-driven experiments, code-driven architecture.** Which
  environment to run and hyperparameters are YAML; the model code doesn't
  change per environment.
- **Self-contained data.** No external dataset downloads — the environment
  *is* the data source.

See `SPEC.md` for the full specification and `ROADMAP.md` for the phased
task breakdown.

## Install (development)

```bash
git clone https://github.com/Mattral/rssmlite.git
cd rssmlite
pip install -e ".[dev]"
pytest tests/
```

## Current public API

```python
from rssmlite import RSSMAgent
import gymnasium as gym

env = gym.make("CartPole-v1")
agent = RSSMAgent.from_env(env)
agent.train(env, steps=200_000, checkpoint_dir="./ckpts")
agent.imagine_rollout(steps=15)
agent.visualize_latent_space()  # needs: pip install rssmlite[viz]
```

Or via the CLI:
```bash
pip install -e ".[dev,envs]"
python scripts/train.py --config configs/cartpole.yaml
python scripts/train.py --config configs/cartpole.yaml --resume checkpoints/checkpoint_50000.pt
```

**Known simplifications (documented, not hidden):**
- Continuous actions (Pendulum) use an unbounded Normal policy — no tanh
  squashing yet, so the actor can in principle propose actions outside the
  env's valid range. Fine for the P1.3 smoke tests; worth fixing before
  reporting real Pendulum results.
- A config's `train.steps` is a **total** budget across the whole run,
  including time before a `--resume`. Resuming from a checkpoint whose
  `env_steps` already exceeds `steps` is a no-op, not an error — bump
  `steps` in the config before resuming a long run.
- `configs/cartpole.yaml` is deliberately minimal (P1.4 formalizes the
  config system across all four target environments; this just unblocks
  P1.3's CLI).

## License

MIT — see [LICENSE](LICENSE).

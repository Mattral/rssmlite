# rssmlite

A lightweight, well-documented, PyTorch-only library for learning RSSM-based
(Recurrent State-Space Model) world models — in the DreamerV2/V3 tradition —
on Gymnasium environments. Designed to run end-to-end on a free Google Colab
T4 GPU with zero external datasets.

> **Status:** active development. The `RSSM` world-model core (encoder,
> GRU dynamics, categorical stochastic latent, decoder, reward/continue
> heads) is implemented and tested. `RSSMAgent` (actor-critic in
> imagination) and `ReplayBuffer` are next — see `ROADMAP.md`.
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
from rssmlite import RSSM

rssm = RSSM(obs_dim=6, action_dim=3)  # e.g. CartPole-style state vector
rollout = rssm.observe(obs_seq, action_seq)     # teacher-forced training pass
losses = rssm.loss(obs_seq, action_seq, reward_seq, continue_seq)
losses["total"].backward()
```

`RSSMAgent` and `ReplayBuffer` (roadmap P1.2–P1.3) will wrap this into a
full training loop.

## License

MIT — see [LICENSE](LICENSE).

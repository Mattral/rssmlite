# `worldmodel` — Roadmap

Priorities follow P0 (do now, blocking everything else) → P1 (core library) →
P2 (research contribution) → P3 (stretch / optional). Within a priority level,
items are roughly ordered but not strictly sequential.

See `SPEC.md` for full context, design philosophy, and success criteria.

---

## P0 — Foundation (blocking, do first)

- [ ] Create PyPI account, enable 2FA, generate API token.
- [ ] Build and upload a minimal placeholder package (`worldmodel==0.0.1`) to
  PyPI to reserve the name permanently. Real description + license + GitHub
  link, not an empty shell.
- [ ] Create `github.com/Mattral/worldmodel` repo (even if just README + LICENSE
  initially) so the GitHub namespace matches the PyPI name.
- [ ] Set up `pyproject.toml` with correct package metadata (name, version,
  author, license, classifiers, Python version requirement).
- [ ] Choose and add a LICENSE (MIT recommended for maximum reuse/citation).
- [ ] Write the initial README with project description, install instructions
  (even if install doesn't do much yet), and a link to `SPEC.md`.

**Exit criteria for P0:** `pip install worldmodel` installs *something* real,
the GitHub repo exists and mirrors the PyPI project, and there is a README a
stranger could read to understand what this is.

---

## P1 — Core Library (Layer 1–2 success criteria)

### P1.1 — RSSM core
- [ ] Implement `RSSM` class: encoder, GRU-based recurrent state, categorical
  stochastic latent, decoder, reward head, continue head.
- [ ] Implement KL balancing + free bits (per DreamerV3).
- [ ] Implement symlog transforms for reward/value scaling.
- [ ] Unit tests for `RSSM` forward pass shapes and loss computation.

### P1.2 — Data pipeline
- [ ] Implement `ReplayBuffer` (sequence storage + fixed-length sampling).
- [ ] Implement environment interaction loop (collect real experience).
- [ ] Smoke test: 100-step end-to-end run on CartPole with no errors.

### P1.3 — Agent & training loop
- [ ] Implement `RSSMAgent`: actor-critic trained on imagined rollouts.
- [ ] Implement `train.py` CLI: `python train.py --config configs/cartpole.yaml`.
- [ ] Implement checkpointing (save/resume), including Colab Drive path support.
- [ ] Get a full CartPole training run working end-to-end on Colab T4 in
  under 30 minutes.

### P1.4 — Config system
- [ ] YAML config schema covering environment, model hyperparameters, training
  schedule, logging/checkpoint paths.
- [ ] `RSSMAgent.from_config(path)` constructor.
- [ ] Configs for all four target environments (CartPole, Acrobot, Pendulum,
  LunarLander).

### P1.5 — Visualization & evaluation
- [ ] `agent.imagine_rollout()` — generate and visualize an imagined trajectory.
- [ ] `agent.visualize_latent_space()` — t-SNE or similar plot of learned latents.
- [ ] Reconstruction quality reporting (real vs. predicted observations).
- [ ] Sample-efficiency comparison plot vs. a simple model-free baseline (PPO
  via Stable-Baselines3 or a minimal custom implementation) on at least one
  environment.

### P1.6 — Notebooks
- [ ] `01_world_model_intro.ipynb`
- [ ] `02_cartpole_experiment.ipynb`
- [ ] `03_lunarlander_experiment.ipynb`
- [ ] Every notebook opens and runs top-to-bottom on a fresh Colab T4 runtime
  with only the standard bootstrap cell.

### P1.7 — Testing & CI
- [ ] `tests/` covering RSSM, replay buffer, and an end-to-end smoke test.
- [ ] GitHub Actions CI running tests on every push/PR.

### P1.8 — Documentation
- [ ] `docs/architecture.md`
- [ ] `docs/getting_started.md`
- [ ] `docs/api_reference.md`
- [ ] `docs/colab_guide.md`
- [ ] README polish: badges (PyPI version, CI status, Colab links), usage
  example, link to docs.

**Exit criteria for P1:** all Layer 1 and Layer 2 success criteria in
`SPEC.md` Section 5 are met. This is the "real, usable library" milestone —
tag as `v0.2.0` or similar.

---

## P2 — Research Contribution (Layer 3 success criteria)

- [ ] Implement `TransformerDynamics` as an alternative backbone to the GRU.
- [ ] Design and run the ablation study: at minimum, compare (a) with/without
  stochastic latents, (b) GRU vs. Transformer backbone, (c) with/without KL
  balancing.
- [ ] `04_ablation_study.ipynb` — reproducible notebook for the above.
- [ ] `docs/ablation_study.md` — written summary of methodology and findings.
- [ ] Decide, based on results, whether findings are substantial enough to
  justify a paper write-up (they don't have to be groundbreaking — a clean,
  honest negative or mixed result is a valid contribution too).

**Exit criteria for P2:** a genuine novelty artifact exists with documented,
reproducible results — not just a reimplementation of existing work.

---

## P3 — Publishing & Community (stretch, optional)

- [ ] Register an ORCID iD.
- [ ] Draft a short paper/technical report from `docs/ablation_study.md`.
- [ ] Seek arXiv endorsement (cs.LG or cs.AI) using the GitHub repo as a
  credibility signal — ask via Hugging Face forums, ResearchGate, or a
  personal network contact who is an existing arXiv author.
- [ ] Submit to an appropriate venue: arXiv preprint first, then consider a
  NeurIPS/ICLR/ICML workshop (higher acceptance rate, explicitly open to
  independent researchers, useful feedback loop).
- [ ] Add `CONTRIBUTING.md` and respond to any external issues/PRs.
- [ ] Optional: write a blog-style post (personal site or Medium) walking
  through the project, separate from the formal paper.

**Exit criteria for P3:** none required — this tier is ambition, not
obligation. Reaching P2 already constitutes project success per `SPEC.md`.

---

## Immediate Next Action

The very first task, before writing any model code, is the first checkbox
under **P0**: reserve the PyPI name. Everything else can wait a day; that
cannot, since names are first-come-first-served with no reservation system.

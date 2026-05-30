# NeuroFade

**Watch your neural network forget in real time.**

Live forgetting visualizer for continual learning — tracks per-layer, per-neuron activation retention as you train on new tasks.

🌐 **[neurofade.gitinxyzbot.io](https://gitinxyzbot.github.io/neurofade/)** · 📦 **[PyPI](https://pypi.org/project/neurofade/)**

---

## The Problem

Catastrophic forgetting is invisible. You fine-tune on task 2, your model silently destroys task 1. You only find out when you run an eval — and by then the damage is done.

NeuroFade makes it undeniable. Green = alive. Red = forgetting. Dark = dead.

## Install

```bash
pip install neurofade
```

## Usage

```python
from neurofade import ForgettingVisualizer

# After training Task 1
viz = ForgettingVisualizer(model)
viz.set_baseline(task1_loader)

# Train Task 2 — watch it forget
with viz.watch():
    trainer.fit(task2_loader)

# See the damage
viz.summary()              # terminal table with per-layer retention
viz.export("forget.gif")   # shareable heatmap animation
```

## Output

```
Layer                                     Avg Retention  Alive %
─────────────────────────────────────────────────────────────────
0                                              100.0%  100.0%
1                                                4.1%    5.1%   ← 🔴 dying
2                                               99.8%  100.0%
3                                               16.5%   17.0%
4                                              100.0%  100.0%
5                                               18.0%   18.0%
6                                              100.0%  100.0%
7                                                0.0%    0.0%   ← 💀 dead
...
```

## Why This Matters

- Know **which layers** to apply EWC, replay, or PackNet — not guesswork
- Catch forgetting **during training**, not after
- Export shareable GIFs of your model's death
- One-line integration with any PyTorch model

## Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.0
- Pillow ≥ 9.0

## License

MIT

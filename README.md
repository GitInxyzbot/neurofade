# NeuroFade

Live Forgetting Visualizer — see your neural network forget in real time.

![Example](assets/demo.gif)

## What

Tracks per-neuron activation retention across training tasks and renders the forgetting process as animated heatmaps. Watch your model lose knowledge in real time.

## Install

```bash
pip install neurofade
```

## Quick Start

```python
from neurofade import ForgettingVisualizer

viz = ForgettingVisualizer(model, baseline_task=train_task_1_loader)

# Wraps your training loop
with viz.watch():
    trainer.fit(task_2_loader)

viz.export("forgetting.mp4")
viz.share()  # Upload to public URL
```

## Why

Catastrophic forgetting is invisible until it tanks your model. NeuroFade makes it undeniable.

- **Layers fade in real-time** — see which neurons die first
- **Export as GIF/MP4** — shareable artifacts
- **Per-task baselines** — measures retention against any previous task
- **Framework-agnostic** — works with PyTorch, TensorFlow, JAX

## The Vibe

Your model is dying. Now you can see it.
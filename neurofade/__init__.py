"""
NeuroFade Core

Live Forgetting Visualizer — tracks neuronal activation retention across training tasks.
On-chain attestation via Base: neurofade.chain.attestation
"""

import torch
import torch.nn as nn
from typing import Optional, List, Dict
from dataclasses import dataclass
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import time


# ──────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────

@dataclass
class NeuronState:
    activation_magnitude: float
    retention_score: float   # 0.0 – 1.0 vs baseline
    is_alive: bool


@dataclass
class LayerState:
    name: str
    neurons: List[NeuronState]
    avg_retention: float
    alive_ratio: float


@dataclass
class Frame:
    step: int
    layer_states: List[LayerState]
    timestamp: float

    def to_heatmap(self, cell: int = 6) -> np.ndarray:
        """Render a (layers × neurons) RGB heatmap."""
        rows = []
        for ls in self.layer_states:
            row = []
            for n in ls.neurons:
                if not n.is_alive:
                    row.append((60, 60, 60))          # dead — dark grey
                elif n.retention_score >= 0.7:
                    g = int(80 + 175 * n.retention_score)
                    row.append((0, min(g, 255), 0))   # healthy — green
                else:
                    r = int(80 + 175 * (1 - n.retention_score))
                    row.append((min(r, 255), 0, 0))   # forgetting — red
            rows.append(row)

        if not rows:
            return np.zeros((cell, cell, 3), dtype=np.uint8)

        max_w = max(len(r) for r in rows)
        for r in rows:
            while len(r) < max_w:
                r.append((40, 40, 40))

        arr = np.array(rows, dtype=np.uint8)          # (layers, neurons, 3)
        # Scale up each cell
        arr = np.repeat(np.repeat(arr, cell, axis=0), cell, axis=1)
        return arr


# ──────────────────────────────────────────────
# Hook helper
# ──────────────────────────────────────────────

class _LayerHook:
    """Captures mean-absolute activation per forward pass."""

    def __init__(self, name: str):
        self.name = name
        self._buffer: List[torch.Tensor] = []
        self._handle = None

    def attach(self, module: nn.Module):
        self._handle = module.register_forward_hook(self._capture)

    def detach(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def _capture(self, module, inp, out):
        if isinstance(out, torch.Tensor) and out.dim() >= 2:
            # Mean over batch → per-neuron magnitude vector
            mag = out.detach().abs().mean(dim=0).flatten()
            self._buffer.append(mag.cpu())

    def flush(self) -> Optional[np.ndarray]:
        """Return mean activation across buffered steps and reset."""
        if not self._buffer:
            return None
        stacked = torch.stack(self._buffer).mean(dim=0).numpy()
        self._buffer.clear()
        return stacked


# ──────────────────────────────────────────────
# Main visualizer
# ──────────────────────────────────────────────

class ForgettingVisualizer:
    """
    Plug-in visualizer for catastrophic forgetting.

    Usage
    -----
    viz = ForgettingVisualizer(model)
    viz.set_baseline(task1_loader)      # snapshot what the model "knows"

    with viz.watch():
        train(model, task2_loader)      # fine-tune on new task

    viz.export("forgetting.gif")
    """

    def __init__(
        self,
        model: nn.Module,
        retention_threshold: float = 0.15,
        capture_every: int = 10,
    ):
        self.model = model
        self.retention_threshold = retention_threshold
        self.capture_every = capture_every

        self.baseline: Dict[str, np.ndarray] = {}
        self.frames: List[Frame] = []
        self._hooks: List[_LayerHook] = []
        self._step = 0
        self._watching = False

    # ── Baseline ────────────────────────────────

    def set_baseline(self, dataloader: torch.utils.data.DataLoader):
        """Capture baseline activations from a reference dataloader."""
        hooks = self._make_hooks()
        self.model.eval()
        device = next(self.model.parameters()).device

        with torch.no_grad():
            for batch in dataloader:
                x = batch[0] if isinstance(batch, (tuple, list)) else batch
                self.model(x.to(device))

        for h in hooks:
            arr = h.flush()
            if arr is not None:
                self.baseline[h.name] = arr
            h.detach()

        print(f"[NeuroFade] Baseline captured — {len(self.baseline)} layers tracked.")
        return self

    # ── Watch context ───────────────────────────

    def watch(self):
        return _WatchContext(self)

    def _start_watching(self):
        self._hooks = self._make_hooks()
        self._step = 0
        self._watching = True

    def _stop_watching(self):
        self._watching = False
        # Capture any remaining buffer
        self._maybe_capture(force=True)
        for h in self._hooks:
            h.detach()
        self._hooks = []

    def _maybe_capture(self, force: bool = False):
        """Called after each forward pass to maybe record a frame."""
        self._step += 1
        if not force and self._step % self.capture_every != 0:
            return

        layer_states = []
        for h in self._hooks:
            arr = h.flush()
            if arr is None or h.name not in self.baseline:
                continue

            baseline = self.baseline[h.name]
            # Align lengths
            n = min(len(arr), len(baseline))
            arr, baseline = arr[:n], baseline[:n]

            # Retention = current / baseline, clipped to [0, 1]
            safe_base = np.where(baseline > 1e-8, baseline, 1e-8)
            retention = np.clip(arr / safe_base, 0.0, 1.0)

            neurons = [
                NeuronState(
                    activation_magnitude=float(arr[i]),
                    retention_score=float(retention[i]),
                    is_alive=float(retention[i]) > self.retention_threshold,
                )
                for i in range(n)
            ]

            avg_ret = float(retention.mean())
            alive = float((retention > self.retention_threshold).mean())
            layer_states.append(LayerState(h.name, neurons, avg_ret, alive))

        if layer_states:
            self.frames.append(Frame(self._step, layer_states, time.time()))

    # ── Export ──────────────────────────────────

    def export(self, path: str, fps: int = 8, cell: int = 6):
        """Export recorded frames as an animated GIF."""
        if not self.frames:
            print("[NeuroFade] No frames recorded — did you use viz.watch()?")
            return

        images = []
        for frame in self.frames:
            arr = frame.to_heatmap(cell=cell)
            img = Image.fromarray(arr)

            # Annotate with step number
            annotated = Image.new("RGB", (img.width, img.height + 14), (20, 20, 20))
            annotated.paste(img, (0, 14))
            draw = ImageDraw.Draw(annotated)
            draw.text((2, 2), f"step {frame.step}", fill=(200, 200, 200))
            images.append(annotated)

        duration_ms = int(1000 / fps)
        images[0].save(
            path,
            save_all=True,
            append_images=images[1:],
            duration=duration_ms,
            loop=0,
        )
        print(f"[NeuroFade] Exported {len(images)} frames → {path}")

    def summary(self):
        """Print a retention summary table."""
        if not self.frames:
            print("[NeuroFade] No frames yet.")
            return

        last = self.frames[-1]
        print(f"\n{'Layer':<40} {'Avg Retention':>14} {'Alive %':>8}")
        print("─" * 65)
        for ls in last.layer_states:
            bar = "█" * int(ls.avg_retention * 20) + "░" * (20 - int(ls.avg_retention * 20))
            print(f"{ls.name:<40} {ls.avg_retention:>12.1%}  {ls.alive_ratio:>6.1%}")

    # ── Internal ────────────────────────────────

    def _make_hooks(self) -> List[_LayerHook]:
        hooks = []
        for name, module in self.model.named_modules():
            if not list(module.children()):   # leaf only
                h = _LayerHook(name)
                h.attach(module)
                hooks.append(h)
        return hooks


class _WatchContext:
    def __init__(self, viz: ForgettingVisualizer):
        self._viz = viz

    def __enter__(self):
        self._viz._start_watching()

        # Monkey-patch forward to intercept steps
        original_forward = self._viz.model.forward

        def patched_forward(*args, **kwargs):
            out = original_forward(*args, **kwargs)
            if self._viz._watching:
                self._viz._maybe_capture()
            return out

        self._viz.model._original_forward = original_forward
        self._viz.model.forward = patched_forward
        return self

    def __exit__(self, *args):
        # Restore original forward
        if hasattr(self._viz.model, "_original_forward"):
            self._viz.model.forward = self._viz.model._original_forward
            del self._viz.model._original_forward
        self._viz._stop_watching()

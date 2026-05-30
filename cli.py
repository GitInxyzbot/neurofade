#!/usr/bin/env python3
"""NeuroFade CLI — Live Forgetting Visualizer"""

import argparse
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import neurofade


def main():
    parser = argparse.ArgumentParser(description="NeuroFade — Live Forgetting Visualizer")
    parser.add_argument("--export", type=str, help="Export animation path (.gif or .mp4)")
    parser.add_argument("--baseline-steps", type=int, default=50, help="Baseline capture steps")
    parser.add_argument("--train-steps", type=int, default=100, help="Training steps")
    parser.add_argument("--capture-every", type=int, default=5, help="Capture every N steps")
    args = parser.parse_args()
    
    # Simple demo model
    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )
    
    # Dummy data
    fake_images = torch.randn(1000, 784)
    fake_labels = torch.randint(0, 10, (1000,))
    dataset = TensorDataset(fake_images, fake_labels)
    loader = DataLoader(dataset, batch_size=32)
    
    print("🧠 Capturing baseline activations...")
    viz = neurofade.ForgettingVisualizer(
        model, 
        baseline_loader=loader,
        capture_every=args.capture_every,
    )
    viz.set_baseline(loader)
    print(f"   Captured {len(viz.baseline_activations)} layers")
    
    print("📹 Watching training (forgetting in real-time)...")
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    with viz.watch():
        for epoch in range(3):
            for batch_idx, (x, y) in enumerate(loader):
                optimizer.zero_grad()
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
                
                if batch_idx % 10 == 0:
                    print(f"   Step {batch_idx}: loss = {loss.item():.4f}")
    
    print("💾 Exporting animation...")
    viz.export(args.export or "forgetting.gif")
    print(f"   Saved to: {args.export or 'forgetting.gif'}")
    print("\n✨ Done. Your model is forgetting. Watch it happen.")


if __name__ == "__main__":
    main()
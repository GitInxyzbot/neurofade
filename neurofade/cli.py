"""NeuroFade CLI"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from neurofade import ForgettingVisualizer


def main():
    parser = argparse.ArgumentParser(
        description="NeuroFade — Live Forgetting Visualizer"
    )
    parser.add_argument("--output", "-o", default="forgetting.gif", help="Output GIF path")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs for demo")
    parser.add_argument("--fps", type=int, default=6, help="GIF frames per second")
    args = parser.parse_args()

    print("🧠 NeuroFade demo — running forgetting simulation...\n")

    model = nn.Sequential(
        nn.Linear(64, 256), nn.ReLU(),
        nn.Linear(256, 128), nn.ReLU(),
        nn.Linear(128, 64), nn.ReLU(),
        nn.Linear(64, 10),
    )

    loader1 = DataLoader(
        TensorDataset(torch.randn(300, 64), torch.randint(0, 10, (300,))),
        batch_size=32
    )
    loader2 = DataLoader(
        TensorDataset(torch.randn(300, 64) * 3.0, torch.randint(0, 10, (300,))),
        batch_size=32
    )

    viz = ForgettingVisualizer(model, capture_every=5)
    viz.set_baseline(loader1)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    criterion = nn.CrossEntropyLoss()

    model.train()
    with viz.watch():
        for epoch in range(args.epochs):
            for x, y in loader2:
                optimizer.zero_grad()
                criterion(model(x), y).backward()
                optimizer.step()
            print(f"  Epoch {epoch + 1}/{args.epochs} done")

    print()
    viz.summary()
    viz.export(args.output, fps=args.fps)
    print(f"\n✅ Open {args.output} to see your model forget.")


if __name__ == "__main__":
    main()

"""
Example: Using NeuroFade with a Real Model
=======================================

This example demonstrates how to integrate NeuroFade with any PyTorch model.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from neurofade import ForgettingVisualizer


class SimpleCNN(nn.Module):
    """Simple CNN for MNIST-like data."""
    
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def run_demo():
    # Create dummy MNIST-like data
    train_data = TensorDataset(
        torch.randn(500, 1, 28, 28),  # images
        torch.randint(0, 10, (500,)),   # labels
    )
    train_loader = DataLoader(train_data, batch_size=32)
    
    # Initialize model
    model = SimpleCNN()
    
    # ===== PHASE 1: Capture Baseline =====
    # After training on Task 1, capture what the model "knows"
    print("📸 Phase 1: Capturing Task 1 baseline...")
    
    # Simulate training on Task 1 first
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(2):
        for x, y in train_loader:
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
    
    # Now capture baseline with NeuroFade
    viz = ForgettingVisualizer(model, baseline_loader=train_loader, capture_every=5)
    viz.set_baseline(train_loader)
    print(f"   Captured baseline from {len(viz.baseline_activations)} layers")
    
    # ===== PHASE 2: Train on New Task, Watch Forgetting =====
    print("📹 Phase 2: Training on Task 2, watching forgetting...")
    
    # New task (different labels/distribution)
    task2_data = TensorDataset(
        torch.randn(500, 1, 28, 28),
        torch.randint(0, 10, (500,)),
    )
    task2_loader = DataLoader(task2_data, batch_size=32)
    
    model.train()
    with viz.watch():
        for epoch in range(5):
            for batch_idx, (x, y) in enumerate(task2_loader):
                optimizer.zero_grad()
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
                
                if batch_idx % 20 == 0:
                    print(f"   Step {batch_idx}: watching...")
    
    # ===== PHASE 3: Export & Share =====
    print("💾 Phase 3: Exporting visualization...")
    viz.export("neurofade_example.gif")
    print("   Saved to: neurofade_example.gif")
    
    # Get stats
    print("\n📊 Layer Retention Summary:")
    for layer_name, baseline in list(viz.baseline_activations.items())[:3]:
        print(f"   {layer_name}: baseline captured")


if __name__ == "__main__":
    run_demo()
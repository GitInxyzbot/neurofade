"""
NeuroFade × Base — On-Chain Model Health Attestation
=====================================================
This example shows how to:
  1. Train a model on Task A (baseline)
  2. Fine-tune on Task B (forgetting happens here)
  3. Detect forgetting with NeuroFade
  4. Post a verifiable health certificate to Base mainnet

The tx hash IS the certificate — immutable, public, composable.

Usage
-----
    python examples/base_attestation.py --key 0xYOUR_PRIVATE_KEY

    # Verify an existing attestation
    python examples/base_attestation.py --verify 0xTX_HASH
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Add parent dir to path when running from repo root
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from neurofade import ForgettingVisualizer
from neurofade.chain.attestation import post_attestation, verify_attestation


# ── Tiny demo model ──────────────────────────────────────────────────────────

class SmallNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        return self.fc3(self.relu2(self.fc2(self.relu1(self.fc1(x)))))


def make_loader(n=200, features=64, classes=10):
    x = torch.randn(n, features)
    y = torch.randint(0, classes, (n,))
    return DataLoader(TensorDataset(x, y), batch_size=32, shuffle=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def run_demo(private_key: str):
    print("=" * 60)
    print("  NeuroFade × Base — Model Health Attestation Demo")
    print("=" * 60)

    model = SmallNet()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    task_a = make_loader()
    task_b = make_loader()

    # ── Step 1: Train on Task A ───────────────────────────────────────────────
    print("\n[1/4] Training on Task A (building baseline)...")
    model.train()
    for epoch in range(3):
        for x, y in task_a:
            optimizer.zero_grad()
            loss_fn(model(x), y).backward()
            optimizer.step()

    # ── Step 2: Capture baseline ──────────────────────────────────────────────
    print("\n[2/4] Capturing baseline activations...")
    viz = ForgettingVisualizer(model, capture_every=5)
    viz.set_baseline(task_a)

    # ── Step 3: Fine-tune on Task B (forgetting!) ─────────────────────────────
    print("\n[3/4] Fine-tuning on Task B (forgetting begins)...")
    with viz.watch():
        for epoch in range(5):
            for x, y in task_b:
                model.train()
                optimizer.zero_grad()
                loss_fn(model(x), y).backward()
                optimizer.step()

    viz.summary()
    viz.export("/tmp/neurofade_base_demo.gif")
    print("\n  GIF saved → /tmp/neurofade_base_demo.gif")

    # ── Step 4: Post attestation to Base ─────────────────────────────────────
    print("\n[4/4] Posting health certificate to Base mainnet...")
    cert = post_attestation(
        viz=viz,
        private_key=private_key,
        model=model,
        verbose=True,
    )

    print("\n" + "=" * 60)
    print("  CERTIFICATE ISSUED")
    print("=" * 60)
    print(f"  Attester      : {cert.attester}")
    print(f"  Avg Retention : {cert.avg_retention:.1%}")
    print(f"  Alive Ratio   : {cert.alive_ratio:.1%}")
    print(f"  Worst Layer   : {cert.worst_layer}")
    print(f"  Report Hash   : {cert.report_hash[:32]}...")
    print(f"  TX (proof)    : https://basescan.org/tx/{cert.tx_hash}")
    print("=" * 60)

    return cert


def run_verify(tx_hash: str):
    print(f"\nVerifying attestation: {tx_hash}")
    result = verify_attestation(tx_hash)
    if result is None:
        print("❌ Not a valid NeuroFade attestation.")
        return
    print("\n✅ Valid NeuroFade health certificate:")
    for k, v in result.items():
        print(f"  {k:<20}: {v}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NeuroFade × Base attestation demo")
    parser.add_argument("--key", help="Private key (0x...) for posting attestation")
    parser.add_argument("--verify", help="TX hash to verify")
    args = parser.parse_args()

    if args.verify:
        run_verify(args.verify)
    elif args.key:
        run_demo(args.key)
    else:
        print("Usage:")
        print("  python examples/base_attestation.py --key 0xYOUR_PRIVATE_KEY")
        print("  python examples/base_attestation.py --verify 0xTX_HASH")

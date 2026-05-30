"""
NeuroFade On-Chain Attestation
-------------------------------
Posts model health certificates to Base mainnet as permanent,
verifiable records. No smart contract deploy required — uses
calldata-only transactions (0-value ETH tx with structured data).

This makes NeuroFade results:
  - Immutable (on-chain, cannot be altered)
  - Public (anyone can verify an AI agent's model health)
  - Composable (other contracts can read attestations via indexers)

Flow:
  1. Run NeuroFade diagnostics → get retention scores
  2. Hash the report (SHA-256)
  3. Post hash + metadata as calldata to Base
  4. Return tx hash = the "health certificate"
"""

import hashlib
import json
import time
from dataclasses import dataclass, asdict
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from neurofade import ForgettingVisualizer

# Base mainnet RPC
BASE_RPC = "https://mainnet.base.org"

# NeuroFade attestation receiver address (burn address — calldata only, no contract needed)
# Anyone can verify by querying txs TO this address with input data prefix 0x4e465244 ("NFRD")
ATTESTATION_ADDRESS = "0x000000000000000000000000000000000000dEaD"

# 4-byte selector prefix: bytes4(keccak256("neurofade.attest"))
ATTEST_PREFIX = b"\x4e\x46\x52\x44"   # "NFRD" in ASCII


@dataclass
class HealthCertificate:
    """The verifiable health report for a model checkpoint."""
    version: str = "1"
    timestamp: int = 0
    model_hash: str = ""          # SHA-256 of model state_dict bytes
    report_hash: str = ""         # SHA-256 of full JSON report
    avg_retention: float = 0.0    # 0.0–1.0 across all layers
    alive_ratio: float = 0.0      # fraction of non-dead neurons
    layer_count: int = 0
    worst_layer: str = ""         # layer with lowest retention
    worst_retention: float = 1.0
    attester: str = ""            # wallet address that posted it
    tx_hash: str = ""             # populated after posting


def build_report(viz: "ForgettingVisualizer") -> dict:
    """Build a JSON-serialisable health report from a ForgettingVisualizer."""
    if not viz.frames:
        raise ValueError("No frames recorded — run set_baseline() then watch() first.")

    last = viz.frames[-1]
    retentions = [ls.avg_retention for ls in last.layer_states]
    alive_ratios = [ls.alive_ratio for ls in last.layer_states]
    worst_idx = int(min(range(len(retentions)), key=lambda i: retentions[i]))

    report = {
        "version": "1",
        "timestamp": int(time.time()),
        "layer_count": len(last.layer_states),
        "avg_retention": float(sum(retentions) / len(retentions)) if retentions else 0.0,
        "alive_ratio": float(sum(alive_ratios) / len(alive_ratios)) if alive_ratios else 0.0,
        "worst_layer": last.layer_states[worst_idx].name if last.layer_states else "",
        "worst_retention": retentions[worst_idx] if retentions else 1.0,
        "layers": [
            {
                "name": ls.name,
                "avg_retention": ls.avg_retention,
                "alive_ratio": ls.alive_ratio,
            }
            for ls in last.layer_states
        ],
    }
    return report


def hash_report(report: dict) -> str:
    """Return SHA-256 hex of the canonical JSON report."""
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def hash_model(model) -> str:
    """Return SHA-256 hex of all model parameters (state_dict)."""
    import torch, io
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return hashlib.sha256(buf.getvalue()).hexdigest()


def encode_calldata(cert: HealthCertificate) -> bytes:
    """
    Encode the certificate as calldata:
    [4-byte prefix][1-byte version][8-byte timestamp][32-byte report_hash]
    + JSON metadata tail for human readability in block explorers.
    """
    prefix = ATTEST_PREFIX
    version_byte = int(cert.version).to_bytes(1, "big")
    ts_bytes = cert.timestamp.to_bytes(8, "big")
    report_hash_bytes = bytes.fromhex(cert.report_hash)

    meta = json.dumps({
        "neurofade": True,
        "avg_retention": round(cert.avg_retention, 4),
        "alive_ratio": round(cert.alive_ratio, 4),
        "layer_count": cert.layer_count,
        "worst_layer": cert.worst_layer,
        "worst_retention": round(cert.worst_retention, 4),
        "model_hash": cert.model_hash[:16],   # first 8 bytes for brevity
    }, separators=(",", ":")).encode()

    return prefix + version_byte + ts_bytes + report_hash_bytes + b"|" + meta


def post_attestation(
    viz: "ForgettingVisualizer",
    private_key: str,
    rpc_url: str = BASE_RPC,
    model=None,
    verbose: bool = True,
) -> HealthCertificate:
    """
    Post a NeuroFade health certificate to Base mainnet.

    Parameters
    ----------
    viz         : ForgettingVisualizer with frames recorded
    private_key : hex private key of the attesting wallet (0x...)
    rpc_url     : Base RPC endpoint (default: public Base mainnet)
    model       : optional nn.Module to include model hash
    verbose     : print status messages

    Returns
    -------
    HealthCertificate with tx_hash populated
    """
    try:
        from web3 import Web3
        from eth_account import Account
    except ImportError:
        raise ImportError("pip install web3 eth-account")

    w3 = Web3(Web3.HTTPProvider(rpc_url))
    if not w3.is_connected():
        raise ConnectionError(f"Cannot connect to Base RPC: {rpc_url}")

    account = Account.from_key(private_key)
    attester = account.address

    # Build report
    report = build_report(viz)
    report_hash = hash_report(report)
    model_hash = hash_model(model) if model is not None else "0" * 64

    cert = HealthCertificate(
        version="1",
        timestamp=report["timestamp"],
        model_hash=model_hash,
        report_hash=report_hash,
        avg_retention=report["avg_retention"],
        alive_ratio=report["alive_ratio"],
        layer_count=report["layer_count"],
        worst_layer=report["worst_layer"],
        worst_retention=report["worst_retention"],
        attester=attester,
    )

    calldata = encode_calldata(cert)

    nonce = w3.eth.get_transaction_count(attester)
    gas_price = w3.eth.gas_price

    tx = {
        "to": ATTESTATION_ADDRESS,
        "value": 0,
        "data": calldata,
        "gas": 100_000,
        "gasPrice": gas_price,
        "nonce": nonce,
        "chainId": 8453,  # Base mainnet
    }

    if verbose:
        print(f"[NeuroFade] Posting attestation from {attester}")
        print(f"  avg_retention : {cert.avg_retention:.1%}")
        print(f"  alive_ratio   : {cert.alive_ratio:.1%}")
        print(f"  worst_layer   : {cert.worst_layer} ({cert.worst_retention:.1%})")
        print(f"  report_hash   : {report_hash[:16]}...")

    signed = account.sign_transaction(tx)
    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    tx_hex = tx_hash.hex()
    if not tx_hex.startswith("0x"):
        tx_hex = "0x" + tx_hex

    cert.tx_hash = tx_hex

    if verbose:
        print(f"[NeuroFade] ✅ Attestation posted!")
        print(f"  TX: https://basescan.org/tx/{tx_hex}")

    return cert


def verify_attestation(tx_hash: str, rpc_url: str = BASE_RPC) -> Optional[dict]:
    """
    Fetch and decode a NeuroFade attestation from a tx hash.

    Returns the decoded certificate dict, or None if not a valid attestation.
    """
    try:
        from web3 import Web3
    except ImportError:
        raise ImportError("pip install web3")

    w3 = Web3(Web3.HTTPProvider(rpc_url))
    tx = w3.eth.get_transaction(tx_hash)
    data = bytes(tx["input"])

    if not data.startswith(ATTEST_PREFIX):
        return None

    version = data[4]
    timestamp = int.from_bytes(data[5:13], "big")
    report_hash = data[13:45].hex()

    # Parse JSON tail
    pipe_idx = data.find(b"|", 45)
    meta = {}
    if pipe_idx != -1:
        try:
            meta = json.loads(data[pipe_idx + 1:])
        except Exception:
            pass

    return {
        "version": version,
        "timestamp": timestamp,
        "report_hash": report_hash,
        "attester": tx["from"],
        "block": tx["blockNumber"],
        "tx_hash": tx_hash,
        **meta,
    }

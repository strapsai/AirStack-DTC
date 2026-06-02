from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from diffphysdrone_px4_wrapper.inference import OnnxPolicyBackend


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a local ONNX inference smoke test for VOXL/offboard dry-run prep."
    )
    parser.add_argument("--onnx", required=True, help="Path to exported ONNX model.")
    parser.add_argument("--dim-obs", type=int, default=10, choices=(7, 10))
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--provider", action="append", default=None)
    args = parser.parse_args()

    backend = OnnxPolicyBackend(Path(args.onnx), args.dim_obs, providers=args.provider)
    depth = torch.ones((1, 1, 12, 16), dtype=torch.float32)
    state = torch.zeros((1, args.dim_obs), dtype=torch.float32)

    start = time.perf_counter()
    last_action = None
    for _ in range(max(1, args.steps)):
        last_action, _ = backend.infer(depth, state)
    elapsed = time.perf_counter() - start
    rate_hz = args.steps / elapsed if elapsed > 0.0 else float("inf")

    print(f"steps={args.steps}")
    print(f"elapsed_sec={elapsed:.4f}")
    print(f"rate_hz={rate_hz:.2f}")
    print(f"last_action={last_action.flatten().tolist() if last_action is not None else []}")


if __name__ == "__main__":
    main()

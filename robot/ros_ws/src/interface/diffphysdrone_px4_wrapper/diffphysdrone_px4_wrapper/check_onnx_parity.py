from __future__ import annotations

import argparse
from pathlib import Path

import torch

from diffphysdrone_px4_wrapper.export_onnx import export_checkpoint_to_onnx
from diffphysdrone_px4_wrapper.inference import (
    HIDDEN_DIM,
    OnnxPolicyBackend,
    load_torch_model,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare PyTorch checkpoint inference against exported ONNX inference."
    )
    parser.add_argument(
        "--checkpoint",
        default="/airlab-storage/chiron/models/diffphysdrone/checkpoint0004.pth",
    )
    parser.add_argument(
        "--onnx",
        default="/airlab-storage/chiron/models/diffphysdrone/checkpoint0004.onnx",
    )
    parser.add_argument("--export", action="store_true", help="Export ONNX before checking.")
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    onnx_path = Path(args.onnx)
    if args.export or not onnx_path.exists():
        export_checkpoint_to_onnx(checkpoint_path, onnx_path)

    torch_model, dim_obs = load_torch_model(checkpoint_path, torch.device("cpu"))
    torch_hidden = torch.zeros((1, HIDDEN_DIM), dtype=torch.float32)
    onnx_backend = OnnxPolicyBackend(onnx_path, dim_obs)

    generator = torch.Generator().manual_seed(7)
    max_action_error = 0.0
    max_hidden_error = 0.0
    for _ in range(max(1, args.steps)):
        depth = torch.rand((1, 1, 12, 16), generator=generator)
        state = torch.randn((1, dim_obs), generator=generator)
        with torch.inference_mode():
            torch_action, _, torch_hidden = torch_model(depth, state, torch_hidden)
        onnx_action, onnx_hidden = onnx_backend.infer(depth, state)
        action_error = float((torch_action - onnx_action).abs().max())
        hidden_error = float((torch_hidden - onnx_hidden).abs().max())
        max_action_error = max(max_action_error, action_error)
        max_hidden_error = max(max_hidden_error, hidden_error)

    print(f"max_action_error={max_action_error:.8g}")
    print(f"max_hidden_error={max_hidden_error:.8g}")
    if max_action_error > args.atol + args.rtol or max_hidden_error > args.atol + args.rtol:
        raise SystemExit(
            "ONNX parity check failed; inspect runtime/version/export compatibility."
        )
    print("ONNX parity check passed")


if __name__ == "__main__":
    main()

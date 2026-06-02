from __future__ import annotations

import argparse
from pathlib import Path

import torch

from diffphysdrone_px4_wrapper.inference import (
    HIDDEN_DIM,
    DiffPhysExportModel,
    load_torch_model,
)


def export_checkpoint_to_onnx(
    checkpoint_path: Path,
    onnx_path: Path,
    *,
    opset: int = 17,
) -> None:
    model, dim_obs = load_torch_model(checkpoint_path, torch.device("cpu"))
    export_model = DiffPhysExportModel(model).eval()

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    depth = torch.rand((1, 1, 12, 16), dtype=torch.float32)
    state = torch.rand((1, dim_obs), dtype=torch.float32)
    hidden = torch.zeros((1, HIDDEN_DIM), dtype=torch.float32)

    torch.onnx.export(
        export_model,
        (depth, state, hidden),
        str(onnx_path),
        input_names=["depth", "state", "hidden"],
        output_names=["action", "next_hidden"],
        dynamic_axes={
            "depth": {0: "batch"},
            "state": {0: "batch"},
            "hidden": {0: "batch"},
            "action": {0: "batch"},
            "next_hidden": {0: "batch"},
        },
        opset_version=opset,
        # Avoid PyTorch's newer dynamo exporter dependency on onnxscript in the
        # robot container; the legacy exporter handles this static GRUCell graph.
        dynamo=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export DiffPhysDrone checkpoint to ONNX.")
    parser.add_argument(
        "--checkpoint",
        default="/airlab-storage/chiron/models/diffphysdrone/checkpoint0004.pth",
        help="Path to checkpoint .pth file.",
    )
    parser.add_argument(
        "--output",
        default="/airlab-storage/chiron/models/diffphysdrone/checkpoint0004.onnx",
        help="Output ONNX path.",
    )
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    export_checkpoint_to_onnx(Path(args.checkpoint), Path(args.output), opset=args.opset)
    print(f"Exported {args.checkpoint} -> {args.output}")


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path
from typing import Protocol, Tuple

import numpy as np
import torch
from torch import nn

from diffphysdrone_px4_wrapper.diffphys_model import DiffPhysModel


HIDDEN_DIM = 192
ACTION_DIM = 6


class PolicyBackend(Protocol):
    dim_obs: int

    def infer(self, depth: torch.Tensor, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ...


class DiffPhysExportModel(nn.Module):
    """ONNX-friendly wrapper: returns action and next recurrent hidden state."""

    def __init__(self, model: DiffPhysModel) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        depth: torch.Tensor,
        state: torch.Tensor,
        hidden: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        action, _, next_hidden = self.model(depth, state, hidden)
        return action, next_hidden


def checkpoint_dimensions(checkpoint_path: Path, map_location: torch.device | str = "cpu") -> tuple[int, int]:
    state_dict = torch.load(checkpoint_path, map_location=map_location)
    dim_obs = int(state_dict["v_proj.weight"].shape[1])
    dim_action = int(state_dict["fc.weight"].shape[0])
    return dim_obs, dim_action


def load_torch_model(checkpoint_path: Path, device: torch.device) -> tuple[DiffPhysModel, int]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"DiffPhysDrone checkpoint not found: {checkpoint_path}")

    state_dict = torch.load(checkpoint_path, map_location=device)
    dim_obs = int(state_dict["v_proj.weight"].shape[1])
    dim_action = int(state_dict["fc.weight"].shape[0])
    if dim_action != ACTION_DIM:
        raise ValueError(f"Expected a {ACTION_DIM}-action DiffPhysDrone checkpoint, got {dim_action}")
    if dim_obs not in (7, 10):
        raise ValueError(f"Expected a 7-D or 10-D DiffPhysDrone state, got {dim_obs}")

    model = DiffPhysModel(dim_obs=dim_obs, dim_action=dim_action).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, dim_obs


class TorchPolicyBackend:
    def __init__(self, checkpoint_path: Path, device: torch.device) -> None:
        self.model, self.dim_obs = load_torch_model(checkpoint_path, device)
        self.device = device
        self.hidden = torch.zeros((1, HIDDEN_DIM), dtype=torch.float32, device=device)

    def infer(self, depth: torch.Tensor, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        action, _, self.hidden = self.model(depth, state, self.hidden)
        return action, self.hidden


class OnnxPolicyBackend:
    def __init__(self, onnx_path: Path, dim_obs: int, providers: list[str] | None = None) -> None:
        if not onnx_path.exists():
            raise FileNotFoundError(f"DiffPhysDrone ONNX model not found: {onnx_path}")
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "onnxruntime is required for backend='onnx'. Install it in the target runtime "
                "or use backend='torch'."
            ) from exc

        self.session = ort.InferenceSession(str(onnx_path), providers=providers)
        self.dim_obs = int(dim_obs)
        self.hidden = np.zeros((1, HIDDEN_DIM), dtype=np.float32)

    def infer(self, depth: torch.Tensor, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        depth_np = depth.detach().cpu().numpy().astype(np.float32, copy=False)
        state_np = state.detach().cpu().numpy().astype(np.float32, copy=False)
        action_np, hidden_np = self.session.run(
            ["action", "next_hidden"],
            {
                "depth": depth_np,
                "state": state_np,
                "hidden": self.hidden,
            },
        )
        self.hidden = hidden_np.astype(np.float32, copy=False)
        return torch.from_numpy(action_np), torch.from_numpy(self.hidden)

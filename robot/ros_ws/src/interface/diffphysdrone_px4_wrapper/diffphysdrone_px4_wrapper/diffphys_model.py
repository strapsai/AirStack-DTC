from typing import Optional, Tuple

import torch
from torch import nn


class DiffPhysModel(nn.Module):
    """Network architecture from HenryHuYu/DiffPhysDrone/model.py."""

    def __init__(self, dim_obs: int = 10, dim_action: int = 6) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 2, 2, bias=False),
            nn.LeakyReLU(0.05),
            nn.Conv2d(32, 64, 3, bias=False),
            nn.LeakyReLU(0.05),
            nn.Conv2d(64, 128, 3, bias=False),
            nn.LeakyReLU(0.05),
            nn.Flatten(),
            nn.Linear(128 * 2 * 4, 192, bias=False),
        )
        self.v_proj = nn.Linear(dim_obs, 192)
        self.gru = nn.GRUCell(192, 192)
        self.fc = nn.Linear(192, dim_action, bias=False)
        self.act = nn.LeakyReLU(0.05)

    def forward(
        self,
        depth: torch.Tensor,
        state: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, None, torch.Tensor]:
        image_features = self.stem(depth)
        x = self.act(image_features + self.v_proj(state))
        hidden = self.gru(x, hidden)
        action = self.fc(self.act(hidden))
        return action, None, hidden

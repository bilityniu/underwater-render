import torch
import torch.nn.functional as F
from torch import nn


class AppearanceBilateralGrid(nn.Module):
    """Continuous appearance correction via bilateral grid sampling."""

    def __init__(
        self,
        grid_xy=32,
        grid_guide=8,
        gain_strength=0.25,
        bias_strength=0.05,
    ):
        super().__init__()
        self.grid_xy = max(8, int(grid_xy))
        self.grid_guide = max(4, int(grid_guide))
        self.gain_strength = float(gain_strength)
        self.bias_strength = float(bias_strength)
        self.affine_grid = nn.Parameter(
            torch.zeros(1, 6, self.grid_guide, self.grid_xy, self.grid_xy, dtype=torch.float32)
        )

    def forward(self, image_rgb):
        # image_rgb: [3, H, W]
        _, h, w = image_rgb.shape
        device = image_rgb.device
        dtype = image_rgb.dtype

        xs = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
        ys = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
        y_grid, x_grid = torch.meshgrid(ys, xs, indexing="ij")
        luma = (0.299 * image_rgb[0] + 0.587 * image_rgb[1] + 0.114 * image_rgb[2]).clamp(0.0, 1.0)
        guide = luma * 2.0 - 1.0

        sample_grid = torch.stack((x_grid, y_grid, guide), dim=-1).unsqueeze(0).unsqueeze(0)
        affine = F.grid_sample(
            self.affine_grid,
            sample_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        ).squeeze(0).squeeze(1)

        gain_delta = affine[0:3]
        bias_delta = affine[3:6]
        gain = torch.clamp(1.0 + self.gain_strength * torch.tanh(gain_delta), min=0.5, max=1.8)
        bias = self.bias_strength * torch.tanh(bias_delta)
        corrected = torch.clamp(image_rgb * gain + bias, 0.0, 1.0)

        return {
            "appearance_gain": gain,
            "appearance_bias": bias,
            "appearance_rgb": corrected,
        }

import torch
from torch import nn


class CausticField(nn.Module):
    """Slowly varying dynamic caustic illumination field."""

    def __init__(self, lambda_c, f1, f2=14.0, detail_strength=0.15, vx=0.08, vy=0.05, light_dir=(0.2, -0.6, 0.77)):
        super().__init__()
        self.lambda_c = nn.Parameter(torch.tensor(float(lambda_c), dtype=torch.float32))
        self.f1 = nn.Parameter(torch.tensor(float(f1), dtype=torch.float32))
        self.f2 = nn.Parameter(torch.tensor(float(f2), dtype=torch.float32))
        self.detail_strength = nn.Parameter(torch.tensor(float(detail_strength), dtype=torch.float32))
        self.vx = nn.Parameter(torch.tensor(float(vx), dtype=torch.float32))
        self.vy = nn.Parameter(torch.tensor(float(vy), dtype=torch.float32))
        light_dir = torch.tensor(light_dir, dtype=torch.float32)
        light_dir = torch.nn.functional.normalize(light_dir, dim=0, eps=1e-8)
        self.register_buffer("light_dir", light_dir.view(3, 1, 1))
        self.max_pattern = 3.0
        self.focus_kappa = 0.35
        self.domain_warp_strength = 0.85
        self.envelope_strength = 0.35

    def forward(self, surface_points_world, time_value, scene_normal, water_outputs=None, visible_mask=None):
        x_world = surface_points_world[0]
        y_world = surface_points_world[1]
        z_world = surface_points_world[2]
        t = self._scalar_time(x_world, time_value)

        warped_x = x_world
        warped_y = y_world

        light_response = torch.clamp((scene_normal * self.light_dir).sum(dim=0, keepdim=True), min=0.0, max=1.0)
        surface_focus = torch.ones_like(light_response)
        focus_gain = torch.ones_like(light_response)

        if water_outputs is not None:
            water_normal = water_outputs.get("surface_normal")
            water_gradient = water_outputs.get("surface_gradient")
            gradient_magnitude = water_outputs.get("surface_gradient_magnitude")
            curvature = water_outputs.get("surface_curvature")
            if water_normal is not None:
                surface_focus = torch.clamp(water_normal[2:3], min=0.0, max=1.0)
            if water_gradient is not None:
                warped_x = warped_x + self.domain_warp_strength * water_gradient[0]
                warped_y = warped_y + self.domain_warp_strength * water_gradient[1]
            if gradient_magnitude is not None or curvature is not None:
                focus_driver = 0.0
                if gradient_magnitude is not None:
                    focus_driver = focus_driver + gradient_magnitude
                if curvature is not None:
                    focus_driver = focus_driver + curvature
                focus_gain = 1.0 + self.focus_kappa * self.normalize_pattern(focus_driver.squeeze(0), visible_mask).unsqueeze(0)

        base_1 = torch.sin((warped_x + self.vx * t) * self.f1) * torch.sin((warped_y + self.vy * t) * (self.f1 * 1.07))
        base_2 = torch.sin((warped_x * 1.41 - self.vx * 0.67 * t) * (self.f1 * 0.63)) * torch.sin((warped_y * 0.79 + self.vy * 1.21 * t) * (self.f1 * 0.71))
        detail = torch.sin((warped_x * 1.73 - self.vx * 0.73 * t) * self.f2) * torch.sin((warped_y * 0.91 + self.vy * 1.19 * t) * (self.f2 * 0.87))
        envelope = 0.5 + self.envelope_strength * torch.sin((warped_x * 0.37 + warped_y * 0.19) + 0.23 * t)
        broken_pattern = torch.relu(base_1 + 0.7 * base_2)
        pattern_raw = torch.relu(envelope * broken_pattern + self.detail_strength * detail)
        pattern = self.normalize_pattern(pattern_raw, visible_mask)

        caustic = pattern.unsqueeze(0) * light_response * surface_focus * focus_gain
        if visible_mask is not None:
            caustic = caustic * visible_mask

        gain = 1.0 + torch.clamp(self.lambda_c, min=0.0) * caustic
        return {
            "caustic_pattern": pattern.unsqueeze(0),
            "caustic_pattern_raw": pattern_raw.unsqueeze(0),
            "warped_world_x": warped_x.unsqueeze(0),
            "warped_world_y": warped_y.unsqueeze(0),
            "world_x": x_world.unsqueeze(0),
            "world_y": y_world.unsqueeze(0),
            "world_z": z_world.unsqueeze(0),
            "caustic": caustic,
            "caustic_gain": gain,
            "light_response": light_response,
            "surface_focus": surface_focus,
            "focus_gain": focus_gain,
        }

    def normalize_pattern(self, pattern, visible_mask=None):
        if pattern.dim() == 3 and pattern.shape[0] == 1:
            pattern = pattern.squeeze(0)
        if visible_mask is not None:
            valid = visible_mask.squeeze(0) > 0.5
        else:
            valid = torch.ones_like(pattern, dtype=torch.bool)
        if valid.any():
            mean_value = pattern[valid].mean()
        else:
            mean_value = pattern.new_tensor(1.0)
        pattern = pattern / (mean_value + 1e-6)
        pattern = torch.clamp(pattern, min=0.0, max=self.max_pattern)
        return pattern

    @staticmethod
    def _scalar_time(reference, time_value):
        if torch.is_tensor(time_value):
            return time_value.to(device=reference.device, dtype=reference.dtype)
        return torch.tensor(float(time_value), device=reference.device, dtype=reference.dtype)

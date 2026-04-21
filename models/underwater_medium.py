import math

import torch
import torch.nn.functional as F
from torch import nn


_SH_Y00 = 0.28209479177387814
_SH_C1 = 0.4886025119029199
_SH_C2 = (
    1.0925484305920792,
    1.0925484305920792,
    0.31539156525252005,
    1.0925484305920792,
    0.5462742152960396,
)


def jerlov_rgb_preset(jerlov_type):
    """
    Return approximate RGB attenuation priors (1/m) for open-ocean Jerlov classes.
    Values are pragmatic RGB fits used as bounded priors, not full spectral curves.
    """
    key = str(jerlov_type).strip().upper()
    presets = {
        "I": {
            "beta_d": [0.14, 0.065, 0.028],
            "beta_b": [0.11, 0.050, 0.020],
            "b_inf": [0.04, 0.22, 0.42],
        },
        "IA": {
            "beta_d": [0.18, 0.085, 0.036],
            "beta_b": [0.14, 0.065, 0.026],
            "b_inf": [0.05, 0.24, 0.44],
        },
        "IB": {
            "beta_d": [0.24, 0.11, 0.05],
            "beta_b": [0.18, 0.085, 0.035],
            "b_inf": [0.06, 0.27, 0.47],
        },
        "II": {
            "beta_d": [0.34, 0.16, 0.075],
            "beta_b": [0.26, 0.12, 0.055],
            "b_inf": [0.08, 0.30, 0.50],
        },
        "III": {
            "beta_d": [0.48, 0.24, 0.12],
            "beta_b": [0.36, 0.18, 0.085],
            "b_inf": [0.10, 0.34, 0.54],
        },
        "1C": {
            "beta_d": [0.26, 0.16, 0.18],
            "beta_b": [0.20, 0.14, 0.16],
            "b_inf": [0.12, 0.30, 0.18],
        },
        "3C": {
            "beta_d": [0.38, 0.23, 0.26],
            "beta_b": [0.28, 0.18, 0.20],
            "b_inf": [0.15, 0.33, 0.17],
        },
        "5C": {
            "beta_d": [0.54, 0.32, 0.36],
            "beta_b": [0.40, 0.25, 0.28],
            "b_inf": [0.18, 0.37, 0.16],
        },
        "7C": {
            "beta_d": [0.74, 0.44, 0.48],
            "beta_b": [0.56, 0.34, 0.38],
            "b_inf": [0.24, 0.42, 0.15],
        },
        "9C": {
            "beta_d": [0.96, 0.58, 0.62],
            "beta_b": [0.74, 0.46, 0.50],
            "b_inf": [0.28, 0.46, 0.14],
        },
    }
    return presets.get(key, presets["I"])


def num_sh_basis(degree):
    return (int(degree) + 1) ** 2


def eval_sh_basis(degree, directions):
    dirs = F.normalize(directions, dim=0, eps=1e-8)
    x = dirs[0]
    y = dirs[1]
    z = dirs[2]
    basis = [torch.ones_like(x) * _SH_Y00]
    if degree >= 1:
        basis.extend(
            [
                -_SH_C1 * y,
                _SH_C1 * z,
                -_SH_C1 * x,
            ]
        )
    if degree >= 2:
        c0, c1, c2, c3, c4 = _SH_C2
        basis.extend(
            [
                c0 * x * y,
                c1 * y * z,
                c2 * (3.0 * z * z - 1.0),
                c3 * x * z,
                c4 * (x * x - y * y),
            ]
        )
    return torch.stack(basis, dim=0)


def inverse_softplus(value):
    x = torch.as_tensor(value, dtype=torch.float32)
    return torch.log(torch.expm1(torch.clamp(x, min=1e-6)))


def inverse_sigmoid(value):
    x = torch.as_tensor(value, dtype=torch.float32).clamp(1e-5, 1.0 - 1e-5)
    return torch.log(x / (1.0 - x))


class UnderwaterMedium(nn.Module):
    """
    Plenodium-inspired underwater medium with spatial-directional explicit fields
    and a light physics-guided advection prior.
    """

    def __init__(
        self,
        beta,
        b_inf,
        spatial_scale=0.08,
        num_basis=4,
        beta_strength=1.0,
        binf_strength=1.0,
        depth_scale=2.0,
        beta_backscatter=None,
        jerlov_type="I",
        use_jerlov_prior=True,
        beta_min=0.01,
        beta_max=1.5,
        grid_resolution=24,
        scene_bounds=None,
        sh_degree=1,
        enable_motion=False,
        motion_grid_resolution=12,
        motion_scale=0.0015,
        directional_strength=1.0,
    ):
        super().__init__()
        if use_jerlov_prior:
            preset = jerlov_rgb_preset(jerlov_type)
            beta_direct_base = preset["beta_d"] if beta is None else beta
            beta_back_base = preset["beta_b"] if beta_backscatter is None else beta_backscatter
            b_inf_base = preset["b_inf"] if b_inf is None else b_inf
        else:
            beta_direct_base = beta
            beta_back_base = beta if beta_backscatter is None else beta_backscatter
            b_inf_base = b_inf

        beta_direct_base = torch.tensor(beta_direct_base, dtype=torch.float32).view(3, 1, 1)
        beta_back_base = torch.tensor(beta_back_base, dtype=torch.float32).view(3, 1, 1)
        b_inf_base = torch.tensor(b_inf_base, dtype=torch.float32).view(3, 1, 1)

        self.register_buffer("beta_direct_base", beta_direct_base)
        self.register_buffer("beta_back_base", beta_back_base)
        self.register_buffer("b_inf_base", b_inf_base)
        self.num_basis = int(num_basis)
        self.spatial_scale = float(spatial_scale)
        self.beta_strength = float(beta_strength)
        self.binf_strength = float(binf_strength)
        self.depth_scale = float(depth_scale)
        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)
        self.grid_resolution = max(8, int(grid_resolution))
        self.sh_degree = int(sh_degree)
        self.sh_basis_count = num_sh_basis(self.sh_degree)
        self.enable_motion = bool(enable_motion)
        self.motion_grid_resolution = max(4, int(motion_grid_resolution))
        self.motion_scale = float(motion_scale)
        self.directional_strength = float(directional_strength)

        scene_min, scene_max = self._resolve_scene_bounds(scene_bounds)
        self.register_buffer("scene_bounds_min", scene_min)
        self.register_buffer("scene_bounds_max", scene_max)
        self.register_buffer("scene_extent", torch.clamp(scene_max - scene_min, min=1e-4))

        self.beta_direct_grid = nn.Parameter(
            torch.zeros(1, 3, self.grid_resolution, self.grid_resolution, self.grid_resolution, dtype=torch.float32)
        )
        self.beta_back_sh_grid = nn.Parameter(
            torch.zeros(
                1,
                3 * self.sh_basis_count,
                self.grid_resolution,
                self.grid_resolution,
                self.grid_resolution,
                dtype=torch.float32,
            )
        )
        self.binf_sh_grid = nn.Parameter(
            torch.zeros(
                1,
                3 * self.sh_basis_count,
                self.grid_resolution,
                self.grid_resolution,
                self.grid_resolution,
                dtype=torch.float32,
            )
        )
        self.motion_flow_grid = nn.Parameter(
            torch.zeros(
                1,
                3,
                self.motion_grid_resolution,
                self.motion_grid_resolution,
                self.motion_grid_resolution,
                dtype=torch.float32,
            ),
            requires_grad=self.enable_motion,
        )

    def forward(
        self,
        scene_rgb,
        scene_depth,
        surface_points_world=None,
        valid_mask=None,
        time_controls=None,
        flow_controls=None,
    ):
        depth = torch.clamp(scene_depth, min=0.0)
        effective_depth = depth * max(self.depth_scale, 1e-4)
        medium_fields = self.query_medium_fields(
            scene_rgb,
            surface_points_world=surface_points_world,
            valid_mask=valid_mask,
            time_controls=time_controls,
            direction_world=self._resolve_view_direction(flow_controls),
        )
        beta_direct_map = medium_fields["beta_d_map"]
        beta_back_map = medium_fields["beta_b_map"]
        b_inf_map = medium_fields["b_inf_map"]
        advected_points_world = medium_fields["advected_surface_points_world"]
        flow_velocity_world = medium_fields["medium_flow_velocity"]
        motion_magnitude = medium_fields["medium_motion_magnitude"]
        directionality = medium_fields["medium_directionality"]

        transmission_direct = torch.exp(-beta_direct_map * effective_depth)
        transmission_back = torch.exp(-beta_back_map * effective_depth)
        backscatter = b_inf_map * (1.0 - transmission_back)
        direct_underwater_rgb = scene_rgb * transmission_direct
        underwater_rgb = direct_underwater_rgb + backscatter

        outputs = {
            "effective_depth": effective_depth,
            "beta_d_map": beta_direct_map,
            "beta_b_map": beta_back_map,
            "beta_map": beta_direct_map,
            "b_inf_map": b_inf_map,
            "backscatter": backscatter,
            "direct_underwater_rgb": direct_underwater_rgb,
            "transmission_direct": transmission_direct,
            "transmission_back": transmission_back,
            "transmission": transmission_direct,
            "underwater_rgb": underwater_rgb,
            "medium_flow_velocity": flow_velocity_world,
            "medium_flow_x": flow_velocity_world[0:1],
            "medium_flow_y": flow_velocity_world[1:2],
            "medium_flow_z": flow_velocity_world[2:3],
            "medium_motion_magnitude": motion_magnitude,
            "medium_directionality": directionality,
        }
        if advected_points_world is not None:
            outputs.update(
                {
                    "advected_surface_points_world": advected_points_world,
                    "advected_world_x": advected_points_world[0:1],
                    "advected_world_y": advected_points_world[1:2],
                    "advected_world_z": advected_points_world[2:3],
                }
            )
        return outputs

    def query_medium_fields(
        self,
        scene_rgb_like,
        surface_points_world=None,
        valid_mask=None,
        time_controls=None,
        direction_world=None,
    ):
        beta_direct_map = self.beta_direct_base.expand_as(scene_rgb_like)
        beta_back_map = self.beta_back_base.expand_as(scene_rgb_like)
        b_inf_map = self.b_inf_base.expand_as(scene_rgb_like)

        advected_points_world = None
        flow_velocity_world = torch.zeros(
            (3, scene_rgb_like.shape[-2], scene_rgb_like.shape[-1]),
            device=scene_rgb_like.device,
            dtype=scene_rgb_like.dtype,
        )
        motion_magnitude = torch.zeros(
            (1, scene_rgb_like.shape[-2], scene_rgb_like.shape[-1]),
            device=scene_rgb_like.device,
            dtype=scene_rgb_like.dtype,
        )
        directionality = torch.zeros(
            (1, scene_rgb_like.shape[-2], scene_rgb_like.shape[-1]),
            device=scene_rgb_like.device,
            dtype=scene_rgb_like.dtype,
        )

        if surface_points_world is not None:
            time_value = self._resolve_time_value(time_controls)
            if self.enable_motion:
                advected_points_world, flow_velocity_world, motion_magnitude, advected_coords = self.advect_points(
                    surface_points_world,
                    time_value,
                )
            else:
                advected_points_world = surface_points_world
                advected_coords = self.world_to_unit_coords(surface_points_world)
            beta_direct_scale = self.sample_log_residual_field(self.beta_direct_grid, advected_coords, self.beta_strength)
            # Backscatter coefficient and asymptotic radiance are physically
            # isotropic: beta_b is an angle-integrated scattering coefficient,
            # and b_inf is the equilibrium of multiply-scattered light. The
            # angular structure of single-scatter is already represented by
            # the HG phase function in the beam channel; duplicating it as a
            # learnable SH field here creates a view-dependent haze tint that
            # manifests as a "shadow that drifts with the camera" and whose
            # visibility grows with distance (since backscatter saturates
            # toward b_inf at depth). We evaluate both fields at the DC
            # (isotropic) component only, so per-point backscatter is a
            # scalar-valued function of position, not of view direction.
            beta_back_scale, directionality = self.sample_directional_log_residual_field(
                self.beta_back_sh_grid,
                advected_coords,
                None,
                self.beta_strength,
            )
            b_inf_spatial, _ = self.sample_directional_binf_field(
                self.binf_sh_grid,
                advected_coords,
                None,
                self.binf_strength,
            )

            beta_direct_spatial = torch.clamp(
                self.beta_direct_base * beta_direct_scale,
                min=self.beta_min,
                max=self.beta_max,
            )
            beta_back_spatial = torch.clamp(
                self.beta_back_base * beta_back_scale,
                min=self.beta_min,
                max=self.beta_max,
            )
            if valid_mask is None:
                beta_direct_map = beta_direct_spatial
                beta_back_map = beta_back_spatial
                b_inf_map = b_inf_spatial
            else:
                valid_rgb = valid_mask.expand_as(scene_rgb_like) > 0.5
                beta_direct_map = torch.where(valid_rgb, beta_direct_spatial, beta_direct_map)
                beta_back_map = torch.where(valid_rgb, beta_back_spatial, beta_back_map)
                b_inf_map = torch.where(valid_rgb, b_inf_spatial, b_inf_map)

        return {
            "beta_d_map": beta_direct_map,
            "beta_b_map": beta_back_map,
            "b_inf_map": b_inf_map,
            "advected_surface_points_world": advected_points_world,
            "medium_flow_velocity": flow_velocity_world,
            "medium_motion_magnitude": motion_magnitude,
            "medium_directionality": directionality,
        }

    def compute_downwelling_transport(
        self,
        scene_rgb_like,
        surface_points_world,
        down_dir_world,
        down_distance,
        valid_mask=None,
        time_controls=None,
    ):
        fields = self.query_medium_fields(
            scene_rgb_like,
            surface_points_world=surface_points_world,
            valid_mask=valid_mask,
            time_controls=time_controls,
            direction_world=down_dir_world,
        )
        effective_distance = torch.clamp(down_distance, min=0.0) * max(self.depth_scale, 1e-4)
        transmission_down_rgb = torch.exp(-fields["beta_d_map"] * effective_distance)
        transmission_down = transmission_down_rgb.mean(dim=0, keepdim=True)
        scatter_density = fields["beta_b_map"].mean(dim=0, keepdim=True)
        if valid_mask is not None and (valid_mask > 0.5).any():
            scatter_mean = torch.clamp(scatter_density[valid_mask > 0.5].mean(), min=1e-4)
        else:
            scatter_mean = torch.clamp(scatter_density.mean(), min=1e-4)
        scatter_norm = torch.clamp(scatter_density / scatter_mean, min=0.5, max=2.0)
        blur_strength = torch.clamp(
            0.08 * scatter_density * effective_distance + 0.03 * fields["medium_directionality"],
            min=0.0,
            max=0.45,
        )
        beam_visibility = torch.clamp(
            (1.0 - transmission_down) * torch.sqrt(torch.clamp(transmission_down, min=1e-4)) * (0.40 + 0.60 * scatter_norm),
            min=0.0,
            max=1.0,
        )
        return {
            "down_transmission_rgb": transmission_down_rgb,
            "down_transmission": transmission_down,
            "down_scatter_density": scatter_density,
            "down_blur_strength": blur_strength,
            "beam_visibility": beam_visibility,
            "down_beta_d_map": fields["beta_d_map"],
            "down_beta_b_map": fields["beta_b_map"],
            "down_b_inf_map": fields["b_inf_map"],
        }

    @torch.no_grad()
    def init_from_stokes_drift(self, surface_module, up_axis=2, scale=1.0):
        """
        Initialise motion_flow_grid from the Stokes drift of a WaterSurface model.

        Stokes drift decays exponentially with depth (exp(−2kd)), so deeper grid
        cells receive proportionally weaker horizontal flow.  This couples
        surface-wave kinematics to the medium's advection field without adding
        any extra learnable parameters.

        The method writes the inverse-tanh of the normalised Stokes drift into
        motion_flow_grid so that tanh(grid) ≈ v_stokes / v_max at every cell.

        Args:
            surface_module: WaterSurface instance with compute_stokes_drift().
            up_axis: index of the vertical axis (0=x,1=y,2=z).
            scale: overall strength multiplier (default 1.0).
        """
        if not hasattr(surface_module, "compute_stokes_drift"):
            return

        res = self.motion_grid_resolution
        device = self.motion_flow_grid.device
        dtype = self.motion_flow_grid.dtype

        # Build a uniform depth grid over the scene extent
        scene_min = self.scene_bounds_min.to(device=device, dtype=dtype)
        scene_max = self.scene_bounds_max.to(device=device, dtype=dtype)
        water_top = scene_max[up_axis]

        # Depth at each level of the grid (0 = top / water surface)
        levels = torch.linspace(0.0, 1.0, res, device=device, dtype=dtype)  # [res]
        depth_vals = levels * (scene_max[up_axis] - scene_min[up_axis])     # [res]

        # Compute Stokes drift for all depth levels at once
        lateral_axes = [i for i in range(3) if i != up_axis]
        new_grid = torch.zeros_like(self.motion_flow_grid)  # [1, 3, res, res, res]

        for z_idx, d in enumerate(depth_vals):
            drift_2d = surface_module.compute_stokes_drift(d)   # [2, 1, 1] (scalar depth)
            drift_u = drift_2d[0].mean().item()
            drift_v = drift_2d[1].mean().item()

            v_max = max(abs(drift_u), abs(drift_v), 1e-6)
            norm_u = float(scale) * drift_u / v_max
            norm_v = float(scale) * drift_v / v_max

            # Inverse tanh so that tanh(grid) ≈ normalised drift
            raw_u = math.atanh(max(-0.999, min(0.999, norm_u)))
            raw_v = math.atanh(max(-0.999, min(0.999, norm_v)))

            new_grid[0, lateral_axes[0], z_idx, :, :] = raw_u
            new_grid[0, lateral_axes[1], z_idx, :, :] = raw_v
            # Vertical component is zero for surface-gravity-wave Stokes drift

        self.motion_flow_grid.data.copy_(new_grid)

    def advect_points(self, surface_points_world, time_value):
        coords01 = self.world_to_unit_coords(surface_points_world)
        flow_velocity = torch.tanh(self.sample_grid(self.motion_flow_grid, coords01))
        flow_displacement = self.motion_scale * flow_velocity
        advected_coords = torch.remainder(coords01 - float(time_value) * flow_displacement, 1.0)
        advected_world = self.unit_coords_to_world(advected_coords)
        flow_velocity_world = flow_velocity * self.scene_extent.view(3, 1, 1)
        motion_magnitude = torch.linalg.norm(flow_displacement * self.scene_extent.view(3, 1, 1), dim=0, keepdim=True)
        return advected_world, flow_velocity_world, motion_magnitude, advected_coords

    def sample_log_residual_field(self, grid, coords01, strength):
        raw = self.sample_grid(grid, coords01)
        return torch.exp(float(strength) * torch.tanh(raw))

    def sample_directional_log_residual_field(self, grid, coords01, view_dir_world, strength):
        coeff = self.sample_grid(grid, coords01)
        coeff = coeff.view(3, self.sh_basis_count, coords01.shape[-2], coords01.shape[-1])
        if view_dir_world is None:
            basis = coeff.new_zeros((self.sh_basis_count, coords01.shape[-2], coords01.shape[-1]))
            basis[0] = _SH_Y00
        else:
            basis = eval_sh_basis(self.sh_degree, view_dir_world)
        dc_raw = coeff[:, 0] * basis[0] / _SH_Y00
        directional_raw = (
            torch.sum(coeff[:, 1:] * basis[1:].unsqueeze(0), dim=1) if self.sh_basis_count > 1 else dc_raw.new_zeros(dc_raw.shape)
        )
        effective_directional_strength = 0.35 * self.directional_strength
        raw = dc_raw + effective_directional_strength * directional_raw
        values = torch.exp(float(strength) * torch.tanh(raw))
        directionality = (effective_directional_strength * directional_raw).abs().mean(dim=0, keepdim=True)
        return values, directionality

    def sample_directional_binf_field(self, grid, coords01, view_dir_world, strength):
        coeff = self.sample_grid(grid, coords01)
        coeff = coeff.view(3, self.sh_basis_count, coords01.shape[-2], coords01.shape[-1])
        if view_dir_world is None:
            basis = coeff.new_zeros((self.sh_basis_count, coords01.shape[-2], coords01.shape[-1]))
            basis[0] = _SH_Y00
        else:
            basis = eval_sh_basis(self.sh_degree, view_dir_world)
        dc_raw = coeff[:, 0] * basis[0] / _SH_Y00
        directional_raw = (
            torch.sum(coeff[:, 1:] * basis[1:].unsqueeze(0), dim=1) if self.sh_basis_count > 1 else dc_raw.new_zeros(dc_raw.shape)
        )
        effective_directional_strength = 0.30 * self.directional_strength
        raw = dc_raw + effective_directional_strength * directional_raw
        base_logits = torch.logit(self.b_inf_base.expand(raw.shape[0], raw.shape[1], raw.shape[2]).clamp(1e-5, 1.0 - 1e-5))
        values = torch.sigmoid(base_logits + float(strength) * torch.tanh(raw))
        directionality = (effective_directional_strength * directional_raw).abs().mean(dim=0, keepdim=True)
        return values, directionality

    def sample_grid(self, grid, coords01):
        grid_coords = self.unit_coords_to_grid(coords01)
        x = torch.clamp(grid_coords[0], -1.0, 1.0)
        y = torch.clamp(grid_coords[1], -1.0, 1.0)
        z = torch.clamp(grid_coords[2], -1.0, 1.0)
        sample_grid = torch.stack((x, y, z), dim=-1).unsqueeze(0).unsqueeze(0)
        return F.grid_sample(
            grid,
            sample_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        ).squeeze(0).squeeze(1)

    def world_to_unit_coords(self, surface_points_world):
        return torch.clamp(
            (surface_points_world - self.scene_bounds_min.view(3, 1, 1)) / self.scene_extent.view(3, 1, 1),
            min=0.0,
            max=1.0,
        )

    def unit_coords_to_world(self, coords01):
        return self.scene_bounds_min.view(3, 1, 1) + coords01 * self.scene_extent.view(3, 1, 1)

    @staticmethod
    def unit_coords_to_grid(coords01):
        return 2.0 * coords01 - 1.0

    @staticmethod
    def _resolve_time_value(time_controls):
        if time_controls is None:
            return 0.0
        if isinstance(time_controls, dict):
            return float(time_controls.get("time_value", 0.0))
        if torch.is_tensor(time_controls):
            return float(time_controls.detach().item())
        return float(time_controls)

    @staticmethod
    def _resolve_view_direction(flow_controls):
        if flow_controls is None:
            return None
        if isinstance(flow_controls, dict):
            view_dir = flow_controls.get("view_dir_world")
            if view_dir is None:
                view_dir = -flow_controls.get("pixel_ray_dir_world") if flow_controls.get("pixel_ray_dir_world") is not None else None
            return view_dir
        return None

    @staticmethod
    def _resolve_scene_bounds(scene_bounds):
        if scene_bounds is None:
            scene_min = torch.tensor([-1.0, -1.0, -1.0], dtype=torch.float32)
            scene_max = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
            return scene_min, scene_max
        scene_min = scene_bounds["min"].detach().float().view(3)
        scene_max = scene_bounds["max"].detach().float().view(3)
        return scene_min, torch.maximum(scene_max, scene_min + 1e-4)

    @staticmethod
    def _resize_init(values, size):
        repeats = math.ceil(size / values.numel())
        return values.repeat(repeats)[:size]

    @staticmethod
    def _resize_matrix(values, size):
        repeats = math.ceil(size / values.shape[1])
        return values.repeat(1, repeats)[:, :size]

    @staticmethod
    def _resize_angles(size):
        return torch.arange(size, dtype=torch.float32) * (math.pi / max(1, size))

    @staticmethod
    def _resize_phase(size):
        return torch.arange(size, dtype=torch.float32) * (2.0 * math.pi / max(1, size))

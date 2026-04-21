import torch
from torch import nn


class WaterLightField(nn.Module):
    """
    Unified water-light field.

    The field is queried at world points x and time t and returns:
    - sigma_t_view: extinction coefficients toward the camera
    - sigma_t_down: extinction coefficients along the downwelling path
    - sigma_s: scattering coefficients toward the camera
    - b_inf: ambient / asymptotic backscatter color
    - L_surface: downwelling radiance on the water-driven source plane
    - L_in: downwelling radiance arriving at x after travelling downward through water
    - depth_down: depth from the local water surface to x
    """

    def __init__(self, medium_module, caustic_module, surface_module=None):
        super().__init__()
        self.medium = medium_module
        self.caustic = caustic_module
        self.surface = surface_module

    def query(
        self,
        x_world,
        t,
        view_dir_world=None,
        down_dir_world=None,
        valid_mask=None,
        caustic_outputs=None,
    ):
        reference = x_world.new_zeros(3, x_world.shape[-2], x_world.shape[-1])
        if self.medium is None or self.caustic is None:
            zeros_rgb = reference
            zeros_scalar = reference[:1]
            return {
                "sigma_t_view": zeros_rgb,
                "sigma_t_down": zeros_rgb,
                "sigma_s": zeros_rgb,
                "b_inf": zeros_rgb,
                "L_surface": zeros_rgb,
                "L_in": zeros_rgb,
                "depth_down": zeros_scalar,
                "downwelling_scalar": zeros_scalar,
            }

        medium_view = self.medium.query_medium_fields(
            reference,
            surface_points_world=x_world,
            valid_mask=valid_mask,
            time_controls={"time_value": t},
            direction_world=view_dir_world,
        )
        if down_dir_world is None:
            down_dir_world = view_dir_world
        medium_down = self.medium.query_medium_fields(
            reference,
            surface_points_world=x_world,
            valid_mask=valid_mask,
            time_controls={"time_value": t},
            direction_world=down_dir_world,
        )
        sigma_t_view = medium_view["beta_d_map"]
        sigma_s = medium_view["beta_b_map"]
        b_inf = medium_view["b_inf_map"]
        sigma_t_down = medium_down["beta_d_map"]

        world_u = x_world[self.caustic.lateral_axes[0]]
        world_v = x_world[self.caustic.lateral_axes[1]]
        down_scalar, _ = self.caustic.query_downwelling(
            world_u,
            world_v,
            t,
            valid_mask=valid_mask,
            surface_sampler=self.surface,
            precomputed=caustic_outputs,
        )
        water_height = self.caustic.query_water_height(
            world_u,
            world_v,
            t,
            surface_sampler=self.surface,
            precomputed=caustic_outputs,
        )
        depth_down = torch.clamp(
            water_height - x_world[self.caustic.up_axis : self.caustic.up_axis + 1],
            min=0.0,
        )

        # Use the medium ambient color as the water-light tint.
        light_color = torch.clamp(0.72 * b_inf + 0.28, 0.0, 1.0)
        L_surface = light_color * down_scalar
        L_in = L_surface * torch.exp(-sigma_t_down * depth_down)

        return {
            "sigma_t_view": sigma_t_view,
            "sigma_t_down": sigma_t_down,
            "sigma_s": sigma_s,
            "b_inf": b_inf,
            "L_surface": L_surface,
            "L_in": L_in,
            "depth_down": depth_down,
            "downwelling_scalar": down_scalar,
        }

import torch
import torch.nn.functional as F
from torch import nn


class RefractionCausticField(nn.Module):
    """Continuous base-field caustics with receiver-footprint detail modulation."""

    AXIS_TO_INDEX = {"x": 0, "y": 1, "z": 2}

    def __init__(
        self,
        lambda_c,
        light_dir=(0.2, -0.6, 0.77),
        eta=1.0 / 1.33,
        plane_res=96,
        plane_padding=0.15,
        water_level_offset=4.0,
        water_level_mode="scene_relative",
        water_level_world=0.0,
        max_pattern=3.0,
        min_lambda=0.05,
        receiver_query_mix=0.32,
        beam_strength=0.12,
        beam_depth_decay=0.35,
        beam_view_power=2.0,
        up_axis="z",
        scene_bounds=None,
    ):
        super().__init__()
        self.lambda_c = nn.Parameter(torch.tensor(float(lambda_c), dtype=torch.float32))
        self.register_buffer(
            "lambda_reference",
            torch.tensor(max(float(lambda_c), float(min_lambda)), dtype=torch.float32),
        )
        light_dir = torch.tensor(light_dir, dtype=torch.float32)
        light_dir = F.normalize(light_dir, dim=0, eps=1e-8)
        self.register_buffer("light_dir", light_dir.view(3))
        self.eta = float(eta)
        self.plane_res = int(plane_res)
        self.plane_padding = float(plane_padding)
        self.water_level_offset = float(water_level_offset)
        self.water_level_mode = str(water_level_mode).lower()
        self.water_level_world = float(water_level_world)
        self.max_pattern = float(max_pattern)
        self.min_lambda = float(min_lambda)
        self.receiver_query_mix = float(receiver_query_mix)
        self.beam_strength = float(beam_strength)
        self.beam_depth_decay = float(beam_depth_decay)
        self.beam_view_power = float(beam_view_power)
        scene_min, scene_max = self._resolve_scene_bounds(scene_bounds)
        self.register_buffer("scene_bounds_min", scene_min)
        self.register_buffer("scene_bounds_max", scene_max)
        self.up_axis = self._resolve_up_axis(up_axis)
        self.lateral_axes = tuple(index for index in range(3) if index != self.up_axis)

    def forward(self, surface_points_world, time_value, scene_normal, water_outputs=None, visible_mask=None, surface_sampler=None):
        device = surface_points_world.device
        dtype = surface_points_world.dtype
        world_u = surface_points_world[self.lateral_axes[0]]
        world_v = surface_points_world[self.lateral_axes[1]]
        world_up = surface_points_world[self.up_axis]
        valid_mask = self._to_valid_mask(visible_mask, world_u)

        bounds = self.compute_world_bounds(device, dtype)
        water_level = self.resolve_water_level(bounds, device, dtype)
        local_refracted_dir = self.compute_local_refracted_dir(
            water_outputs,
            surface_sampler,
            world_u,
            world_v,
            time_value,
            device,
            dtype,
        )
        refraction_spread = self.compute_refraction_spread(local_refracted_dir, valid_mask, water_outputs=water_outputs)
        base_plane_map, base_focus_weight = self.compute_base_plane_caustic(
            bounds,
            water_level,
            time_value,
            surface_sampler,
            device,
            dtype,
        )
        downwelling_base_map = self.build_downwelling_base_map(base_plane_map)
        base_pattern = self.query_plane_map(downwelling_base_map, world_u, world_v, bounds, valid_mask)
        base_pattern = self.normalize_pattern(base_pattern, valid_mask)
        # Caustic focus is already captured by the photon density on the plane
        # grid (compute_focus_weights). A second Jacobian in pixel space over
        # receiver footprints was unreliable on 3DGS geometry: depth and normal
        # discontinuities make spatial_derivative explode, producing spurious
        # bright/dark rims at object silhouettes and unstable global mean
        # normalisation. We drop that path entirely. project_to_receiver_surface
        # is retained only to produce receiver_support, a soft gate telling the
        # renderer which pixels actually receive refracted light.
        receiver_state = self.project_to_receiver_surface(
            surface_points_world,
            scene_normal,
            world_u,
            world_v,
            water_level,
            local_refracted_dir,
            valid_mask,
            water_outputs=water_outputs,
        )
        sampled_caustic = base_pattern
        detail_modulation = torch.ones_like(base_pattern)
        receiver_query_pattern = base_pattern
        receiver_focus_weight = torch.zeros_like(base_pattern)
        receiver_blur_weight = torch.zeros_like(base_pattern)
        detail_pattern = torch.zeros_like(base_pattern)

        light_response = torch.clamp(
            (scene_normal * self.light_dir.view(3, 1, 1)).sum(dim=0, keepdim=True),
            min=0.0,
            max=1.0,
        )
        surface_focus = torch.ones_like(light_response)
        focus_gain = torch.ones_like(light_response)
        if water_outputs is not None:
            water_normal = water_outputs.get("surface_normal")
            gradient_magnitude = water_outputs.get("surface_gradient_magnitude")
            curvature = water_outputs.get("surface_curvature")
            if water_normal is not None:
                surface_focus = torch.clamp(water_normal[2:3], min=0.0, max=1.0)
            if gradient_magnitude is not None or curvature is not None:
                focus_driver = 0.0
                if gradient_magnitude is not None:
                    focus_driver = focus_driver + gradient_magnitude
                if curvature is not None:
                    focus_driver = focus_driver + curvature
                focus_driver = self.blur_plane_map(focus_driver).clamp_min(0.0)
                # Stronger focus lobe from the water slope/curvature, applied as
                # a surface-attached detail layer over the low-pass caustic.
                focus_gain = 1.0 + 0.35 * self.normalize_pattern(focus_driver.squeeze(0), valid_mask).unsqueeze(0)
        attached_base_pattern = base_pattern * focus_gain.squeeze(0)

        receiver_support = (0.80 + 0.20 * receiver_state["receiver_alignment"]).unsqueeze(0)
        final_caustic = sampled_caustic.unsqueeze(0) * light_response * surface_focus * focus_gain * receiver_support
        final_caustic = final_caustic * valid_mask.float()
        # Physical lambda: geometric-optics prior from wave slope variance +
        # learnable residual. Caustic focusing gain ∝ σ²_slope (small-angle approx).
        lambda_phys = self.compute_physical_lambda(water_outputs, valid_mask)
        effective_lambda = torch.clamp(lambda_phys + self.lambda_c, min=self.min_lambda)
        caustic_gain = 1.0 + effective_lambda * final_caustic

        return {
            "caustic_plane_map": base_plane_map,
            "downwelling_base_map": downwelling_base_map,
            "caustic_pattern": sampled_caustic.unsqueeze(0),
            "sampled_caustic": sampled_caustic.unsqueeze(0),
            "caustic": final_caustic,
            "caustic_gain": caustic_gain,
            "base_caustic_pattern": base_pattern.unsqueeze(0),
            "receiver_query_pattern": receiver_query_pattern.unsqueeze(0),
            "attached_base_pattern": attached_base_pattern.unsqueeze(0),
            "surface_attached_pattern": attached_base_pattern.unsqueeze(0),
            "base_plane_focus_weight": base_focus_weight.unsqueeze(0),
            "footprint_detail_pattern": detail_pattern.unsqueeze(0),
            "detail_modulation": detail_modulation.unsqueeze(0),
            "light_response": light_response,
            "surface_focus": surface_focus,
            "focus_gain": focus_gain,
            "world_x": surface_points_world[0:1],
            "world_y": surface_points_world[1:2],
            "world_z": surface_points_world[2:3],
            "world_up": world_up.unsqueeze(0),
            "plane_hit_x": receiver_state["receiver_hit_world"][self.lateral_axes[0] : self.lateral_axes[0] + 1],
            "plane_hit_y": receiver_state["receiver_hit_world"][self.lateral_axes[1] : self.lateral_axes[1] + 1],
            "plane_focus_weight": base_focus_weight.unsqueeze(0),
            "plane_query_x": receiver_state["receiver_hit_world"][self.lateral_axes[0] : self.lateral_axes[0] + 1],
            "plane_query_y": receiver_state["receiver_hit_world"][self.lateral_axes[1] : self.lateral_axes[1] + 1],
            "receiver_blur_weight": receiver_blur_weight.unsqueeze(0),
            "receiver_alignment": receiver_state["receiver_alignment"].unsqueeze(0),
            "refraction_spread": refraction_spread.unsqueeze(0),
            "receiver_footprint_u": receiver_state["receiver_footprint_u"].unsqueeze(0),
            "receiver_footprint_v": receiver_state["receiver_footprint_v"].unsqueeze(0),
            "receiver_focus_weight": receiver_focus_weight.unsqueeze(0),
            "receiver_support": receiver_state["receiver_support"].unsqueeze(0),
            "refracted_dir_x": local_refracted_dir[0:1],
            "refracted_dir_y": local_refracted_dir[1:2],
            "refracted_dir_z": local_refracted_dir[2:3],
            "local_refracted_dir_x": local_refracted_dir[0:1],
            "local_refracted_dir_y": local_refracted_dir[1:2],
            "local_refracted_dir_z": local_refracted_dir[2:3],
            "water_level": torch.ones_like(world_u).unsqueeze(0) * water_level,
        }

    def compute_base_plane_caustic(self, bounds, water_level, time_value, surface_sampler, device, dtype):
        plane_grid_u, plane_grid_v = self.build_plane_grid(bounds, device, dtype)
        # Sample the wave surface at fixed world coordinates. Time evolution of
        # the caustic pattern comes from the Gerstner wave's own phase term
        # (frequency·dir·x − speed·t) inside sample_world: multi-component
        # interference naturally produces the non-rigid "twinkle + slow drift"
        # look of real water. Applying an additional rigid translation here
        # (amplitude²-weighted mean phase velocity × t) double-counted the
        # phase velocity and collapsed all motion into one uniform direction,
        # which made the flicker look periodic and conveyor-belt-like.
        water_surface = self.sample_water_surface(surface_sampler, plane_grid_u, plane_grid_v, time_value, dtype)
        plane_surface_level = water_level + water_surface["surface_height"].squeeze(0)
        plane_water_normal = self.to_world_vector(water_surface["surface_normal"])
        incident_dir = self.get_incident_direction(device, dtype).view(3, 1, 1).expand_as(plane_water_normal)
        refracted_dir = self.refract(incident_dir, plane_water_normal, self.eta)
        ref_level = 0.35 * bounds["up_min"] + 0.65 * bounds["up_center"]
        hit_u, hit_v, ray_valid = self.project_to_reference_plane(
            plane_grid_u,
            plane_grid_v,
            plane_surface_level,
            refracted_dir,
            ref_level,
        )
        focus_weight = self.compute_focus_weights(hit_u, hit_v, ray_valid)
        plane_map = self.accumulate_plane_density(hit_u, hit_v, ray_valid, focus_weight, bounds, device, dtype)
        plane_map = self.blur_plane_map(plane_map)
        return plane_map, focus_weight

    def resolve_water_level(self, bounds, device, dtype):
        if self.water_level_mode == "absolute":
            return torch.tensor(self.water_level_world, device=device, dtype=dtype)
        return bounds["up_max"] + self.water_level_offset

    def compute_world_bounds(self, device, dtype):
        scene_min = self.scene_bounds_min.to(device=device, dtype=dtype)
        scene_max = self.scene_bounds_max.to(device=device, dtype=dtype)
        u_axis, v_axis = self.lateral_axes
        u_min = scene_min[u_axis]
        u_max = scene_max[u_axis]
        v_min = scene_min[v_axis]
        v_max = scene_max[v_axis]
        up_min = scene_min[self.up_axis]
        up_max = scene_max[self.up_axis]
        u_pad = (u_max - u_min) * self.plane_padding + 1e-4
        v_pad = (v_max - v_min) * self.plane_padding + 1e-4
        return {
            "x_min": u_min - u_pad,
            "x_max": u_max + u_pad,
            "y_min": v_min - v_pad,
            "y_max": v_max + v_pad,
            "up_min": up_min,
            "up_max": up_max,
            "up_center": 0.5 * (up_min + up_max),
        }

    def build_plane_grid(self, bounds, device, dtype):
        ys, xs = torch.meshgrid(
            torch.linspace(bounds["y_min"], bounds["y_max"], self.plane_res, device=device, dtype=dtype),
            torch.linspace(bounds["x_min"], bounds["x_max"], self.plane_res, device=device, dtype=dtype),
            indexing="ij",
        )
        return xs, ys

    def sample_water_surface(self, surface_sampler, plane_grid_u, plane_grid_v, time_value, dtype):
        if surface_sampler is None:
            zeros = torch.zeros_like(plane_grid_u)
            ones = torch.ones_like(plane_grid_u)
            return {
                "surface_height": zeros.unsqueeze(0),
                "surface_normal": torch.stack((zeros, zeros, ones), dim=0),
            }
        # Prefer the caustic-specific low-band surface when available. The full
        # spectrum (detail + packet + flow warp) drives the visible water
        # appearance, but feeding those sub-coherence-scale features into the
        # refraction Jacobian makes caustics flicker at the capillary tempo
        # instead of the swell tempo.
        if hasattr(surface_sampler, "sample_caustic_surface"):
            return surface_sampler.sample_caustic_surface(plane_grid_u, plane_grid_v, time_value)
        return surface_sampler.sample_world(plane_grid_u, plane_grid_v, time_value)

    def get_incident_direction(self, device, dtype):
        incident = -self.light_dir.to(device=device, dtype=dtype)
        return F.normalize(incident, dim=0, eps=1e-8)

    def refract(self, incident, normal, eta):
        cos_i = torch.clamp(-(incident * normal).sum(dim=0, keepdim=True), min=-1.0, max=1.0)
        k = 1.0 - eta * eta * (1.0 - cos_i * cos_i)
        sqrt_k = torch.sqrt(torch.clamp(k, min=1e-8))
        refracted = eta * incident + (eta * cos_i - sqrt_k) * normal
        refracted = F.normalize(refracted, dim=0, eps=1e-8)
        invalid = k <= 0.0
        refracted = torch.where(invalid.expand_as(refracted), incident, refracted)
        return refracted

    def project_to_reference_plane(self, plane_grid_u, plane_grid_v, surface_level, refracted_dir, ref_level):
        dir_up = refracted_dir[self.up_axis]
        ray_valid = torch.abs(dir_up) > 1e-6
        tau = (ref_level - surface_level) / torch.where(ray_valid, dir_up, torch.ones_like(dir_up))
        ray_valid = ray_valid & (tau > 0)
        hit_u = plane_grid_u + tau * refracted_dir[self.lateral_axes[0]]
        hit_v = plane_grid_v + tau * refracted_dir[self.lateral_axes[1]]
        return hit_u, hit_v, ray_valid

    def accumulate_plane_density(self, hit_u, hit_v, ray_valid, focus_weight, bounds, device, dtype):
        res = self.plane_res
        u = (hit_u - bounds["x_min"]) / (bounds["x_max"] - bounds["x_min"] + 1e-6) * (res - 1)
        v = (hit_v - bounds["y_min"]) / (bounds["y_max"] - bounds["y_min"] + 1e-6) * (res - 1)

        u0 = torch.floor(u).long()
        v0 = torch.floor(v).long()
        u1 = u0 + 1
        v1 = v0 + 1

        wu1 = (u - u0.float()).clamp(0.0, 1.0)
        wv1 = (v - v0.float()).clamp(0.0, 1.0)
        wu0 = 1.0 - wu1
        wv0 = 1.0 - wv1

        plane = torch.zeros(res * res, device=device, dtype=dtype)
        self._splat(plane, u0, v0, wu0 * wv0 * focus_weight, ray_valid, res)
        self._splat(plane, u1, v0, wu1 * wv0 * focus_weight, ray_valid, res)
        self._splat(plane, u0, v1, wu0 * wv1 * focus_weight, ray_valid, res)
        self._splat(plane, u1, v1, wu1 * wv1 * focus_weight, ray_valid, res)
        plane = plane.view(1, res, res)
        plane = F.avg_pool2d(plane.unsqueeze(0), kernel_size=3, stride=1, padding=1).squeeze(0)
        return plane

    def build_downwelling_base_map(self, plane_map):
        # Multi-scale Gaussian-style smoothing on the plane map. We do not
        # mix in the raw plane_map anywhere — its cell-edge stairsteps
        # (plane_res=96 gives ~0.1 m cells, while a caustic stripe is finer
        # than a cell) would otherwise pass straight through grid_sample as
        # digital-particle aliasing at the pixel edges. Real caustics are
        # already blurred by the sun's ~0.5° angular size on the receiver
        # plane, which justifies removing the unsmoothed component.
        fine = self.blur_plane_map(plane_map)
        coarse = self.blur_plane_map(fine)
        downwelling = 0.55 * fine + 0.45 * coarse
        downwelling = torch.pow(torch.clamp(downwelling, min=0.0), 0.95)
        return downwelling

    def query_downwelling(
        self,
        world_u,
        world_v,
        time_value,
        valid_mask=None,
        surface_sampler=None,
        precomputed=None,
    ):
        device = world_u.device
        dtype = world_u.dtype
        valid = self._to_valid_mask(valid_mask, world_u)
        bounds = self.compute_world_bounds(device, dtype)

        if precomputed is not None and precomputed.get("downwelling_base_map") is not None:
            downwelling_map = precomputed["downwelling_base_map"]
        else:
            water_level = self.resolve_water_level(bounds, device, dtype)
            base_plane_map, _ = self.compute_base_plane_caustic(
                bounds,
                water_level,
                time_value,
                surface_sampler,
                device,
                dtype,
            )
            downwelling_map = self.build_downwelling_base_map(base_plane_map)

        sampled = self.query_plane_map(downwelling_map, world_u, world_v, bounds, valid)
        return sampled.unsqueeze(0), {
            "bounds": bounds,
            "downwelling_base_map": downwelling_map,
        }

    def query_water_height(
        self,
        world_u,
        world_v,
        time_value,
        surface_sampler=None,
        precomputed=None,
    ):
        device = world_u.device
        dtype = world_u.dtype
        if precomputed is not None and precomputed.get("water_level") is not None:
            water_level = precomputed["water_level"].to(device=device, dtype=dtype)
            if water_level.numel() > 1:
                water_level = water_level.mean()
        else:
            bounds = self.compute_world_bounds(device, dtype)
            water_level = self.resolve_water_level(bounds, device, dtype)
        if surface_sampler is None:
            return torch.ones_like(world_u).unsqueeze(0) * water_level
        surface_outputs = surface_sampler.sample_world(world_u, world_v, time_value)
        return water_level + surface_outputs["surface_height"]

    def _splat(self, plane, u_idx, v_idx, weight, ray_valid, res):
        valid = ray_valid & (u_idx >= 0) & (u_idx < res) & (v_idx >= 0) & (v_idx < res)
        if not valid.any():
            return
        flat_idx = v_idx[valid] * res + u_idx[valid]
        plane.scatter_add_(0, flat_idx.reshape(-1), weight[valid].reshape(-1))

    def query_plane_map(self, plane_map, world_u, world_v, bounds, valid_mask):
        grid_x = 2.0 * (world_u - bounds["x_min"]) / (bounds["x_max"] - bounds["x_min"] + 1e-6) - 1.0
        grid_y = 2.0 * (world_v - bounds["y_min"]) / (bounds["y_max"] - bounds["y_min"] + 1e-6) - 1.0
        grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0)
        sampled = F.grid_sample(plane_map.unsqueeze(0), grid, mode="bilinear", padding_mode="border", align_corners=True)
        sampled = sampled.squeeze(0).squeeze(0) * valid_mask.squeeze(0)
        return sampled

    @staticmethod
    def blur_plane_map(plane_map):
        blurred = plane_map.unsqueeze(0)
        blurred = F.avg_pool2d(blurred, kernel_size=5, stride=1, padding=2)
        blurred = F.avg_pool2d(blurred, kernel_size=5, stride=1, padding=2)
        return blurred.squeeze(0)

    def compute_focus_weights(self, hit_u, hit_v, ray_valid):
        hit_u_dx = self.spatial_derivative_x(hit_u)
        hit_u_dy = self.spatial_derivative_y(hit_u)
        hit_v_dx = self.spatial_derivative_x(hit_v)
        hit_v_dy = self.spatial_derivative_y(hit_v)
        det_j = torch.abs(hit_u_dx * hit_v_dy - hit_u_dy * hit_v_dx)
        focus_weight = 1.0 / torch.clamp(det_j, min=1e-3)
        if ray_valid.any():
            valid_weight = focus_weight[ray_valid]
            focus_weight = focus_weight / (valid_weight.mean() + 1e-6)
        focus_weight = torch.clamp(focus_weight, min=0.0, max=6.0)
        return focus_weight * ray_valid.float()

    def compute_local_refracted_dir(self, water_outputs, surface_sampler, world_u, world_v, time_value, device, dtype):
        if water_outputs is not None and water_outputs.get("surface_normal") is not None:
            local_water_normal = self.to_world_vector(water_outputs["surface_normal"])
        elif surface_sampler is not None:
            local_water_normal = self.to_world_vector(
                surface_sampler.sample_world(world_u, world_v, time_value)["surface_normal"]
            )
        else:
            zeros = torch.zeros_like(world_u)
            ones = torch.ones_like(world_u)
            local_water_normal = self.to_world_vector(torch.stack((zeros, zeros, ones), dim=0))

        incident_dir = self.get_incident_direction(device, dtype)
        local_incident = incident_dir.view(3, 1, 1).expand_as(local_water_normal)
        return self.refract(local_incident, local_water_normal, self.eta)

    def project_to_receiver_surface(
        self,
        surface_points_world,
        scene_normal,
        world_u,
        world_v,
        water_level,
        local_refracted_dir,
        valid_mask,
        water_outputs=None,
    ):
        receiver_normal = F.normalize(scene_normal, dim=0, eps=1e-8)
        water_surface_points = surface_points_world.clone()
        water_surface_points[self.up_axis : self.up_axis + 1] = water_level
        if water_outputs is not None and water_outputs.get("surface_height") is not None:
            water_surface_points[self.up_axis : self.up_axis + 1] = water_level + water_outputs["surface_height"]

        tangent_u, tangent_v = self.build_receiver_tangent_basis(receiver_normal, world_u)
        numer = torch.sum(receiver_normal * (surface_points_world - water_surface_points), dim=0)
        denom = torch.sum(receiver_normal * local_refracted_dir, dim=0)
        valid = valid_mask.squeeze(0).bool()
        receiver_valid = valid & (torch.abs(denom) > 1e-4)
        safe_denom = torch.where(receiver_valid, denom, torch.ones_like(denom))
        travel = numer / safe_denom
        receiver_valid = receiver_valid & (travel > 0.0)
        safe_travel = torch.where(receiver_valid, travel, torch.zeros_like(travel))

        receiver_hit_world = water_surface_points + safe_travel.unsqueeze(0) * local_refracted_dir
        receiver_offset = receiver_hit_world - surface_points_world
        footprint_u = torch.sum(receiver_offset * tangent_u, dim=0)
        footprint_v = torch.sum(receiver_offset * tangent_v, dim=0)
        footprint_radius = torch.sqrt(torch.clamp(footprint_u * footprint_u + footprint_v * footprint_v, min=0.0))

        receiver_alignment_raw = torch.clamp(
            torch.abs(torch.sum(local_refracted_dir * receiver_normal, dim=0)),
            min=0.0,
            max=1.0,
        )
        receiver_alignment = 0.55 + 0.45 * receiver_alignment_raw
        if receiver_valid.any():
            radius_scale = footprint_radius[receiver_valid].mean() + 1e-6
        else:
            radius_scale = footprint_radius.new_tensor(1.0)
        receiver_support = (0.45 + 0.55 * torch.exp(-footprint_radius / (4.5 * radius_scale))) * receiver_valid.float()

        footprint_u = footprint_u * receiver_valid.float()
        footprint_v = footprint_v * receiver_valid.float()
        receiver_hit_world = torch.where(receiver_valid.unsqueeze(0), receiver_hit_world, surface_points_world)
        return {
            "receiver_hit_world": receiver_hit_world,
            "receiver_footprint_u": footprint_u,
            "receiver_footprint_v": footprint_v,
            "receiver_travel": safe_travel.abs(),
            "receiver_alignment": receiver_alignment * receiver_valid.float(),
            "receiver_valid": receiver_valid,
            "receiver_support": receiver_support,
        }

    def compute_receiver_focus_weights(self, footprint_u, footprint_v, receiver_valid, valid_mask, refraction_spread=None):
        det_j = torch.abs(
            self.spatial_derivative_x(footprint_u) * self.spatial_derivative_y(footprint_v)
            - self.spatial_derivative_y(footprint_u) * self.spatial_derivative_x(footprint_v)
        )
        focus_weight = 1.0 / torch.clamp(det_j, min=1e-4)
        valid = receiver_valid & valid_mask.squeeze(0).bool()
        if valid.any():
            focus_weight = focus_weight / (focus_weight[valid].mean() + 1e-6)
        if refraction_spread is not None:
            focus_weight = focus_weight / (1.0 + 0.15 * refraction_spread)
        focus_weight = torch.clamp(focus_weight, min=0.0, max=6.0)
        return focus_weight * receiver_valid.float()

    def compute_receiver_blur_weight(self, receiver_travel, scene_normal, valid_mask, receiver_alignment=None, refraction_spread=None):
        valid = valid_mask.squeeze(0).bool()
        if valid.any():
            travel_norm = receiver_travel / (receiver_travel[valid].mean() + 1e-6)
        else:
            travel_norm = receiver_travel.new_ones(receiver_travel.shape)
        receiver_tilt = 1.0 - torch.clamp(scene_normal[self.up_axis].abs(), min=0.0, max=1.0)
        if receiver_alignment is None:
            receiver_alignment = torch.ones_like(receiver_travel)
        if refraction_spread is None:
            refraction_spread = torch.zeros_like(receiver_travel)
        if valid.any():
            spread_norm = refraction_spread / (refraction_spread[valid].mean() + 1e-6)
        else:
            spread_norm = refraction_spread.new_zeros(refraction_spread.shape)
        blur_weight = 0.10 * travel_norm + 0.48 * receiver_tilt + 0.10 * (1.0 - receiver_alignment) + 0.20 * spread_norm
        blur_weight = torch.clamp(blur_weight, min=0.0, max=0.72)
        return blur_weight * valid_mask.squeeze(0).float()

    def compute_refraction_spread(self, local_refracted_dir, valid_mask, water_outputs=None):
        dir_u = local_refracted_dir[self.lateral_axes[0]]
        dir_v = local_refracted_dir[self.lateral_axes[1]]
        spread = (
            self.spatial_derivative_x(dir_u).abs()
            + self.spatial_derivative_y(dir_u).abs()
            + self.spatial_derivative_x(dir_v).abs()
            + self.spatial_derivative_y(dir_v).abs()
        )
        if water_outputs is not None and water_outputs.get("surface_gradient_magnitude") is not None:
            spread = spread + 0.35 * water_outputs["surface_gradient_magnitude"].squeeze(0)
        valid = valid_mask.squeeze(0).bool()
        if valid.any():
            spread = spread / (spread[valid].mean() + 1e-6)
        return torch.clamp(spread, min=0.0, max=3.0) * valid_mask.squeeze(0).float()

    def compute_physical_lambda(self, water_outputs, valid_mask):
        """
        Geometric-optics prior for caustic intensity.

        For a sinusoidal wave surface, the light focusing gain above the mean
        irradiance scales with wave slope variance σ²:
            λ_phys = λ_min + k_focus × σ²_slope

        k_focus = 2.0 is calibrated so that typical ocean swell (σ ≈ 0.15 rad)
        yields λ_phys ≈ 0.10–0.15, producing clearly visible caustics.

        The learnable self.lambda_c acts as a scene-specific residual correction
        on top of this physics-derived base.
        """
        if water_outputs is None:
            return torch.zeros(1, device=self.light_dir.device, dtype=self.light_dir.dtype)

        grad_mag = water_outputs.get("surface_gradient_magnitude")
        curvature = water_outputs.get("surface_curvature")

        device = self.light_dir.device
        dtype = self.light_dir.dtype

        if grad_mag is not None:
            src = grad_mag.squeeze(0)
        elif curvature is not None:
            src = curvature.squeeze(0).abs() * 0.5
        else:
            return torch.zeros(1, device=device, dtype=dtype)

        valid = valid_mask.squeeze(0).bool()
        if valid.any():
            slope_sq = (src[valid] ** 2).mean().clamp(0.0, 1.0)
        else:
            slope_sq = (src ** 2).mean().clamp(0.0, 1.0)

        return (self.min_lambda + 2.0 * slope_sq).to(device=device, dtype=dtype)

    def normalize_pattern(self, pattern, valid_mask):
        valid = valid_mask.squeeze(0).bool()
        if valid.any():
            mean_value = pattern[valid].mean()
        else:
            mean_value = pattern.new_tensor(1.0)
        pattern = pattern / (mean_value + 1e-6)
        pattern = torch.clamp(pattern, min=0.0, max=self.max_pattern)
        return pattern

    @staticmethod
    def spatial_derivative_x(value):
        grad = torch.zeros_like(value)
        grad[:, 1:-1] = 0.5 * (value[:, 2:] - value[:, :-2])
        grad[:, 0] = value[:, 1] - value[:, 0]
        grad[:, -1] = value[:, -1] - value[:, -2]
        return grad

    @staticmethod
    def spatial_derivative_y(value):
        grad = torch.zeros_like(value)
        grad[1:-1, :] = 0.5 * (value[2:, :] - value[:-2, :])
        grad[0, :] = value[1, :] - value[0, :]
        grad[-1, :] = value[-1, :] - value[-2, :]
        return grad

    def _to_valid_mask(self, visible_mask, reference):
        if visible_mask is None:
            return torch.ones((1, reference.shape[-2], reference.shape[-1]), device=reference.device, dtype=torch.bool)
        return visible_mask > 0.5

    def to_world_vector(self, local_vector):
        world_vector = torch.zeros_like(local_vector)
        world_vector[self.lateral_axes[0]] = local_vector[0]
        world_vector[self.lateral_axes[1]] = local_vector[1]
        world_vector[self.up_axis] = local_vector[2]
        return F.normalize(world_vector, dim=0, eps=1e-8)

    def build_receiver_tangent_basis(self, receiver_normal, reference):
        ref_axis = torch.zeros_like(receiver_normal)
        ref_axis[self.up_axis : self.up_axis + 1] = 1.0
        tangent_u = torch.cross(ref_axis, receiver_normal, dim=0)
        tangent_norm = torch.linalg.norm(tangent_u, dim=0, keepdim=True)

        fallback_axis = torch.zeros_like(receiver_normal)
        fallback_axis[self.lateral_axes[0] : self.lateral_axes[0] + 1] = 1.0
        fallback_tangent = torch.cross(fallback_axis, receiver_normal, dim=0)
        tangent_u = torch.where(tangent_norm > 1e-4, tangent_u, fallback_tangent)
        tangent_u = F.normalize(tangent_u, dim=0, eps=1e-8)
        tangent_v = F.normalize(torch.cross(receiver_normal, tangent_u, dim=0), dim=0, eps=1e-8)
        return tangent_u, tangent_v

    @classmethod
    def _resolve_up_axis(cls, up_axis):
        key = str(up_axis).strip().lower()
        if key not in cls.AXIS_TO_INDEX:
            raise ValueError(f"Unsupported up axis '{up_axis}'. Expected one of {tuple(cls.AXIS_TO_INDEX.keys())}.")
        return cls.AXIS_TO_INDEX[key]

    @staticmethod
    def _resolve_scene_bounds(scene_bounds):
        if scene_bounds is None:
            scene_min = torch.tensor([-1.0, -1.0, -1.0], dtype=torch.float32)
            scene_max = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
            return scene_min, scene_max
        scene_min = scene_bounds["min"].detach().float().view(3)
        scene_max = scene_bounds["max"].detach().float().view(3)
        return scene_min, torch.maximum(scene_max, scene_min + 1e-4)

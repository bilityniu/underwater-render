import os

import torch

from models import RefractionCausticField, UnderwaterMedium, UnderwaterRenderer, WaterLightField, WaterSurface


AXIS_TO_INDEX = {"x": 0, "y": 1, "z": 2}


SURFACE_PRESETS = {
    "custom": {},
    "beaufort1": {
        "A1": 0.035,
        "A2": 0.020,
        "w1": 4.8,
        "w2": 6.2,
        "s1": 0.16,
        "s2": 0.12,
        "num_waves": 6,
        "num_bands": 2,
        "directional_spread": 0.35,
        "detail_strength": 0.10,
        "flow_strength": 0.025,
        "flow_speed": 0.035,
        "min_wave_amplitude": 0.004,
    },
    "beaufort3": {
        "A1": 0.08,
        "A2": 0.05,
        "w1": 6.0,
        "w2": 7.5,
        "s1": 0.35,
        "s2": 0.28,
        "num_waves": 8,
        "num_bands": 3,
        "directional_spread": 0.60,
        "detail_strength": 0.24,
        "flow_strength": 0.06,
        "flow_speed": 0.08,
        "min_wave_amplitude": 0.008,
    },
    "beaufort5": {
        "A1": 0.13,
        "A2": 0.08,
        "w1": 5.8,
        "w2": 7.0,
        "s1": 0.55,
        "s2": 0.42,
        "num_waves": 10,
        "num_bands": 4,
        "directional_spread": 0.85,
        "detail_strength": 0.32,
        "flow_strength": 0.09,
        "flow_speed": 0.12,
        "min_wave_amplitude": 0.012,
    },
}


def compute_scene_bounds_from_gaussians(gaussians, padding_ratio=0.08):
    xyz = gaussians.get_xyz.detach()
    scene_min = xyz.min(dim=0).values
    scene_max = xyz.max(dim=0).values
    extent = torch.clamp(scene_max - scene_min, min=1e-4)
    padding = extent * float(padding_ratio)
    return {
        "min": scene_min - padding,
        "max": scene_max + padding,
        "center": 0.5 * (scene_min + scene_max),
        "extent": extent + 2.0 * padding,
    }


def resolve_surface_coord_bounds(scene_bounds, up_axis="z"):
    if scene_bounds is None:
        return None, None
    axis_index = AXIS_TO_INDEX[str(up_axis).strip().lower()]
    lateral_axes = tuple(index for index in range(3) if index != axis_index)
    scene_min = scene_bounds["min"].detach().cpu()
    scene_max = scene_bounds["max"].detach().cpu()
    coord_min = torch.stack((scene_min[lateral_axes[0]], scene_min[lateral_axes[1]]))
    coord_max = torch.stack((scene_max[lateral_axes[0]], scene_max[lateral_axes[1]]))
    return coord_min, coord_max


def resolve_surface_preset(surface_preset):
    preset_name = str(surface_preset).strip().lower()
    if preset_name not in SURFACE_PRESETS:
        print(f"[WaterSurface] Unknown preset '{surface_preset}', falling back to 'beaufort3'.")
        preset_name = "beaufort3"
    return preset_name, SURFACE_PRESETS[preset_name]


def build_underwater_renderer(uw_args, scene_bounds=None):
    surface_coord_min, surface_coord_max = resolve_surface_coord_bounds(scene_bounds, uw_args.water_up_axis)
    use_jerlov_prior = bool(int(getattr(uw_args, "medium_use_jerlov_prior", 1)))
    beta_direct = None if use_jerlov_prior else [uw_args.uw_beta_r, uw_args.uw_beta_g, uw_args.uw_beta_b]
    beta_back = None if use_jerlov_prior else [uw_args.uw_beta_b_r, uw_args.uw_beta_b_g, uw_args.uw_beta_b_b]
    b_inf = None if use_jerlov_prior else [uw_args.uw_binf_r, uw_args.uw_binf_g, uw_args.uw_binf_b]

    medium = UnderwaterMedium(
        beta=beta_direct,
        beta_backscatter=beta_back,
        b_inf=b_inf,
        jerlov_type=uw_args.medium_jerlov_type,
        use_jerlov_prior=use_jerlov_prior,
        beta_min=uw_args.medium_beta_min,
        beta_max=uw_args.medium_beta_max,
        grid_resolution=uw_args.medium_grid_resolution,
        depth_scale=uw_args.medium_depth_scale,
        spatial_scale=uw_args.medium_spatial_scale,
        num_basis=uw_args.medium_num_basis,
        beta_strength=uw_args.medium_beta_strength,
        binf_strength=uw_args.medium_binf_strength,
        scene_bounds=scene_bounds,
        sh_degree=uw_args.medium_sh_degree,
        enable_motion=uw_args.medium_enable_motion,
        motion_grid_resolution=uw_args.medium_motion_grid_resolution,
        motion_scale=uw_args.medium_motion_scale,
        directional_strength=uw_args.medium_directional_strength,
    )
    surface_preset_name, surface_preset = resolve_surface_preset(getattr(uw_args, "surface_preset", "beaufort3"))
    surface_kwargs = {
        "A1": uw_args.surface_A1,
        "A2": uw_args.surface_A2,
        "w1": uw_args.surface_w1,
        "w2": uw_args.surface_w2,
        "s1": uw_args.surface_s1,
        "s2": uw_args.surface_s2,
        "spatial_scale": uw_args.surface_spatial_scale,
        "num_waves": uw_args.surface_num_waves,
        "num_bands": uw_args.surface_num_bands,
        "dominant_angle": uw_args.surface_dominant_angle,
        "directional_spread": uw_args.surface_directional_spread,
        "detail_strength": uw_args.surface_detail_strength,
        "base_height": uw_args.surface_base_height,
        "tide_amplitude": uw_args.surface_tide_amplitude,
        "tide_speed": uw_args.surface_tide_speed,
        "tide_phase": uw_args.surface_tide_phase,
        "flow_strength": uw_args.surface_flow_strength,
        "flow_speed": uw_args.surface_flow_speed,
        "min_wave_amplitude": uw_args.surface_min_wave_amplitude,
        "min_wave_speed": uw_args.surface_min_wave_speed,
        "min_tide_speed": uw_args.surface_min_tide_speed,
        "max_tide_amplitude": uw_args.surface_max_tide_amplitude,
        "max_base_height_offset": uw_args.surface_max_base_height_offset,
        "coord_min": surface_coord_min,
        "coord_max": surface_coord_max,
    }
    surface_kwargs.update(surface_preset)
    surface = WaterSurface(**surface_kwargs)
    surface.learnable = bool(getattr(uw_args, "surface_learnable", False))
    surface.preset_name = surface_preset_name
    if not surface.learnable:
        for param in surface.parameters():
            param.requires_grad_(False)
    caustic = RefractionCausticField(
        lambda_c=uw_args.caustic_lambda,
        light_dir=(
            uw_args.caustic_light_dir_x,
            uw_args.caustic_light_dir_y,
            uw_args.caustic_light_dir_z,
        ),
        eta=uw_args.caustic_refract_eta,
        plane_res=uw_args.caustic_plane_res,
        plane_padding=uw_args.caustic_plane_padding,
        water_level_offset=uw_args.caustic_water_level_offset,
        water_level_mode=uw_args.caustic_water_level_mode,
        water_level_world=uw_args.caustic_water_level_world,
        max_pattern=uw_args.caustic_max_pattern,
        min_lambda=uw_args.caustic_min_lambda,
        receiver_query_mix=uw_args.caustic_receiver_query_mix,
        beam_strength=uw_args.caustic_beam_strength,
        beam_depth_decay=uw_args.caustic_beam_depth_decay,
        beam_view_power=uw_args.caustic_beam_view_power,
        up_axis=uw_args.water_up_axis,
        scene_bounds=scene_bounds,
    )
    water_light_field = WaterLightField(
        medium_module=medium,
        caustic_module=caustic,
        surface_module=surface,
    )
    appearance = None
    enable_interface = bool(getattr(uw_args, "enable_interface", False) or getattr(uw_args, "enable_surface", False))

    # Initialise the medium's motion flow field from Stokes drift so that
    # subsurface currents are physically consistent with the wave field at t=0.
    # The network can subsequently refine this with learnable residuals.
    if getattr(uw_args, "medium_enable_motion", False):
        up_axis_idx = {"x": 0, "y": 1, "z": 2}.get(str(uw_args.water_up_axis).lower(), 2)
        medium.init_from_stokes_drift(surface, up_axis=up_axis_idx, scale=1.0)

    return UnderwaterRenderer(
        medium_module=medium,
        surface_module=surface,
        caustic_module=caustic,
        water_light_field=water_light_field,
        appearance_module=appearance,
        enable_underwater=uw_args.enable_underwater,
        enable_surface=uw_args.enable_surface,
        enable_caustic=uw_args.enable_caustic,
        enable_beam=bool(getattr(uw_args, "enable_beam", True)),
        enable_interface=enable_interface,
        enable_appearance=False,
        submerged_only=uw_args.submerged_only,
        water_level_mode=uw_args.caustic_water_level_mode,
        water_level_offset=uw_args.caustic_water_level_offset,
        water_level_world=uw_args.caustic_water_level_world,
        water_level_quantile=uw_args.water_level_quantile,
        waterline_softness=uw_args.waterline_softness,
        interface_band_width=uw_args.interface_band_width,
        interface_tint_strength=uw_args.interface_tint_strength,
        interface_reflection_strength=uw_args.interface_reflection_strength,
        interface_fresnel_power=uw_args.interface_fresnel_power,
        beam_num_steps=uw_args.caustic_beam_num_steps,
        beam_phase_g=uw_args.caustic_beam_phase_g,
        up_axis=uw_args.water_up_axis,
        scene_bounds=scene_bounds,
    )


def underwater_checkpoint_dir(model_path):
    return os.path.join(model_path, "underwater")


def save_underwater_checkpoint(model_path, iteration, underwater_renderer, optimizer, extra_state=None):
    out_dir = os.path.join(underwater_checkpoint_dir(model_path), f"iteration_{iteration}")
    os.makedirs(out_dir, exist_ok=True)
    payload = {
        "iteration": iteration,
        "model_state_dict": underwater_renderer.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "extra_state": extra_state or {},
    }
    checkpoint_path = os.path.join(out_dir, "underwater.pth")
    import torch

    torch.save(payload, checkpoint_path)
    return checkpoint_path


def resolve_underwater_checkpoint(model_path, checkpoint_path=None):
    if checkpoint_path:
        return checkpoint_path
    base_dir = underwater_checkpoint_dir(model_path)
    if not os.path.isdir(base_dir):
        return None
    iterations = []
    for name in os.listdir(base_dir):
        if not name.startswith("iteration_"):
            continue
        try:
            iterations.append(int(name.split("_")[-1]))
        except ValueError:
            continue
    if not iterations:
        return None
    latest_iter = max(iterations)
    return os.path.join(base_dir, f"iteration_{latest_iter}", "underwater.pth")


def load_underwater_checkpoint(model_path, underwater_renderer, optimizer=None, checkpoint_path=None, strict=True):
    resolved_path = resolve_underwater_checkpoint(model_path, checkpoint_path)
    if resolved_path is None or not os.path.isfile(resolved_path):
        return 0, None
    import torch

    payload = torch.load(resolved_path)
    model_state = payload["model_state_dict"]
    current_state = underwater_renderer.state_dict()
    if strict:
        underwater_renderer.load_state_dict(model_state, strict=True)
    else:
        compatible_state = {}
        skipped_keys = []
        skip_surface_state = (
            getattr(underwater_renderer, "surface_module", None) is not None
            and not bool(getattr(underwater_renderer.surface_module, "learnable", True))
        )
        for key, value in model_state.items():
            if skip_surface_state and key.startswith("surface_module."):
                skipped_keys.append(key)
                continue
            if key not in current_state:
                skipped_keys.append(key)
                continue
            if current_state[key].shape != value.shape:
                skipped_keys.append(key)
                continue
            if torch.is_tensor(value) and (torch.isnan(value).any() or torch.isinf(value).any()):
                safe_value = current_state[key].to(device=value.device, dtype=value.dtype)
                mask = torch.isnan(value) | torch.isinf(value)
                value = torch.where(mask, safe_value, value)
            compatible_state[key] = value
        underwater_renderer.load_state_dict(compatible_state, strict=False)
        if skipped_keys:
            print(
                f"[UnderwaterCheckpoint] Skipped {len(skipped_keys)} incompatible keys "
                f"(e.g., {', '.join(skipped_keys[:5])})."
            )
    if getattr(underwater_renderer, "surface_module", None) is not None and hasattr(underwater_renderer.surface_module, "sanitize_parameters_"):
        surface_fallback = {}
        for key, value in current_state.items():
            if key.startswith("surface_module."):
                surface_fallback[key.split("surface_module.", 1)[1]] = value
        underwater_renderer.surface_module.sanitize_parameters_(fallback_state=surface_fallback)
    if optimizer is not None and payload.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    return int(payload.get("iteration", 0)), resolved_path

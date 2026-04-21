import torch
from torch import nn


class UnderwaterRenderer(nn.Module):
    """Composes medium, surface, and caustics on top of PGSR scene outputs."""

    AXIS_TO_INDEX = {"x": 0, "y": 1, "z": 2}

    def __init__(
        self,
        medium_module,
        surface_module=None,
        caustic_module=None,
        water_light_field=None,
        appearance_module=None,
        enable_underwater=True,
        enable_surface=True,
        enable_caustic=True,
        enable_beam=True,
        enable_interface=True,
        enable_appearance=False,
        submerged_only=True,
        water_level_mode="scene_relative",
        water_level_offset=0.0,
        water_level_world=0.0,
        water_level_quantile=0.98,
        waterline_softness=0.08,
        interface_band_width=0.18,
        interface_tint_strength=0.18,
        interface_reflection_strength=0.35,
        interface_fresnel_power=5.0,
        beam_num_steps=6,
        beam_phase_g=0.85,
        up_axis="z",
        scene_bounds=None,
    ):
        super().__init__()
        self.medium_module = medium_module
        self.surface_module = surface_module
        self.caustic_module = caustic_module
        self.water_light_field = water_light_field
        self.appearance_module = appearance_module
        self.enable_underwater = bool(enable_underwater)
        self.enable_surface = bool(enable_surface)
        self.enable_caustic = bool(enable_caustic)
        self.enable_beam = bool(enable_beam)
        self.enable_interface = bool(enable_interface)
        self.enable_appearance = bool(enable_appearance)
        self.submerged_only = bool(submerged_only)
        self.water_level_mode = str(water_level_mode).lower()
        self.water_level_offset = float(water_level_offset)
        self.water_level_world = float(water_level_world)
        self.water_level_quantile = float(water_level_quantile)
        self.waterline_softness = float(waterline_softness)
        self.interface_band_width = float(interface_band_width)
        self.interface_tint_strength = float(interface_tint_strength)
        self.interface_reflection_strength = float(interface_reflection_strength)
        self.interface_fresnel_power = float(interface_fresnel_power)
        self.beam_num_steps = int(beam_num_steps)
        self.beam_phase_g = float(beam_phase_g)
        self.up_axis = self._resolve_up_axis(up_axis)
        self.lateral_axes = tuple(index for index in range(3) if index != self.up_axis)
        scene_min, scene_max = self._resolve_scene_bounds(scene_bounds)
        self.register_buffer("scene_bounds_min", scene_min)
        self.register_buffer("scene_bounds_max", scene_max)
        self.register_buffer("background_depth", torch.clamp((scene_max - scene_min).max() * 1.25, min=1.0).view(1))

    def forward(self, scene_outputs, camera, time_value):
        scene_rgb = scene_outputs["scene_rgb"]
        scene_rgb_raw = scene_outputs.get("scene_rgb_raw", scene_rgb)
        scene_depth = scene_outputs["scene_depth"]
        scene_normal = scene_outputs["scene_normal"]
        surface_points_world = scene_outputs["surface_points_world"]
        view_dir_world = scene_outputs.get("view_dir_world", torch.zeros_like(surface_points_world))
        pixel_ray_dir_world = scene_outputs.get("pixel_ray_dir_world", torch.zeros_like(surface_points_world))
        camera_center_world = scene_outputs.get("camera_center_world", torch.zeros_like(surface_points_world))
        surface_valid_mask = scene_outputs.get("surface_valid_mask", (scene_depth > 0).float())

        visible_mask = (scene_depth > 0).float() * surface_valid_mask

        outputs = {
            "scene_rgb": scene_rgb,
            "scene_rgb_raw": scene_rgb_raw,
            "scene_depth": scene_depth,
            "scene_normal": scene_normal,
            "surface_points_world": surface_points_world,
            "surface_position_world": scene_outputs.get("surface_position_world", surface_points_world),
            "surface_valid_mask": surface_valid_mask,
            "camera_center_world": camera_center_world,
            "view_dir_world": view_dir_world,
            "pixel_ray_dir_world": pixel_ray_dir_world,
        }

        medium_depth = scene_depth
        if self.submerged_only:
            background_depth = self.background_depth.to(device=scene_depth.device, dtype=scene_depth.dtype).view(1, 1, 1)
            medium_depth = torch.where(visible_mask > 0.5, scene_depth, torch.ones_like(scene_depth) * background_depth)

        transmission = torch.ones_like(scene_rgb)
        underwater_rgb = scene_rgb_raw
        if self.enable_underwater and self.medium_module is not None:
            medium_outputs = self.medium_module(
                scene_rgb_raw,
                medium_depth,
                surface_points_world=surface_points_world,
                valid_mask=visible_mask,
                time_controls={"time_value": time_value},
                flow_controls={
                    "view_dir_world": view_dir_world,
                    "pixel_ray_dir_world": pixel_ray_dir_world,
                },
            )
            transmission = medium_outputs["transmission"]
            underwater_rgb = medium_outputs["underwater_rgb"]
            outputs.update(medium_outputs)
        else:
            outputs.update({"transmission": transmission, "underwater_rgb": underwater_rgb, "effective_depth": medium_depth})

        zero_world = torch.zeros_like(scene_depth)
        surface_outputs = {
            "surface_height": torch.zeros_like(scene_depth),
            "surface_base_height": torch.zeros_like(scene_depth),
            "surface_tide_height": torch.zeros_like(scene_depth),
            "surface_gradient": torch.zeros(
                (2, scene_depth.shape[-2], scene_depth.shape[-1]),
                device=scene_depth.device,
                dtype=scene_depth.dtype,
            ),
            "surface_normal": torch.zeros_like(scene_normal),
            "surface_gradient_magnitude": torch.zeros_like(scene_depth),
            "surface_curvature": torch.zeros_like(scene_depth),
            "surface_local_variation": torch.zeros_like(scene_depth),
            "surface_wave_packet_energy": torch.zeros_like(scene_depth),
        }
        if self.enable_surface and self.surface_module is not None:
            surface_outputs = self.surface_module.sample_world(
                surface_points_world[self.lateral_axes[0]],
                surface_points_world[self.lateral_axes[1]],
                time_value,
            )
        outputs.update(surface_outputs)
        water_surface_normal_world = self.local_to_world_vector(surface_outputs["surface_normal"])

        caustic_outputs = {
            "caustic_pattern": torch.zeros_like(scene_depth),
            "caustic": torch.zeros_like(scene_depth),
            "caustic_gain": torch.ones_like(scene_depth),
            "light_beam": torch.zeros_like(scene_depth),
            "light_beam_rgb": torch.zeros_like(scene_rgb),
            "light_response": torch.zeros_like(scene_depth),
            "surface_focus": torch.zeros_like(scene_depth),
            "focus_gain": torch.ones_like(scene_depth),
            "receiver_blur_weight": torch.zeros_like(scene_depth),
            "world_x": surface_points_world[0:1],
            "world_y": surface_points_world[1:2],
            "world_z": surface_points_world[2:3],
            "world_up": surface_points_world[self.up_axis : self.up_axis + 1],
        }
        if self.enable_caustic and self.caustic_module is not None:
            caustic_outputs = self.caustic_module(
                surface_points_world,
                time_value,
                scene_normal,
                water_outputs=surface_outputs,
                visible_mask=visible_mask,
                surface_sampler=self.surface_module,
            )
        outputs.update(caustic_outputs)

        water_level = self.resolve_water_level(scene_depth)
        if self.enable_surface and self.surface_module is not None:
            water_height_world = water_level + surface_outputs["surface_height"]
        else:
            water_height_world = torch.ones_like(scene_depth) * water_level

        world_up = surface_points_world[self.up_axis : self.up_axis + 1]
        interface_distance = water_height_world - world_up
        water_surface_point_world = self.build_water_surface_points(surface_points_world, water_height_world)
        softness = max(self.waterline_softness, 1e-4)
        if self.submerged_only:
            underwater_mask = torch.ones_like(scene_depth)
        else:
            underwater_mask = torch.sigmoid(interface_distance / softness) * visible_mask

        down_dir_world = torch.stack(
            (
                caustic_outputs.get("refracted_dir_x", torch.zeros_like(scene_depth)),
                caustic_outputs.get("refracted_dir_y", torch.zeros_like(scene_depth)),
                caustic_outputs.get("refracted_dir_z", -torch.ones_like(scene_depth)),
            ),
            dim=0,
        ).squeeze(1)
        down_dir_world = torch.nn.functional.normalize(down_dir_world, p=2, dim=0, eps=1e-8)

        illumination_transport = self.build_illumination_transport(
            scene_rgb=scene_rgb_raw,
            surface_points_world=surface_points_world,
            interface_distance=interface_distance,
            visible_mask=visible_mask,
            caustic_outputs=caustic_outputs,
            time_value=time_value,
        )
        outputs.update(illumination_transport)

        field_outputs = None
        if self.water_light_field is not None:
            field_outputs = self.water_light_field.query(
                surface_points_world,
                time_value,
                view_dir_world=view_dir_world,
                down_dir_world=down_dir_world,
                valid_mask=visible_mask,
                caustic_outputs=caustic_outputs,
            )
            outputs["water_light_sigma_t_view"] = field_outputs["sigma_t_view"]
            outputs["water_light_sigma_t_down"] = field_outputs["sigma_t_down"]
            outputs["water_light_sigma_s"] = field_outputs["sigma_s"]
            outputs["water_light_surface"] = field_outputs["L_surface"]
            outputs["water_light_in"] = field_outputs["L_in"]
            outputs["water_light_depth_down"] = field_outputs["depth_down"]
            outputs["global_downwelling_irradiance"] = field_outputs["downwelling_scalar"]

        # ---- Unified single-scatter ray-march ----
        # Along each camera ray we integrate two volumetric sources that share
        # the same transmittance T(s) = exp(-∫σ_t ds):
        #   S_direct(p)  = σ_s(p) · φ_HG(θ) · L_direct(p)
        #                    single-scatter of the downwelling column toward
        #                    the camera — this is what the eye reads as light
        #                    shafts/beams.
        #   S_ambient(p) = σ_t(p) · b_inf(p)
        #                    diffuse in-scatter equivalent to Jaffe's haze
        #                    term; it converges to the closed-form backscatter
        #                    in the homogeneous limit.
        # The boundary radiance is scene_rgb_raw attenuated by the same column
        # and shaded by the surface-attached caustic gain G_surface. The three
        # effects (beams, haze, surface caustics) therefore fall out of a
        # single integral rather than being composited from separate modules.
        zeros_rgb_ref = torch.zeros_like(scene_rgb)
        beam_rgb = zeros_rgb_ref.clone()
        ambient_rgb = zeros_rgb_ref.clone()
        phase_map = torch.zeros_like(scene_depth)

        if field_outputs is not None and self.water_light_field is not None:
            cos_theta = torch.clamp(
                ((-pixel_ray_dir_world) * down_dir_world).sum(dim=0, keepdim=True),
                min=-1.0,
                max=1.0,
            )
            g = self.beam_phase_g
            denom = torch.clamp(1.0 + g * g - 2.0 * g * cos_theta, min=1e-4)
            phase = (1.0 - g * g) / (4.0 * torch.pi * torch.pow(denom, 1.5))
            phase_map = phase

            scene_distance = torch.linalg.norm(
                surface_points_world - camera_center_world, dim=0, keepdim=True
            )
            background_distance = self.background_depth.to(
                device=scene_depth.device, dtype=scene_depth.dtype
            ).view(1, 1, 1)
            scene_distance = torch.where(
                visible_mask > 0.5,
                scene_distance,
                torch.ones_like(scene_distance) * background_distance,
            )
            if self.submerged_only:
                t_surface = torch.zeros_like(scene_depth)
            else:
                ray_interface = self.estimate_ray_interface(
                    camera_center_world=camera_center_world,
                    pixel_ray_dir_world=pixel_ray_dir_world,
                    water_level=water_level,
                    visible_mask=visible_mask,
                    surface_points_world=surface_points_world,
                    time_value=time_value,
                )
                t_surface = torch.clamp(ray_interface["ray_interface_distance"], min=0.0)
            D = torch.clamp(scene_distance - t_surface, min=1e-4)

            step_count = max(int(self.beam_num_steps), 1)
            dt = D / float(step_count)
            optical_depth = zeros_rgb_ref.clone()

            for step_idx in range(step_count):
                alpha = (step_idx + 0.5) / float(step_count)
                t_i = t_surface + alpha * D
                p_i = camera_center_world + pixel_ray_dir_world * t_i
                field_i = self.water_light_field.query(
                    p_i,
                    time_value,
                    view_dir_world=-pixel_ray_dir_world,
                    down_dir_world=down_dir_world,
                    valid_mask=visible_mask,
                    caustic_outputs=caustic_outputs,
                )
                sigma_t_i = field_i["sigma_t_view"]
                sigma_s_i = field_i["sigma_s"]
                L_in_i = field_i["L_in"]
                b_inf_i = field_i["b_inf"]
                down_scalar_i = torch.clamp(field_i["downwelling_scalar"], min=0.0, max=4.0)

                optical_depth = optical_depth + sigma_t_i * dt
                T_i = torch.exp(-optical_depth)
                # Column emphasis on the direct source: the caustic pattern
                # modulates the single-scatter magnitude without changing its
                # mean, so bright columns appear as shafts and the integral
                # stays energy-consistent.
                S_direct = sigma_s_i * phase * L_in_i * down_scalar_i
                S_ambient = sigma_t_i * b_inf_i
                beam_rgb = beam_rgb + S_direct * T_i * dt
                ambient_rgb = ambient_rgb + S_ambient * T_i * dt

            # Upper-hemisphere gate mirrors physical shaft visibility: only
            # rays pointing toward the surface cross the caustic columns.
            ray_up = pixel_ray_dir_world[self.up_axis : self.up_axis + 1]
            upper_weight = torch.sigmoid((ray_up + 0.02) / 0.30)
            beam_rgb = beam_rgb * upper_weight * visible_mask
            beam_rgb = torch.clamp(
                torch.nn.functional.avg_pool2d(
                    beam_rgb.unsqueeze(0), kernel_size=3, stride=1, padding=1
                ).squeeze(0),
                0.0,
                1.0,
            )
            ambient_rgb = ambient_rgb * visible_mask
            direct_underwater_rgb = scene_rgb_raw * outputs.get(
                "transmission", torch.exp(-optical_depth)
            )
        else:
            direct_underwater_rgb = outputs.get("direct_underwater_rgb")
            if direct_underwater_rgb is None:
                direct_underwater_rgb = scene_rgb_raw * outputs.get(
                    "transmission", torch.ones_like(scene_rgb)
                )
            fallback_backscatter = outputs.get("backscatter")
            if fallback_backscatter is None:
                fallback_backscatter = torch.clamp(underwater_rgb - direct_underwater_rgb, min=0.0)
            ambient_rgb = fallback_backscatter

        attached_pattern = caustic_outputs.get(
            "surface_attached_pattern",
            caustic_outputs.get(
                "attached_base_pattern",
                caustic_outputs.get("base_caustic_pattern", torch.ones_like(scene_depth)),
            ),
        )
        receiver_support = caustic_outputs.get(
            "receiver_support", torch.ones_like(scene_depth)
        ).clamp(0.0, 1.0)
        if self.enable_caustic and self.caustic_module is not None:
            lambda_phys = self.caustic_module.compute_physical_lambda(surface_outputs, visible_mask)
            effective_lambda = torch.clamp(
                lambda_phys + self.caustic_module.lambda_c,
                min=self.caustic_module.min_lambda,
            )
        else:
            effective_lambda = torch.ones_like(scene_depth) * 0.15

        local_delta = (attached_pattern - 1.0) * receiver_support
        surface_detail = effective_lambda * local_delta
        G_surface = torch.clamp(1.0 + surface_detail, min=0.55, max=1.75)

        ambient_compressed = ambient_rgb / (1.0 + ambient_rgb.detach())
        direct_shaded = direct_underwater_rgb * G_surface
        underwater_base = torch.clamp(direct_shaded + ambient_compressed, 0.0, 1.0)
        underwater_lit = torch.clamp(underwater_base + beam_rgb, 0.0, 1.0)

        outputs["direct_underwater_rgb"] = direct_underwater_rgb
        outputs["backscatter"] = ambient_rgb
        outputs["surface_caustic_base"] = torch.zeros_like(scene_depth)
        outputs["surface_caustic_detail"] = surface_detail
        outputs["global_irradiance_variation"] = torch.ones_like(scene_depth)
        outputs["caustic"] = G_surface - 1.0
        outputs["caustic_gain"] = G_surface
        outputs["global_irradiance_mod"] = torch.ones_like(scene_depth)
        outputs["light_beam"] = beam_rgb.mean(dim=0, keepdim=True)
        outputs["light_beam_rgb"] = beam_rgb
        outputs["light_beam_source"] = torch.zeros_like(scene_depth)
        outputs["light_beam_alignment"] = phase_map
        outputs["light_beam_depth"] = torch.zeros_like(scene_depth)
        outputs["underwater_base"] = underwater_base
        outputs["underwater_lit"] = underwater_lit
        interface_outputs = self.build_visible_interface(
            scene_rgb_raw=scene_rgb_raw,
            underwater_rgb=underwater_lit,
            scene_depth=scene_depth,
            visible_mask=visible_mask,
            surface_points_world=surface_points_world,
            camera_center_world=camera_center_world,
            pixel_ray_dir_world=pixel_ray_dir_world,
            water_level=water_level,
            water_height_world=water_height_world,
            interface_distance=interface_distance,
            water_surface_normal_world=water_surface_normal_world,
            underwater_mask=underwater_mask,
            time_value=time_value,
            b_inf_map=outputs.get("b_inf_map"),
        )
        outputs.update(interface_outputs)

        if self.submerged_only:
            final_rgb = underwater_lit
        else:
            final_rgb = scene_rgb_raw * (1.0 - underwater_mask) + underwater_lit * underwater_mask
        if self.enable_interface:
            interface_alpha = interface_outputs["visible_interface_alpha"]
            final_rgb = final_rgb * (1.0 - interface_alpha) + interface_outputs["visible_interface_rgb"] * interface_alpha

        outputs["final_rgb_pre_app"] = torch.clamp(final_rgb, 0.0, 1.0)
        if self.enable_appearance and self.appearance_module is not None:
            app_outputs = self.appearance_module(outputs["final_rgb_pre_app"])
            outputs.update(app_outputs)
            outputs["final_rgb"] = app_outputs["appearance_rgb"]
        else:
            outputs["final_rgb"] = outputs["final_rgb_pre_app"]

        outputs["visible_mask"] = visible_mask
        outputs["water_level"] = torch.ones_like(scene_depth) * water_level
        outputs["water_height_world"] = water_height_world
        outputs["world_up"] = world_up
        outputs["interface_distance"] = interface_distance
        outputs["water_surface_point_world"] = water_surface_point_world
        outputs["water_surface_point_x"] = water_surface_point_world[0:1]
        outputs["water_surface_point_y"] = water_surface_point_world[1:2]
        outputs["water_surface_point_z"] = water_surface_point_world[2:3]
        outputs["water_surface_normal_world"] = water_surface_normal_world
        outputs["water_surface_normal_world_x"] = water_surface_normal_world[0:1]
        outputs["water_surface_normal_world_y"] = water_surface_normal_world[1:2]
        outputs["water_surface_normal_world_z"] = water_surface_normal_world[2:3]
        outputs["view_dir_x"] = view_dir_world[0:1]
        outputs["view_dir_y"] = view_dir_world[1:2]
        outputs["view_dir_z"] = view_dir_world[2:3]
        outputs["pixel_ray_dir_x"] = pixel_ray_dir_world[0:1]
        outputs["pixel_ray_dir_y"] = pixel_ray_dir_world[1:2]
        outputs["pixel_ray_dir_z"] = pixel_ray_dir_world[2:3]
        outputs["underwater_mask"] = underwater_mask
        outputs["underwater_gbuffer"] = {
            "water_level": outputs["water_level"],
            "water_height_world": outputs["water_height_world"],
            "interface_distance": outputs["interface_distance"],
            "underwater_mask": outputs["underwater_mask"],
            "water_surface_point_world": outputs["water_surface_point_world"],
            "water_surface_normal_world": outputs["water_surface_normal_world"],
            "surface_local_variation": outputs.get("surface_local_variation"),
            "surface_wave_packet_energy": outputs.get("surface_wave_packet_energy"),
            "surface_interface_band": outputs.get("surface_interface_band"),
            "ray_interface_mask": outputs.get("ray_interface_mask"),
            "interface_fresnel": outputs.get("interface_fresnel"),
            "visible_interface_alpha": outputs.get("visible_interface_alpha"),
            "beta_d_map": outputs.get("beta_d_map"),
            "beta_b_map": outputs.get("beta_b_map"),
            "b_inf_map": outputs.get("b_inf_map"),
            "medium_directionality": outputs.get("medium_directionality"),
            "medium_motion_magnitude": outputs.get("medium_motion_magnitude"),
            "medium_flow_velocity": outputs.get("medium_flow_velocity"),
            "transmission": outputs.get("transmission"),
            "caustic_pattern": outputs.get("caustic_pattern"),
            "caustic_gain": outputs.get("caustic_gain"),
            "light_beam": outputs.get("light_beam"),
            "light_beam_source": outputs.get("light_beam_source"),
            "light_beam_alignment": outputs.get("light_beam_alignment"),
            "light_beam_depth": outputs.get("light_beam_depth"),
            "receiver_blur_weight": outputs.get("receiver_blur_weight"),
            "receiver_alignment": outputs.get("receiver_alignment"),
            "refraction_spread": outputs.get("refraction_spread"),
            "receiver_footprint_u": outputs.get("receiver_footprint_u"),
            "receiver_footprint_v": outputs.get("receiver_footprint_v"),
            "receiver_focus_weight": outputs.get("receiver_focus_weight"),
            "receiver_support": outputs.get("receiver_support"),
        }
        return outputs

    def build_visible_light_beam(
        self,
        scene_depth,
        visible_mask,
        surface_points_world,
        camera_center_world,
        pixel_ray_dir_world,
        water_level,
        time_value,
        caustic_outputs,
        water_height_world=None,
        n_beam_steps=6,
    ):
        zeros_scalar = torch.zeros_like(scene_depth)
        zeros_rgb = torch.zeros((3, scene_depth.shape[-2], scene_depth.shape[-1]), device=scene_depth.device, dtype=scene_depth.dtype)
        outputs = {
            "light_beam": zeros_scalar,
            "light_beam_rgb": zeros_rgb,
            "light_beam_source": zeros_scalar,
            "light_beam_alignment": zeros_scalar,
            "light_beam_depth": zeros_scalar,
        }
        if (not self.enable_beam) or (not self.enable_caustic) or self.caustic_module is None:
            return outputs

        ray_interface = self.estimate_ray_interface(
            camera_center_world=camera_center_world,
            pixel_ray_dir_world=pixel_ray_dir_world,
            water_level=water_level,
            visible_mask=visible_mask,
            surface_points_world=surface_points_world,
            time_value=time_value,
        )
        ray_up = pixel_ray_dir_world[self.up_axis : self.up_axis + 1]
        upper_weight = torch.sigmoid((ray_up + 0.02) / 0.30)

        # beam_valid: submerged camera vs. above-water camera.
        # ray_interface_mask requires the water surface to lie between camera and
        # scene — correct for above-water cameras, but wrong for submerged ones
        # where every scene pixel is already inside the water volume.
        if self.submerged_only:
            beam_mask = visible_mask * upper_weight
            beam_valid = visible_mask > 0.5
        else:
            beam_mask = ray_interface["ray_interface_mask"] * upper_weight
            beam_valid = beam_mask > 1e-3
        if not beam_valid.any():
            return outputs

        down_dir = torch.cat(
            (
                caustic_outputs.get("refracted_dir_x", torch.zeros_like(scene_depth)),
                caustic_outputs.get("refracted_dir_y", torch.zeros_like(scene_depth)),
                caustic_outputs.get("refracted_dir_z", -torch.ones_like(scene_depth)),
            ),
            dim=0,
        )
        down_dir = torch.nn.functional.normalize(down_dir, p=2, dim=0, eps=1e-8)
        cos_theta = torch.clamp(
            torch.sum((-pixel_ray_dir_world) * down_dir, dim=0, keepdim=True),
            min=-1.0,
            max=1.0,
        )
        g = self.beam_phase_g
        denom = torch.clamp(1.0 + g * g - 2.0 * g * cos_theta, min=1e-4)
        phase = (1.0 - g * g) / (4.0 * torch.pi * torch.pow(denom, 1.5))

        scene_distance = torch.linalg.norm(
            surface_points_world - camera_center_world, dim=0, keepdim=True
        )
        background_distance = self.background_depth.to(device=scene_depth.device, dtype=scene_depth.dtype).view(1, 1, 1)
        scene_distance = torch.where(
            visible_mask > 0.5, scene_distance, torch.ones_like(scene_distance) * background_distance
        )
        if self.submerged_only:
            t_surface = torch.zeros_like(scene_depth)
        else:
            t_surface = torch.clamp(ray_interface["ray_interface_distance"], min=0.0)
        D = torch.clamp(scene_distance - t_surface, min=1e-4)

        step_count = max(int(n_beam_steps), 1)
        dt = D / float(step_count)
        beam_rgb = zeros_rgb.clone()
        optical_depth = zeros_rgb.clone()
        L_surface_accum = zeros_rgb.clone()
        T_cam_accum = torch.zeros_like(scene_depth)

        for step_idx in range(step_count):
            alpha = (step_idx + 0.5) / float(step_count)
            t_i = t_surface + alpha * D
            p_i = camera_center_world + pixel_ray_dir_world * t_i

            field_i = self.water_light_field.query(
                p_i,
                time_value,
                view_dir_world=-pixel_ray_dir_world,
                down_dir_world=down_dir,
                valid_mask=beam_valid.float(),
                caustic_outputs=caustic_outputs,
            )
            sigma_t_i = field_i["sigma_t_view"]
            sigma_s_i = field_i["sigma_s"]
            L_in_i = field_i["L_in"]
            # Column emphasis: multiply by the local downwelling pattern so
            # bright caustic columns get boosted single-scatter (visible as
            # shafts) and dim inter-column regions get suppressed. The mean
            # of down_scalar is ≈ 1 by construction, so total beam energy is
            # roughly preserved while the spatial structure becomes shaft-like.
            # (The diffuse-ambient in-scatter is already represented by the
            # medium backscatter channel; beams only need to carry the
            # caustic-driven anisotropic component.)
            down_scalar_i = torch.clamp(field_i["downwelling_scalar"], min=0.0, max=4.0)
            L_shaft_i = L_in_i * down_scalar_i

            optical_depth = optical_depth + sigma_t_i * dt
            T_cam_i = torch.exp(-optical_depth)
            dI = sigma_s_i * phase * L_shaft_i * T_cam_i * dt

            beam_rgb = beam_rgb + dI
            L_surface_accum = L_surface_accum + field_i["L_surface"] * dt
            T_cam_accum = T_cam_accum + T_cam_i.mean(dim=0, keepdim=True) * dt

        beam_rgb = torch.clamp(beam_rgb * beam_mask, min=0.0, max=1.0)
        beam_rgb = torch.clamp(
            torch.nn.functional.avg_pool2d(beam_rgb.unsqueeze(0), kernel_size=3, stride=1, padding=1).squeeze(0),
            0.0,
            1.0,
        )
        beam_intensity = beam_rgb.mean(dim=0, keepdim=True)
        L_surface_avg = L_surface_accum / torch.clamp(D, min=1e-4)
        T_cam_avg = T_cam_accum / torch.clamp(D, min=1e-4)
        outputs.update(
            {
                "light_beam": beam_intensity,
                "light_beam_rgb": beam_rgb,
                "light_beam_source": L_surface_avg.mean(dim=0, keepdim=True),
                "light_beam_alignment": phase,
                "light_beam_depth": T_cam_avg,
            }
        )
        return outputs

    def build_illumination_transport(
        self,
        scene_rgb,
        surface_points_world,
        interface_distance,
        visible_mask,
        caustic_outputs,
        time_value,
    ):
        zeros = torch.zeros_like(interface_distance)
        outputs = {
            "down_transmission": torch.ones_like(interface_distance),
            "down_blur_strength": zeros,
            "beam_visibility": zeros,
            "down_distance": zeros,
        }
        if self.medium_module is None or self.caustic_module is None:
            return outputs

        down_dir_world = torch.stack(
            (
                caustic_outputs.get("refracted_dir_x", zeros),
                caustic_outputs.get("refracted_dir_y", zeros),
                caustic_outputs.get("refracted_dir_z", -torch.ones_like(interface_distance)),
            ),
            dim=0,
        ).squeeze(1)
        down_dir_world = torch.nn.functional.normalize(down_dir_world, p=2, dim=0, eps=1e-8)
        down_dir_up = torch.clamp(down_dir_world[self.up_axis : self.up_axis + 1].abs(), min=0.20)
        submerged_depth = torch.clamp(interface_distance, min=0.0)
        down_distance = submerged_depth / down_dir_up

        transport = self.medium_module.compute_downwelling_transport(
            scene_rgb,
            surface_points_world,
            down_dir_world,
            down_distance,
            valid_mask=visible_mask,
            time_controls={"time_value": time_value},
        )
        transport["down_distance"] = down_distance
        return transport

    @staticmethod
    def soft_blur_scalar(value):
        blurred = value.unsqueeze(0)
        blurred = torch.nn.functional.avg_pool2d(blurred, kernel_size=5, stride=1, padding=2)
        blurred = torch.nn.functional.avg_pool2d(blurred, kernel_size=5, stride=1, padding=2)
        return blurred.squeeze(0)

    @staticmethod
    def masked_mean(value, valid_mask):
        valid = valid_mask > 0.5
        if valid.any():
            return value[valid].mean()
        return value.mean()

    def make_unit_mean_gain(self, field, valid_mask, scale=1.0, clamp_min=0.8, clamp_max=1.2):
        mean_value = self.masked_mean(field, valid_mask).clamp(min=1e-6)
        normalized = field / mean_value
        gain = 1.0 + float(scale) * (normalized - 1.0)
        return torch.clamp(gain, min=clamp_min, max=clamp_max)

    def resolve_water_level(self, reference_depth):
        device = reference_depth.device
        dtype = reference_depth.dtype
        if self.water_level_mode == "absolute":
            return torch.tensor(self.water_level_world, device=device, dtype=dtype)
        scene_top = self.scene_bounds_max.to(device=device, dtype=dtype)[self.up_axis]
        return scene_top + self.water_level_offset

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

    def local_to_world_vector(self, local_vector):
        world_vector = torch.zeros_like(local_vector)
        world_vector[self.lateral_axes[0]] = local_vector[0]
        world_vector[self.lateral_axes[1]] = local_vector[1]
        world_vector[self.up_axis] = local_vector[2]
        return torch.nn.functional.normalize(world_vector, p=2, dim=0, eps=1e-8)

    def build_water_surface_points(self, surface_points_world, water_height_world):
        water_surface_point_world = surface_points_world.clone()
        water_surface_point_world[self.up_axis : self.up_axis + 1] = water_height_world
        return water_surface_point_world

    def build_default_world_normal(self, reference_tensor):
        world_normal = torch.zeros_like(reference_tensor)
        world_normal[self.up_axis : self.up_axis + 1] = 1.0
        return world_normal

    def resolve_interface_color(self, reference_tensor, b_inf_map=None):
        if b_inf_map is not None:
            return b_inf_map
        if self.medium_module is not None and hasattr(self.medium_module, "b_inf_base"):
            return self.medium_module.b_inf_base.to(
                device=reference_tensor.device,
                dtype=reference_tensor.dtype,
            ).expand(3, reference_tensor.shape[-2], reference_tensor.shape[-1])
        color = torch.tensor([0.05, 0.25, 0.45], device=reference_tensor.device, dtype=reference_tensor.dtype).view(3, 1, 1)
        return color.expand(3, reference_tensor.shape[-2], reference_tensor.shape[-1])

    def estimate_ray_interface(self, camera_center_world, pixel_ray_dir_world, water_level, visible_mask, surface_points_world, time_value):
        ray_up = pixel_ray_dir_world[self.up_axis : self.up_axis + 1]
        camera_up = camera_center_world[self.up_axis : self.up_axis + 1]
        eps = 1e-5
        safe_ray_up = torch.where(ray_up.abs() > eps, ray_up, torch.ones_like(ray_up))
        flat_t = (torch.ones_like(ray_up) * water_level - camera_up) / safe_ray_up
        valid = (ray_up.abs() > eps) & (flat_t > 0.0)

        flat_hit = camera_center_world + pixel_ray_dir_world * flat_t
        hit_x = flat_hit[self.lateral_axes[0]]
        hit_y = flat_hit[self.lateral_axes[1]]

        if self.enable_surface and self.surface_module is not None:
            hit_surface_outputs = self.surface_module.sample_world(hit_x, hit_y, time_value)
            hit_height = water_level + hit_surface_outputs["surface_height"]
            hit_t = (hit_height - camera_up) / safe_ray_up
            valid = valid & (hit_t > 0.0)
            hit_point = camera_center_world + pixel_ray_dir_world * hit_t
            hit_normal_world = self.local_to_world_vector(hit_surface_outputs["surface_normal"])
        else:
            hit_t = flat_t
            hit_height = torch.ones_like(flat_t) * water_level
            hit_point = flat_hit
            hit_normal_world = self.build_default_world_normal(pixel_ray_dir_world)

        scene_distance = torch.linalg.norm(surface_points_world - camera_center_world, dim=0, keepdim=True)
        inf_distance = torch.ones_like(scene_distance) * 1e8
        scene_distance = torch.where(visible_mask > 0.5, scene_distance, inf_distance)
        interface_before_surface = valid.float() * torch.where(
            visible_mask > 0.5,
            (hit_t < scene_distance).float(),
            torch.ones_like(hit_t),
        )
        return {
            "ray_surface_valid": valid.float(),
            "ray_interface_distance": hit_t,
            "ray_interface_mask": interface_before_surface,
            "ray_interface_point_world": hit_point,
            "ray_interface_height_world": hit_height,
            "ray_interface_normal_world": hit_normal_world,
        }

    def build_visible_interface(
        self,
        scene_rgb_raw,
        underwater_rgb,
        scene_depth,
        visible_mask,
        surface_points_world,
        camera_center_world,
        pixel_ray_dir_world,
        water_level,
        water_height_world,
        interface_distance,
        water_surface_normal_world,
        underwater_mask,
        time_value,
        b_inf_map=None,
    ):
        zeros_scalar = torch.zeros_like(scene_depth)
        zeros_rgb = torch.zeros_like(scene_rgb_raw)
        ones_scalar = torch.ones_like(scene_depth)
        outputs = {
            "surface_interface_band": zeros_scalar,
            "ray_interface_mask": zeros_scalar,
            "ray_interface_distance": zeros_scalar,
            "interface_fresnel": zeros_scalar,
            "visible_interface_alpha": zeros_scalar,
            "visible_interface_rgb": zeros_rgb,
            "ray_interface_point_x": zeros_scalar,
            "ray_interface_point_y": zeros_scalar,
            "ray_interface_point_z": zeros_scalar,
        }
        if not self.enable_interface:
            return outputs

        band_width = max(self.interface_band_width, 1e-4)
        surface_interface_band = torch.exp(-torch.square(interface_distance / band_width)) * visible_mask

        ray_interface = self.estimate_ray_interface(
            camera_center_world=camera_center_world,
            pixel_ray_dir_world=pixel_ray_dir_world,
            water_level=water_level,
            visible_mask=visible_mask,
            surface_points_world=surface_points_world,
            time_value=time_value,
        )
        hit_normal_world = ray_interface["ray_interface_normal_world"]
        cos_theta = torch.clamp(
            torch.abs(torch.sum((-pixel_ray_dir_world) * hit_normal_world, dim=0, keepdim=True)),
            min=0.0,
            max=1.0,
        )
        fresnel = self.schlick_fresnel(cos_theta, power=self.interface_fresnel_power)
        water_color = self.resolve_interface_color(scene_depth, b_inf_map=b_inf_map)
        surface_rgb = torch.clamp(0.6 * underwater_rgb + 0.4 * water_color, 0.0, 1.0)
        reflection_rgb = torch.clamp(0.5 + 0.75 * water_color, 0.0, 1.0)

        surface_alpha = torch.clamp(self.interface_tint_strength * surface_interface_band, 0.0, 1.0)
        ray_alpha = torch.clamp(
            self.interface_reflection_strength * ray_interface["ray_interface_mask"] * (0.25 + 0.75 * fresnel),
            0.0,
            1.0,
        )
        total_alpha = torch.clamp(surface_alpha + ray_alpha, 0.0, 1.0)
        denom = torch.clamp(surface_alpha + ray_alpha, min=1e-6)
        visible_interface_rgb = torch.clamp(
            (surface_rgb * surface_alpha + reflection_rgb * ray_alpha) / denom,
            0.0,
            1.0,
        )

        outputs.update(
            {
                "surface_interface_band": surface_interface_band,
                "ray_interface_mask": ray_interface["ray_interface_mask"],
                "ray_interface_distance": ray_interface["ray_interface_distance"],
                "interface_fresnel": fresnel,
                "visible_interface_alpha": total_alpha,
                "visible_interface_rgb": visible_interface_rgb,
                "ray_interface_point_x": ray_interface["ray_interface_point_world"][0:1],
                "ray_interface_point_y": ray_interface["ray_interface_point_world"][1:2],
                "ray_interface_point_z": ray_interface["ray_interface_point_world"][2:3],
                "ray_interface_height_world": ray_interface["ray_interface_height_world"],
                "camera_water_distance": water_level - camera_center_world[self.up_axis : self.up_axis + 1],
            }
        )
        return outputs

    @staticmethod
    def schlick_fresnel(cos_theta, f0=0.020059312199524774, power=5.0):
        one_minus_cos = torch.clamp(1.0 - cos_theta, min=0.0, max=1.0)
        return f0 + (1.0 - f0) * torch.pow(one_minus_cos, power)

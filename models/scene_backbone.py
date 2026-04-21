import torch

from gaussian_renderer import render


class SceneBackboneRenderer:
    """Thin wrapper around PGSR rendering with stable output names."""

    def __call__(
        self,
        camera,
        gaussians,
        pipe,
        background,
        app_model=None,
        scaling_modifier=1.0,
        require_geometry=True,
        prefer_app_image=True,
    ):
        render_pkg = render(
            camera,
            gaussians,
            pipe,
            background,
            scaling_modifier=scaling_modifier,
            app_model=app_model,
            return_plane=require_geometry,
            return_depth_normal=require_geometry,
        )

        if prefer_app_image:
            scene_rgb = render_pkg.get("app_image", render_pkg["render"])
        else:
            scene_rgb = render_pkg["render"]
        outputs = {
            "scene_rgb": scene_rgb,
            "scene_rgb_raw": render_pkg["render"],
            "raw_render_pkg": render_pkg,
        }
        if require_geometry:
            surface_points_world, valid_mask = self.reconstruct_surface_points(
                render_pkg["plane_depth"],
                camera,
            )
            camera_center_world = self.build_camera_center_map(camera, surface_points_world.shape[-2:], depth_device=surface_points_world.device, depth_dtype=surface_points_world.dtype)
            view_dir_world = self.compute_view_directions(surface_points_world, camera_center_world, valid_mask)
            pixel_ray_dir_world = -view_dir_world
            outputs.update(
                {
                    "scene_depth": render_pkg["plane_depth"],
                    "scene_normal": self._normalize_normals(render_pkg["rendered_normal"]),
                    "depth_normal": render_pkg.get("depth_normal"),
                    "rendered_distance": render_pkg.get("rendered_distance"),
                    "surface_points_world": surface_points_world,
                    "surface_position_world": surface_points_world,
                    "surface_valid_mask": valid_mask,
                    "camera_center_world": camera_center_world,
                    "view_dir_world": view_dir_world,
                    "pixel_ray_dir_world": pixel_ray_dir_world,
                }
            )
            outputs["geometry_gbuffer"] = {
                "scene_rgb_raw": outputs["scene_rgb_raw"],
                "scene_depth": outputs["scene_depth"],
                "scene_normal": outputs["scene_normal"],
                "surface_position_world": outputs["surface_position_world"],
                "surface_valid_mask": outputs["surface_valid_mask"],
                "camera_center_world": outputs["camera_center_world"],
                "view_dir_world": outputs["view_dir_world"],
                "pixel_ray_dir_world": outputs["pixel_ray_dir_world"],
            }
        return outputs

    @staticmethod
    def _normalize_normals(normal_map):
        return torch.nn.functional.normalize(normal_map, p=2, dim=0, eps=1e-8)

    @staticmethod
    def reconstruct_surface_points(scene_depth, camera):
        depth = scene_depth.squeeze(0)
        height = int(camera.image_height)
        width = int(camera.image_width)
        device = depth.device
        dtype = depth.dtype

        ys, xs = torch.meshgrid(
            torch.arange(height, device=device, dtype=dtype),
            torch.arange(width, device=device, dtype=dtype),
            indexing="ij",
        )

        x_cam = (xs - camera.Cx) / camera.Fx * depth
        y_cam = (ys - camera.Cy) / camera.Fy * depth
        z_cam = depth
        cam_points = torch.stack((x_cam, y_cam, z_cam), dim=-1).reshape(-1, 3)

        _, cam_to_world = camera.get_calib_matrix_nerf()
        cam_to_world = cam_to_world.to(device=device, dtype=dtype)
        cam_points_h = torch.cat([cam_points, torch.ones_like(cam_points[:, :1])], dim=-1)
        world_points_h = cam_points_h @ cam_to_world
        world_points = world_points_h[:, :3].reshape(height, width, 3).permute(2, 0, 1)

        valid_mask = (
            torch.isfinite(depth).unsqueeze(0)
            & (depth > float(camera.znear)).unsqueeze(0)
            & (depth < float(camera.zfar)).unsqueeze(0)
        )
        world_points = world_points * valid_mask.float()
        return world_points, valid_mask.float()

    @staticmethod
    def build_camera_center_map(camera, spatial_shape, depth_device, depth_dtype):
        height, width = spatial_shape
        if hasattr(camera, "camera_center") and camera.camera_center is not None:
            camera_center = camera.camera_center.to(device=depth_device, dtype=depth_dtype).view(3, 1, 1)
        else:
            _, cam_to_world = camera.get_calib_matrix_nerf()
            cam_to_world = cam_to_world.to(device=depth_device, dtype=depth_dtype)
            camera_center = cam_to_world[3, :3].view(3, 1, 1)
        return camera_center.expand(3, height, width)

    @staticmethod
    def compute_view_directions(surface_points_world, camera_center_world, valid_mask):
        view_dir_world = torch.nn.functional.normalize(
            camera_center_world - surface_points_world,
            p=2,
            dim=0,
            eps=1e-8,
        )
        return view_dir_world * valid_mask.float()

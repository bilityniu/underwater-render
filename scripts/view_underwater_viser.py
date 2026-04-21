import os
import sys
import time
import copy
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
import torch
import viser
import viser.transforms as vtf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arguments import ModelParams, PipelineParams, UnderwaterParams, get_combined_args
from models import SceneBackboneRenderer
from models.underwater_medium import jerlov_rgb_preset
from scene import GaussianModel, Scene
from utils.graphics_utils import getProjectionMatrix
from utils.general_utils import safe_state
from utils.sh_utils import SH2RGB
from utils.underwater_utils import build_underwater_renderer, compute_scene_bounds_from_gaussians, load_underwater_checkpoint


DISPLAY_OPTIONS = [
    "final_rgb",
    "scene_rgb",
    "underwater_rgb",
    "light_beam_rgb",
    "scene_depth",
    "scene_normal",
    "transmission",
    "backscatter",
    "beta_d_map",
    "beta_b_map",
    "b_inf_map",
    "caustic",
    "caustic_pattern",
    "base_caustic_pattern",
    "footprint_detail_pattern",
    "detail_modulation",
    "caustic_gain",
    "light_beam",
    "underwater_mask",
    "effective_depth",
    "surface_height",
    "surface_domain_warp",
    "surface_wave_packet_energy",
    "receiver_blur_weight",
    "receiver_alignment",
    "refraction_spread",
    "receiver_focus_weight",
]

NORM_RANGES = {
    "water_level_quantile": (0.5, 1.0),
    "waterline_softness": (0.01, 0.5),
    "surface_dominant_angle": (-3.14, 3.14),
    "surface_directional_spread": (0.05, 1.8),
    "surface_detail_strength": (0.0, 1.0),
    "surface_tide_amplitude": (0.0, 0.2),
    "surface_tide_speed": (0.01, 1.5),
    "caustic_lambda": (0.0, 6.0),
    "caustic_max_pattern": (0.5, 6.0),
    "medium_density_scale": (0.2, 2.2),
    "medium_backscatter_scale": (0.2, 2.5),
    "medium_tone": (-1.0, 1.0),
    "medium_variation": (0.0, 1.2),
    "medium_depth_scale": (0.2, 6.0),
}


def inverse_softplus(value):
    x = max(float(value), 1e-5)
    return np.log(np.expm1(x))


def inverse_sigmoid(value):
    x = np.clip(float(value), 1e-5, 1.0 - 1e-5)
    return np.log(x / (1.0 - x))


def norm_value(name, actual):
    if name not in NORM_RANGES:
        return float(actual)
    lo, hi = NORM_RANGES[name]
    if hi - lo < 1e-8:
        return 0.0
    actual_value = float(actual)
    if name == "caustic_lambda":
        actual_value = max(actual_value, lo)
        t = np.sqrt((actual_value - lo) / (hi - lo))
    else:
        t = (actual_value - lo) / (hi - lo)
    return float(np.clip(t, 0.0, 1.0))


def denorm_value(name, normalized):
    if name not in NORM_RANGES:
        return float(normalized)
    lo, hi = NORM_RANGES[name]
    t = float(np.clip(normalized, 0.0, 1.0))
    if name == "caustic_lambda":
        t = t * t
    return float(lo + t * (hi - lo))


def camera_to_c2w(camera):
    w2c = camera.world_view_transform.transpose(0, 1).detach().cpu().numpy()
    return np.linalg.inv(w2c).astype(np.float32)


def set_client_camera_from_dataset_view(client, dataset_camera):
    c2w_colmap = camera_to_c2w(dataset_camera)
    c2w_viewer = c2w_colmap.copy()
    c2w_viewer[:3, 1:3] *= -1.0
    rot_viewer_with_flip = c2w_viewer[:3, :3]
    rot_viewer = np.matmul(rot_viewer_with_flip, vtf.SO3.from_x_radians(-np.pi).as_matrix())
    client.camera.position = c2w_viewer[:3, 3]
    client.camera.wxyz = vtf.SO3.from_matrix(rot_viewer).wxyz
    client.camera.fov = float(dataset_camera.FoVy)


def build_runtime_camera_from_viser(template_camera, viser_camera, image_size):
    max_size = int(max(32, image_size))
    aspect = float(getattr(viser_camera, "aspect", 1.0))
    if not np.isfinite(aspect) or aspect <= 1e-6:
        aspect = 1.0
    image_height = max_size
    image_width = int(round(image_height * aspect))
    if image_width > max_size:
        image_width = max_size
        image_height = int(round(image_width / max(aspect, 1e-6)))
    image_width = max(32, image_width)
    image_height = max(32, image_height)

    rot_so3 = vtf.SO3(wxyz=np.asarray(viser_camera.wxyz))
    rot = np.matmul(rot_so3.as_matrix(), vtf.SO3.from_x_radians(np.pi).as_matrix()).astype(np.float32)
    pos = np.asarray(viser_camera.position, dtype=np.float32)

    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = rot
    c2w[:3, 3] = pos
    c2w[:3, 1:3] *= -1.0
    w2c = np.linalg.inv(c2w).astype(np.float32)

    cam = copy.copy(template_camera)
    cam.image_width = image_width
    cam.image_height = image_height
    cam.resolution = (image_width, image_height)

    cam.R = w2c[:3, :3].T
    cam.T = w2c[:3, 3]

    cam.FoVy = float(getattr(viser_camera, "fov", template_camera.FoVy))
    tan_half_y = np.tan(0.5 * cam.FoVy)
    tan_half_x = tan_half_y * (float(image_width) / float(max(1, image_height)))
    cam.FoVx = float(2.0 * np.arctan(max(tan_half_x, 1e-6)))

    cam.Fx = float(image_width) / max(2.0 * tan_half_x, 1e-6)
    cam.Fy = float(image_height) / max(2.0 * tan_half_y, 1e-6)
    cam.Cx = 0.5 * float(image_width)
    cam.Cy = 0.5 * float(image_height)

    cam.world_view_transform = torch.tensor(w2c, device="cuda", dtype=torch.float32).transpose(0, 1)
    cam.projection_matrix = getProjectionMatrix(
        znear=cam.znear,
        zfar=cam.zfar,
        fovX=cam.FoVx,
        fovY=cam.FoVy,
    ).transpose(0, 1).cuda()
    cam.full_proj_transform = (
        cam.world_view_transform.unsqueeze(0).bmm(cam.projection_matrix.unsqueeze(0))
    ).squeeze(0)
    cam.camera_center = cam.world_view_transform.inverse()[3, :3]
    return cam


def rgb_chw_to_u8(tensor):
    image = torch.clamp(tensor, 0.0, 1.0).detach().float().cpu()
    if image.dim() == 2:
        image = image.unsqueeze(0)
    if image.shape[0] == 1:
        image = image.repeat(3, 1, 1)
    image = image.permute(1, 2, 0).numpy()
    return np.clip(image * 255.0, 0, 255).astype(np.uint8)


def normal_chw_to_u8(tensor):
    normal = tensor.detach().float().cpu()
    normal = torch.nn.functional.normalize(normal, p=2, dim=0, eps=1e-8)
    normal = normal.permute(1, 2, 0).numpy()
    return np.clip((normal + 1.0) * 127.5, 0, 255).astype(np.uint8)


def scalar_chw_to_colormap(tensor, colormap):
    value = tensor.detach().float().cpu()
    if value.dim() == 2:
        value = value.unsqueeze(0)
    if value.shape[0] > 1:
        value = value.mean(dim=0, keepdim=True)
    value_np = value.squeeze(0).numpy()
    valid = np.isfinite(value_np)
    if not np.any(valid):
        return np.zeros((value_np.shape[0], value_np.shape[1], 3), dtype=np.uint8)
    lo = np.percentile(value_np[valid], 2.0)
    hi = np.percentile(value_np[valid], 98.0)
    if hi - lo < 1e-8:
        lo = float(np.min(value_np[valid]))
        hi = float(np.max(value_np[valid]) + 1e-8)
    norm = np.clip((value_np - lo) / (hi - lo + 1e-8), 0.0, 1.0)
    value_u8 = np.clip(norm * 255.0, 0, 255).astype(np.uint8)
    bgr = cv2.applyColorMap(value_u8, colormap)
    return bgr[:, :, ::-1]


def tensor_to_display(outputs, key):
    if key not in outputs:
        key = "final_rgb"
    tensor = outputs[key]

    if key in ("scene_rgb", "underwater_rgb", "light_beam_rgb", "final_rgb_pre_app", "final_rgb", "visible_interface_rgb"):
        return rgb_chw_to_u8(tensor)
    if key in ("scene_normal", "water_surface_normal_world"):
        return normal_chw_to_u8(tensor)
    if key == "scene_depth":
        return scalar_chw_to_colormap(tensor, cv2.COLORMAP_OCEAN)
    if key in (
        "transmission",
        "light_response",
        "surface_focus",
        "underwater_mask",
        "surface_interface_band",
        "ray_interface_mask",
        "visible_interface_alpha",
        "interface_fresnel",
    ):
        return scalar_chw_to_colormap(tensor, cv2.COLORMAP_BONE)
    if key in ("backscatter", "b_inf_map"):
        return scalar_chw_to_colormap(tensor, cv2.COLORMAP_OCEAN)
    if key in ("caustic", "caustic_pattern", "caustic_plane_map"):
        return scalar_chw_to_colormap(tensor, cv2.COLORMAP_HOT)
    return scalar_chw_to_colormap(tensor, cv2.COLORMAP_TURBO)


def resolve_views(scene, view_split):
    if view_split == "train":
        views = scene.getTrainCameras()
    else:
        views = scene.getTestCameras()
    if len(views) == 0:
        views = scene.getTrainCameras()
    return views


def covariance_6_to_mat(cov6):
    cov = torch.zeros((cov6.shape[0], 3, 3), device=cov6.device, dtype=cov6.dtype)
    cov[:, 0, 0] = cov6[:, 0]
    cov[:, 0, 1] = cov6[:, 1]
    cov[:, 1, 0] = cov6[:, 1]
    cov[:, 0, 2] = cov6[:, 2]
    cov[:, 2, 0] = cov6[:, 2]
    cov[:, 1, 1] = cov6[:, 3]
    cov[:, 1, 2] = cov6[:, 4]
    cov[:, 2, 1] = cov6[:, 4]
    cov[:, 2, 2] = cov6[:, 5]
    return cov


def add_scene_geometry(
    server,
    gaussians,
    views,
    max_points,
    point_size,
    frustum_stride,
    frustum_scale,
    geometry_mode="gaussians",
    opacity_thresh=0.05,
    scale_min=1e-4,
    scale_max=0.25,
    gaussian_scale=1.0,
):
    server.scene.add_frame("/world", axes_length=0.4, axes_radius=0.01)
    server.scene.add_grid(
        "/grid",
        width=8.0,
        height=8.0,
        plane="xy",
        cell_size=0.5,
        section_size=1.0,
        infinite_grid=True,
        plane_opacity=0.06,
    )

    xyz_all = gaussians.get_xyz
    opacity_all = gaussians.get_opacity.squeeze(-1)
    scaling_all = gaussians.get_scaling
    max_axis = scaling_all.max(dim=1).values

    valid = (
        (opacity_all > float(opacity_thresh))
        & (max_axis > float(scale_min))
        & (max_axis < float(scale_max))
    )
    valid_count = int(valid.sum().item())
    if valid_count == 0:
        valid = opacity_all > float(opacity_thresh * 0.5)
        valid_count = int(valid.sum().item())
    if valid_count == 0:
        valid = opacity_all > 0.0
        valid_count = int(valid.sum().item())

    num_show = min(int(max_points), valid_count)
    score = opacity_all * torch.clamp(max_axis, min=float(scale_min), max=float(scale_max))
    score = torch.where(valid, score, torch.full_like(score, -1.0))
    if num_show < score.shape[0]:
        idx = torch.topk(score, k=max(1, num_show), largest=True, sorted=False).indices
    else:
        idx = torch.arange(score.shape[0], device=xyz_all.device)
    print(
        "[Viewer] Geometry mode={}, selected {}/{} gaussians.".format(
            geometry_mode, idx.shape[0], xyz_all.shape[0]
        )
    )

    xyz = xyz_all[idx]
    opacity = gaussians.get_opacity[idx]
    rgb = torch.clamp(SH2RGB(gaussians._features_dc[idx].squeeze(1)), 0.0, 1.0)

    if str(geometry_mode).lower() == "gaussians":
        scaling = scaling_all[idx].clamp(min=float(scale_min), max=float(scale_max))
        rotation = gaussians.get_rotation[idx]
        cov = gaussians.covariance_activation(scaling, 1.0, rotation)
        cov = covariance_6_to_mat(cov)
        server.scene.add_gaussian_splats(
            "/scene/gaussian_splats",
            centers=xyz.detach().float().cpu().numpy(),
            covariances=cov.detach().float().cpu().numpy(),
            rgbs=rgb.detach().float().cpu().numpy(),
            opacities=opacity.detach().float().cpu().numpy(),
            scale=float(gaussian_scale),
        )
    else:
        intensity = np.clip(opacity.detach().float().cpu().numpy().reshape(-1) * 255.0, 20, 255).astype(np.uint8)
        colors = np.stack((intensity // 2, intensity, np.clip(intensity + 40, 0, 255)), axis=-1)
        server.scene.add_point_cloud(
            "/scene/gaussian_centers",
            points=xyz.detach().float().cpu().numpy(),
            colors=colors,
            point_size=float(point_size),
        )

    stride = max(1, int(frustum_stride))
    for i, camera in enumerate(views[::stride]):
        c2w = camera_to_c2w(camera)
        position = tuple(c2w[:3, 3].tolist())
        rotation = c2w[:3, :3]
        wxyz = tuple(vtf.SO3.from_matrix(rotation).wxyz.tolist())
        server.scene.add_camera_frustum(
            "/scene/cameras/{:03d}".format(i),
            fov=float(camera.FoVy),
            aspect=float(camera.image_width) / float(camera.image_height),
            scale=float(frustum_scale),
            line_width=1.0,
            color=(238, 168, 74),
            wxyz=wxyz,
            position=position,
        )


def apply_live_controls(underwater_renderer, controls):
    underwater_renderer.enable_underwater = bool(controls["enable_underwater"].value)
    underwater_renderer.enable_surface = bool(controls["enable_surface"].value)
    underwater_renderer.enable_caustic = bool(controls["enable_caustic"].value)
    underwater_renderer.enable_beam = bool(controls["enable_beam"].value)
    underwater_renderer.enable_appearance = False
    underwater_renderer.submerged_only = bool(controls["submerged_only"].value)
    underwater_renderer.waterline_softness = max(denorm_value("waterline_softness", controls["waterline_softness"].value), 1e-4)

    level_mode = str(controls["water_level_mode"].value)
    level_offset = float(controls["water_level_offset"].value)
    level_world = float(controls["water_level_world"].value)
    underwater_renderer.water_level_mode = level_mode
    underwater_renderer.water_level_offset = level_offset
    underwater_renderer.water_level_world = level_world
    underwater_renderer.water_level_quantile = denorm_value("water_level_quantile", controls["water_level_quantile"].value)

    medium = underwater_renderer.medium_module
    if medium is not None:
        preset = jerlov_rgb_preset(str(controls["medium_water_type"].value))
        density_scale = denorm_value("medium_density_scale", controls["medium_density_scale"].value)
        backscatter_scale = denorm_value("medium_backscatter_scale", controls["medium_backscatter_scale"].value)
        tone = denorm_value("medium_tone", controls["medium_tone"].value)
        variation = denorm_value("medium_variation", controls["medium_variation"].value)

        beta_direct = torch.tensor(
            preset["beta_d"],
            device=medium.beta_direct_base.device,
            dtype=medium.beta_direct_base.dtype,
        ) * density_scale
        beta_back = torch.tensor(
            preset["beta_b"],
            device=medium.beta_back_base.device,
            dtype=medium.beta_back_base.dtype,
        ) * density_scale
        b_inf = torch.tensor(
            preset["b_inf"],
            device=medium.b_inf_base.device,
            dtype=medium.b_inf_base.dtype,
        )

        tone_green = 1.0 + 0.30 * max(-tone, 0.0) - 0.10 * max(tone, 0.0)
        tone_blue = 1.0 + 0.30 * max(tone, 0.0) - 0.10 * max(-tone, 0.0)
        tone_red = max(0.2, 1.0 - 0.08 * abs(tone))
        b_inf = b_inf * torch.tensor(
            [tone_red, tone_green, tone_blue],
            device=b_inf.device,
            dtype=b_inf.dtype,
        ) * backscatter_scale

        medium.beta_direct_base.data.copy_(beta_direct.view(3, 1, 1))
        medium.beta_back_base.data.copy_(beta_back.view(3, 1, 1))
        medium.b_inf_base.data.copy_(torch.clamp(b_inf, 0.0, 1.0).view(3, 1, 1))
        medium.beta_strength = variation
        medium.binf_strength = variation
        medium.depth_scale = max(denorm_value("medium_depth_scale", controls["medium_depth_scale"].value), 1e-4)

    surface = underwater_renderer.surface_module
    if surface is not None:
        if hasattr(surface, "dominant_angle"):
            surface.dominant_angle.data.fill_(denorm_value("surface_dominant_angle", controls["surface_dominant_angle"].value))
        if hasattr(surface, "raw_directional_spread"):
            spread = max(denorm_value("surface_directional_spread", controls["surface_directional_spread"].value), 1e-4)
            surface.raw_directional_spread.data.fill_(inverse_softplus(spread))
        if hasattr(surface, "raw_detail_strength"):
            detail_strength = denorm_value("surface_detail_strength", controls["surface_detail_strength"].value)
            surface.raw_detail_strength.data.fill_(inverse_sigmoid(detail_strength))
        if hasattr(surface, "raw_tide_amplitude"):
            tide_amp = max(denorm_value("surface_tide_amplitude", controls["surface_tide_amplitude"].value), 0.0)
            surface.raw_tide_amplitude.data.fill_(inverse_softplus(tide_amp + 1e-6))
        if hasattr(surface, "raw_tide_speed"):
            tide_speed = max(denorm_value("surface_tide_speed", controls["surface_tide_speed"].value), 1e-4)
            surface.raw_tide_speed.data.fill_(inverse_softplus(tide_speed))

    caustic = underwater_renderer.caustic_module
    if caustic is not None and hasattr(caustic, "lambda_c"):
        caustic.lambda_c.data.fill_(denorm_value("caustic_lambda", controls["caustic_lambda"].value))
        caustic.max_pattern = denorm_value("caustic_max_pattern", controls["caustic_max_pattern"].value)
        if hasattr(caustic, "water_level_mode"):
            caustic.water_level_mode = level_mode
            caustic.water_level_offset = level_offset
            caustic.water_level_world = level_world


@torch.no_grad()
def render_current(backbone, underwater_renderer, gaussians, pipeline, background, app_model, view, time_value):
    scene_outputs = backbone(
        view,
        gaussians,
        pipeline,
        background,
        app_model=app_model,
        require_geometry=True,
        prefer_app_image=False,
    )
    return underwater_renderer(scene_outputs, view, time_value)


def main():
    parser = ArgumentParser(description="Interactive underwater PGSR viewer with Viser.")
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)
    underwater = UnderwaterParams(parser)

    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--underwater_checkpoint", type=str, default=None)
    parser.add_argument("--view_split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--viewer_target_fps", type=float, default=12.0)
    parser.add_argument("--viewer_preview_scale", type=float, default=1.0)
    parser.add_argument("--viewer_use_free_camera", type=int, default=1)
    parser.add_argument("--viewer_res_moving", type=int, default=640)
    parser.add_argument("--viewer_res_static", type=int, default=1280)
    parser.add_argument("--viewer_move_settle", type=float, default=0.25)
    parser.add_argument("--viewer_max_points", type=int, default=120000)
    parser.add_argument("--viewer_point_size", type=float, default=0.01)
    parser.add_argument("--viewer_geometry_mode", type=str, default="gaussians", choices=["gaussians", "points"])
    parser.add_argument("--viewer_show_geometry", type=int, default=0)
    parser.add_argument("--viewer_opacity_thresh", type=float, default=0.05)
    parser.add_argument("--viewer_scale_min", type=float, default=0.001)
    parser.add_argument("--viewer_scale_max", type=float, default=0.25)
    parser.add_argument("--viewer_gaussian_scale", type=float, default=1.0)
    parser.add_argument("--viewer_frustum_stride", type=int, default=6)
    parser.add_argument("--viewer_frustum_scale", type=float, default=0.09)
    parser.add_argument("--save_dir", type=str, default=None)

    args = get_combined_args(parser)
    safe_state(args.quiet)

    dataset = model.extract(args)
    pipe = pipeline.extract(args)
    uw_args = underwater.extract(args)
    dataset.preload_img = False
    uw_args.enable_appearance = False

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    app_model = None

    scene_bounds = compute_scene_bounds_from_gaussians(gaussians, padding_ratio=uw_args.scene_bounds_padding)
    underwater_renderer = build_underwater_renderer(uw_args, scene_bounds=scene_bounds).cuda()
    _, checkpoint_path = load_underwater_checkpoint(
        dataset.model_path,
        underwater_renderer,
        checkpoint_path=args.underwater_checkpoint,
        strict=False,
    )
    if checkpoint_path is None:
        print("[Viewer] No underwater checkpoint found. Using current module initialization.")
    else:
        print("[Viewer] Loaded underwater checkpoint: {}".format(checkpoint_path))
    underwater_renderer.eval()
    medium_module = underwater_renderer.medium_module

    views = resolve_views(scene, args.view_split)
    if len(views) == 0:
        raise RuntimeError("No cameras found in scene.")

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    backbone = SceneBackboneRenderer()

    server = viser.ViserServer(host=args.host, port=args.port)
    print("[Viewer] Open: http://{}:{}".format(args.host, args.port))

    if bool(args.viewer_show_geometry):
        add_scene_geometry(
            server,
            gaussians,
            views,
            max_points=args.viewer_max_points,
            point_size=args.viewer_point_size,
            frustum_stride=args.viewer_frustum_stride,
            frustum_scale=args.viewer_frustum_scale,
            geometry_mode=args.viewer_geometry_mode,
            opacity_thresh=args.viewer_opacity_thresh,
            scale_min=args.viewer_scale_min,
            scale_max=args.viewer_scale_max,
            gaussian_scale=args.viewer_gaussian_scale,
        )

    with server.gui.add_folder("Render"):
        display_key = server.gui.add_dropdown("Display", options=DISPLAY_OPTIONS, initial_value="final_rgb")
        viewport_bg = server.gui.add_checkbox("Viewport Background", initial_value=True)
        use_free_camera = server.gui.add_checkbox("Use Free Camera", initial_value=bool(args.viewer_use_free_camera))
        camera_index = server.gui.add_slider(
            "Camera Index",
            min=0,
            max=max(0, len(views) - 1),
            step=1,
            initial_value=min(int(args.video_camera_index), len(views) - 1),
        )
        snap_camera_button = server.gui.add_button("Snap FreeCam To Selected View")
        res_moving = server.gui.add_slider(
            "Res Moving",
            min=256,
            max=1920,
            step=32,
            initial_value=int(np.clip(args.viewer_res_moving, 256, 1920)),
        )
        res_static = server.gui.add_slider(
            "Res Static",
            min=256,
            max=2560,
            step=32,
            initial_value=int(np.clip(args.viewer_res_static, 256, 2560)),
        )
        move_settle = server.gui.add_number("Move Settle (s)", initial_value=float(args.viewer_move_settle), min=0.01, step=0.01)
        jpeg_quality = server.gui.add_slider("JPEG Quality", min=40, max=100, step=1, initial_value=85)
        time_value = server.gui.add_number("World Time", initial_value=float(args.video_time_start), step=0.01)
        time_play = server.gui.add_checkbox("Play Time", initial_value=True)
        time_speed = server.gui.add_slider("Time Speed", min=0.0, max=2.0, step=0.01, initial_value=0.35)
        preview_scale = server.gui.add_slider(
            "Preview Scale",
            min=0.25,
            max=1.0,
            step=0.05,
            initial_value=float(np.clip(args.viewer_preview_scale, 0.25, 1.0)),
        )
        save_button = server.gui.add_button("Save Current Frame")
        status = server.gui.add_markdown("`Waiting for first render...`")

    with server.gui.add_folder("Switches"):
        enable_underwater = server.gui.add_checkbox("Enable Underwater", initial_value=bool(uw_args.enable_underwater))
        enable_surface = server.gui.add_checkbox("Enable Surface", initial_value=bool(uw_args.enable_surface))
        enable_caustic = server.gui.add_checkbox("Enable Caustic", initial_value=bool(uw_args.enable_caustic))
        enable_beam = server.gui.add_checkbox("Enable Light Beam", initial_value=bool(getattr(uw_args, "enable_beam", True)))
        submerged_only = server.gui.add_checkbox("Submerged Only", initial_value=bool(uw_args.submerged_only))

    with server.gui.add_folder("Waterline"):
        water_level_mode = server.gui.add_dropdown(
            "Water Level Mode",
            options=["scene_relative", "absolute"],
            initial_value=str(uw_args.caustic_water_level_mode),
        )
        water_level_offset = server.gui.add_number("Water Level Offset", initial_value=float(uw_args.caustic_water_level_offset), step=0.05)
        water_level_world = server.gui.add_number("Water Level World", initial_value=float(uw_args.caustic_water_level_world), step=0.05)
        water_level_quantile = server.gui.add_slider(
            "Water Level Quantile (0-1)",
            min=0.0,
            max=1.0,
            step=0.001,
            initial_value=norm_value("water_level_quantile", float(uw_args.water_level_quantile)),
        )
        waterline_softness = server.gui.add_slider(
            "Waterline Softness (0-1)",
            min=0.0,
            max=1.0,
            step=0.001,
            initial_value=norm_value("waterline_softness", float(uw_args.waterline_softness)),
        )

    with server.gui.add_folder("Water Surface"):
        surface_dominant_angle = server.gui.add_slider(
            "Dominant Angle (0-1)",
            min=0.0,
            max=1.0,
            step=0.001,
            initial_value=norm_value("surface_dominant_angle", float(uw_args.surface_dominant_angle)),
        )
        surface_directional_spread = server.gui.add_slider(
            "Directional Spread (0-1)",
            min=0.0,
            max=1.0,
            step=0.001,
            initial_value=norm_value("surface_directional_spread", float(uw_args.surface_directional_spread)),
        )
        surface_detail_strength = server.gui.add_slider(
            "Detail Strength (0-1)",
            min=0.0,
            max=1.0,
            step=0.001,
            initial_value=norm_value("surface_detail_strength", float(uw_args.surface_detail_strength)),
        )
        surface_tide_amplitude = server.gui.add_slider(
            "Tide Amplitude (0-1)",
            min=0.0,
            max=1.0,
            step=0.001,
            initial_value=norm_value("surface_tide_amplitude", float(uw_args.surface_tide_amplitude)),
        )
        surface_tide_speed = server.gui.add_slider(
            "Tide Speed (0-1)",
            min=0.0,
            max=1.0,
            step=0.001,
            initial_value=norm_value("surface_tide_speed", float(uw_args.surface_tide_speed)),
        )

    with server.gui.add_folder("Caustic"):
        caustic_lambda = server.gui.add_slider(
            "Caustic Lambda (0-1)",
            min=0.0,
            max=1.0,
            step=0.001,
            initial_value=norm_value("caustic_lambda", float(uw_args.caustic_lambda)),
        )
        caustic_max_pattern = server.gui.add_slider(
            "Caustic Max Pattern (0-1)",
            min=0.0,
            max=1.0,
            step=0.001,
            initial_value=norm_value("caustic_max_pattern", float(uw_args.caustic_max_pattern)),
        )

    with server.gui.add_folder("Medium"):
        medium_water_type = server.gui.add_dropdown(
            "Water Type",
            options=["I", "IA", "IB", "II", "III", "1C", "3C", "5C", "7C", "9C"],
            initial_value=str(uw_args.medium_jerlov_type).upper(),
        )
        medium_density_scale = server.gui.add_slider(
            "Water Density (0-1)",
            min=0.0,
            max=1.0,
            step=0.001,
            initial_value=norm_value("medium_density_scale", 1.0),
        )
        medium_backscatter_scale = server.gui.add_slider(
            "Backscatter Strength (0-1)",
            min=0.0,
            max=1.0,
            step=0.001,
            initial_value=norm_value("medium_backscatter_scale", 1.0),
        )
        medium_tone = server.gui.add_slider(
            "Water Tone Green->Blue (0-1)",
            min=0.0,
            max=1.0,
            step=0.001,
            initial_value=norm_value("medium_tone", 0.0),
        )
        medium_variation = server.gui.add_slider(
            "Spatial Variation (0-1)",
            min=0.0,
            max=1.0,
            step=0.001,
            initial_value=norm_value("medium_variation", float(uw_args.medium_beta_strength)),
        )
        medium_depth_scale = server.gui.add_slider(
            "Medium Depth Scale (0-1)",
            min=0.0,
            max=1.0,
            step=0.001,
            initial_value=norm_value("medium_depth_scale", float(uw_args.medium_depth_scale)),
        )

    current_image = np.zeros((int(views[0].image_height), int(views[0].image_width), 3), dtype=np.uint8)
    image_handle = server.gui.add_image(current_image, label="Render")

    controls = {
        "enable_underwater": enable_underwater,
        "enable_surface": enable_surface,
        "enable_caustic": enable_caustic,
        "enable_beam": enable_beam,
        "submerged_only": submerged_only,
        "water_level_mode": water_level_mode,
        "water_level_offset": water_level_offset,
        "water_level_world": water_level_world,
        "water_level_quantile": water_level_quantile,
        "waterline_softness": waterline_softness,
        "surface_dominant_angle": surface_dominant_angle,
        "surface_directional_spread": surface_directional_spread,
        "surface_detail_strength": surface_detail_strength,
        "surface_tide_amplitude": surface_tide_amplitude,
        "surface_tide_speed": surface_tide_speed,
        "caustic_lambda": caustic_lambda,
        "caustic_max_pattern": caustic_max_pattern,
        "medium_water_type": medium_water_type,
        "medium_density_scale": medium_density_scale,
        "medium_backscatter_scale": medium_backscatter_scale,
        "medium_tone": medium_tone,
        "medium_variation": medium_variation,
        "medium_depth_scale": medium_depth_scale,
    }

    dirty = {"flag": True}
    saved = {"counter": 0}
    current_rgb = {"image": current_image}
    client_state = {
        "client": None,
        "last_move_time": 0.0,
        "is_moving": False,
    }

    def mark_dirty(_=None):
        dirty["flag"] = True

    @server.on_client_connect
    def _(client):
        client_state["client"] = client
        client_state["last_move_time"] = time.perf_counter()
        client_state["is_moving"] = True
        if bool(use_free_camera.value):
            view_idx = int(np.clip(int(camera_index.value), 0, len(views) - 1))
            set_client_camera_from_dataset_view(client, views[view_idx])
        mark_dirty()

        @client.camera.on_update
        def _(_cam):
            client_state["last_move_time"] = time.perf_counter()
            client_state["is_moving"] = True
            mark_dirty()

    for handle in [
        display_key,
        viewport_bg,
        use_free_camera,
        camera_index,
        res_moving,
        res_static,
        move_settle,
        jpeg_quality,
        time_value,
        preview_scale,
        enable_underwater,
        enable_surface,
        enable_caustic,
        enable_beam,
        submerged_only,
        water_level_mode,
        water_level_offset,
        water_level_world,
        water_level_quantile,
        waterline_softness,
        surface_dominant_angle,
        surface_directional_spread,
        surface_detail_strength,
        surface_tide_amplitude,
        surface_tide_speed,
        caustic_lambda,
        caustic_max_pattern,
        medium_water_type,
        medium_density_scale,
        medium_backscatter_scale,
        medium_tone,
        medium_variation,
        medium_depth_scale,
    ]:
        handle.on_update(mark_dirty)

    @save_button.on_click
    def _(_event):
        output_dir = args.save_dir or os.path.join(dataset.model_path, "viser_snapshots")
        os.makedirs(output_dir, exist_ok=True)
        view_idx = int(np.clip(int(camera_index.value), 0, len(views) - 1))
        filename = "{:04d}_view{:03d}_t{:.3f}_{}.png".format(
            saved["counter"],
            view_idx,
            float(time_value.value),
            display_key.value,
        )
        path = os.path.join(output_dir, filename)
        cv2.imwrite(path, current_rgb["image"][:, :, ::-1])
        saved["counter"] += 1
        status.content = "`Saved: {}`".format(path)

    @snap_camera_button.on_click
    def _(_event):
        client = client_state["client"]
        if client is None:
            status.content = "`No active browser client connected.`"
            return
        view_idx = int(np.clip(int(camera_index.value), 0, len(views) - 1))
        set_client_camera_from_dataset_view(client, views[view_idx])
        client_state["last_move_time"] = time.perf_counter()
        client_state["is_moving"] = True
        mark_dirty()

    frame_dt = 1.0 / max(float(args.viewer_target_fps), 1.0)
    last_tick = time.perf_counter()

    try:
        while True:
            now = time.perf_counter()
            dt = now - last_tick
            last_tick = now

            if bool(time_play.value):
                time_value.value = float(time_value.value) + float(time_speed.value) * dt
                dirty["flag"] = True

            if bool(use_free_camera.value) and client_state["client"] is not None and bool(client_state["is_moving"]):
                settle_seconds = max(0.01, float(move_settle.value))
                if (time.perf_counter() - float(client_state["last_move_time"])) >= settle_seconds:
                    dirty["flag"] = True

            if dirty["flag"]:
                dirty["flag"] = False
                apply_live_controls(underwater_renderer, controls)
                view_idx = int(np.clip(int(camera_index.value), 0, len(views) - 1))
                render_tag = "dataset_view={:03d}".format(view_idx)
                moving_flag = False
                client = client_state["client"]
                if bool(use_free_camera.value) and client is not None:
                    settle_seconds = max(0.01, float(move_settle.value))
                    elapsed_move = time.perf_counter() - float(client_state["last_move_time"])
                    moving_flag = bool(client_state["is_moving"]) and elapsed_move < settle_seconds
                    target_res = int(res_moving.value if moving_flag else res_static.value)
                    target_res = max(256, min(2560, target_res))
                    view = build_runtime_camera_from_viser(views[view_idx], client.camera, target_res)
                    render_tag = "freecam res={}".format(target_res)
                    if not moving_flag and bool(client_state["is_moving"]):
                        client_state["is_moving"] = False
                else:
                    view = views[view_idx]
                outputs = render_current(
                    backbone,
                    underwater_renderer,
                    gaussians,
                    pipe,
                    background,
                    app_model,
                    view,
                    float(time_value.value),
                )
                frame_rgb = tensor_to_display(outputs, str(display_key.value))
                scale = float(np.clip(preview_scale.value, 0.25, 1.0))
                if scale < 0.999:
                    h, w = frame_rgb.shape[:2]
                    frame_rgb = cv2.resize(
                        frame_rgb,
                        (max(1, int(w * scale)), max(1, int(h * scale))),
                        interpolation=cv2.INTER_AREA,
                    )
                image_handle.image = frame_rgb
                if bool(viewport_bg.value):
                    server.scene.set_background_image(
                        frame_rgb,
                        format="jpeg",
                        jpeg_quality=int(jpeg_quality.value),
                    )
                else:
                    server.scene.set_background_image(None)
                current_rgb["image"] = frame_rgb
                water_level_text = "n/a"
                if "water_level" in outputs:
                    water_level_text = "{:.3f}".format(float(outputs["water_level"].mean().item()))
                status.content = "`{}  moving={}  time={:.3f}  map={}  water_level={}`".format(
                    render_tag,
                    int(moving_flag),
                    float(time_value.value),
                    display_key.value,
                    water_level_text,
                )

            elapsed = time.perf_counter() - now
            if elapsed < frame_dt:
                time.sleep(frame_dt - elapsed)
    except KeyboardInterrupt:
        print("\n[Viewer] Stopped.")


if __name__ == "__main__":
    main()

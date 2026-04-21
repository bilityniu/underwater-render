import os
import sys
import copy
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arguments import ModelParams, PipelineParams, UnderwaterParams, get_combined_args
from models import SceneBackboneRenderer
from scene import GaussianModel, Scene
from utils.graphics_utils import getProjectionMatrix
from utils.general_utils import safe_state
from utils.underwater_utils import build_underwater_renderer, compute_scene_bounds_from_gaussians, load_underwater_checkpoint
from utils.visualization import dump_underwater_visualizations


def resolve_video_views(scene, uw_args):
    if uw_args.video_view == "train":
        views = scene.getTrainCameras()
    else:
        views = scene.getTestCameras()
    if len(views) == 0:
        views = scene.getTrainCameras()
    return views


def build_video_schedule(views, uw_args):
    if len(views) == 0:
        raise ValueError("No cameras available for video rendering.")

    mode = uw_args.video_mode
    frame_count = int(uw_args.video_frames)
    if mode == "camera_path":
        schedule = build_interpolated_schedule(views, frame_count, uw_args.video_camera_cycles)
        output_name = "camera_path"
    else:
        camera_index = min(max(0, int(uw_args.video_camera_index)), len(views) - 1)
        fixed_view = views[camera_index]
        schedule = [fixed_view for _ in range(frame_count)]
        output_name = fixed_view.image_name
    return schedule, output_name


def build_interpolated_schedule(views, frame_count, camera_cycles):
    if len(views) == 1:
        return [views[0] for _ in range(frame_count)]
    schedule = []
    num_views = len(views)
    for frame_idx in range(frame_count):
        frame_progress = frame_idx / max(1, frame_count - 1)
        progress = frame_progress * float(camera_cycles) * num_views
        left_idx = int(np.floor(progress)) % num_views
        right_idx = (left_idx + 1) % num_views
        alpha = float(progress - np.floor(progress))
        schedule.append(interpolate_camera(views[left_idx], views[right_idx], alpha, frame_idx))
    return schedule


def interpolate_camera(cam_a, cam_b, alpha, frame_idx):
    alpha = float(np.clip(alpha, 0.0, 1.0))
    if alpha <= 1e-6:
        return cam_a

    cam = copy.copy(cam_a)
    c2w_a = camera_to_c2w(cam_a)
    c2w_b = camera_to_c2w(cam_b)

    rot_a = c2w_a[:3, :3]
    rot_b = c2w_b[:3, :3]
    trans_a = c2w_a[:3, 3]
    trans_b = c2w_b[:3, 3]

    rot_interp = orthonormalize_rotation((1.0 - alpha) * rot_a + alpha * rot_b)
    trans_interp = (1.0 - alpha) * trans_a + alpha * trans_b

    c2w_interp = np.eye(4, dtype=np.float32)
    c2w_interp[:3, :3] = rot_interp
    c2w_interp[:3, 3] = trans_interp
    w2c_interp = np.linalg.inv(c2w_interp).astype(np.float32)

    cam.R = w2c_interp[:3, :3].T
    cam.T = w2c_interp[:3, 3]
    cam.image_name = f"camera_path_{frame_idx:04d}"
    cam.world_view_transform = torch.tensor(w2c_interp, device="cuda", dtype=torch.float32).transpose(0, 1)
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


def camera_to_c2w(camera):
    w2c = camera.world_view_transform.transpose(0, 1).detach().cpu().numpy()
    return np.linalg.inv(w2c).astype(np.float32)


def orthonormalize_rotation(rotation_matrix):
    u, _, vh = np.linalg.svd(rotation_matrix, full_matrices=False)
    r = u @ vh
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1.0
        r = u @ vh
    return r.astype(np.float32)


def compute_world_time(frame_idx, uw_args):
    if uw_args.video_time_mode == "fixed":
        return uw_args.video_time_start
    frame_progress = frame_idx / max(1, int(uw_args.video_frames) - 1)
    return uw_args.video_time_start + uw_args.video_time_scale * frame_progress * uw_args.video_time_duration


@torch.no_grad()
def render_video(dataset, iteration, pipeline, uw_args, underwater_checkpoint):
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    app_model = None

    scene_bounds = compute_scene_bounds_from_gaussians(gaussians, padding_ratio=uw_args.scene_bounds_padding)
    underwater_renderer = build_underwater_renderer(uw_args, scene_bounds=scene_bounds).cuda()
    _, checkpoint_path = load_underwater_checkpoint(
        dataset.model_path,
        underwater_renderer,
        checkpoint_path=underwater_checkpoint,
        strict=False,
    )
    print(f"Using underwater checkpoint: {checkpoint_path}")

    views = resolve_video_views(scene, uw_args)
    schedule, output_stem = build_video_schedule(views, uw_args)
    first_view = schedule[0]
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    backbone = SceneBackboneRenderer()

    output_dir = os.path.join(dataset.model_path, "video_underwater")
    frames_dir = os.path.join(output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{output_stem}.mp4")

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(uw_args.video_fps),
        (int(first_view.image_width), int(first_view.image_height)),
    )

    for frame_idx, view in enumerate(tqdm(schedule, desc="Rendering underwater video")):
        time_value = compute_world_time(frame_idx, uw_args)
        scene_outputs = backbone(
            view,
            gaussians,
            pipeline,
            background,
            app_model=app_model,
            require_geometry=True,
            prefer_app_image=not uw_args.disable_pgsr_app_model,
        )
        outputs = underwater_renderer(scene_outputs, view, time_value)
        frame = outputs["final_rgb"].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
        frame_bgr = (frame[:, :, ::-1] * 255.0).astype("uint8")
        writer.write(frame_bgr)
        cv2.imwrite(os.path.join(frames_dir, f"{frame_idx:04d}.png"), frame_bgr)
        if uw_args.debug_dump_intermediate and frame_idx % max(1, uw_args.video_fps) == 0:
            dump_underwater_visualizations(output_dir, f"frame_{frame_idx:04d}", outputs)

    writer.release()
    print(f"Saved underwater video to {output_path}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Render underwater temporal video")
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)
    underwater = UnderwaterParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--underwater_checkpoint", type=str, default=None)

    args = get_combined_args(parser)
    safe_state(args.quiet)
    render_video(
        model.extract(args),
        args.iteration,
        pipeline.extract(args),
        underwater.extract(args),
        args.underwater_checkpoint,
    )

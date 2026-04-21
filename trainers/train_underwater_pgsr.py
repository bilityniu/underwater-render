import os
import random
import shutil
import sys
import uuid
from argparse import ArgumentParser, Namespace
from pathlib import Path
from random import randint

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arguments import ModelParams, OptimizationParams, PipelineParams, UnderwaterParams
from models import SceneBackboneRenderer
from scene import GaussianModel, Scene
from utils.general_utils import safe_state
from utils.loss_utils import (
    spatial_smoothness_loss,
    temporal_smoothness_loss,
    underwater_reconstruction_loss,
    vector_field_smoothness_loss,
    weak_caustic_regularization,
)
from utils.underwater_utils import (
    build_underwater_renderer,
    compute_scene_bounds_from_gaussians,
    load_underwater_checkpoint,
    save_underwater_checkpoint,
)
from utils.visualization import dump_underwater_visualizations

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def prepare_output_and_logger(args):
    if not args.model_path:
        unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args_underwater"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # TensorBoard event writer can be unstable in some envs (crc32 thread crash).
    # Keep training stable by default; opt-in via env var.
    if os.environ.get("PGSR_ENABLE_TENSORBOARD", "0") != "1":
        print("TensorBoard logging is disabled by default for stability. Set PGSR_ENABLE_TENSORBOARD=1 to enable.")
        return None

    if not TENSORBOARD_FOUND:
        return None
    return SummaryWriter(os.path.join(args.model_path, "tensorboard_underwater"))


def disable_tb_writer(tb_writer, error):
    print(f"[Warn] Disabling TensorBoard logging due to writer error: {error}")
    try:
        tb_writer.close()
    except Exception:
        pass
    return None


def backup_underwater_code(model_path):
    backup_root = os.path.join(model_path, "code_underwater")
    os.makedirs(backup_root, exist_ok=True)
    for rel_path in ["models", "trainers", "utils", "arguments", "render.py"]:
        src = os.path.join(".", rel_path)
        dst = os.path.join(backup_root, os.path.basename(rel_path))
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        elif os.path.isfile(src):
            shutil.copy2(src, dst)


def freeze_stage1_modules(gaussians, app_model, uw_args):
    geometry_params = [gaussians._xyz, gaussians._rotation, gaussians._scaling]
    if uw_args.freeze_geometry:
        for param in geometry_params:
            param.requires_grad_(False)
    if uw_args.freeze_opacity:
        gaussians._opacity.requires_grad_(False)
    if uw_args.freeze_sh:
        gaussians._features_dc.requires_grad_(False)
        gaussians._features_rest.requires_grad_(False)
    if uw_args.freeze_app_model and app_model is not None:
        for param in app_model.parameters():
            param.requires_grad_(False)


def collect_trainable_parameters(gaussians, app_model, underwater_renderer, uw_args):
    underwater_params = []
    surface_params = []
    for name, param in underwater_renderer.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("surface_module."):
            surface_params.append(param)
        else:
            underwater_params.append(param)

    params = []
    if underwater_params:
        params.append({"params": underwater_params, "lr": uw_args.underwater_lr, "name": "underwater"})
    if surface_params:
        params.append(
            {
                "params": surface_params,
                "lr": uw_args.underwater_lr * uw_args.surface_lr_scale,
                "name": "surface",
            }
        )
    if not uw_args.freeze_sh:
        params.append(
            {
                "params": [gaussians._features_dc, gaussians._features_rest],
                "lr": uw_args.underwater_lr * 0.1,
                "name": "scene_color",
            }
        )
    if not uw_args.freeze_app_model and app_model is not None:
        params.append({"params": list(app_model.parameters()), "lr": uw_args.underwater_lr * 0.1, "name": "app_model"})
    return params


def build_proxy_target(scene_outputs, underwater_renderer):
    scene_rgb_raw = scene_outputs["scene_rgb_raw"]
    scene_depth = scene_outputs["scene_depth"]
    visible_mask = scene_outputs.get("surface_valid_mask", (scene_depth > 0).float()) * (scene_depth > 0).float()

    proxy_depth = scene_depth
    if underwater_renderer.submerged_only:
        background_depth = underwater_renderer.background_depth.to(device=scene_depth.device, dtype=scene_depth.dtype).view(1, 1, 1)
        proxy_depth = torch.where(visible_mask > 0.5, scene_depth, torch.ones_like(scene_depth) * background_depth)

    if underwater_renderer.medium_module is None:
        return scene_rgb_raw.detach()

    with torch.no_grad():
        proxy_outputs = underwater_renderer.medium_module(
            scene_rgb_raw,
            proxy_depth,
            surface_points_world=None,
            valid_mask=None,
        )
    return proxy_outputs["underwater_rgb"].detach()


def build_proxy_lit_target(scene_outputs, underwater_renderer, outputs):
    proxy_underwater = build_proxy_target(scene_outputs, underwater_renderer)

    direct_proxy = outputs.get("direct_underwater_rgb")
    if direct_proxy is None:
        direct_proxy = proxy_underwater
    direct_proxy = direct_proxy.detach()

    backscatter = outputs.get("backscatter")
    if backscatter is None:
        backscatter = torch.clamp(proxy_underwater - direct_proxy, min=0.0)
    backscatter = backscatter.detach()

    unified_gain = outputs.get("caustic_gain")
    if unified_gain is None:
        unified_gain = torch.ones_like(proxy_underwater[:, :1])
    unified_gain = unified_gain.detach()
    proxy_base = torch.clamp(direct_proxy * unified_gain + backscatter, 0.0, 1.0)

    beam_rgb = outputs.get("light_beam_rgb")
    if beam_rgb is None:
        beam_rgb = torch.zeros_like(proxy_underwater)
    beam_rgb = beam_rgb.detach()
    proxy_lit = torch.clamp(proxy_base + beam_rgb, 0.0, 1.0)
    return proxy_lit


@torch.no_grad()
def render_previous_caustic_only(scene_outputs, underwater_renderer, previous_time):
    scene_depth = scene_outputs["scene_depth"]
    scene_normal = scene_outputs["scene_normal"]
    surface_points_world = scene_outputs["surface_points_world"]
    visible_mask = scene_outputs.get("surface_valid_mask", (scene_depth > 0).float()) * (scene_depth > 0).float()

    if not underwater_renderer.enable_caustic or underwater_renderer.caustic_module is None:
        return {"caustic": torch.zeros_like(scene_depth)}

    surface_outputs = {
        "surface_height": torch.zeros_like(scene_depth),
        "surface_normal": torch.zeros_like(scene_normal),
        "surface_gradient": torch.zeros(
            (2, scene_depth.shape[-2], scene_depth.shape[-1]),
            device=scene_depth.device,
            dtype=scene_depth.dtype,
        ),
        "surface_gradient_magnitude": torch.zeros_like(scene_depth),
        "surface_curvature": torch.zeros_like(scene_depth),
    }
    if underwater_renderer.enable_surface and underwater_renderer.surface_module is not None:
        surface_outputs = underwater_renderer.surface_module.sample_world(
            surface_points_world[underwater_renderer.lateral_axes[0]],
            surface_points_world[underwater_renderer.lateral_axes[1]],
            previous_time,
        )

    caustic_outputs = underwater_renderer.caustic_module(
        surface_points_world,
        previous_time,
        scene_normal,
        water_outputs=surface_outputs,
        visible_mask=visible_mask,
        surface_sampler=underwater_renderer.surface_module,
    )
    return {"caustic": caustic_outputs["caustic"]}


def underwater_training(dataset, opt, pipe, uw_args, test_iterations, save_iterations, checkpoint_path):
    tb_writer = prepare_output_and_logger(dataset)
    backup_underwater_code(dataset.model_path)

    gaussians = GaussianModel(dataset.sh_degree)
    point_cloud_root = os.path.join(dataset.model_path, "point_cloud")
    load_iteration = -1 if os.path.isdir(point_cloud_root) else None
    if load_iteration is None:
        print("No stage-1 PGSR checkpoint found under model_path/point_cloud, falling back to fresh scene initialization for smoke testing.")
    scene = Scene(dataset, gaussians, load_iteration=load_iteration, shuffle=True)
    gaussians.training_setup(opt)

    app_model = None

    freeze_stage1_modules(gaussians, app_model, uw_args)

    backbone = SceneBackboneRenderer()
    scene_bounds = compute_scene_bounds_from_gaussians(gaussians, padding_ratio=uw_args.scene_bounds_padding)
    underwater_renderer = build_underwater_renderer(uw_args, scene_bounds=scene_bounds).cuda()
    optimizer = torch.optim.Adam(
        collect_trainable_parameters(gaussians, app_model, underwater_renderer, uw_args),
        betas=(0.9, 0.99),
    )

    first_iter = 1
    if checkpoint_path:
        first_iter, resolved = load_underwater_checkpoint(
            dataset.model_path,
            underwater_renderer,
            optimizer,
            checkpoint_path,
            strict=False,
        )
        first_iter += 1
        print(f"Loaded underwater checkpoint: {resolved}")

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    debug_dir = os.path.join(scene.model_path, "debug_underwater")
    os.makedirs(debug_dir, exist_ok=True)
    viewpoint_stack = None
    progress_bar = tqdm(range(first_iter, uw_args.underwater_iterations + 1), desc="Underwater training")

    for iteration in progress_bar:
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        gt_image = None
        if uw_args.underwater_loss_mode == "dry_gt":
            gt_image, _ = viewpoint_cam.get_image()

        bg = torch.rand((3), device="cuda") if opt.random_background else background
        scene_outputs = backbone(
            viewpoint_cam,
            gaussians,
            pipe,
            bg,
            app_model=app_model,
            require_geometry=True,
            prefer_app_image=not uw_args.disable_pgsr_app_model,
        )

        current_time = iteration * uw_args.time_delta * uw_args.time_stride
        previous_time = current_time - uw_args.time_delta

        outputs = underwater_renderer(scene_outputs, viewpoint_cam, current_time)
        previous_outputs = render_previous_caustic_only(scene_outputs, underwater_renderer, previous_time)

        if uw_args.underwater_loss_mode == "dry_gt":
            rec_target = gt_image
            rec_source = outputs["final_rgb"]
        else:
            rec_target = build_proxy_lit_target(scene_outputs, underwater_renderer, outputs)
            rec_source = outputs.get("underwater_lit", outputs["underwater_rgb"])
        rec_loss, rec_terms = underwater_reconstruction_loss(
            rec_source,
            rec_target,
            lambda_dssim=opt.lambda_dssim,
            use_ssim=uw_args.use_ssim,
        )
        temporal_loss = (0.5 * uw_args.temporal_smoothness_weight) * temporal_smoothness_loss(
            outputs["caustic"], previous_outputs["caustic"]
        )
        if getattr(uw_args, "surface_learnable", False):
            surface_loss = uw_args.surface_smoothness_weight * spatial_smoothness_loss(outputs["surface_height"])
        else:
            surface_loss = outputs["scene_depth"].new_zeros(())
        caustic_reg = uw_args.caustic_regularization_weight * weak_caustic_regularization(outputs["caustic"])
        medium_flow_loss = uw_args.medium_flow_smoothness_weight * vector_field_smoothness_loss(outputs.get("medium_flow_velocity"))
        medium_motion_loss = uw_args.medium_motion_magnitude_weight * outputs.get("medium_motion_magnitude", outputs["scene_depth"].new_zeros(())).mean()

        total_loss = rec_loss + temporal_loss + surface_loss + caustic_reg + medium_flow_loss + medium_motion_loss

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()

        progress_bar.set_postfix(
            {
                "rec": f"{rec_loss.item():.4f}",
                "temp": f"{temporal_loss.item():.4f}",
                "surf": f"{surface_loss.item():.4f}",
                "creg": f"{caustic_reg.item():.4f}",
                "mflow": f"{medium_flow_loss.item():.4f}",
            }
        )

        if tb_writer is not None:
            try:
                tb_writer.add_scalar("underwater/total_loss", total_loss.item(), iteration)
                tb_writer.add_scalar("underwater/reconstruction", rec_loss.item(), iteration)
                tb_writer.add_scalar("underwater/l1", rec_terms["l1"].item(), iteration)
                tb_writer.add_scalar("underwater/ssim", rec_terms["ssim"].item(), iteration)
                tb_writer.add_scalar("underwater/temporal", temporal_loss.item(), iteration)
                tb_writer.add_scalar("underwater/surface", surface_loss.item(), iteration)
                tb_writer.add_scalar("underwater/caustic_reg", caustic_reg.item(), iteration)
                tb_writer.add_scalar("underwater/medium_flow", medium_flow_loss.item(), iteration)
                tb_writer.add_scalar("underwater/medium_motion", medium_motion_loss.item(), iteration)
                if iteration % max(1, uw_args.debug_dump_interval) == 0:
                    tb_writer.add_images(f"underwater/{viewpoint_cam.image_name}/final", outputs["final_rgb"][None], iteration)
                    tb_writer.add_images(f"underwater/{viewpoint_cam.image_name}/scene", outputs["scene_rgb_raw"][None], iteration)
            except Exception as tb_error:
                tb_writer = disable_tb_writer(tb_writer, tb_error)

        if uw_args.debug_dump_intermediate and iteration % max(1, uw_args.debug_dump_interval) == 0:
            dump_underwater_visualizations(debug_dir, f"{iteration:05d}_{viewpoint_cam.image_name}", outputs)

        if iteration in test_iterations:
            tb_writer = evaluate_underwater(
                tb_writer,
                iteration,
                scene,
                gaussians,
                pipe,
                background,
                app_model,
                backbone,
                underwater_renderer,
                uw_args,
            )

        if iteration in save_iterations:
            save_underwater_checkpoint(dataset.model_path, iteration, underwater_renderer, optimizer)

    save_underwater_checkpoint(dataset.model_path, uw_args.underwater_iterations, underwater_renderer, optimizer)


@torch.no_grad()
def evaluate_underwater(tb_writer, iteration, scene, gaussians, pipe, background, app_model, backbone, underwater_renderer, uw_args):
    validation_views = scene.getTestCameras() if len(scene.getTestCameras()) > 0 else scene.getTrainCameras()[:5]
    total_l1 = 0.0
    for index, view in enumerate(validation_views[:5]):
        scene_outputs = backbone(
            view,
            gaussians,
            pipe,
            background,
            app_model=app_model,
            require_geometry=True,
            prefer_app_image=not uw_args.disable_pgsr_app_model,
        )
        outputs = underwater_renderer(scene_outputs, view, iteration * uw_args.time_delta * uw_args.time_stride)
        if uw_args.underwater_loss_mode == "dry_gt":
            gt_image, _ = view.get_image()
            eval_target = gt_image
            eval_source = outputs["final_rgb"]
        else:
            eval_target = build_proxy_lit_target(scene_outputs, underwater_renderer, outputs)
            eval_source = outputs.get("underwater_lit", outputs["underwater_rgb"])
        l1_value = torch.abs(eval_source - eval_target).mean().item()
        total_l1 += l1_value
        if tb_writer is not None:
            try:
                tb_writer.add_images(f"eval/{view.image_name}/final", outputs["final_rgb"][None], iteration)
                tb_writer.add_images(f"eval/{view.image_name}/caustic", outputs["caustic"][None], iteration)
            except Exception as tb_error:
                tb_writer = disable_tb_writer(tb_writer, tb_error)
    total_l1 /= max(1, min(5, len(validation_views)))
    if tb_writer is not None:
        try:
            tb_writer.add_scalar("eval/l1", total_l1, iteration)
        except Exception as tb_error:
            tb_writer = disable_tb_writer(tb_writer, tb_error)
    print(f"[ITER {iteration}] Underwater eval L1 {total_l1:.6f}")
    return tb_writer


if __name__ == "__main__":
    parser = ArgumentParser(description="Underwater PGSR training parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    uw = UnderwaterParams(parser)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--underwater_checkpoint", type=str, default=None)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1000, 5000, 10000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1000, 5000, 10000])
    args = parser.parse_args(sys.argv[1:])

    safe_state(args.quiet)
    underwater_training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        uw.extract(args),
        args.test_iterations,
        args.save_iterations,
        args.underwater_checkpoint,
    )

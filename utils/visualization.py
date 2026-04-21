import os

import cv2
import numpy as np
import torch
import torchvision


def _to_numpy_chw(tensor):
    if tensor is None:
        return None
    if torch.is_tensor(tensor):
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)
        return tensor.detach().float().cpu().numpy()
    array = np.asarray(tensor, dtype=np.float32)
    if array.ndim == 2:
        array = array[None]
    return array


def colorize_depth(depth):
    depth_np = _to_numpy_chw(depth)[0]
    valid = np.isfinite(depth_np)
    if not valid.any():
        return np.zeros((depth_np.shape[0], depth_np.shape[1], 3), dtype=np.uint8)
    depth_min = depth_np[valid].min()
    depth_max = depth_np[valid].max()
    depth_norm = (depth_np - depth_min) / (depth_max - depth_min + 1e-8)
    depth_u8 = np.clip(depth_norm * 255.0, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(depth_u8, cv2.COLORMAP_OCEAN)


def colorize_scalar_map(tensor, colormap=cv2.COLORMAP_JET):
    value = _to_numpy_chw(tensor)
    if value is None:
        value = np.zeros((1, 1), dtype=np.float32)
    elif value.ndim == 0:
        value = np.array([[float(value)]], dtype=np.float32)
    elif value.ndim == 1:
        value = value[None, :]
    elif value.ndim >= 3:
        if value.shape[0] in (1, 3):
            value = value[0]
        else:
            value = value[..., 0]
    value = value - value.min()
    value = value / (value.max() + 1e-8)
    value_u8 = np.ascontiguousarray(np.clip(value * 255.0, 0, 255).astype(np.uint8))
    return cv2.applyColorMap(value_u8, colormap)


def colorize_normal(normal):
    normal_np = _to_numpy_chw(normal).transpose(1, 2, 0)
    normal_np = normal_np / (np.linalg.norm(normal_np, axis=-1, keepdims=True) + 1e-8)
    return np.clip((normal_np + 1.0) * 127.5, 0, 255).astype(np.uint8)


def save_rgb(path, image):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torchvision.utils.save_image(torch.clamp(image, 0.0, 1.0), path)


def save_bgr(path, image_bgr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, image_bgr)


def dump_underwater_visualizations(output_dir, stem, outputs):
    os.makedirs(output_dir, exist_ok=True)
    if "scene_rgb" in outputs:
        save_rgb(os.path.join(output_dir, f"{stem}_scene_rgb.png"), outputs["scene_rgb"])
    if "scene_rgb_raw" in outputs:
        save_rgb(os.path.join(output_dir, f"{stem}_scene_rgb_raw.png"), outputs["scene_rgb_raw"])
    if "underwater_rgb" in outputs:
        save_rgb(os.path.join(output_dir, f"{stem}_underwater_rgb.png"), outputs["underwater_rgb"])
    if "light_beam_rgb" in outputs:
        save_rgb(os.path.join(output_dir, f"{stem}_light_beam_rgb.png"), outputs["light_beam_rgb"])
    if "final_rgb_pre_app" in outputs:
        save_rgb(os.path.join(output_dir, f"{stem}_final_rgb_pre_app.png"), outputs["final_rgb_pre_app"])
    if "final_rgb" in outputs:
        save_rgb(os.path.join(output_dir, f"{stem}_final_rgb.png"), outputs["final_rgb"])
    if "visible_interface_rgb" in outputs:
        save_rgb(os.path.join(output_dir, f"{stem}_visible_interface_rgb.png"), outputs["visible_interface_rgb"])
    if "scene_depth" in outputs:
        save_bgr(os.path.join(output_dir, f"{stem}_scene_depth.jpg"), colorize_depth(outputs["scene_depth"]))
    if "scene_normal" in outputs:
        save_bgr(os.path.join(output_dir, f"{stem}_scene_normal.jpg"), colorize_normal(outputs["scene_normal"]))
    if "transmission" in outputs:
        save_bgr(
            os.path.join(output_dir, f"{stem}_transmission.jpg"),
            colorize_scalar_map(outputs["transmission"].mean(dim=0, keepdim=True), cv2.COLORMAP_BONE),
        )
    if "transmission_back" in outputs:
        save_bgr(
            os.path.join(output_dir, f"{stem}_transmission_back.jpg"),
            colorize_scalar_map(outputs["transmission_back"].mean(dim=0, keepdim=True), cv2.COLORMAP_BONE),
        )
    if "backscatter" in outputs:
        save_bgr(
            os.path.join(output_dir, f"{stem}_backscatter.jpg"),
            colorize_scalar_map(outputs["backscatter"].mean(dim=0, keepdim=True), cv2.COLORMAP_OCEAN),
        )
    if "beta_map" in outputs:
        save_bgr(
            os.path.join(output_dir, f"{stem}_beta_map.jpg"),
            colorize_scalar_map(outputs["beta_map"].mean(dim=0, keepdim=True), cv2.COLORMAP_TURBO),
        )
    if "beta_d_map" in outputs:
        save_bgr(
            os.path.join(output_dir, f"{stem}_beta_d_map.jpg"),
            colorize_scalar_map(outputs["beta_d_map"].mean(dim=0, keepdim=True), cv2.COLORMAP_TURBO),
        )
    if "beta_b_map" in outputs:
        save_bgr(
            os.path.join(output_dir, f"{stem}_beta_b_map.jpg"),
            colorize_scalar_map(outputs["beta_b_map"].mean(dim=0, keepdim=True), cv2.COLORMAP_TURBO),
        )
    if "b_inf_map" in outputs:
        save_bgr(
            os.path.join(output_dir, f"{stem}_b_inf_map.jpg"),
            colorize_scalar_map(outputs["b_inf_map"].mean(dim=0, keepdim=True), cv2.COLORMAP_OCEAN),
        )
    if "medium_directionality" in outputs:
        save_bgr(
            os.path.join(output_dir, f"{stem}_medium_directionality.jpg"),
            colorize_scalar_map(outputs["medium_directionality"], cv2.COLORMAP_HOT),
        )
    if "medium_motion_magnitude" in outputs:
        save_bgr(
            os.path.join(output_dir, f"{stem}_medium_motion_magnitude.jpg"),
            colorize_scalar_map(outputs["medium_motion_magnitude"], cv2.COLORMAP_TURBO),
        )
    if "medium_flow_x" in outputs:
        save_bgr(os.path.join(output_dir, f"{stem}_medium_flow_x.jpg"), colorize_scalar_map(outputs["medium_flow_x"], cv2.COLORMAP_TURBO))
    if "medium_flow_y" in outputs:
        save_bgr(os.path.join(output_dir, f"{stem}_medium_flow_y.jpg"), colorize_scalar_map(outputs["medium_flow_y"], cv2.COLORMAP_TURBO))
    if "medium_flow_z" in outputs:
        save_bgr(os.path.join(output_dir, f"{stem}_medium_flow_z.jpg"), colorize_scalar_map(outputs["medium_flow_z"], cv2.COLORMAP_TURBO))
    if "advected_world_x" in outputs:
        save_bgr(os.path.join(output_dir, f"{stem}_advected_world_x.jpg"), colorize_scalar_map(outputs["advected_world_x"], cv2.COLORMAP_TURBO))
    if "advected_world_y" in outputs:
        save_bgr(os.path.join(output_dir, f"{stem}_advected_world_y.jpg"), colorize_scalar_map(outputs["advected_world_y"], cv2.COLORMAP_TURBO))
    if "advected_world_z" in outputs:
        save_bgr(os.path.join(output_dir, f"{stem}_advected_world_z.jpg"), colorize_scalar_map(outputs["advected_world_z"], cv2.COLORMAP_TURBO))
    if "caustic" in outputs:
        save_bgr(os.path.join(output_dir, f"{stem}_caustic.jpg"), colorize_scalar_map(outputs["caustic"], cv2.COLORMAP_HOT))
    if "caustic_pattern" in outputs:
        save_bgr(
            os.path.join(output_dir, f"{stem}_caustic_pattern_world.jpg"),
            colorize_scalar_map(outputs["caustic_pattern"], cv2.COLORMAP_HOT),
        )
    if "base_caustic_pattern" in outputs:
        save_bgr(
            os.path.join(output_dir, f"{stem}_base_caustic_pattern.jpg"),
            colorize_scalar_map(outputs["base_caustic_pattern"], cv2.COLORMAP_HOT),
        )
    if "footprint_detail_pattern" in outputs:
        save_bgr(
            os.path.join(output_dir, f"{stem}_footprint_detail_pattern.jpg"),
            colorize_scalar_map(outputs["footprint_detail_pattern"], cv2.COLORMAP_HOT),
        )
    if "detail_modulation" in outputs:
        save_bgr(
            os.path.join(output_dir, f"{stem}_detail_modulation.jpg"),
            colorize_scalar_map(outputs["detail_modulation"], cv2.COLORMAP_TURBO),
        )
    if "caustic_plane_map" in outputs:
        save_bgr(
            os.path.join(output_dir, f"{stem}_caustic_plane_map.jpg"),
            colorize_scalar_map(outputs["caustic_plane_map"], cv2.COLORMAP_HOT),
        )
    if "light_response" in outputs:
        save_bgr(
            os.path.join(output_dir, f"{stem}_light_response.jpg"),
            colorize_scalar_map(outputs["light_response"], cv2.COLORMAP_BONE),
        )
    if "surface_focus" in outputs:
        save_bgr(
            os.path.join(output_dir, f"{stem}_surface_focus.jpg"),
            colorize_scalar_map(outputs["surface_focus"], cv2.COLORMAP_BONE),
        )
    if "caustic_gain" in outputs:
        save_bgr(
            os.path.join(output_dir, f"{stem}_caustic_gain.jpg"),
            colorize_scalar_map(outputs["caustic_gain"], cv2.COLORMAP_TURBO),
        )
    if "light_beam" in outputs:
        save_bgr(
            os.path.join(output_dir, f"{stem}_light_beam.jpg"),
            colorize_scalar_map(outputs["light_beam"], cv2.COLORMAP_BONE),
        )
    if "light_beam_source" in outputs:
        save_bgr(
            os.path.join(output_dir, f"{stem}_light_beam_source.jpg"),
            colorize_scalar_map(outputs["light_beam_source"], cv2.COLORMAP_HOT),
        )
    if "light_beam_alignment" in outputs:
        save_bgr(
            os.path.join(output_dir, f"{stem}_light_beam_alignment.jpg"),
            colorize_scalar_map(outputs["light_beam_alignment"], cv2.COLORMAP_BONE),
        )
    if "light_beam_depth" in outputs:
        save_bgr(
            os.path.join(output_dir, f"{stem}_light_beam_depth.jpg"),
            colorize_scalar_map(outputs["light_beam_depth"], cv2.COLORMAP_OCEAN),
        )
    if "surface_height" in outputs:
        save_bgr(
            os.path.join(output_dir, f"{stem}_surface_height.jpg"),
            colorize_scalar_map(outputs["surface_height"], cv2.COLORMAP_TURBO),
        )
    if "surface_base_height" in outputs:
        save_bgr(
            os.path.join(output_dir, f"{stem}_surface_base_height.jpg"),
            colorize_scalar_map(outputs["surface_base_height"], cv2.COLORMAP_TURBO),
        )
    if "water_level" in outputs:
        save_bgr(
            os.path.join(output_dir, f"{stem}_water_level.jpg"),
            colorize_scalar_map(outputs["water_level"], cv2.COLORMAP_TURBO),
        )
    if "water_height_world" in outputs:
        save_bgr(
            os.path.join(output_dir, f"{stem}_water_height_world.jpg"),
            colorize_scalar_map(outputs["water_height_world"], cv2.COLORMAP_TURBO),
        )
    if "interface_distance" in outputs:
        save_bgr(
            os.path.join(output_dir, f"{stem}_interface_distance.jpg"),
            colorize_scalar_map(outputs["interface_distance"], cv2.COLORMAP_TURBO),
        )
    if "surface_interface_band" in outputs:
        save_bgr(
            os.path.join(output_dir, f"{stem}_surface_interface_band.jpg"),
            colorize_scalar_map(outputs["surface_interface_band"], cv2.COLORMAP_BONE),
        )
    if "ray_interface_mask" in outputs:
        save_bgr(
            os.path.join(output_dir, f"{stem}_ray_interface_mask.jpg"),
            colorize_scalar_map(outputs["ray_interface_mask"], cv2.COLORMAP_BONE),
        )
    if "visible_interface_alpha" in outputs:
        save_bgr(
            os.path.join(output_dir, f"{stem}_visible_interface_alpha.jpg"),
            colorize_scalar_map(outputs["visible_interface_alpha"], cv2.COLORMAP_BONE),
        )
    if "interface_fresnel" in outputs:
        save_bgr(
            os.path.join(output_dir, f"{stem}_interface_fresnel.jpg"),
            colorize_scalar_map(outputs["interface_fresnel"], cv2.COLORMAP_BONE),
        )
    if "ray_interface_distance" in outputs:
        save_bgr(
            os.path.join(output_dir, f"{stem}_ray_interface_distance.jpg"),
            colorize_scalar_map(outputs["ray_interface_distance"], cv2.COLORMAP_TURBO),
        )
    if "camera_water_distance" in outputs:
        save_bgr(
            os.path.join(output_dir, f"{stem}_camera_water_distance.jpg"),
            colorize_scalar_map(outputs["camera_water_distance"], cv2.COLORMAP_TURBO),
        )
    if "underwater_mask" in outputs:
        save_bgr(
            os.path.join(output_dir, f"{stem}_underwater_mask.jpg"),
            colorize_scalar_map(outputs["underwater_mask"], cv2.COLORMAP_BONE),
        )
    if "effective_depth" in outputs:
        save_bgr(
            os.path.join(output_dir, f"{stem}_effective_depth.jpg"),
            colorize_scalar_map(outputs["effective_depth"], cv2.COLORMAP_OCEAN),
        )
    if "surface_tide_height" in outputs:
        save_bgr(
            os.path.join(output_dir, f"{stem}_surface_tide_height.jpg"),
            colorize_scalar_map(outputs["surface_tide_height"], cv2.COLORMAP_TURBO),
        )
    if "surface_domain_warp" in outputs:
        save_bgr(
            os.path.join(output_dir, f"{stem}_surface_domain_warp.jpg"),
            colorize_scalar_map(outputs["surface_domain_warp"], cv2.COLORMAP_TURBO),
        )
    if "surface_local_variation" in outputs:
        save_bgr(
            os.path.join(output_dir, f"{stem}_surface_local_variation.jpg"),
            colorize_scalar_map(outputs["surface_local_variation"], cv2.COLORMAP_TURBO),
        )
    if "surface_wave_packet_energy" in outputs:
        save_bgr(
            os.path.join(output_dir, f"{stem}_surface_wave_packet_energy.jpg"),
            colorize_scalar_map(outputs["surface_wave_packet_energy"], cv2.COLORMAP_TURBO),
        )
    if "world_x" in outputs:
        save_bgr(os.path.join(output_dir, f"{stem}_X_world_x.jpg"), colorize_scalar_map(outputs["world_x"], cv2.COLORMAP_TURBO))
    if "world_y" in outputs:
        save_bgr(os.path.join(output_dir, f"{stem}_X_world_y.jpg"), colorize_scalar_map(outputs["world_y"], cv2.COLORMAP_TURBO))
    if "world_z" in outputs:
        save_bgr(os.path.join(output_dir, f"{stem}_X_world_z.jpg"), colorize_scalar_map(outputs["world_z"], cv2.COLORMAP_TURBO))
    if "world_up" in outputs:
        save_bgr(os.path.join(output_dir, f"{stem}_X_world_up.jpg"), colorize_scalar_map(outputs["world_up"], cv2.COLORMAP_TURBO))
    if "view_dir_x" in outputs:
        save_bgr(os.path.join(output_dir, f"{stem}_view_dir_x.jpg"), colorize_scalar_map(outputs["view_dir_x"], cv2.COLORMAP_TURBO))
    if "view_dir_y" in outputs:
        save_bgr(os.path.join(output_dir, f"{stem}_view_dir_y.jpg"), colorize_scalar_map(outputs["view_dir_y"], cv2.COLORMAP_TURBO))
    if "view_dir_z" in outputs:
        save_bgr(os.path.join(output_dir, f"{stem}_view_dir_z.jpg"), colorize_scalar_map(outputs["view_dir_z"], cv2.COLORMAP_TURBO))
    if "pixel_ray_dir_x" in outputs:
        save_bgr(os.path.join(output_dir, f"{stem}_pixel_ray_dir_x.jpg"), colorize_scalar_map(outputs["pixel_ray_dir_x"], cv2.COLORMAP_TURBO))
    if "pixel_ray_dir_y" in outputs:
        save_bgr(os.path.join(output_dir, f"{stem}_pixel_ray_dir_y.jpg"), colorize_scalar_map(outputs["pixel_ray_dir_y"], cv2.COLORMAP_TURBO))
    if "pixel_ray_dir_z" in outputs:
        save_bgr(os.path.join(output_dir, f"{stem}_pixel_ray_dir_z.jpg"), colorize_scalar_map(outputs["pixel_ray_dir_z"], cv2.COLORMAP_TURBO))
    if "water_surface_point_x" in outputs:
        save_bgr(os.path.join(output_dir, f"{stem}_water_surface_point_x.jpg"), colorize_scalar_map(outputs["water_surface_point_x"], cv2.COLORMAP_TURBO))
    if "water_surface_point_y" in outputs:
        save_bgr(os.path.join(output_dir, f"{stem}_water_surface_point_y.jpg"), colorize_scalar_map(outputs["water_surface_point_y"], cv2.COLORMAP_TURBO))
    if "water_surface_point_z" in outputs:
        save_bgr(os.path.join(output_dir, f"{stem}_water_surface_point_z.jpg"), colorize_scalar_map(outputs["water_surface_point_z"], cv2.COLORMAP_TURBO))
    if "water_surface_normal_world" in outputs:
        save_bgr(
            os.path.join(output_dir, f"{stem}_water_surface_normal_world.jpg"),
            colorize_normal(outputs["water_surface_normal_world"]),
        )
    if "ray_interface_point_x" in outputs:
        save_bgr(os.path.join(output_dir, f"{stem}_ray_interface_point_x.jpg"), colorize_scalar_map(outputs["ray_interface_point_x"], cv2.COLORMAP_TURBO))
    if "ray_interface_point_y" in outputs:
        save_bgr(os.path.join(output_dir, f"{stem}_ray_interface_point_y.jpg"), colorize_scalar_map(outputs["ray_interface_point_y"], cv2.COLORMAP_TURBO))
    if "ray_interface_point_z" in outputs:
        save_bgr(os.path.join(output_dir, f"{stem}_ray_interface_point_z.jpg"), colorize_scalar_map(outputs["ray_interface_point_z"], cv2.COLORMAP_TURBO))
    if "plane_hit_x" in outputs:
        save_bgr(os.path.join(output_dir, f"{stem}_plane_hit_x.jpg"), colorize_scalar_map(outputs["plane_hit_x"], cv2.COLORMAP_TURBO))
    if "plane_hit_y" in outputs:
        save_bgr(os.path.join(output_dir, f"{stem}_plane_hit_y.jpg"), colorize_scalar_map(outputs["plane_hit_y"], cv2.COLORMAP_TURBO))
    if "plane_focus_weight" in outputs:
        save_bgr(
            os.path.join(output_dir, f"{stem}_plane_focus_weight.jpg"),
            colorize_scalar_map(outputs["plane_focus_weight"], cv2.COLORMAP_HOT),
        )
    if "receiver_blur_weight" in outputs:
        save_bgr(
            os.path.join(output_dir, f"{stem}_receiver_blur_weight.jpg"),
            colorize_scalar_map(outputs["receiver_blur_weight"], cv2.COLORMAP_HOT),
        )
    if "receiver_alignment" in outputs:
        save_bgr(
            os.path.join(output_dir, f"{stem}_receiver_alignment.jpg"),
            colorize_scalar_map(outputs["receiver_alignment"], cv2.COLORMAP_BONE),
        )
    if "refraction_spread" in outputs:
        save_bgr(
            os.path.join(output_dir, f"{stem}_refraction_spread.jpg"),
            colorize_scalar_map(outputs["refraction_spread"], cv2.COLORMAP_HOT),
        )
    if "receiver_footprint_u" in outputs:
        save_bgr(
            os.path.join(output_dir, f"{stem}_receiver_footprint_u.jpg"),
            colorize_scalar_map(outputs["receiver_footprint_u"], cv2.COLORMAP_TURBO),
        )
    if "receiver_footprint_v" in outputs:
        save_bgr(
            os.path.join(output_dir, f"{stem}_receiver_footprint_v.jpg"),
            colorize_scalar_map(outputs["receiver_footprint_v"], cv2.COLORMAP_TURBO),
        )
    if "receiver_focus_weight" in outputs:
        save_bgr(
            os.path.join(output_dir, f"{stem}_receiver_focus_weight.jpg"),
            colorize_scalar_map(outputs["receiver_focus_weight"], cv2.COLORMAP_HOT),
        )
    if "receiver_support" in outputs:
        save_bgr(
            os.path.join(output_dir, f"{stem}_receiver_support.jpg"),
            colorize_scalar_map(outputs["receiver_support"], cv2.COLORMAP_BONE),
        )
    if "plane_query_x" in outputs:
        save_bgr(os.path.join(output_dir, f"{stem}_plane_query_x.jpg"), colorize_scalar_map(outputs["plane_query_x"], cv2.COLORMAP_TURBO))
    if "plane_query_y" in outputs:
        save_bgr(os.path.join(output_dir, f"{stem}_plane_query_y.jpg"), colorize_scalar_map(outputs["plane_query_y"], cv2.COLORMAP_TURBO))
    if "refracted_dir_x" in outputs:
        save_bgr(
            os.path.join(output_dir, f"{stem}_refracted_dir_x.jpg"),
            colorize_scalar_map(outputs["refracted_dir_x"], cv2.COLORMAP_TURBO),
        )
    if "refracted_dir_y" in outputs:
        save_bgr(
            os.path.join(output_dir, f"{stem}_refracted_dir_y.jpg"),
            colorize_scalar_map(outputs["refracted_dir_y"], cv2.COLORMAP_TURBO),
        )
    if "refracted_dir_z" in outputs:
        save_bgr(
            os.path.join(output_dir, f"{stem}_refracted_dir_z.jpg"),
            colorize_scalar_map(outputs["refracted_dir_z"], cv2.COLORMAP_TURBO),
        )
    if "local_refracted_dir_x" in outputs:
        save_bgr(
            os.path.join(output_dir, f"{stem}_local_refracted_dir_x.jpg"),
            colorize_scalar_map(outputs["local_refracted_dir_x"], cv2.COLORMAP_TURBO),
        )
    if "local_refracted_dir_y" in outputs:
        save_bgr(
            os.path.join(output_dir, f"{stem}_local_refracted_dir_y.jpg"),
            colorize_scalar_map(outputs["local_refracted_dir_y"], cv2.COLORMAP_TURBO),
        )
    if "local_refracted_dir_z" in outputs:
        save_bgr(
            os.path.join(output_dir, f"{stem}_local_refracted_dir_z.jpg"),
            colorize_scalar_map(outputs["local_refracted_dir_z"], cv2.COLORMAP_TURBO),
        )

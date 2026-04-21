#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        self.preload_img = True
        self.ncc_scale = 1.0
        self.multi_view_num = 8
        self.multi_view_max_angle = 30
        self.multi_view_min_dis = 0.01
        self.multi_view_max_dis = 1.5
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.001
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        self.scale_loss_weight = 100.0
        
        self.wo_image_weight = False
        self.single_view_weight = 0.015
        self.single_view_weight_from_iter = 7000

        self.use_virtul_cam = False
        self.virtul_cam_prob = 0.5
        self.use_multi_view_trim = True
        self.multi_view_ncc_weight = 0.15
        self.multi_view_geo_weight = 0.03
        self.multi_view_weight_from_iter = 7000
        self.multi_view_patch_size = 3
        self.multi_view_sample_num = 102400
        self.multi_view_pixel_noise_th = 1.0
        self.wo_use_geo_occ_aware = False

        self.opacity_cull_threshold = 0.005
        self.densify_abs_grad_threshold = 0.0008
        self.abs_split_radii2D_threshold = 20
        self.max_abs_split_points = 50_000
        self.max_all_points = 6000_000
        self.exposure_compensation = False
        self.random_background = False
        super().__init__(parser, "Optimization Parameters")

class UnderwaterParams(ParamGroup):
    def __init__(self, parser):
        self.render_mode = "clean_pgsr"
        self.enable_underwater = False
        self.enable_surface = False
        self.enable_caustic = False
        self.enable_beam = True
        self.enable_interface = False
        self.enable_appearance = False
        self.submerged_only = True

        self.freeze_geometry = True
        self.freeze_sh = True
        self.freeze_opacity = True
        self.freeze_app_model = True
        self.disable_pgsr_app_model = True

        self.uw_beta_r = 0.15
        self.uw_beta_g = 0.07
        self.uw_beta_b = 0.03
        self.uw_beta_b_r = 0.11
        self.uw_beta_b_g = 0.05
        self.uw_beta_b_b = 0.02
        self.uw_binf_r = 0.05
        self.uw_binf_g = 0.25
        self.uw_binf_b = 0.45
        self.medium_jerlov_type = "I"
        self.medium_use_jerlov_prior = 1
        self.medium_beta_min = 0.01
        self.medium_beta_max = 1.5
        self.medium_grid_resolution = 24
        self.medium_sh_degree = 1
        self.medium_enable_motion = False
        self.medium_motion_grid_resolution = 12
        self.medium_motion_scale = 0.0015
        self.medium_directional_strength = 0.35
        self.medium_depth_scale = 1.0
        self.medium_spatial_scale = 0.08
        self.medium_num_basis = 4
        self.medium_beta_strength = 1.0
        self.medium_binf_strength = 1.0
        self.medium_flow_smoothness_weight = 0.01
        self.medium_motion_magnitude_weight = 0.002
        self.appearance_grid_xy = 32
        self.appearance_grid_guide = 8
        self.appearance_gain_strength = 0.25
        self.appearance_bias_strength = 0.05
        self.water_up_axis = "z"
        self.scene_bounds_padding = 0.08

        self.surface_A1 = 0.08
        self.surface_A2 = 0.05
        self.surface_w1 = 6.0
        self.surface_w2 = 7.5
        self.surface_s1 = 0.35
        self.surface_s2 = 0.28
        self.surface_preset = "beaufort3"
        self.surface_learnable = False
        self.surface_lr_scale = 0.1
        self.surface_spatial_scale = 1.0
        self.surface_num_waves = 8
        self.surface_num_bands = 3
        self.surface_dominant_angle = 0.0
        self.surface_directional_spread = 0.6
        self.surface_detail_strength = 0.24
        self.surface_base_height = 0.0
        self.surface_tide_amplitude = 0.02
        self.surface_tide_speed = 0.15
        self.surface_tide_phase = 0.0
        self.surface_flow_strength = 0.06
        self.surface_flow_speed = 0.08
        self.surface_min_wave_amplitude = 0.008
        self.surface_min_wave_speed = 0.15
        self.surface_min_tide_speed = 0.03
        self.surface_max_tide_amplitude = 0.2
        self.surface_max_base_height_offset = 0.5
        self.surface_smoothness_weight = 0.01

        self.caustic_lambda = 0.15
        self.caustic_min_lambda = 0.05
        self.caustic_light_dir_x = 0.2
        self.caustic_light_dir_y = -0.6
        self.caustic_light_dir_z = 0.77
        self.caustic_refract_eta = 1.0 / 1.33
        self.caustic_plane_res = 96
        self.caustic_plane_padding = 0.15
        self.caustic_water_level_offset = 0.0
        self.caustic_water_level_mode = "scene_relative"
        self.caustic_water_level_world = 0.0
        self.caustic_receiver_query_mix = 0.45
        self.caustic_beam_strength = 0.12
        self.caustic_beam_depth_decay = 0.35
        self.caustic_beam_view_power = 2.0
        self.caustic_beam_num_steps = 6
        self.caustic_beam_phase_g = 0.70
        self.water_level_quantile = 0.98
        self.waterline_softness = 0.08
        self.interface_band_width = 0.18
        self.interface_tint_strength = 0.18
        self.interface_reflection_strength = 0.35
        self.interface_fresnel_power = 5.0
        self.caustic_max_pattern = 3.0
        self.caustic_regularization_weight = 0.01
        self.temporal_smoothness_weight = 0.05

        self.underwater_lr = 0.005
        self.underwater_iterations = 10_000
        self.time_delta = 0.033333
        self.time_stride = 1.0
        self.use_ssim = True
        self.underwater_loss_mode = "proxy"
        self.debug_dump_intermediate = False
        self.debug_dump_interval = 200
        self.video_frames = 120
        self.video_fps = 24
        self.video_view = "test"
        self.video_camera_index = 0
        self.video_mode = "fixed_camera"
        self.video_time_mode = "dynamic"
        self.video_time_start = 0.0
        self.video_time_scale = 1.0
        self.video_camera_cycles = 1.0
        self.video_time_duration = 4.0
        super().__init__(parser, "Underwater Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except (TypeError, FileNotFoundError):
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None or k not in merged_dict:
            merged_dict[k] = v
    return Namespace(**merged_dict)

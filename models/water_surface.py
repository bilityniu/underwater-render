import math

import torch
import torch.nn.functional as F
from torch import nn


class WaterSurface(nn.Module):
    """Trainable multi-scale directional wave-spectrum surface."""

    def __init__(
        self,
        A1,
        A2,
        w1,
        w2,
        s1,
        s2,
        spatial_scale=1.0,
        num_waves=8,
        num_bands=3,
        dominant_angle=0.0,
        directional_spread=0.6,
        detail_strength=0.3,
        base_height=0.0,
        tide_amplitude=0.02,
        tide_speed=0.15,
        tide_phase=0.0,
        min_wave_amplitude=0.008,
        min_wave_speed=0.15,
        min_tide_speed=0.03,
        max_tide_amplitude=0.2,
        max_base_height_offset=0.5,
        flow_strength=0.08,
        flow_speed=0.10,
        coord_min=None,
        coord_max=None,
        local_field_res=12,
    ):
        super().__init__()
        self.spatial_scale = float(spatial_scale)
        self.num_waves = max(4, int(num_waves))
        self.num_bands = max(2, min(int(num_bands), self.num_waves))
        self.min_wave_amplitude = float(min_wave_amplitude)
        self.min_wave_speed = float(min_wave_speed)
        self.min_tide_speed = float(min_tide_speed)
        self.max_tide_amplitude = float(max_tide_amplitude)
        self.max_base_height_offset = float(max_base_height_offset)
        self.max_flow_strength = 0.35
        self.local_field_res = max(6, int(local_field_res))
        coord_min_tensor, coord_max_tensor = self._resolve_coord_bounds(coord_min, coord_max)
        self.register_buffer("coord_min", coord_min_tensor)
        self.register_buffer("coord_max", coord_max_tensor)

        band_ids = self._build_band_ids(self.num_waves, self.num_bands)
        wave_offsets = torch.linspace(-1.0, 1.0, self.num_waves, dtype=torch.float32)
        band_frequency = self._build_band_frequency(w1, w2, self.num_bands)
        band_amplitude = self._build_band_amplitude(A1, A2, self.num_bands)

        amp_init = band_amplitude[band_ids] * (1.0 - 0.35 * wave_offsets.abs())
        freq_init = band_frequency[band_ids] * (1.0 + 0.18 * wave_offsets)
        freq_init = torch.clamp(freq_init, min=0.15)

        base_speed = max(1e-3, 0.5 * (float(s1) + float(s2)))
        speed_init = base_speed * torch.sqrt(freq_init / torch.clamp(freq_init.mean(), min=1e-6))
        speed_init = speed_init * (1.0 + 0.08 * wave_offsets)
        speed_init = torch.clamp(speed_init, min=0.02)
        amp_init_eff = torch.clamp(amp_init - self.min_wave_amplitude, min=1e-4)
        speed_init_eff = torch.clamp(speed_init - self.min_wave_speed, min=1e-4)

        phase_init = 2.0 * math.pi * torch.arange(self.num_waves, dtype=torch.float32) / float(self.num_waves)
        direction_offsets = 0.85 * wave_offsets
        flow_count = max(4, min(6, self.num_bands * 2))
        flow_offsets = torch.linspace(-1.0, 1.0, flow_count, dtype=torch.float32)
        flow_amp_init = torch.clamp(
            0.05 * (1.0 - 0.35 * flow_offsets.abs()),
            min=0.02,
        )
        flow_freq_init = torch.logspace(
            math.log10(max(0.08, float(min(w1, w2)) * 0.18)),
            math.log10(max(0.18, float(min(w1, w2)) * 0.65)),
            steps=flow_count,
            dtype=torch.float32,
        )
        flow_wave_speed_init = torch.clamp(
            0.6 * max(1e-3, 0.5 * (float(s1) + float(s2))) * (1.0 + 0.15 * flow_offsets),
            min=0.02,
        )
        flow_phase_init = 2.0 * math.pi * torch.arange(flow_count, dtype=torch.float32) / float(flow_count)
        flow_angle_offsets = 1.4 * flow_offsets

        self.register_buffer("band_ids", band_ids)
        self.register_buffer("wave_offsets", wave_offsets)
        self.register_buffer("flow_offsets", flow_offsets)
        packet_count = max(4, min(8, self.num_waves))
        packet_offsets = torch.linspace(-1.0, 1.0, packet_count, dtype=torch.float32)
        packet_angles = 2.0 * math.pi * torch.arange(packet_count, dtype=torch.float32) / float(packet_count)
        packet_amp_init = torch.clamp(
            0.22 * band_amplitude[0] * (1.0 - 0.30 * packet_offsets.abs()),
            min=0.003,
            max=0.025,
        )
        packet_freq_init = torch.clamp(
            0.85 * band_frequency[min(self.num_bands - 1, 1)] * (1.0 + 0.35 * packet_offsets),
            min=0.35,
            max=12.0,
        )
        packet_speed_init = torch.clamp(
            0.55 * base_speed * (1.0 + 0.15 * packet_offsets),
            min=0.03,
            max=2.5,
        )
        packet_sigma_u_init = torch.clamp(0.34 - 0.08 * packet_offsets.abs(), min=0.16, max=0.40)
        packet_sigma_v_init = torch.clamp(0.26 - 0.05 * packet_offsets.abs(), min=0.12, max=0.32)
        packet_center_u_init = 0.58 * torch.sin(packet_angles + 0.25)
        packet_center_v_init = 0.52 * torch.cos(1.35 * packet_angles - 0.30)
        packet_phase_init = 2.0 * math.pi * torch.arange(packet_count, dtype=torch.float32) / float(packet_count)
        packet_angle_offsets = 0.9 * packet_offsets

        self.raw_amplitudes = nn.Parameter(self._inv_softplus(amp_init_eff))
        self.raw_frequencies = nn.Parameter(self._inv_softplus(freq_init))
        self.raw_speeds = nn.Parameter(self._inv_softplus(speed_init_eff))
        self.direction_offsets = nn.Parameter(direction_offsets)
        self.phases = nn.Parameter(phase_init)
        self.dominant_angle = nn.Parameter(torch.tensor(float(dominant_angle), dtype=torch.float32))
        self.raw_directional_spread = nn.Parameter(
            self._inv_softplus(torch.tensor(float(directional_spread), dtype=torch.float32))
        )
        self.raw_detail_strength = nn.Parameter(
            self._inv_sigmoid(torch.tensor(float(detail_strength), dtype=torch.float32))
        )
        self.base_height = nn.Parameter(torch.tensor(float(base_height), dtype=torch.float32))
        tide_amplitude_init = torch.clamp(
            torch.tensor(float(tide_amplitude), dtype=torch.float32),
            min=1e-4,
            max=max(1e-4, self.max_tide_amplitude),
        )
        tide_speed_init = torch.clamp(
            torch.tensor(float(tide_speed), dtype=torch.float32) - self.min_tide_speed,
            min=1e-4,
        )
        self.raw_tide_amplitude = nn.Parameter(self._inv_softplus(tide_amplitude_init))
        self.raw_tide_speed = nn.Parameter(self._inv_softplus(tide_speed_init))
        self.tide_phase = nn.Parameter(torch.tensor(float(tide_phase), dtype=torch.float32))
        flow_strength_init = torch.clamp(
            torch.tensor(float(flow_strength), dtype=torch.float32),
            min=1e-4,
            max=self.max_flow_strength,
        )
        flow_speed_init = torch.clamp(
            torch.tensor(float(flow_speed), dtype=torch.float32),
            min=1e-4,
        )
        self.raw_flow_strength = nn.Parameter(self._inv_sigmoid(flow_strength_init / self.max_flow_strength))
        self.raw_flow_speed = nn.Parameter(self._inv_softplus(flow_speed_init))
        self.raw_flow_amplitudes = nn.Parameter(self._inv_softplus(flow_amp_init))
        self.raw_flow_frequencies = nn.Parameter(self._inv_softplus(flow_freq_init))
        self.raw_flow_speeds = nn.Parameter(self._inv_softplus(flow_wave_speed_init))
        self.flow_direction_offsets = nn.Parameter(flow_angle_offsets)
        self.flow_phases = nn.Parameter(flow_phase_init)
        self.local_amp_field = nn.Parameter(torch.zeros(1, 1, self.local_field_res, self.local_field_res, dtype=torch.float32))
        self.local_angle_field = nn.Parameter(torch.zeros(1, 1, self.local_field_res, self.local_field_res, dtype=torch.float32))
        self.local_phase_field = nn.Parameter(torch.zeros(1, 1, self.local_field_res, self.local_field_res, dtype=torch.float32))
        self.local_detail_field = nn.Parameter(torch.zeros(1, 1, self.local_field_res, self.local_field_res, dtype=torch.float32))
        self.register_buffer("packet_offsets", packet_offsets)
        self.packet_center_u = nn.Parameter(self._inv_sigmoid(0.5 * (packet_center_u_init + 1.0)))
        self.packet_center_v = nn.Parameter(self._inv_sigmoid(0.5 * (packet_center_v_init + 1.0)))
        self.raw_packet_sigma_u = nn.Parameter(self._inv_softplus(packet_sigma_u_init))
        self.raw_packet_sigma_v = nn.Parameter(self._inv_softplus(packet_sigma_v_init))
        self.raw_packet_amplitudes = nn.Parameter(self._inv_softplus(packet_amp_init))
        self.raw_packet_frequencies = nn.Parameter(self._inv_softplus(packet_freq_init))
        self.raw_packet_speeds = nn.Parameter(self._inv_softplus(packet_speed_init))
        self.packet_angle_offsets = nn.Parameter(packet_angle_offsets)
        self.packet_phases = nn.Parameter(packet_phase_init)

    def forward(self, grid_xy, time_value):
        return self.sample_world(grid_xy[0], grid_xy[1], time_value)

    def sample_world(self, x_world, y_world, time_value):
        x = x_world * self.spatial_scale
        y = y_world * self.spatial_scale
        t = self._scalar_time(x_world, time_value)
        x_norm, y_norm, x_norm_dx, y_norm_dy = self.normalized_world_coords(x_world, y_world)

        spread = torch.clamp(F.softplus(self.raw_directional_spread), min=1e-4, max=2.0)
        detail_strength = torch.sigmoid(self.raw_detail_strength)
        tide_amplitude = torch.clamp(F.softplus(self.raw_tide_amplitude), min=0.0, max=self.max_tide_amplitude)
        tide_speed = torch.clamp(self.min_tide_speed + F.softplus(self.raw_tide_speed), min=self.min_tide_speed, max=5.0)
        flow_strength = torch.clamp(
            torch.sigmoid(self.raw_flow_strength) * self.max_flow_strength,
            min=0.0,
            max=self.max_flow_strength,
        )
        flow_speed_scale = torch.clamp(0.6 * F.softplus(self.raw_flow_speed), min=0.01, max=3.0)
        local_fields = self.sample_local_fields(x_world, y_world)
        local_angle_offset = local_fields["angle_offset"]
        local_amp_scale = local_fields["amp_scale"]
        local_phase_offset = local_fields["phase_offset"]
        local_detail_scale = local_fields["detail_scale"]
        direction = self.dominant_angle + spread * torch.tanh(self.direction_offsets).view(-1, 1, 1)
        direction = direction + (0.75 + 0.25 * self.wave_offsets).view(-1, 1, 1) * local_angle_offset.unsqueeze(0)
        direction_x = torch.cos(direction)
        direction_y = torch.sin(direction)
        amplitude = torch.clamp(self.min_wave_amplitude + F.softplus(self.raw_amplitudes), min=self.min_wave_amplitude, max=1.0).view(-1, 1, 1)
        frequency = torch.clamp(F.softplus(self.raw_frequencies), min=0.15, max=40.0).view(-1, 1, 1)
        speed = torch.clamp(self.min_wave_speed + F.softplus(self.raw_speeds), min=self.min_wave_speed, max=12.0).view(-1, 1, 1)
        phase = self.phases.view(-1, 1, 1)
        flow_direction = self.dominant_angle + 0.5 * math.pi + spread * torch.tanh(self.flow_direction_offsets)
        flow_dir_x = torch.cos(flow_direction).view(-1, 1, 1)
        flow_dir_y = torch.sin(flow_direction).view(-1, 1, 1)
        flow_amplitude = torch.clamp(0.42 * detail_strength * F.softplus(self.raw_flow_amplitudes), min=0.0, max=0.24).view(-1, 1, 1)
        flow_frequency = torch.clamp(F.softplus(self.raw_flow_frequencies), min=0.04, max=8.0).view(-1, 1, 1)
        flow_speed = torch.clamp(F.softplus(self.raw_flow_speeds) * flow_speed_scale, min=0.01, max=6.0).view(-1, 1, 1)
        flow_phase = self.flow_phases.view(-1, 1, 1)

        flow_coord = flow_dir_x * x.unsqueeze(0) + flow_dir_y * y.unsqueeze(0)
        flow_wave_phase = flow_frequency * flow_coord - flow_speed * t + flow_phase
        flow_phase_y = flow_wave_phase + 0.37 + 0.9 * self.flow_offsets.view(-1, 1, 1)
        flow_sin_x = torch.sin(flow_wave_phase)
        flow_cos_x = torch.cos(flow_wave_phase)
        flow_sin_y = torch.sin(flow_phase_y)
        flow_cos_y = torch.cos(flow_phase_y)

        warp_x = torch.sum(flow_amplitude * flow_sin_x, dim=0)
        warp_y = torch.sum(flow_amplitude * flow_sin_y, dim=0)

        warp_x_dx = torch.sum(flow_amplitude * flow_frequency * flow_dir_x * flow_cos_x, dim=0)
        warp_x_dy = torch.sum(flow_amplitude * flow_frequency * flow_dir_y * flow_cos_x, dim=0)
        warp_y_dx = torch.sum(flow_amplitude * flow_frequency * flow_dir_x * flow_cos_y, dim=0)
        warp_y_dy = torch.sum(flow_amplitude * flow_frequency * flow_dir_y * flow_cos_y, dim=0)

        warp_x_dxx = torch.sum(
            -flow_amplitude * (flow_frequency ** 2) * (flow_dir_x ** 2) * flow_sin_x,
            dim=0,
        )
        warp_x_dyy = torch.sum(
            -flow_amplitude * (flow_frequency ** 2) * (flow_dir_y ** 2) * flow_sin_x,
            dim=0,
        )
        warp_y_dxx = torch.sum(
            -flow_amplitude * (flow_frequency ** 2) * (flow_dir_x ** 2) * flow_sin_y,
            dim=0,
        )
        warp_y_dyy = torch.sum(
            -flow_amplitude * (flow_frequency ** 2) * (flow_dir_y ** 2) * flow_sin_y,
            dim=0,
        )

        advected_x = x + flow_strength * warp_x
        advected_y = y + flow_strength * warp_y
        advected_x_dx = 1.0 + flow_strength * warp_x_dx
        advected_x_dy = flow_strength * warp_x_dy
        advected_y_dx = flow_strength * warp_y_dx
        advected_y_dy = 1.0 + flow_strength * warp_y_dy
        advected_x_dxx = flow_strength * warp_x_dxx
        advected_x_dyy = flow_strength * warp_x_dyy
        advected_y_dxx = flow_strength * warp_y_dxx
        advected_y_dyy = flow_strength * warp_y_dyy

        amplitude_envelope = 1.0 + 0.10 * flow_strength * torch.tanh(0.9 * warp_x + 0.55 * warp_y)
        wave_amplitude = amplitude * amplitude_envelope.unsqueeze(0) * local_amp_scale.unsqueeze(0)
        phase_warp_scale = (0.12 + 0.06 * detail_strength) * flow_strength
        phase_warp_gain = (1.0 + 0.30 * self.wave_offsets).view(-1, 1, 1)
        phase_warp = phase_warp_gain * phase_warp_scale * (0.85 * warp_x - 0.60 * warp_y).unsqueeze(0)
        phase_warp_dx = phase_warp_gain * phase_warp_scale * (0.85 * warp_x_dx - 0.60 * warp_y_dx).unsqueeze(0)
        phase_warp_dy = phase_warp_gain * phase_warp_scale * (0.85 * warp_x_dy - 0.60 * warp_y_dy).unsqueeze(0)
        phase_warp_dxx = phase_warp_gain * phase_warp_scale * (0.85 * warp_x_dxx - 0.60 * warp_y_dxx).unsqueeze(0)
        phase_warp_dyy = phase_warp_gain * phase_warp_scale * (0.85 * warp_x_dyy - 0.60 * warp_y_dyy).unsqueeze(0)
        shared_phase_warp = phase_warp.mean(dim=0, keepdim=True)
        shared_phase_warp_dx = phase_warp_dx.mean(dim=0, keepdim=True)
        shared_phase_warp_dy = phase_warp_dy.mean(dim=0, keepdim=True)
        shared_phase_warp_dxx = phase_warp_dxx.mean(dim=0, keepdim=True)
        shared_phase_warp_dyy = phase_warp_dyy.mean(dim=0, keepdim=True)
        local_phase_gain = (1.0 + 0.35 * self.wave_offsets).view(-1, 1, 1)
        local_phase = local_phase_gain * local_phase_offset.unsqueeze(0)

        directional_coord = direction_x * advected_x.unsqueeze(0) + direction_y * advected_y.unsqueeze(0)
        directional_dx = direction_x * advected_x_dx.unsqueeze(0) + direction_y * advected_y_dx.unsqueeze(0)
        directional_dy = direction_x * advected_x_dy.unsqueeze(0) + direction_y * advected_y_dy.unsqueeze(0)
        directional_dxx = direction_x * advected_x_dxx.unsqueeze(0) + direction_y * advected_y_dxx.unsqueeze(0)
        directional_dyy = direction_x * advected_x_dyy.unsqueeze(0) + direction_y * advected_y_dyy.unsqueeze(0)
        wave_phase = frequency * directional_coord + phase_warp + local_phase - speed * t + phase
        detail_coord_scale = 1.75
        detail_time_scale = 0.68
        detail_amplitude = 0.32 * wave_amplitude * local_detail_scale.unsqueeze(0)
        detail_phase = (
            detail_coord_scale * frequency * directional_coord
            + 0.65 * phase_warp
            + 0.55 * local_phase
            - detail_time_scale * speed * t
            + 0.37 * phase
        )
        detail_phase_dx = detail_coord_scale * frequency * directional_dx + 0.65 * phase_warp_dx
        detail_phase_dy = detail_coord_scale * frequency * directional_dy + 0.65 * phase_warp_dy
        detail_phase_dxx = detail_coord_scale * frequency * directional_dxx + 0.65 * phase_warp_dxx
        detail_phase_dyy = detail_coord_scale * frequency * directional_dyy + 0.65 * phase_warp_dyy

        packet_center_u = 1.70 * torch.sigmoid(self.packet_center_u).view(-1, 1, 1) - 0.85
        packet_center_v = 1.70 * torch.sigmoid(self.packet_center_v).view(-1, 1, 1) - 0.85
        packet_sigma_u = torch.clamp(F.softplus(self.raw_packet_sigma_u), min=0.12, max=0.45).view(-1, 1, 1)
        packet_sigma_v = torch.clamp(F.softplus(self.raw_packet_sigma_v), min=0.10, max=0.36).view(-1, 1, 1)
        packet_amp = torch.clamp(
            (0.55 + 0.45 * detail_strength) * F.softplus(self.raw_packet_amplitudes),
            min=0.0,
            max=0.045,
        ).view(-1, 1, 1)
        packet_frequency = torch.clamp(F.softplus(self.raw_packet_frequencies), min=0.30, max=16.0).view(-1, 1, 1)
        packet_speed = torch.clamp(F.softplus(self.raw_packet_speeds), min=0.02, max=3.0).view(-1, 1, 1)
        packet_direction = self.dominant_angle + 0.75 * spread * torch.tanh(self.packet_angle_offsets).view(-1, 1, 1)
        packet_dir_x = torch.cos(packet_direction)
        packet_dir_y = torch.sin(packet_direction)
        packet_phase_bias = self.packet_phases.view(-1, 1, 1)

        packet_du = x_norm.unsqueeze(0) - packet_center_u
        packet_dv = y_norm.unsqueeze(0) - packet_center_v
        inv_sigma_u2 = 1.0 / torch.clamp(packet_sigma_u * packet_sigma_u, min=1e-5)
        inv_sigma_v2 = 1.0 / torch.clamp(packet_sigma_v * packet_sigma_v, min=1e-5)
        packet_envelope = torch.exp(-0.5 * (packet_du * packet_du * inv_sigma_u2 + packet_dv * packet_dv * inv_sigma_v2))
        packet_envelope_dx = packet_envelope * (-packet_du * inv_sigma_u2 * x_norm_dx)
        packet_envelope_dy = packet_envelope * (-packet_dv * inv_sigma_v2 * y_norm_dy)
        packet_envelope_dxx = packet_envelope * (((packet_du * packet_du) * inv_sigma_u2 * inv_sigma_u2 - inv_sigma_u2) * (x_norm_dx ** 2))
        packet_envelope_dyy = packet_envelope * (((packet_dv * packet_dv) * inv_sigma_v2 * inv_sigma_v2 - inv_sigma_v2) * (y_norm_dy ** 2))

        packet_coord = packet_dir_x * advected_x.unsqueeze(0) + packet_dir_y * advected_y.unsqueeze(0)
        packet_coord_dx = packet_dir_x * advected_x_dx.unsqueeze(0) + packet_dir_y * advected_y_dx.unsqueeze(0)
        packet_coord_dy = packet_dir_x * advected_x_dy.unsqueeze(0) + packet_dir_y * advected_y_dy.unsqueeze(0)
        packet_coord_dxx = packet_dir_x * advected_x_dxx.unsqueeze(0) + packet_dir_y * advected_y_dxx.unsqueeze(0)
        packet_coord_dyy = packet_dir_x * advected_x_dyy.unsqueeze(0) + packet_dir_y * advected_y_dyy.unsqueeze(0)
        packet_phase = packet_frequency * packet_coord + 0.35 * shared_phase_warp + packet_phase_bias - 0.55 * packet_speed * t
        packet_phase_dx = packet_frequency * packet_coord_dx + 0.35 * shared_phase_warp_dx
        packet_phase_dy = packet_frequency * packet_coord_dy + 0.35 * shared_phase_warp_dy
        packet_phase_dxx = packet_frequency * packet_coord_dxx + 0.35 * shared_phase_warp_dxx
        packet_phase_dyy = packet_frequency * packet_coord_dyy + 0.35 * shared_phase_warp_dyy

        sin_phase = torch.sin(wave_phase)
        cos_phase = torch.cos(wave_phase)
        detail_sin = torch.sin(detail_phase)
        detail_cos = torch.cos(detail_phase)
        packet_sin = torch.sin(packet_phase)
        packet_cos = torch.cos(packet_phase)

        primary_height = torch.sum(wave_amplitude * sin_phase, dim=0)
        detail_height = torch.sum(detail_amplitude * detail_sin, dim=0)
        packet_height = torch.sum(packet_amp * packet_envelope * packet_sin, dim=0)
        tide_height = torch.ones_like(primary_height) * (tide_amplitude * torch.sin(tide_speed * t + self.tide_phase))
        base_height = torch.clamp(self.base_height, min=-self.max_base_height_offset, max=self.max_base_height_offset)
        height = base_height + tide_height + primary_height + detail_strength * detail_height + packet_height

        phase_dx = frequency * directional_dx + phase_warp_dx
        phase_dy = frequency * directional_dy + phase_warp_dy
        phase_dxx = frequency * directional_dxx + phase_warp_dxx
        phase_dyy = frequency * directional_dyy + phase_warp_dyy

        primary_dx = torch.sum(wave_amplitude * cos_phase * phase_dx, dim=0)
        primary_dy = torch.sum(wave_amplitude * cos_phase * phase_dy, dim=0)
        detail_dx = torch.sum(detail_amplitude * detail_cos * detail_phase_dx, dim=0)
        detail_dy = torch.sum(detail_amplitude * detail_cos * detail_phase_dy, dim=0)
        packet_dx = torch.sum(packet_amp * (packet_envelope_dx * packet_sin + packet_envelope * packet_cos * packet_phase_dx), dim=0)
        packet_dy = torch.sum(packet_amp * (packet_envelope_dy * packet_sin + packet_envelope * packet_cos * packet_phase_dy), dim=0)
        dh_dx = primary_dx + detail_strength * detail_dx + packet_dx
        dh_dy = primary_dy + detail_strength * detail_dy + packet_dy

        primary_d2x = wave_amplitude * (-sin_phase * phase_dx * phase_dx + cos_phase * phase_dxx)
        primary_d2y = wave_amplitude * (-sin_phase * phase_dy * phase_dy + cos_phase * phase_dyy)
        primary_laplacian = torch.sum(primary_d2x + primary_d2y, dim=0)

        detail_d2x = detail_amplitude * (-detail_sin * detail_phase_dx * detail_phase_dx + detail_cos * detail_phase_dxx)
        detail_d2y = detail_amplitude * (-detail_sin * detail_phase_dy * detail_phase_dy + detail_cos * detail_phase_dyy)
        detail_laplacian = torch.sum(detail_d2x + detail_d2y, dim=0)
        packet_d2x = packet_amp * (
            packet_envelope_dxx * packet_sin
            + 2.0 * packet_envelope_dx * packet_cos * packet_phase_dx
            + packet_envelope * (-packet_sin * packet_phase_dx * packet_phase_dx + packet_cos * packet_phase_dxx)
        )
        packet_d2y = packet_amp * (
            packet_envelope_dyy * packet_sin
            + 2.0 * packet_envelope_dy * packet_cos * packet_phase_dy
            + packet_envelope * (-packet_sin * packet_phase_dy * packet_phase_dy + packet_cos * packet_phase_dyy)
        )
        packet_laplacian = torch.sum(packet_d2x + packet_d2y, dim=0)
        laplacian = primary_laplacian + detail_strength * detail_laplacian + packet_laplacian

        normal = torch.stack((-dh_dx, -dh_dy, torch.ones_like(height)), dim=0)
        normal = torch.nan_to_num(normal, nan=0.0, posinf=0.0, neginf=0.0)
        normal[2].fill_(1.0)
        normal = F.normalize(normal, p=2, dim=0, eps=1e-8)
        gradient = torch.nan_to_num(torch.stack((dh_dx, dh_dy), dim=0), nan=0.0, posinf=0.0, neginf=0.0)
        gradient_magnitude = torch.sqrt(torch.clamp(gradient[0] * gradient[0] + gradient[1] * gradient[1], min=0.0) + 1e-8).unsqueeze(0)
        curvature = torch.nan_to_num(laplacian.abs().unsqueeze(0), nan=0.0, posinf=0.0, neginf=0.0)
        wave_packet_energy = torch.nan_to_num(torch.sum(packet_amp * packet_envelope, dim=0).unsqueeze(0), nan=0.0, posinf=0.0, neginf=0.0)
        return {
            "surface_height": torch.nan_to_num(height.unsqueeze(0), nan=0.0, posinf=0.0, neginf=0.0),
            "surface_base_height": torch.nan_to_num(torch.ones_like(height).unsqueeze(0) * base_height, nan=0.0, posinf=0.0, neginf=0.0),
            "surface_tide_height": torch.nan_to_num(tide_height.unsqueeze(0), nan=0.0, posinf=0.0, neginf=0.0),
            "surface_gradient": gradient,
            "surface_normal": normal,
            "surface_gradient_magnitude": gradient_magnitude,
            "surface_curvature": curvature,
            "surface_domain_warp": torch.nan_to_num((flow_strength * torch.sqrt(warp_x * warp_x + warp_y * warp_y + 1e-8)).unsqueeze(0), nan=0.0, posinf=0.0, neginf=0.0),
            "surface_local_variation": torch.nan_to_num(local_fields["variation"].unsqueeze(0), nan=0.0, posinf=0.0, neginf=0.0),
            "surface_wave_packet_energy": wave_packet_energy,
        }

    def sample_caustic_surface(self, x_world, y_world, time_value):
        """Low-band surface used specifically for caustic formation.

        The sun has a finite angular size (~0.5°), which sets a spatial
        coherence window on the water surface: features shorter than that
        window do not focus coherently and blur out before reaching the
        receiver plane. We therefore form caustics from the dominant low-band
        Gerstner mode only, dropping the detail waves, wave packets, and
        domain-warp flow terms that drive the fast capillary tempo. The
        visible water surface still uses the full spectrum; only the caustic
        pipeline sees this band-limited surface.
        """
        x = x_world * self.spatial_scale
        y = y_world * self.spatial_scale
        t = self._scalar_time(x_world, time_value)

        spread = torch.clamp(F.softplus(self.raw_directional_spread), min=1e-4, max=2.0)
        direction = self.dominant_angle + spread * torch.tanh(self.direction_offsets).view(-1, 1, 1)
        direction_x = torch.cos(direction)
        direction_y = torch.sin(direction)
        amplitude = torch.clamp(
            self.min_wave_amplitude + F.softplus(self.raw_amplitudes),
            min=self.min_wave_amplitude,
            max=1.0,
        ).view(-1, 1, 1)
        frequency = torch.clamp(F.softplus(self.raw_frequencies), min=0.15, max=40.0).view(-1, 1, 1)
        speed = torch.clamp(
            self.min_wave_speed + F.softplus(self.raw_speeds),
            min=self.min_wave_speed,
            max=12.0,
        ).view(-1, 1, 1)
        phase = self.phases.view(-1, 1, 1)

        band_mask = (self.band_ids == 0).to(dtype=amplitude.dtype).view(-1, 1, 1)
        wave_amplitude = amplitude * band_mask

        directional_coord = direction_x * x.unsqueeze(0) + direction_y * y.unsqueeze(0)
        wave_phase = frequency * directional_coord - speed * t + phase
        sin_phase = torch.sin(wave_phase)
        cos_phase = torch.cos(wave_phase)

        height = torch.sum(wave_amplitude * sin_phase, dim=0)
        phase_dx = frequency * direction_x
        phase_dy = frequency * direction_y
        dh_dx = torch.sum(wave_amplitude * cos_phase * phase_dx, dim=0)
        dh_dy = torch.sum(wave_amplitude * cos_phase * phase_dy, dim=0)

        normal = torch.stack((-dh_dx, -dh_dy, torch.ones_like(height)), dim=0)
        normal = F.normalize(normal, p=2, dim=0, eps=1e-8)
        return {
            "surface_height": torch.nan_to_num(height.unsqueeze(0), nan=0.0, posinf=0.0, neginf=0.0),
            "surface_normal": torch.nan_to_num(normal, nan=0.0, posinf=0.0, neginf=0.0),
        }

    def compute_stokes_drift(self, depth_from_surface):
        """
        Stokes drift velocity at depth d below the water surface.

        For a superposition of linear surface-gravity waves, the depth-averaged
        mass-transport velocity (Stokes drift) is:

            v_stokes(d) = Σ_i  A_i² · k_i · ω_i · exp(−2 k_i · d) · k̂_i

        where k_i = spatial frequency, ω_i = k_i · c_i (temporal frequency),
        d = depth from surface (positive downward).

        This provides a physics-derived prior for the medium's motion_flow_grid,
        coupling surface-wave dynamics to subsurface current without extra parameters.

        Args:
            depth_from_surface: scalar or tensor [H, W] or [1, H, W], depth ≥ 0.

        Returns:
            Tensor [2, *spatial] – (lateral_u, lateral_v) Stokes drift velocity.
        """
        device = self.raw_amplitudes.device
        dtype = self.raw_amplitudes.dtype

        amplitude = torch.clamp(
            self.min_wave_amplitude + F.softplus(self.raw_amplitudes), min=self.min_wave_amplitude
        ).view(-1, 1, 1)
        frequency = torch.clamp(F.softplus(self.raw_frequencies), min=0.15).view(-1, 1, 1)
        speed = torch.clamp(
            self.min_wave_speed + F.softplus(self.raw_speeds), min=self.min_wave_speed
        ).view(-1, 1, 1)
        spread = torch.clamp(F.softplus(self.raw_directional_spread), min=1e-4, max=2.0)
        direction = self.dominant_angle + spread * torch.tanh(self.direction_offsets).view(-1, 1, 1)
        dir_x = torch.cos(direction)
        dir_y = torch.sin(direction)

        omega = frequency * speed  # temporal angular frequency ω_i = k_i · c_i

        if not torch.is_tensor(depth_from_surface):
            depth_from_surface = torch.tensor(float(depth_from_surface), device=device, dtype=dtype)
        d = depth_from_surface.to(device=device, dtype=dtype)
        if d.dim() < 3:
            d = d.unsqueeze(0)

        # Depth decay: exp(−2 k_i d)
        decay = torch.exp(-2.0 * frequency * d)            # [N, H, W]

        # Stokes drift contribution per wave component
        weight = (amplitude ** 2) * frequency * omega * decay  # [N, H, W]
        drift_u = (weight * dir_x).sum(dim=0)                  # [H, W]
        drift_v = (weight * dir_y).sum(dim=0)                  # [H, W]

        return torch.stack([drift_u, drift_v], dim=0)          # [2, H, W]

    @torch.no_grad()
    def compute_mean_phase_velocity(self):
        """Amplitude²-weighted mean phase velocity vector for caustic pattern drift.

        Returns the group-mean advection velocity (u, v) in world units per time unit.
        Used to translate the caustic map laterally so the pattern moves with the waves.
        """
        amplitude = torch.clamp(
            self.min_wave_amplitude + F.softplus(self.raw_amplitudes), min=self.min_wave_amplitude
        )
        speed = torch.clamp(self.min_wave_speed + F.softplus(self.raw_speeds), min=self.min_wave_speed)
        spread = torch.clamp(F.softplus(self.raw_directional_spread), min=1e-4, max=2.0)
        direction = self.dominant_angle + spread * torch.tanh(self.direction_offsets)
        weight = amplitude ** 2
        weight = weight / (weight.sum() + 1e-6)
        vel_u = (weight * speed * torch.cos(direction)).sum()
        vel_v = (weight * speed * torch.sin(direction)).sum()
        return torch.stack([vel_u, vel_v])  # [2], world-units / time-unit

    @staticmethod
    def _scalar_time(reference, time_value):
        if torch.is_tensor(time_value):
            return time_value.to(device=reference.device, dtype=reference.dtype)
        return torch.tensor(float(time_value), device=reference.device, dtype=reference.dtype)

    @staticmethod
    def _build_band_ids(num_waves, num_bands):
        wave_ids = torch.arange(num_waves, dtype=torch.long)
        return torch.div(wave_ids * num_bands, max(1, num_waves), rounding_mode="floor")

    @staticmethod
    def _build_band_frequency(w1, w2, num_bands):
        low = max(0.15, min(float(w1), float(w2)) * 0.45)
        high = max(low * 1.5, max(float(w1), float(w2)) * 1.8)
        if num_bands == 1:
            return torch.tensor([(low + high) * 0.5], dtype=torch.float32)
        return torch.logspace(math.log10(low), math.log10(high), steps=num_bands, dtype=torch.float32)

    @staticmethod
    def _build_band_amplitude(A1, A2, num_bands):
        base = 0.6 * float(A1) + 0.4 * float(A2)
        band_ids = torch.arange(1, num_bands + 1, dtype=torch.float32)
        falloff = torch.pow(band_ids, -1.15)
        amp = base * falloff
        amp[0] = max(float(A1), 1e-4)
        if num_bands > 1:
            amp[1] = max(0.65 * float(A1) + 0.35 * float(A2), 1e-4)
        return torch.clamp(amp, min=1e-4)

    def sample_local_fields(self, x_world, y_world):
        grid = self._build_local_grid(x_world, y_world)
        amp_scale = 1.0 + 0.22 * torch.tanh(self._sample_grid_field(self.local_amp_field, grid))
        angle_offset = 0.28 * torch.tanh(self._sample_grid_field(self.local_angle_field, grid))
        phase_offset = 0.40 * torch.tanh(self._sample_grid_field(self.local_phase_field, grid))
        detail_scale = 1.0 + 0.30 * torch.tanh(self._sample_grid_field(self.local_detail_field, grid))
        variation = 0.55 * (amp_scale - 1.0).abs() + 0.85 * angle_offset.abs() + 0.45 * phase_offset.abs()
        return {
            "amp_scale": amp_scale,
            "angle_offset": angle_offset,
            "phase_offset": phase_offset,
            "detail_scale": detail_scale,
            "variation": variation,
        }

    def _build_local_grid(self, x_world, y_world):
        coord_min = self.coord_min.to(device=x_world.device, dtype=x_world.dtype)
        coord_max = self.coord_max.to(device=x_world.device, dtype=x_world.dtype)
        extent = torch.clamp(coord_max - coord_min, min=1e-4)
        grid_x = 2.0 * (x_world - coord_min[0]) / extent[0] - 1.0
        grid_y = 2.0 * (y_world - coord_min[1]) / extent[1] - 1.0
        grid = torch.stack((grid_x.clamp(-1.0, 1.0), grid_y.clamp(-1.0, 1.0)), dim=-1)
        return grid.unsqueeze(0)

    def normalized_world_coords(self, x_world, y_world):
        coord_min = self.coord_min.to(device=x_world.device, dtype=x_world.dtype)
        coord_max = self.coord_max.to(device=x_world.device, dtype=x_world.dtype)
        extent = torch.clamp(coord_max - coord_min, min=1e-4)
        x_norm = 2.0 * (x_world - coord_min[0]) / extent[0] - 1.0
        y_norm = 2.0 * (y_world - coord_min[1]) / extent[1] - 1.0
        x_norm_dx = (2.0 / extent[0]).to(device=x_world.device, dtype=x_world.dtype)
        y_norm_dy = (2.0 / extent[1]).to(device=y_world.device, dtype=y_world.dtype)
        return x_norm, y_norm, x_norm_dx, y_norm_dy

    @staticmethod
    def _sample_grid_field(field, grid):
        sampled = F.grid_sample(field, grid, mode="bilinear", padding_mode="border", align_corners=True)
        return sampled.squeeze(0).squeeze(0)

    @staticmethod
    def _resolve_coord_bounds(coord_min, coord_max):
        if coord_min is None or coord_max is None:
            return (
                torch.tensor([-1.0, -1.0], dtype=torch.float32),
                torch.tensor([1.0, 1.0], dtype=torch.float32),
            )
        coord_min = torch.as_tensor(coord_min, dtype=torch.float32).view(2)
        coord_max = torch.as_tensor(coord_max, dtype=torch.float32).view(2)
        safe_extent = torch.clamp(coord_max - coord_min, min=1e-4)
        return coord_min, coord_min + safe_extent

    @staticmethod
    def _inv_softplus(value):
        value = torch.clamp(value, min=1e-4)
        return torch.log(torch.expm1(value))

    @staticmethod
    def _inv_sigmoid(value):
        value = torch.clamp(value, min=1e-4, max=1.0 - 1e-4)
        return torch.log(value / (1.0 - value))

    def sanitize_parameters_(self, fallback_state=None):
        with torch.no_grad():
            for name, param in self.named_parameters():
                if not torch.is_tensor(param):
                    continue
                if torch.isnan(param).any() or torch.isinf(param).any():
                    if fallback_state is not None and name in fallback_state:
                        safe_value = fallback_state[name].to(device=param.device, dtype=param.dtype)
                        mask = torch.isnan(param) | torch.isinf(param)
                        param.copy_(torch.where(mask, safe_value, param))
                    else:
                        param.copy_(torch.nan_to_num(param, nan=0.0, posinf=0.0, neginf=0.0))

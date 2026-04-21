# Underwater Quickstart

This project uses a two-stage workflow:

1. Train a clean PGSR backbone.
2. Train the underwater rendering branch on top of that backbone.

Activate your environment first:

```bash
conda activate pgsr
```

## 1. Train Clean PGSR Backbone

Example with the `garden` scene:

```bash
python trainers/train_clean_pgsr.py \
  -s data/garden \
  -m output/garden_clean \
  -r 4
```

This creates the clean PGSR model in:

```bash
output/garden_clean
```

## 2. Train Underwater Branch

Use the same `-m` directory so stage 2 can load the stage-1 PGSR checkpoint:

```bash
python trainers/train_underwater_pgsr.py \
  -s data/garden \
  -m output/garden_clean \
  -r 4 \
  --render_mode underwater_pgsr \
  --enable_underwater \
  --enable_surface \
  --enable_caustic \
  --underwater_iterations 10000 \
  --test_iterations 1000 5000 10000 \
  --save_iterations 1000 5000 10000
```

Underwater checkpoints are saved to:

```bash
output/garden_clean/underwater
```

## 3. Render Underwater Images

```bash
python render.py \
  -s data/garden \
  -m output/garden_clean \
  -r 4 \
  --iteration -1 \
  --render_mode underwater_pgsr \
  --enable_underwater \
  --enable_surface \
  --enable_caustic
```

Typical outputs:

```bash
output/garden_clean/train/underwater_<iter>/final
output/garden_clean/train/underwater_<iter>/debug
```

The `debug` folder includes intermediate maps such as:

- `scene_rgb`
- `scene_depth`
- `scene_normal`
- `transmission`
- `underwater_rgb`
- `surface_height`
- `caustic`
- `final_rgb`

## 4. Render Underwater Video

```bash
python scripts/render_underwater_video.py \
  -s data/garden \
  -m output/garden_clean \
  -r 4 \
  --iteration -1 \
  --render_mode underwater_pgsr \
  --enable_underwater \
  --enable_surface \
  --enable_caustic \
  --video_frames 120 \
  --video_fps 24 \
  --video_camera_index 0
```

Video output:

```bash
output/garden_clean/video_underwater
```

## 5. Interactive Viser Viewer

Use Viser to tune underwater parameters with sliders and inspect the scene in 3D:

```bash
python scripts/view_underwater_viser.py \
  -s data/garden \
  -m output/garden_clean \
  -r 4 \
  --iteration -1 \
  --render_mode underwater_pgsr \
  --enable_underwater \
  --enable_surface \
  --enable_caustic \
  --viewer_use_free_camera 1 \
  --viewer_res_moving 640 \
  --viewer_res_static 1280 \
  --host 0.0.0.0 \
  --port 8080
```

Optional speed-up flags:

```bash
--viewer_preview_scale 0.5
--viewer_target_fps 10
```

## Resolution Notes

Use `-r` as the main resolution control:

- `-r 8`: fastest, lowest memory
- `-r 4`: recommended default
- `-r 2`: higher quality
- `-r 1`: full resolution
- `-r -1`: PGSR automatic behavior

## Scene Switch

To use another dataset, replace `garden` with your scene name:

```bash
-s data/<your_scene>
-m output/<your_scene>_clean
```

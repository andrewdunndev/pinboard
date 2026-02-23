# Pinboard

Ken Burns-style video generator that composites photos as vintage Polaroids on
panoramic backgrounds, with smooth camera movement and music.

Drop your photos, a wide background image, and some music into `input/` — get
a 4K video back.

## Quick Start

```bash
make venv          # create virtual environment
# place assets in input/ (see below)
make render        # full pipeline (~15 min)
```

### Input Structure

```
input/
  photos/         # Photos (jpg, jpeg, png — any resolution)
  background/     # Panoramic background image(s) (PNG, ideally 8K+)
  audio/          # Up to three music tracks (any ffmpeg-supported format)
```

Photos are auto-classified as portrait or landscape from dimensions. EXIF
orientation is applied automatically.

### Two-Background Mode

Place two PNG files in `input/background/`. They are matched by alphabetical
sort order (first file = segment 1, second = segment 2). Music tracks in
`input/audio/` are also matched alphabetically: the first two play over
segments 1 and 2 with a crossfade, and an optional third plays over the end
title card.

Photos are split evenly across both backgrounds.

### Output

`output/pinboard.mp4` (or `output/test.mp4` in test mode).

## How It Works

`pipeline.py` is a single-file pipeline with six stages:

| Stage | What it does |
|-------|-------------|
| **1. Source Scan** | Detect photos, backgrounds, and music files in `input/` |
| **2. Load Background** | Open panoramic image(s), resize to canonical dimensions |
| **3. Layout** | Scatter photos using Poisson disk sampling, SAT-based overlap detection, and force-based relaxation |
| **4. Weathering** | Age each photo — desaturation, warm tint, sepia, vignette, paper texture — then frame as a Polaroid |
| **5. Composite** | Paste all Polaroids onto the background at computed positions and rotations |
| **6. Video** | Frame-by-frame Ken Burns render piped directly to ffmpeg |

### Video Structure

The camera follows a 4-phase path over each background:

```
Phase 1 — Wide establishing shot
           Full background visible, gentle drift

Phase 2 — Zoom in
           Cosine-eased zoom to close_zoom_pct of the background

Phase 3 — Zig-zag meander
           Camera pans across the image visiting clusters of photos
           Cosine easing with loiter (dwell) at each waypoint

Phase 4 — Zoom out
           Pull back to wide shot, fade to black
```

With two backgrounds the camera traverses left-to-right on the first,
right-to-left on the second (a "return trip"). Audio crossfades between
tracks. An end title card fades in over the final music track.

## Build Commands

```bash
make venv          # create virtual environment
make render        # full pipeline (~15 min with optimizations)
make test          # 120s test clip (~2 min)
make layout        # layout stage only (preview placement)
make composite     # layout + composite (preview final image)
make clean         # remove output/
make clean-all     # remove output/ and .venv/
```

## Customization

All parameters live in the `CONFIG` dict at the top of `pipeline.py`.

Key things to tune:

| Parameter | What it controls |
|-----------|-----------------|
| `video_width`, `video_height` | Output video resolution (default 4K) |
| `close_zoom_pct` | How far the camera zooms in during the meander |
| `desaturation`, `sepia_strength`, `vignette_intensity` | Photo weathering intensity |
| `polaroid_border`, `polaroid_bottom`, `min_gap` | Frame border widths and spacing |
| `secs_per_waypoint`, `loiter_ratio` | Camera timing and dwell at each waypoint |
| `title_lines`, `title_font_size` | End title card text and styling |

## Dependencies

- Python 3.10+
- [Pillow](https://pillow.readthedocs.io/) — image manipulation
- [NumPy](https://numpy.org/) — array operations for weathering and rendering
- [ffmpeg 7+](https://ffmpeg.org/) — video encoding and audio mixing

```bash
pip install numpy Pillow
brew install ffmpeg   # or your system's package manager
```

## Development

This project is developed on [GitLab](https://gitlab.com/dunn.dev/pinboard), where issues, merge requests, and CI/CD live. It's mirrored to [GitHub](https://github.com/andrewdunndev/pinboard) for discoverability — if you're solving a similar problem or want to discuss approaches, please reach out!

## License

MIT

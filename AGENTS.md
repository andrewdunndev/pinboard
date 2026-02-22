# Pinboard — AI Agent Context

## Project Overview

Ken Burns-style video generator that composites photos as vintage Polaroid
snapshots on panoramic backgrounds. Vintage photo wall — aged Polaroid
snapshots on a panoramic background with warm weathering effects, deliberate
pacing, and smooth camera movement.

## Project Structure

```
pinboard/
  input/
    photos/           # Source photos (jpg/jpeg, gitignored)
    background/       # 8K panoramic PNG(s) (gitignored) + README.md
    audio/            # Music tracks as .mp4/.m4a/.mp3 (gitignored)
  output/             # Generated artifacts (gitignored)
  pipeline.py         # Single-file pipeline (all six stages)
  fonts/              # Bundled EB Garamond (SIL OFL licensed)
    EBGaramond.ttf
    OFL.txt
  Makefile            # Build targets
  pyproject.toml      # Python packaging + ruff config
  requirements.txt    # Python dependencies (numpy, Pillow)
  README.md           # Project overview, quick start, and customization guide
  icon.png            # Project icon
  LICENSE             # MIT
  AGENTS.md           # This file
```

## Key Files

- **pipeline.py** — Single-file pipeline with `CONFIG` dict at top (lines 55-168)
  containing all tunable parameters. Six stages:
  1. `scan_sources()` — detect photos, backgrounds, and music files
  2. `load_backgrounds()` — open and resize 8K panoramic images
  3. `generate_layout()` — Poisson disk scatter + force-based relaxation
  4. `weather_photo()` / `create_polaroid()` — age photos with warm tint,
     sepia, vignette, and Polaroid border
  5. `composite_photos()` — paste all Polaroids onto the background
  6. `create_video()` — frame-by-frame Ken Burns render piped to ffmpeg
- **fonts/EBGaramond.ttf** — Bundled serif font. Cross-platform font loading
  tries this first, then system Georgia, then default.
- **Makefile** — Build targets (see below).

## Build Commands

```bash
make venv       # Create virtual environment
make render     # Full pipeline (~15 min)
make test       # 120s test clip (~2 min), reuses existing composite
make layout     # Generate layout.json only
make composite  # Generate layout + composited image only
make clean      # Remove output/
make clean-all  # Remove output/ + .venv/
```

Pipeline CLI flags: `--test`, `--video-only`, `--layout-only`, `--composite-only`.

## Lint

```bash
ruff check pipeline.py
```

## Source Assets

### Background Images
- Located in `input/background/`
- 8K panoramic PNG(s), resized to 12624x4736 at load time
- Supports one or two backgrounds (matched by filename keywords)
- Has keep-out zones for corner badge areas (configurable in CONFIG)

### Photos
- Located in `input/photos/` (`.jpg`, `.jpeg`, `.png`)
- Auto-classified as portrait or landscape by pixel dimensions
- EXIF orientation auto-applied via `ImageOps.exif_transpose()`
- Split proportionally across backgrounds based on song duration

### Music
- Located in `input/audio/` as `.mp4`, `.m4a`, or `.mp3` files
- Matched by alphabetical sort order (track_1, track_2, optional track_3)
- Supports 2-3 tracks with automatic crossfade mixing
- Trailing silence is detected and stripped via `detect_audio_end()`

## Video Structure

```
Segment 1   Music track 1, camera L→R meander
             Wide shot → zoom in → zig-zag across waypoints → zoom out
             → fade to black

Segment 2   Music track 2, camera R→L meander (return trip)
             Fade from black → zoom in → zig-zag → zoom out → hold wide
             → fade to black

Title Card   Optional third track
             White text on black, centered, fade in → hold → fade out
```

Camera path has four phases per segment: static wide shot, smooth zoom in,
zig-zag meander via waypoints with eased loiter, zoom out + pan back.

## Dependencies

- Python 3.10+ with venv
- Pillow (PIL) — image manipulation
- NumPy — array operations for weathering and frame rendering
- ffmpeg 7+ — video encoding, audio mixing
- ffprobe — audio duration detection (bundled with ffmpeg)

## Key Design Decisions

- **SAT overlap detection** — Separating Axis Theorem for rotated rectangles
  (`rect_corners` + `frames_overlap`) instead of AABB, since frames are rotated
- **Poisson disk sampling** — Bridson's algorithm for initial placement with
  configurable retry/shrink, followed by force-based relaxation to resolve overlaps
- **Frame-by-frame rendering** — NumPy array crops piped to ffmpeg stdin instead
  of ffmpeg's `zoompan` filter, which couldn't handle the 8K canvas or eased
  camera paths
- **amix over acrossfade** — ffmpeg's `acrossfade` filter produced truncated
  output (126s shorter); replaced with `amix` + `apad` + individual fade filters
- **Trailing silence detection** — `silencedetect` filter strips padded silence
  from music containers (e.g. 194s container → 143s content) to prevent dead air
  during crossfades
- **Threaded frame pipeline** — Producer-consumer pattern with `ThreadPoolExecutor`
  (4 workers, 8-frame buffer) for frame preparation ahead of ffmpeg consumption

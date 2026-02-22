#!/usr/bin/env python3
"""
Pinboard — Ken Burns Video Pipeline

Generates a Ken Burns-style pan across a panoramic background with photos
composited as aged Polaroid snapshots, set to music.

Usage:
    python pipeline.py                  # Full pipeline: layout + composite + video
    python pipeline.py --test           # 120s test clip (fast preset, no audio)
    python pipeline.py --video-only     # Skip layout/composite, reuse existing
    python pipeline.py --layout-only    # Generate layout.json only
    python pipeline.py --composite-only # Generate layout + composited image only

The pipeline has six stages:
    1. Source Scan     — detect photos, background, and music files
    2. Load Background — open the panoramic image(s)
    3. Layout          — Poisson disk scatter + force-based relaxation
    4. Weathering      — age photos with warm tint, sepia, vignette
    5. Composite       — paste all Polaroids onto the background
    6. Video           — frame-by-frame Ken Burns render piped to ffmpeg
"""

import json
import math
import os
import random
import subprocess
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance, ImageFont, ImageOps


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# All tunable parameters live here. The pipeline reads from this dict so you
# can adjust any value without hunting through the code. Parameters are
# grouped by the pipeline stage they affect.
#
# If you're adapting this for your own project, the most likely things to
# change are:
#   - Frame sizes (portrait_width/height, landscape_width/height)
#   - Photo weathering intensity (desaturation, sepia_strength)
#   - Camera timing (phase durations, secs_per_waypoint)
#   - close_zoom_pct (how far the camera zooms in during the meander)
# ---------------------------------------------------------------------------
CONFIG = {
    # --- Video output ---
    "fps": 30,
    "video_width": 3840,          # 4K UHD
    "video_height": 2160,
    "crf_production": 18,         # High quality (lower = bigger file, better quality)
    "crf_test": 23,               # Good enough for previewing camera path
    "preset_production": "slow",  # Slower encode = smaller file at same quality
    "preset_test": "fast",        # Quick encode for iteration
    "test_duration": 120.0,       # Seconds of video in --test mode

    # --- Polaroid frame sizes (pixels) ---
    # These were iterated down from 575x766 to fit 50+ frames in the available
    # placement zone without excessive overlap. The aspect ratios match
    # standard 4x6 / 6x4 prints.
    "portrait_width": 495,
    "portrait_height": 660,
    "landscape_width": 660,
    "landscape_height": 495,
    "polaroid_border": 20,        # Top and side border width
    "polaroid_bottom": 50,        # Thicker bottom border (classic Polaroid look)
    "min_gap": 40,                # Minimum pixel gap between frame edges

    # --- Layout ---
    "random_seed": 42,
    "margin_x": 400,              # Horizontal margin from canvas edge
    "margin_y_pct": (0.15, 0.85), # Vertical placement band (% of canvas height)
                                  # Wider than 5-95% caused photos to fall off map
                                  # edges; narrower than 20-80% didn't fit 50+ frames.

    # Max scale/rotation for bounding box calculation. Individual frames get
    # weighted random values within these bounds (see generate_layout).
    "max_scale": 1.10,
    "max_rotation_deg": 10,

    # Keep-out zones: areas of the background that should remain uncovered.
    # Each zone is an (x1, y1, x2, y2) bounding box in canvas pixels.
    # Set to [] if your background has no areas to protect.
    "keepout_zones": [],
    "keepout_padding": 300,        # Extra margin around keepout zones

    # One photo intentionally overlaps a keep-out zone to make the layout
    # feel hand-placed. Set to None to disable.
    "badge_overlap_zone": None,

    # Relaxation parameters: the force-based algorithm that pushes overlapping
    # frames apart after Poisson disk placement.
    "relaxation_max_iterations": 200,
    "relaxation_push_force": 30,   # Pixels of push per overlap per iteration
    "relaxation_boundary_margin": 350,

    # Poisson disk sampling parameters
    "poisson_candidates_per_point": 50,  # Standard Bridson's algorithm value
    "poisson_max_retries": 8,            # Attempts with decreasing min_dist
    "poisson_retry_shrink": 0.92,        # Multiply min_dist by this each retry

    # --- Photo weathering ---
    # These values were dialed back from initial stronger settings that made
    # photos look muddy and washed out. The current values give a subtle
    # vintage warmth without destroying the original photo quality.
    "desaturation": 0.20,         # Blend 20% toward grayscale
    "red_boost": 1.10,            # Warm shift: boost reds 10%
    "blue_reduce": 0.85,          # Warm shift: reduce blues 15%
    "sepia_color": (112, 66, 20), # Classic sepia tone (RGB)
    "sepia_strength": 0.15,       # 15% sepia overlay
    "vignette_intensity": 0.15,   # 15% edge darkening
    "border_color": (245, 230, 200, 255),  # Cream (not pure white — aged look)
    "border_noise_sigma": 8,      # Gaussian noise on border for paper texture
    "shadow_offset": 6,           # Drop shadow offset in pixels
    "shadow_blur": 8,             # Drop shadow blur radius
    "shadow_alpha": 80,           # Drop shadow opacity (0-255)

    # --- Camera path ---
    # The Ken Burns effect has four phases:
    #   Phase 1: Static wide shot (establishes the full map concept)
    #   Phase 2: Smooth zoom in toward the left side
    #   Phase 3: Zig-zag meander across the map via waypoints
    #   Phase 4: Zoom out, pan back left, fade to black
    "phase1_duration": 3,         # Static wide shot (seconds)
    "phase2_duration": 10,        # Zoom in (seconds)
    "zoom_out_duration": 6,       # Phase 4: zoom out at right end
    "pan_back_duration": 8,       # Phase 4: pan back left while zoomed out
    "fade_in_duration": 1,        # Video fade from black (seconds)
    "fade_out_duration": 4,       # Video fade to black (seconds, completes before title card)
    "close_zoom_pct": 0.40,       # Crop width as % of canvas when zoomed in
    "secs_per_waypoint": 32,      # Time spent traversing between waypoints
    "loiter_ratio": 0.20,         # Fraction of waypoint time spent dwelling
    "camera_seed": 99,            # Separate seed for camera Y-jitter
    "y_range_scale": 0.8,         # How far up/down the camera zig-zags

    # --- Title card ---
    "title_lines": [
        "Your Title Here",
    ],
    "title_fade_in": 2.0,         # Seconds to fade title text in
    "title_hold": 4.0,            # Seconds to hold title at full opacity
    "title_fade_out": 2.0,        # Seconds to fade title text out
    "title_font_size": 80,        # Font size in pixels

    # --- Audio ---
    "crossfade_duration": 3,      # Seconds of overlap between tracks
    "audio_bitrate": "192k",

    # --- Performance ---
    # These settings control the threaded frame preparation pipeline.
    "render_workers": 4,          # Thread pool size for frame preparation
    "render_buffer_ahead": 8,     # Frames to prepare ahead of ffmpeg consumption
}

# ---------------------------------------------------------------------------
# Derived paths — all relative to this script's location
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent
INPUT_DIR = ROOT / "input"
PHOTOS_DIR = INPUT_DIR / "photos"
BACKGROUND_DIR = INPUT_DIR / "background"
AUDIO_DIR = INPUT_DIR / "audio"
OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Stage 1: Source Scanning
# ---------------------------------------------------------------------------
# Scans input directories for photos, background image(s), and music files.
# Photos are classified as portrait or landscape by their pixel dimensions.
# Backgrounds and music are matched by alphabetical sort order.
# ---------------------------------------------------------------------------

def scan_sources():
    """Scan input directories for all pipeline assets.

    Backgrounds: all PNG files in input/background/, sorted alphabetically.
    One background = single-segment mode. Two = two-segment mode with
    fade-to-black transition at the song changeover.

    Music: all audio files in input/audio/, sorted alphabetically.
    The first two tracks play over segments 1 and 2 with a crossfade.
    An optional third track plays over the end title card.

    Photos are split proportionally between backgrounds (50/50 by default).

    Returns:
        backgrounds: list of Paths to background PNGs (1 or 2)
        portrait_photos: list of Paths to portrait-orientation photos
        landscape_photos: list of Paths to landscape-orientation photos
        music: dict with keys "track_1", "track_2", and optionally "track_3"
    """
    print("=== Scanning sources ===")
    backgrounds = []
    photos = []
    music = {}

    for f in sorted(BACKGROUND_DIR.iterdir()):
        if f.is_file() and f.suffix.lower() == ".png":
            backgrounds.append(f)

    for f in sorted(PHOTOS_DIR.iterdir()):
        if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png"):
            photos.append(f)

    audio_files = sorted(
        f for f in AUDIO_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in (".mp4", ".m4a", ".mp3")
    )
    for i, f in enumerate(audio_files):
        music[f"track_{i + 1}"] = f

    portrait = []
    landscape = []
    for p in photos:
        try:
            img = Image.open(p)
            img = ImageOps.exif_transpose(img)
            w, h = img.size
            img.close()
            if h > w:
                portrait.append(p)
            else:
                landscape.append(p)
        except Exception:
            continue

    bg_names = [b.name for b in backgrounds]
    print(f"  Backgrounds: {len(backgrounds)} ({', '.join(bg_names)})")
    print(f"  Photos: {len(portrait)} portrait, {len(landscape)} landscape ({len(portrait) + len(landscape)} total)")
    print(f"  Music: {len(music)} tracks ({', '.join(f.name for f in audio_files)})")

    return backgrounds, portrait, landscape, music


def load_backgrounds(bg_paths, target_size=None):
    """Load background images and optionally resize to match.

    When two backgrounds have different dimensions (common with AI-generated
    images), we resize both to match the smaller one. This ensures consistent
    zoom levels and camera behavior across the transition.

    Args:
        bg_paths: List of Paths to background PNGs
        target_size: Optional (w, h) tuple to resize to. If None and multiple
                     backgrounds, uses the smallest dimensions.

    Returns:
        List of PIL Images, all at the same dimensions
    """
    print("=== Loading backgrounds ===")
    images = []
    for p in bg_paths:
        img = Image.open(p)
        print(f"  {p.name}: {img.size[0]}x{img.size[1]} ({img.size[0] * img.size[1] / 1e6:.1f}MP)")
        images.append(img)

    if len(images) > 1 and target_size is None:
        # Use the smallest dimensions to avoid upscaling
        target_w = min(img.size[0] for img in images)
        target_h = min(img.size[1] for img in images)
        target_size = (target_w, target_h)

    if target_size:
        resized = []
        for img in images:
            if img.size != target_size:
                print(f"  Resizing {img.size[0]}x{img.size[1]} -> {target_size[0]}x{target_size[1]}")
                img = img.resize(target_size, Image.LANCZOS)
            resized.append(img)
        images = resized

    print(f"  Canvas: {images[0].size[0]}x{images[0].size[1]}")
    return images


# ---------------------------------------------------------------------------
# Stage 3: Layout — Poisson Disk Sampling + Force-Based Relaxation
# ---------------------------------------------------------------------------
# This is the most iterated part of the pipeline. The goal is to place 50+
# photo frames on the map with an organic, "pinned over time" look — not a
# rigid grid.
#
# Iteration history:
#   1. Grid layout (v1): Rigid rows/columns. Looked like a PowerPoint slide.
#   2. Pure Poisson disk: Organic scatter, but the minimum distance needed to
#      guarantee zero overlaps was too large to fit 50+ frames in the
#      available space.
#   3. Overlap detection experiments:
#      - AABB (axis-aligned bounding box): Doesn't account for rotation.
#        Massive false positives — frames that don't visually overlap were
#        reported as overlapping because their AABBs intersect.
#      - Circle-based distance: Too conservative. Treats each rectangular
#        frame as its circumscribed circle, wasting ~36% of usable space.
#      - Center-to-center distance threshold: Same problem. Couldn't pack
#        densely enough for 50+ frames.
#      - SAT with arbitrary safe distance: Worked, but the initial Poisson
#        disk distance was hard to tune.
#   4. Final approach — SAT with diagonal-based distance:
#      Calculate the diagonal of the worst-case bounding box (maximum scale,
#      maximum rotation), use 80% of that as the Poisson disk minimum
#      distance. This is tight enough to fit 50+ frames but close enough to
#      collision-free that force-based relaxation converges quickly (~90
#      iterations). The SAT check uses precise rotated-rectangle collision
#      with MIN_GAP built into the projection test.
#
# The relaxation pass uses a simple force-based algorithm: for each pair of
# overlapping frames, push them apart along the vector between their centers.
# Boundary clamping prevents frames from drifting off the map. The algorithm
# typically converges to 0 overlaps within 90-100 iterations.
# ---------------------------------------------------------------------------

def poisson_disk_sample(width, height, min_dist, margin_x, margin_y_top,
                        margin_y_bottom, num_points, rng):
    """Generate candidate positions using Poisson disk sampling (Bridson's algorithm).

    Uses three seed points spread across the horizontal span to ensure even
    coverage. The keepout zones are avoided with configurable padding.

    Args:
        width, height: Canvas dimensions
        min_dist: Minimum distance between any two points
        margin_x: Horizontal margin from canvas edge
        margin_y_top, margin_y_bottom: Vertical placement bounds
        num_points: Target number of points (algorithm may produce more)
        rng: random.Random instance for reproducibility

    Returns:
        List of (x, y) tuples — candidate positions
    """
    cell_size = min_dist / math.sqrt(2)
    grid = {}

    def grid_key(x, y):
        return (int((x - margin_x) / cell_size), int((y - margin_y_top) / cell_size))

    def is_valid(px, py):
        if px < margin_x or px > width - margin_x:
            return False
        if py < margin_y_top or py > margin_y_bottom:
            return False
        # Check neighborhood in the spatial hash grid — standard Bridson's
        # algorithm uses a 5x5 cell neighborhood to find nearby points.
        gx, gy = grid_key(px, py)
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                key = (gx + dx, gy + dy)
                if key in grid:
                    ox, oy = grid[key]
                    if math.sqrt((px - ox) ** 2 + (py - oy) ** 2) < min_dist:
                        return False
        return True

    def in_keepout(px, py):
        """Check if a point falls within any keepout zone (with padding)."""
        pad = CONFIG["keepout_padding"]
        for x1, y1, x2, y2 in CONFIG["keepout_zones"]:
            if x1 - pad < px < x2 + pad and y1 - pad < py < y2 + pad:
                return True
        return False

    points = []
    active = []

    # Seed points: three evenly-spaced positions across the horizontal span.
    # This prevents the Poisson disk from starting in one corner and running
    # out of space before reaching the other side.
    seeds = [
        (margin_x + 500, (margin_y_top + margin_y_bottom) / 2),
        (width / 2, (margin_y_top + margin_y_bottom) / 2),
        (width - margin_x - 500, (margin_y_top + margin_y_bottom) / 2),
    ]
    for sx, sy in seeds:
        if is_valid(sx, sy) and not in_keepout(sx, sy):
            points.append((sx, sy))
            active.append((sx, sy))
            grid[grid_key(sx, sy)] = (sx, sy)

    # Standard Bridson's: for each active point, try k candidates at random
    # angles and distances in [min_dist, 1.5 * min_dist]. If none are valid,
    # remove the point from the active list.
    k = CONFIG["poisson_candidates_per_point"]
    while active:
        idx = rng.randint(0, len(active) - 1)
        bx, by = active[idx]
        found = False
        for _ in range(k):
            angle = rng.uniform(0, 2 * math.pi)
            dist = rng.uniform(min_dist, min_dist * 1.5)
            nx = bx + dist * math.cos(angle)
            ny = by + dist * math.sin(angle)
            if is_valid(nx, ny) and not in_keepout(nx, ny):
                points.append((nx, ny))
                active.append((nx, ny))
                grid[grid_key(nx, ny)] = (nx, ny)
                found = True
                break
        if not found:
            active.pop(idx)

    return points


def rect_corners(cx, cy, w, h, angle_deg):
    """Compute the four corners of a rectangle rotated about its center.

    Used by the SAT overlap test to get precise rotated-rectangle geometry.
    """
    angle = math.radians(angle_deg)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    hw, hh = w / 2, h / 2
    corners = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
    return [(cx + x * cos_a - y * sin_a, cy + x * sin_a + y * cos_a) for x, y in corners]


def frames_overlap(f1, f2):
    """Test whether two rotated Polaroid frames overlap using SAT.

    The Separating Axis Theorem (SAT) is the only overlap detection method
    that correctly handles rotated rectangles without being overly
    conservative. We tested AABB, circle-based, and center-distance
    approaches first — all either produced false positives (wasting layout
    space) or couldn't handle rotation.

    The MIN_GAP is incorporated directly into the projection comparison,
    ensuring frames maintain a minimum visual separation even when they
    don't technically intersect.

    How SAT works for two convex shapes:

        For each edge of both rectangles, compute the perpendicular
        (normal) axis. Project all corners of both rectangles onto that
        axis to get a 1D interval [min, max] per rectangle.

            Axis ──────────────────────────────────────►
                    [===R1===]         [===R2===]
                              ↑ gap ↑
                    If any axis shows a gap → separated (no overlap).
                    If ALL axes show overlap → collision.

        The min_gap widens each interval so frames maintain visual spacing
        even when edges don't touch.

    Args:
        f1, f2: Frame dicts with center_x, center_y, width, height,
                rotation_deg, and attachment keys.

    Returns:
        True if the frames overlap (including gap violation), False otherwise.
    """
    min_gap = CONFIG["min_gap"]
    border = CONFIG["polaroid_border"]
    bottom = CONFIG["polaroid_bottom"]

    def total_dims(f):
        b = bottom if f["attachment"] == "polaroid" else border
        tw = f["width"] + 2 * border
        th = f["height"] + border + b
        return tw, th

    tw1, th1 = total_dims(f1)
    tw2, th2 = total_dims(f2)
    c1 = rect_corners(f1["center_x"], f1["center_y"], tw1, th1, f1["rotation_deg"])
    c2 = rect_corners(f2["center_x"], f2["center_y"], tw2, th2, f2["rotation_deg"])

    def get_axes(corners):
        """Extract edge-normal axes for SAT projection."""
        axes = []
        for i in range(len(corners)):
            edge = (corners[(i + 1) % len(corners)][0] - corners[i][0],
                    corners[(i + 1) % len(corners)][1] - corners[i][1])
            length = math.sqrt(edge[0] ** 2 + edge[1] ** 2)
            if length > 0:
                axes.append((-edge[1] / length, edge[0] / length))
        return axes

    def project(corners, axis):
        """Project all corners onto an axis, return (min, max) interval."""
        dots = [c[0] * axis[0] + c[1] * axis[1] for c in corners]
        return min(dots), max(dots)

    # SAT: if we find ANY axis where the projections don't overlap (with gap),
    # the rectangles are separated. If all axes show overlap, they collide.
    for axis in get_axes(c1) + get_axes(c2):
        min1, max1 = project(c1, axis)
        min2, max2 = project(c2, axis)
        if max1 + min_gap < min2 or max2 + min_gap < min1:
            return False
    return True


def generate_layout(canvas_w, canvas_h, num_portrait, num_landscape):
    """Generate the complete frame layout for all photos.

    This is the core layout algorithm that evolved through multiple iterations:

    1. Compute the "safe distance" — the diagonal of the worst-case rotated
       bounding box. This is the minimum center-to-center distance that
       guarantees no overlap for the largest, most-rotated frame.
    2. Start Poisson disk sampling at 80% of safe distance. This intentionally
       allows some overlaps but produces much denser packing than 100%.
    3. If not enough candidates are generated, reduce min_dist by 8% and retry
       (up to 8 attempts).
    4. Select positions evenly spaced across the horizontal span (left to right,
       left to right).
    5. Optionally place one frame overlapping a keep-out zone (feels organic).
    6. Assign scale, rotation, and attachment style with weighted distributions.
    7. Run force-based relaxation to push any remaining overlaps apart.

    The relaxation typically converges in ~90 iterations. Frame sizes were
    reduced from the original 575x766 to 495x660 to fit 50+ frames.

    Args:
        canvas_w, canvas_h: Background image dimensions
        num_portrait: Number of portrait-orientation photos
        num_landscape: Number of landscape-orientation photos

    Returns:
        Layout dict with "canvas", "viewport", and "frames" keys.
        Also writes layout.json to the output directory.
    """
    print("=== Generating layout ===")
    rng = random.Random(CONFIG["random_seed"])
    total = num_portrait + num_landscape

    margin_x = CONFIG["margin_x"]
    margin_y_top = int(canvas_h * CONFIG["margin_y_pct"][0])
    margin_y_bottom = int(canvas_h * CONFIG["margin_y_pct"][1])

    # --- Compute the Poisson disk minimum distance ---
    # This is the key insight that made the layout work: use the diagonal of
    # the worst-case rotated bounding box as the basis for spacing.
    #
    # For a frame at maximum scale with maximum rotation:
    #   1. Compute total Polaroid dimensions (frame + border)
    #   2. Compute the axis-aligned bounding box of the rotated rectangle
    #   3. Take the diagonal of that bounding box
    #   4. Add MIN_GAP for visual separation
    #   5. Use 80% of that as the Poisson disk min_dist
    #
    # The 80% factor is critical: it allows the Poisson disk to pack more
    # tightly than strictly collision-free, relying on the relaxation pass
    # to resolve any overlaps. Without this, we couldn't fit 50+ frames.
    max_scale = CONFIG["max_scale"]
    max_rot = math.radians(5)  # Use 5 degrees for bounding calc (not the full 10)
    border = CONFIG["polaroid_border"]
    bottom = CONFIG["polaroid_bottom"]
    pw = int(CONFIG["portrait_width"] * max_scale) + 2 * border
    ph = int(CONFIG["portrait_height"] * max_scale) + border + bottom
    bw = abs(pw * math.cos(max_rot)) + abs(ph * math.sin(max_rot))
    bh = abs(pw * math.sin(max_rot)) + abs(ph * math.cos(max_rot))
    safe_dist = math.sqrt(bw ** 2 + bh ** 2) + CONFIG["min_gap"]
    min_dist = int(safe_dist * 0.80)

    usable_w = canvas_w - 2 * margin_x
    usable_h = margin_y_bottom - margin_y_top
    print(f"  Placement zone: x=[{margin_x}..{canvas_w - margin_x}], y=[{margin_y_top}..{margin_y_bottom}]")
    print(f"  Usable area: {usable_w}x{usable_h} ({usable_w * usable_h / 1e6:.1f}M px^2)")
    print(f"  Min distance between frame centers: {min_dist}px")

    # --- Poisson disk sampling with retry ---
    # If the first attempt doesn't produce enough candidates, we shrink the
    # minimum distance and try again. Each retry uses a different seed to
    # avoid getting stuck in the same configuration.
    max_retries = CONFIG["poisson_max_retries"]
    shrink = CONFIG["poisson_retry_shrink"]
    candidates = None
    for attempt in range(max_retries):
        candidates = poisson_disk_sample(
            canvas_w, canvas_h, min_dist,
            margin_x, margin_y_top, margin_y_bottom,
            total, rng
        )
        print(f"  Attempt {attempt + 1}: {len(candidates)} candidates at min_dist={min_dist} (need {total})")
        if len(candidates) >= total:
            break
        min_dist = int(min_dist * shrink)
        rng = random.Random(CONFIG["random_seed"] + attempt + 1)

    # Sort candidates left-to-right
    candidates.sort(key=lambda p: p[0])

    # Select positions evenly spaced across the candidate pool to maintain
    # uniform horizontal distribution.
    if len(candidates) > total:
        step = len(candidates) / total
        selected = [candidates[int(i * step)] for i in range(total)]
    else:
        selected = candidates[:total]

    if CONFIG["badge_overlap_zone"]:
        bx1, by1, bx2, by2 = CONFIG["badge_overlap_zone"]
        badge_point = (rng.uniform(bx1, bx2), rng.uniform(by1, by2))
        if selected:
            closest_idx = min(range(len(selected)), key=lambda i: selected[i][0])
            selected[closest_idx] = badge_point
            print(f"  Overlap zone: frame at ({int(badge_point[0])}, {int(badge_point[1])})")

    # Re-sort after badge insertion
    selected.sort(key=lambda p: p[0])

    # --- Distribute landscape frames evenly across the timeline ---
    # Without this, landscape frames would cluster together. We space them
    # evenly with a small random offset for natural variation.
    land_spacing = total / max(num_landscape, 1)
    land_positions = set()
    for i in range(num_landscape):
        pos = int(i * land_spacing + rng.uniform(0, land_spacing * 0.5))
        pos = max(0, min(total - 1, pos))
        while pos in land_positions:
            pos = (pos + 1) % total
        land_positions.add(pos)

    # --- Assign properties to each frame ---
    # Scale and rotation use weighted distributions:
    #   Scale: 60% cluster near 1.0x (gaussian sigma=0.03), 25% are 0.90-0.97x,
    #          15% are 1.03-1.10x. Most frames look similar-sized, with
    #          occasional larger/smaller ones for visual interest.
    #   Rotation: 8% near-straight (0.5 deg), 42% slight tilt (gaussian
    #          sigma=2.5 deg), 30% moderate tilt (3-6 deg), 20% strong tilt
    #          (6-10 deg). This mimics photos pinned casually — most are
    #          roughly straight with occasional dramatic angles.
    #   Attachment: 60% polaroid (thick bottom border), 25% tape (uniform
    #          border), 15% pushpin (uniform border). Only affects border
    #          dimensions — visual tape/pin rendering was attempted but
    #          looked artificial and was removed.
    frames = []
    for i, (cx, cy) in enumerate(selected[:total]):
        is_landscape = i in land_positions

        # Scale distribution (weighted)
        r = rng.random()
        if r < 0.6:
            scale = rng.gauss(1.0, 0.03)
        elif r < 0.85:
            scale = rng.uniform(0.90, 0.97)
        else:
            scale = rng.uniform(1.03, 1.10)
        scale = max(0.90, min(1.10, scale))

        # Rotation distribution (weighted)
        r2 = rng.random()
        if r2 < 0.08:
            rotation = rng.uniform(-0.5, 0.5)
        elif r2 < 0.50:
            rotation = rng.gauss(0, 2.5)
        elif r2 < 0.80:
            rotation = rng.choice([-1, 1]) * rng.uniform(3, 6)
        else:
            rotation = rng.choice([-1, 1]) * rng.uniform(6, 10)
        rotation = max(-10, min(10, rotation))

        # Attachment style distribution
        r3 = rng.random()
        if r3 < 0.60:
            attachment = "polaroid"
        elif r3 < 0.85:
            attachment = "tape"
        else:
            attachment = "pushpin"

        if is_landscape:
            fw = int(CONFIG["landscape_width"] * scale)
            fh = int(CONFIG["landscape_height"] * scale)
            ftype = "landscape"
        else:
            fw = int(CONFIG["portrait_width"] * scale)
            fh = int(CONFIG["portrait_height"] * scale)
            ftype = "portrait"

        frames.append({
            "id": i + 1,
            "type": ftype,
            "center_x": int(cx),
            "center_y": int(cy),
            "width": fw,
            "height": fh,
            "rotation_deg": round(rotation, 1),
            "attachment": attachment,
        })

    # --- Count initial overlaps ---
    overlap_count = 0
    for i in range(len(frames)):
        for j in range(i + 1, len(frames)):
            if frames_overlap(frames[i], frames[j]):
                overlap_count += 1
    print(f"  Initial overlap: {overlap_count} pairs")

    # --- Force-based relaxation ---
    # For each pair of overlapping frames, compute a repulsion vector along
    # the line between their centers. The push force is constant (not
    # proportional to overlap depth) which prevents oscillation. Boundary
    # clamping keeps frames within the placement zone.
    max_iter = CONFIG["relaxation_max_iterations"]
    push = CONFIG["relaxation_push_force"]
    boundary = CONFIG["relaxation_boundary_margin"]
    for iteration in range(max_iter):
        moved = False
        for i in range(len(frames)):
            push_x, push_y = 0.0, 0.0
            for j in range(len(frames)):
                if i == j:
                    continue
                if frames_overlap(frames[i], frames[j]):
                    dx = frames[i]["center_x"] - frames[j]["center_x"]
                    dy = frames[i]["center_y"] - frames[j]["center_y"]
                    dist = math.sqrt(dx * dx + dy * dy)
                    if dist < 1:
                        dx, dy, dist = 1, 0, 1
                    push_x += dx / dist * push
                    push_y += dy / dist * push
            if abs(push_x) > 0.5 or abs(push_y) > 0.5:
                new_x = frames[i]["center_x"] + int(push_x)
                new_y = frames[i]["center_y"] + int(push_y)
                new_x = max(margin_x + boundary, min(canvas_w - margin_x - boundary, new_x))
                new_y = max(margin_y_top + boundary, min(margin_y_bottom - boundary, new_y))
                frames[i]["center_x"] = new_x
                frames[i]["center_y"] = new_y
                moved = True
        if not moved:
            print(f"  Relaxation converged at iteration {iteration + 1}")
            break

    # Verify final overlap count
    overlap_count = 0
    for i in range(len(frames)):
        for j in range(i + 1, len(frames)):
            if frames_overlap(frames[i], frames[j]):
                overlap_count += 1
    print(f"  Final overlap: {overlap_count} pairs after relaxation")

    # Re-sort and re-number left-to-right
    frames.sort(key=lambda f: f["center_x"])
    for i, f in enumerate(frames):
        f["id"] = i + 1

    layout = {
        "canvas": {"width": canvas_w, "height": canvas_h},
        "viewport": {"width": CONFIG["video_width"], "height": CONFIG["video_height"]},
        "frames": frames,
    }

    out_path = OUTPUT_DIR / "layout.json"
    with open(out_path, "w") as f:
        json.dump(layout, f, indent=2)

    port_count = sum(1 for fr in frames if fr["type"] == "portrait")
    land_count = sum(1 for fr in frames if fr["type"] == "landscape")
    print(f"  Placed {len(frames)} frames: {port_count} portrait, {land_count} landscape")
    print(f"  Layout -> {out_path}")
    return layout


# ---------------------------------------------------------------------------
# Stage 4: Photo Weathering & Polaroid Creation
# ---------------------------------------------------------------------------
# Each photo is aged to match the vintage map aesthetic:
#   - Partial desaturation (toward grayscale)
#   - Warm color shift (boost reds, reduce blues)
#   - Sepia overlay at low opacity
#   - Radial vignette (darken edges)
#
# The weathering intensity was dialed back from initial stronger settings.
# The first pass used 40% desaturation and 25% sepia, which made photos
# look muddy. Current values (20% desat, 15% sepia) give a subtle warmth
# that keeps the photos recognizable while fitting the vintage map style.
# ---------------------------------------------------------------------------

def weather_photo(photo, frame_w, frame_h):
    """Apply vintage weathering effects to a photo.

    All operations are done in numpy for performance. The photo is first
    resized to the target frame dimensions, then processed through four
    sequential effects that each modify the pixel array in-place.

    Args:
        photo: PIL Image (RGB)
        frame_w, frame_h: Target dimensions after weathering

    Returns:
        PIL Image with weathering applied
    """
    photo = photo.resize((frame_w, frame_h), Image.LANCZOS)
    arr = np.array(photo, dtype=np.float32)

    # Step 1: Partial desaturation — blend toward grayscale
    gray = np.mean(arr, axis=2, keepdims=True)
    desat = CONFIG["desaturation"]
    arr = arr * (1 - desat) + gray * desat

    # Step 2: Warm color shift — boost reds, reduce blues
    arr[:, :, 0] = np.clip(arr[:, :, 0] * CONFIG["red_boost"], 0, 255)
    arr[:, :, 2] = np.clip(arr[:, :, 2] * CONFIG["blue_reduce"], 0, 255)

    # Step 3: Sepia overlay — blend toward a sepia tone
    sepia = np.array(CONFIG["sepia_color"], dtype=np.float32)
    strength = CONFIG["sepia_strength"]
    arr = arr * (1 - strength) + sepia * strength

    # Step 4: Radial vignette — darken edges using a distance mask
    h, w = arr.shape[:2]
    cy, cx = h / 2, w / 2
    max_dist = math.sqrt(cx ** 2 + cy ** 2)
    y_coords, x_coords = np.ogrid[:h, :w]
    dist = np.sqrt((x_coords - cx) ** 2 + (y_coords - cy) ** 2) / max_dist
    vignette = 1.0 - CONFIG["vignette_intensity"] * dist
    arr *= vignette[:, :, np.newaxis]

    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def create_polaroid(photo_img, frame, rng_noise):
    """Create a weathered Polaroid with border, noise texture, and drop shadow.

    The Polaroid effect consists of:
      1. Weather the photo (warm tint, sepia, vignette)
      2. Create a cream-colored border (not white — aged paper look)
      3. Add Gaussian noise to the border for paper texture
      4. Paste the weathered photo into the border
      5. Add a soft drop shadow behind the Polaroid

    The border dimensions vary by attachment style:
      - "polaroid": thick bottom border (classic Polaroid caption area)
      - "tape" / "pushpin": uniform border on all sides

    Args:
        photo_img: PIL Image (RGB) — the original photo
        frame: Frame dict from the layout
        rng_noise: Integer seed for reproducible noise generation

    Returns:
        PIL Image (RGBA) with the complete Polaroid + shadow
    """
    fw, fh = frame["width"], frame["height"]
    weathered = weather_photo(photo_img, fw, fh)

    border = CONFIG["polaroid_border"]
    bottom = CONFIG["polaroid_bottom"] if frame["attachment"] == "polaroid" else border
    total_w = fw + 2 * border
    total_h = fh + border + bottom

    # Create cream border with paper-like noise texture
    cream = CONFIG["border_color"]
    polaroid = Image.new("RGBA", (total_w, total_h), cream)

    rstate = np.random.RandomState(rng_noise)
    noise = rstate.normal(0, CONFIG["border_noise_sigma"], (total_h, total_w, 3))
    noise_img = np.array(polaroid.convert("RGB"), dtype=np.float32) + noise
    noise_img = np.clip(noise_img, 0, 255).astype(np.uint8)
    polaroid = Image.fromarray(noise_img).convert("RGBA")
    alpha = Image.new("L", (total_w, total_h), 255)
    polaroid.putalpha(alpha)

    # Paste weathered photo into the border
    polaroid.paste(weathered, (border, border))

    # Add drop shadow: create a larger canvas, draw a semi-transparent
    # rectangle offset from center, blur it, then paste the Polaroid on top.
    shadow_pad = 12
    shadow = Image.new("RGBA",
                        (total_w + shadow_pad * 2, total_h + shadow_pad * 2),
                        (0, 0, 0, 0))
    shadow_rect = Image.new("RGBA", (total_w, total_h),
                             (0, 0, 0, CONFIG["shadow_alpha"]))
    shadow.paste(shadow_rect,
                  (shadow_pad + CONFIG["shadow_offset"],
                   shadow_pad + CONFIG["shadow_offset"]))
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=CONFIG["shadow_blur"]))
    shadow.paste(polaroid, (shadow_pad // 2, shadow_pad // 2), polaroid)

    return shadow


# ---------------------------------------------------------------------------
# Stage 5: Compositing
# ---------------------------------------------------------------------------

def composite_photos(background, layout, portrait_photos, landscape_photos,
                     output_path=None):
    """Paste all weathered Polaroids onto the background image.

    Photos are shuffled (deterministically) and matched to frame slots by
    orientation. Each Polaroid is rotated according to its layout rotation
    and pasted at its center position.

    Args:
        background: PIL Image — the 8K panoramic map
        layout: Layout dict from generate_layout
        portrait_photos: List of Paths to portrait photos
        landscape_photos: List of Paths to landscape photos
        output_path: Optional Path for the output file. Defaults to
            OUTPUT_DIR / "composited.png".

    Returns:
        Path to the saved composited image
    """
    print("=== Compositing photos ===")
    rng = random.Random(CONFIG["random_seed"])

    p_photos = list(portrait_photos)
    l_photos = list(landscape_photos)
    rng.shuffle(p_photos)
    rng.shuffle(l_photos)

    canvas = background.copy().convert("RGBA")
    p_idx = 0
    l_idx = 0

    for i, frame in enumerate(layout["frames"]):
        # Match photo orientation to frame type, with fallback
        if frame["type"] == "portrait" and p_idx < len(p_photos):
            photo_path = p_photos[p_idx]
            p_idx += 1
        elif frame["type"] == "landscape" and l_idx < len(l_photos):
            photo_path = l_photos[l_idx]
            l_idx += 1
        elif p_idx < len(p_photos):
            photo_path = p_photos[p_idx]
            p_idx += 1
        elif l_idx < len(l_photos):
            photo_path = l_photos[l_idx]
            l_idx += 1
        else:
            continue

        photo = ImageOps.exif_transpose(Image.open(photo_path)).convert("RGB")
        polaroid = create_polaroid(photo, frame, rng_noise=i)
        photo.close()

        # Rotate the Polaroid. Note: negative rotation because PIL's rotate
        # is counter-clockwise but our layout angles are clockwise-positive.
        rotated = polaroid.rotate(
            -frame["rotation_deg"], expand=True, resample=Image.BICUBIC
        )

        # Center the rotated Polaroid at the frame's layout position
        paste_x = frame["center_x"] - rotated.width // 2
        paste_y = frame["center_y"] - rotated.height // 2
        canvas.paste(rotated, (paste_x, paste_y), rotated)

        if (i + 1) % 10 == 0:
            print(f"  Composited {i + 1}/{len(layout['frames'])} photos")

    final = canvas.convert("RGB")
    out_path = output_path or (OUTPUT_DIR / "composited.png")
    final.save(out_path, quality=95)
    print(f"  Composited image -> {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Stage 6: Ken Burns Video Rendering
# ---------------------------------------------------------------------------
# The video is rendered by cropping the composited 8K image at each frame's
# camera position, resizing to 4K, and piping raw RGB frames to ffmpeg.
#
# Why not ffmpeg's zoompan filter?
#   We tried zoompan first. It caused aspect ratio distortion when panning
#   across a non-16:9 source image — the filter assumes the source matches
#   the output aspect ratio. Pan direction was also unreliable. Frame-by-
#   frame rendering with Pillow gives full control over the camera path.
#
# Camera path design:
#   Phase 1 (0-3s):     Static wide shot — establishes the full map concept
#   Phase 2 (3-13s):    Smooth zoom toward the left end
#   Phase 3 (13s-~460s): Zig-zag meander across the map via waypoints
#                        with loiter+inertia easing at each stop
#   Phase 4 (~460s-end): Zoom out at right end, pan back left, fade to black
#
# The meander uses waypoints that alternate Y positions (up/down zig-zag)
# to prevent the camera from just sliding monotonically left-to-right.
# The ease_with_loiter function creates a dwell at each waypoint with
# smooth acceleration/deceleration between them — like someone slowly
# browsing the map, pausing to look at photos, then moving on.
# ---------------------------------------------------------------------------

def ease_in_out(t):
    """Cosine easing — smooth acceleration and deceleration.

    Maps [0, 1] -> [0, 1] with zero velocity at both endpoints.
    """
    return (1 - math.cos(math.pi * max(0.0, min(1.0, t)))) / 2


def ease_with_loiter(t, loiter_ratio=None):
    """Easing function that dwells at start and end of each segment.

    The loiter_ratio controls what fraction of the segment time is spent
    stationary (split evenly between start and end). The remaining time
    is spent moving with cosine easing.

    For example, with loiter_ratio=0.35:
      - First 17.5% of time: stationary at start position
      - Middle 65% of time: smooth ease from start to end
      - Last 17.5% of time: stationary at end position

    This creates the feeling of someone pausing to look at photos at each
    waypoint, then smoothly moving to the next group.

    Args:
        t: Progress through this segment [0, 1]
        loiter_ratio: Fraction of time spent dwelling (default from CONFIG)

    Returns:
        Eased progress [0, 1]
    """
    if loiter_ratio is None:
        loiter_ratio = CONFIG["loiter_ratio"]
    if t <= 0:
        return 0.0
    if t >= 1:
        return 1.0
    move_ratio = 1.0 - loiter_ratio
    half_loiter = loiter_ratio / 2
    if t < half_loiter:
        return 0.0
    elif t < half_loiter + move_ratio:
        move_t = (t - half_loiter) / move_ratio
        return ease_in_out(move_t)
    else:
        return 1.0


def _clamp_crop(cx, cy, cw, ch, canvas_w, canvas_h):
    """Convert center+dimensions to top-left corner, clamp to canvas, ensure even."""
    x = int(cx - cw / 2)
    y = int(cy - ch / 2)
    cw = int(cw)
    ch = int(ch)
    x = max(0, min(canvas_w - cw, x))
    y = max(0, min(canvas_h - ch, y))
    if cw % 2 != 0:
        cw -= 1
    if ch % 2 != 0:
        ch -= 1
    return (x, y, cw, ch)


def generate_camera_path(canvas_w, canvas_h, segment_frames, num_segments=1):
    """Generate camera paths for one or two background segments.

    The camera simulates a physical dolly move across the panoramic background.
    A crop rectangle (the "viewport") moves across the large canvas, and each
    frame is the contents of that viewport scaled to the output resolution.

    Two-segment mode (two backgrounds, two music tracks):

        Segment 1 (L→R):
        ├─ wide ─┤├── zoom in ──┤├──── meander L→R ────┤├─ zoom out ─┤├ hold ┤
        0s       3s            13s                                    -8s    -2s  end
                                    ╭─╮   ╭─╮   ╭─╮
                 camera moves:      │ │   │ │   │ │     (zig-zag through
                                    ╰─╯   ╰─╯   ╰─╯     waypoints)
                                                                    fade to black

        Segment 2 (R→L, mirrored — "return trip"):
        ├ hold ┤├── zoom in ──┤├──── meander R→L ────┤├─ zoom out ─┤├── hold ──┤
        0s     2s            12s                                   -10s  -4s   end
        fade from black                                            hold at wide + fade to black

    Single-segment mode:
        ├─ wide ─┤├── zoom in ──┤├──── meander ────┤├─ zoom out ─┤├─ pan back ─┤├─ fade ─┤

    The meander phase interpolates between randomly-generated waypoints that
    zig-zag vertically across the canvas. Each waypoint transition uses cosine
    easing with loiter (dwell time at each waypoint, simulating pausing to look
    at photos).

    Args:
        canvas_w, canvas_h: Background image dimensions (same for both)
        segment_frames: List of frame counts, one per segment
        num_segments: 1 or 2

    Returns:
        List of lists: one camera path per segment, each is a list of
        (x, y, crop_w, crop_h) tuples.
    """
    fps = CONFIG["fps"]
    aspect = CONFIG["video_width"] / CONFIG["video_height"]

    wide_h = canvas_h
    wide_w = int(wide_h * aspect)
    if wide_w > canvas_w:
        wide_w = canvas_w
        wide_h = int(wide_w / aspect)

    close_w = int(canvas_w * CONFIG["close_zoom_pct"])
    close_h = int(close_w / aspect)

    pan_margin = close_w / 2 + 100
    pan_x_start = pan_margin
    pan_x_end = canvas_w - pan_margin
    y_center = canvas_h / 2

    secs_per_wp = CONFIG["secs_per_waypoint"]

    all_paths = []

    for seg_idx in range(num_segments):
        total_frames = segment_frames[seg_idx]
        # Reverse direction for segment 2 (R→L meander = return trip)
        reverse = (seg_idx == 1 and num_segments == 2)
        seed = CONFIG["camera_seed"] + seg_idx * 100

        if num_segments == 1:
            # Single-background mode: original 4-phase camera path
            phase1_end = CONFIG["phase1_duration"] * fps
            zoom_in_end = (CONFIG["phase1_duration"] + CONFIG["phase2_duration"]) * fps
            zoom_out_dur = CONFIG["zoom_out_duration"] * fps
            pan_back_dur = CONFIG["pan_back_duration"] * fps
            fade_out_dur = CONFIG["fade_out_duration"] * fps
            end_section = zoom_out_dur + pan_back_dur + fade_out_dur
            meander_end = total_frames - end_section
            if meander_end <= zoom_in_end:
                meander_end = total_frames
        else:
            # Two-background mode: each segment has its own intro/outro
            if seg_idx == 0:
                # Segment 1: wide intro → zoom in → meander → zoom out to wide → hold wide (for fade)
                phase1_end = CONFIG["phase1_duration"] * fps
                zoom_in_end = (CONFIG["phase1_duration"] + CONFIG["phase2_duration"]) * fps
                # End with zoom-out to wide shot + brief hold for the fade-to-black
                zoom_out_dur = CONFIG["zoom_out_duration"] * fps
                hold_dur = 2 * fps  # 2s hold at wide for fade
                meander_end = total_frames - zoom_out_dur - hold_dur
                if meander_end <= zoom_in_end:
                    meander_end = total_frames
            else:
                # Segment 2: wide shot (fading in) → zoom in → meander → zoom out → hold wide → fade out
                phase1_end = 2 * fps  # 2s wide hold during fade-in
                zoom_in_end = phase1_end + CONFIG["phase2_duration"] * fps
                zoom_out_dur = CONFIG["zoom_out_duration"] * fps
                wide_hold = 4 * fps  # 4s hold at wide shot before fade
                fade_to_black = CONFIG["fade_out_duration"] * fps
                hold_dur = wide_hold + fade_to_black
                meander_end = total_frames - zoom_out_dur - hold_dur
                if meander_end <= zoom_in_end:
                    meander_end = total_frames

        # Generate zig-zag waypoints
        rng = random.Random(seed)
        traverse_frames = meander_end - zoom_in_end
        traverse_dur = max(1, traverse_frames) / fps
        num_waypoints = max(4, int(traverse_dur / secs_per_wp))
        seg_label = "Segment 1 (L→R)" if not reverse else "Segment 2 (R→L)"
        print(f"  {seg_label}: {traverse_dur:.0f}s meander, {num_waypoints} waypoints (~{secs_per_wp}s each)")

        waypoints = []
        y_range = (canvas_h - close_h) / 2 * CONFIG["y_range_scale"]
        for i in range(num_waypoints):
            t = i / (num_waypoints - 1)
            if reverse:
                # R→L: start at right edge, end at left
                wx = pan_x_end - (pan_x_end - pan_x_start) * t
            else:
                # L→R: start at left edge, end at right
                wx = pan_x_start + (pan_x_end - pan_x_start) * t
            if i % 2 == 0:
                wy = y_center + rng.uniform(y_range * 0.3, y_range)
            else:
                wy = y_center - rng.uniform(y_range * 0.3, y_range)
            waypoints.append((wx, wy))

        directions = ['down' if wy > y_center else 'up' for _, wy in waypoints]
        print(f"    Waypoints Y: {directions}")

        def interp_waypoint(progress, wps=waypoints, nw=num_waypoints):
            pos = progress * (nw - 1)
            idx = int(pos)
            idx = max(0, min(nw - 2, idx))
            local_t = pos - idx
            e = ease_with_loiter(local_t)
            wx = wps[idx][0] + (wps[idx + 1][0] - wps[idx][0]) * e
            wy = wps[idx][1] + (wps[idx + 1][1] - wps[idx][1]) * e
            return wx, wy

        # First waypoint position (for zoom-in target)
        first_wp_x, first_wp_y = interp_waypoint(0)
        # Last waypoint position (for zoom-out origin)
        last_wp_x, last_wp_y = interp_waypoint(1.0)

        path = []
        for frame in range(total_frames):
            if frame < phase1_end:
                # Wide shot (static hold)
                cw, ch = wide_w, wide_h
                cx, cy = canvas_w / 2, canvas_h / 2

            elif frame < zoom_in_end:
                # Zoom from wide to close, easing toward first waypoint
                t = (frame - phase1_end) / (zoom_in_end - phase1_end)
                e = ease_in_out(t)
                cw = wide_w + (close_w - wide_w) * e
                ch = wide_h + (close_h - wide_h) * e
                cx = (canvas_w / 2) + (first_wp_x - canvas_w / 2) * e
                cy = (canvas_h / 2) + (first_wp_y - canvas_h / 2) * e

            elif frame < meander_end:
                # Zig-zag meander at close zoom
                t = (frame - zoom_in_end) / max(1, meander_end - zoom_in_end)
                cw, ch = close_w, close_h
                cx, cy = interp_waypoint(t)

            elif frame < meander_end + zoom_out_dur:
                # Zoom out from close to wide
                t = (frame - meander_end) / zoom_out_dur
                e = ease_in_out(t)
                cw = close_w + (wide_w - close_w) * e
                ch = close_h + (wide_h - close_h) * e
                cx = last_wp_x + (canvas_w / 2 - last_wp_x) * e
                cy = last_wp_y + (canvas_h / 2 - last_wp_y) * e

            else:
                # Hold at wide shot (for fade in/out)
                cw, ch = wide_w, wide_h
                cx, cy = canvas_w / 2, canvas_h / 2

            path.append(_clamp_crop(cx, cy, cw, ch, canvas_w, canvas_h))

        all_paths.append(path)

    return all_paths


def detect_audio_end(path, container_dur):
    """Find where the last non-silent audio ends using ffmpeg silencedetect.

    Many music files have long trailing silence padded into the container.
    Using container durations would create dead air in the crossfade gap.
    """
    result = subprocess.run(
        ["ffmpeg", "-i", str(path), "-af", "silencedetect=noise=-35dB:d=1",
         "-f", "null", "-"],
        capture_output=True, text=True
    )
    last_silence_start = None
    for line in result.stderr.split("\n"):
        if "silence_start:" in line:
            try:
                last_silence_start = float(line.split("silence_start:")[1].strip().split()[0])
            except (ValueError, IndexError):
                pass
    if last_silence_start and last_silence_start < container_dur - 5:
        return last_silence_start
    return container_dur


def create_video(composited_paths, canvas_w, canvas_h, music, test_mode=False):
    """Render the Ken Burns video with audio crossfade and background transition.

    Supports one or two composited background images. With two backgrounds:
      - Segment 1 plays over track 1 (L→R meander)
      - Segment 2 plays over track 2 (R→L meander, mirrored "return trip")
      - The transition happens at the audio crossfade point with a fade-to-black

    The fade-to-black transition:
      - Last 1.5s of segment 1: fade video to black
      - First 1.5s of segment 2: fade video from black
      This creates a 3s visual transition aligned with the audio crossfade.

    Audio crossfade implementation note:
      We tried ffmpeg's `acrossfade` (produced 126s shorter output due to AAC
      container issues), then `asetpts` (only shifts PTS, doesn't insert silence
      — both tracks played simultaneously). The final approach uses `adelay` to
      insert actual silence before track 2, plus `atrim` to strip trailing
      silence from both source files.

    Performance: 4.3x speedup via BICUBIC resize, 4-thread frame prep with
    8-frame lookahead, and numpy array backing for thread-safe cropping.

    Args:
        composited_paths: List of Paths to composited PNGs (1 or 2)
        canvas_w, canvas_h: Background dimensions (same for all)
        music: Dict with "track_1" and "track_2" paths (optionally "track_3")
        test_mode: If True, render 120s test clip without audio

    Returns:
        Path to the output video file
    """
    print("=== Creating Ken Burns video ===")
    fps = CONFIG["fps"]
    video_w = CONFIG["video_width"]
    video_h = CONFIG["video_height"]
    num_segments = len(composited_paths)

    music1 = music["track_1"]
    music2 = music["track_2"]
    music3 = music.get("track_3")

    # Probe audio durations (content, not container)
    audio1_container = float(subprocess.check_output([
        "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
        "-of", "csv=p=0", str(music1)
    ]).decode().strip())
    audio2_container = float(subprocess.check_output([
        "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
        "-of", "csv=p=0", str(music2)
    ]).decode().strip())

    audio1_dur = detect_audio_end(music1, audio1_container)
    audio2_dur = detect_audio_end(music2, audio2_container)

    crossfade_dur = CONFIG["crossfade_duration"]
    songs_dur = audio1_dur + audio2_dur - crossfade_dur

    title_fade_in_cfg = CONFIG["title_fade_in"]
    title_fade_out_cfg = CONFIG["title_fade_out"]
    title_gap = 1.0

    if music3 and not test_mode:
        audio3_container = float(subprocess.check_output([
            "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
            "-of", "csv=p=0", str(music3)
        ]).decode().strip())
        audio3_dur = detect_audio_end(music3, audio3_container)
        title_total_cfg = audio3_dur
        title_hold_cfg = audio3_dur - title_fade_in_cfg - title_fade_out_cfg
    else:
        audio3_dur = 0
        title_hold_cfg = CONFIG["title_hold"]
        title_total_cfg = title_fade_in_cfg + title_hold_cfg + title_fade_out_cfg

    if test_mode:
        total_dur = CONFIG["test_duration"]
    else:
        if music3 and CONFIG.get("title_lines"):
            total_dur = songs_dur + title_gap + title_total_cfg
        else:
            total_dur = songs_dur
    total_frames = int(total_dur * fps)
    songs_frames = int(songs_dur * fps) if not test_mode else total_frames

    print(f"  Track 1: {audio1_dur:.1f}s content ({audio1_container:.1f}s container)")
    print(f"  Track 2: {audio2_dur:.1f}s content ({audio2_container:.1f}s container)")
    if music3 and not test_mode:
        print(f"  Track 3: {audio3_dur:.1f}s content ({audio3_container:.1f}s container)")
    if test_mode:
        print(f"  TEST MODE: {total_dur:.1f}s ({total_frames} frames)")
    else:
        print(f"  Songs: {songs_dur:.1f}s, Title: {title_gap + title_total_cfg:.1f}s, "
              f"Total: {total_dur:.1f}s ({total_frames} frames)")

    # --- Compute per-segment frame counts ---
    # Segment frames only cover the songs portion (not the title card).
    if num_segments == 2:
        seg1_frames = min(int(audio1_dur * fps), songs_frames)
        seg2_frames = max(0, songs_frames - seg1_frames)
        segment_frames = [seg1_frames, seg2_frames]
        print(f"  Segment 1: {seg1_frames} frames ({seg1_frames / fps:.1f}s)")
        print(f"  Segment 2: {seg2_frames} frames ({seg2_frames / fps:.1f}s)")
    else:
        segment_frames = [songs_frames]

    print(f"  Generating camera paths ({songs_frames} song frames)...")
    paths = generate_camera_path(canvas_w, canvas_h, segment_frames, num_segments)

    # --- Audio ---
    # Build a crossfaded audio mix from 2 or 3 music tracks.
    #
    # Signal flow (3-track mode):
    #
    #   [Track 1] ─► atrim ─► afade(out) ─► apad ──────────────────────┐
    #   [Track 2] ─► atrim ─► afade(in) ─► afade(out) ─► adelay ─► apad ┤─► amix ─► afade(in) ─► AAC
    #   [Track 3] ─► atrim ─► afade(in) ─► afade(out) ─► adelay ─► apad ┘
    #
    #   Timeline:
    #   ├── Track 1 ──────┤
    #                 ├── Track 2 ──────┤        (overlap = crossfade_dur)
    #                              ├─ gap ─┤
    #                                      ├── Track 3 ──────┤
    #
    # Each track is trimmed to its detected audio end (stripping container
    # silence), faded at boundaries, delayed to its start position, then
    # padded to total duration. amix combines them without normalization
    # (normalize=0) to preserve relative levels.
    crossfade_audio_path = OUTPUT_DIR / "crossfade_audio.m4a"
    fade_in_dur = CONFIG["fade_in_duration"]
    fade_out_dur = CONFIG["fade_out_duration"]
    if not test_mode:
        xfade_start = audio1_dur - crossfade_dur
        xfade_start_ms = int(xfade_start * 1000)

        track2_fade_out_dur = fade_out_dur + title_gap

        if music3:
            title_start_time = songs_dur + title_gap
            title_start_ms = int(title_start_time * 1000)

            subprocess.run([
                "ffmpeg", "-y",
                "-i", str(music1),
                "-i", str(music2),
                "-i", str(music3),
                "-filter_complex",
                f"[0:a]atrim=0:{audio1_dur}[a1trim];"
                f"[1:a]atrim=0:{audio2_dur}[a2trim];"
                f"[2:a]atrim=0:{audio3_dur}[a3trim];"
                f"[a1trim]afade=t=out:st={xfade_start}:d={crossfade_dur},"
                f"apad=whole_dur={total_dur}[a1pad];"
                f"[a2trim]afade=t=in:st=0:d={crossfade_dur},"
                f"afade=t=out:st={audio2_dur - track2_fade_out_dur}:d={track2_fade_out_dur}[a2fade];"
                f"[a2fade]adelay={xfade_start_ms}|{xfade_start_ms},"
                f"apad=whole_dur={total_dur}[a2del];"
                f"[a3trim]afade=t=in:st=0:d={title_fade_in_cfg},"
                f"afade=t=out:st={audio3_dur - title_fade_out_cfg}:d={title_fade_out_cfg}[a3fade];"
                f"[a3fade]adelay={title_start_ms}|{title_start_ms},"
                f"apad=whole_dur={total_dur}[a3del];"
                f"[a1pad][a2del][a3del]amix=inputs=3:duration=first:normalize=0,"
                f"afade=t=in:st=0:d={fade_in_dur}[a]",
                "-map", "[a]",
                "-c:a", "aac", "-b:a", CONFIG["audio_bitrate"],
                str(crossfade_audio_path),
            ], check=True, capture_output=True)
        else:
            subprocess.run([
                "ffmpeg", "-y",
                "-i", str(music1),
                "-i", str(music2),
                "-filter_complex",
                f"[0:a]atrim=0:{audio1_dur}[a1trim];"
                f"[1:a]atrim=0:{audio2_dur}[a2trim];"
                f"[a1trim]afade=t=out:st={xfade_start}:d={crossfade_dur},"
                f"apad=whole_dur={total_dur}[a1pad];"
                f"[a2trim]afade=t=in:st=0:d={crossfade_dur}[a2fade];"
                f"[a2fade]adelay={xfade_start_ms}|{xfade_start_ms}[a2del];"
                f"[a1pad][a2del]amix=inputs=2:duration=first:normalize=0,"
                f"afade=t=in:st=0:d={fade_in_dur},"
                f"afade=t=out:st={total_dur - fade_out_dur}:d={fade_out_dur}[a]",
                "-map", "[a]",
                "-c:a", "aac", "-b:a", CONFIG["audio_bitrate"],
                str(crossfade_audio_path),
            ], check=True, capture_output=True)
        print(f"  Audio -> {crossfade_audio_path}")

    if test_mode:
        output_video = OUTPUT_DIR / "test.mp4"
    else:
        output_video = OUTPUT_DIR / "pinboard.mp4"

    # --- Load composited images into numpy arrays ---
    # Thread-safe and faster than PIL.crop() for large images.
    comp_arrays = []
    for p in composited_paths:
        print(f"  Loading {p.name}...")
        img = Image.open(p)
        comp_arrays.append(np.array(img))
        img.close()

    # --- Build ffmpeg command ---
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{video_w}x{video_h}",
        "-r", str(fps),
        "-i", "pipe:0",
    ]
    if not test_mode:
        cmd += ["-i", str(crossfade_audio_path)]
    cmd += [
        "-c:v", "libx264",
        "-preset", CONFIG["preset_test"] if test_mode else CONFIG["preset_production"],
        "-crf", str(CONFIG["crf_test"] if test_mode else CONFIG["crf_production"]),
        "-pix_fmt", "yuv420p",
    ]
    if not test_mode:
        cmd += [
            "-c:a", "aac", "-b:a", CONFIG["audio_bitrate"],
            "-map", "0:v", "-map", "1:a",
        ]
    cmd += ["-t", str(total_dur), str(output_video)]

    print(f"  Piping {total_frames} frames to ffmpeg...")
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    # Stderr drain thread (prevents deadlock — see README)
    stderr_chunks = []
    def drain_stderr():
        while True:
            chunk = proc.stderr.read(4096)
            if not chunk:
                break
            stderr_chunks.append(chunk)
    stderr_thread = threading.Thread(target=drain_stderr, daemon=True)
    stderr_thread.start()

    # --- Build a flat frame map ---
    # For two segments, we need to know which composited array and which
    # camera path position to use for each global frame index.
    # We also compute per-frame fade alpha for:
    #   - Overall video fade in (first 1s) and fade out (last 5s)
    #   - Segment transition fade-to-black (last 1.5s of seg1, first 1.5s of seg2)
    workers = CONFIG["render_workers"]
    buffer_ahead = CONFIG["render_buffer_ahead"]
    transition_fade_dur = 1.5  # seconds of fade to/from black at segment boundary
    transition_fade_frames = int(transition_fade_dur * fps)
    overall_fade_in = int(fade_in_dur * fps)
    overall_fade_out = int(fade_out_dur * fps)

    title_fade_in_dur = title_fade_in_cfg
    title_hold_dur = title_hold_cfg
    title_fade_out_dur = title_fade_out_cfg
    title_total_dur = title_total_cfg
    title_total_frames = int(title_total_dur * fps)
    title_start_frame = total_frames - title_total_frames

    title_img = None
    if not test_mode and CONFIG.get("title_lines"):
        title_img = Image.new("RGB", (video_w, video_h), (0, 0, 0))
        draw = ImageDraw.Draw(title_img)
        font_paths = [
            ROOT / "fonts" / "EBGaramond.ttf",
            Path("/System/Library/Fonts/Supplemental/Georgia.ttf"),
            Path("/usr/share/fonts/truetype/ebgaramond/EBGaramond12-Regular.otf"),
        ]
        font = None
        for fp in font_paths:
            if fp.exists():
                try:
                    font = ImageFont.truetype(str(fp), CONFIG["title_font_size"])
                    break
                except OSError:
                    continue
        if font is None:
            font = ImageFont.load_default()
        lines = CONFIG["title_lines"]
        line_bboxes = [draw.textbbox((0, 0), line, font=font) for line in lines]
        line_heights = [bb[3] - bb[1] for bb in line_bboxes]
        line_widths = [bb[2] - bb[0] for bb in line_bboxes]
        line_spacing = 20
        block_h = sum(line_heights) + line_spacing * (len(lines) - 1)
        y_start = (video_h - block_h) // 2
        for i, line in enumerate(lines):
            lx = (video_w - line_widths[i]) // 2
            ly = y_start + sum(line_heights[:i]) + line_spacing * i
            draw.text((lx, ly), line, fill=(255, 255, 255), font=font)
        title_arr = np.array(title_img, dtype=np.float32)
        print(f"  Title card: {title_total_dur:.0f}s ({title_fade_in_dur:.0f}s in, "
              f"{title_hold_dur:.0f}s hold, {title_fade_out_dur:.0f}s out)")

    # Pre-compute segment boundaries: (start_global_frame, comp_array_idx, path)
    seg_boundaries = []
    offset = 0
    for seg_idx in range(num_segments):
        seg_boundaries.append((offset, seg_idx, paths[seg_idx]))
        offset += segment_frames[seg_idx]

    black_bytes = bytes(video_w * video_h * 3)

    def prepare_frame(global_i):
        """Prepare a single output frame.

        For song frames (global_i < songs_frames): crop from composited image,
        apply fades. For title card frames: black background with text overlay.
        """
        if global_i >= songs_frames and title_img is not None:
            title_local = global_i - title_start_frame
            fade_in_frames = int(title_fade_in_dur * fps)
            hold_frames = int(title_hold_dur * fps)
            if title_local < 0:
                return global_i, black_bytes
            elif title_local < fade_in_frames:
                t_alpha = title_local / fade_in_frames
            elif title_local < fade_in_frames + hold_frames:
                t_alpha = 1.0
            else:
                remaining = title_total_frames - title_local
                fade_out_frames = int(title_fade_out_dur * fps)
                t_alpha = max(0.0, remaining / fade_out_frames)
            if t_alpha <= 0:
                return global_i, black_bytes
            frame = Image.fromarray((title_arr * t_alpha).astype(np.uint8))
            return global_i, frame.tobytes()

        if global_i >= songs_frames:
            return global_i, black_bytes

        seg_idx = 0
        local_i = global_i
        for s in range(num_segments):
            if global_i < seg_boundaries[s][0] + segment_frames[s]:
                seg_idx = s
                local_i = global_i - seg_boundaries[s][0]
                break

        comp_arr = comp_arrays[seg_idx]
        seg_path = seg_boundaries[seg_idx][2]
        seg_len = segment_frames[seg_idx]

        x, y, cw, ch = seg_path[local_i]
        cropped = Image.fromarray(comp_arr[y:y + ch, x:x + cw])
        frame = cropped.resize((video_w, video_h), Image.BICUBIC)

        # Three independent fade multipliers, combined via min():
        #   1. Overall fade-in:  black → visible at video start
        #   2. Overall fade-out: visible → black before title card
        #   3. Segment transition: fade between backgrounds (2-segment mode)
        # Using min() means the darkest fade always wins.
        alpha = 1.0

        if not test_mode:
            if global_i < overall_fade_in:
                alpha = min(alpha, global_i / overall_fade_in)

            fade_out_start = songs_frames - overall_fade_out
            if global_i >= fade_out_start:
                alpha = min(alpha, (songs_frames - global_i) / overall_fade_out)

            if num_segments > 1:
                if seg_idx == 0 and local_i > seg_len - transition_fade_frames:
                    t_alpha = (seg_len - local_i) / transition_fade_frames
                    alpha = min(alpha, t_alpha)
                if seg_idx == 1 and local_i < transition_fade_frames:
                    t_alpha = local_i / transition_fade_frames
                    alpha = min(alpha, t_alpha)

        if alpha < 1.0:
            arr = np.array(frame, dtype=np.float32) * max(0.0, alpha)
            frame = Image.fromarray(arr.astype(np.uint8))

        return global_i, frame.tobytes()

    # --- Threaded frame rendering pipeline ---
    # A bounded producer-consumer that prepares frames ahead of ffmpeg:
    #
    #   [ThreadPool] ─► prepare_frame(i) ─► results{i: bytes}
    #                    prepare_frame(i+1) ─►     │
    #                    prepare_frame(i+2) ─►     │
    #                                              ▼
    #   [Main thread] ─► write results in order ─► ffmpeg stdin
    #
    # Why not just pool.map()? We need:
    #   1. Ordered output (frames must reach ffmpeg in sequence)
    #   2. Bounded memory (only buffer_ahead frames in flight)
    #   3. Overlap between preparation and I/O (pipelining)
    #
    # The pending deque tracks submitted futures; results dict buffers
    # out-of-order completions until write_idx catches up.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = deque()
        write_idx = 0

        for i in range(min(buffer_ahead, total_frames)):
            pending.append(pool.submit(prepare_frame, i))
        next_submit = min(buffer_ahead, total_frames)

        results = {}
        try:
            while write_idx < total_frames:
                if pending:
                    fut = pending.popleft()
                    idx, data = fut.result()
                    results[idx] = data

                if next_submit < total_frames:
                    pending.append(pool.submit(prepare_frame, next_submit))
                    next_submit += 1

                while write_idx in results:
                    proc.stdin.write(results.pop(write_idx))
                    write_idx += 1

                    if write_idx % (fps * 10) == 0:
                        elapsed = write_idx / fps
                        pct = write_idx / total_frames * 100
                        print(f"    {elapsed:.0f}s / {total_dur:.0f}s ({pct:.0f}%)")
        except BrokenPipeError:
            print(f"  WARNING: ffmpeg closed pipe at frame {write_idx}/{total_frames}")

    proc.stdin.close()
    stderr_thread.join(timeout=30)
    proc.wait()
    stderr_data = b"".join(stderr_chunks)
    if proc.returncode != 0:
        print(f"  FFMPEG STDERR:\n{stderr_data.decode(errors='replace')[-3000:]}")
        raise RuntimeError("ffmpeg failed")

    print(f"  Output -> {output_video}")
    return output_video


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def split_photos(portrait, landscape, num_segments, ratios):
    """Split photos across segments proportional to given ratios.

    Maintains the portrait/landscape ratio within each segment.

    Args:
        portrait: list of portrait photo Paths
        landscape: list of landscape photo Paths
        num_segments: number of segments to split into
        ratios: list of floats (e.g. [0.42, 0.58]) summing to ~1.0

    Returns:
        List of (portrait_subset, landscape_subset) tuples, one per segment.
    """
    rng = random.Random(CONFIG["random_seed"])
    p = list(portrait)
    l = list(landscape)
    rng.shuffle(p)
    rng.shuffle(l)

    segments = []
    p_offset = 0
    l_offset = 0
    for i in range(num_segments):
        if i == num_segments - 1:
            seg_p = p[p_offset:]
            seg_l = l[l_offset:]
        else:
            p_count = int(len(p) * ratios[i])
            l_count = int(len(l) * ratios[i])
            seg_p = p[p_offset:p_offset + p_count]
            seg_l = l[l_offset:l_offset + l_count]
            p_offset += p_count
            l_offset += l_count
        segments.append((seg_p, seg_l))

    return segments


def main():
    """Pipeline entry point with CLI flag handling.

    Supports single-background (1 composited image) and two-background mode
    (2 composited images with fade-to-black transition at song changeover).

    Flags:
        --test:           Render 120s test clip (fast preset, no audio)
        --video-only:     Skip layout/composite, reuse existing composited images
        --layout-only:    Generate layout.json only, then exit
        --composite-only: Generate layout + composited images, then exit
    """
    test_mode = "--test" in sys.argv
    video_only = "--video-only" in sys.argv
    layout_only = "--layout-only" in sys.argv
    composite_only = "--composite-only" in sys.argv

    print("Pinboard — Ken Burns Video Pipeline")
    if test_mode:
        print(f"*** TEST MODE ({CONFIG['test_duration']:.0f}s clip) ***")
    print("=" * 40)

    backgrounds, portrait_photos, landscape_photos, music = scan_sources()
    num_segments = len(backgrounds)

    if not backgrounds:
        print("ERROR: No background images found in input/background/")
        sys.exit(1)
    if not video_only and not layout_only:
        if not music.get("track_1") or not music.get("track_2"):
            print(f"ERROR: Need at least 2 music tracks in input/audio/. Found: {list(music.keys())}")
            sys.exit(1)

    bg_images = load_backgrounds(backgrounds)
    canvas_w, canvas_h = bg_images[0].size

    # --- Layout and composite for each segment ---
    # Two-background mode: split photos evenly across backgrounds.
    # Each background gets its own layout and composited image.
    composited_names = [f"composited_{i+1}.png" for i in range(num_segments)]
    if num_segments == 1:
        composited_names = ["composited.png"]

    composited_paths = [OUTPUT_DIR / name for name in composited_names]

    if video_only and all(p.exists() for p in composited_paths):
        print(f"  Reusing {num_segments} existing composited image(s)")
    else:
        # Split photos across segments
        if num_segments == 2:
            ratios = [0.50, 0.50]
        else:
            ratios = [1.0]
        photo_segments = split_photos(portrait_photos, landscape_photos, num_segments, ratios)

        for seg_idx in range(num_segments):
            seg_p, seg_l = photo_segments[seg_idx]
            total_seg = len(seg_p) + len(seg_l)
            seg_label = f"Segment {seg_idx + 1}" if num_segments > 1 else "All"
            print(f"\n--- {seg_label}: {len(seg_p)} portrait + {len(seg_l)} landscape = {total_seg} photos ---")

            layout = generate_layout(canvas_w, canvas_h, len(seg_p), len(seg_l))

            if layout_only:
                continue

            composite_photos(bg_images[seg_idx], layout, seg_p, seg_l,
                             output_path=composited_paths[seg_idx])

        if layout_only:
            print(f"\n=== Done (layout only) ===")
            return

        if composite_only:
            print(f"\n=== Done (composite only) ===")
            for p in composited_paths:
                print(f"  Composited: {p}")
            return

    video = create_video(composited_paths, canvas_w, canvas_h, music, test_mode=test_mode)

    print(f"\n=== Done! ===")
    for p in composited_paths:
        print(f"  Composited: {p}")
    print(f"  Video:      {video}")


if __name__ == "__main__":
    main()

# Retrospective

Observations and discoveries from building the photo wall pipeline.
Built under deadline pressure for a real event.

## Architecture Decisions

### Three-Phase Evolution Plan

- **v1.0:** Single-file pipeline, presentable for open-source
- **v2.0:** Module refactor with typed config (Pydantic), testable modules
- **v3.0:** Declarative scene graph — scene description IS the program

### Layout Algorithm Evolution

Started with a simple grid, moved to Poisson disk sampling, then added
SAT (Separating Axis Theorem) for rotated rectangle overlap detection,
and finally force-based relaxation to push overlapping frames apart.

AABB overlap detection produced false positives on rotated rectangles.
Circle-based detection was too conservative and couldn't handle the
actual footprint of rotated frames. SAT was the only correct approach
for arbitrary rotated rectangles — project onto each edge normal and
check for separating axes.

### Video Rendering Approach

Initially tried ffmpeg's `zoompan` filter. It couldn't handle multi-phase
camera paths (zoom in, meander, zoom out) or per-frame effects like fades.

Switched to frame-by-frame rendering: Python generates each frame as raw
RGB bytes and pipes directly to ffmpeg's stdin. This gives full control
over camera position, zoom, and per-frame compositing at the cost of
doing the work in Python.

Achieved a 4.3x speedup with a threaded frame preparation pipeline
(producer-consumer with bounded buffer). See Performance section.

## Audio Pipeline

### Crossfade Architecture

ffmpeg's `acrossfade` filter couldn't handle three tracks or give precise
timing control over overlap regions.

Replaced with a manual approach: `adelay` + `atrim` + `amix` + `apad`.
Each track is trimmed to its content duration, faded in/out, delayed to
its start position, padded to total length, then mixed together.

Critical detail: `normalize=0` on `amix` preserves relative levels.
The default normalization divides by the number of inputs, which
destroys the mix when tracks overlap during crossfades.

### Trailing Silence Detection

Music files had long trailing silence in their containers — up to 100+
seconds beyond the actual audio content.

Used ffmpeg's `silencedetect` filter to find the actual audio end point.
This was critical for correct video duration calculation and crossfade
timing. Without it, the video would either run long with silence or the
crossfade math would place transitions in the wrong positions.

## Visual Effects

### Weathering Intensity

Initial weathering settings (strong sepia, heavy desaturation) made
photos look muddy and indistinct. The composite lost the character of
individual photos.

Dialed back to subtle vintage warmth:

- 20% desaturation
- 15% sepia tint
- 15% vignette

These values preserve original photo quality while adding a cohesive
aged feel across the entire wall. The restraint matters more than the
effect.

### Background Images

Compared three AI image generators for panoramic map illustrations:

| Tool | Result |
|------|--------|
| DALL-E | Inconsistent style at panoramic aspect ratios |
| Midjourney | Good quality but difficult to control composition |
| Ideogram | Best results for 8K panoramic illustrations with consistent style |

Two backgrounds create a journey-and-return structure: the camera pans
left-to-right across the first, then right-to-left across the second.

## Camera Path

### Four-Phase Motion

Each segment follows four phases:

1. **Wide establishing shot** — full background visible
2. **Zoom in** — camera tightens to show photo detail
3. **Zig-zag meander** — camera visits waypoints across the wall
4. **Zoom out** — pulls back to wide shot, fade to black

Cosine easing with a loiter ratio creates a natural "pause to look"
feel at each waypoint. The camera decelerates into a waypoint, dwells
briefly, then accelerates toward the next.

Two-segment mode mirrors direction (L→R then R→L) for the return trip.

### EXIF Orientation

Photos from phones often carry EXIF rotation metadata — the pixel data
is stored in one orientation while a metadata flag indicates how to
display it.

`ImageOps.exif_transpose()` must be applied before any processing.
Without it, portrait photos render sideways and layout calculations
use wrong dimensions. This is easy to miss because most image viewers
auto-apply the rotation.

## Performance

### Threaded Frame Pipeline

Frame preparation (crop from background + resize + apply fade) is
CPU-bound and independent per frame. The main thread must write frames
to ffmpeg's stdin in strict order.

Solution: custom bounded producer-consumer using `ThreadPoolExecutor`.
Worker threads prepare frames ahead of the write cursor. The main
thread picks up completed frames in sequence and writes to ffmpeg.

- 4 worker threads
- 8-frame lookahead buffer
- Benchmarked as optimal for this workload on the target hardware

Key constraint: frames MUST reach ffmpeg in order. No parallel writes
to stdin. The bounded buffer prevents memory from growing unboundedly
while keeping the ffmpeg pipe fed continuously.

## Font Loading

EB Garamond is bundled as the fallback serif font (SIL OFL licensed).

Font resolution chain:

1. Bundled `fonts/EBGaramond.ttf`
2. macOS system Georgia (`/Library/Fonts/Georgia.ttf`)
3. Pillow default bitmap font

Cross-platform: macOS and Linux paths are checked in sequence. The
bundled font ensures consistent rendering regardless of system fonts.

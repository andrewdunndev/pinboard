# Pinboard Roadmap

Three phases, each a tagged release.

## Phase 1: Single-File Pipeline — `v1.0` ✓ COMPLETE

A working single-file pipeline (`pipeline.py`) that generates 4K Ken Burns-style
photo wall videos. All parameters live in a `CONFIG` dict at the top of the file.
No architecture changes needed to ship.

## Phase 2: Module Refactor — `v2.0`

Decompose `pipeline.py` into testable modules with typed configuration.
Same algorithms, better structure.

```
src/
  config.py          Pydantic Settings model, loads from config.toml
  assets.py          Source scanning -> AssetManifest dataclass
  layout.py          Poisson disk + SAT + force relaxation
  effects.py         Weathering filter chain
  composite.py       Layout + effects + background -> composited image
  camera.py          Composable timeline of CameraSegment objects
  audio.py           Silence detection, ffmpeg filter graph generation
  render.py          Threaded frame pipeline -> ffmpeg
  title.py           Title card frame generator
  pipeline.py        Orchestrator and CLI
tests/
  test_layout.py     SAT correctness, relaxation convergence
  test_camera.py     Waypoint interpolation, easing functions
  test_effects.py    Pixel-level weathering checks
  conftest.py        Fixtures with synthetic test images
config.toml          Default configuration
```

Key changes:

- Pydantic config with nested groups replaces flat CONFIG dict
- Intermediate artifacts as first-class dataclasses (AssetManifest, Layout, CameraPath)
- Camera path as composable timeline of segment objects
- Effects as composable filter chain
- Unit-testable without real assets
- Add argparse (replace sys.argv matching)
- Add ruff linting, type hints
- Verify output matches v1 frame-for-frame

## Phase 3: Declarative Scene Graph — `v3.0`

Scene description IS the program. Fork the project, edit `scene.toml`, produce
a completely different video without touching Python.

```python
scene = Scene(
    segments=[
        PhotoWall(
            background=0,
            layout=PoissonLayout(seed=42, min_gap=40),
            camera=Camera([
                Hold(wide(), duration=3),
                ZoomTo(close(0.40), duration=10),
                Meander(waypoints=auto, secs_per_wp=32),
                ZoomTo(wide(), duration=6),
            ]),
            audio=Track("music-1.mp4", fade_out=3),
        ),
        Transition(fade_to_black=1.5),
        TitleCard(lines=[...], audio=Track("music-3.mp4")),
    ],
)
render(scene)
```

```
src/
  scene.py          Scene, Segment ABC, PhotoWall, TitleCard, Transition
  camera.py         Hold, ZoomTo, Meander, PanTo operations
  layout.py         LayoutStrategy protocol, PoissonLayout, GridLayout
  effects.py        Effect protocol, composable chain
  frame_styles.py   Polaroid, Borderless, TapeFrame
  audio.py          Track probing, ffmpeg filter graph generation
  planner.py        Scene -> RenderPlan
  renderer.py       RenderPlan -> video frames -> ffmpeg
  assets.py         Photo scanning, EXIF, classification
  cli.py            python -m pinboard render scene.toml
scene.toml          Scene as pure data
```

Key changes:

- Segments are polymorphic (PhotoWall, TitleCard, Transition)
- Camera operations are first-class composable objects
- Layout strategies pluggable via protocol
- Two-pass execution: Plan then Render
- Scene loadable from TOML

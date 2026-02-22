# Background Images

The background is one or two panoramic images, ideally 8K+ resolution
(e.g., 12288x4608 or similar wide aspect ratio). These are the canvas that
photos are composited onto and the camera pans across.

## Requirements

- PNG format
- Wide aspect ratio (~2.5:1 or wider recommended)
- High resolution — the camera zooms in to ~40% crop, so detail matters
- A calmer horizontal band in the center helps photos stand out

## AI Generation Tips

If generating backgrounds with AI image tools:

| Tool | Strengths | Weaknesses |
|------|-----------|------------|
| **Ideogram** | Wide aspect ratios natively, respects text constraints | Slightly less artistic than Midjourney |
| **Midjourney** | Best illustration quality | Cannot suppress unwanted text generation |
| **DALL-E** | Easy API, conversational | Max 1024x1024 — stitching produces visible seams |

Ideogram was the most reliable for producing wide, cohesive panoramic
illustrations without unwanted text artifacts.

## Layout Interaction

The pipeline has configurable keep-out zones (`keepout_zones` in CONFIG) to
protect specific areas of the background from photo placement. One photo is
intentionally placed to overlap a keep-out zone for a hand-placed feel
(`badge_overlap_zone` in CONFIG).

Adjust these values when using a different background image.

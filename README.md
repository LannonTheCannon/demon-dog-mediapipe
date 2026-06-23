# demon-dog 🐕‍🦺

Hand-gesture summoning, inspired by *Chainsaw Man*: frame a patch of space with
your hands and a demon dog appears — registered to your fingers, not just pasted
near them. Built with **OpenCV** + **MediaPipe** on a plain webcam.

> The trick that makes the anime shot land is *registration* — the dog's ears
> resolve onto the fingertips. That's the north star here.

## Status

**v0 — camera loop.** Opens the webcam and shows the live feed. Hand tracking,
gesture detection, and the summon get layered on top from here.

## Quick start

```bash
# one-time setup (Python 3.11 — MediaPipe doesn't ship 3.13 wheels yet)
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# generate the placeholder demon asset
python scripts/make_placeholder.py

# run it
python -m src.main            # webcam window — press q / esc to quit
python -m src.main --selftest # headless: grab a few frames and report
```

First run on macOS will prompt for **camera permission** for your terminal.

## Roadmap — the realism ladder

The architecture keeps placement (`compositor`) separate from sensing
(`gesture`), so we can climb this without rewrites:

| Tier | Technique | Payoff |
|---|---|---|
| 0 ✅ | camera loop | feed on screen |
| 1 | oriented placement (affine from hand vector) | demon tilts & scales with your hands |
| 2 | angle-aware sprite atlas | different angles of the demon |
| 3 | anchor rig (ears → fingertips) | **the Chainsaw Man effect** |
| 4 | skeletal 2D (Live2D / Spine) | ears flex, dog breathes |
| 5 | mesh warp to hand outline | silhouette conforms to your fingers |
| 6 | 3D model + pose estimation | occlusion, shadow, photoreal |
| 7 | generative (depth + ControlNet) | indistinguishable from real |

## Layout

```
src/
  config.py        # every tunable knob
  main.py          # the capture → display loop
scripts/
  make_placeholder.py   # generates assets/demon.png
assets/
  demon.png        # placeholder demon (generated)
```

## Credits

Inspired by *Chainsaw Man* (Tatsuki Fujimoto). Code under MIT.

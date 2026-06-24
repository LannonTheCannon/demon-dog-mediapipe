"""Generate the fox-demon asset (assets/demon.png) — a Chainsaw Man-style Fox Devil.

The Fox Devil is an oversized pale/arctic fox head covered in *multiple* glowing
eyes, each made of concentric rings. This draws that look procedurally so we have
a striking placeholder; a hand-made/3D asset can replace it later (enterprise phase).

Run:
    python scripts/make_placeholder.py
"""

from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "assets" / "demon.png"

SIZE = 768  # square RGBA canvas

FUR = (234, 237, 245, 255)        # pale arctic white
INNER_EAR = (120, 86, 104, 255)
NOSE = (22, 10, 16, 255)
MOUTH = (28, 0, 8, 255)
TOOTH = (245, 246, 250, 255)


def _eye(d: ImageDraw.ImageDraw, cx: float, cy: float, r: float) -> None:
    """A demonic concentric-ring eye with a slit pupil."""
    rings = [
        (r, (70, 0, 8, 255)),              # dark maroon rim
        (r * 0.80, (165, 14, 14, 255)),    # deep red
        (r * 0.58, (230, 60, 22, 255)),    # red-orange
        (r * 0.38, (255, 150, 45, 255)),   # orange
        (r * 0.20, (255, 232, 165, 255)),  # hot center
    ]
    for rad, col in rings:
        d.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill=col)
    pw = max(2.0, r * 0.09)
    d.ellipse([cx - pw, cy - r * 0.22, cx + pw, cy + r * 0.22], fill=(8, 0, 0, 255))


def build() -> Image.Image:
    img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    cx = SIZE // 2

    # --- ears (large, pointed; tips near the top so they sit at the frame corners) ---
    d.polygon([(cx - 238, 360), (cx - 205, int(SIZE * 0.10)), (cx - 60, 330)], fill=FUR)
    d.polygon([(cx + 238, 360), (cx + 205, int(SIZE * 0.10)), (cx + 60, 330)], fill=FUR)
    d.polygon([(cx - 196, 330), (cx - 186, int(SIZE * 0.17)), (cx - 110, 320)], fill=INNER_EAR)
    d.polygon([(cx + 196, 330), (cx + 186, int(SIZE * 0.17)), (cx + 110, 320)], fill=INNER_EAR)

    # --- head + muzzle ---
    d.ellipse([cx - 232, 248, cx + 232, 648], fill=FUR)
    d.ellipse([cx - 120, 510, cx + 120, int(SIZE * 0.93)], fill=FUR)

    # --- snout: nose + toothy mouth (snout tip ~ y 0.85) ---
    ny = int(SIZE * 0.85)
    d.polygon([(cx - 36, ny - 28), (cx + 36, ny - 28), (cx, ny + 22)], fill=NOSE)
    d.polygon([(cx - 96, ny + 14), (cx + 96, ny + 14), (cx + 60, ny + 74), (cx - 60, ny + 74)], fill=MOUTH)
    for i in range(-2, 3):
        tx = cx + i * 32
        d.polygon([(tx - 13, ny + 16), (tx + 13, ny + 16), (tx, ny + 46)], fill=TOOTH)

    # --- the signature: MANY concentric eyes across the face ---
    eyes = [
        (cx - 96, 432, 62), (cx + 96, 432, 62),    # primary pair
        (cx, 332, 36),                             # forehead
        (cx - 152, 352, 30), (cx + 152, 352, 30),  # upper cheeks
        (cx - 156, 512, 32), (cx + 156, 512, 32),  # lower cheeks
        (cx - 58, 548, 22), (cx + 58, 548, 22),    # near snout
        (cx, 470, 24),                             # extra center
    ]
    for ex, ey, er in eyes:
        _eye(d, ex, ey, er)

    return img


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    img = build()
    img.save(OUT)
    print(f"wrote {OUT}  ({img.size[0]}x{img.size[1]}, RGBA)")


if __name__ == "__main__":
    main()

"""Generate a placeholder transparent demon-dog PNG so we're never blocked on art.

This draws a simple stylized dog-head silhouette with pointy ears and glowing
eyes onto a transparent canvas. It's intentionally crude — the point is to have
*something* with an alpha channel to composite while the real asset gets made.

Run:
    python scripts/make_placeholder.py
"""

from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "assets" / "demon.png"

SIZE = 512  # square canvas, RGBA


def build() -> Image.Image:
    img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    body = (18, 18, 22, 255)      # near-black demon body
    glow = (220, 30, 30, 255)     # red eye glow

    cx = SIZE // 2

    # --- ears: two tall triangles near the top, splayed like the framed fingers ---
    d.polygon([(cx - 150, 250), (cx - 120, 60), (cx - 40, 230)], fill=body)   # left ear
    d.polygon([(cx + 150, 250), (cx + 120, 60), (cx + 40, 230)], fill=body)   # right ear

    # --- head: rounded blob ---
    d.ellipse([cx - 150, 180, cx + 150, 430], fill=body)

    # --- snout ---
    d.ellipse([cx - 70, 360, cx + 70, 470], fill=body)
    d.ellipse([cx - 26, 430, cx + 26, 478], fill=(8, 8, 10, 255))  # nose

    # --- eyes: glowing ---
    for ex in (cx - 70, cx + 70):
        d.ellipse([ex - 26, 280, ex + 26, 332], fill=glow)
        d.ellipse([ex - 10, 296, ex + 10, 320], fill=(255, 200, 120, 255))  # hot center

    return img


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    img = build()
    img.save(OUT)
    print(f"wrote {OUT}  ({img.size[0]}x{img.size[1]}, RGBA)")


if __name__ == "__main__":
    main()

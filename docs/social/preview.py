#!/usr/bin/env python3

import argparse
import io
from pathlib import Path

import cairosvg
from PIL import Image

SVG_SOURCE = Path(__file__).with_name("preview.html")
svg_text = SVG_SOURCE.read_text(encoding="utf-8")
# Split the HTML file into SVG snippets using a delimiter.
SVG_CONTENT = [item.strip() for item in svg_text.split("<!-- SVG -->") if item.strip()]

icon_size = 200
color = "#000000"

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("-o", default="cover.png")
args = parser.parse_args()

cover = Image.open(Path(__file__).with_name("bg.png")).convert("RGBA")
width, height = cover.size
center_y = height // 2
left = icon_size / 2
right = width - (icon_size / 2)
step = (right - left) / 2
centers = [round(left + (step * i)) for i in range(3)]

for i, svg in enumerate(SVG_CONTENT):
    styled_svg = svg.replace("<svg ", f'<svg style="color:{color};" ', 1)
    png_bytes = cairosvg.svg2png(
        bytestring=styled_svg.encode("utf-8"),
        output_width=icon_size,
        output_height=icon_size,
    )
    icon = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    x = centers[i] - (icon.width // 2)
    y = center_y - (icon.height // 2)
    cover.alpha_composite(icon, (x, y))

cover.save(args.o)
print(f"Wrote {args.o}")

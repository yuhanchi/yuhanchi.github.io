#!/usr/bin/env python3
"""Tile screenshots into one contact sheet: sheet.py OUT.png COLS img1 img2 ..."""
import sys
from PIL import Image
out, cols, files = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
ims = [Image.open(f).convert("RGB") for f in files]
w = 640; ims = [im.resize((w, round(im.height * w / im.width))) for im in ims]
h = max(im.height for im in ims); rows = -(-len(ims) // cols)
sheet = Image.new("RGB", (cols * (w + 10), rows * (h + 10)), "white")
for i, im in enumerate(ims):
    sheet.paste(im, ((i % cols) * (w + 10), (i // cols) * (h + 10)))
sheet.save(out)

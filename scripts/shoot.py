#!/usr/bin/env python3
"""Screenshot pages of a locally served build: shoot.py OUTDIR path1 path2 ...  (env W=width, FULL=1)"""
import os, sys
from playwright.sync_api import sync_playwright
out, paths = sys.argv[1], sys.argv[2:]
os.makedirs(out, exist_ok=True)
w = int(os.environ.get("W", 1280)); full = os.environ.get("FULL", "1") == "1"
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": w, "height": 900}, device_scale_factor=1)
    for path in paths:
        pg.goto("http://localhost:4567" + path, wait_until="networkidle", timeout=45000)
        pg.wait_for_timeout(600); pg.evaluate("document.fonts.ready.then(() => 1)"); pg.evaluate("(window.MathJax && MathJax.startup && MathJax.startup.promise) ? MathJax.startup.promise.then(() => 1) : 1"); pg.wait_for_timeout(300)
        name = (path.strip("/").replace("/", "_") or "root") + f"_{w}.png"
        pg.screenshot(path=os.path.join(out, name), full_page=full)
        print(name)
    b.close()

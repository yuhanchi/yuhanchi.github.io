#!/usr/bin/env python3
"""Generate the per-design stub pages.

For every design in _data/styles.yml this writes
  <id>/index.md          home
  <id>/about.md          about (the only page with the photo)
  <id>/writing.md        post archive
  <id>/writing/<slug>.md one page per post in _posts/
Stubs contain front matter only; all content comes from _includes/ and _posts/.
Run again after adding, renaming or deleting posts or designs.
"""
import pathlib, re, shutil, yaml

root = pathlib.Path(__file__).resolve().parent.parent
styles = yaml.safe_load((root / "_data/styles.yml").read_text(encoding="utf-8"))
slugs = []
for f in sorted((root / "_posts").glob("*.md")):
    m = re.match(r"\d{4}-\d{2}-\d{2}-(.+)\.md$", f.name)
    if m:
        slugs.append(m.group(1))

def stub(path, **fm):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["---", "layout: default"] + [f"{k}: {v}" for k, v in fm.items()] + ["---", ""]
    path.write_text("\n".join(lines), encoding="utf-8")

for st in styles:
    sid = st["id"]
    d = root / sid
    if (d / "writing").exists():
        shutil.rmtree(d / "writing")
    stub(d / "index.md", style=sid, kind="home", permalink=f"/{sid}/")
    stub(d / "about.md", style=sid, kind="about", permalink=f"/{sid}/about/")
    stub(d / "writing.md", style=sid, kind="writing", permalink=f"/{sid}/writing/")
    for slug in slugs:
        stub(d / "writing" / f"{slug}.md", style=sid, kind="post",
             post_slug=slug, permalink=f"/{sid}/writing/{slug}/")

print(f"{len(styles)} designs × {len(slugs)} posts")

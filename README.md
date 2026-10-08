# yuhanchi.github.io

Personal academic homepage of Yuhan Chi (池裕涵), School of Mathematical Sciences, Fudan University.
Plain Jekyll on GitHub Pages (no plugins), design: **Scholar**.

## Pages

| URL | Source |
|---|---|
| `/` About (entry page, the only page with the photo) | `_includes/about.md` + `_data/profile.yml` |
| `/writing/` and `/writing/<slug>/` | `_posts/` |
| `/cv/` | `_includes/cv.md`, PDF in `files/Yuhan_Chi_CV.pdf` |

## Editing

- **Bio** — `_includes/about.md`
- **Publications, experience, projects, talks, service, honors, links** — `_data/profile.yml`
- **CV** — `_includes/cv.md` (and replace `files/Yuhan_Chi_CV.pdf`)
- **Photo** — `assets/img/yuhan.jpg` (rectangular, 3:4)
- **Look** — `assets/css/scholar.css` (palette, fonts, photo size via `--photo-w`); shared spacing and components in `assets/css/base.css`

## Writing a post

Add `_posts/YYYY-MM-DD-slug.md`, commit and push; it appears at `/writing/slug/`.

```markdown
---
title: "My title"
description: "One-sentence summary shown in lists."
category: "tech"        # or "musings"
tags: ["notes"]
math: true              # loads MathJax: $inline$, $$display$$
---
```

Code blocks are highlighted; ` ```mermaid ` blocks render as diagrams.

## Other designs

The other 19 candidate designs were moved, without personal data, into a separate reusable kit
(`academic-homepage-templates`), which has its own README.

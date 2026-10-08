# yuhanchi.github.io

Hi, I am Yuhan Chi. This is the source of my personal site, built with plain Jekyll on GitHub Pages (no plugins).

## Design previews

The site currently contains **10 candidate designs** built from the same content, so they can be compared side by side:

| # | Design | URL |
|---|--------|-----|
| 1 | Paper — single serif column on warm ivory | [/paper/](https://yuhanchi.github.io/paper/) |
| 2 | Sage — calm sidebar layout, pale sage | [/sage/](https://yuhanchi.github.io/sage/) |
| 3 | Mist — cool grey-blue, hairline lists | [/mist/](https://yuhanchi.github.io/mist/) |
| 4 | Linen — sand & terracotta, Garamond, split home | [/linen/](https://yuhanchi.github.io/linen/) |
| 5 | Forest — soft dark green theme | [/forest/](https://yuhanchi.github.io/forest/) |
| 6 | Ledger — margin labels, monospace details | [/ledger/](https://yuhanchi.github.io/ledger/) |
| 7 | Washi — Japanese-inspired, Mincho type | [/washi/](https://yuhanchi.github.io/washi/) |
| 8 | Clay — warm rounded cards | [/clay/](https://yuhanchi.github.io/clay/) |
| 9 | Dusk — lavender-grey, big serif masthead | [/dusk/](https://yuhanchi.github.io/dusk/) |
| 10 | Stone — Swiss grid, big type, olive accent | [/stone/](https://yuhanchi.github.io/stone/) |

The root page lists them all. Every page has a small switcher at the bottom (‹ ›) that jumps to the *same page* in the previous/next design.

Each design has four page types: home, `about/` (the only page with my photo), `writing/` (archive) and `writing/<post>/`.

## Where things live

```
_config.yml            site settings (default_style = design used for /posts/<slug>/)
_data/profile.yml      name, links, projects, publications, talks
_data/styles.yml       the list of designs
_includes/about.md     About page text (Markdown)
_posts/                blog posts (Markdown; `math: true` enables KaTeX)
_includes/k/           page templates shared by all designs (home, about, writing, post)
_includes/styles/<id>.html   page shell of each design
assets/css/base.css    shared typography / reading styles
assets/css/<id>.css    each design's colours and layout
assets/img/yuhan.jpg   profile photo (used only on About)
scripts/make_previews.py   regenerates <id>/… stub pages
```

The four posts in `_posts/` are sample content for judging readability; replace them freely.

## Writing a post

1. Add `_posts/YYYY-MM-DD-some-slug.md`:
   ```markdown
   ---
   title: My title
   description: One-sentence summary shown in lists.
   tags: [notes]
   math: true        # optional
   ---
   Text…
   ```
2. Run `python3 scripts/make_previews.py` (creates the per-design copies of the post page), commit, push.

## Picking the final design

Once one design is chosen: set `default_style` in `_config.yml`, point `/` at that design's home, delete the other designs' folders/CSS/shells and the `switcher` include.

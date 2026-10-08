# yuhanchi.github.io

Personal academic homepage of Yuhan Chi (池裕涵), School of Mathematical Sciences, Fudan University.
Plain Jekyll on GitHub Pages (no plugins).

## Design candidates

The root site (`/`) uses the design set in `_config.yml → default_style` (currently **scholar**).
All candidates are listed at **[/styles/](https://yuhanchi.github.io/styles/)**; each page has a small switcher (‹ ›) that jumps to the same page in the previous/next design.

| # | Design | URL | Description |
|---|--------|-----|-------------|
| 1 | Scholar | [/scholar/](https://yuhanchi.github.io/scholar/) | Classic academic homepage — ivory paper, navy ink, EB Garamond with small-caps headings. |
| 2 | Article | [/article/](https://yuhanchi.github.io/article/) | Typeset like a LaTeX paper — Latin Modern, centered title block, an abstract, numbered sections. |
| 3 | Tufte | [/tufte/](https://yuhanchi.github.io/tufte/) | Edward Tufte's handout style — ET Book on cream, a wide margin for the photo, dates and notes. |
| 4 | Classic | [/classic/](https://yuhanchi.github.io/classic/) | The familiar researcher page — name and bio beside a photo, a tidy publication list, Lato and calm blue links. |
| 5 | Faculty | [/faculty/](https://yuhanchi.github.io/faculty/) | A university profile page — a deep teal header band, a contact card beside the bio, Noto Sans and Serif. |
| 6 | Oxford | [/oxford/](https://yuhanchi.github.io/oxford/) | Formal and collegiate — Oxford blue on parchment, Caslon type, double rules and small capitals. |
| 7 | Claret | [/claret/](https://yuhanchi.github.io/claret/) | Burgundy and warm white; a sticky profile column with contact details and Crimson Pro text. |
| 8 | Ivy | [/ivy/](https://yuhanchi.github.io/ivy/) | Deep ivy green, Libre Baskerville text and Cormorant small-caps headings; quietly traditional. |
| 9 | Monograph | [/monograph/](https://yuhanchi.github.io/monograph/) | Book-like — running head, centered small-caps section titles, ornaments and a drop cap, set in Spectral. |
| 10 | Distill | [/distill/](https://yuhanchi.github.io/distill/) | A research-journal look in the spirit of Distill — sans headings, serif text, a byline grid on posts. |
| 11 | Paper | [/paper/](https://yuhanchi.github.io/paper/) | A single quiet column of serif text on warm ivory, like a well-set essay. |
| 12 | Sage | [/sage/](https://yuhanchi.github.io/sage/) | A calm sidebar layout in pale sage and soft green, sans-serif with serif reading pages. |
| 13 | Mist | [/mist/](https://yuhanchi.github.io/mist/) | Cool grey-blue, centered header, IBM Plex; clean lists separated by hairlines. |
| 14 | Linen | [/linen/](https://yuhanchi.github.io/linen/) | Sand and terracotta with an elegant Garamond display face. |
| 15 | Forest | [/forest/](https://yuhanchi.github.io/forest/) | A soft dark theme — deep green-charcoal, cream text and moss accents. |
| 16 | Ledger | [/ledger/](https://yuhanchi.github.io/ledger/) | A structured index with section labels in the margin, monospace details and an ochre accent. |
| 17 | Washi | [/washi/](https://yuhanchi.github.io/washi/) | Japanese-inspired restraint — rice-paper tones, indigo ink, Mincho type. |
| 18 | Clay | [/clay/](https://yuhanchi.github.io/clay/) | Warm and friendly — soft cards with a muted clay accent, Figtree and Lora. |
| 19 | Dusk | [/dusk/](https://yuhanchi.github.io/dusk/) | Muted lavender-grey with a large serif name; modern and editorial. |
| 20 | Stone | [/stone/](https://yuhanchi.github.io/stone/) | Swiss-style grid on warm grey — big typographic name, strict alignment, one olive accent. |

Every design has the same pages: About (entry page, the only page with the photo), `writing/`, `writing/<post>/`, and `cv/`.

## Where things live

```
_config.yml                 site settings; default_style = design used at the root
_data/profile.yml           name, links, publications, experience, projects, talks, service, honors
_data/styles.yml            the list of designs
_includes/about.md          About-page bio (Markdown)
_includes/cv.md             CV page (Markdown); PDF in files/Yuhan_Chi_CV.pdf
_posts/                     posts (Markdown; `math: true` loads MathJax, `$…$` inline)
_includes/k/                page templates shared by all designs (about, writing, post, cv)
_includes/styles/<id>.html  page shell of each design
assets/css/base.css         shared spacing, components and reading typography
assets/css/<id>.css         each design's palette, type and layout
scripts/make_previews.py    regenerates the /<id>/… stub pages
```

## Writing a post

1. Add `_posts/YYYY-MM-DD-slug.md`:
   ```markdown
   ---
   title: "My title"
   description: "One-sentence summary shown in lists."
   category: "tech"        # or "musings"
   tags: ["notes"]
   math: true              # optional
   ---
   ```
2. Run `python3 scripts/make_previews.py`, commit, push.

## Choosing the final design

Set `default_style` in `_config.yml`. To retire the previews: delete the other `<id>/` folders, their CSS and shells, `styles/`, and the `{% include switcher.html %}` line in each remaining shell.

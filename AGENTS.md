# Project guidance

## Scope and sources

This repository is Yuhan Chi's personal academic website and English-language
blog. Keep this file in English. Follow the user's current request when it
supersedes these defaults; discuss edits in the user's preferred language.

These guidelines consolidate explicit writing preferences from conversation
history, the project README, repository configuration, and Git history. Treat
historical implementation choices as project context, not permanent restrictions
on what the author may write. Do not infer new personal preferences from a
single old commit.

## Author-first English editing

The author writes the English draft first. The assistant corrects and lightly
polishes it. The primary goal is to help the author improve their English while
publishing work that still sounds like them.

- Preserve the author's voice, narrative order, examples, personal observations,
  and conclusions. Prefer small, local edits over rewriting whole paragraphs.
- Correct spelling, grammar, tense, collocations, and unclear wording. Keep
  expressions that already work; do not replace simple language merely to sound
  more sophisticated, academic, or fluent.
- Preserve candid admissions, uncertainty, humor, rhetorical questions, and
  first-person reflection. Do not turn tentative ideas into confident claims.
- Chinese notes embedded in an English draft usually explain the intended
  meaning. Integrate that meaning naturally into the corresponding English
  passage, rather than leaving an editorial aside or adding a parallel Chinese
  translation, unless requested.
- Improve transitions and the logic of newly added passages without inventing
  experiences, facts, evidence, or arguments. Clarify ambiguities that would
  materially change the meaning; resolve routine language choices directly.
- When asked about a word, explain its meaning and natural collocations in the
  actual sentence. Respect the author's preferred wording when it works. For
  example, "accumulation" can express accumulated knowledge, but should not be
  rejected simply because a different phrase sounds more polished.
- After substantive language edits, briefly explain a few useful changes using
  the original expression, the revision, and the reason. Focus on reusable
  language lessons rather than a long correction inventory. Skip or shorten
  this explanation when the user requests direct edits or a concise result.
- Distinguish advice from permission to edit: if the user says "just tell me"
  or asks how to edit in a terminal, answer that request. If asked to update the
  file, make the edits directly.
- Keep metadata summaries faithful to the body; do not exaggerate novelty or
  significance. Do not add a generic concluding paragraph unless requested or
  needed to complete the author's intended argument.

## Editorial judgment and attribution

Favor complete posts with a clear result, useful explanation, or considered
personal insight. An earlier cleanup removed unfinished drafts and retellings,
but later posts explicitly include personal reflections: that cleanup is not a
ban on diary-like writing or musings. Do not delete, unpublish, or restore posts
based solely on an inferred quality threshold.

Keep original analysis distinct from results attributed to other researchers.
When adding or checking substantive claims, use appropriate original sources.
Do not silently expand a language-editing task into a new technical argument.
Retain qualifications and limitations, including those that make a story less
impressive but more accurate.

## Deciding whether a post needs citation instructions

Ask: could a reader reasonably cite a specific result, explanation, method, or
viewpoint from this post in their own writing or research?

- Strong candidates: original experiments with quantitative results, robustness
  checks, and stated limits.
- Good candidates: a coherent analytical framework, distinctive explanation,
  or synthesis across fields. Cite the underlying sources for existing results.
- A short viewpoint essay can qualify when it has a clear central thesis; an
  experiment or long article is not required. Do not imply unverified priority
  or originality.
- Reusable technical investigations can qualify when they document the actual
  environment, failed attempts, and a working solution. Include a version or
  update date where needed to identify the relevant experience.
- Basic study notes do not automatically need a prominent citation section.
  Personal reflections, everyday experiences, and visit notes usually need only
  an ordinary link when someone refers to them.
- Assess unpublished drafts once ready for publication and once sources and
  personal contributions are clear. Do not add citation blocks site-wide.

The posts currently selected for citation blocks are:

- `_posts/2026-05-04-code-not-text.md` — When CoT Surface Features Transfer.
- `_posts/2026-08-14-choice-and-alignment.md` — Choice and Alignment.
- `_posts/2026-09-30-proofs-as-observations.md` — Proofs as Observations.
- `_posts/2026-11-06-time-series-as-tokens.md` — Time Series as Tokens: Notes
  on Chronos-2 (requested by the author).

## Citation presentation

Use a discreet end-of-post `<details>` block with the summary `Cite this post`.
Provide both a readable citation and a copyable BibTeX entry. The established
introduction is:

> If you find this post useful in your work, you can cite it as:

Use `Yuhan Chi` as the author (`Chi, Yuhan` in BibTeX), the actual post title,
publication date, and canonical URL. Use `@misc` with `howpublished = {Blog post}`;
protect title capitalization such as CoT and AI. Do not invent a journal, DOI,
or publication status. Give each entry a distinct citation key.

If a post summarizes a report, paper, or code project with existing citation
information, distinguish citing the blog explanation from citing that underlying
work. A "Cite this post" block does not replace the post's own references.

The adopted readable-citation-plus-BibTeX structure was informed by Lilian
Weng's blog, for example:
https://lilianweng.github.io/posts/2025-05-01-thinking/#citation

## Repository map and design

This is plain Jekyll on GitHub Pages, without custom plugins. The chosen design
is **Scholar**: a restrained white academic page with navy text and EB Garamond.
Other candidate designs were moved out of this repository. Preserve this design
unless a redesign is requested; reuse existing components and styles.

- `_includes/about.md`: biography and About content; About is the entry page.
- `_data/profile.yml`: profile information, publications, experience, projects,
  talks, service, honors, and links.
- `_includes/cv.md` and `files/Yuhan_Chi_CV.pdf`: CV page and downloadable CV.
- `assets/img/yuhan.jpg`: rectangular 3:4 profile photo, shown on About.
- `_posts/YYYY-MM-DD-slug.md`: blog content (canonical English posts).
- `_zh/<slug>.md`: optional Chinese translation of a post, sharing the exact same `slug` as `_posts/YYYY-MM-DD-<slug>.md` (set `slug: "<slug>"` and `date: "YYYY-MM-DD"` in front matter). Configured as a Jekyll collection in `_config.yml` outputting to `/writing/<slug>/zh/`, so it does not duplicate entries on the main Writing index or RSS feed.
- `_includes/k/post.html`: post header (including the per-article `EN / 中文` switcher in `.post-topbar` when a matching `_zh/<slug>.md` exists), byline, body, and navigation.
- `_includes/k/writing.html` and `_includes/parts/post-list.html`: writing index.
- `assets/css/scholar.css`: design-specific appearance.
- `assets/css/base.css`: shared spacing and components, including details boxes.
- `_config.yml`: URLs, Markdown settings, defaults, and exclusions.

## Post format and visibility

Use the existing front matter fields: `title`, `date`, `description`, `category`
(`tech` or `musings`), and `tags`. Set `math: true` when MathJax is needed.
The default layout and post kind are supplied by `_config.yml`.

Canonical URLs follow `https://yuhanchi.github.io/writing/<slug>/`. Preserve
existing slugs when changing titles so links and citations remain valid. Use
Jekyll `post_url` tags for links between posts; use `relative_url` or the existing
base-prefix convention for other internal links and assets.

Kramdown uses GFM input. MathJax supports inline and display math, and fenced
`mermaid` blocks render diagrams. Reuse existing formatting unless a change
improves readability or fixes rendering.

- To hide a whole post while retaining its source, use `published: false`.
- To hide a section, use `{% comment %}` and `{% endcomment %}`. This also avoids
  HTML comments ending unexpectedly on Mermaid arrows containing `-->`.
- Keep intentionally hidden content hidden unless the user asks otherwise.
  The ONNX robot deployment post is currently unpublished; the ECG paper
  analysis in the Obermeyer visit post is currently commented out.
- Do not publish empty drafts or unrelated files just because they are present
  in the working tree.

## Verification and Git workflow

Inspect the working tree before edits. Preserve unrelated changes and stage
explicit paths for the requested task rather than using a blanket `git add`.

For prose and metadata changes, review the diff, check whitespace, verify link
targets against filenames and permalink settings, and check relevant Markdown
or Liquid rendering when available. For layout changes, also check affected
pages and responsive behavior. Do not claim a full site build or live deployment
was verified when only a fragment or local diff was checked.

Commit and push when requested or already authorized within the ongoing task;
do not ask again for an authorized push. Otherwise, state clearly that changes
remain local. Report the commit and push result concisely. A successful push is
not itself evidence that the GitHub Pages deployment has completed.

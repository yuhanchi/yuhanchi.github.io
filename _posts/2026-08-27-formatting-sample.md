---
title: A formatting sample
description: Headings, lists, quotes, code, tables and footnotes — a page for checking how long-form reading feels on this site.
tags: [meta]
---

This post exists to show how different kinds of content look on this site. Good reading pages need a comfortable line length, enough space between lines, and clear but quiet structure.

## Paragraphs and emphasis

Body text should be easy to scan and easy to read for a long time. Here is some *emphasis*, some **strong text**, a bit of `inline code`, and [a link](https://github.com/Chi-Shan0707). Footnotes keep side remarks out of the way.[^1]

## Lists

1. Write down the problem in plain words.
2. Find the smallest example that still shows the difficulty.
3. Only then reach for heavier tools.

- Unordered lists work too
- with short items
  - and nested ones.

## A quotation

> The purpose of computing is insight, not numbers.
>
> — Richard Hamming

## Code

```python
def bisect(f, lo, hi, tol=1e-10):
    """Find a root of f in [lo, hi], assuming f(lo) and f(hi) differ in sign."""
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if f(lo) * f(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2
```

## A table

| Method        | Needs a model? | Typical use               |
|---------------|:--------------:|---------------------------|
| Value iteration | yes          | small, known MDPs         |
| Q-learning    | no             | discrete actions          |
| PPO           | no             | continuous control, LLMs  |

---

That is everything a typical post uses.

[^1]: Like this one.

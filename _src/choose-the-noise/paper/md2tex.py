"""Convert the blog post (Kramdown/GFM + MathJax) into the LaTeX body of the PDF version.

The converter only handles the constructs used in this post: headings, paragraphs,
bold/italic, links, inline and display math, lists, GFM tables, <figure> blocks,
<details> blocks, and the reference list.  Author-year citations, links to cited papers, and
"Figure/Table/Section N" become internal PDF links.  Run:  python3 md2tex.py  (writes body.tex, refs.tex).
"""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
POST = os.path.join(HERE, "..", "..", "..", "_posts", "2026-10-18-choose-the-noise.md")
SITE = "https://yuhanchi.github.io"

TEXT_ESC = [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("#", r"\#"),
            ("_", r"\_"), ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"),
            ("^", r"\textasciicircum{}")]


def esc_text(s):
    for a, b in TEXT_ESC:
        s = s.replace(a, b)
    return s


def esc_url(u):
    return u.replace("%", r"\%").replace("#", r"\#").replace("_", r"\_")


TOKENS = []

# Cross-references: Figure/Table/Section numbers in the post -> LaTeX labels.
FIG_LABELS = {"1": "fig:vae-correspondence", "2": "fig:tilt-noise-tilts-policy", "3": "fig:five-routes",
              "4": "fig:experiment-training", "5": "fig:experiment-routes"}

# Citations: filled by load_refs() from the post's reference list.
REFS = []        # [(key, surname, year, text)] in the post's order
REF_URLS = {}    # url -> key
# Method names used in the text, with a marker that identifies their reference entry.
METHODS = {"PARROT": "PARROT:", "PLAS": "PLAS:", "DSRL": "(DSRL)", "LP-DS": "(LP-DS)",
           "GoldenStart": "GoldenStart:", "FlowDAgger": "FlowDAgger:", "UniSteer": "UniSteer:",
           "Flow Reversal Steering": "Flow Reversal Steering", "ReFORM": "ReFORM:",
           "Noise Hypernetworks": "Noise Hypernetworks:", "LPS": "(LPS)", "Noise PPO": "(Noise PPO)",
           "GoRL": "GoRL:", "Golden Ticket": "Golden Ticket", "PDP": "(PDP)", "LAFM": "(LAFM)",
           "ReNO": "ReNO:", "D-Flow": "D-Flow:"}
METHOD_KEYS = {}
STATE = {"internal": True}   # False while rendering the reference list itself


def load_refs(md):
    refs = md.split("## References\n", 1)[1].split("\n<details>", 1)[0]
    for n, it in enumerate(l[2:] for l in refs.strip().split("\n") if l.startswith("- ")):
        surname = it.split(",", 1)[0]
        year = re.search(r"\((\d{4})\)", it).group(1)
        key = f"ref:{n + 1}"
        REFS.append((key, surname, year, it))
        for url in re.findall(r"\]\(([^)]+)\)", it):
            REF_URLS[url] = key
    for name, marker in METHODS.items():
        hits = [k for k, _, _, t in REFS if marker in t]
        assert len(hits) == 1, name
        METHOD_KEYS[name] = hits[0]


def cite_key(authors, year, before):
    """Resolve an author-year citation; `before` is the text preceding it (for method names)."""
    surname = re.split(r" et al\.| & |, ", authors)[0].strip()
    hits = [k for k, sn, y, _ in REFS if sn == surname and y == year]
    if len(hits) > 1:
        hits = [METHOD_KEYS[n] for n in METHODS if before.rstrip(" (").endswith(n) and METHOD_KEYS[n] in hits]
    return hits[0] if len(hits) == 1 else None


def inline(s, top=True, methods=False):
    """Convert one paragraph of inline Markdown into LaTeX."""
    tokens = TOKENS

    def keep(x):
        tokens.append(x)
        return f"\x00{len(tokens) - 1}\x00"

    # math first, so nothing inside it is touched
    s = re.sub(r"\$\$(.+?)\$\$", lambda m: keep("$" + m.group(1).strip() + "$"), s)
    s = re.sub(r"\\\((.+?)\\\)", lambda m: keep("$" + m.group(1).strip() + "$"), s)
    # Liquid links to site files
    s = re.sub(r"\{\{ '([^']+)' \| relative_url \}\}", lambda m: SITE + m.group(1), s)

    def link(m):
        text, url = m.group(1), m.group(2)
        if STATE["internal"] and url in REF_URLS:      # a cited paper: jump to its reference entry
            return keep(r"\hyperlink{" + REF_URLS[url] + "}{" + inline(text, False) + "}")
        return keep(r"\href{" + esc_url(url) + "}{" + inline(text, False) + "}")
    s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", link, s)

    if STATE["internal"]:
        def cite(m, narrative):
            authors, year = m.group(1), m.group(2)
            key = cite_key(authors, year, m.string[:m.start()])
            if key is None:
                return m.group(0)
            if narrative:                                  # Eyring et al. (2025)
                return keep(r"\hyperlink{" + key + "}{" + inline(authors, False) + " (" + year + ")}")
            return "(" + keep(r"\hyperlink{" + key + "}{" + inline(authors, False) + ", " + year + "}") + ")"
        name = r"([A-Z][\w-]+(?: et al\.| & [A-Z][\w-]+)?)"
        s = re.sub(r"\(" + name + r", (\d{4})\)", lambda m: cite(m, False), s)
        s = re.sub(name + r" \((\d{4})\)", lambda m: cite(m, True), s)
        if methods:                                        # method names in the map table
            for n in sorted(METHODS, key=len, reverse=True):
                s = re.sub(r"(?<![\w-])" + re.escape(n) + r"(?![\w-])",
                           lambda m: keep(r"\hyperlink{" + METHOD_KEYS[m.group(0)] + "}{" + esc_text(m.group(0)) + "}"), s)
        s = re.sub(r"\b(Figure|Table|Section) (\d)([ab]?)\b",
                   lambda m: keep(m.group(1) + "~\\ref{" + {"Figure": FIG_LABELS, "Table": TABLE_LABELS}.get(
                       m.group(1), {}).get(m.group(2), "sec:" + m.group(2)) + "}" + m.group(3)), s)
    s = re.sub(r"<strong>(.+?)</strong>", lambda m: keep(r"\textbf{" + inline(m.group(1), False) + "}"), s)
    s = re.sub(r"\*\*(.+?)\*\*", lambda m: keep(r"\textbf{" + inline(m.group(1), False) + "}"), s)
    s = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", lambda m: keep(r"\emph{" + inline(m.group(1), False) + "}"), s)
    s = esc_text(s)
    # typography
    s = re.sub(r'"(\S[^"]*?)"', r"``\1''", s)
    s = re.sub(r"(?<=\w)'(?=\w)", "'", s)
    s = s.replace("’", "'").replace("“", "``").replace("”", "''")
    s = s.replace("—", "---").replace("–", "--")
    s = s.replace("±", r"$\pm$").replace("°", r"$^\circ$").replace(" ms", r"\,ms")
    s = re.sub(r"(\d)%", r"\1\\%", s)
    if not top:
        return s
    while "\x00" in s:
        s = re.sub(r"\x00(\d+)\x00", lambda m: tokens[int(m.group(1))], s)
    return s


def table(lines, caption=None, label=None, spec=None):
    rows = [[c.strip() for c in l.strip().strip("|").split("|")] for l in lines]
    head, body = rows[0], rows[2:]
    n = len(head)
    spec = spec or ("l" + "X" * (n - 1))
    out = [r"\begin{table}[htbp]", r"\centering" + (r"\footnotesize" if label == "tab:map" else r"\small")]
    if caption:
        out.append(r"\caption{" + inline(caption) + "}")
    if label:
        out.append(r"\label{" + label + "}")
    out += [r"\begin{tabularx}{\linewidth}{" + spec + "}", r"\toprule",
            " & ".join(inline(c) for c in head) + r" \\", r"\midrule"]
    out += [" & ".join(inline(c, methods=(label == "tab:map" and k == 1)) for k, c in enumerate(r)) + r" \\"
            for r in body]
    out += [r"\bottomrule", r"\end{tabularx}", r"\end{table}"]
    return "\n".join(out)


TABLE_SPECS = {
    "tab:notation": r"@{}p{0.2\linewidth}>{\raggedright\arraybackslash}X@{}",
    "tab:vae": r"@{}>{\raggedright\arraybackslash}X>{\raggedright\arraybackslash}X@{}",
    "tab:routes": r"@{}>{\raggedright\arraybackslash}p{0.27\linewidth}>{\raggedright\arraybackslash}Xcc@{}",
    "tab:map": r"@{}>{\raggedright}p{0.12\linewidth}>{\raggedright}p{0.14\linewidth}>{\raggedright}p{0.17\linewidth}>{\raggedright}X>{\raggedright\arraybackslash}p{0.2\linewidth}@{}",
}


# The PDF is more formal than the post: third person, no web-only asides.
FORMAL = [
    ("It is a useful map. I wanted to look more closely at one family that is not brought together as a group in that survey.",
     "It is a useful map. This note examines one family that the survey does not treat as a group."),
    ("Imagine a robot policy", "Consider a robot policy"),
    ("This post follows that thread", "This note follows that thread"),
    ("The post makes two points and then checks them on a small example.",
     "The note makes two points and then checks them on a small example."),
    ("Table 1 (click to expand) collects the notation used in the rest of the post;",
     "Table 1 collects the notation used in the rest of the note;"),
    ("In practice, it turned out to be.", "Empirically, DSRL finds that it is."),
    ("Two facts make this reading useful rather than decorative.", "Two facts make this reading useful."),
    ("To see the five routes at work, I trained the same noise proposer",
     "To see the five routes at work, we train the same noise proposer"),
    ("The code for this section and all figures is [on GitHub]",
     "Code for this section and all figures is available [on GitHub]"),
    ("Three things stand out.", "Three observations follow."),
    ("Three neighbors are worth knowing.", "Three neighboring lines of work are relevant."),
    ("At \\(\\beta = \\infty\\) we recover the base policy.", "At \\(\\beta = \\infty\\) the base policy is recovered."),
]

PDF_TABLE_CAPTIONS = {
    "1": r"Notation. The decoder parameters \(\theta\) are pretrained and frozen; only the proposer parameters \(\phi\) are learned.",
    "2": "Correspondence between the terms of a VAE and KL-regularized noise-space RL.",
    "4": "Noise-space methods and close relatives, grouped by the route through which value reaches the noise. "
         "``Prior control'' distinguishes actual constraints from initialization or supervision alone.",
}
TABLE_LABELS = {"1": "tab:notation", "2": "tab:vae", "3": "tab:routes", "4": "tab:map"}


def convert(md):
    for a, b in FORMAL:
        assert md.count(a) == 1, a
        md = md.replace(a, b)
    md = md.split("\n---\n", 1)[1]                              # drop front matter
    md = md.split("\n<details>\n<summary>Cite this post", 1)[0]   # cite block handled in main.tex
    lines = md.split("\n")
    out, i, sec = [], 0, 0
    para = []

    def flush():
        if para:
            out.append(inline(" ".join(para)))
            out.append("")
            para.clear()

    while i < len(lines):
        l = lines[i]
        if l.startswith("*This post is also available as a [PDF]"):   # web-only line
            i += 1
            continue
        if l.startswith("## "):
            flush()
            title = re.sub(r"^\d+\.\s*", "", l[3:])
            if title == "References":
                break
            out.append(r"\section{" + inline(title) + "}")
            out.append(r"\label{sec:" + str(sec := sec + 1) + "}")
            i += 1
            continue
        if l.strip() == "$$":                                      # display math
            flush()
            j = i + 1
            while lines[j].strip() != "$$":
                j += 1
            body = "\n".join(lines[i + 1:j]).strip()
            if body.startswith(r"\begin{aligned}"):
                body = body.replace(r"\begin{aligned}", r"\begin{aligned}", 1)
            out.append(r"\begin{equation}" + "\n" + body + "\n" + r"\end{equation}")
            i = j + 1
            continue
        if l.startswith('<figure class="table"'):
            flush()
            j = i
            while not lines[j].startswith("</figure>"):
                j += 1
            block = lines[i:j]
            num, cap = re.search(r"<figcaption[^>]*><strong>Table (\d+)\.</strong>\s*(.*?)</figcaption>", "\n".join(block)).groups()
            tl = [b for b in block if b.startswith("|")]
            label = TABLE_LABELS[num]
            out.append(table(tl, cap, label, TABLE_SPECS[label]))
            out.append("")
            i = j + 1
            continue
        if l.startswith("<figure>"):
            flush()
            j = i
            while not lines[j].startswith("</figure>"):
                j += 1
            block = "\n".join(lines[i:j])
            name = re.search(r"choose-the-noise/([\w-]+)\.png", block).group(1)
            cap = re.search(r"<figcaption><strong>Figure \d+\.</strong>\s*(.*?)</figcaption>", block, re.S).group(1)
            width = {"five-routes": r"0.82\linewidth"}.get(name, r"\linewidth")
            out += [r"\begin{figure}[htbp]", r"\centering",
                    r"\includegraphics[width=" + width + "]{figures/" + name + ".pdf}",
                    r"\caption{" + inline(cap) + "}", r"\label{fig:" + name + "}", r"\end{figure}", ""]
            i = j + 1
            continue
        if l.startswith("<details"):
            flush()
            j = i
            while not lines[j].startswith("</details>"):
                j += 1
            block = [b for b in lines[i + 1:j] if not b.startswith("<summary>")]
            summary = re.search(r"<summary>(.*?)</summary>", "\n".join(lines[i:j])).group(1)
            m = re.match(r"<strong>Table (\d+)\.</strong>", summary)
            if m:
                label = TABLE_LABELS[m.group(1)]
                tl = [b for b in block if b.startswith("|")]
                out.append(table(tl, PDF_TABLE_CAPTIONS[m.group(1)], label, TABLE_SPECS[label]))
                out.append("")
                i = j + 1
                continue
            out.append(r"\begin{proof}[" + inline(summary) + "]")
            out.append(convert_simple(block))
            out.append(r"\end{proof}")
            out.append("")
            i = j + 1
            continue
        if re.match(r"^(\d+\.|-) ", l):                            # lists
            flush()
            env = "enumerate" if l[0].isdigit() else "itemize"
            out.append(r"\begin{" + env + "}")
            while i < len(lines) and re.match(r"^(\d+\.|-) ", lines[i]):
                out.append(r"\item " + inline(re.sub(r"^(\d+\.|-) ", "", lines[i])))
                i += 1
            out.append(r"\end{" + env + "}")
            out.append("")
            continue
        if l.strip() == "":
            flush()
        else:
            para.append(l.strip())
        i += 1
    flush()
    tex = "\n".join(out)
    tex = re.sub(r"^\\textbf\{([^{}]*(?:\{[^{}]*\}[^{}]*)*?[.?])\} ", r"\\paragraph{\1} ", tex, flags=re.M)
    return formalize(tex)


def formalize(tex):
    """Turn the two exact results of Section 2 into stated propositions."""
    a = tex.index(r"\paragraph{Tilting the noise tilts the policy.}")
    b = tex.index(r"\end{equation}", a) + len(r"\end{equation}")
    head = tex[a:b].replace(r"\paragraph{Tilting the noise tilts the policy.} ", "")
    tex = tex[:a] + r"\begin{proposition}[Tilting the noise tilts the policy]\label{prop:tilt}" + "\n" + head + "\n" + r"\end{proposition}" + tex[b:]
    tex = tex.replace(r"\begin{proof}[Why the tilt commutes with decoding]", r"\begin{proof}[Proof of Proposition~\ref{prop:tilt}]")
    a = tex.index(r"\paragraph{The noise-space KL is a conservative proxy.}")
    b = tex.index(r"\end{equation}", a) + len(r"\end{equation}")
    head = tex[a:b].replace(r"\paragraph{The noise-space KL is a conservative proxy.} ", "")
    tex = tex[:a] + r"\begin{proposition}[The noise-space KL is a conservative proxy]\label{prop:dpi}" + "\n" + head + "\n" + r"\end{proposition}" + tex[b:]
    return tex


def convert_simple(block):
    out, i = [], 0
    while i < len(block):
        if block[i].strip() == "$$":
            j = i + 1
            while block[j].strip() != "$$":
                j += 1
            out.append(r"\begin{equation*}" + "\n" + "\n".join(block[i + 1:j]).strip() + "\n" + r"\end{equation*}")
            i = j + 1
        else:
            if block[i].strip():
                out.append(inline(block[i]))
            i += 1
    return "\n".join(out)


def references():
    """The reference list, alphabetical, with a link target for every entry."""
    STATE["internal"] = False
    out = [r"\section*{References}", r"\begin{list}{}{\leftmargin=1.5em \itemindent=-1.5em \itemsep=2pt \parsep=0pt}", r"\small"]
    for key, _, _, it in sorted(REFS, key=lambda r: (r[1].lower(), r[2], r[3].lower())):
        out.append(r"\item \raisebox{\ht\strutbox}{\hypertarget{" + key + "}{}}" + inline(it))
    out.append(r"\end{list}")
    STATE["internal"] = True
    return "\n".join(out)


if __name__ == "__main__":
    md = open(POST, encoding="utf-8").read()
    load_refs(md)
    body = convert(md)
    open(os.path.join(HERE, "body.tex"), "w", encoding="utf-8").write(body + "\n")
    open(os.path.join(HERE, "refs.tex"), "w", encoding="utf-8").write(references() + "\n")
    print("wrote body.tex and refs.tex")

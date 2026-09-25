#!/usr/bin/env python3
"""
render.py — data  ->  the published site.

    data/content.json       profile, intro, research categories, per-paper figure + synopsis
    data/publications.json  the publication list (verified against Crossref)
    data/talks.json         invited talks and minisymposia (from the CV)
    data/jobs.json          open and upcoming positions, with their flyers
    data/interests.json     research-interest topics (text + generated figure)
    data/papers.json        paper facts, regenerated from the archive by scan_papers.py
    assets/                 source images: photo, heroes/, paperfigs/, interests/
    build/theme.py          the design
            |
            v
    docs/   index.html (Research) · interests.html · publications.html · talks.html
            jobs.html · CV_Hahn.pdf (compiled from the CV .tex) · jobs/<flyer>.html · assets/...

The build is all-or-nothing: every input is checked first, the site is written to a
temporary sibling directory, and only a complete build replaces docs/.  Any missing
figure or unknown paper key stops the build with a list of problems.

Usage
  python build/render.py                 # build into docs/
  python build/render.py --out _preview  # build into another subfolder of site/
"""
from __future__ import annotations

import argparse
import base64
import html
import json
import shutil
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    sys.exit("Pillow is required:  pip install --user Pillow")

import theme

HERE = Path(__file__).resolve().parent
SITE = HERE.parent
DATA = SITE / "data"

HERO_W = 720      # hero images: shown <= 360 px, 2x for retina
FIG_W = 360       # paper figures: shown <= 150 px
TOPIC_W = 840     # research-interest figures: shown <= 420 px
PHOTO_W = 320

SELF = "Jooyoung Hahn"
PROBLEMS: list[str] = []          # collected during rendering; any entry fails the build


def esc(s) -> str:
    return html.escape(str(s or ""), quote=True)


# Hyphenated tokens that must not break at the end of a line.
_NOBREAK = {"3-D": "3‑D", "2-D": "2‑D", "p-Poisson": "p‑Poisson", "p-elastica": "p‑elastica"}


def txt(s) -> str:
    """Escape visible prose and keep short hyphenated tokens on one line."""
    out = esc(s)
    for a, b in _NOBREAK.items():
        out = out.replace(a, b)
    return out


def load(name: str):
    p = DATA / name
    if not p.exists():
        sys.exit(f"render.py: missing {p}")
    return json.loads(p.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- images
def web_image(src: Path, dst: Path, width: int, fmt: str = "JPEG") -> tuple[int, int] | None:
    """Resize src to at most `width` px wide, save to dst; return (w, h)."""
    if not src.exists():
        PROBLEMS.append(f"missing image {src}")
        return None
    try:
        with Image.open(src) as im:
            im = im.convert("RGB")
            if im.width > width:
                im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
            dst.parent.mkdir(parents=True, exist_ok=True)
            if fmt == "WEBP":
                im.save(dst, "WEBP", quality=84, method=6)
            else:
                im.save(dst, "JPEG", quality=84, optimize=True, progressive=True)
            return im.size
    except Exception as exc:
        PROBLEMS.append(f"image failed {src}: {exc}")
        return None


def img_tag(src: str, size, alt: str, cls: str = "", lazy: bool = True) -> str:
    w, h = size
    c = f' class="{cls}"' if cls else ""
    ld = ' loading="lazy" decoding="async"' if lazy else ""
    return f'<img{c} src="{esc(src)}" width="{w}" height="{h}" alt="{esc(alt)}"{ld}>'


# --------------------------------------------------------------------------- references
def fmt_authors(names: list[str]) -> str:
    out = [f"<b>{esc(n)}</b>" if n == SELF else esc(n) for n in names]
    if len(out) <= 1:
        return "".join(out)
    if len(out) == 2:
        return f"{out[0]} and {out[1]}"
    return ", ".join(out[:-1]) + ", and " + out[-1]


def fmt_source(e: dict) -> str:
    """'<i>Journal</i>, 64(4), 1443–1477, 2026' in the usual reference order."""
    status = e.get("status", "published")
    if e.get("type") == "presentation":
        note = f" · {esc(e['note'])}" if e.get("note") else ""
        return f"{esc(e.get('venue'))}, {esc(e.get('place'))}, {esc(e.get('date') or e.get('year'))}{note}"
    if e.get("type") == "patent":
        return f"{esc(e.get('venue'))}, {e.get('year')}"
    if e.get("type") == "preprint":
        return f"Preprint, {esc(e.get('venue'))}, {esc(status)}, {e.get('year')}"
    if e.get("type") == "submitted":
        return f"Submitted to <i>{esc(e.get('venue'))}</i>, {e.get('year')}"
    parts = [f"<i>{esc(e.get('venue'))}</i>"]
    if status in ("accepted", "in press"):
        parts.append(esc(status))
    vol, iss, pages = e.get("volume"), e.get("issue"), e.get("pages")
    if vol:
        parts.append(esc(vol) + (f"({esc(iss)})" if iss else ""))
    if pages:
        parts.append(esc(pages))
    if e.get("year"):
        parts.append(str(e["year"]))
    return ", ".join(parts)


def paper_url(e: dict) -> str:
    """Where a click on the paper goes: an explicit link, else the DOI, else a URL."""
    if e.get("link"):
        return e["link"]
    if e.get("doi"):
        return "https://doi.org/" + e["doi"]
    return e.get("url") or ""


def link_html(e: dict) -> str:
    nt = ' target="_blank" rel="noopener"'
    if e.get("link"):
        return f' · <a class="doi" href="{esc(e["link"])}"{nt}>{esc(e.get("link_label") or "Paper")}</a>'
    if e.get("doi"):
        return f' · <a class="doi" href="https://doi.org/{esc(e["doi"])}"{nt}>doi:{esc(e["doi"])}</a>'
    if e.get("url"):
        u = e["url"]
        label = "Google Patents" if e.get("type") == "patent" else u.split("//", 1)[-1]
        return f' · <a class="doi" href="{esc(u)}"{nt}>{esc(label)}</a>'
    return ""


# --------------------------------------------------------------------------- page shell
NAV = [  # id, sidebar label, phone label, file
    ("research", "Research", "Research", "index.html"),
    ("interests", "Research Interests", "Interests", "interests.html"),
    ("publications", "Publications", "Papers", "publications.html"),
    ("funding", "Funding", "Funding", "funding.html"),
    ("talks", "Talks", "Talks", "talks.html"),
    ("jobs", "Jobs", "Jobs", "jobs.html"),
]


def shell(content: dict, page: str, fname: str, title: str, body: str, build: dict) -> str:
    p = content["profile"]
    u = base64.b64encode(p["email_user"].encode()).decode()
    d = base64.b64encode(p["email_domain"].encode()).decode()

    def nav_links(short: bool) -> str:
        out = []
        for pid, label, slabel, href in NAV:
            cur = ' aria-current="page"' if pid == page else ""
            out.append(f'<a href="{href}"{cur}>{esc(slabel if short else label)}</a>')
        if build.get("cv_href"):
            out.append(f'<a href="{esc(build["cv_href"])}" target="_blank" rel="noopener">CV'
                       f'<span class="ext">PDF</span></a>')
        return "".join(out)

    mail = f'<a href="#" class="mail" data-u="{u}" data-d="{d}">Email</a>'
    profiles = mail + "".join(f'<a href="{esc(l["href"])}" target="_blank" rel="noopener">{esc(l["label"])}</a>'
                              for l in p.get("profiles", []))
    affil = "<br>".join(esc(a) for a in p.get("affiliation", []))
    photo = img_tag("assets/photo.jpg", build["photo_size"], f"Photo of {p['name']}", "photo", lazy=False)

    who = (f'<p class="name"><a href="index.html">{esc(p["name"])}</a></p>'
           f'<p class="t">{esc(p["title"])}</p>'
           f'<p class="g">{esc(p.get("group", ""))}</p>')

    desc = content.get("meta_description") or f"{p['name']}, {p['title']}"
    page_title = p["name"] if page == "research" else f"{title} · {p['name']}"
    base = build["site_url"]
    page_url = base + ("" if fname == "index.html" else fname)
    canon = (f'<link rel="canonical" href="{esc(page_url)}">\n'
             f'<meta property="og:url" content="{esc(page_url)}">\n'
             f'<meta property="og:image" content="{esc(base)}assets/photo.jpg">') if base else ""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(page_title)}</title>
<meta name="description" content="{esc(desc)}">
<meta property="og:title" content="{esc(page_title)}">
<meta property="og:description" content="{esc(desc)}">
<meta property="og:type" content="website">
{canon}
<meta name="theme-color" content="#D8E5F0" media="(prefers-color-scheme: light)">
<meta name="theme-color" content="#182633" media="(prefers-color-scheme: dark)">
<link rel="icon" href="assets/favicon.svg" type="image/svg+xml">
<link rel="stylesheet" href="assets/site.css">
</head>
<body>
<div class="shell">
  <aside class="side">
    {photo}
    <div class="who">{who}<p class="a">{affil}</p></div>
    <nav class="nav" aria-label="Site">{nav_links(False)}</nav>
    <div class="links">{profiles}</div>
  </aside>
  <header class="mtop">
    {photo}
    <div class="who">{who}</div>
  </header>
  <nav class="mnav" aria-label="Site"><div class="strip">{nav_links(True)}</div></nav>
  <main>
    <div class="wrap">
{body}
    <footer class="site">
      <div class="mlinks">{profiles}</div>
      <p>{affil.replace("<br>", ", ")}</p>
      <p>Updated {esc(build["updated"])}</p>
    </footer>
    </div>
  </main>
</div>
<script>
(function(){{
  Array.prototype.forEach.call(document.querySelectorAll('.mail'),function(a){{
    a.addEventListener('click',function(e){{e.preventDefault();
      location.href='mailto:'+atob(a.getAttribute('data-u'))+String.fromCharCode(64)+atob(a.getAttribute('data-d'));}});
  }});
  Array.prototype.forEach.call(document.querySelectorAll('[data-until]'),function(el){{
    if(new Date()>new Date(el.getAttribute('data-until')+'T23:59:59'))el.hidden=true;
  }});
  var cur=document.querySelector('.mnav [aria-current]');
  if(cur){{var st=cur.parentNode;st.scrollLeft=Math.max(0,cur.offsetLeft-(st.clientWidth-cur.offsetWidth)/2);}}
  function openHash(){{
    var id=location.hash.slice(1);if(!id)return;
    var d=document.getElementById(id);if(d&&d.tagName==='DETAILS'){{d.open=true;}}
  }}
  openHash();window.addEventListener('hashchange',openHash);
  Array.prototype.forEach.call(document.querySelectorAll('details.cat'),function(d){{
    d.addEventListener('toggle',function(){{
      var s=d.querySelector('.open');if(s)s.textContent=d.open?'Hide papers':'Show papers';}});
  }});
}})();
</script>
</body>
</html>
"""


# --------------------------------------------------------------------------- pages
def page_research(content, known, pubs_by_key, out: Path) -> str:
    parts = ['<h1 class="vh">Jooyoung Hahn</h1>']

    intro = (content.get("intro") or "").strip()
    if intro:
        parts.append('<div class="intro">' + "".join(
            f"<p>{txt(x.strip())}</p>" for x in intro.split("\n\n") if x.strip()) + "</div>")

    pmeta = content.get("papers", {})
    blocks = []
    for ci, cat in enumerate(content.get("categories", [])):
        cid = cat["id"]
        rows = []
        for key in cat.get("papers", []):
            meta, ed = known.get(key), pmeta.get(key, {})
            if not meta:
                PROBLEMS.append(f"category {cid}: paper key {key} not in papers.json")
                continue
            if not ed.get("include", True):
                continue
            pub = pubs_by_key.get(key, {})
            thumb = '<span class="nofig"></span>'
            if ed.get("figure_src"):
                size = web_image(SITE / ed["figure_src"], out / "assets" / "fig" / f"{key}.jpg", FIG_W)
                if size:
                    thumb = img_tag(f"assets/fig/{key}.jpg", size, ed.get("caption", ""), "pth")
            title = pub.get("title") or meta.get("title") or key
            url = paper_url(pub)
            title_html = (f'<a href="{esc(url)}" target="_blank" rel="noopener">{txt(title)}</a>'
                          if url else txt(title))
            year = pub.get("year") or meta.get("year")
            if pub.get("type") == "submitted":
                venue = f" · submitted to <i>{esc(pub['venue'])}</i>"
            elif pub.get("type") == "preprint":
                venue = f" · preprint, {esc(pub['venue'])}"
            else:
                venue = f" · <i>{esc(pub['venue'])}</i>" if pub.get("venue") else ""
            rows.append(f'<li>{thumb}<div>'
                        f'<p class="pt">{title_html}</p>'
                        f'<p class="pv">{esc(year)}{venue}</p>'
                        f'<p class="po">{txt(ed.get("synopsis", ""))}</p></div></li>')
        if not rows:
            continue

        hero = cat.get("hero") or {}
        hero_html = ""
        if hero.get("src"):
            size = web_image(SITE / hero["src"], out / "assets" / "hero" / f"{cid}.jpg", HERO_W)
            if size:
                cap = hero.get("caption", "")
                alt = "" if cap else cat["name"]          # the caption is read out already
                hero_html = (f'<span class="hero">'
                             f'{img_tag(f"assets/hero/{cid}.jpg", size, alt, lazy=ci > 0)}'
                             f'{f"<span class=cap>{txt(cap)}</span>" if cap else ""}</span>')

        blocks.append(
            f'<details class="cat" id="{esc(cid)}"><summary>{hero_html}'
            f'<h3>{txt(cat["name"])}</h3>'
            f'<span class="cd">{txt(cat.get("blurb", ""))}</span>'
            f'<span class="cm"><span class="n">{len(rows)} papers</span><span class="open">Show papers</span></span>'
            f'</summary><ul class="plist">{"".join(rows)}</ul></details>')

    parts.append('<h2 class="sec" style="margin-top:2.2rem">Research</h2>')
    parts.append('<div class="cats">' + "\n".join(blocks) + "</div>")
    return "\n".join(parts)


JUMP_JS = """<script>
(function(){
  var bar=document.querySelector('.jump');if(!bar)return;
  var links=[].slice.call(bar.querySelectorAll('a'));
  var secs=links.map(function(a){return document.getElementById(a.getAttribute('href').slice(1));});
  var root=document.documentElement;
  function sizes(){
    var m=document.querySelector('.mnav');
    var mh=m&&m.offsetParent!==null?m.offsetHeight:0;
    root.style.setProperty('--mnav',mh+'px');
    root.style.setProperty('--jump',bar.offsetHeight+'px');
  }
  function mark(){
    var cur=0,pad=parseFloat(getComputedStyle(root).scrollPaddingTop)||0;
    secs.forEach(function(s,i){
      if(s&&s.getBoundingClientRect().top<=pad+(parseFloat(getComputedStyle(s).scrollMarginTop)||0)+8)cur=i;});
    if(window.innerHeight+window.scrollY>=document.body.scrollHeight-2)cur=secs.length-1;
    links.forEach(function(a,i){if(i===cur)a.setAttribute('aria-current','true');else a.removeAttribute('aria-current');});
  }
  sizes();mark();
  window.addEventListener('resize',function(){sizes();mark();});
  window.addEventListener('scroll',mark,{passive:true});
})();
</script>"""


def jump_bar(label: str, items) -> str:
    """In-page tabs: every section stays on the page, a tab only scrolls to it."""
    links = "".join(f'<a href="#{esc(i)}"><span class="lb">{txt(t)}</span><span class="nm">{n}</span></a>'
                    for i, t, n in items)
    return f'<nav class="ptabs jump" aria-label="{esc(label)}">{links}</nav>'


def page_interests(interests, out: Path) -> str:
    parts = [f'<h1 class="page">{esc(interests.get("title", "Research Interests"))}</h1>']
    if interests.get("lede"):
        parts.append(f'<p class="lede">{interests["lede"]}</p>')
    topics = interests.get("topics", [])
    groups = interests.get("groups", [])
    for t in topics:
        if t.get("group") not in groups:
            PROBLEMS.append(f"interests: topic {t.get('id')} has unknown group {t.get('group')!r}")
    bar, secs = [], []
    for gi, group in enumerate(groups):
        items = [t for t in topics if t.get("group") == group]
        if not items:
            continue
        gid = f"g{gi + 1}"
        bar.append((gid, interests.get("tab_labels", {}).get(group, group), len(items)))
        lis = []
        for t in items:
            fig = ""
            if t.get("figure"):
                size = web_image(SITE / t["figure"], out / "assets" / "interests" / f"{t['id']}.webp", TOPIC_W, "WEBP")
                if size:
                    fig = img_tag(f"assets/interests/{t['id']}.webp", size, t.get("alt", ""))
            who = (f'<p class="with">With {txt(t["collaborators"])}.</p>' if t.get("collaborators") else "")
            lis.append(f'<li class="topic" id="{esc(t["id"])}">{fig}<div>'
                       f'<h3>{txt(t["label"])}</h3><p>{txt(t["description"])}</p>{who}</div></li>')
        secs.append(f'<section class="jsec" id="{gid}" aria-labelledby="{gid}-h">'
                    f'<h2 class="sec band" id="{gid}-h">{txt(group)}</h2><ul class="topics">{"".join(lis)}</ul></section>')
    parts.append(jump_bar("Research areas", bar))
    parts.extend(secs)
    parts.append(JUMP_JS)
    return "\n".join(parts)


PUB_TABS = [  # section id, tab label, entry types, section heading
    ("journal", "Journal", ("journal",), "Journal articles"),
    ("conference", "Conference", ("proceedings", "presentation"), "Conference papers and presentations"),
    ("preprint", "Preprint", ("preprint",), "Preprints"),
    ("patent", "Patent", ("patent",), "Patents"),
]


def pub_years(entries) -> str:
    order = {"proceedings": 0, "presentation": 1}
    out = []
    for y in sorted({e["year"] for e in entries}, reverse=True):
        ys = sorted((e for e in entries if e["year"] == y), key=lambda e: (order.get(e["type"], 0), e["title"]))
        lis = []
        for e in ys:
            kind = '<span class="kind">Proceedings</span>' if e["type"] == "proceedings" else ""
            lis.append(f'<li><p class="ti">{txt(e["title"])}{kind}</p>'
                       f'<p class="au">{fmt_authors(e["authors"])}</p>'
                       f'<p class="ven">{fmt_source(e)}{link_html(e)}</p></li>')
        out.append(f'<section class="yr"><h3>{y}</h3><ol class="pubs">{"".join(lis)}</ol></section>')
    return "".join(out)


def page_publications(pubs) -> str:
    entries = pubs.get("entries", [])
    parts = ['<h1 class="page">Publications</h1>']
    if pubs.get("lede"):
        parts.append(f'<p class="lede">{pubs["lede"]}</p>')
    bar, secs = [], []
    for tid, label, types, heading in PUB_TABS:
        items = [e for e in entries if e.get("type") in types]
        if not items:
            continue
        bar.append((tid, label, len(items)))
        note = (f'<p class="pnote">{esc(pubs["conference_note"])}</p>'
                if tid == "conference" and pubs.get("conference_note") else "")
        secs.append(f'<section class="jsec ppanel" id="{tid}" aria-labelledby="{tid}-h">'
                    f'<h2 class="sec band" id="{tid}-h">{heading}</h2>{note}{pub_years(items)}</section>')
    parts.append(jump_bar("Publication type", bar))
    parts.extend(secs)
    parts.append(JUMP_JS)
    return "\n".join(parts)


def page_funding(funding) -> str:
    parts = ['<h1 class="page">Funding</h1>']
    if funding.get("lede"):
        parts.append(f'<p class="lede">{esc(funding["lede"])}</p>')
    for g in funding.get("groups", []):
        lis = []
        for it in g.get("items", []):
            meta = " · ".join(x for x in (it.get("funder"), it.get("role")) if x)
            note = f'<p class="te">{txt(it["note"])}</p>' if it.get("note") else ""
            lis.append(f'<li><span class="td">{esc(it["period"])}</span><div>'
                       f'<p class="tt">{txt(it["title"])}</p>'
                       + (f'<p class="te">{txt(meta)}</p>' if meta else "") + f'{note}</div></li>')
        parts.append(f'<h2 class="sec">{txt(g["title"])}</h2><ol class="dlist">{"".join(lis)}</ol>')
    return "\n".join(x for x in parts if x)


def page_talks(talks) -> str:
    def items(lst):
        out = []
        for t in lst:
            links = " · ".join(f'<a href="{esc(l["href"])}" target="_blank" rel="noopener">{esc(l["label"])}</a>'
                               for l in t.get("links", []))
            links = f" · {links}" if links else ""
            out.append(f'<li><span class="td">{esc(t["date"])}</span><div>'
                       f'<p class="tt">{txt(t["title"])}</p>'
                       f'<p class="te">{esc(t["event"])}, {esc(t["place"])}{links}</p></div></li>')
        return "".join(out)

    parts = ['<h1 class="page">Talks</h1>']
    if talks.get("lede"):
        parts.append(f'<p class="lede">{talks["lede"]}</p>')
    parts.append(f'<h2 class="sec">Invited talks</h2><ol class="dlist">{items(talks.get("invited", []))}</ol>')
    parts.append(f'<h2 class="sec">Minisymposia organized</h2><ol class="dlist">{items(talks.get("minisymposia", []))}</ol>')
    return "\n".join(parts)


def page_jobs(jobs) -> str:
    """Each item: period, title, status, optional where, note and links; a link with
    "file" points at a flyer that build() copies into jobs/ unchanged."""
    nt = ' target="_blank" rel="noopener"'
    parts = ['<h1 class="page">Jobs</h1>']
    if jobs.get("lede"):
        parts.append(f'<p class="lede">{txt(jobs["lede"])}</p>')
    for g in jobs.get("groups", []):
        lis = []
        for it in g.get("items", []):
            meta = " · ".join(x for x in (it.get("where"), f'<span class="st">{esc(it["status"])}</span>'
                                          if it.get("status") else "") if x)
            note = f'<p class="te">{txt(it["note"])}</p>' if it.get("note") else ""
            links = " · ".join(f'<a href="{esc(l["href"])}"{nt}>{esc(l["label"])}</a>' for l in it.get("links", []))
            links = f'<p class="te">{links}</p>' if links else ""
            lis.append(f'<li><span class="td">{esc(it.get("period", ""))}</span><div>'
                       f'<p class="tt">{txt(it["title"])}</p>'
                       f'<p class="te">{meta}</p>{note}{links}</div></li>')
        parts.append(f'<h2 class="sec">{txt(g["title"])}</h2><ol class="dlist">{"".join(lis)}</ol>')
    return "\n".join(parts)


def compile_cv(tex: Path, dst: Path) -> None:
    """latexmk into a throw-away folder, so the CV folder gets no aux files."""
    with tempfile.TemporaryDirectory() as t:
        r = subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error",
                            f"-outdir={t}", tex.name], cwd=tex.parent, capture_output=True, text=True)
        pdf = Path(t) / (tex.stem + ".pdf")
        if r.returncode or not pdf.is_file():
            sys.exit(f"render.py: CV did not compile ({tex})\n" + r.stdout[-2000:])
        shutil.copyfile(pdf, dst)


# --------------------------------------------------------------------------- build
FAVICON = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">
<rect width="64" height="64" rx="10" fill="#D8E5F0"/>
<text x="32" y="42" text-anchor="middle" font-family="Georgia,serif" font-size="28" font-weight="600" fill="#10243A">JH</text>
</svg>
"""
KEEP = {"CNAME"}                   # files in docs/ that are not generated and must survive a rebuild


def safe_out(out_dir: str) -> Path:
    """Only ever (re)write a generated folder inside site/."""
    out = (SITE / out_dir).resolve()
    if out == SITE or SITE not in out.parents or out.name in {"build", "data", "assets"}:
        sys.exit(f"render.py: refusing to write to {out}")
    if out.exists() and any(out.iterdir()) and not (out / ".nojekyll").exists():
        sys.exit(f"render.py: {out} is not empty and was not made by render.py; refusing to replace it")
    return out


def build(out_dir: str = "docs") -> Path:
    out = safe_out(out_dir)
    content = load("content.json")
    papers = load("papers.json")
    pubs = load("publications.json")
    talks = load("talks.json")
    funding = load("funding.json")
    jobs = load("jobs.json")
    interests = load("interests.json")
    known = {p["key"]: p for p in papers.get("papers", [])}
    pubs["entries"] = [e for e in pubs.get("entries", []) if not e.get("hidden")]
    pubs_by_key = {e["paper_key"]: e for e in pubs["entries"] if e.get("paper_key")}

    b = content.get("build", {})
    cv_on = bool(b.get("cv_tex")) and b.get("cv_link", True)
    sources = {"photo": SITE / b.get("photo", "")}
    if cv_on:
        sources["cv_tex"] = (SITE / b["cv_tex"]).resolve()
    flyers = {}                                   # published name -> source file
    for g in jobs.get("groups", []):
        for it in g.get("items", []):
            for l in it.get("links", []):
                if l.get("file"):
                    src = (SITE / l["file"]).resolve()
                    sources[f'flyer {l["href"]}'] = src
                    flyers[l["href"]] = src
    missing = [f"{k}: {v}" for k, v in sources.items() if not v.is_file()]
    if missing:
        sys.exit("render.py: missing inputs\n  " + "\n  ".join(missing))

    tmp = out.with_name(out.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    (tmp / "assets").mkdir(parents=True)
    (tmp / "assets" / "site.css").write_text(theme.site_css(), encoding="utf-8")
    (tmp / "assets" / "favicon.svg").write_text(FAVICON, encoding="utf-8")
    (tmp / ".nojekyll").write_text("", encoding="utf-8")

    photo_size = web_image(sources["photo"], tmp / "assets" / "photo.jpg", PHOTO_W)
    cv_name = ""
    if cv_on:
        cv_name = b.get("cv_name", "CV_Hahn.pdf")
        compile_cv(sources["cv_tex"], tmp / cv_name)
    for href, src in flyers.items():            # byte-for-byte copies; the flyers are not edited
        (tmp / href).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, tmp / href)

    site_url = b.get("site_url", "")
    if site_url and not site_url.endswith("/"):
        site_url += "/"
    bld = {"photo_size": photo_size, "cv_href": cv_name,
           "site_url": site_url, "updated": b.get("updated") or date.today().strftime("%B %Y")}

    pages = {
        "index.html": ("research", "Research", page_research(content, known, pubs_by_key, tmp)),
        "interests.html": ("interests", "Research Interests", page_interests(interests, tmp)),
        "publications.html": ("publications", "Publications", page_publications(pubs)),
        "funding.html": ("funding", "Funding", page_funding(funding)),
        "talks.html": ("talks", "Talks", page_talks(talks)),
        "jobs.html": ("jobs", "Jobs", page_jobs(jobs)),
    }
    if PROBLEMS:
        shutil.rmtree(tmp)
        sys.exit("render.py: build stopped, docs/ left unchanged\n  " + "\n  ".join(PROBLEMS))
    for fname, (pid, title, body) in pages.items():
        (tmp / fname).write_text(shell(content, pid, fname, title, body, bld), encoding="utf-8")

    # swap in the finished build, carrying over the files that are not generated
    if out.exists():
        for k in KEEP:
            if (out / k).exists():
                shutil.copy2(out / k, tmp / k)
        shutil.rmtree(out)
    tmp.rename(out)

    n = {d: len(list((out / "assets" / d).glob("*"))) for d in ("hero", "fig", "interests")
         if (out / "assets" / d).exists()}
    kb = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) // 1024
    print(f"built {out}: {', '.join(pages)}  ({kb} KB)")
    print(f"  images: {n} · {cv_name or 'CV off'} · flyers: {len(flyers)}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs", help="output subfolder of site/ (default: docs)")
    build(ap.parse_args().out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

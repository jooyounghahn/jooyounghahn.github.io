#!/usr/bin/env python3
"""
scan_papers.py — read the paper archive and produce data/papers.json

This is the ONLY place that touches ../Ref_Papers/PAPERS_HAHN.  Everything
downstream (editor, renderer) reads data/papers.json, so the site always stays
consistent with the archive: add a paper there, re-run this, and it shows up.

Outputs
  data/papers.json          metadata + full figure inventory for every paper
  assets/thumbs/<key>/*.jpg small thumbnails, used by the editor's figure picker

Usage
  python build/scan_papers.py
"""
from __future__ import annotations

import io
import os
import json
import re
import sys
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    sys.exit("Pillow is required:  pip install --user Pillow")

HERE = Path(__file__).resolve().parent
SITE = HERE.parent
ARCHIVE = (SITE / ".." / ".." / "Ref_Papers" / "PAPERS_HAHN").resolve()
DATA = SITE / "data"
THUMBS = SITE / "assets" / "thumbs"

THUMB_W = 320          # picker thumbnail width
MIN_USEFUL_PX = 130    # below this the crop is almost certainly a fragment

IMG_EXT = {".jpg", ".jpeg", ".png", ".gif", ".webp"}


# --------------------------------------------------------------------------- md parsing
def read_md(md_path: Path) -> dict:
    """Pull title / authors / abstract out of a Mathpix markdown extraction."""
    try:
        text = md_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}

    body = text
    # strip the auto-generated YAML provenance block
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            body = text[end + 4:]

    title = ""
    m = re.search(r"^#\s+(.+?)\s*$", body, re.MULTILINE)
    if m:
        title = re.sub(r"\s+", " ", m.group(1)).strip()
        title = re.sub(r"\\?\$.*?\\?\$", "", title).strip()      # drop inline math
        title = re.sub(r"[*_`]", "", title).strip()

    abstract = ""
    m = re.search(r"#{1,3}\s*abstract\s*\n+(.+?)(?=\n#{1,3}\s|\Z)",
                  body, re.IGNORECASE | re.DOTALL)
    if m:
        abstract = re.sub(r"\s+", " ", m.group(1)).strip()
    if not abstract:
        m = re.search(r"\bAbstract\b[.:—-]*\s*(.{200,2000}?)(?=\n#|\bKeywords\b)",
                      body, re.IGNORECASE | re.DOTALL)
        if m:
            abstract = re.sub(r"\s+", " ", m.group(1)).strip()
    abstract = abstract[:1500]

    authors = ""
    if title:
        after = body.split(title, 1)[-1][:600]
        first = [ln.strip() for ln in after.splitlines() if ln.strip()]
        if first:
            cand = re.sub(r"<br\s*/?>", " ", first[0])
            cand = re.sub(r"\$\{?\s*\}?\^?\{?[^$]*\}?\$", "", cand)
            cand = re.sub(r"[*_`{}\\^]", "", cand)
            cand = re.sub(r"\s+", " ", cand).strip(" ,;")
            if 3 < len(cand) < 300 and not cand.lower().startswith(("abstract", "keywords")):
                authors = cand
    return {"title": title, "authors": authors, "abstract": abstract}


# --------------------------------------------------------------------------- figures
def scan_figures(assets_dir: Path, key: str) -> list[dict]:
    """Inventory every figure crop, with size info and a generated thumbnail."""
    if not assets_dir.is_dir():
        return []

    out_dir = THUMBS / key
    out_dir.mkdir(parents=True, exist_ok=True)

    figs = []
    for p in sorted(assets_dir.iterdir()):
        if p.suffix.lower() not in IMG_EXT or not p.is_file():
            continue
        try:
            with Image.open(p) as im:
                w, h = im.size
                rgb = im.convert("RGB")
                rgb.thumbnail((THUMB_W, THUMB_W), Image.LANCZOS)
                thumb_name = p.stem + ".jpg"
                buf = io.BytesIO()
                rgb.save(buf, "JPEG", quality=72, optimize=True)
                (out_dir / thumb_name).write_bytes(buf.getvalue())
        except Exception as exc:                                  # unreadable image
            print(f"    ! skipped {p.name}: {exc}")
            continue

        figs.append({
            "file": p.name,
            "w": w,
            "h": h,
            "bytes": p.stat().st_size,
            "thumb": f"assets/thumbs/{key}/{thumb_name}",
            # a wide, reasonably large crop is far more likely to be a real
            # result figure than a narrow sliver or a single table cell
            "fragment": (min(w, h) < MIN_USEFUL_PX or w * h < 90_000),
        })

    # best first: prefer big area, mild preference for landscape result panels
    figs.sort(key=lambda f: (f["fragment"], -(f["w"] * f["h"])))
    return figs


# --------------------------------------------------------------------------- main
def main() -> int:
    if not ARCHIVE.is_dir():
        sys.exit(f"paper archive not found: {ARCHIVE}")

    DATA.mkdir(parents=True, exist_ok=True)
    THUMBS.mkdir(parents=True, exist_ok=True)

    keys = sorted(p.stem for p in ARCHIVE.glob("*.md"))
    print(f"archive : {ARCHIVE}")
    print(f"papers  : {len(keys)}\n")

    papers = []
    for key in keys:
        md = ARCHIVE / f"{key}.md"
        pdf = ARCHIVE / f"{key}.pdf"
        assets = ARCHIVE / f"{key}.assets"

        meta = read_md(md)
        figs = scan_figures(assets, key)
        ym = re.search(r"(19|20)\d{2}", key)

        papers.append({
            "key": key,
            "year": int(ym.group(0)) if ym else 0,
            "title": meta.get("title", ""),
            "authors": meta.get("authors", ""),
            "abstract": meta.get("abstract", ""),
            "pdf": f"{key}.pdf" if pdf.exists() else "",
            "assets_dir": os.path.relpath(assets, SITE) if assets.is_dir() else "",
            "figures": figs,
            "n_figures": len(figs),
            "n_real": sum(1 for f in figs if not f["fragment"]),
        })
        good = papers[-1]["n_real"]
        print(f"  {key[:52]:52s} {papers[-1]['year']}  {len(figs):3d} figs ({good} usable)")

    # Two files.  papers.json is published with the site, so it holds only the
    # facts the pages need: no abstracts (publisher copyright) and no author
    # e-mail addresses, which the markdown often carries.  The full inventory,
    # used when choosing a figure, stays local (see .gitignore).
    slim = [{"key": p["key"], "year": p["year"], "title": p["title"]} for p in papers]
    (DATA / "papers.json").write_text(
        json.dumps({"source": os.path.relpath(ARCHIVE, SITE), "count": len(slim), "papers": slim},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    (DATA / "papers.local.json").write_text(
        json.dumps({"source": str(ARCHIVE), "count": len(papers), "papers": papers},
                   ensure_ascii=False, indent=1), encoding="utf-8")

    total = sum(p["n_figures"] for p in papers)
    print(f"\nwrote {DATA/'papers.json'} (published) and {DATA/'papers.local.json'} (local only)")
    print(f"  {len(papers)} papers, {total} figures, thumbnails in {THUMBS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

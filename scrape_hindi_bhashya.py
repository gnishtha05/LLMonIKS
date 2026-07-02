import argparse
import json
import random
import re
import time
from typing import Dict, List, Tuple, Optional

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

CHAPTER_VERSE_COUNT = {
    1: 47, 2: 72, 3: 43, 4: 42, 5: 29, 6: 47, 7: 30, 8: 28, 9: 34,
    10: 42, 11: 55, 12: 20, 13: 35, 14: 27, 15: 20, 16: 24, 17: 28, 18: 78
}

BASE_URL = "https://www.gitasupersite.iitk.ac.in/srimad"

DISPLAY_MARKER = "Display Selected Translations"
MOOL_MARKERS = ["मूल श्लोकः", "Mool Shloka"]
HTSHG_MARKER_SUBSTRS = [
    "Hindi Translation Of Sri Shankaracharya's Sanskrit Commentary",
    "Harikrishnadas Goenka",
]

STOP_MARKERS = [
    "Copyright",
    "Design by",
    "Translations and Commentaries",
    "Audio",
    "Script",
    "Chapter",
    "Shloka",
    "Display Selected Translations",
    "Sanskrit Commentary By",
    "Hindi Translation By",
    "English Translation By",
    "English Commentary By",
    "Hindi Commentary By",
]

VERSE_TAG_RE = re.compile(r"[।\.]{2}\s*\d+\.\d+\s*[।\.]{2}")

def clean_lines(lines: List[str]) -> List[str]:
    out = []
    for ln in lines:
        ln = (ln or "").strip()
        if not ln:
            continue
        ln = re.sub(r"\s+", " ", ln)
        out.append(ln)
    return out

def find_last_index(lines: List[str], predicate) -> int:
    idx = -1
    for i, ln in enumerate(lines):
        if predicate(ln):
            idx = i
    return idx

def slice_after_marker(lines: List[str], marker: str) -> List[str]:
    idx = find_last_index(lines, lambda x: x == marker)
    return lines[idx + 1:] if idx != -1 else lines

def find_block(lines: List[str], heading_pred) -> Tuple[int, int]:
    """
    Returns (start_idx, end_idx_exclusive) for block content AFTER a heading line.
    """
    start = find_last_index(lines, heading_pred)
    if start == -1:
        return (-1, -1)

    i = start + 1
    while i < len(lines):
        ln = lines[i]
        if any(ln.startswith(m) or ln == m for m in STOP_MARKERS):
            break
        i += 1
    return (start + 1, i)

def looks_like_htshg_heading(line: str) -> bool:
    return all(s in line for s in HTSHG_MARKER_SUBSTRS[:1])

def normalize_shloka(text: str) -> str:
    text = VERSE_TAG_RE.sub("", text).strip()
    return text

def normalize_htshg(text: str) -> str:
    text = text.strip()
    return text

def extract_main_text(html: str) -> List[str]:
    soup = BeautifulSoup(html, "lxml")

    main = (
        soup.select_one("main")
        or soup.select_one("div.region-content")
        or soup.select_one("div#content")
        or soup.select_one("div#main")
        or soup.body
    )

    raw_lines = list(main.stripped_strings) if main else list(soup.stripped_strings)
    lines = clean_lines(raw_lines)

    lines = slice_after_marker(lines, DISPLAY_MARKER)
    return lines

def parse_single_verse_page(html: str, key: str) -> Tuple[str, str]:
    """
    Returns (sanskrit_shloka, htshg_hindi_commentary_translation).
    Missing parts -> "".
    """
    lines = extract_main_text(html)

    mool_heading_idx = -1
    for mk in MOOL_MARKERS:
        idx = find_last_index(lines, lambda x, mk=mk: x == mk)
        if idx != -1:
            mool_heading_idx = idx
            break

    shloka_text = ""
    if mool_heading_idx != -1:
        i = mool_heading_idx + 1
        buf = []
        while i < len(lines):
            ln = lines[i]
            if looks_like_htshg_heading(ln):
                break
            if any(ln.startswith(m) or ln == m for m in STOP_MARKERS):
                break
            buf.append(ln)
            i += 1
        shloka_text = normalize_shloka("\n".join(buf).strip())

    def htshg_heading_pred(ln: str) -> bool:
        return "Hindi Translation Of Sri Shankaracharya's Sanskrit Commentary" in ln

    s, e = find_block(lines, htshg_heading_pred)
    htshg_text = ""
    if s != -1:
        htshg_text = normalize_htshg("\n".join(lines[s:e]).strip())

    return shloka_text, htshg_text

def fetch(session: requests.Session, chapter: int, verse: int, timeout: int = 30) -> str:
    params = {
        "language": "dv",               
        "choose": "1",                 
        "show_mool": "1",              
        "htshg": "1",                   
        "field_chapter_value": str(chapter),
        "field_nsutra_value": str(verse),
    }
    r = session.get(BASE_URL, params=params, timeout=timeout)
    r.raise_for_status()
    return r.text

def scrape_all(out_path: str,
               chapters: Optional[List[int]] = None,
               min_delay: float = 0.7,
               max_delay: float = 1.3,
               max_retries: int = 4) -> Dict[str, List[str]]:

    chapters = chapters or list(range(1, 19))

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (compatible; gita-scraper/1.0; +https://www.iitk.ac.in/)"
    })

    data: Dict[str, List[str]] = {}

    total = sum(CHAPTER_VERSE_COUNT[c] for c in chapters)
    pbar = tqdm(total=total, desc="Scraping", unit="verse")

    for ch in chapters:
        for v in range(1, CHAPTER_VERSE_COUNT[ch] + 1):
            key = f"{ch}.{v}"

            if key in data:
                pbar.update(1)
                continue

            last_err = None
            for attempt in range(1, max_retries + 1):
                try:
                    html = fetch(session, ch, v)
                    shloka_sa, htshg_hi = parse_single_verse_page(html, key)
                    data[key] = [shloka_sa or "", htshg_hi or ""]
                    last_err = None
                    break
                except Exception as e:
                    last_err = e
                    sleep_s = (2 ** (attempt - 1)) * 0.6 + random.random() * 0.4
                    time.sleep(sleep_s)

            if last_err is not None:
                data[key] = ["", ""]
                print(f"\n[WARN] Failed {key}: {last_err}")

            time.sleep(random.uniform(min_delay, max_delay))

            if (len(data) % 50) == 0:
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)

            pbar.update(1)

    pbar.close()

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return data

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default='hindi_bhashya.json', help="Output JSON file path")
    ap.add_argument("--chapters", default="1-18",
                    help="Chapters to scrape, e.g. '1-18' or '1,2,3'")
    ap.add_argument("--min-delay", type=float, default=0.7)
    ap.add_argument("--max-delay", type=float, default=1.3)
    args = ap.parse_args()

    if "-" in args.chapters:
        a, b = args.chapters.split("-", 1)
        chapters = list(range(int(a), int(b) + 1))
    else:
        chapters = [int(x.strip()) for x in args.chapters.split(",") if x.strip()]

    scrape_all(
        out_path=args.out,
        chapters=chapters,
        min_delay=args.min_delay,
        max_delay=args.max_delay,
    )

if __name__ == "__main__":
    main()

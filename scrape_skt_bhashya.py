import argparse
import json
import random
import time
from typing import Dict, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

BASE_URL = "https://www.gitasupersite.iitk.ac.in/srimad"

CHAPTER_VERSE_COUNT = {
    1: 47, 2: 72, 3: 43, 4: 42, 5: 29, 6: 47, 7: 30, 8: 28, 9: 34,
    10: 42, 11: 55, 12: 20, 13: 35, 14: 27, 15: 20, 16: 24, 17: 28, 18: 78
}

DISPLAY_MARKER = "Display Selected Translations"
MOOL_MARKERS = ["मूल श्लोकः", "Mool Shloka"]
SCSH_HEADING = "Sanskrit Commentary By Sri Shankaracharya"

STOP_MARKERS = {
    "Copyright",
    "Design by",
    "Translations and Commentaries",
    "Audio",
    "Script",
    "Chapter",
    "Shloka",
    "Display Selected Translations",
}

DID_NOT_COMMENT_PHRASE = "did not comment"

def norm(s: str) -> str:
    s = (s or "").replace("\u200c", "").replace("\u200d", "").replace("\ufeff", "")
    s = " ".join(s.split())
    return s.strip()

def extract_main_lines(html: str) -> List[str]:
    soup = BeautifulSoup(html, "lxml")
    main = (
        soup.select_one("main")
        or soup.select_one("div.region-content")
        or soup.select_one("div#content")
        or soup.select_one("div#main")
        or soup.body
    )
    raw = list(main.stripped_strings) if main else list(soup.stripped_strings)
    lines = [norm(x) for x in raw]
    lines = [x for x in lines if x] 

    if DISPLAY_MARKER in lines:
        idx = len(lines) - 1 - lines[::-1].index(DISPLAY_MARKER)
        lines = lines[idx + 1 :]
    return lines

def find_last_index(lines: List[str], target: str) -> int:
    for i in range(len(lines) - 1, -1, -1):
        if lines[i] == target:
            return i
    return -1

def parse_scsh_page(html: str) -> Tuple[str, str]:
    """
    Returns (sanskrit_shloka, shankara_sanskrit_commentary).
    Missing -> "".
    """
    lines = extract_main_lines(html)

    mool_idx = -1
    for mk in MOOL_MARKERS:
        mool_idx = find_last_index(lines, mk)
        if mool_idx != -1:
            break

    shloka = ""
    if mool_idx != -1:
        buf = []
        i = mool_idx + 1
        while i < len(lines):
            ln = lines[i]
            if ln == SCSH_HEADING:
                break
            if any(ln.startswith(x) or ln == x for x in STOP_MARKERS):
                break
            buf.append(ln)
            i += 1
        shloka = "\n".join(buf).strip()

    scsh_idx = find_last_index(lines, SCSH_HEADING)
    commentary = ""
    if scsh_idx != -1:
        buf = []
        i = scsh_idx + 1
        while i < len(lines):
            ln = lines[i]
            if any(ln.startswith(x) or ln == x for x in STOP_MARKERS):
                break
            buf.append(ln)
            i += 1
        commentary = "\n".join(buf).strip()

        if DID_NOT_COMMENT_PHRASE.lower() in commentary.lower():
            commentary = ""

    return shloka, commentary

def fetch(session: requests.Session, chapter: int, verse: int, timeout: int = 30) -> str:
    params = {
        "language": "dv",
        "choose": "1",
        "show_mool": "1",
        "scsh": "1",
        "field_chapter_value": str(chapter),
        "field_nsutra_value": str(verse),
    }
    r = session.get(BASE_URL, params=params, timeout=timeout)
    r.raise_for_status()
    return r.text

def scrape(out_path: str,
           chapters: Optional[List[int]] = None,
           min_delay: float = 0.7,
           max_delay: float = 1.3,
           max_retries: int = 4,
           key_mode: str = "chapter.verse") -> Dict[str, List[str]]:

    chapters = chapters or list(range(1, 19))

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (compatible; gita-scraper/1.0)"
    })

    data: Dict[str, List[str]] = {}
    total = sum(CHAPTER_VERSE_COUNT[c] for c in chapters)
    pbar = tqdm(total=total, desc="Scraping scsh", unit="verse")

    def make_key(ch: int, v: int) -> str:
        return f"{ch}.{v}" if key_mode == "chapter.verse" else str(v)

    for ch in chapters:
        for v in range(1, CHAPTER_VERSE_COUNT[ch] + 1):
            key = make_key(ch, v)

            last_err = None
            for attempt in range(1, max_retries + 1):
                try:
                    html = fetch(session, ch, v)
                    shloka_sa, shankara_sa = parse_scsh_page(html)
                    data[key] = [shloka_sa or "", shankara_sa or ""]
                    last_err = None
                    break
                except Exception as e:
                    last_err = e
                    backoff = (2 ** (attempt - 1)) * 0.6 + random.random() * 0.4
                    time.sleep(backoff)

            if last_err is not None:
                data[key] = ["", ""]
                print(f"\n[WARN] Failed {ch}.{v}: {last_err}")

            if len(data) % 50 == 0:
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)

            time.sleep(random.uniform(min_delay, max_delay))
            pbar.update(1)

    pbar.close()

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return data

def parse_chapters_arg(s: str) -> List[int]:
    s = s.strip()
    if "-" in s:
        a, b = s.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(x.strip()) for x in s.split(",") if x.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default='skt_bhashya.json', help="Output JSON path")
    ap.add_argument("--chapters", default="1-18", help="e.g. '1-18' or '1,2,3'")
    ap.add_argument("--min-delay", type=float, default=0.7)
    ap.add_argument("--max-delay", type=float, default=1.3)
    ap.add_argument("--keymode", choices=["chapter.verse", "verse"], default="chapter.verse")
    args = ap.parse_args()

    chapters = parse_chapters_arg(args.chapters)
    scrape(args.out, chapters=chapters, min_delay=args.min_delay, max_delay=args.max_delay, key_mode=args.keymode)

if __name__ == "__main__":
    main()

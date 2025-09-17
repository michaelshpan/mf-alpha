import os
import re
import time
import argparse
import pathlib
import requests
import zipfile
from urllib.parse import urljoin

from lxml import html  # pip install lxml

DATASET_PAGE = "https://www.sec.gov/data-research/sec-markets-data/mutual-fund-prospectus-riskreturn-summary-data-sets"
OUTDIR = pathlib.Path("sec_rr_datasets")

HREF_PAT = re.compile(r"/files/.*?/(\d{4})\s*Q\s*([1-4]).*?\.zip$", re.I)  # robust to spaces/case
LABEL_PAT = re.compile(r"(\d{4})\s*Q\s*([1-4])", re.I)                     # parse link text like "2025 Q2 MFRR"

def sec_headers():
    ua = os.getenv("SEC_USER_AGENT")
    if not ua or "@" not in ua:
        raise SystemExit(
            "SEC_USER_AGENT not set or missing an email.\n"
            "Example:\n  export SEC_USER_AGENT=\"Michael Pan michael.sh.pan@gmail.com\""
        )
    return {
        "User-Agent": ua,
        "Accept-Encoding": "gzip, deflate",
        "Host": "www.sec.gov",
        "Connection": "keep-alive",
    }

def discover_quarter_links():
    """Return list of (yyyy, q, url) found on the dataset page."""
    r = requests.get(DATASET_PAGE, headers=sec_headers(), timeout=30)
    r.raise_for_status()
    tree = html.fromstring(r.text)
    links = []
    for a in tree.xpath("//a[@href]"):
        href = a.get("href")
        if not href:
            continue
        # Normalize to absolute
        url = urljoin("https://www.sec.gov/", href)
        # Try to parse from label text (preferred)
        txt = "".join(a.itertext()).strip()
        m = LABEL_PAT.search(txt) or HREF_PAT.search(href)
        if m:
            yyyy, q = int(m.group(1)), int(m.group(2))
            if url.lower().endswith(".zip"):
                links.append((yyyy, q, url))
    # Deduplicate by (yyyy, q), prefer https links
    uniq = {}
    for yyyy, q, url in links:
        uniq[(yyyy, q)] = url
    # Sort newest first
    out = sorted(((y, q, u) for (y, q), u in uniq.items()), key=lambda t: (t[0], t[1]), reverse=True)
    return out

def download_zip(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, headers=sec_headers(), stream=True, timeout=120) as r:
        if r.status_code == 403:
            raise RuntimeError(
                f"403 Forbidden from SEC for {url}. "
                "Double-check SEC_USER_AGENT (must include a real email) and avoid parallel requests."
            )
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 15):
                if chunk:
                    f.write(chunk)

def unzip_file(zip_path, extract_to=None):
    """Extract a zip file to a directory"""
    if extract_to is None:
        # Extract to a directory with the same name as the zip file (without extension)
        extract_to = zip_path.parent / zip_path.stem
    
    extract_to.mkdir(parents=True, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    
    return extract_to

def main(years=3, sleep_secs=0.5, unzip=True):
    quarters_needed = years * 4
    discovered = discover_quarter_links()
    if not discovered:
        raise SystemExit("No ZIP links discovered on the dataset page. The page layout may have changed.")

    # Select last N quarters that actually exist
    to_get = discovered[:quarters_needed]

    print(f"Found {len(discovered)} posted quarters on the SEC page; checking existing data...")
    
    # Check what we already have and filter out existing data from download requests
    to_download = []
    skipped_count = 0
    
    for (yyyy, q, url) in to_get:
        fname = f"{yyyy}q{q}_mfrr.zip"
        out = OUTDIR / fname
        extract_dir = OUTDIR / f"{yyyy}q{q}_mfrr"
        
        # Check if already extracted with key files
        key_files = ['sub.tsv', 'num.tsv', 'tag.tsv']
        if extract_dir.exists() and all((extract_dir / f).exists() for f in key_files):
            print(f"[SKIP] {yyyy}Q{q} already extracted with all key files in {extract_dir}")
            skipped_count += 1
            continue
            
        # Check if zip exists but not properly extracted
        if out.exists() and out.stat().st_size > 0:
            print(f"[FOUND] {fname} exists, checking extraction...")
            if unzip:
                try:
                    extracted_to = unzip_file(out, extract_dir)
                    # Verify key files were extracted
                    if all((extract_dir / f).exists() for f in key_files):
                        print(f"[UNZIP] successfully extracted to {extracted_to}")
                        skipped_count += 1
                        continue
                    else:
                        print(f"[WARN] extraction incomplete, will re-download")
                        to_download.append((yyyy, q, url))
                except Exception as e:
                    print(f"[FAIL] unzip {fname}: {e}, will re-download")
                    to_download.append((yyyy, q, url))
            else:
                print(f"[SKIP] {fname} exists, unzip disabled")
                skipped_count += 1
            continue
            
        # Need to download this quarter
        to_download.append((yyyy, q, url))
    
    print(f"SEC RR Data Status: {skipped_count} quarters already available, {len(to_download)} quarters to download")
    
    # Download missing data
    for (yyyy, q, url) in to_download:
        fname = f"{yyyy}q{q}_mfrr.zip"
        out = OUTDIR / fname
        extract_dir = OUTDIR / f"{yyyy}q{q}_mfrr"
        
        print(f"[GET ] {fname}  ←  {url}")
        try:
            download_zip(url, out)
            print(f"[OK  ] wrote {out} ({out.stat().st_size:,} bytes)")
            
            # Automatically unzip after download
            if unzip:
                try:
                    extracted_to = unzip_file(out, extract_dir)
                    print(f"[UNZIP] extracted to {extracted_to}")
                except Exception as e:
                    print(f"[FAIL] unzip {fname}: {e}")
                    
        except Exception as e:
            print(f"[FAIL] {fname}: {e}")
        time.sleep(sleep_secs)  # be polite
    
    if len(to_download) == 0:
        print("[INFO] All requested SEC RR quarters are already available - no downloads needed")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, default=3, help="How many years (quarters*4) to fetch, default 3.")
    ap.add_argument("--sleep", type=float, default=0.5, help="Seconds to sleep between requests.")
    ap.add_argument("--no-unzip", action="store_true", help="Skip automatic unzipping of downloaded files.")
    args = ap.parse_args()
    main(years=args.years, sleep_secs=args.sleep, unzip=not args.no_unzip)
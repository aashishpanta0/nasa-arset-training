"""
Download NEX-GDDP-CMIP6 NetCDFs for GISS-E2-1-G (pr, tasmax, tasmin) with MD5 verification.

Quick start
-----------
# 1) Edit the USER PARAMETERS section below (e.g., OUTPUT_DIR = "./data")
# 2) Run:
#    python download_nex_gddp.py

Optional CLI overrides (take precedence over the user parameters)
-----------------------------------------------------------------
python grab_nex_gddp_giss.py \
  --out ./nex_data \
  --scenarios historical ssp245 \
  --members r1i1p1f1 \
  --vars pr tasmax tasmin \
  --workers 8
"""
from __future__ import annotations

# ================================ #
#  USER PARAMETERS - EDIT HERE     #
# ================================ #
OUTPUT_DIR = "./"  # Where files will be saved; directory structure is preserved under this root
#SCENARIOS  = ["historical", "ssp126", "ssp245", "ssp370", "ssp585"]  # subset as you like
SCENARIOS  = ["historical", "ssp585"]  # subset as you like
MEMBERS    = None  # e.g., ["r1i1p1f1"] or None for all available
VARS       = ["pr", "tasmax", "tasmin"]  # variables to download
WORKERS    = 6  # parallel downloads
MODEL      = "GISS-E2-1-G"  # keep as-is per your request; can be changed if needed
MD5_URL    = "https://nex-gddp-cmip6.s3-us-west-2.amazonaws.com/index_v2.0_md5.txt"
# ================================ #
#  END USER PARAMETERS             #
# ================================ #

import argparse
import concurrent.futures as cf
import hashlib
import os
import re
import sys
import time
from pathlib import Path
from typing import Iterable, List, Optional, Tuple
import urllib.parse

import requests

REQUEST_TIMEOUT = (10, 60)  # (connect, read) seconds
MAX_RETRIES = 3
RETRY_BACKOFF = 3.0  # seconds
S3_BASE = "https://nex-gddp-cmip6.s3-us-west-2.amazonaws.com"


def md5sum(path: Path, chunk: int = 1024 * 1024) -> str:
    m = hashlib.md5()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            m.update(block)
    return m.hexdigest()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def http_get(url: str, dest: Path) -> None:
    """Stream a file to disk with retries."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with requests.get(url, stream=True, timeout=REQUEST_TIMEOUT) as r:
                r.raise_for_status()
                ensure_parent(dest)
                tmp = dest.with_suffix(dest.suffix + ".part")
                with tmp.open("wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                tmp.replace(dest)
            return
        except Exception as e:
            if attempt == MAX_RETRIES:
                raise
            time.sleep(RETRY_BACKOFF * attempt)


def parse_md5_index(text: str) -> List[Tuple[str, str]]:
    """
    Parse lines like:
      <md5>  <relative_path>
    Returns list of (md5, relative_path)
    """
    pairs: List[Tuple[str, str]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = re.split(r"\s+", line, maxsplit=1)
        if len(parts) != 2:
            continue
        checksum, relpath = parts
        pairs.append((checksum, relpath))
    return pairs


def filter_entries(
    pairs: Iterable[Tuple[str, str]],
    model: str,
    variables: Iterable[str],
    scenarios: Optional[Iterable[str]],
    members: Optional[Iterable[str]],
    suffixes: Tuple[str, ...] = (".nc",),
) -> List[Tuple[str, str]]:
    variables = set(variables)
    scenarios = set(scenarios) if scenarios else None
    members = set(members) if members else None

    kept: List[Tuple[str, str]] = []
    for md5, rel in pairs:
        if not rel.endswith(suffixes):
            continue
        if f"/{model}/" not in rel:
            continue
        if not any(f"/{v}/" in rel for v in variables):
            continue
        if scenarios is not None and not any(f"/{s}/" in rel for s in scenarios):
            continue
        if members is not None and not any(f"/{m}/" in rel for m in members):
            continue
        kept.append((md5, rel))
    return kept


def build_url(relpath: str) -> str:
    return f"{S3_BASE}/{urllib.parse.quote(relpath)}"


def work_one(base_out: Path, md5_hex: str, relpath: str) -> Tuple[str, str, str]:
    url = build_url(relpath)
    dest = base_out / relpath  # preserve folder structure

    # Skip if exists and matches checksum
    if dest.is_file():
        try:
            current = md5sum(dest)
            if current.lower() == md5_hex.lower():
                return ("skip", relpath, "exists+md5_ok")
            else:
                dest.unlink(missing_ok=True)
        except Exception:
            pass

    try:
        http_get(url, dest)
        got = md5sum(dest)
        if got.lower() != md5_hex.lower():
            dest.unlink(missing_ok=True)
            return ("error", relpath, f"md5_mismatch (got {got}, want {md5_hex})")
        return ("ok", relpath, "downloaded")
    except Exception as e:
        return ("error", relpath, f"{type(e).__name__}: {e}")


def main(argv: Optional[List[str]] = None) -> int:
    # CLI with defaults coming from the USER PARAMETERS
    p = argparse.ArgumentParser(description="Download NEX-GDDP-CMIP6 files for GISS-E2-1-G with MD5 verification.")
    p.add_argument("--out", type=Path, default=Path(OUTPUT_DIR),
                   help=f"Output directory root (default from USER PARAMETERS: {OUTPUT_DIR}).")
    p.add_argument("--scenarios", nargs="*", default=SCENARIOS,
                   help=f"Scenarios to include (default from USER PARAMETERS: {' '.join(SCENARIOS)}).")
    p.add_argument("--members", nargs="*", default=MEMBERS,
                   help="Optional ensemble members (e.g., r1i1p1f1). If omitted, include all.")
    p.add_argument("--workers", type=int, default=WORKERS,
                   help=f"Parallel downloads (default from USER PARAMETERS: {WORKERS}).")
    p.add_argument("--vars", nargs="*", default=VARS,
                   help=f"Variables to include (default from USER PARAMETERS: {' '.join(VARS)}).")
    p.add_argument("--model", default=MODEL,
                   help=f"Model name filter (default from USER PARAMETERS: {MODEL}).")
    p.add_argument("--md5_url", default=MD5_URL,
                   help=f"URL of the md5 index (default from USER PARAMETERS).")
    args = p.parse_args(argv)

    print(f"Fetching MD5 index from: {args.md5_url}", flush=True)
    r = requests.get(args.md5_url, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    pairs = parse_md5_index(r.text)

    print("Filtering entries...", flush=True)
    entries = filter_entries(
        pairs,
        model=args.model,
        variables=args.vars,
        scenarios=args.scenarios,
        members=args.members,
        suffixes=(".nc",),
    )

    if not entries:
        print("No matching entries found. Check your filters (scenarios/members/vars/model).", file=sys.stderr)
        return 2

    print(f"Matched files: {len(entries)}")
    base_out = args.out
    tasks = [(base_out, md5_hex, rel) for (md5_hex, rel) in entries]

    ok = skip = err = 0
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(work_one, *t) for t in tasks]
        for f in cf.as_completed(futs):
            status, rel, msg = f.result()
            if status == "ok":
                ok += 1
                print(f"[OK]   {rel}  ({msg})")
            elif status == "skip":
                skip += 1
                print(f"[SKIP] {rel}  ({msg})")
            else:
                err += 1
                print(f"[ERR]  {rel}  ({msg})", file=sys.stderr)

    print(f"\nSummary: ok={ok} skip={skip} err={err}")
    return 0 if err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())


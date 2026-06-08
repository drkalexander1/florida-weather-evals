"""Download FAWN yearly QAQC zip archives into data/fawn_raw/."""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "fawn_raw"
BASE_URL = "https://fawn.ifas.ufl.edu/data/fawn_data_qaqc_pub/{year}.csv.zip"
DEFAULT_YEAR_MIN = 2005
DEFAULT_YEAR_MAX = 2024


def download_year(year: int, dest_dir: Path, *, force: bool = False) -> bool:
    dest = dest_dir / f"{year}.csv.zip"
    if dest.exists() and not force:
        print(f"skip {year} (already exists: {dest.name}, {dest.stat().st_size:,} bytes)")
        return True

    url = BASE_URL.format(year=year)
    print(f"download {year} from {url}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".zip.part")

    try:
        with urllib.request.urlopen(url, timeout=120) as response:
            total = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 1024 * 1024
            with tmp.open("wb") as out:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    out.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded * 100 // total
                        print(f"  {year}: {downloaded:,}/{total:,} bytes ({pct}%)", end="\r")
        print()
    except urllib.error.HTTPError as exc:
        print(f"  failed {year}: HTTP {exc.code}", file=sys.stderr)
        if tmp.exists():
            tmp.unlink()
        return False
    except urllib.error.URLError as exc:
        print(f"  failed {year}: {exc.reason}", file=sys.stderr)
        if tmp.exists():
            tmp.unlink()
        return False

    tmp.replace(dest)
    print(f"  saved {dest.name} ({dest.stat().st_size:,} bytes)")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download FAWN yearly QAQC zip files")
    parser.add_argument("--dest", type=Path, default=RAW_DIR)
    parser.add_argument("--year-min", type=int, default=DEFAULT_YEAR_MIN)
    parser.add_argument("--year-max", type=int, default=DEFAULT_YEAR_MAX)
    parser.add_argument("--force", action="store_true", help="Re-download even if file exists")
    args = parser.parse_args(argv)

    ok = 0
    failed: list[int] = []
    for year in range(args.year_min, args.year_max + 1):
        if download_year(year, args.dest, force=args.force):
            ok += 1
        else:
            failed.append(year)

    total = args.year_max - args.year_min + 1
    print(f"Done: {ok}/{total} years in {args.dest}")
    if failed:
        print(f"Failed years: {', '.join(map(str, failed))}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

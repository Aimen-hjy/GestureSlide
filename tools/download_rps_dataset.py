"""Download the lightweight Rock-Paper-Scissors image dataset.

The dataset is small enough for classroom experiments and maps naturally to:
  rock     -> FIST
  paper    -> OPEN_PALM
  scissors -> PEACE_UP

It is used as optional supplemental data, not as the main training source.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

URLS = {
    "rps.zip": "https://storage.googleapis.com/learning-datasets/rps.zip",
    "rps-test-set.zip": "https://storage.googleapis.com/learning-datasets/rps-test-set.zip",
}


def download(url: str, out_path: Path, force: bool = False) -> None:
    if out_path.exists() and out_path.stat().st_size > 0 and not force:
        print(f"Already exists: {out_path}")
        return
    print(f"Downloading {url}\n  -> {out_path}")
    with urllib.request.urlopen(url, timeout=60) as response, out_path.open("wb") as f:
        total = response.headers.get("Content-Length")
        total_size = int(total) if total and total.isdigit() else None
        downloaded = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if total_size:
                pct = downloaded / total_size * 100
                print(f"\r  {downloaded / 1024 / 1024:.1f} MiB / {total_size / 1024 / 1024:.1f} MiB ({pct:.1f}%)", end="")
        print()


def extract(zip_path: Path, out_dir: Path) -> None:
    marker = out_dir / f".{zip_path.stem}.extracted"
    if marker.exists():
        print(f"Already extracted: {zip_path.name}")
        return
    print(f"Extracting {zip_path.name} -> {out_dir}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)
    marker.write_text("ok\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download lightweight RPS supplemental dataset")
    parser.add_argument("--output-dir", default="datasets/rps")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for filename, url in URLS.items():
        zip_path = out_dir / filename
        try:
            download(url, zip_path, force=args.force)
            extract(zip_path, out_dir)
        except Exception as exc:
            print(f"Failed to download/extract {filename}: {exc}", file=sys.stderr)
            print("You can also manually download the dataset and then use tools/import_image_folder.py.", file=sys.stderr)
            return 1

    print("\nDone. Expected folders:")
    print(f"  {out_dir / 'rps'}")
    print(f"  {out_dir / 'rps-test-set'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

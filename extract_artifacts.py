"""Verify and extract the data archives shipped with the reproduction package."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path, PurePosixPath
from zipfile import ZipFile


PACKAGE_ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = PACKAGE_ROOT / "artifacts"
CHECKSUM_FILE = ARTIFACTS_DIR / "SHA256SUMS.txt"


def read_checksums() -> dict[str, str]:
    checksums: dict[str, str] = {}
    for line in CHECKSUM_FILE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, filename = line.split(maxsplit=1)
        checksums[filename.strip()] = digest.lower()
    return checksums


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_target(member_name: str) -> Path:
    member = PurePosixPath(member_name)
    if member.is_absolute() or ".." in member.parts:
        raise ValueError(f"Unsafe archive member: {member_name}")
    target = (PACKAGE_ROOT / Path(*member.parts)).resolve()
    if target != PACKAGE_ROOT and PACKAGE_ROOT not in target.parents:
        raise ValueError(f"Archive member escapes package root: {member_name}")
    return target


def extract_archive(path: Path, *, force: bool) -> tuple[int, int]:
    extracted = 0
    skipped = 0
    with ZipFile(path) as archive:
        for info in archive.infolist():
            target = safe_target(info.filename)
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if target.exists() and not force:
                skipped += 1
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, target.open("wb") as destination:
                while chunk := source.read(1024 * 1024):
                    destination.write(chunk)
            extracted += 1
    return extracted, skipped


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        action="append",
        metavar="ARCHIVE",
        help="Process only this archive filename; repeat for multiple archives.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify SHA-256 hashes without extracting files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite files that have already been extracted.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    checksums = read_checksums()
    selected = args.only or list(checksums)
    unknown = sorted(set(selected) - set(checksums))
    if unknown:
        raise SystemExit(f"Unknown archive(s): {', '.join(unknown)}")

    for filename in selected:
        path = ARTIFACTS_DIR / filename
        if not path.is_file():
            raise SystemExit(f"Missing archive: {path}")
        actual = sha256_file(path)
        expected = checksums[filename]
        if actual != expected:
            raise SystemExit(
                f"Checksum mismatch for {filename}: expected {expected}, got {actual}"
            )
        print(f"Verified {filename}")
        if not args.verify_only:
            extracted, skipped = extract_archive(path, force=args.force)
            print(f"Extracted {filename}: written={extracted}, skipped={skipped}")


if __name__ == "__main__":
    main()

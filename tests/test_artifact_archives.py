from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path, PurePosixPath
from zipfile import ZipFile


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PACKAGE_ROOT / "artifacts"

EXPECTED_PREFIXES = {
    "package_data.zip": "data/",
    "rq2_data.zip": "rqs/rq2/data/",
    "rq4_data.zip": "rqs/rq4/data/",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checksums() -> dict[str, str]:
    result: dict[str, str] = {}
    for line in (ARTIFACTS_DIR / "SHA256SUMS.txt").read_text(
        encoding="utf-8"
    ).splitlines():
        digest, filename = line.split(maxsplit=1)
        result[filename.strip()] = digest
    return result


def test_release_archives_have_expected_hashes_and_safe_paths() -> None:
    expected_hashes = checksums()
    assert set(expected_hashes) == set(EXPECTED_PREFIXES)

    for filename, prefix in EXPECTED_PREFIXES.items():
        path = ARTIFACTS_DIR / filename
        assert path.is_file(), filename
        assert path.stat().st_size < 100 * 1024 * 1024, filename
        assert sha256_file(path) == expected_hashes[filename]
        with ZipFile(path) as archive:
            names = archive.namelist()
            assert names, filename
            assert all(name.startswith(prefix) for name in names), filename
            for name in names:
                member = PurePosixPath(name)
                assert not member.is_absolute(), name
                assert ".." not in member.parts, name


def test_artifact_verification_cli() -> None:
    result = subprocess.run(
        [sys.executable, "extract_artifacts.py", "--verify-only"],
        cwd=PACKAGE_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    for filename in EXPECTED_PREFIXES:
        assert f"Verified {filename}" in result.stdout


def test_extracted_artifact_directories_are_ignored() -> None:
    text = (PACKAGE_ROOT / ".gitignore").read_text(encoding="utf-8")
    expected = (
        "/data/",
        "/rqs/rq2/data/",
        "/rqs/rq4/data/",
    )
    for pattern in expected:
        assert pattern in text

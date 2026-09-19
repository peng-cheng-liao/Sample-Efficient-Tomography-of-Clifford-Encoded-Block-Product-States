#!/usr/bin/env python3
"""Create the task-only patch and verification ZIP for this investigation."""

from __future__ import annotations

import difflib
import hashlib
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[2]
TASK_ROOT = ROOT / "tests" / "fixed_error_d1_bimodality"
REPORT = ROOT / "Reports" / "fixed_error_d1_bimodality_investigation_report.md"
DIFF = ROOT / "Reports" / "fixed_error_d1_bimodality_investigation_task.diff"
ARCHIVE = ROOT / "Reports" / "fixed_error_d1_bimodality_investigation_verification.zip"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def task_files() -> list[Path]:
    files = [
        path
        for path in TASK_ROOT.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
    ]
    files.append(REPORT)
    return sorted(files, key=lambda path: str(path.relative_to(ROOT)))


def is_text(path: Path) -> bool:
    return path.suffix.lower() not in {".png", ".zip"}


def write_task_diff(files: list[Path]) -> None:
    chunks = [
        "# Task-only patch for fixed_error_d1_bimodality investigation\n",
        "# Every listed path was absent at task start. Binary plots are identified by SHA256.\n",
    ]
    for path in files:
        relative = str(path.relative_to(ROOT))
        if is_text(path):
            lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
            chunks.extend(
                difflib.unified_diff(
                    [],
                    lines,
                    fromfile="/dev/null",
                    tofile=f"b/{relative}",
                    lineterm="\n",
                )
            )
        else:
            chunks.append(f"Binary file /dev/null and b/{relative} differ\n")
            chunks.append(f"SHA256 {sha256(path)}  {relative}\n")
    DIFF.write_text("".join(chunks), encoding="utf-8")


def write_archive(files: list[Path]) -> None:
    with zipfile.ZipFile(ARCHIVE, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in [*files, DIFF]:
            archive.write(path, arcname=str(path.relative_to(ROOT)))


def main() -> int:
    files = task_files()
    if not REPORT.exists():
        raise FileNotFoundError(REPORT)
    write_task_diff(files)
    write_archive(files)
    print(f"task diff: {DIFF.relative_to(ROOT)} ({DIFF.stat().st_size} bytes)")
    print(f"verification ZIP: {ARCHIVE.relative_to(ROOT)} ({ARCHIVE.stat().st_size} bytes)")
    print(f"ZIP payload files: {len(files) + 1}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

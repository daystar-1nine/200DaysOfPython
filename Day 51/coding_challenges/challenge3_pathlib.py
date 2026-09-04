"""
===============================================================================
DAY 51 — CODING CHALLENGE 3: PATHLIB FILESYSTEM OPERATIONS
===============================================================================
This module creates a directory structure, writes text files using pathlib.Path,
and lists all matching .txt files.
===============================================================================
"""

from pathlib import Path
from typing import List


def create_and_list_reports(base_dir: Path) -> List[str]:
    """Create directory structure, write 3 files, and list .txt filenames."""
    # What is used: Path.mkdir with parents=True and exist_ok=True.
    # Why it is used: Ensures folder exists without raising FileExistsError.
    # How it works: Recursively creates parent directories if needed.
    reports_dir = base_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # What is used: Path.write_text file writing.
    # Why it is used: Writes text content to file cleanly in a single method call.
    # How it works: Opens file, writes string, and closes descriptor.
    for i in range(1, 4):
        file_path = reports_dir / f"report_{i}.txt"
        file_path.write_text(f"Report content {i}", encoding="utf-8")

    # What is used: Path.glob pattern matching.
    # Why it is used: Iterates over all matching .txt files in directory.
    # How it works: Returns generator yielding Path objects matching *.txt.
    txt_files = sorted([p.name for p in reports_dir.glob("*.txt")])
    return txt_files


if __name__ == "__main__":
    test_dir = Path("Day 51/scratch_test_pathlib")
    files = create_and_list_reports(test_dir)
    print("Created Text Files:", files)
    assert len(files) == 3
    assert files == ["report_1.txt", "report_2.txt", "report_3.txt"]

    # Cleanup temporary test directory
    for f in (test_dir / "reports").glob("*.txt"):
        f.unlink()
    (test_dir / "reports").rmdir()
    if test_dir.exists():
        test_dir.rmdir()
    print("[OK] Challenge 3 Passed!")

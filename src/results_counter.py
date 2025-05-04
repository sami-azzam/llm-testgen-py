#!/usr/bin/env python3
"""
Count files that end with either “GptTest.java” (Kept tests)
or “GptTest.java.disabled” (Disabled tests).

For every *run* directory you list in RUN_DIRS it:

1. Looks only at first-level sub-directories whose names  
   • start with a project name in `PROJECTS`, **and**  
   • end with “_gpt”.  
   (e.g.  “Collections_base_gpt”, “Math_something_gpt”, …)

2. Recursively counts the matching files inside each of those folders.

Output is grouped by project and by run directory:

```

\=== runs\_20250503\_122824 ===
Collections
Kept: 12
Disabled: 35

Gson
Kept: 33
Disabled: 4
...

```
"""

from pathlib import Path
from typing import Tuple, Dict, List

# ---------- configuration ----------------------------------------------------
PROJECTS = [
    "Chart", "Cli", "Closure", "Codec", "Collections", "Compress", "Csv", "Gson",
    "JacksonCore", "JacksonDatabind", "JacksonXml", "Jsoup", "JxPath", "Lang",
    "Math", "Mockito", "Time",
]

LATEST_RESULTS = {
    "gpt-4o-mini": "runs_20250503_122824",
    "o4-mini":     "runs_20250503_154817",
}

ROOT        = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"

RUN_DIRS = (
    RESULTS_DIR / LATEST_RESULTS["gpt-4o-mini"],
    RESULTS_DIR / LATEST_RESULTS["o4-mini"],
)
# -----------------------------------------------------------------------------

def count_test_files(folder: Path) -> Tuple[int, int]:
    """Return (kept, disabled) counts inside *folder* (recursive)."""
    kept = disabled = 0
    for p in folder.rglob("*"):
        if not p.is_file():
            continue
        name = p.name
        if name.endswith("Test.java.disabled"):
            disabled += 1
        elif name.endswith("Test.java"):
            kept += 1
    return kept, disabled


def find_project_gpt_dirs(run_dir: Path, project: str) -> List[Path]:
    """Return first-level sub-dirs that start with *project* and end with '_gpt'."""
    return [
        d for d in run_dir.iterdir()
        if d.is_dir() and d.name.startswith(project) and d.name.endswith("_gpt")
    ]


def count_per_project(run_dir: Path) -> Dict[str, Tuple[int, int]]:
    """Count kept/disabled tests for every present project in *run_dir*."""
    results: Dict[str, Tuple[int, int]] = {}
    for project in PROJECTS:
        gpt_dirs = find_project_gpt_dirs(run_dir, project)
        if not gpt_dirs:
            continue  # project not present in this run
        kept = disabled = 0
        for d in gpt_dirs:
            k, dis = count_test_files(d)
            kept += k
            disabled += dis
        results[project] = (kept, disabled)
    return results


def main() -> None:
    for run_dir in RUN_DIRS:
        print(f"\n=== {run_dir.name} ===")
        if not run_dir.exists():
            print(f"[!] Directory not found: {run_dir}")
            continue

        project_counts = count_per_project(run_dir)
        if not project_counts:
            print("No *_gpt project folders found.")
            continue

        for project, (kept, disabled) in sorted(project_counts.items()):
            print(f"{project}")
            print(f"    Kept:     {kept}")
            print(f"    Disabled: {disabled}\n")


if __name__ == "__main__":
    main()

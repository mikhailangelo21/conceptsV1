"""Build an immutable-by-default, portable V3 research handoff folder.

Run from anywhere: python analysis/build_v3_handoff.py
The copied files form an evidence snapshot, not a new experiment run.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
DEST = PROJECT / "handoff" / "quality-suite-v3"
RUNS = {
    "quality_suite_v3_laptop-88fad55b46b0": "primary V3 result",
    "quality_suite_v3_smoke-55cb21caf007": "integration smoke check",
    "pilot_m2-792030855aa5": "original size-only pilot comparison",
}
SKIP_NAMES = {".DS_Store", "__pycache__", "behavior_shards", "static_embeddings.npz"}


def selected() -> list[tuple[Path, Path, str]]:
    rows: list[tuple[Path, Path, str]] = []

    def add(source: Path, relative: Path, role: str) -> None:
        if not source.is_file():
            raise FileNotFoundError(source)
        rows.append((source, relative, role))

    for name in ("pyproject.toml", "requirements-lock.txt", "populate_quality_suite.py"):
        add(PROJECT / name, Path(name), "project setup")
    add(PROJECT / "README.md", Path("source_material/project_README.md"), "original project overview")
    for path in sorted((PROJECT / "configs").iterdir()):
        if path.is_file() and path.suffix in {".yaml", ".json"}:
            add(path, path.relative_to(PROJECT), "V3 and legacy test configuration")
    for path in sorted((PROJECT / "docs").rglob("*")):
        if path.is_file() and path.suffix in {".md", ".csv", ".json"}:
            add(path, path.relative_to(PROJECT), "design and implementation documentation")
    for name in ("codex-llm-quality-dimensions.md", "codex-pilot-refinement-v2.md"):
        add(WORKSPACE / name, Path("source_material") / name, "historical user-supplied specification")
    for path in sorted((PROJECT / "data" / "quality_suite_v3").rglob("*")):
        if path.is_file():
            add(path, path.relative_to(PROJECT), "complete versioned fixture bank")
    for path in sorted((PROJECT / "data").glob("*.csv")):
        add(path, path.relative_to(PROJECT), "original pilot inputs and offline test fixture")
    for path in sorted((PROJECT / "src" / "quality_dimensions").glob("*.py")):
        add(path, path.relative_to(PROJECT), "analysis and inference implementation")
    for path in sorted((PROJECT / "tests").glob("*.py")):
        add(path, path.relative_to(PROJECT), "software verification")
    for name in ("v3_findings.py", "v3_report.py", "build_v3_handoff.py"):
        add(PROJECT / "analysis" / name, Path("analysis") / name, "reproduction and handoff script")
    for path in sorted((PROJECT / "reports" / "quality_suite_v3_findings").glob("*")):
        if path.is_file() and path.name not in SKIP_NAMES:
            add(path, path.relative_to(PROJECT), "interpreted findings and figures")

    for run_name, role in RUNS.items():
        run = PROJECT / "runs" / run_name
        if not run.is_dir():
            raise FileNotFoundError(run)
        for path in sorted(run.iterdir()):
            if path.name in SKIP_NAMES:
                continue
            if path.is_file() and path.suffix in {".csv", ".json", ".md"}:
                add(path, Path("runs") / run_name / path.name, role)
        if run_name.startswith("quality_suite_v3_laptop"):
            nested = (("fits", {".npz"}), ("plots", {".png", ".svg"}))
        else:
            nested = (("plots", {".png", ".svg"}),)
        for subdir, suffixes in nested:
            if (run / subdir).is_dir():
                for path in sorted((run / subdir).rglob("*")):
                    if path.is_file() and path.suffix in suffixes:
                        add(path, Path("runs") / run_name / path.relative_to(run), role)
    return rows


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    rows = selected()
    previous_path = DEST / "FILE_MANIFEST.json"
    previous = ({item["path"]: item["sha256"] for item in json.loads(previous_path.read_text())["files"]}
                if previous_path.exists() else {})
    manifest = []
    for source, relative, role in rows:
        target = DEST / relative
        source_hash = sha256(source)
        if target.exists():
            if sha256(target) != source_hash:
                if sha256(target) != previous.get(relative.as_posix()):
                    raise FileExistsError(f"Existing handoff file was edited outside the builder: {target}")
                shutil.copy2(source, target)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        manifest.append({
            "path": relative.as_posix(),
            "role": role,
            "bytes": target.stat().st_size,
            "sha256": source_hash,
            "source": str(source),
        })
    payload = {
        "schema": "quality-suite-v3-handoff-1",
        "description": "Copied research evidence; paths are relative to this folder and SHA-256 values verify content.",
        "file_count": len(manifest),
        "total_bytes": sum(item["bytes"] for item in manifest),
        "exclusions": [
            "Pretrained Qwen model weights and Hugging Face download cache",
            "External activation shard cache; indexes and identity records are included",
            "Per-question behavior_shards because behavior_scores.csv retains the scored rows",
            "static_embeddings.npz because it is a 36 MiB feature cache, not a result table",
            "Virtual environment and operating-system files",
        ],
        "files": manifest,
        "handoff_guides": [
            {"path": name, "bytes": (DEST / name).stat().st_size,
             "sha256": sha256(DEST / name)}
            for name in ("README.md", "START_HERE.md", "METHODS.md", "RESULTS.md", "REPRODUCE.md", "VALIDATION.md")
        ],
    }
    (DEST / "FILE_MANIFEST.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Handoff: {DEST}")
    print(f"Verified files: {len(manifest)}; copied bytes: {payload['total_bytes']:,}")


if __name__ == "__main__":
    main()

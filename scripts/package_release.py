"""Create deterministic, text-free release archives from preserved run files."""
from __future__ import annotations

import gzip
import hashlib
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def archive(path: Path, members: list[Path]) -> str:
    with path.open("wb") as file, gzip.GzipFile(filename="", mode="wb", fileobj=file, mtime=0, compresslevel=9) as zipped:
        with tarfile.open(fileobj=zipped, mode="w") as tar:
            for source in sorted(members):
                info = tar.gettarinfo(str(source), arcname=str(source.relative_to(ROOT)))
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                info.mtime = 0
                with source.open("rb") as stream:
                    tar.addfile(info, stream)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    stats = json.loads((RESULTS / "statistics_with_upper.json").read_text())
    raw = [ROOT / detail["path"] for source in stats["sources"].values() for detail in source["raw"].values()]
    if len(raw) != 35 or not all(path.exists() for path in raw):
        raise ValueError("expected 35 complete model/dataset prediction files")
    completed_run = RESULTS / "runs" / "20260923T005308Z_upper-qwen38-27b-v1"
    history = list(completed_run.rglob("*")) + list(RESULTS.glob("summary*.json"))
    history = [path for path in history if path.is_file()]
    history.append(RESULTS / "statistics_baseline.json")
    hashes = {
        "raw-complete.tar.gz": archive(RESULTS / "raw-complete.tar.gz", raw),
        "history.tar.gz": archive(RESULTS / "history.tar.gz", history),
    }
    (RESULTS / "ARCHIVES.sha256").write_text("".join(f"{digest}  results/{name}\n" for name, digest in hashes.items()))
    print("Wrote deterministic result archives and results/ARCHIVES.sha256")


if __name__ == "__main__":
    main()

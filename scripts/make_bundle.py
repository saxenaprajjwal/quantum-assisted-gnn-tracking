"""Build the lxplus deployment bundle of HSF Part II (run on the PC, from the Part-II folder):

    python scripts/make_bundle.py            -> dist/hsf2_bundle_<DD-MM-YYYY-HH-MM>.tgz

Everything goes under hsf2* names, so it can be unpacked INSIDE the Phase-1 Optimization folder without touching any
Phase-1 file (installation: hsf2_scripts/deploy_lxplus.sh):

    hsf2/          the package (no __pycache__)
    hsf2_condor/   condor/run.sh, *.sub, *.dag
    hsf2_scripts/  scripts/*.sh (env, setup, download, hardware, deploy, discovery) and scripts/*.py (odd_export.py)
    hsf2_tests/    tests/test_<x>.py renamed test_hsf2_<x>.py (no clash with Phase-1 tests/test_core.py)
    hsf2_docs/     guide PDF, README, requirements, DEPLOYED_REV (code revision + source hash), MANIFEST (sha1 per file)

Not included: data, models, runs, figures, LaTeX sources, literature, .claude, git files.  Text files get LF line
endings and shell scripts the executable bit.
"""
from __future__ import annotations

import hashlib
import io
import subprocess
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEXT = {".py", ".sh", ".sub", ".dag", ".txt", ".md", ".json"}
# pytest run from anywhere on lxplus: rootdir = hsf2_tests, no .pytest_cache written next to the Phase-1 code
PYTEST_INI = b"[pytest]\naddopts = -p no:cacheprovider\n"


def files():
    out = []
    for f in sorted((ROOT / "hsf2").rglob("*")):
        if f.is_file() and "__pycache__" not in f.parts and f.suffix != ".pyc":
            out.append((f, "hsf2/" + f.relative_to(ROOT / "hsf2").as_posix()))
    for f in sorted((ROOT / "condor").iterdir()):
        if f.suffix in (".sub", ".dag", ".sh"):
            out.append((f, "hsf2_condor/" + f.name))
    for f in sorted((ROOT / "scripts").glob("*.sh")):
        out.append((f, "hsf2_scripts/" + f.name))
    for f in sorted((ROOT / "scripts").glob("*.py")):       # odd_export.py runs in the ACTS environment on lxplus
        if f.name != "make_bundle.py":
            out.append((f, "hsf2_scripts/" + f.name))
    for f in sorted((ROOT / "tests").glob("test_*.py")):
        out.append((f, "hsf2_tests/test_hsf2_" + f.name[len("test_"):]))
    for src, dst in (("docs/HSF_PartII_Guide.pdf", "hsf2_docs/HSF_PartII_Guide.pdf"), ("README.md", "hsf2_docs/README.md"),
                     ("requirements.txt", "hsf2_docs/requirements_hsf2.txt"), ("docs/PROGRESS.md", "hsf2_docs/PROGRESS.md")):
        if (ROOT / src).exists():
            out.append((ROOT / src, dst))
    return out


def content(path: Path) -> bytes:
    data = path.read_bytes()
    return data.replace(b"\r\n", b"\n") if path.suffix in TEXT else data


def main() -> int:
    sys.path.insert(0, str(ROOT))
    from hsf2.utils import source_hash
    try:
        rev = subprocess.run(["git", "-C", str(ROOT), "describe", "--always", "--dirty"], capture_output=True,
                             text=True, timeout=10).stdout.strip() or "nogit"
    except Exception:
        rev = "nogit"
    stamp = time.strftime("%d-%m-%Y-%H-%M")
    items = files()
    manifest = [f"{hashlib.sha1(content(src)).hexdigest()}  {dst}" for src, dst in items]
    manifest.append(f"{hashlib.sha1(PYTEST_INI).hexdigest()}  hsf2_tests/pytest.ini")
    deployed = f"{rev}+src{source_hash()}\nbuilt {stamp} from {ROOT}\n"
    out = ROOT / "dist" / f"hsf2_bundle_{stamp}.tgz"
    out.parent.mkdir(exist_ok=True)
    with tarfile.open(out, "w:gz") as tar:
        def add(name: str, data: bytes, mode: int) -> None:
            info = tarfile.TarInfo(name)
            info.size, info.mode, info.mtime = len(data), mode, int(time.time())
            tar.addfile(info, io.BytesIO(data))
        for src, dst in items:
            add(dst, content(src), 0o755 if src.suffix == ".sh" else 0o644)
        add("hsf2_docs/DEPLOYED_REV", deployed.encode(), 0o644)
        add("hsf2_tests/pytest.ini", PYTEST_INI, 0o644)
        add("hsf2_docs/MANIFEST.txt", ("\n".join(manifest) + "\n").encode(), 0o644)
    print(f"{out}  ({len(items)} files, {out.stat().st_size / 1e6:.2f} MB, code {deployed.splitlines()[0]})")
    print("copy to lxplus (PowerShell):")
    print(f'  scp "{out}" <cern-user>@lxplus.cern.ch:~/')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

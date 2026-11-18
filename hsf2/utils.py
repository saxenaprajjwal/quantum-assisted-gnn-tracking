"""Naming, crash-safe writing and provenance helpers (same conventions as Phase-1 acts_qaoa.io_utils).

Rules (standing project rules):
  * every time stamp in a file or folder name is DD-MM-YYYY-HH-MM (local time);
  * a result file is never silently overwritten: an older, different version moves to history/;
  * every command writes its files to a new folder <run>/outputs/<DD-MM-YYYY-HH-MM>_<command>/,
    together with the configuration that produced them.
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, List, Optional

import numpy as np

STAMP_FORMAT = "%d-%m-%Y-%H-%M"
STAMPED_NAME = re.compile(r"^\d\d-\d\d-\d{4}-\d\d-\d\d(_.*)?$")


def stamp(t: Optional[float] = None) -> str:
    """Time stamp used in all file and folder names, e.g. 19-10-2026-14-05."""
    return time.strftime(STAMP_FORMAT, time.localtime(t))


def clean_label(label: Optional[str]) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(label or "")).strip("._-")


def ensure_dir(path: Path, wait_s: float = 60.0) -> Path:
    """mkdir -p that survives concurrent creation on EOS.

    When several batch nodes create the same folder at once, EOS answers EEXIST to all but one while their FUSE
    caches may still report the folder as missing, so pathlib's exist_ok check raises FileExistsError (stage A,
    eval_c1, 30-10-2026). Retry with back-off until the folder is visible.
    """
    path = Path(path)
    delay, t0 = 0.05, time.monotonic()
    while True:
        try:
            path.mkdir(parents=True, exist_ok=True)
            return path
        except (FileExistsError, FileNotFoundError):
            if os.path.isdir(path):
                return path
            if time.monotonic() - t0 > wait_s:
                raise
        time.sleep(delay)
        delay = min(2 * delay, 2.0)


def unique_dir(parent: Path, name: str) -> Path:
    """Create parent/name atomically; if it exists, parent/name_2, name_3, ... (a folder is never reused)."""
    parent = ensure_dir(parent)
    for k in range(1, 10000):
        cand = parent / (name if k == 1 else f"{name}_{k}")
        try:
            cand.mkdir()
            return cand
        except FileExistsError:
            continue
    raise RuntimeError(f"could not create a new folder {name}_* in {parent}")


def unique_path(path: Path) -> Path:
    """path itself if it does not exist yet, else path with _2, _3, ... before the extension."""
    path = Path(path)
    k, cand = 2, path
    while cand.exists():
        cand = path.with_name(f"{path.stem}_{k}{path.suffix}")
        k += 1
    return cand


def new_run_dir(root: Path, label: Optional[str] = None) -> Path:
    """root/<DD-MM-YYYY-HH-MM>[_label], created atomically (same minute -> _2, _3, ...)."""
    name = stamp()
    lab = clean_label(label)
    if lab:
        name += "_" + lab
    return unique_dir(root, name)


def command_dir(run_dir: Path, command: str) -> Path:
    """<run>/outputs/<DD-MM-YYYY-HH-MM>_<command>/ for one invocation of a command."""
    return new_run_dir(Path(run_dir) / "outputs", command)


def latest_dir(root: Path, marker: str) -> Optional[Path]:
    """Most recently created sub-folder of root that contains the file `marker` (written once, at creation)."""
    cands = [p.parent for p in Path(root).glob(f"*/{marker}")]
    if not cands:
        return None
    return max(cands, key=lambda d: (d / marker).stat().st_mtime)


def resolve_dir(arg: str, root: Path, marker: str) -> Path:
    """A folder given on the command line; 'latest' = newest folder in root that contains `marker`."""
    if str(arg) == "latest":
        d = latest_dir(root, marker)
        if d is None:
            raise FileNotFoundError(f"no folder with {marker} in {root}")
        return d
    d = Path(arg)
    if not (d / marker).exists():
        raise FileNotFoundError(f"{d} does not contain {marker}")
    return d


# ---------------------------------------------------------------------------
# JSON / text output
# ---------------------------------------------------------------------------
def _default(o: Any):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    if hasattr(o, "to_dict"):
        return o.to_dict()
    raise TypeError(f"not JSON serialisable: {type(o)}")


def to_json(obj: Any) -> str:
    return json.dumps(obj, indent=2, default=_default, allow_nan=True)


def keep_previous(path: Path, new_text: Optional[str] = None) -> Optional[Path]:
    """Before a result file is rewritten, move the old version to <dir>/history/<stem>__<stamp><ext>.

    Nothing is moved if the file does not exist or already contains exactly new_text.
    """
    path = Path(path)
    if not path.exists():
        return None
    if new_text is not None:
        try:
            if path.read_text() == new_text:
                return None
        except Exception:
            pass
    hist = path.parent / "history"
    hist.mkdir(exist_ok=True)
    dest = unique_path(hist / f"{path.stem}__{stamp(path.stat().st_mtime)}{path.suffix}")
    os.replace(path, dest)
    return dest


def write_text_atomic(path: Path, text: str) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", newline="\n") as f:
        f.write(text)
    os.replace(tmp, path)


def save_json(path: Path, obj: Any, keep_history: bool = True) -> Path:
    text = to_json(obj)
    if keep_history:
        keep_previous(path, text)
    write_text_atomic(path, text)
    return Path(path)


def save_text(path: Path, text: str, keep_history: bool = True) -> Path:
    if keep_history:
        keep_previous(path, text)
    write_text_atomic(path, text)
    return Path(path)


def save_csv(df, path: Path, keep_history: bool = True) -> Path:
    try:
        text = df.to_csv(index=False, lineterminator="\n")
    except TypeError:                       # pandas < 1.5
        text = df.to_csv(index=False, line_terminator="\n")
    return save_text(path, text, keep_history)


def load_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


def append_jsonl(path: Path, obj: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "a", newline="\n") as f:
        f.write(json.dumps(obj, default=_default, allow_nan=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path: Path) -> List[Any]:
    path = Path(path)
    if not path.exists():
        return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue                 # tolerate a truncated last line after a crash
    return out


# ---------------------------------------------------------------------------
# Provenance and logging
# ---------------------------------------------------------------------------
_SOURCE_HASH: Optional[str] = None


def source_hash() -> str:
    """SHA-1 (16 hex) over every hsf2/**/*.py file (path + content): identifies the code wherever it was copied."""
    global _SOURCE_HASH
    if _SOURCE_HASH is None:
        import hashlib
        pkg = Path(__file__).resolve().parent
        h = hashlib.sha1()
        for f in sorted(p for p in pkg.rglob("*.py") if "__pycache__" not in p.parts):
            h.update(f.relative_to(pkg).as_posix().encode())
            h.update(b"\0")
            h.update(f.read_bytes())
        _SOURCE_HASH = h.hexdigest()[:16]
    return _SOURCE_HASH


def _git(root: Path, *args: str) -> Optional[str]:
    try:
        r = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, timeout=10)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


def code_version() -> dict:
    """Provenance of the code behind a result: package version, source hash of hsf2/, the git commit only if hsf2/
    is tracked by the enclosing repository (on lxplus hsf2/ sits untracked inside the ACTS checkout, whose commit
    says nothing about Part II), a dirty flag, $HSF2_CODE_REV (set at deployment) and the versions that change
    results."""
    from . import __version__
    out = {"hsf2": __version__, "source_sha1": source_hash(), "python": sys.version.split()[0]}
    root = Path(__file__).resolve().parents[1]
    top = _git(root, "rev-parse", "--show-toplevel")
    if top and Path(top).resolve() == root and _git(root, "ls-files", "hsf2/__init__.py"):
        out["git"] = _git(root, "rev-parse", "--short", "HEAD")
        out["git_dirty"] = bool(_git(root, "status", "--porcelain", "--", "hsf2"))
    if os.environ.get("HSF2_CODE_REV"):
        out["deployed_rev"] = os.environ["HSF2_CODE_REV"]
    for mod in ("numpy", "scipy", "pandas", "torch", "qiskit"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            pass
    for mod in ("qiskit_aer", "qiskit_ibm_runtime", "optuna", "acts_qaoa"):      # recorded when the run used them
        if mod in sys.modules:
            out[mod] = getattr(sys.modules[mod], "__version__", None)
    return out


def setup_logging(level: str = "INFO", logfile: Optional[Path] = None) -> logging.Logger:
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    for h in list(root.handlers):
        root.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%d-%m-%Y %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(sh)
    if logfile is not None:
        Path(logfile).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(logfile)
        fh.setFormatter(fmt)
        root.addHandler(fh)
    for noisy in ("qiskit", "qiskit_ibm_runtime", "qiskit_aer", "stevedore", "urllib3", "matplotlib",
                  "optuna", "requests", "numexpr", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return root


class Timer:
    """with Timer() as t: ...;  t.s = elapsed seconds (wall clock)."""

    def __enter__(self):
        self.t0 = time.perf_counter()
        self.s = 0.0
        return self

    def __exit__(self, *exc):
        self.s = time.perf_counter() - self.t0
        return False

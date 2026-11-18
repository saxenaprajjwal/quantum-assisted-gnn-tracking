#!/usr/bin/env python3
"""HSF Part II: export the ACTS CKF chain of Phase 1 on the OpenDataDetector together with its truth (simhits,
measurements with geometry ids, particles, the final CKF tracks), so that the GNN can be run on exactly the same events.

Runs in the ACTS environment (the shell in which `python -c "import acts"` works), NOT in the Part-II venv.  Phase 1's
ckf_odd.py is imported read-only and its run_ckf() is used unchanged: same Fatras, digitisation, seeding and CKF
configuration, same RandomNumbers(seed), same particles_seed<seed>.root.  Three things are added from outside, by
wrapping two functions for the duration of the run: the Sequencer gets `skip` (--skip); addCKFTracks() is called with
writeTrackStates=True (Phase 1 leaves it False; it only adds the writer of trackstates_ckf.root, the measurement lists
of the CKF tracks); and the CSV writers are added to the Sequencer right after addCKFTracks().  The rest of the CKF's
ROOT output (performance_finding_ckf.root, tracksummary_ckf.root, timing.csv) is written by ckf_odd as in Phase 1.

  python odd_export.py --check [--with-geometry]     what this ACTS build offers (writer classes + config fields, the
                                                      whiteboard collection names found in ACTS's Python sources, ODD,
                                                      uproot); --with-geometry builds the ODD (~1 min) and compares the
                                                      geometry-id bit layout with ACTS's GeometryIdentifier. Exits.
  python odd_export.py --events 5 --seed 1000000 --label test     the export; output only in hsf2_work/odd_export/
                                                      <DD-MM-YYYY-HH-MM>_<label>/  (never overwritten, never Phase 1)
  python odd_export.py --post <export folder>        redo only the post-processing (performance json, CKF tracks from
                                                      trackstates_ckf.root) of an export whose ACTS run has finished

Outputs: see hsf2/data/odd.py (layout of an export folder).  A writer class or config field that this ACTS version
names differently is reported with the list of what exists, and nothing is written for it.
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent                 # <Optimization>/hsf2_scripts (lxplus) or <Part II>/scripts (PC)


def _setup_paths() -> None:
    for d in (HERE.parent, HERE.parent.parent):
        if (d / "hsf2" / "__init__.py").is_file():
            if str(d) not in sys.path:
                sys.path.insert(0, str(d))
            return
    sys.exit(f"hsf2 package not found next to {HERE}")


_setup_paths()

try:
    import acts
except ImportError:                                    # --help / --post work outside the ACTS environment
    acts = None

from hsf2 import _phase1                                # noqa: E402
from hsf2.data import odd                               # noqa: E402

WRITER_SPECS = {                                        # class name -> (config fields that must exist, optional ones)
    "CsvSimHitWriter": (("inputSimHits", "outputDir"), ("outputStem", "outputPrecision")),
    "CsvMeasurementWriter": (("inputMeasurements", "inputMeasurementSimHitsMap", "outputDir"), ("outputPrecision",)),
    "CsvParticleWriter": (("inputParticles", "outputDir"), ("outputStem", "outputPrecision")),
}
WRITER_MODULES = ("acts.examples", "acts.examples.csv", "acts.examples.root")
SOURCE_MODULES = ("acts.examples.simulation", "acts.examples.reconstruction")
# whiteboard names tried, in this order, after the ones found in ACTS's Python sources
FALLBACK_INITIAL = ("particles_initial", "particles_simulated_initial", "particles_simulated")
FALLBACK_SELECTED = ("particles_selected",)
FALLBACK_MAP = ("measurement_simhits_map", "measurement_simhit_map")


class ExportError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# ACTS introspection
# ---------------------------------------------------------------------------
def acts_version() -> str:
    for get in (lambda: acts.version.version, lambda: acts.__version__, lambda: acts.version):
        try:
            v = get()
            if isinstance(v, str):
                return v
        except Exception:
            continue
    return "?"


def _modules():
    out = []
    for name in WRITER_MODULES:
        try:
            out.append((name, importlib.import_module(name)))
        except Exception:
            pass
    return out


def available_writers() -> dict:
    return {name: sorted(n for n in dir(mod) if "Writer" in n) for name, mod in _modules()}


def config_fields(cls):
    """Names of the config fields of an ACTS python class (None if the bindings do not expose a Config type)."""
    cfg = getattr(cls, "Config", None)
    if cfg is None:
        return None
    return sorted(n for n in dir(cfg) if not n.startswith("_"))


def find_writer(name: str):
    for modname, mod in _modules():
        if hasattr(mod, name):
            return getattr(mod, name)
    raise ExportError(f"writer class {name} not found; available writers: {json.dumps(available_writers(), indent=1)}")


def make_writer(name: str, level, **kw):
    cls = find_writer(name)
    need, opt = WRITER_SPECS[name]
    fields = config_fields(cls)
    if fields is not None:
        missing = [k for k in need if k not in fields]
        if missing:
            raise ExportError(f"{name} has no config field(s) {missing}; its fields: {fields}")
        kw = {k: v for k, v in kw.items() if k in need or k in fields}
    return cls(level=level, **kw)


def source_literals() -> dict:
    """Whiteboard collection names found as string literals in ACTS's own python sources (simulation.py,
    reconstruction.py): particle collections and measurement maps."""
    found = {"particles": set(), "maps": set(), "tracks": set()}
    for name in SOURCE_MODULES:
        try:
            src = inspect.getsource(importlib.import_module(name))
        except Exception:
            continue
        for lit in set(re.findall(r'"([A-Za-z0-9_]+)"', src)):
            if lit.startswith("particles_") or lit == "particles":
                found["particles"].add(lit)
            elif lit.startswith("measurement") and "map" in lit:
                found["maps"].add(lit)
            elif "tracks" in lit and "_" in lit or lit == "tracks":
                found["tracks"].add(lit)
    return {k: sorted(v) for k, v in found.items()}


def collection_profiles(forced: dict) -> list:
    """Candidate (initial, selected, map) whiteboard names; the first that the Sequencer accepts is used."""
    lit = source_literals()
    initial = [forced["initial"]] if forced.get("initial") else \
        [n for n in FALLBACK_INITIAL if n in lit["particles"]] + [n for n in FALLBACK_INITIAL if n not in lit["particles"]]
    selected = [forced["selected"]] if forced.get("selected") else \
        [n for n in FALLBACK_SELECTED if n in lit["particles"]] + [n for n in FALLBACK_SELECTED if n not in lit["particles"]] + [None]
    maps = [forced["map"]] if forced.get("map") else \
        [n for n in FALLBACK_MAP if n in lit["maps"]] + [n for n in FALLBACK_MAP if n not in lit["maps"]]
    profiles = []
    for m in maps[:1]:
        for i in initial:
            for s in selected:
                profiles.append({"initial": i, "selected": s, "map": m})
    # a missing 'selected' collection must not hide a good 'initial' one: try without it before another 'initial'
    profiles.sort(key=lambda p: (initial.index(p["initial"]) if p["initial"] in initial else 99, p["selected"] is None))
    return profiles[:6]


def _call(obj, name):
    v = getattr(obj, name)
    return v() if callable(v) else v


def geometry_table(trackingGeometry) -> dict:
    """geometry_id (as acts reports it), volume, layer, sensitive of every sensitive surface of the ODD."""
    items = []
    try:
        for gid, _surf in trackingGeometry.geoIdSurfaceMap().items():
            items.append(gid)
    except AttributeError:
        trackingGeometry.visitSurfaces(lambda s: items.append(s.geometryId()))
    rows = []
    for g in items:
        try:
            value = int(_call(g, "value"))
        except Exception:
            value = int(g)
        rows.append((value, int(_call(g, "volume")), int(_call(g, "layer")), int(_call(g, "sensitive"))))
    rows = sorted({r for r in rows if r[3] != 0})
    if not rows:
        raise ExportError("the tracking geometry has no sensitive surfaces with a geometry id")
    import numpy as np
    arr = np.array(rows, dtype=np.uint64)
    return {"geometry_id": arr[:, 0], "volume": arr[:, 1].astype(np.int64), "layer": arr[:, 2].astype(np.int64),
            "sensitive": arr[:, 3].astype(np.int64)}


def check_geometry_bits(tab: dict) -> dict:
    """Compare the bit-mask decoding of hsf2.data.odd with the values ACTS reports for every sensitive surface."""
    d = odd.geo_decode(tab["geometry_id"])
    mism = {k: int((d[k] != tab[k]).sum()) for k in ("volume", "layer", "sensitive")}
    return {"n_surfaces": int(len(tab["geometry_id"])), "mismatches": mism, "bit_layout_verified": not any(mism.values()),
            "volumes": sorted(set(tab["volume"].tolist()))}


def write_surfaces(path: Path, tab: dict) -> None:
    with open(path, "w", newline="\n") as f:
        f.write("geometry_id,volume,layer,sensitive\n")
        for g, v, l, s in zip(tab["geometry_id"].tolist(), tab["volume"].tolist(), tab["layer"].tolist(),
                              tab["sensitive"].tolist()):
            f.write(f"{g},{v},{l},{s}\n")


# ---------------------------------------------------------------------------
# --check
# ---------------------------------------------------------------------------
def import_ckf_odd(phase1_dir: str = None):
    cands = [Path(phase1_dir)] if phase1_dir else []
    cands += _phase1.candidates()
    for d in cands:
        if (Path(d) / "ckf_odd.py").is_file():
            if str(d) not in sys.path:
                sys.path.append(str(d))
            return importlib.import_module("ckf_odd"), Path(d).resolve()
    raise ExportError("ckf_odd.py not found; set HSF_PHASE1_DIR or --phase1-dir (tried "
                      + ", ".join(str(c) for c in cands) + ")")


def check(a) -> int:
    ok = True
    ckf_odd = None
    print(f"python {sys.version.split()[0]}   acts {acts_version() if acts else 'NOT IMPORTABLE'}")
    if acts is None:
        print("ACTS python bindings not found: run this from the ACTS shell")
        return 1
    try:
        ckf_odd, p1 = import_ckf_odd(a.phase1_dir)
        print(f"ckf_odd.py (read-only)   {p1 / 'ckf_odd.py'}")
        print(f"  run_ckf / build_odd / particle_file / select_generated present: "
              f"{all(hasattr(ckf_odd, n) for n in ('run_ckf', 'build_odd', 'particle_file', 'select_generated'))}")
        pf = ckf_odd.particle_file(Path(a.indir) if a.indir else p1 / "odd_events", int(a.seed))
        print(f"  event file for seed {a.seed}: {pf}")
        ok &= pf is not None
    except ExportError as exc:
        print(exc)
        ok = False
    print("writer classes of this ACTS build:")
    for mod, names in available_writers().items():
        print(f"  {mod}: {' '.join(names)}")
    for name, (need, opt) in WRITER_SPECS.items():
        try:
            cls = find_writer(name)
            fields = config_fields(cls)
            miss = [k for k in need if fields is not None and k not in fields]
            print(f"  {name}: found; config fields {fields}" + (f"  <-- MISSING {miss}" if miss else ""))
            ok &= not miss
        except ExportError as exc:
            print(f"  {name}: NOT FOUND  <-- required")
            ok = False
    seq_fields = config_fields(acts.examples.Sequencer)
    print(f"Sequencer config fields: {seq_fields}")
    if seq_fields is not None and "skip" not in seq_fields:
        print("  no 'skip' in Sequencer config: --skip will not work (use 0)")
    lit = source_literals()
    print(f"collection names in ACTS's python sources: {json.dumps(lit)}")
    print(f"collection candidates that will be tried: {collection_profiles({})}")
    for fn in ("acts.examples.reconstruction.addCKFTracks", "acts.examples.simulation.addDigiParticleSelection"):
        mod, _, attr = fn.rpartition(".")
        try:
            print(f"  {attr}{inspect.signature(getattr(importlib.import_module(mod), attr))}")
        except Exception as exc:
            print(f"  {attr}: {type(exc).__name__}: {exc}")
    try:
        import uproot
        print(f"uproot {uproot.__version__}")
    except ImportError:
        print("uproot NOT importable  <-- needed for the CKF tracks (trackstates_ckf.root)")
        ok = False
    if a.with_geometry and ckf_odd is not None:
        t0 = time.time()
        geo = ckf_odd.build_odd()
        tab = geometry_table(geo[0])
        res = check_geometry_bits(tab)
        print(f"ODD built in {time.time() - t0:.0f} s; geometry-id check: {json.dumps(res)}")
        if not res["bit_layout_verified"]:
            print("  the bit masks of hsf2/data/odd.py DISAGREE with ACTS: the reader will use surfaces.csv instead")
    print("CHECK", "PASSED" if ok else "FOUND PROBLEMS (see above)")
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# the export
# ---------------------------------------------------------------------------
def install_hooks(skip: int, add_writers, ckf_odd_module):
    """Wrap Sequencer (adds `skip`) and addCKFTracks (turns on its track-states writer, adds the writers after it) for one
    run_ckf call."""
    import acts.examples as ex
    import acts.examples.reconstruction as rec
    real_seq, real_add = ex.Sequencer, rec.addCKFTracks

    def seq_factory(*args, **kw):
        if skip:
            kw["skip"] = int(skip)
        return real_seq(*args, **kw)

    def add_hook(s, *args, **kw):
        # Phase 1 leaves writeTrackStates at its default (False); the export needs trackstates_ckf.root for the
        # measurement lists of the CKF tracks.  This only adds RootTrackStatesWriter; the CKF itself is unchanged.
        kw["writeTrackStates"] = True
        out = real_add(s, *args, **kw)
        add_writers(s)
        return out

    ex.Sequencer, rec.addCKFTracks = seq_factory, add_hook

    def restore():
        ex.Sequencer, rec.addCKFTracks = real_seq, real_add
    return restore


def unique_export_dir(label: str) -> Path:
    from hsf2.datasets import ensure_inside_work, work_root
    from hsf2.utils import clean_label, stamp, unique_dir
    parent = ensure_inside_work(work_root() / "odd_export", "odd_export output folder")
    name = stamp() + (("_" + clean_label(label)) if clean_label(label) else "")
    return unique_dir(parent, name)


def count_events_in_particle_file(path: Path):
    try:
        import uproot
        with uproot.open(str(path)) as f:
            names = [k.split(";")[0] for k in f.keys()]
            if "particles" in names:
                return int(f["particles"].num_entries)
            print(f"  (cannot count events: no tree 'particles' in {path.name}; trees {names})")
    except Exception as exc:
        print(f"  (cannot count events of {path.name}: {type(exc).__name__}: {exc})")
    return None


def run_attempt(a, ckf_odd, geo, adir: Path, profile: dict, n_events: int, pfile: Path, extra_args: list):
    truth = adir / "truth"
    for d in odd.TRUTH_DIRS.values():
        (adir / d).mkdir(parents=True, exist_ok=True)
    (adir / "truth" / "particles_selected").mkdir(parents=True, exist_ok=True)
    ckf_dir = adir / "ckf"
    ckf_dir.mkdir(parents=True, exist_ok=True)
    level = acts.logging.INFO

    def add_writers(s):
        s.addWriter(make_writer("CsvSimHitWriter", level, inputSimHits="simhits",
                                outputDir=str(adir / odd.TRUTH_DIRS["simhits"]), outputStem="simhits",
                                outputPrecision=12))
        s.addWriter(make_writer("CsvMeasurementWriter", level, inputMeasurements="measurements",
                                inputMeasurementSimHitsMap=profile["map"],
                                outputDir=str(adir / odd.TRUTH_DIRS["measurements"]), outputPrecision=12))
        s.addWriter(make_writer("CsvParticleWriter", level, inputParticles=profile["initial"],
                                outputDir=str(adir / odd.TRUTH_DIRS["particles"]), outputStem="particles",
                                outputPrecision=12))
        if profile["selected"]:
            s.addWriter(make_writer("CsvParticleWriter", level, inputParticles=profile["selected"],
                                    outputDir=str(truth / "particles_selected"), outputStem="particles_selected",
                                    outputPrecision=12))

    os.environ[ckf_odd.FPE_ENV] = "1"                   # as ckf_odd's main: before the first Sequencer is created
    opts = ckf_odd.getArgumentParser().parse_args(
        [f"--indir={Path(pfile).parent}", f"--output={ckf_dir}", f"--nEvents={n_events}", f"--seed={a.seed}",
         f"--nThreads={a.nThreads}"] + extra_args)
    restore = install_hooks(a.skip, add_writers, ckf_odd)
    try:
        ckf_odd.run_ckf(opts, geo=geo, report=False)
    finally:
        restore()
    return opts


def post_process(export_dir: Path) -> dict:
    """performance json (Phase 1's read_metrics on performance_finding_ckf.root), CKF tracks from trackstates, counts of
    the selected particles; updates the manifest."""
    from hsf2.utils import save_json
    ex = Path(export_dir)
    man = odd.load_manifest(ex)
    ckf_dir = ex / "ckf"
    res = {"post_started": time.strftime("%d-%m-%Y-%H-%M")}
    try:
        p1 = Path(man.get("phase1_dir", "")) if man.get("phase1_dir") else None
        if p1 and str(p1) not in sys.path:
            sys.path.append(str(p1))
        from acts_qaoa.ckf_runner import read_metrics, read_timing
        m = read_metrics(ckf_dir)
        keep = {k: float(v) for k, v in m.items() if k in ("efficiency", "fakerate", "duplicaterate")}
        keep.update({"timing": {k: float(v) for k, v in read_timing(ckf_dir).items()}})
        (ex / odd.PERFORMANCE_JSON).write_text(json.dumps(keep, indent=2) + "\n")
        res["performance"] = keep
    except Exception as exc:
        res["performance_error"] = f"{type(exc).__name__}: {exc}"
    states = sorted(ckf_dir.glob("*trackstates*ckf*.root")) or sorted(ckf_dir.glob("*trackstates*.root"))
    if not states:
        res["tracks_error"] = f"no trackstates*.root in {ckf_dir}: files there: {sorted(p.name for p in ckf_dir.iterdir())}"
    else:
        try:
            res["tracks"] = odd.convert_trackstates_root(ex, states[0])
            (ex / "ckf_tracks_report.json").write_text(json.dumps(res["tracks"], indent=2) + "\n")
        except Exception as exc:
            res["tracks_error"] = f"{type(exc).__name__}: {exc}"
    sel = ex / "truth" / "particles_selected"
    if sel.is_dir():
        counts = {}
        for p in sel.glob("event*-*.csv"):
            mm = re.match(r"event0*(\d+)-", p.name)
            with open(p) as fh:
                counts[int(mm.group(1))] = max(sum(1 for _ in fh) - 1, 0)
        res["n_selected_particles"] = {str(k): counts[k] for k in sorted(counts)}
    man.update({"post": res, "status": "complete" if not any(k.endswith("_error") for k in res) else "post_failed"})
    save_json(ex / odd.EXPORT_MANIFEST, man)
    return res


def export(a) -> int:
    from hsf2.utils import save_json
    if acts is None:
        raise ExportError("ACTS python bindings not found: run this from the ACTS shell")
    ckf_odd, p1 = import_ckf_odd(a.phase1_dir)
    indir = Path(a.indir) if a.indir else p1 / "odd_events"
    pfile = ckf_odd.particle_file(indir, int(a.seed))
    if pfile is None:
        raise ExportError(f"no particles_seed{a.seed}.root (or particles.root) in {indir}: Phase 1 would then generate "
                          "events on the fly; the export needs the event file so that the events are reproducible")
    n_file = count_events_in_particle_file(pfile)
    need = int(a.skip) + int(a.events)
    if n_file is not None and need > n_file:
        raise ExportError(f"{pfile.name} holds {n_file} events, --skip {a.skip} + --events {a.events} = {need}")
    out = unique_export_dir(a.label)
    export_id = out.name
    cmdline = "python " + shlex.join(sys.argv)
    manifest = {"export_id": export_id, "label": a.label, "seed": int(a.seed), "skip": int(a.skip),
                "n_events_requested": int(a.events), "events": [], "status": "running", "acts_version": acts_version(),
                "phase1_dir": str(p1), "particle_file": str(pfile), "particle_file_events": n_file,
                "command": cmdline, "started": time.strftime("%d-%m-%Y-%H-%M"), "host": os.uname().nodename
                if hasattr(os, "uname") else "?", "ckf_extra_args": a.ckf_args}
    save_json(out / odd.EXPORT_MANIFEST, manifest)
    print(f"export folder {out}", flush=True)
    t0 = time.time()
    geo = ckf_odd.build_odd()
    print(f"ODD built in {time.time() - t0:.0f} s", flush=True)
    try:
        tab = geometry_table(geo[0])
        manifest["geometry_check"] = check_geometry_bits(tab)
        write_surfaces(out / odd.SURFACES_CSV, tab)
        if not manifest["geometry_check"]["bit_layout_verified"]:
            print(f"WARNING: geometry-id bit masks differ from ACTS: {manifest['geometry_check']['mismatches']}; "
                  "the reader uses surfaces.csv (ACTS values)", flush=True)
    except Exception as exc:
        manifest["geometry_check"] = {"available": False, "reason": f"{type(exc).__name__}: {exc}"}
        print(f"WARNING: no geometry table ({exc}); the reader will decode geometry ids by bit masks", flush=True)
    save_json(out / odd.EXPORT_MANIFEST, manifest)
    forced = {"initial": a.initial_particles, "selected": a.selected_particles, "map": a.measurement_simhit_map}
    profiles = collection_profiles({k: v for k, v in forced.items() if v})
    extra = shlex.split(a.ckf_args or "")
    failures = []
    for k, prof in enumerate(profiles, 1):
        adir = out / f"attempt{k}"
        print(f"[attempt {k}/{len(profiles)}] collections {prof}", flush=True)
        try:
            run_attempt(a, ckf_odd, geo, adir, prof, int(a.events), pfile, extra)
        except ExportError:
            raise
        except Exception as exc:                          # wrong whiteboard name etc.: keep the folder, try the next
            failures.append({"profile": prof, "error": f"{type(exc).__name__}: {exc}", "folder": adir.name})
            print(f"  attempt {k} failed: {type(exc).__name__}: {exc}", flush=True)
            continue
        for sub in ("truth", "ckf"):                      # success: move the results up (nothing is deleted)
            os.replace(adir / sub, out / sub)
        manifest["collections"] = prof
        break
    else:
        manifest.update({"status": "failed", "failed_attempts": failures})
        save_json(out / odd.EXPORT_MANIFEST, manifest)
        raise ExportError("every collection-name attempt failed (see the manifest); run --check and pass "
                          "--initial-particles / --selected-particles / --measurement-simhit-map explicitly. "
                          f"Errors: {json.dumps(failures, indent=1)}")
    manifest["failed_attempts"] = failures
    found = sorted({int(re.match(r"event0*(\d+)-", p.name).group(1))
                    for p in (out / odd.TRUTH_DIRS["measurements"]).glob("event*-measurements.csv")})
    manifest["events"] = found
    manifest["seconds_acts"] = round(time.time() - t0, 1)
    if len(found) != int(a.events):
        manifest["status"] = "wrong_event_count"
        save_json(out / odd.EXPORT_MANIFEST, manifest)
        raise ExportError(f"expected {a.events} events but found measurement files of {len(found)} ({found[:5]}...)")
    save_json(out / odd.EXPORT_MANIFEST, manifest)
    # post-processing in a fresh process: ACTS closes its ROOT files only when its writers are destroyed
    r = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--post", str(out)])
    if r.returncode != 0:
        raise ExportError(f"post-processing failed (exit {r.returncode}); the ACTS output is complete in {out}; "
                          f"after fixing, run: python {Path(__file__).name} --post {out}")
    print(f"export complete: {out}   ({manifest['seconds_acts']} s)", flush=True)
    print(out)
    return 0


def getArgumentParser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--events", type=int, default=5, help="number of events to export")
    p.add_argument("--seed", type=int, default=1000000,
                   help="seed of RandomNumbers and of particles_seed<seed>.root (Phase 1's confirmation samples: "
                        "1000000 ... 6000000, 100 events each)")
    p.add_argument("--label", default="odd", help="folder name suffix of hsf2_work/odd_export/<stamp>_<label>")
    p.add_argument("--skip", type=int, default=0, help="skip the first N events of the particle file")
    p.add_argument("--check", action="store_true", help="diagnose this ACTS build and exit")
    p.add_argument("--with-geometry", action="store_true", help="--check: also build the ODD and verify geometry ids")
    p.add_argument("--post", default=None, metavar="EXPORT_DIR", help="only the post-processing of an export folder")
    p.add_argument("--indir", default=None, help="folder with particles_seed<seed>.root (default <Phase 1>/odd_events)")
    p.add_argument("--phase1-dir", default=None, help="folder with ckf_odd.py (default: $HSF_PHASE1_DIR, ...)")
    p.add_argument("--nThreads", type=int, default=1, help="Sequencer threads (1: stable, as the Phase-1 timing)")
    p.add_argument("--ckf-args", default="", help='extra ckf_odd options in one string, e.g. "--sf_deltaRMax=100"; '
                                                  'default: the ACTS-default configuration of the Phase-1 confirmation')
    p.add_argument("--initial-particles", default=None, help="whiteboard name of the initial-state particles")
    p.add_argument("--selected-particles", default=None, help="whiteboard name of the digitisation-selected particles")
    p.add_argument("--measurement-simhit-map", default=None, help="whiteboard name of the measurement-simhit map")
    return p


def main() -> int:
    a = getArgumentParser().parse_args()
    try:
        if a.post:
            res = post_process(Path(a.post))
            print(json.dumps({k: v for k, v in res.items() if k != "tracks"}, indent=1))
            if "tracks" in res:
                print(f"tracks: {res['tracks']['n_states']} states, {res['tracks']['n_states_unmatched']} unmatched")
            return 0 if not any(k.endswith("_error") for k in res) else 1
        if a.check:
            return check(a)
        if a.events < 1:
            raise ExportError("--events must be >= 1")
        return export(a)
    except ExportError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())

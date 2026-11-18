"""Command line:  python -m hsf2 <command> [options]      (python -m hsf2 <command> --help)

Data and models
  selftest       end-to-end check on the synthetic detector (no data needed, ~1 min)
  synth          write a synthetic data folder (same format as TrackML; for tests and dry runs)
  trackml-prep   TrackML zip (local, or zenodo:train_1.zip via HTTP range requests) -> data folder
  odd-prep       exports of scripts/odd_export.py (ACTS ODD chain of Phase 1) -> data folder (same layout as trackml-prep)
  compare-ckf    GNN versus the exported ACTS CKF tracks on the same events, one matching / metric definition,
                 paired bootstrap intervals, cross-check against the CKF's own performance file
  train          train the Interaction Network edge classifier -> models/<DD-MM-YYYY-HH-MM>_<label>/
  evaluate       reconstruct tracks with one parameter set on a split -> <model>/outputs/<stamp>_evaluate_<split>/
  scan           1-D scan of one tracking parameter -> <model>/outputs/<stamp>_scan_<param>/
                 (evaluate/scan --level S0|S0N|S1: evaluate the model on another hit selection)
  compare-evals  paired per-event difference B - A of two evaluate folders (same events) -> compare.json
  info          summary of a data or model folder

Tuning (Phase-1 surrogate -> QUBO -> QAOA pipeline on the GNN objective; runs/<DD-MM-YYYY-HH-MM>_<label>/)
  objective options of tune-compare / tune-loop: --score-eff efficiency_dm (S_DM), --k-time 20, --dm-floor 0.99
  tune-landscape QX1: exhaustive 16-bit landscape of the real objective (+ independent verification)
  tune-compare   QX1: optimisers on the exact landscape (QUBO loops with exact/SA/QAOA samplers vs baselines)
  tune-objectives QX1: optimum of S and of its sensitivity variants (S_DM, t/20, DM floor) on a landscape
  tune-loop      QX2: 24-bit closed loop on the real objective, baselines, confirmation on held-out events
  tune-scaling   QX6: QAOA on the same tuning problem at 16-32 qubits (Aer statevector) vs exact, SA and random

Quantum track selection
  select         QX3: candidate-conflict QUBO per conflict component (exact / SA / QAOA) vs greedy resolution
  triplet        QX3: Bapst-style triplet QUBO on barrel sectors (exact / SA / QAOA on components)

Exploratory (emulation)
  hhl            QX4: linear-system tracking (TrackHHL): classical solve vs HHL / 1-bit-filter emulation, gate counts
  qgnn           QX5: variational quantum edge classifier vs classical classifiers on the same inputs
  report         figures of the guide from saved output folders (figures_manifest.json records the sources)
  ibm-status     IBM account usage and calibration medians of the account's QPUs (metadata only, no QPU time)
  hw-validate    hardware samples of a QX2 run folder on the real objective, against uniform and ideal-circuit
                 controls of the same size and the exact QUBO optimum (0 QPU s)
  hw-analyze     noise analysis of the hardware samples: F_E, F_XEB, tails, correlation-order fidelity s_k, CVaR bounds
                 (0 QPU s, no GNN)
  hw-run         one hardware job of the suppression ladder (settings a-f): --mode dry (plan only), local (fake backend
                 + Aer, 0 QPU s) or submit (typed confirmation 'submit', QPU ledger, budget guard)
  hw-ladder      the settings of the ladder one after the other, for one or more K (couplings kept)
  hw-readout     readout calibration of the qubits of a job (|0..0>, |1..1>) and mthree correction of its counts
  hw-retrieve    rebuild the outputs of a submitted job from its id (lost shell, long queue)
  hw-ledger      QPU-time ledger of the IBM account (budget, usage per job, remaining)
  hw-components  step 3: QX3b components (hard pieces, plus a benchmark sample) of a triplet run folder on hardware, one
                 component per PUB, --per-job PUBs per job; --list shows the selection; modes as hw-run; then runs
                 hw-components-analyze
  hw-components-analyze  per component: P(ground), F_E / F_XEB, best-of-N, single-flip descent and the uniform / ideal /
                 white-noise controls at the same N; aggregates by group (hard / benchmark) (0 QPU s)
  (hw-analyze and hw-components-analyze take --readout <readout_cal.json> [--readout-tags REGEX]: raw and readout-mitigated
                 values side by side)
  hw-zne         step 4: zero-noise extrapolation jobs (folded circuits, lambda = 1, 3, 5) for QX2 K values of a QX2 run folder
                 or QX3b components of a triplet run folder; modes as hw-run
  hw-zne-analyze linear / Richardson / exponential extrapolation of F_E, F_XEB, tails, P(ground), s_k to lambda = 0 with
                 bootstrap errors and the ideal value (0 QPU s)
  hw-qx4         step 5: QX4 on hardware, the 1-bit filter of TrackHHL on the 2x3 (N = 8) and 2x5 (N = 16) toy events at folding
                 factors 1, 3, 5, plus an optional HHL fidelity PUB; read-out calibration right after; modes as hw-run
  hw-qx4-analyze leakage L = P(isolated segment | ancilla 0), P(post), filter decisions, distances, ratio ZNE (numerator and
                 denominator extrapolated separately), raw and read-out-corrected (0 QPU s)
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from . import __version__
from .config import (ExperimentRecord, GraphConfig, HitSelection, ModelConfig, ScoreConfig, SplitConfig,
                     TargetConfig, TrackingParams, TRACKING_DEFAULTS, TRACKING_PARAM_NAMES, TrainConfig)
from .datasets import data_root, load_split, models_root, resolve_data, resolve_model, runs_root, splits
from .utils import (append_jsonl, code_version, command_dir, load_json, new_run_dir, save_csv, save_json,
                    save_text, setup_logging, stamp)

log = logging.getLogger("hsf2")


_ARGV: List[str] = []


def _cmdline() -> str:
    """The command line of this invocation, as recorded next to every result."""
    import shlex
    return "python -m hsf2 " + shlex.join(_ARGV)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _hit_selection(level: str, pt: float) -> HitSelection:
    level = level.upper()
    if level == "S0N":                      # S0 hits + all noise hits
        return HitSelection("S0N", float(pt), True)
    return HitSelection.preset(level, pt)


def parse_sets(items: Optional[List[str]], base: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """['cut_hi=0.6', 'max_degree=2'] -> dict on top of `base` (default: TRACKING_DEFAULTS)."""
    out = dict(TRACKING_DEFAULTS if base is None else base)
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"--set expects NAME=VALUE, got {it!r}")
        k, v = it.split("=", 1)
        k = k.strip()
        if k not in TrackingParams.__dataclass_fields__:
            raise SystemExit(f"unknown tracking parameter {k!r}; known: {list(TrackingParams.__dataclass_fields__)}")
        out[k] = float(v)
    return out


def _df(rows):
    import pandas as pd
    return pd.DataFrame(rows)


def _summary_line(m: dict) -> str:
    return ("eff %.4f  eff_dm %.4f  fake %.4f  dup %.4f  TrackML %.4f  tracks/evt %.0f  edges %.0f  t %.2fs/evt"
            % (m["efficiency"], m["efficiency_dm"], m["fakerate"], m["duplicaterate"], m["trackml_score"],
               m["n_tracks"] / max(m["n_events"], 1), m["edges_filtered_per_event"], m["t_per_event"]))


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------
def cmd_selftest(args) -> int:
    from .selftest import run_selftest
    return 0 if run_selftest(quick=args.quick) else 1


def cmd_synth(args) -> int:
    from .data.synthetic import ToyConfig, ToyDetector, generate_event
    out = new_run_dir(data_root(), args.label)
    setup_logging(args.log_level, out / "log.txt")
    cfg = ToyConfig(n_particles=args.particles, seed=args.seed, noise_frac=args.noise_frac)
    det = ToyDetector()
    rows = []
    for i in range(args.n_events):
        ev = generate_event(det, cfg, i)
        ev.save(out / "events" / f"{ev.event_id}.npz")
        rows.append({"event_id": ev.event_id, "n_hits": ev.n_hits, "n_particles": ev.n_particles,
                     "n_noise": int((ev.particle_id == 0).sum())})
    sc = SplitConfig(args.n_train, args.n_val, args.n_tune, args.n_confirm, args.n_confirm_sets)
    save_json(out / "splits.json", sc.make([r["event_id"] for r in rows]))
    save_csv(_df(rows), out / "events.csv")
    save_json(out / "manifest.json", {"source": "synthetic", "created": stamp(), "n_events": len(rows),
                                      "generator": cfg.__dict__, "splits": sc.to_dict(), "code": code_version()})
    log.info("synthetic data folder: %s (%d events)", out, len(rows))
    print(out)
    return 0


def cmd_trackml_prep(args) -> int:
    from .data.trackml import check_detectors, prep_from_zip
    if args.zip is None:
        args.zip = str(data_root() / "raw" / "train_1.zip")
        if not Path(args.zip).exists():
            raise SystemExit(f"{args.zip} not found: download it (hsf2_scripts/trackml_download.sh) or pass --zip zenodo:train_1.zip")
    if args.detectors == "auto":
        cand = data_root() / "raw" / "detectors.csv"
        args.detectors = str(cand) if cand.exists() else None
    elif args.detectors == "none":
        args.detectors = None
    out = new_run_dir(data_root(), args.label)
    setup_logging(args.log_level, out / "log.txt")
    if args.detectors:
        check_detectors(Path(args.detectors))
        log.info("detectors.csv layout OK (48 layers)")
    sc = SplitConfig(args.n_train, args.n_val, args.n_tune, args.n_confirm, args.n_confirm_sets)
    n_events = args.n_events or (sc.n_train + sc.n_val + sc.n_tune + sc.n_confirm * sc.n_confirm_sets)
    save_json(out / "prep_config.json", {"zip": args.zip, "n_events": n_events, "skip": args.skip,
                                         "workers": args.workers, "splits": sc.to_dict(), "started": stamp(),
                                         "code": code_version()})
    t0 = time.time()
    rows = prep_from_zip(args.zip, out, n_events, args.skip, args.workers)
    save_json(out / "splits.json", sc.make([r["event_id"] for r in rows]))
    df = _df(rows)
    save_csv(df, out / "events.csv")
    stats = {k: float(df[k].mean()) for k in df.columns if k != "event_id"}
    save_json(out / "manifest.json", {"source": "trackml", "zip": args.zip, "created": stamp(), "n_events": len(rows),
                                      "mean_per_event": stats, "splits": sc.to_dict(), "skip": args.skip,
                                      "seconds": round(time.time() - t0, 1), "code": code_version()})
    log.info("TrackML data folder: %s  (%d events; mean hits %.0f, pT>1 hits %.0f, noise %.0f)", out, len(rows),
             stats["n_hits"], stats["n_hits_pt1"], stats["n_noise"])
    print(out)
    return 0


def cmd_odd_prep(args) -> int:
    from .data.odd import prep_from_exports
    out = new_run_dir(data_root(), args.label)
    setup_logging(args.log_level, out / "log.txt")
    exports = [Path(p) for p in args.export]
    sc = SplitConfig(args.n_train, args.n_val, args.n_tune, args.n_confirm, args.n_confirm_sets)
    n_events = args.n_events or (sc.n_train + sc.n_val + sc.n_tune + sc.n_confirm * sc.n_confirm_sets)
    save_json(out / "prep_config.json", {"command": _cmdline(), "exports": [str(p) for p in exports],
                                         "n_events": n_events, "splits": sc.to_dict(), "started": stamp(),
                                         "code": code_version()})
    t0 = time.time()
    rows, info, layer_index, frame_info = prep_from_exports(exports, out, n_events, log.info, positions=args.positions,
                                                            n_frame_events=sc.n_train)
    save_json(out / "splits.json", sc.make([r["event_id"] for r in rows]))
    df = _df(rows)
    save_csv(df, out / "events.csv")
    stats = {k: float(df[k].mean()) for k in df.columns if k != "event_id"}
    save_json(out / "manifest.json", {"source": "odd", "created": stamp(), "n_events": len(rows),
                                      "exports": info, "n_layers": len(layer_index),
                                      "layers": [[v, l] for (v, l) in layer_index],
                                      "position_source": "measured" if args.positions == "measured" else "simhit_truth",
                                      "frames": frame_info, "mean_per_event": stats, "splits": sc.to_dict(),
                                      "seconds": round(time.time() - t0, 1), "code": code_version()})
    if frame_info.get("n_positions_without_frame"):
        log.warning("%d measurements lie on surfaces without a fitted frame and keep their simhit position",
                    frame_info["n_positions_without_frame"])
    log.info("ODD data folder: %s  (%d events, %d layers; mean hits %.0f, pT>1 hits %.0f)", out, len(rows),
             len(layer_index), stats["n_hits"], stats["n_hits_pt1"])
    if stats["n_hits_particle_missing"] > 0:
        log.warning("%.1f hits per event belong to a particle that is not in the exported particle table "
                    "(never a target): check the particle collection of the export", stats["n_hits_particle_missing"])
    print(out)
    return 0


def cmd_compare_ckf(args) -> int:
    from .datasets import ensure_inside_work
    from .tracking import compare as cmp
    from .tracking.pipeline import GNNPipeline
    model_dir, model, data_dir = _load_model_and_data(args)
    out = command_dir(model_dir, f"compare-ckf_{args.split}")
    setup_logging(args.log_level, out / "log.txt")
    params = TrackingParams(**parse_sets(args.set))
    targets = TargetConfig(pt_min=args.target_pt, eta_max=args.target_eta, min_hits=args.target_min_hits)
    exports = cmp.export_map([ensure_inside_work(Path(p), "--export") for p in args.export])
    rec = ExperimentRecord(command=_cmdline(), hits=model.hit_selection.to_dict(), targets=targets.to_dict(),
                           model=model.mcfg.to_dict(), tracking=params.to_dict(), data_dir=str(data_dir),
                           model_dir=str(model_dir), split=args.split,
                           events=splits(data_dir)[args.split][: args.max_events or None], code=code_version(),
                           extra={"exports": {k: str(v) for k, v in exports.items()}, "n_boot": args.n_boot,
                                  "boot_seed": args.boot_seed})
    save_json(out / "config.json", rec.to_dict())
    events = load_split(data_dir, args.split, args.max_events)
    selection = dict(cmp.CKF_SELECTION) if args.select == "ckf" else None
    position_source = {e.meta.get("position_source", "simhit_truth") for e in events}
    skip_layers: List[int] = []
    if selection is not None and position_source == {"measured"}:
        from .data import odd
        skip_layers = odd.one_coordinate_layers(odd.load_frames(Path(data_dir) / odd.FRAMES_FILE),
                                                load_json(Path(data_dir) / "manifest.json")["layers"])
        log.info("selection refit without the hits of %d one-coordinate layers %s", len(skip_layers), skip_layers)
    res = cmp.compare_events(GNNPipeline(model, targets), params, events, exports, targets, selection=selection,
                             fit_skip_layers=skip_layers)
    summary = cmp.summarize_comparison(res, args.n_boot, args.boot_seed)
    if position_source == {"measured"}:
        pos_note = ("positions: the GNN sees the digitised measurements (surface frames fitted on the training events; "
                    "1-D long-strip hits on the strip centre line), as the CKF does")
    else:
        pos_note = "positions: the GNN sees simhit truth positions, the CKF the smeared measurements"
    caveats = [pos_note, "timing is indicative only (different code, GNN = inference + track building on this node)"]
    if model.hit_selection.pt_keep > 0 or not model.hit_selection.keep_noise:
        caveats.insert(0, f"the model uses hit selection {model.hit_selection.level} (truth pT >= "
                          f"{model.hit_selection.pt_keep:g} GeV): a truth-informed filter the CKF does not have; "
                          "train with --level S1 for the like-for-like comparison")
    summary["caveats"] = caveats
    summary["model_hit_selection"] = model.hit_selection.to_dict()
    save_json(out / "summary.json", summary)
    save_csv(_df([{"method": m, **r} for m in ("gnn", "ckf") for r in res[m]]), out / "per_event.csv")
    if not args.no_crosscheck:
        cc = [cmp.crosscheck_export(p, targets, args.crosscheck_max_events) for p in exports.values()]
        save_json(out / "crosscheck.json", cc)
        for c in cc:
            log.info("cross-check %s: Python %s | ACTS %s | difference %s", c["export_id"], c["python"],
                     c.get("acts"), c.get("difference_python_minus_acts"))
    text = cmp.format_report(summary, caveats)
    save_text(out / "compare.txt", text + "\n")
    log.info("\n%s", text)
    print(out)
    return 0


def cmd_train(args) -> int:
    import torch
    from .data.event import select_hits
    from .graph.construct import build_graph, learn_pair_map
    from .model.train import train_model
    data_dir = resolve_data(args.data)
    sel = _hit_selection(args.level, args.pt)
    out = new_run_dir(models_root(), args.label or f"in_{sel.level}")
    setup_logging(args.log_level, out / "log.txt")
    gcfg = GraphConfig(pt_min_map=args.pt)
    mcfg = ModelConfig(hidden=args.hidden, n_iter=args.n_iter, mlp_layers=args.mlp_layers, shared=not args.unshared,
                       label=args.edge_label)
    tcfg = TrainConfig(epochs=args.epochs, lr=args.lr, lr_min=args.lr_min, seed=args.seed, threads=args.threads,
                       max_train_events=args.max_events, max_pos_weight=args.max_pos_weight)
    if tcfg.threads:
        torch.set_num_threads(tcfg.threads)
    sp = splits(data_dir)
    train_ids = sp["train"][: args.max_events] if args.max_events else sp["train"]
    val_ids = sp["val"][: args.max_val_events] if args.max_val_events else sp["val"]
    rec = ExperimentRecord(command=_cmdline(), hits=sel.to_dict(), graph=gcfg.to_dict(),
                           model=mcfg.to_dict(), train=tcfg.to_dict(), data_dir=str(data_dir), split="train",
                           events=train_ids, code=code_version(), extra={"val_events": val_ids})
    save_json(out / "train_config.json", rec.to_dict())
    log.info("model folder %s | data %s | %s hits, pT >= %.2f GeV | %d train / %d val events | torch threads %d",
             out, data_dir, sel.level, sel.pt_keep, len(train_ids), len(val_ids), torch.get_num_threads())
    t0 = time.time()
    train_ev = [select_hits(e, sel) for e in load_split(data_dir, "train", len(train_ids))]
    val_ev = [select_hits(e, sel) for e in load_split(data_dir, "val", len(val_ids))]
    pmap = learn_pair_map(train_ev, gcfg)
    log.info("layer-pair map: %s", pmap.stats)
    train_set = [(e, build_graph(e, pmap, truth_pt_min=args.pt)) for e in train_ev]
    val_set = [(e, build_graph(e, pmap, truth_pt_min=args.pt)) for e in val_ev]
    gstats = {}
    for name, ds in (("train", train_set), ("val", val_set)):
        keys = ("n_nodes", "n_edges", "n_true_edges", "edge_purity", "truth_edge_eff")
        gstats[name] = {k: float(np.mean([g.meta[k] for _, g in ds])) for k in keys}
    gstats["pair_map"] = pmap.stats
    save_json(out / "graph_stats.json", gstats)
    log.info("graphs (%.0f s): train mean %s", time.time() - t0,
             {k: round(v, 4) for k, v in gstats["train"].items()})

    def progress(r: dict) -> None:
        append_jsonl(out / "history.jsonl", r)
        log.info("epoch %3d  train %.4f  val %.4f  AUC %.4f  eff %.3f  pur %.3f  lr %.2e  %.0f s%s", r["epoch"],
                 r["train_loss"], r["val_loss"], r["auc"], r["edge_eff"], r["edge_purity"], r["lr"], r["time_s"],
                 "  *" if r.get("best") else "")

    model, history = train_model(train_set, val_set, mcfg, tcfg, pmap, sel, progress)
    model.meta.update({"data_dir": str(data_dir), "created": stamp(), "train_seconds": round(time.time() - t0, 1),
                       "graph_stats": gstats})
    model.save(out, {"code": code_version()})
    save_csv(_df(history), out / "history.csv")
    save_json(out / "summary.json", {"best_epoch": model.meta["best_epoch"], "best_val": model.meta["best_val"],
                                     "n_parameters": model.meta["n_parameters"], "seconds": model.meta["train_seconds"]})
    log.info("done: best epoch %s %s -> %s", model.meta["best_epoch"], model.meta["best_val"], out)
    print(out)
    return 0


def _load_model_and_data(args):
    from .model.train import EdgeModel
    model_dir = resolve_model(args.model)
    model = EdgeModel.load(model_dir)
    data_dir = resolve_data(args.data) if args.data else Path(model.meta.get("data_dir", ""))
    if args.data is None and not (data_dir / "manifest.json").exists():
        tc = load_json(model_dir / "train_config.json")
        data_dir = Path(tc["data_dir"])
    return model_dir, model, data_dir


def _targets(args, model) -> TargetConfig:
    return TargetConfig(pt_min=args.target_pt if args.target_pt is not None else max(model.hit_selection.pt_keep, 0.0) or 1.0,
                        eta_max=args.target_eta, min_hits=args.target_min_hits)


def _apply_level(model, level: Optional[str]) -> dict:
    """evaluate/scan --level: score the model on another hit selection (S0 model on S0N events = the mismatch case).
    The model on disk is untouched; returns the hit selection actually used and the model's own."""
    model.trained_level = model.hit_selection.level
    own = model.hit_selection.to_dict()
    if level and level.upper() != model.hit_selection.level:
        model.hit_selection = _hit_selection(level, model.hit_selection.pt_keep or 1.0)
        log.info("hit level override: model trained on %s, evaluated on %s", own["level"], model.hit_selection.level)
    return {"model_hits": own, "level_override": model.hit_selection.level != own["level"]}


def _evaluate(model, data_dir, split, max_events, params: TrackingParams, targets: TargetConfig, out: Path,
              tag: str) -> dict:
    from .model.train import edge_metrics
    from .tracking.metrics import efficiency_by
    from .tracking.pipeline import GNNPipeline
    pipe = GNNPipeline(model, targets)
    events = load_split(data_dir, split, max_events)
    rows, tracks_all, sel_events, ys, ss = [], [], [], [], []
    for ev in events:
        tracks, counts, (es, g, keep, s, cands) = pipe.run_event(ev, params)
        counts["event_id"] = ev.event_id
        rows.append(counts)
        tracks_all.append(tracks)
        sel_events.append(es)
        ys.append(g.y[keep]); ss.append(s)
    from .tracking.pipeline import aggregate_rows
    from .tracking.metrics import edge_pr_summary
    for r in rows:                                      # per-event ratios, for paired comparisons (compare-evals)
        r["eff"] = r["n_targets_found"] / r["n_targets"] if r["n_targets"] else float("nan")
        r["eff_dm"] = r["n_targets_found_dm"] / r["n_targets"] if r["n_targets"] else float("nan")
        r["fake"] = r["n_fake"] / r["n_tracks"] if r["n_tracks"] else float("nan")
        r["dup"] = r["n_duplicate"] / r["n_tracks"] if r["n_tracks"] else float("nan")
        r["trackml"] = r["trackml_score_raw"] / max(r["trackml_weight_total"], 1e-12)
    m = aggregate_rows(rows)
    y_all = np.concatenate(ys) if ys else np.zeros(0, bool)
    s_all = np.concatenate(ss) if ss else np.zeros(0)
    em = edge_metrics(y_all, s_all, params.cut_hi)
    m.update({f"edge_{k}" if not k.startswith("edge") else k: v for k, v in em.items()})
    m.update(edge_pr_summary(y_all, s_all))             # PR-AUC and purity at fixed recall (AUC alone hides purity)
    m["edge_recall"] = m["edge_eff"]                    # edge-level recall and precision at the working cut_hi
    m["edge_precision"] = m["edge_purity"]
    m["hit_level"] = model.hit_selection.level           # the hit selection actually used
    m["model_hit_level"] = getattr(model, "trained_level", model.hit_selection.level)
    m["hits_per_event"] = float(np.mean([r["n_hits"] for r in rows])) if rows else 0.0
    save_json(out / f"metrics{tag}.json", m)
    save_csv(_df(rows), out / f"per_event{tag}.csv")
    for key, bins in (("pt", np.array([1, 1.5, 2, 3, 5, 10, 100.0])), ("eta", np.linspace(-4, 4, 17))):
        eb = efficiency_by(sel_events, tracks_all, targets, key, bins)
        save_csv(_df([{"lo": a, "hi": b, "n": n, "eff": e} for a, b, n, e in zip(eb["lo"], eb["hi"], eb["n"], eb["eff"])]),
                 out / f"eff_vs_{key}{tag}.csv")
    return m


def cmd_evaluate(args) -> int:
    model_dir, model, data_dir = _load_model_and_data(args)
    out = command_dir(model_dir, f"evaluate_{args.split}")
    setup_logging(args.log_level, out / "log.txt")
    params = TrackingParams(**parse_sets(args.set))
    lvl = _apply_level(model, getattr(args, "level", None))
    targets = _targets(args, model)
    rec = ExperimentRecord(command=_cmdline(), hits=model.hit_selection.to_dict(),
                           targets=targets.to_dict(), model=model.mcfg.to_dict(), tracking=params.to_dict(),
                           data_dir=str(data_dir), model_dir=str(model_dir), split=args.split,
                           events=splits(data_dir)[args.split][: args.max_events or None], code=code_version(),
                           extra=lvl)
    save_json(out / "config.json", rec.to_dict())
    m = _evaluate(model, data_dir, args.split, args.max_events, params, targets, out, "")
    log.info("%s | %s", params.to_dict(), _summary_line(m))
    log.info("edge AUC %.4f, edge eff %.4f, purity %.4f (cut %.2f) -> %s", m["edge_auc"], m["edge_eff"],
             m["edge_purity"], params.cut_hi, out)
    print(out)
    return 0


def cmd_scan(args) -> int:
    model_dir, model, data_dir = _load_model_and_data(args)
    out = command_dir(model_dir, f"scan_{args.param}")
    setup_logging(args.log_level, out / "log.txt")
    base = parse_sets(args.set)
    lvl = _apply_level(model, getattr(args, "level", None))
    targets = _targets(args, model)
    values = [float(v) for v in args.values.split(",")]
    save_json(out / "config.json", {"command": _cmdline(), "param": args.param, "values": values, **lvl,
                                    "hits": model.hit_selection.to_dict(),
                                    "base": base, "targets": targets.to_dict(), "split": args.split,
                                    "max_events": args.max_events, "model_dir": str(model_dir), "data_dir": str(data_dir),
                                    "code": code_version()})
    from .config import ScoreConfig
    sc = ScoreConfig()
    rows = []
    for v in values:
        p = dict(base)
        p[args.param] = v
        m = _evaluate(model, data_dir, args.split, args.max_events, TrackingParams(**p), targets, out,
                      f"_{args.param}_{v:g}")
        row = {args.param: v, **{k: m[k] for k in ("efficiency", "efficiency_dm", "fakerate", "duplicaterate",
                                                     "trackml_score", "time_proxy", "t_per_event", "n_tracks")}}
        row["score"] = sc.score(m)
        rows.append(row)
        log.info("%s = %g | %s | score %.4f", args.param, v, _summary_line(m), row["score"])
    save_csv(_df(rows), out / "scan.csv")
    print(out)
    return 0


COMPARE_METRICS = (("efficiency", "eff"), ("efficiency_dm", "eff_dm"), ("fakerate", "fake"), ("duplicaterate", "dup"),
                   ("trackml_score", "trackml"))


def compare_evals(dir_a: Path, dir_b: Path, n_boot: int = 2000, seed: int = 0) -> dict:
    """Paired per-event difference B - A of two evaluate folders on the same events: mean, standard error, and the
    68 % bootstrap interval (16th-84th percentile of the resampled mean difference).  Refuses different event lists."""
    import pandas as pd
    dfs, cfgs = [], []
    for d in (Path(dir_a), Path(dir_b)):
        if not (d / "per_event.csv").is_file():
            raise SystemExit(f"{d}: no per_event.csv (not an evaluate output folder)")
        df = pd.read_csv(d / "per_event.csv", dtype={"event_id": str})
        miss = [c for _, c in COMPARE_METRICS if c not in df.columns]
        if miss:
            raise SystemExit(f"{d}: per_event.csv lacks {miss}; re-run evaluate with this version")
        dfs.append(df)
        cfgs.append(load_json(d / "config.json") if (d / "config.json").is_file() else {})
    ids = [list(df["event_id"]) for df in dfs]
    if sorted(ids[0]) != sorted(ids[1]) or len(set(ids[0])) != len(ids[0]):
        raise SystemExit(f"the two evaluations are not on the same events ({len(ids[0])} vs {len(ids[1])}; "
                         f"only in A: {sorted(set(ids[0]) - set(ids[1]))[:5]}, only in B: {sorted(set(ids[1]) - set(ids[0]))[:5]})")
    a, b = (df.set_index("event_id").loc[ids[0]] for df in dfs)
    rng = np.random.default_rng(seed)
    res: dict = {"n_events": len(ids[0]), "n_boot": n_boot, "A": str(dir_a), "B": str(dir_b), "metrics": {}}
    for name, col in COMPARE_METRICS:
        x, y = a[col].to_numpy(float), b[col].to_numpy(float)
        ok = np.isfinite(x) & np.isfinite(y)                  # events without targets / tracks drop out of that metric
        d = (y - x)[ok]
        n = int(len(d))
        if n == 0:
            res["metrics"][name] = {"n": 0, "mean_A": float("nan"), "mean_B": float("nan"), "diff": float("nan"),
                                    "stderr": float("nan"), "lo68": float("nan"), "hi68": float("nan")}
            continue
        boots = d[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
        res["metrics"][name] = {"n": n, "mean_A": float(x[ok].mean()), "mean_B": float(y[ok].mean()),
                                "diff": float(d.mean()), "stderr": float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan"),
                                "lo68": float(np.percentile(boots, 16)), "hi68": float(np.percentile(boots, 84))}
    res["hits_A"] = cfgs[0].get("hits")
    res["hits_B"] = cfgs[1].get("hits")
    return res


def cmd_compare_evals(args) -> int:
    dir_a, dir_b = Path(args.a), Path(args.b)
    res = compare_evals(dir_a, dir_b, args.n_boot, args.seed)
    out = new_run_dir(dir_a.parent, "compare_evals")
    setup_logging(args.log_level, out / "log.txt")
    res.update({"command": _cmdline(), "code": code_version()})
    save_json(out / "compare.json", res)
    for k, v in res["metrics"].items():
        log.info("%-14s B-A = %+.5f +- %.5f  (68%% boot [%+.5f, %+.5f], n=%d)", k, v["diff"], v["stderr"], v["lo68"], v["hi68"], v["n"])
    print(out)
    return 0


def _seeds(text: str) -> List[int]:
    """'0-19' -> 0..19;  '0,3,7' -> [0, 3, 7]."""
    if "-" in text and "," not in text:
        a, b = text.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in text.split(",") if x.strip()]


def _score(args) -> ScoreConfig:
    """The objective: S as specified, or a sensitivity variant (--score-eff / --k-time / --dm-floor)."""
    return ScoreConfig(efficiency_key=args.score_eff, k_time=args.k_time, dm_floor=args.dm_floor)


def _label_with_score(label: str, sc: ScoreConfig) -> str:
    return f"{label}_{sc.tag()}" if sc.tag() and not label.endswith(sc.tag()) else label


def _landscape_folder(text: str) -> Path:
    if str(text) == "latest":                           # newest finished landscape (HTCondor DAG)
        from .utils import latest_dir
        land = latest_dir(runs_root(), "landscape.npz")
        if land is None:
            raise SystemExit(f"no folder with landscape.npz in {runs_root()}")
        return land
    return Path(text)


def cmd_tune_objectives(args) -> int:
    from .tuning.landscape import objective_sensitivity
    land = _landscape_folder(args.landscape)
    out = command_dir(land, "objectives")
    setup_logging(args.log_level, out / "log.txt")
    res = objective_sensitivity(land)
    res.update({"command": _cmdline(), "code": code_version()})
    save_json(out / "objectives.json", res)
    for name, o in res["objectives"].items():
        b = o["optimum"]
        log.info("%-24s optimum %5d  eff %.4f DM %.4f TrackML %.4f t %.3f | default rank %5d, S-optimum rank %5d",
                 name, b["index"], b["efficiency"], b["efficiency_dm"], b["trackml_score"], b["time_proxy"],
                 o["rank_of_default"], o["rank_of_S_optimum"])
    for name, b in res["edge_reduction_at_no_physics_cost"].items():
        log.info("smallest t with %-26s t %.3f  DM %.4f TrackML %.4f", name, b["time_proxy"], b["efficiency_dm"],
                 b["trackml_score"])
    print(out)
    return 0


def cmd_tune_landscape(args) -> int:
    from .tuning.landscape import compute_landscape, verify_landscape
    model_dir, model, data_dir = _load_model_and_data(args)
    if args.out:                                   # resume an existing landscape (never a new folder elsewhere)
        from .datasets import ensure_inside_work
        out = ensure_inside_work(Path(args.out), "--out")
        if not (out / "spec.json").exists():
            raise SystemExit(f"--out {out} is not an existing landscape folder (no spec.json)")
    else:
        out = new_run_dir(runs_root(), args.label)
    setup_logging(args.log_level, out / "log.txt")
    append_jsonl(out / "commands.jsonl", {"command": _cmdline(), "time": stamp(),
                                          "code": code_version()})
    targets = _targets(args, model)
    s = compute_landscape(model_dir, data_dir, out, args.split, args.max_events, targets, args.workers)
    o, d = s["optimum"], s["default"]
    log.info("QX1 optimum S=%.5f (eff %.4f fake %.4f dup %.4f t %.3f) %s; %d exact tie(s)", o["score"],
             o["efficiency"], o["fakerate"], o["duplicaterate"], o["time_proxy"], o["params"], s["n_exact_ties"])
    log.info("default S=%.5f, rank %d of %d; bootstrap: the optimum stays the argmax in %.0f%% of resamples",
             d["score"], d["rank"], s["n_configurations"], 100 * s["bootstrap"]["argmax_is_optimum_fraction"])
    if not args.no_verify:
        v = verify_landscape(out, args.n_verify, args.workers)
        log.info("verification with the reference implementation: %d configurations, max |difference| %.2e -> %s",
                 v["n_checked"], v["max_abs_difference"], "PASSED" if v["passed"] else "FAILED")
        if not v["passed"]:
            return 1
    print(out)
    return 0


def cmd_tune_scaling(args) -> int:
    from .tuning.scaling import resolve_source, run_study
    if args.out:                                   # resume an existing study (never a new folder elsewhere)
        from .datasets import ensure_inside_work
        out = ensure_inside_work(Path(args.out), "--out")
        if not (out / "spec.json").exists():
            raise SystemExit(f"--out {out} is not an existing scaling folder (no spec.json)")
        source = Path(load_json(out / "spec.json")["source"])
    else:
        source = resolve_source(args.source, runs_root())
        out = new_run_dir(runs_root(), args.label)
    setup_logging(args.log_level, out / "log.txt")
    logging.getLogger("acts_qaoa").setLevel(logging.WARNING)
    append_jsonl(out / "commands.jsonl", {"command": _cmdline(), "time": stamp(), "code": code_version()})
    validate = None
    if not args.no_validate:
        from .tuning.runner import GNNRunner
        model_dir, model, data_dir = _load_model_and_data(args)
        targets = _targets(args, model)

        def validate(todo):
            """The picks of every size on the real GNN objective (after all simulations, to keep memory free)."""
            runner = GNNRunner(model_dir, data_dir, out, TRACKING_PARAM_NAMES, args.split, args.max_events, targets,
                               workers=args.workers)
            rows = []
            try:
                for name, prm in todo:
                    rec = runner.evaluate(prm, tag=name)
                    m = rec.get("metrics") or {}
                    rows.append({"config": name, "params": prm, "objective": rec.get("objective"),
                                 **{k: m.get(k) for k in ("efficiency", "efficiency_dm", "fakerate", "duplicaterate",
                                                          "time_proxy", "trackml_score")}})
                    log.info("[qx6] %-12s S %s  %s", name, rec.get("objective"), prm)
            finally:
                runner.close()
            return rows
    sizes = [int(x) for x in args.sizes.split(",") if x.strip()]
    run_study(source, out, sizes, args.shots, args.seed, args.k_low, args.sparse_top_k, args.threads,
              args.max_memory_gb, not args.no_noise, args.noise_backend, args.precision, validate,
              extra={"command": _cmdline(), "code": code_version()})
    print(out)
    return 0


def cmd_ibm_status(args) -> int:
    from .tuning.phase1 import ibm_status
    setup_logging(args.log_level)
    s = ibm_status(args.creds)
    if s["usage"]:
        u = s["usage"]
        log.info("QPU usage in the current window: %s s of %s s used, %s s remaining (%s)",
                 u.get("usage_consumed_seconds"), u.get("usage_limit_seconds"), u.get("usage_remaining_seconds"),
                 u.get("usage_period"))
    for r in s["backends"]:
        if "median_2q_error" in r:
            log.info("%-15s %3d qubits  operational %-5s  queue %5d  median 2q error %.2e  median readout %.2e  (%s)",
                     r["name"], r["n_qubits"], r["operational"], r["queue"], r["median_2q_error"],
                     r["median_readout_error"], r["calibrated"])
        else:
            log.info("%-15s %3d qubits  details not available (%s)", r["name"], r["n_qubits"], r.get("error"))
    log.info("lowest median two-qubit error: %s", s["best"])
    return 0


def cmd_hw_validate(args) -> int:
    from .datasets import ensure_inside_work
    from .tuning.hardware import validate_hardware
    from .tuning.phase1 import make_run
    from .tuning.runner import EVENT_SEED_STRIDE, GNNRunner
    from .tuning.space import full_encoding, sub_encoding
    run_dir = ensure_inside_work(Path(args.run_dir), "--run-dir")
    bits = load_json(run_dir / "run_config.json")["bits"]
    enc = full_encoding() if bits == full_encoding().bits else sub_encoding(full_encoding())
    model_dir, model, data_dir = _load_model_and_data(args)
    out = command_dir(run_dir, "hw_validate")
    setup_logging(args.log_level, out / "log.txt")
    logging.getLogger("acts_qaoa").setLevel(logging.WARNING)
    append_jsonl(out / "commands.jsonl", {"command": _cmdline(), "time": stamp(), "code": code_version()})
    runner = GNNRunner(model_dir, data_dir, out, enc.space.names, args.split, args.max_events, _targets(args, model),
                       workers=args.workers)
    try:
        run = make_run(run_dir, enc.space, enc.bits, runner, runner.loop_n_events, TRACKING_DEFAULTS)
        validate_hardware(run, out, args.top_k, args.min_distance, args.seed, EVENT_SEED_STRIDE, TRACKING_DEFAULTS)
    finally:
        runner.close()
    print(out)
    return 0


def cmd_hw_analyze(args) -> int:
    from .datasets import ensure_inside_work
    from .hw.analysis import analyze_run
    run_dir = ensure_inside_work(Path(args.run_dir), "--run-dir")
    out = command_dir(run_dir, "hw_analyze")
    setup_logging(args.log_level, out / "log.txt")
    logging.getLogger("acts_qaoa").setLevel(logging.WARNING)
    append_jsonl(out / "commands.jsonl", {"command": _cmdline(), "time": stamp(), "code": code_version()})
    if args.readout_tags and not args.readout:
        raise SystemExit("hw-analyze: --readout-tags needs --readout <readout_cal.json>")
    analyze_run(run_dir, out, args.tags, args.kmax, Path(args.readout) if args.readout else None, args.readout_tags,
                args.prefer, args.readout_method)
    print(out)
    return 0


def _hw_config(args, **over):
    """HwConfig from the command line on top of the environment (HSF2_IBM_CREDS, HSF2_QPU_LEDGER, HSF2_QPU_BUDGET,
    HSF2_IBM_BACKEND); the credentials file is only passed on as a path."""
    from .hw.config import HwConfig
    kw = dict(backend=args.backend, creds_file=args.creds, ledger=args.ledger, budget_s=args.budget, mode=args.mode,
              primitive=args.primitive, shots=args.shots, fake_backend=args.fake_backend,
              real_target=args.real_target or None, transpile_seeds=args.seeds, reserve_s=args.reserve)
    kw.update(over)
    return HwConfig.from_env(**kw)


def _hw_start(args, command: str):
    """Output folder <run>/outputs/<stamp>_<command>/ with log, commands.jsonl and the arguments of this call."""
    from .datasets import ensure_inside_work
    run_dir = ensure_inside_work(Path(args.run_dir), "run_dir")
    out = command_dir(run_dir, command)
    setup_logging(args.log_level, out / "log.txt")
    logging.getLogger("acts_qaoa").setLevel(logging.WARNING)
    creds = getattr(args, "creds", None)
    cmd = _cmdline()
    if creds:                                     # record the credentials file by its name only, never its path
        cmd = cmd.replace(str(creds), Path(creds).name)
    rec = {"command": cmd, "time": stamp(), "code": code_version()}
    append_jsonl(out / "commands.jsonl", rec)
    save_json(out / "config.json", dict(rec, args={k: (Path(v).name if k == "creds" and v else v)
                                                   for k, v in vars(args).items() if k != "func"}))
    return run_dir, out


def _hw_guard(fn):
    """Run a hardware command body; a refused budget or an unconfirmed submit is a normal stop (exit code 1)."""
    from .hw.ledger import BudgetError
    from .hw.run import SubmitAborted
    try:
        return fn()
    except (BudgetError, SubmitAborted) as exc:
        log.error("%s", exc)
        return 1


def cmd_hw_run(args) -> int:
    from .hw.run import run_job
    run_dir, out = _hw_start(args, "hw_run")
    cfg = _hw_config(args)

    def body():
        res = run_job(run_dir, args.setting, args.K or None, cfg, args.step, args.gauges, _cmdline(), wait=not args.no_wait,
                      server_dry_run=args.server_dry_run, seed=args.sim_seed)
        log.info("job folder %s (state %s)", res["job_dir"], res["manifest"]["state"])
        print(res["job_dir"])
        return 0
    return _hw_guard(body)


def cmd_hw_ladder(args) -> int:
    from .hw.run import run_ladder
    run_dir, out = _hw_start(args, "hw_ladder")
    cfg = _hw_config(args)
    ks = [k or None for k in args.K]

    def body():
        rows = run_ladder(run_dir, ks, args.settings, cfg, args.gauges, args.step, _cmdline(), out, args.readout,
                          seed=args.sim_seed)
        for r in rows:
            log.info("K=%-5s %-2s %-10s billed %s s  %s", r["K"] or "full", r["setting"], r["state"], r["billed_s"], r["job_dir"])
        print(out)
        return 0
    return _hw_guard(body)


def cmd_hw_readout(args) -> int:
    from .hw.run import apply_readout, run_readout
    run_dir, out = _hw_start(args, "hw_readout")
    cfg = _hw_config(args)

    def body():
        if args.apply:
            if not args.cal:
                raise SystemExit("--apply needs --cal <readout_cal.json>")
            res = apply_readout(Path(args.apply), Path(args.cal), args.prefer)
            log.info("readout correction by %s; mean energies %s", res["mitigation"], res.get("mean_energy"))
            print(Path(args.apply) / "readout_mitigated.json")
            return 0
        if not args.from_job:
            raise SystemExit("give --from-job <job folder> (calibrate its qubits) or --apply <job folder> --cal <file>")
        res = run_readout(run_dir, Path(args.from_job), cfg, args.step, _cmdline(), args.cal_shots, seed=args.sim_seed,
                          wait=not args.no_wait)
        log.info("readout calibration folder %s (state %s)", res["job_dir"], res["manifest"]["state"])
        if res["manifest"]["state"] == "submitted":
            log.info("submitted without waiting; later: python -m hsf2 hw-retrieve %s", res["manifest"]["job_id"])
        print(res["job_dir"])
        return 0
    return _hw_guard(body)


def cmd_hw_retrieve(args) -> int:
    from .datasets import ensure_inside_work
    from .hw.components import retrieve_any            # component jobs of hw-components; every other job: run.retrieve_job
    setup_logging(args.log_level)
    logging.getLogger("acts_qaoa").setLevel(logging.WARNING)
    cfg = _hw_config(args)
    run_dir = ensure_inside_work(Path(args.run_dir), "--run-dir") if args.run_dir else None
    root = ensure_inside_work(Path(args.search_root), "--search-root") if args.search_root else runs_root()
    res = retrieve_any(args.job_id, cfg, run_dir, None if run_dir else root)
    log.info("job %s: outputs rebuilt in %s (billed %s)", args.job_id, res["job_dir"], res["manifest"].get("billed"))
    print(res["job_dir"])
    return 0


def cmd_hw_ledger(args) -> int:
    from .hw.config import HwConfig
    from .hw.ledger import HwLedger
    setup_logging(args.log_level)
    cfg = HwConfig.from_env(creds_file=args.creds, ledger=args.ledger, budget_s=args.budget)
    led = HwLedger(cfg.ledger_path(), cfg.budget_s, creds_file=cfg.creds_file)
    print(led.table())
    if args.service:                       # IBM's own figure (metadata only, no QPU time)
        from .hw import session
        svc = session.get_service(cfg.creds_file)
        print(f"IBM reports: consumed {session.account_consumed(svc)} s, remaining {session.service_remaining(svc)} s")
    return 0


def _components_selection(args, run_dir):
    """(all components of a triplet run folder, the selected ones, the number of components in the size range, the number of
    hard ones among them) for the selection options of hw-components."""
    from .hw import components as C
    comps = C.load_components(run_dir)
    try:
        sel = C.select_components(comps, args.nmin, args.nmax, args.hard, args.benchmark, args.seed, args.max)
    except ValueError as exc:
        raise SystemExit(f"hw-components: {exc}") from None
    pool = [c for c in comps if args.nmin <= c.n <= args.nmax]
    return comps, sel, len(pool), sum(c.hard for c in pool)


def cmd_hw_components(args) -> int:
    from .datasets import ensure_inside_work
    from .hw import components as C
    setup_logging(args.log_level)
    _, sel, pool, pool_hard = _components_selection(args, ensure_inside_work(Path(args.run_dir), "run_dir"))
    if args.list:                                   # read-only: nothing is written
        print(C.selection_table(sel, pool, pool_hard))
        return 0
    if args.per_job < 1:
        raise SystemExit("hw-components: --per-job must be >= 1")
    if args.top_k < 0:
        raise SystemExit("hw-components: --top-k must be >= 0 (0: all couplings)")
    if args.no_wait and -(-len(sel) // args.per_job) > 1:
        raise SystemExit(f"hw-components: --no-wait needs a single job: --per-job must be at least {len(sel)}, the number of "
                         "selected components")
    run_dir, out = _hw_start(args, "hw_components")
    cfg = _hw_config(args)
    params = {"nmin": args.nmin, "nmax": args.nmax, "hard": args.hard, "benchmark": args.benchmark, "seed": args.seed,
              "max": args.max, "pool": pool, "pool_hard": pool_hard}
    save_json(out / "selection.json", {"params": params, "components": [c.to_record(run_dir) for c in sel]})
    log.info("selection:\n%s", C.selection_table(sel, pool, pool_hard))

    cmd = _cmdline()
    if args.creds:                                  # the manifests record the credentials file by its name only
        cmd = cmd.replace(str(args.creds), Path(args.creds).name)

    def body():
        res = C.run_components(run_dir, sel, args.setting, cfg, args.per_job, args.top_k or None, args.step, cmd, out,
                               wait=not args.no_wait, seed=args.sim_seed, sim_method=args.sim_method,
                               selection=dict(params, components=[c.name for c in sel]), analysis_seed=args.seed)
        for r in res["jobs"]:
            m = r["manifest"]
            log.info("job %d/%d %s: %d PUBs, state %s, job id %s, estimated %.1f s, billed %s s", m["job_index"], m["n_jobs"],
                     r["job_dir"].name, len(m["pubs"]), m["state"], m["job_id"], m["estimate_s"],
                     (m.get("billed") or {}).get("used"))
        log.info("estimated QPU time of all jobs: %.1f s", res["estimate_total_s"])
        print(out)
        return 0
    return _hw_guard(body)


def cmd_hw_components_analyze(args) -> int:
    from .datasets import ensure_inside_work
    from .hw.comp_analysis import analyze_jobs
    jobs = [ensure_inside_work(Path(p), "job folder") for p in args.job_dirs]
    if any(j.parent.name != "hw" for j in jobs):
        raise SystemExit("hw-components-analyze: give job folders <triplet run>/hw/<stamp>_comp_<setting>_<k>")
    args.run_dir = str(jobs[0].parents[1])          # the command folder goes into the run folder that holds the jobs
    if args.readout_tags and not args.readout:
        raise SystemExit("hw-components-analyze: --readout-tags needs --readout <readout_cal.json>")
    run_dir, out = _hw_start(args, "hw_components_analyze")
    analyze_jobs(jobs, out, args.seed, readout=Path(args.readout) if args.readout else None, readout_tags=args.readout_tags,
                 prefer=args.prefer)
    print(out)
    return 0


def cmd_hw_zne(args) -> int:
    from .hw import zne as Z
    if bool(args.K) == bool(args.components):
        raise SystemExit("hw-zne: give either --K (QX2 run folder) or --components (triplet run folder)")
    run_dir, out = _hw_start(args, "hw_zne")
    cfg = _hw_config(args)
    cmd = _cmdline()
    if args.creds:                                  # the manifests record the credentials file by its name only
        cmd = cmd.replace(str(args.creds), Path(args.creds).name)

    fold_mode = args.fold_gates is not None or args.from_job or args.subsets != 1
    if args.from_job:
        from .datasets import ensure_inside_work
        args.from_job = ensure_inside_work(Path(args.from_job), "job folder")
    lams = [int(x) if float(x).is_integer() else float(x) for x in args.lambdas]    # whole values stay ints (file names, legacy checks)

    def body():
        res = Z.run_zne(run_dir, args.K or [], args.components or [], args.setting, cfg, lams,
                        args.per_job or (Z.FOLD_MAX_PUBS_PER_JOB if fold_mode else Z.MAX_PUBS_PER_JOB),
                        args.top_k or None, args.step, cmd, out, seed=args.sim_seed, sim_method=args.sim_method,
                        fold_gates=args.fold_gates, subsets=args.subsets, from_job=args.from_job, fold_seed=args.fold_seed,
                        randomizations=args.randomizations, wait=not args.no_wait)
        for r in res["jobs"]:
            m = r["manifest"]
            log.info("job %d/%d %s: %d PUBs, state %s, job id %s, estimated %.1f s, billed %s s", m["job_index"], m["n_jobs"],
                     r["job_dir"].name, len(m["pubs"]), m["state"], m["job_id"], m["estimate_s"],
                     (m.get("billed") or {}).get("used"))
        log.info("estimated QPU time of all jobs: %.1f s", res["estimate_total_s"])
        print(out)
        return 0
    return _hw_guard(body)


def cmd_hw_zne_analyze(args) -> int:
    from .datasets import ensure_inside_work
    from .hw.zne import analyze_zne
    jobs = [ensure_inside_work(Path(p), "job folder") for p in args.job_dirs]
    if any(j.parent.name != "hw" for j in jobs):
        raise SystemExit("hw-zne-analyze: give job folders <run>/hw/<stamp>_zne_<setting>_<k>")
    args.run_dir = str(jobs[0].parents[1])
    run_dir, out = _hw_start(args, "hw_zne_analyze")
    analyze_zne(jobs, out, args.boot_n, args.seed, Path(args.readout) if args.readout else None)
    print(out)
    return 0


def cmd_hw_qx4(args) -> int:
    from .datasets import ensure_inside_work
    from .hw import qx4hw as Q
    try:
        events, lams, lams_other = Q.parse_events(args.event), Q.parse_lams(args.lams), Q.parse_lams(args.lams_other)
    except ValueError as exc:
        raise SystemExit(f"hw-qx4: {exc}") from None
    # the folder of the QX4 jobs: <run>/hw/...; a new run folder <runs>/<stamp>_qx4_hw unless --run-dir names one
    args.run_dir = str(ensure_inside_work(Path(args.run_dir), "--run-dir") if args.run_dir
                       else new_run_dir(runs_root(), args.label))
    run_dir, out = _hw_start(args, "hw_qx4")
    cfg = _hw_config(args)
    cmd = _cmdline()
    if args.creds:                                  # the manifests record the credentials file by its name only
        cmd = cmd.replace(str(args.creds), Path(args.creds).name)

    def body():
        res = Q.run_qx4(run_dir, events, args.setting, cfg, lams, lams_other, args.hhl, not args.no_readout, args.cal_shots,
                        args.event_seed, args.alpha, args.beta, args.eps, args.threshold, args.hhl_clock, args.step, cmd, out,
                        seed=args.sim_seed, boot_n=args.boot_n, analysis_seed=args.seed)
        m = res["manifest"]
        log.info("job %s: %d PUBs, state %s, job id %s, estimated %.1f s (with read-out calibration %.1f s), billed %s s",
                 res["job_dir"].name, len(m["pubs"]), m["state"], m["job_id"], m["estimate_s"], res["estimate_total_s"],
                 (m.get("billed") or {}).get("used"))
        print(out)
        return 0
    return _hw_guard(body)


def cmd_hw_qx4_analyze(args) -> int:
    from .datasets import ensure_inside_work
    from .hw.qx4hw import analyse_qx4
    job = ensure_inside_work(Path(args.job_dir), "job folder")
    if job.parent.name != "hw":
        raise SystemExit("hw-qx4-analyze: give a job folder <run>/hw/<stamp>_qx4_<setting>_<k>")
    args.run_dir = str(job.parents[1])
    run_dir, out = _hw_start(args, "hw_qx4_analyze")
    analyse_qx4(job, out, Path(args.readout) if args.readout else None, args.boot_n, args.seed)
    print(out)
    return 0


def cmd_tune_compare(args) -> int:
    from .tuning.compare import noise_model_for, run_comparison
    from .tuning.landscape import LandscapeTable
    from .tuning.phase1 import TaggedView, make_run, phase1_info
    from .tuning.runner import LookupRunner
    land = _landscape_folder(args.landscape)
    args.landscape = str(land)
    sc = _score(args)
    tab = LandscapeTable(land, sc)
    out = command_dir(land, _label_with_score("compare", sc))
    setup_logging(args.log_level, out / "log.txt")
    logging.getLogger("acts_qaoa").setLevel(logging.WARNING)
    log.info("objective: %s", sc.describe())
    base = LookupRunner(tab.enc, tab.metrics_of_index, tab.n_events, score=sc)
    noise = noise_model_for(make_run(out / "noise_model", tab.enc.space, tab.enc.bits, TaggedView(base),
                                     tab.n_events, TRACKING_DEFAULTS, score=sc), args.n_ws)
    log.info("noise model: full circuit %d two-qubit gates -> F = %.3f; top-24 circuit %d -> F = %.3f",
             noise["full"]["n_2q"], noise["full"]["fidelity"], noise["top24"]["n_2q"], noise["top24"]["fidelity"])
    methods = [m.strip() for m in args.methods.split(",")]
    summ = run_comparison(out, base, tab.enc, TRACKING_DEFAULTS, tab.n_events, _seeds(args.seeds), methods,
                          args.budget, args.n_ws, args.top_k, args.shots, args.min_distance, noise,
                          truth=lambda i: float(tab.score[i]), best_true=tab.best,
                          extra={"landscape": str(args.landscape), "objective": "exact landscape (QX1)",
                                 "phase1": phase1_info(), "code": code_version(),
                                 "command": _cmdline()}, score=sc)
    for m, d in summ["methods_summary"].items():
        log.info("  %-24s regret median %.5f [q25 %.5f, q75 %.5f]  P(optimum found) %.2f", m, d["regret"]["median"],
                 d["regret"]["q25"], d["regret"]["q75"], d["p_found_optimum"])
    print(out)
    return 0


def cmd_tune_loop(args) -> int:
    from .tuning.compare import noise_model_for, run_comparison
    from .tuning.phase1 import TaggedView, make_run, phase1_info
    from .tuning.runner import EVENT_SEED_STRIDE, GNNRunner
    from .tuning.space import describe_space, full_encoding
    model_dir, model, data_dir = _load_model_and_data(args)
    sc = _score(args)
    out = new_run_dir(runs_root(), _label_with_score(args.label, sc))
    setup_logging(args.log_level, out / "log.txt")
    logging.getLogger("acts_qaoa").setLevel(logging.WARNING)
    log.info("objective: %s", sc.describe())
    enc = full_encoding()
    targets = _targets(args, model)
    base = GNNRunner(model_dir, data_dir, out, enc.space.names, args.split, args.max_events, targets,
                     score=sc, workers=args.workers)
    n_ev = base.loop_n_events
    methods = [m.strip() for m in args.methods.split(",")]
    seeds = _seeds(args.seeds)
    save_json(out / "config.json", {"command": _cmdline(), "model_dir": str(model_dir),
                                    "data_dir": str(data_dir), "split": args.split,
                                    "events": base.events_for(None, None), "targets": targets.to_dict(),
                                    "space": describe_space(enc), "methods": methods, "seeds": seeds,
                                    "budget": args.budget, "n_ws": args.n_ws, "top_k": args.top_k,
                                    "score_definition": sc.describe(), "score_config": sc.to_dict(),
                                    "phase1": phase1_info(), "model_fp": base.model_fp, "code": code_version()})
    try:
        noise = noise_model_for(make_run(out / "noise_model", enc.space, enc.bits, TaggedView(base), n_ev,
                                         TRACKING_DEFAULTS, score=sc), args.n_ws)
        log.info("noise model: full circuit %d two-qubit gates -> F = %.3f; top-24 circuit %d -> F = %.3f",
                 noise["full"]["n_2q"], noise["full"]["fidelity"], noise["top24"]["n_2q"], noise["top24"]["fidelity"])
        run_comparison(out, base, enc, TRACKING_DEFAULTS, n_ev, seeds, methods, args.budget, args.n_ws, args.top_k,
                       args.shots, args.min_distance, noise,
                       extra={"objective": f"real GNN objective on split {args.split} ({n_ev} events)",
                              "phase1": phase1_info(), "code": code_version()}, score=sc)
        _confirm_picks(base, out, EVENT_SEED_STRIDE)
    finally:
        base.close()
    print(out)
    return 0


def _confirm_picks(base, out: Path, stride: int) -> None:
    """The pick of every method and seed, and the default, on every held-out confirmation split."""
    res = load_json(out / "results.json")["results"]
    picks = {"default": dict(TRACKING_DEFAULTS)}
    for r in res:
        recs = [x for x in r["records"] if x["objective"] is not None]
        if recs:
            picks[f"{r['method']}/seed{r['seed']}"] = max(recs, key=lambda x: x["objective"])["params"]
    ks = sorted(int(k[len("confirm"):]) for k in base.splits if k.startswith("confirm") and base.splits[k])
    rows = []
    for name, prm in picks.items():
        tune = base.evaluate(prm, tag="pick")
        row = {"config": name, "params": prm, "tune_objective": tune["objective"], "confirm": {}}
        for k in ks:
            rec = base.evaluate(prm, tag="confirm", seed=k * stride)
            row["confirm"][f"confirm{k}"] = {"objective": rec["objective"], "n_events": rec["n_events"],
                                             **{m: rec["metrics"].get(m) for m in
                                                ("efficiency", "fakerate", "duplicaterate", "time_proxy",
                                                 "trackml_score", "efficiency_dm")}}
        vals = [v["objective"] for v in row["confirm"].values() if v["objective"] is not None]
        row["confirm_mean"] = float(np.mean(vals)) if vals else None
        rows.append(row)
        log.info("[confirm] %-34s tune %.5f  confirm %s", name, tune["objective"] or float("nan"),
                 " ".join(f"{v['objective']:.5f}" for v in row["confirm"].values() if v["objective"] is not None))
    save_json(out / "confirm.json", {"confirm_splits": [f"confirm{k}" for k in ks], "rows": rows})


def cmd_select(args) -> int:
    from .qubo.selection import select
    from .tracking.build import build_candidates
    from .tracking.metrics import event_counts, summarize
    from .tracking.pipeline import GNNPipeline
    from .tuning.phase1 import fidelity_from_2q
    model_dir, model, data_dir = _load_model_and_data(args)
    out = command_dir(model_dir, f"select_{args.split}")
    setup_logging(args.log_level, out / "log.txt")
    params = TrackingParams(**parse_sets(args.set))
    targets = _targets(args, model)
    methods = [m.strip() for m in args.methods.split(",")]
    save_json(out / "config.json", {"command": _cmdline(), "tracking": params.to_dict(),
                                    "targets": targets.to_dict(), "methods": methods, "max_vars": args.max_vars,
                                    "penalty_factor": args.penalty_factor, "shots": args.shots,
                                    "events": splits(data_dir)[args.split][: args.max_events or None],
                                    "model_dir": str(model_dir), "data_dir": str(data_dir), "code": code_version()})
    pipe = GNNPipeline(model, targets)
    fid_cache: Dict[tuple, dict] = {}

    def fidelity_of(qubo) -> float:
        """Measured-fidelity model at the transpiled (fake ibm_fez) two-qubit gate count of this component."""
        from .tuning.phase1 import transpiled_2q
        iu, ju = np.nonzero(np.triu(qubo.W, 1))
        key = (qubo.n, tuple(zip(iu.tolist(), ju.tolist())))
        if key not in fid_cache:
            ising = qubo.to_ising()
            fid_cache[key] = transpiled_2q(ising, ising.scale(), seeds=(0,))
        return fid_cache[key]["fidelity_model"]

    rows, per_method = [], {m: [] for m in methods}
    comp_sizes: List[int] = []
    amp: Dict[str, list] = {}
    for ev in load_split(data_dir, args.split, args.max_events):
        es, g = pipe.prepare(ev)
        keep, sc, _ = pipe.scores(ev, params)
        cands = build_candidates(es.n_hits, g.src[keep], g.dst[keep], sc, params)
        for m in methods:
            res = select(cands, es.n_hits, m, args.max_vars, args.penalty_factor, args.seed, args.shots,
                         fidelity_of if m == "qaoa-noise" else None)
            tracks = [cands.tracks[i] for i in res["selected"]]
            c = event_counts(es, tracks, targets)
            per_method[m].append(c)
            st = res["stats"]
            if "ground_amplification" in st:
                amp.setdefault(m, []).extend(zip(st["component_n_for_amp"], st["ground_amplification"]))
            rows.append({"event_id": ev.event_id, "method": m, **{k: v for k, v in st.items() if k not in
                                                                   ("component_sizes", "p_ground", "ground_amplification",
                                                                    "component_n_for_amp")},
                         "mean_p_ground": float(np.mean(st["p_ground"])) if st["p_ground"] else None,
                         **{k: c[k] for k in ("n_tracks", "n_fake", "n_duplicate", "n_targets_found")}})
            if m == methods[0]:
                comp_sizes += st["component_sizes"]
            log.info("%s %-10s cands %d comps %d (too large %d) optimal %d/%d feasible %d weight %.1f %.1f s",
                     ev.event_id, m, st["n_candidates"], st["n_components"], st["n_too_large"], st["optimal"],
                     st["n_fit"], st["feasible"], st["selected_weight"], st["seconds"])
    import pandas as pd
    df = pd.DataFrame(rows)
    save_csv(df, out / "per_event.csv")
    sizes = np.array(comp_sizes)
    multi = sizes[sizes > 1]
    summary = {"components": {"n": int(len(sizes)), "n_with_conflicts": int(len(multi)),
                              "fraction_fit_max_vars": float(np.mean(multi <= args.max_vars)) if len(multi) else None,
                              "candidates_in_fitting_components": float(sizes[sizes <= args.max_vars].sum() / max(sizes.sum(), 1)),
                              "size_quantiles": {q: float(np.quantile(multi, q)) for q in (0.5, 0.9, 0.99)} if len(multi) else {},
                              "max_size": int(sizes.max()) if len(sizes) else 0},
               "methods": {}}
    sc_cfg = ScoreConfig()
    for m in methods:
        mm = summarize(per_method[m])
        g = df[df["method"] == m]
        summary["methods"][m] = {**{k: mm[k] for k in ("efficiency", "efficiency_dm", "fakerate", "duplicaterate",
                                                        "trackml_score", "n_tracks")},
                                 "physics_score": sc_cfg.physics(mm),
                                 "optimal_components": int(g["optimal"].sum()), "fit_components": int(g["n_fit"].sum()),
                                 "feasible_components": int(g["feasible"].sum()),
                                 "selected_weight": float(g["selected_weight"].sum()),
                                 "mean_p_ground": float(g["mean_p_ground"].dropna().mean()) if g["mean_p_ground"].notna().any() else None,
                                 "seconds": float(g["seconds"].sum())}
        d = summary["methods"][m]
        log.info("%-10s eff %.4f fake %.4f dup %.4f TrackML %.4f | optimal %d/%d components, weight %.1f", m,
                 d["efficiency"], d["fakerate"], d["duplicaterate"], d["trackml_score"], d["optimal_components"],
                 d["fit_components"], d["selected_weight"])
    for m, pairs in amp.items():          # QAOA ground-state probability / uniform, by component size
        a = np.array(pairs, float)
        summary["methods"][m]["ground_amplification_by_size"] = {
            int(n): {"n_components": int((a[:, 0] == n).sum()), "median": float(np.median(a[a[:, 0] == n, 1]))}
            for n in np.unique(a[:, 0]).astype(int)}
    if fid_cache:
        summary["noise_model"] = {"n_transpiled_structures": len(fid_cache),
                                  "fidelity_mean": float(np.mean([v["fidelity_model"] for v in fid_cache.values()])),
                                  "n_2q_mean": float(np.mean([v["n_2q"] for v in fid_cache.values()]))}
    save_json(out / "summary.json", summary)
    print(out)
    return 0


def cmd_triplet(args) -> int:
    from .data.event import select_hits
    from .data.trackml import LAYER_INDEX
    from .qubo.triplet import TRACKML_BARREL, run_study
    from ._phase1 import ensure_phase1
    ensure_phase1()
    data_dir = resolve_data(args.data)
    out = new_run_dir(runs_root(), args.label)
    setup_logging(args.log_level, out / "log.txt")
    events = load_split(data_dir, args.split, args.max_events)
    sectors = [(float(i) * args.dphi, args.dphi) for i in range(args.n_sectors)]
    fracs = [float(f) for f in args.fracs.split(",")]
    methods = [m.strip() for m in args.methods.split(",")]
    save_json(out / "config.json", {"command": _cmdline(), "data_dir": str(data_dir),
                                    "split": args.split, "events": [e.event_id for e in events], "sectors": sectors,
                                    "fracs": fracs, "methods": methods, "barrel_layers": list(TRACKML_BARREL),
                                    "code": code_version()})
    comp_rows = []
    (out / "components").mkdir()
    rows = run_study(events, TRACKML_BARREL, sectors, fracs, methods, args.max_vars, args.seed,
                     component_rows=comp_rows, export_dir=out / "components")
    df = _df(rows)
    save_csv(df.drop(columns=["solved_by"], errors="ignore"), out / "triplet.csv")
    save_csv(_df(comp_rows), out / "components.csv")
    summ = {}
    for (f, m), g in df.groupby(["frac", "method"], sort=False):
        D = g[["D_true", "D_rec", "D_matched", "D_oa"]].sum()
        summ[f"frac{f}/{m}"] = {"efficiency": float(D["D_matched"] / max(D["D_true"], 1)),
                                "purity": float(D["D_matched"] / max(D["D_rec"] - D["D_oa"], 1)),
                                "n_triplets": int(g["n_triplets"].sum()),
                                "triplets_in_components_le24": float(g["triplets_in_components_le24"].mean()),
                                "max_component": int(g["max_component"].max()), "seconds": float(g["seconds"].sum())}
        log.info("frac %.2f %-13s doublet eff %.4f purity %.4f | triplets %d, in components <= 24: %.3f, max %d",
                 f, m, summ[f"frac{f}/{m}"]["efficiency"], summ[f"frac{f}/{m}"]["purity"],
                 summ[f"frac{f}/{m}"]["n_triplets"], summ[f"frac{f}/{m}"]["triplets_in_components_le24"],
                 summ[f"frac{f}/{m}"]["max_component"])
    save_json(out / "summary.json", summ)
    print(out)
    return 0


def cmd_hhl(args) -> int:
    from .qubo.linear import run_study
    out = new_run_dir(runs_root(), args.label)
    setup_logging(args.log_level, out / "log.txt")
    sizes = [tuple(int(v) for v in x.split("x")) for x in args.sizes.split(",")]
    seeds = list(range(args.seeds))
    save_json(out / "config.json", {"command": _cmdline(), "sizes": sizes, "seeds": seeds,
                                    "alpha": args.alpha, "beta": args.beta, "eps": args.eps, "T": args.threshold,
                                    "noise_frac": args.noise_frac, "sigma": args.sigma, "n_clock": args.n_clock,
                                    "code": code_version()})
    res = run_study(sizes, seeds, args.alpha, args.beta, args.eps, args.threshold, noise_frac=args.noise_frac,
                    sigma=args.sigma, circuit_max_qubits=args.circuit_max_qubits, n_clock=args.n_clock)
    df = _df(res["rows"])
    save_csv(df, out / "systems.csv")
    save_csv(_df(res["circuits"]), out / "circuits.csv")
    agg = df.groupby(["n_particles", "n_layers"]).mean(numeric_only=True).reset_index()
    save_csv(agg, out / "systems_mean.csv")
    for _, r in agg.iterrows():
        log.info("%2d particles x %d layers: %5.0f segments  kappa %.2f | classical eff %.3f pur %.3f | 1-bit (10N shots)"
                 " eff %.3f pur %.3f | HHL ideal (10N) eff %.3f pur %.3f", r["n_particles"], r["n_layers"],
                 r["n_segments"], r.get("condition_number", float("nan")), r["classical_efficiency"],
                 r["classical_purity"], r.get("onebit_eff_shots10N", float("nan")), r.get("onebit_pur_shots10N", float("nan")),
                 r.get("hhl-ideal_eff_shots10N", float("nan")), r.get("hhl-ideal_pur_shots10N", float("nan")))
    for c in res["circuits"]:
        log.info("circuit N=%d (%d system qubits, %d Pauli terms): 1-bit %d qubits %d CX | HHL(n_c=%d) %d qubits %d CX",
                 c["n_segments"], c["system_qubits"], c["pauli_terms"], c["onebit_qubits"], c["onebit_cx"],
                 c["n_clock"], c["hhl_qubits"], c["hhl_cx"])
    print(out)
    return 0


def cmd_qgnn(args) -> int:
    import torch
    from .data.event import select_hits
    from .model.quantum_head import run_study
    model_dir, model, data_dir = _load_model_and_data(args)
    out = command_dir(model_dir, "qgnn")
    setup_logging(args.log_level, out / "log.txt")
    if args.threads:
        torch.set_num_threads(args.threads)
    sel = model.hit_selection
    tr = [select_hits(e, sel) for e in load_split(data_dir, "train", args.train_events)]
    te = [select_hits(e, sel) for e in load_split(data_dir, args.split, args.test_events)]
    qubits = tuple(int(q) for q in args.qubits.split(","))
    save_json(out / "config.json", {"command": _cmdline(), "train_events": [e.event_id for e in tr],
                                    "test_events": [e.event_id for e in te], "qubits": qubits, "layers": args.layers,
                                    "n_per_class": args.n_per_class, "epochs": args.epochs, "seed": args.seed,
                                    "model_dir": str(model_dir), "code": code_version()})
    res = run_study(model, tr, te, qubits, args.layers, args.n_per_class, args.epochs, args.seed)
    save_json(out / "summary.json", res)
    for key, d in res.items():
        if key == "interaction_network":
            log.info("%-16s AUC %.4f (%d parameters)", key, d["auc"], d["parameters"])
            continue
        log.info("%-16s quantum AUC %.4f (%d params, %d qubits) | logistic %.4f (%d) | MLP %.4f (%d)", key,
                 d["quantum"]["auc"], d["quantum"]["parameters"], d["quantum"]["qubits"], d["logistic"]["auc"],
                 d["logistic"]["parameters"], d["mlp_same_params"]["auc"], d["mlp_same_params"]["parameters"])
    print(out)
    return 0


def cmd_report(args) -> int:
    from .report import make_all
    from .datasets import figures_root
    from .datasets import ensure_inside_work
    out = ensure_inside_work(Path(args.out), "--out") if args.out else figures_root()
    man = make_all(out, args.model, args.eval, args.landscape, args.compare, args.loop, True, args.select,
                   args.hhl, args.qgnn)
    for k, v in man.items():
        print(f"{k}: {v}")
    return 0


def cmd_info(args) -> int:
    d = Path(args.dir)
    for name in ("manifest.json", "summary.json", "graph_stats.json", "train_config.json"):
        if (d / name).exists():
            print(f"--- {name}")
            print(open(d / name).read()[:4000])
    return 0


# ---------------------------------------------------------------------------
def _add_score_args(s) -> None:
    s.add_argument("--score-eff", default="efficiency", choices=["efficiency", "efficiency_dm"],
                   help="efficiency in the objective: efficiency (S, as specified) or efficiency_dm (S_DM)")
    s.add_argument("--k-time", type=float, default=5.0, help="weight 1/k of the time term (S: 5)")
    s.add_argument("--dm-floor", type=float, default=0.0,
                   help="penalty 10 * max(0, floor - DM efficiency); 0 = none (S)")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m hsf2", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=f"hsf2 {__version__}")
    sub = p.add_subparsers(dest="cmd")

    def add(name, func, help_):
        sp = sub.add_parser(name, help=help_)
        sp.set_defaults(func=func)
        sp.add_argument("--log-level", default="INFO")
        return sp

    s = add("selftest", cmd_selftest, "end-to-end check on the synthetic detector")
    s.add_argument("--quick", action="store_true")

    s = add("synth", cmd_synth, "write a synthetic data folder")
    s.add_argument("--n-events", type=int, default=120)
    s.add_argument("--particles", type=int, default=400)
    s.add_argument("--noise-frac", type=float, default=0.05)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--label", default="toy")
    for k, v in (("n-train", 80), ("n-val", 10), ("n-tune", 10), ("n-confirm", 10), ("n-confirm-sets", 2)):
        s.add_argument(f"--{k}", type=int, default=v)

    s = add("trackml-prep", cmd_trackml_prep, "TrackML zip -> data folder")
    s.add_argument("--zip", default=None, help="path to train_1.zip, or zenodo:train_1.zip (HTTP range requests); "
                                              "default: <data root>/raw/train_1.zip")
    s.add_argument("--detectors", default="auto", help="detectors.csv for the layout check (auto: <data root>/raw/detectors.csv if present; none: skip)")
    s.add_argument("--n-events", type=int, default=0, help="default: the sum of the split sizes (680)")
    s.add_argument("--skip", type=int, default=0, help="skip the first N events of the zip")
    s.add_argument("--workers", type=int, default=1, help="parallel zip readers (useful for zenodo:)")
    s.add_argument("--label", default="trackml")
    for k, v in (("n-train", 300), ("n-val", 30), ("n-tune", 50), ("n-confirm", 100), ("n-confirm-sets", 3)):
        s.add_argument(f"--{k}", type=int, default=v)

    s = add("odd-prep", cmd_odd_prep, "ODD exports (odd_export.py) -> data folder")
    s.add_argument("--export", nargs="+", required=True, help="export folder(s) hsf2_work/odd_export/<stamp>_<label> "
                                                              "(one per seed; events sorted by seed, event number)")
    s.add_argument("--n-events", type=int, default=0, help="default: the sum of the split sizes (370)")
    s.add_argument("--positions", default="simhit", choices=["simhit", "measured"],
                   help="hit positions: simhit truth, or the digitised measurements placed with surface frames fitted "
                        "on the training events (the exports carry no surface transforms)")
    s.add_argument("--label", default="odd")
    for k, v in (("n-train", 200), ("n-val", 20), ("n-tune", 0), ("n-confirm", 50), ("n-confirm-sets", 3)):
        s.add_argument(f"--{k}", type=int, default=v)

    s = add("compare-ckf", cmd_compare_ckf, "GNN versus the exported ACTS CKF tracks on the same ODD events")
    s.add_argument("--model", default="latest")
    s.add_argument("--data", default=None, help="odd-prep data folder (default: the folder the model was trained on)")
    s.add_argument("--export", nargs="+", required=True, help="the export folder(s) the data folder was made from")
    s.add_argument("--split", default="confirm1")
    s.add_argument("--max-events", type=int, default=0)
    s.add_argument("--set", nargs="*", default=[], metavar="NAME=VALUE",
                   help=f"tracking parameters (default {TRACKING_DEFAULTS})")
    s.add_argument("--target-pt", type=float, default=1.0, help="Phase-1 target definition: pT >= 1 GeV ...")
    s.add_argument("--target-eta", type=float, default=3.0, help="... |eta| < 3 ...")
    s.add_argument("--target-min-hits", type=int, default=9, help="... >= 9 measurements")
    s.add_argument("--n-boot", type=int, default=2000, help="paired per-event bootstrap resamples")
    s.add_argument("--boot-seed", type=int, default=0)
    s.add_argument("--no-crosscheck", action="store_true", help="skip Python matching of all exported CKF tracks vs "
                                                                "performance_finding_ckf")
    s.add_argument("--crosscheck-max-events", type=int, default=0, help="0 = all events of each export")
    s.add_argument("--select", default="none", choices=["none", "ckf"],
                   help="ckf: also report both methods after the CKF's own track selection (pT >= 1 GeV, |eta| < 3, "
                        "|d0| < 4 mm, >= 7 measurements) applied to a helix refit of every track's hits")

    s = add("train", cmd_train, "train the Interaction Network")
    s.add_argument("--data", default="latest")
    s.add_argument("--level", default="S0", help="S0 (pT >= --pt hits only), S0N (S0 + noise), S1 (all hits)")
    s.add_argument("--pt", type=float, default=1.0)
    s.add_argument("--epochs", type=int, default=25)
    s.add_argument("--hidden", type=int, default=64)
    s.add_argument("--n-iter", type=int, default=3)
    s.add_argument("--mlp-layers", type=int, default=2)
    s.add_argument("--unshared", action="store_true", help="separate weights per message-passing step")
    s.add_argument("--edge-label", default="consecutive", choices=["consecutive", "pid"])
    s.add_argument("--lr", type=float, default=2e-3)
    s.add_argument("--lr-min", type=float, default=1e-4)
    s.add_argument("--max-pos-weight", type=float, default=20.0)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--threads", type=int, default=0, help="torch threads (0 = OMP_NUM_THREADS / default)")
    s.add_argument("--max-events", type=int, default=0, help="use only the first N training events")
    s.add_argument("--max-val-events", type=int, default=0)
    s.add_argument("--label", default=None)

    for name, func, help_ in (("evaluate", cmd_evaluate, "reconstruct and score one parameter set"),
                              ("scan", cmd_scan, "1-D scan of one tracking parameter")):
        s = add(name, func, help_)
        s.add_argument("--model", default="latest")
        s.add_argument("--data", default=None, help="default: the data folder the model was trained on")
        s.add_argument("--split", default="val")
        s.add_argument("--max-events", type=int, default=0)
        s.add_argument("--set", nargs="*", default=[], metavar="NAME=VALUE",
                       help=f"tracking parameters (default {TRACKING_DEFAULTS})")
        s.add_argument("--target-pt", type=float, default=None)
        s.add_argument("--target-eta", type=float, default=4.0)
        s.add_argument("--target-min-hits", type=int, default=5)
        s.add_argument("--level", default=None, choices=["S0", "S0N", "S1"],
                       help="hit selection to evaluate on (default: the model's own level)")
        if name == "scan":
            s.add_argument("--param", required=True)
            s.add_argument("--values", required=True, help="comma-separated, e.g. 0.2,0.3,0.5,0.7")

    for name, func, help_ in (("tune-landscape", cmd_tune_landscape, "QX1: exhaustive 16-bit landscape"),
                              ("tune-loop", cmd_tune_loop, "QX2: 24-bit closed loop on the real objective")):
        s = add(name, func, help_)
        s.add_argument("--model", default="latest")
        s.add_argument("--data", default=None, help="default: the data folder the model was trained on")
        s.add_argument("--split", default="tune")
        s.add_argument("--max-events", type=int, default=0)
        s.add_argument("--workers", type=int, default=4, help="worker processes (one core each)")
        s.add_argument("--target-pt", type=float, default=None)
        s.add_argument("--target-eta", type=float, default=4.0)
        s.add_argument("--target-min-hits", type=int, default=5)
        s.add_argument("--label", default="qx1_landscape" if name == "tune-landscape" else "qx2_loop")
        if name == "tune-landscape":
            s.add_argument("--out", default=None, help="existing landscape folder to resume")
            s.add_argument("--no-verify", action="store_true")
            s.add_argument("--n-verify", type=int, default=25, help="random configurations re-evaluated")
        else:
            s.add_argument("--methods", default="qubo-exact,qubo-sa,qubo-qaoa,qubo-qaoa-noise-top24,qubo-random,"
                                                "random-grid,tpe-grid,sa-grid")
            s.add_argument("--seeds", default="0-2")
            s.add_argument("--budget", type=int, default=100, help="distinct real evaluations per method and seed")
            s.add_argument("--n-ws", type=int, default=60, help="Latin-hypercube warm start of the QUBO loops")
            s.add_argument("--top-k", type=int, default=10, help="evaluations per closed-loop round")
            s.add_argument("--shots", type=int, default=4096)
            s.add_argument("--min-distance", type=int, default=2)
            _add_score_args(s)

    s = add("tune-scaling", cmd_tune_scaling, "QX6: QAOA on the same tuning problem at 16-32 qubits (Aer)")
    s.add_argument("--source", default="qx2", help="qx2 (newest finished closed loop), qx1 (newest landscape) "
                                                   "or a folder: the real evaluations the surrogate is fitted on")
    s.add_argument("--sizes", default="16,20,24,26,28,30,32")
    s.add_argument("--shots", type=int, default=1024)
    s.add_argument("--k-low", type=int, default=1024, help="P(K lowest states) is recorded for this K")
    s.add_argument("--sparse-top-k", type=int, default=24, help="couplings kept in the hardware circuit (0: none)")
    s.add_argument("--threads", type=int, default=0, help="Aer OpenMP threads (0: Aer's default)")
    s.add_argument("--max-memory-gb", type=float, default=0.0, help="skip sizes whose statevector would not fit")
    s.add_argument("--noise-backend", default="ibm_fez", help="fake backend for the two-qubit gate counts")
    s.add_argument("--precision", default="auto", choices=["auto", "single", "double"],
                   help="Aer statevector precision; auto = single if it passes a check in a subprocess")
    s.add_argument("--no-noise", action="store_true", help="skip the transpilation for the noise model")
    s.add_argument("--no-validate", action="store_true", help="do not evaluate the picks on the real objective")
    s.add_argument("--model", default="latest")
    s.add_argument("--data", default=None)
    s.add_argument("--split", default="tune")
    s.add_argument("--max-events", type=int, default=0)
    s.add_argument("--workers", type=int, default=4)
    s.add_argument("--target-pt", type=float, default=None)
    s.add_argument("--target-eta", type=float, default=4.0)
    s.add_argument("--target-min-hits", type=int, default=5)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--label", default="qx6_scaling")
    s.add_argument("--out", default=None, help="existing scaling folder to resume")

    s = add("hw-validate", cmd_hw_validate, "hardware samples of a QX2 run folder on the real objective (0 QPU s)")
    s.add_argument("--run-dir", required=True, help="Phase-1 run folder with samples/ibm-*.json (inside hsf2_work)")
    s.add_argument("--top-k", type=int, default=10, help="best distinct states per sample set evaluated")
    s.add_argument("--min-distance", type=int, default=2, help="candidates differ in >= this many knobs")
    s.add_argument("--seed", type=int, default=0, help="seed of the uniform and ideal-circuit controls")
    s.add_argument("--model", default="latest")
    s.add_argument("--data", default=None)
    s.add_argument("--split", default="tune")
    s.add_argument("--max-events", type=int, default=0)
    s.add_argument("--workers", type=int, default=8)
    s.add_argument("--target-pt", type=float, default=None)
    s.add_argument("--target-eta", type=float, default=4.0)
    s.add_argument("--target-min-hits", type=int, default=5)

    s = add("hw-analyze", cmd_hw_analyze, "noise analysis of the hardware samples of a run folder (0 QPU s)")
    s.add_argument("--run-dir", required=True, help="Phase-1 run folder with run_config.json, problem.json and "
                                                    "samples/ibm-*_main.json (+ _control.json)")
    s.add_argument("--tags", nargs="+", default=None, help="hardware main tags (default: all)")
    s.add_argument("--kmax", type=int, default=5, help="highest correlation order k of s_k")
    s.add_argument("--readout", default=None, help="readout_cal.json (hw-readout): the selected main samples are "
                                                   "readout-mitigated; raw and mitigated values are reported side by side")
    s.add_argument("--readout-tags", default=None, help="regex (re.search) of the main tags to mitigate (default: all analysed)")
    s.add_argument("--prefer", default="auto", choices=["auto", "mthree", "numpy"], help="readout correction backend (m3)")
    s.add_argument("--readout-method", default="tensor", choices=["tensor", "m3"],
                   help="tensor: exact inverse of the per-qubit read-out model on all 2^n states (default; corrects broad "
                        "samples); m3: reduced-subspace solve on the observed strings (a no-op on broad 24-qubit samples)")

    def add_hw(name, func, help_, with_run=True, run_help=None, shots_help=None):
        sp = add(name, func, help_)
        if with_run:
            sp.add_argument("run_dir", help=run_help or "Part-II run folder with run_config.json, problem.json and "
                                                        "angles.json (inside hsf2_work)")
        sp.add_argument("--mode", default="dry", choices=["dry", "local", "submit"],
                        help="dry: plan, transpile, estimate (fake backend; --real-target: the real backend's calibration); "
                             "local: fake backend + Aer noise simulation (0 QPU s); submit: IBM hardware, typed "
                             "confirmation 'submit' for every job")
        sp.add_argument("--shots", type=int, default=4000,
                        help=shots_help or "shots of the main PUB and of the control PUB (each)")
        sp.add_argument("--backend", default=None, help="IBM backend (default $HSF2_IBM_BACKEND, else ibm_kingston)")
        sp.add_argument("--fake-backend", default="FakeFez", help="calibration snapshot of dry (without --real-target) "
                                                                  "and local mode")
        sp.add_argument("--real-target", action="store_true",
                        help="dry mode: read the real backend's current target (IBM login, nothing submitted, 0 QPU s)")
        sp.add_argument("--primitive", default="sampler_v2", choices=["sampler_v2", "executor"],
                        help="legacy SamplerV2 (default, comparable with the noise curve) or the executor Sampler")
        sp.add_argument("--creds", default=None, help="credentials file path (default $HSF2_IBM_CREDS; only the path is "
                                                      "passed on, the file is read by Phase 1)")
        sp.add_argument("--ledger", default=None, help="QPU ledger (default $HSF2_QPU_LEDGER, else one per credentials file)")
        sp.add_argument("--budget", type=float, default=None, help="QPU seconds per 28 days (default $HSF2_QPU_BUDGET, 400)")
        sp.add_argument("--reserve", type=float, default=0.0, help="QPU seconds that a job must leave free")
        sp.add_argument("--seeds", type=int, default=4, help="transpiler seeds tried per circuit")
        sp.add_argument("--step", default="step2", help="hardware-plan step recorded in the manifest")
        sp.add_argument("--sim-seed", type=int, default=0, help="seed of the local simulation")
        return sp

    s = add_hw("hw-run", cmd_hw_run, "one hardware job of the suppression ladder (dry / local / submit)")
    s.add_argument("--setting", required=True, choices=list("abcdef"),
                   help="a bare, b DD, c twirl, d both, e fractional gates, f gauges")
    s.add_argument("--K", type=int, default=24, help="couplings kept in the circuit (0: all)")
    s.add_argument("--gauges", type=int, default=8, help="number of gauges of setting f (shots are split equally)")
    s.add_argument("--no-wait", action="store_true", help="submit and return; rebuild the outputs later with hw-retrieve")
    s.add_argument("--server-dry-run", action="store_true",
                   help="executor primitive only: IBM's server-side dry_run (0 QPU s, mock data, no samples are written)")

    s = add_hw("hw-ladder", cmd_hw_ladder, "the suppression-ladder settings one after the other, for one or more K")
    s.add_argument("--K", type=int, nargs="+", default=[12, 48], help="couplings kept (0: all); one ladder per K")
    s.add_argument("--settings", nargs="+", default=list("abcdefd"), choices=list("abcdef"),
                   help="in this order; d is repeated at the end to expose drift")
    s.add_argument("--gauges", type=int, default=8)
    s.add_argument("--readout", action="store_true", help="after every job calibrate the readout of its qubits "
                                                          "(own job, own confirmation)")

    s = add_hw("hw-readout", cmd_hw_readout, "readout calibration of a job's qubits, or mthree correction of its counts")
    s.add_argument("--from-job", default=None, help="job folder whose measured qubits and measurement-twirl switch are calibrated")
    s.add_argument("--cal-shots", type=int, default=None, help="shots per calibration circuit (default: the job's shots per PUB)")
    s.add_argument("--apply", default=None, help="job folder whose counts are corrected (needs --cal)")
    s.add_argument("--cal", default=None, help="readout_cal.json of a calibration folder")
    s.add_argument("--prefer", default="auto", choices=["auto", "mthree", "numpy"])
    s.add_argument("--no-wait", action="store_true",
                   help="submit the calibration and return; hw-retrieve <job id> writes readout_cal.json later")

    s = add_hw("hw-retrieve", cmd_hw_retrieve, "rebuild the outputs of a submitted job from its id", with_run=False)
    s.add_argument("job_id")
    s.add_argument("--run-dir", default=None, help="run folder that holds the job (else --search-root)")
    s.add_argument("--search-root", default=None, help="folder searched for hw/*/manifest.json (default: the runs folder)")

    s = add("hw-ledger", cmd_hw_ledger, "QPU-time ledger of the IBM account")
    s.add_argument("--creds", default=None)
    s.add_argument("--ledger", default=None)
    s.add_argument("--budget", type=float, default=None)
    s.add_argument("--service", action="store_true", help="also ask IBM for its usage figures (metadata only)")

    s = add_hw("hw-components", cmd_hw_components, "step 3: QX3b components (hard pieces first) on hardware, one per PUB",
               run_help="triplet run folder with components.csv and components/ (written by `triplet`; inside hsf2_work)",
               shots_help="shots of each PUB (one PUB per component; equal in every PUB of every job)")
    s.set_defaults(step="step3")
    s.add_argument("--nmin", type=int, default=14, help="smallest component (variables)")
    s.add_argument("--nmax", type=int, default=20, help="largest component (variables, at most 24)")
    s.add_argument("--hard", action="store_true",
                   help="only the hard pieces: both classical descents miss the exact optimum (refuses --benchmark)")
    s.add_argument("--benchmark", type=int, default=0,
                   help="N ordinary (not hard) pieces of the size range, a seeded random sample, run as the benchmark group")
    s.add_argument("--seed", type=int, default=0,
                   help="seed of the random choices (benchmark sample, hard subset of --max) and of the analysis controls")
    s.add_argument("--max", type=int, default=None,
                   help="at most this many components in total: a seeded random subset of the hard pieces is kept, the "
                        "benchmark sample is never reduced")
    s.add_argument("--list", action="store_true", help="print the selection and stop (nothing is written)")
    s.add_argument("--setting", default="d", choices=list("abcde"),
                   help="a bare, b DD, c twirl, d both (default), e fractional gates; f (gauges) is not supported here")
    s.add_argument("--per-job", type=int, default=10, help="PUBs (components) per job; one typed confirmation per job")
    s.add_argument("--top-k", type=int, default=0,
                   help="couplings kept in the circuit (0: all; the conflict couplings are essential)")
    s.add_argument("--no-wait", action="store_true",
                   help="submit and return (one job only); rebuild the outputs later with hw-retrieve")
    s.add_argument("--sim-method", default="auto", choices=["auto", "statevector", "matrix_product_state", "white-noise"],
                   help="local mode: statevector = Aer noise trajectories (small components), matrix_product_state = the "
                        "hw-run simulator (far too slow for these dense circuits), white-noise = model F_est p_ideal + "
                        "(1 - F_est) / 2^n without device noise, auto = statevector up to 14 qubits, white-noise above")

    s = add("hw-components-analyze", cmd_hw_components_analyze,
            "ground-state statistics, descent and controls of component jobs (0 QPU s)")
    s.add_argument("job_dirs", nargs="+", help="job folders <triplet run>/hw/<stamp>_comp_<setting>_<k> of one triplet run "
                                               "folder (inside hsf2_work)")
    s.add_argument("--seed", type=int, default=None, help="seed of the uniform / ideal / white-noise controls (default: "
                                                          "the selection seed recorded in the jobs, as after a run)")
    s.add_argument("--readout", default=None, help="readout_cal.json whose qubits cover the components' measured qubits: the "
                                                   "selected components are readout-mitigated next to the raw values")
    s.add_argument("--readout-tags", default=None, help="regex (re.search) of the component names to mitigate (default: all)")
    s.add_argument("--prefer", default="auto", choices=["auto", "mthree", "numpy"], help="readout correction backend")

    s = add_hw("hw-zne", cmd_hw_zne, "step 4: zero-noise extrapolation jobs (folded circuits) of QX2 K values or QX3b components",
               run_help="QX2 run folder (with --K) or triplet run folder (with --components), inside hsf2_work",
               shots_help="shots of each PUB (one PUB per circuit and lambda; equal in every PUB)")
    s.set_defaults(step="step4")
    s.add_argument("--K", nargs="+", default=None, metavar="K[:l,l,..]",
                   help="QX2 couplings kept, each with its noise factors, e.g. 12:1,3,5 48:1,3 (default factors: --lambdas)")
    s.add_argument("--components", nargs="+", default=None, metavar="NAME[:l,l,..]",
                   help="component JSON names of a triplet run folder (stem, or without the event prefix: 1335_s3_f1_c20)")
    s.add_argument("--lambdas", type=float, nargs="+", default=[1, 3, 5],
                   help="default noise factors (odd integers; in fold mode any numbers in [1, 9], 1 included)")
    s.add_argument("--setting", default="b", choices=list("abcd"), help="b DD without gate twirling (default); a-d allowed")
    s.add_argument("--per-job", type=int, default=None,
                   help="PUBs per job; one typed confirmation per job (default and maximum 10; fold mode: 30)")
    s.add_argument("--top-k", type=int, default=0, help="components: couplings kept (0: all)")
    s.add_argument("--sim-method", default="auto", choices=["auto", "statevector", "matrix_product_state"],
                   help="local mode: statevector up to 14 active qubits, else matrix product state (auto)")
    s.add_argument("--fold-gates", nargs="+", default=None, choices=["cz", "all"],
                   help="fold mode (folding diagnosis): the gate sets folded, one arm each; cz = CZ only, all = CZ, SX and X "
                        "(SX^dagger = RZ(pi) SX RZ(pi)); local unitary folding to fractional lambda on random gate subsets")
    s.add_argument("--subsets", type=int, default=1,
                   help="fold mode: random fold subsets per circuit; the shots are split equally between them (must divide --shots)")
    s.add_argument("--from-job", default=None, metavar="ZNE_JOB_DIR",
                   help="fold mode: a zne job folder <run>/hw/<stamp>_zne_<setting>_<k> of this run folder; its lambda = 1 PUBs "
                        "(the same transpiled circuits, layouts and angles) are the base circuits; --K / --components select")
    s.add_argument("--fold-seed", type=int, default=0, help="fold mode: seed of the random fold subsets")
    s.add_argument("--randomizations", type=int, default=None,
                   help="twirling randomisations of the setting (default 16; the twirled folding arm uses 32 with --setting d)")
    s.add_argument("--no-wait", action="store_true",
                   help="submit every job and return without waiting; hw-retrieve <job id> rebuilds the outputs later")

    s = add("hw-zne-analyze", cmd_hw_zne_analyze, "linear / Richardson / exponential extrapolation of zne jobs (0 QPU s)")
    s.add_argument("job_dirs", nargs="+", help="job folders <run>/hw/<stamp>_zne_<setting>_<k> of one run folder (inside hsf2_work)")
    s.add_argument("--boot-n", type=int, default=1000, help="bootstrap replicates over the shots")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--readout", default=None, help="readout_cal.json covering the jobs' measured qubits (hw-readout "
                                                   "--from-job <zne job>): every observable is read-out-corrected exactly "
                                                   "(tensor inverse) before the extrapolation")

    s = add_hw("hw-qx4", cmd_hw_qx4, "step 5: QX4 1-bit filter (and an HHL fidelity PUB) on the smallest toy events on hardware",
               with_run=False, shots_help="shots of each PUB (equal in every PUB)")
    s.set_defaults(step="step5", seeds=6)
    s.add_argument("--run-dir", default=None, help="run folder that holds hw/ (default: a new <runs>/<stamp>_<label> folder)")
    s.add_argument("--label", default="qx4_hw", help="label of the new run folder")
    s.add_argument("--event", default="2x3,2x5", help="<particles>x<layers>, comma separated; the first is the headline event")
    s.add_argument("--lams", default="1,3,5", help="noise factors (folding) of the first event, e.g. 1,3,5 (numbers >= 1)")
    s.add_argument("--lams-other", default="1", help="noise factors of the other events and of the HHL PUB")
    s.add_argument("--hhl", action=argparse.BooleanOptionalAction, default=True,
                   help="add one HHL PUB (1x3 event, N = 2, --hhl-clock clock qubits): post-selection probability check")
    s.add_argument("--hhl-clock", type=int, default=2)
    s.add_argument("--setting", default="b", choices=list("abcd"), help="b DD without gate twirling (default); a-d allowed")
    s.add_argument("--no-readout", action="store_true", help="skip the read-out calibration job after the main job")
    s.add_argument("--cal-shots", type=int, default=None, help="shots per calibration circuit (default: --shots)")
    s.add_argument("--event-seed", type=int, default=0)
    s.add_argument("--alpha", type=float, default=0.25)
    s.add_argument("--beta", type=float, default=0.5)
    s.add_argument("--eps", type=float, default=1e-4)
    s.add_argument("--threshold", type=float, default=0.8, help="classical solution threshold T")
    s.add_argument("--boot-n", type=int, default=1000, help="bootstrap replicates of the analysis that follows a job")
    s.add_argument("--seed", type=int, default=0, help="seed of the bootstrap and of the uniform control")

    s = add("hw-qx4-analyze", cmd_hw_qx4_analyze, "leakage, filter decisions and ratio zero-noise extrapolation of a qx4 job (0 QPU s)")
    s.add_argument("job_dir", help="job folder <run>/hw/<stamp>_qx4_<setting>_1 (inside hsf2_work)")
    s.add_argument("--boot-n", type=int, default=1000, help="bootstrap replicates over the shots")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--readout", default=None, help="readout_cal.json covering the measured qubits (the calibration job that "
                                                   "hw-qx4 ran after the main job): adds the read-out-corrected variant")

    s = add("ibm-status", cmd_ibm_status, "IBM account usage and QPU calibration medians (no QPU time)")
    s.add_argument("--creds", default=None, help="credentials file (default: Phase 1's ~/ibm_creds.json)")

    s = add("tune-compare", cmd_tune_compare, "QX1: optimisers on the exact landscape")
    s.add_argument("--landscape", required=True, help="folder written by tune-landscape, or latest")
    s.add_argument("--methods", default="qubo-exact,qubo-sa,qubo-qaoa,qubo-qaoa-noise,qubo-qaoa-noise-top24,"
                                        "qubo-random,random-grid,tpe-grid,sa-grid")
    s.add_argument("--seeds", default="0-19")
    s.add_argument("--budget", type=int, default=100)
    s.add_argument("--n-ws", type=int, default=60)
    s.add_argument("--top-k", type=int, default=10)
    s.add_argument("--shots", type=int, default=4096)
    s.add_argument("--min-distance", type=int, default=2)
    _add_score_args(s)

    s = add("tune-objectives", cmd_tune_objectives, "QX1: optimum of S and of its variants on a landscape")
    s.add_argument("--landscape", default="latest", help="folder written by tune-landscape, or latest")

    s = add("select", cmd_select, "QX3: QUBO selection of track candidates")
    s.add_argument("--model", default="latest")
    s.add_argument("--data", default=None)
    s.add_argument("--split", default="tune")
    s.add_argument("--max-events", type=int, default=0)
    s.add_argument("--set", nargs="*", default=["max_degree=3", "cut_hi=0.6", "cut_lo_frac=0.5", "min_hits=5"],
                   metavar="NAME=VALUE", help="tracking knobs that produce overlapping candidates")
    s.add_argument("--methods", default="greedy,exact,sa,random,qaoa,qaoa-noise")
    s.add_argument("--max-vars", type=int, default=24)
    s.add_argument("--penalty-factor", type=float, default=1.5)
    s.add_argument("--shots", type=int, default=1024)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--target-pt", type=float, default=None)
    s.add_argument("--target-eta", type=float, default=4.0)
    s.add_argument("--target-min-hits", type=int, default=5)

    s = add("triplet", cmd_triplet, "QX3: Bapst-style triplet QUBO on TrackML barrel sectors")
    s.add_argument("--data", default="latest")
    s.add_argument("--split", default="tune")
    s.add_argument("--max-events", type=int, default=2)
    s.add_argument("--n-sectors", type=int, default=2)
    s.add_argument("--dphi", type=float, default=0.4, help="sector width in phi (rad)")
    s.add_argument("--fracs", default="0.1,0.25", help="fractions of particles (and noise) kept")
    s.add_argument("--methods", default="exact,sa,qaoa")
    s.add_argument("--max-vars", type=int, default=16, help="largest component solved by QAOA (statevector)")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--label", default="qx3_triplet")

    s = add("hhl", cmd_hhl, "QX4: linear-system tracking emulation and gate counts")
    s.add_argument("--sizes", default="2x3,3x3,4x3,5x3,4x5,8x5,10x5,20x5,40x5",
                   help="particles x layers of the straight-track toy (TrackHHL toy model)")
    s.add_argument("--seeds", type=int, default=10)
    s.add_argument("--alpha", type=float, default=0.25)
    s.add_argument("--beta", type=float, default=0.5)
    s.add_argument("--eps", type=float, default=1e-4)
    s.add_argument("--threshold", type=float, default=0.8)
    s.add_argument("--noise-frac", type=float, default=0.0)
    s.add_argument("--sigma", type=float, default=0.0)
    s.add_argument("--n-clock", type=int, default=3)
    s.add_argument("--circuit-max-qubits", type=int, default=6)
    s.add_argument("--label", default="qx4_hhl")

    s = add("qgnn", cmd_qgnn, "QX5: quantum edge classifier vs classical classifiers")
    s.add_argument("--model", default="latest")
    s.add_argument("--data", default=None)
    s.add_argument("--split", default="tune", help="test events (training edges come from the train split)")
    s.add_argument("--train-events", type=int, default=4)
    s.add_argument("--test-events", type=int, default=3)
    s.add_argument("--qubits", default="4,6")
    s.add_argument("--layers", type=int, default=3)
    s.add_argument("--n-per-class", type=int, default=5000)
    s.add_argument("--epochs", type=int, default=40)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--threads", type=int, default=0)

    s = add("report", cmd_report, "figures of the guide from saved output folders")
    s.add_argument("--out", default=None, help="default: docs/figures (repository) or <work>/figures (deployed)")
    for opt in ("model", "landscape", "compare", "loop", "select", "hhl", "qgnn"):
        s.add_argument(f"--{opt}", default=None)
    s.add_argument("--eval", nargs="*", default=None, help="evaluate output folders (efficiency vs pT)")

    s = add("compare-evals", cmd_compare_evals, "paired per-event difference B - A of two evaluate folders")
    s.add_argument("a", help="evaluate output folder A")
    s.add_argument("b", help="evaluate output folder B (same events)")
    s.add_argument("--n-boot", type=int, default=2000)
    s.add_argument("--seed", type=int, default=0)

    s = add("info", cmd_info, "summary of a data or model folder")
    s.add_argument("dir")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    global _ARGV
    _ARGV = list(sys.argv[1:] if argv is None else argv)
    p = build_parser()
    args = p.parse_args(argv)
    if not getattr(args, "cmd", None):
        p.print_help()
        return 2
    if args.cmd in ("evaluate", "scan", "train"):
        setup_logging(args.log_level)
    return int(args.func(args) or 0)

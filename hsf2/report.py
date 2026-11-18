"""Figures of the guide, made only from saved output folders (python -m hsf2 report ...).

Every figure records the folder it was made from in <out>/figures_manifest.json, so each plotted number can be traced
to an experiment output.  Static figures for a printed PDF: one y-axis per panel, thin marks, recessive grid, a
legend for >= 2 series, and colours from the first three slots of the reference categorical palette (the slots that
stay distinguishable for colour-vision deficiencies in every pairing); charts with more series use small multiples or
single-colour dot charts instead of more hues.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"        # reference palette, light mode, slots 1-3
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                         "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2.0,
                         "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight",
                         "text.color": INK, "axes.titlesize": 9, "axes.titleweight": "bold"})
    return plt


def _save(fig, out: Path, name: str, manifest: dict, sources: List[str]) -> None:
    fig.savefig(out / f"{name}.pdf")
    manifest[name] = sources


def fig_training(model_dir: Path, eval_dirs: List[Path], out: Path, manifest: dict) -> None:
    import pandas as pd
    plt = _style()
    h = pd.read_csv(model_dir / "history.csv")
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.2))
    ax[0].plot(h["epoch"], h["train_loss"], color=C1, label="train")
    ax[0].plot(h["epoch"], h["val_loss"], color=C2, label="validation")
    ax[0].set_yscale("log"); ax[0].set_xlabel("epoch"); ax[0].set_title("weighted BCE loss"); ax[0].legend()
    ax[1].plot(h["epoch"], h["edge_purity"], color=C1)
    ax[1].set_xlabel("epoch"); ax[1].set_title("validation edge purity (cut 0.5)")
    pts, effs, errs = [], [], []
    tabs = [pd.read_csv(d / "eff_vs_pt.csv") for d in eval_dirs]
    n = sum(t["n"].values for t in tabs)
    k = sum((t["eff"].fillna(0) * t["n"]).values for t in tabs)
    lo, hi = tabs[0]["lo"].values, tabs[0]["hi"].values
    eff = np.where(n > 0, k / np.maximum(n, 1), np.nan)
    err = np.sqrt(np.clip(eff * (1 - eff), 0, None) / np.maximum(n, 1))
    mid = np.sqrt(lo * hi)
    ax[2].errorbar(mid, eff, yerr=err, xerr=[mid - lo, hi - mid], fmt="o", ms=4, color=C1, lw=1.0)
    ax[2].set_xscale("log"); ax[2].set_xlabel(r"$p_\mathrm{T}$ [GeV]"); ax[2].set_title("track efficiency (confirm1+2)")
    ax[2].set_ylim(min(0.98, np.nanmin(eff) - 0.005), 1.001)
    fig.tight_layout()
    _save(fig, out, "fig_training", manifest, [str(model_dir)] + [str(d) for d in eval_dirs])
    plt.close(fig)


def fig_landscape(land_dir: Path, out: Path, manifest: dict) -> None:
    plt = _style()
    summ = json.loads((land_dir / "summary.json").read_text())
    with np.load(land_dir / "landscape.npz") as f:
        s, t, phys = f["score"], f["time_proxy"], f["physics"]
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.4))
    ax[0].hist(s, bins=120, color=C1)
    for v, lab, c in ((summ["default"]["score"], "default", C2), (summ["optimum"]["score"], "optimum", C3)):
        ax[0].axvline(v, color=c, lw=1.5)
        ax[0].text(v, ax[0].get_ylim()[1] * 0.92, " " + lab, color=INK2, fontsize=7.5, ha="right" if lab == "optimum" else "left")
    ax[0].set_xlabel("S = eff - fake - dup/5 - t/5"); ax[0].set_ylabel("configurations"); ax[0].set_title("QX1: all 65,536 configurations")
    hb = ax[1].hexbin(t, phys, gridsize=45, bins="log", cmap="Blues", mincnt=1)
    ax[1].scatter([summ["default"]["time_proxy"]], [summ["default"]["physics"]], s=30, color=C2, zorder=3, label="default")
    ax[1].scatter([summ["optimum"]["time_proxy"]], [summ["optimum"]["physics"]], s=30, color=C3, zorder=3, label="optimum of S")
    ax[1].set_xlabel("time proxy t"); ax[1].set_ylabel("eff - fake - dup/5"); ax[1].set_title("physics vs time term")
    ax[1].legend(loc="lower right")
    fig.colorbar(hb, ax=ax[1], label="configurations (log)")
    fig.tight_layout()
    _save(fig, out, "fig_qx1_landscape", manifest, [str(land_dir)])
    plt.close(fig)
    # marginals: best and mean score per level of every knob (small multiples)
    marg = summ["marginals"]
    fig, axs = plt.subplots(2, 4, figsize=(7.2, 3.2), sharey=True)
    for a, (name, d) in zip(axs.ravel(), marg.items()):
        x = np.arange(len(d["values"]))
        a.plot(x, d["best_score"], "o-", color=C1, ms=4, label="best")
        a.plot(x, d["mean_score"], "o-", color=C2, ms=4, label="mean")
        a.set_xticks(x)
        a.set_xticklabels([f"{v:g}" if isinstance(v, (int, float)) else str(v) for v in
                           [round(v, 2) for v in d["values"]]], fontsize=7)
        a.set_title(name.replace("_", " "), fontsize=8)
    axs[0, 0].legend(fontsize=7)
    axs[0, 0].set_ylabel("S"); axs[1, 0].set_ylabel("S")
    fig.tight_layout()
    _save(fig, out, "fig_qx1_marginals", manifest, [str(land_dir)])
    plt.close(fig)


def fig_compare(cmp_dir: Path, out: Path, manifest: dict, name: str = "fig_qx1_compare") -> None:
    plt = _style()
    c = json.loads((cmp_dir / "comparison.json").read_text())
    ms = c["methods_summary"]
    order = sorted(ms, key=lambda m: ms[m]["regret"]["median"])
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [1.0, 1.25]})
    y = np.arange(len(order))[::-1]
    med = [ms[m]["regret"]["median"] for m in order]
    q25 = [ms[m]["regret"]["q25"] for m in order]
    q75 = [ms[m]["regret"]["q75"] for m in order]
    ax[0].hlines(y, q25, q75, color=C1, lw=3, alpha=0.45)
    ax[0].plot(med, y, "o", color=C1, ms=5)
    ax[0].set_yticks(y); ax[0].set_yticklabels(order, fontsize=7.5)
    ax[0].set_xlabel("regret = S* - S(pick)  (median, IQR)")
    ax[0].set_title(f"final regret, budget {c['budget']}, {len(c['seeds'])} seeds")
    # lines: the reference palette's 8 light-mode slots pass the adjacent-pair checks; 7 methods, fixed order
    slots = [C1, C2, C3, "#eda100", "#e87ba4", "#008300", "#6250d6"]
    shown = [m for m in ("qubo-exact", "qubo-sa", "qubo-qaoa", "qubo-qaoa-noise-top24", "random-grid", "tpe-grid",
                         "sa-grid") if m in ms]
    for m, colr in zip(shown, slots):
        cur = np.array(ms[m]["regret_curve_median"], float)
        ax[1].plot(np.arange(1, len(cur) + 1), cur, color=colr, lw=1.4, label=m)
    ax[1].axvline(c["n_ws"], color=INK2, lw=0.8, ls=":")
    ax[1].set_yscale("symlog", linthresh=1e-3); ax[1].set_xlabel("distinct real evaluations (dotted: end of warm start)")
    ax[1].set_ylabel("median regret of best so far")
    ax[1].set_title("convergence")
    ax[1].legend(fontsize=6.5, ncol=2)
    fig.tight_layout()
    _save(fig, out, name, manifest, [str(cmp_dir)])
    plt.close(fig)


def fig_surrogate(run_dir: Path, out: Path, manifest: dict, name: str) -> None:
    """Surrogate-predicted vs measured objective of every closed-loop evaluation, and CV R^2 per round."""
    plt = _style()
    res = json.loads((run_dir / "results.json").read_text())["results"]
    pred, meas, r2 = [], [], {}
    for r in res:
        d = r.get("detail") or {}
        for e in d.get("evaluations", []):
            if e.get("objective") is not None:
                pred.append(e["predicted"])
                meas.append(e["objective"])
        for rd in d.get("rounds", []):
            r2.setdefault(rd["round"], []).append(rd["cv_r2"])
    if not pred:
        return
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6))
    ax[0].scatter(meas, pred, s=6, color=C1, alpha=0.5, lw=0)
    lo, hi = min(min(pred), min(meas)), max(max(pred), max(meas))
    ax[0].plot([lo, hi], [lo, hi], color=INK2, lw=0.8, ls="--")
    ax[0].set_xlabel("measured S"); ax[0].set_ylabel("surrogate prediction")
    ax[0].set_title(f"closed-loop proposals ({len(pred)} evaluations)")
    rounds = sorted(r2)
    ax[1].boxplot([r2[k] for k in rounds], positions=rounds, widths=0.5, patch_artist=True,
                  boxprops={"facecolor": "#cde0f6", "edgecolor": C1}, medianprops={"color": C1},
                  whiskerprops={"color": C1}, capprops={"color": C1}, flierprops={"markeredgecolor": C1, "markersize": 3})
    ax[1].set_xlabel("round"); ax[1].set_ylabel("nested-CV $R^2$ of the surrogate"); ax[1].set_title("surrogate quality")
    fig.tight_layout()
    _save(fig, out, name, manifest, [str(run_dir)])
    plt.close(fig)


def fig_loop_confirm(loop_dir: Path, out: Path, manifest: dict) -> None:
    plt = _style()
    rows = json.loads((loop_dir / "confirm.json").read_text())["rows"]
    by: Dict[str, List[float]] = {}
    tune: Dict[str, List[float]] = {}
    for r in rows:
        m = r["config"].split("/")[0]
        by.setdefault(m, []).extend(v["objective"] for v in r["confirm"].values() if v["objective"] is not None)
        tune.setdefault(m, []).append(r["tune_objective"])
    order = sorted(by, key=lambda m: -np.mean(by[m]))
    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    y = np.arange(len(order))[::-1]
    for yi, m in zip(y, order):
        ax.plot(by[m], [yi] * len(by[m]), "o", color=C1, ms=4, alpha=0.6)
        ax.plot([np.mean(by[m])], [yi], "|", color=INK, ms=12, mew=2)
        ax.plot(tune[m], [yi + 0.25] * len(tune[m]), "^", color=C2, ms=4, alpha=0.8)
    ax.set_yticks(y); ax.set_yticklabels(order, fontsize=7.5)
    ax.set_xlabel("S on held-out events (circles: confirm sets x seeds; bar: mean; triangles: tune)")
    ax.set_title("QX2: picks re-evaluated on independent events")
    fig.tight_layout()
    _save(fig, out, "fig_qx2_confirm", manifest, [str(loop_dir)])
    plt.close(fig)


def qx6_pick_table(scaling_dirs: List[Path]) -> Dict[str, object]:
    """Per number of qubits: the surrogate's prediction at its own optimum (summary.json), the real S of the exact
    optimum and of the best QAOA sample (validation.json, 50 tuning events), and the best real evaluation of the loop
    the surrogate was fitted on (surrogate.json y_max, the same tuning events).  A later folder overrides an earlier
    one at the same size; all folders must share one surrogate source (spec.json), so a run fitted on another loop
    (the first 32-qubit job, on the t/20 loop) cannot be mixed in."""
    pred: Dict[int, float] = {}
    exact: Dict[int, float] = {}
    qaoa: Dict[int, float] = {}
    sources, y_max = set(), set()
    for d in scaling_dirs:
        d = Path(d)
        spec = json.loads((d / "spec.json").read_text())
        sources.add(Path(str(spec["source"]).replace("\\", "/")).name)
        y_max.add(round(float(json.loads((d / "surrogate.json").read_text())["y_max"]), 9))
        for n, s in json.loads((d / "summary.json").read_text())["sizes"].items():
            pred[int(n)] = float(s["predicted_optimum"])
        for r in json.loads((d / "validation.json").read_text())["rows"]:
            if "/" not in r["config"]:
                continue
            tag, kind = r["config"].split("/", 1)
            (exact if kind == "exact" else qaoa)[int(tag.lstrip("n"))] = float(r["objective"])
    if len(sources) != 1 or len(y_max) != 1:
        raise ValueError(f"the runs were fitted on different loops ({sorted(sources)}); plot one objective at a time")
    ns = sorted(pred)
    return {"n": ns, "pred": [pred[n] for n in ns], "exact": [exact[n] for n in ns], "qaoa": [qaoa[n] for n in ns],
            "loop_best": y_max.pop(), "source": sources.pop()}


def fig_qx6_picks(scaling_dirs: List[Path], out: Path, manifest: dict, name: str = "fig_qx6_picks") -> None:
    """QX6 on the real objective against the number of qubits (one y-axis: S on the 50 tuning events)."""
    plt = _style()
    t = qx6_pick_table(scaling_dirs)
    n = np.asarray(t["n"])
    fig, ax = plt.subplots(figsize=(5.6, 2.7))
    ax.plot(n, t["pred"], ls="--", marker="s", ms=4.5, color=C3, label="surrogate's prediction at its optimum")
    ax.plot(n, t["exact"], marker="o", ms=5, color=C2, label="exact surrogate optimum, real $S$")
    ax.plot(n, t["qaoa"], marker="^", ms=5.5, color=C1, label="best QAOA sample, real $S$")
    ax.axhline(t["loop_best"], color=INK2, lw=1.0, ls=":")
    ax.text(n[0], t["loop_best"] + 0.0012, f"best real evaluation of the QX2 loop ({t['loop_best']:.4f})",
            fontsize=7.5, color=INK2, va="bottom")
    k = len(n) - 1
    xb = n[k] + 1.1                                       # bracket right of the last markers
    ax.annotate("", xy=(xb, t["exact"][k]), xytext=(xb, t["pred"][k]),
                arrowprops={"arrowstyle": "<->", "color": INK2, "lw": 0.9, "shrinkA": 0, "shrinkB": 0})
    ax.text(xb + 0.3, 0.5 * (t["pred"][k] + t["loop_best"]),
            f"over-\nestimate\n{t['pred'][k] - t['exact'][k]:.3f}", fontsize=7, color=INK2, va="center")
    ax.set_xticks(n)
    ax.set_xlim(n[0] - 1, n[-1] + 3.2)
    lo = min(min(t["exact"]), min(t["qaoa"]))
    ax.set_ylim(lo - 0.006, max(max(t["pred"]), t["loop_best"]) + 0.006)
    ax.set_xlabel("qubits (bits of the tuning QUBO)")
    ax.set_ylabel("$S$ on 50 tuning events")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.24), ncol=3, fontsize=7.5, handlelength=2.2)
    fig.tight_layout()
    _save(fig, out, name, manifest, [str(d) for d in scaling_dirs])
    plt.close(fig)


def fold_repeat_table(dirs: Dict[str, Path], observable: str = "F_XEB", arm: str = "cz") -> Dict[str, Dict[str, dict]]:
    """Points and exponential extrapolation of hw-zne-analyze outputs: {run label: {target: {'lam', 'value', 'err',
    'zne', 'zne_err'}}}.  Only the folding arm `arm` (target names end in '[<arm>]'); a run without it raises."""
    import pandas as pd
    out: Dict[str, Dict[str, dict]] = {}
    for label, d in dirs.items():
        p, e = pd.read_csv(Path(d) / "zne_points.csv"), pd.read_csv(Path(d) / "zne_extrapolation.csv")
        p = p[(p["observable"] == observable) & p["target"].str.endswith(f"[{arm}]")]
        e = e[(e["observable"] == observable) & e["target"].str.endswith(f"[{arm}]") & (e["model"] == "exponential")]
        if p.empty or e.empty:
            raise ValueError(f"{d}: no '{observable}' rows of the folding arm '{arm}'")
        out[label] = {}
        for tgt, g in p.groupby("target"):
            g = g.sort_values("lam")
            z = e[e["target"] == tgt]
            if len(z) != 1:
                raise ValueError(f"{d}: {len(z)} exponential extrapolations of {tgt}")
            out[label][tgt[: -len(arm) - 2]] = {"lam": g["lam"].to_numpy(float), "value": g["value"].to_numpy(float),
                                                "err": g["err"].to_numpy(float), "zne": float(z["value"].iloc[0]),
                                                "zne_err": float(z["err"].iloc[0])}
    return out


def fig_fold_repeat(first: Dict[str, Path], repeat: Dict[str, Path], out: Path, manifest: dict,
                    pieces: Optional[List[str]] = None, name: str = "fig_fold_repeat",
                    labels=("first run", "repeat"), titles: Optional[Dict[str, str]] = None) -> None:
    """Folding with and without gate twirling, two runs: F_XEB against the noise factor lambda, and the exponential
    zero-noise extrapolation drawn at lambda = 0.  `first`, `repeat`: {'dd': <hw-zne-analyze output>, 'twirl': <...>}.
    One panel per piece (small multiples, shared y-axis); colour = setting, line style and marker fill = run.
    `titles`: panel title per piece (default: the piece name without its event prefix)."""
    plt = _style()
    runs = {(k, s): fold_repeat_table({"x": d[s]})["x"] for k, d in (("first", first), ("repeat", repeat))
            for s in ("dd", "twirl")}
    pieces = pieces or sorted(set.intersection(*[set(v) for v in runs.values()]))
    if not pieces:
        raise ValueError("the four analysis folders have no piece in common")
    fig, axes = plt.subplots(1, len(pieces), figsize=(3.1 * len(pieces) + 0.5, 2.9), sharey=True, squeeze=False)
    colour = {"dd": C1, "twirl": C2}
    shift = {("first", "dd"): -0.12, ("repeat", "dd"): -0.04, ("first", "twirl"): 0.04, ("repeat", "twirl"): 0.12}
    for ax, piece in zip(axes[0], pieces):
        for (k, s), tab in runs.items():
            r = tab[piece]
            first_run = k == "first"
            kw = {"color": colour[s], "ls": "-" if first_run else "--", "marker": "o" if s == "dd" else "s", "ms": 4.5,
                  "mfc": colour[s] if first_run else "white", "mew": 1.2, "lw": 1.6, "capsize": 2, "elinewidth": 0.9}
            ax.errorbar(r["lam"], r["value"], yerr=r["err"], **kw)
            x0 = shift[(k, s)]                          # the four extrapolated values side by side, not on top of each other
            ax.plot([x0, r["lam"][0]], [r["zne"], r["value"][0]], color=colour[s], ls=":", lw=1.0)
            ax.errorbar([x0], [r["zne"]], yerr=[r["zne_err"]], color=colour[s], marker="D", ms=5,
                        mfc=colour[s] if first_run else "white", mew=1.2, capsize=2, elinewidth=0.9, ls="none")
        ax.axvline(0, color=INK2, lw=0.6)
        ax.set_title((titles or {}).get(piece) or (piece.split("event")[-1].lstrip("0") if "event" in piece else piece))
        ax.set_xlabel(r"noise factor $\lambda$")
        ax.set_xlim(-0.25, 3.2)
        ax.set_xticks([0, 1, 1.5, 2, 3])
        ax.set_xticklabels(["0 (ZNE)", "1", "1.5", "2", "3"])
    axes[0][0].set_ylabel(r"$F_\mathrm{XEB}$ (ideal: 1)")
    axes[0][0].set_ylim(0, 1.0)
    from matplotlib.lines import Line2D
    # one entry per drawn style (setting x run), in exactly the colour, line style and marker fill of its curve
    handles = []
    for s, lab_s, mk in (("dd", "DD", "o"), ("twirl", "DD + twirling", "s")):
        for k, lab_r in ((0, labels[0]), (1, labels[1])):
            handles.append(Line2D([], [], color=colour[s], marker=mk, ms=4.5, lw=1.6, ls="-" if k == 0 else "--",
                                  mfc=colour[s] if k == 0 else "white", mew=1.2, label=f"{lab_s}, {lab_r}"))
    # no separate legend glyph for the extrapolated points (a neutral diamond would match no plotted marker): they are
    # identified by the x tick "0 (ZNE)" and drawn as diamonds in the colour and fill of their own curve
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=7.5, handlelength=2.6,
               bbox_to_anchor=(0.5, -0.13))
    fig.tight_layout()
    _save(fig, out, name, manifest, [str(d[s]) for d in (first, repeat) for s in ("dd", "twirl")])
    plt.close(fig)


def fig_compare_measured(loop_dir: Path, out: Path, manifest: dict) -> None:
    """QX2: best measured S per method and seed (tune split) and the median best-so-far curves."""
    plt = _style()
    c = json.loads((loop_dir / "comparison.json").read_text())
    ms = c["methods_measured"]
    res = json.loads((loop_dir / "results.json").read_text())["results"]
    order = sorted(ms, key=lambda m: -ms[m]["best_measured"]["median"])
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [1.0, 1.25]})
    y = np.arange(len(order))[::-1]
    for yi, m in zip(y, order):
        vals = [max(x["objective"] for x in r["records"] if x["objective"] is not None) for r in res if r["method"] == m]
        ax[0].plot(vals, [yi] * len(vals), "o", color=C1, ms=5, alpha=0.8)
    ax[0].set_yticks(y); ax[0].set_yticklabels(order, fontsize=7.5)
    ax[0].set_xlabel("best measured S (tune split), one dot per seed")
    ax[0].set_title(f"QX2: budget {c['budget']} real evaluations")
    slots = [C1, C2, C3, "#eda100", "#e87ba4", "#008300", "#6250d6"]
    shown = [m for m in ("qubo-exact", "qubo-sa", "qubo-qaoa", "qubo-qaoa-noise-top24", "random-grid", "tpe-grid",
                         "sa-grid") if m in ms]
    for m, colr in zip(shown, slots):
        cur = np.array(ms[m]["best_so_far_median"], float)
        ax[1].plot(np.arange(1, len(cur) + 1), cur, color=colr, lw=1.4, label=m)
    ax[1].axvline(c["n_ws"], color=INK2, lw=0.8, ls=":")
    lo = np.nanmin([np.nanmin(np.array(ms[m]["best_so_far_median"], float)[c["n_ws"] // 2:]) for m in shown])
    ax[1].set_ylim(bottom=lo)
    ax[1].set_xlabel("distinct real evaluations (dotted: end of warm start)"); ax[1].set_ylabel("median best measured S")
    ax[1].set_title("convergence"); ax[1].legend(fontsize=6.5, ncol=2)
    fig.tight_layout()
    _save(fig, out, "fig_qx2_compare", manifest, [str(loop_dir)])
    plt.close(fig)


def fig_select(sel_dir: Path, out: Path, manifest: dict) -> None:
    import pandas as pd
    plt = _style()
    df = pd.read_csv(sel_dir / "per_event.csv")
    summ = json.loads((sel_dir / "summary.json").read_text())
    fig, ax = plt.subplots(figsize=(3.6, 2.4))
    g = df[df["method"] == "exact"]
    ax.bar(["fit (<= 24)", "too large"], [g["n_fit"].sum(), g["n_too_large"].sum()], color=C1, width=0.5)
    ax.set_yscale("log"); ax.set_ylabel("conflict components")
    ax.set_title(f"QX3: {summ['components']['n']} components, max size {summ['components']['max_size']}")
    fig.tight_layout()
    _save(fig, out, "fig_qx3_components", manifest, [str(sel_dir)])
    plt.close(fig)


def fig_hhl(hhl_dir: Path, out: Path, manifest: dict) -> None:
    import pandas as pd
    plt = _style()
    m = pd.read_csv(hhl_dir / "systems_mean.csv").sort_values("n_segments")
    c = pd.read_csv(hhl_dir / "circuits.csv").sort_values("n_segments")
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for col, lab, colr, ls, lw in (("classical_purity", "classical solve", C1, "-", 3.0),
                                   ("onebit_pur_shots10N", "1-bit filter, 10N shots (= classical)", C2, "--", 1.6),
                                   ("hhl-ideal_pur_shots10N", "ideal HHL, 10N shots", C3, "-", 1.6)):
        ok = m[col].notna()
        ax[0].plot(m.loc[ok, "n_segments"], m.loc[ok, col], "o", ls=ls, lw=lw, color=colr, ms=4, label=lab)
    ax[0].set_xscale("log"); ax[0].set_xlabel("segments N"); ax[0].set_ylabel("segment purity")
    # legends below the panels: inside, they covered the data (guide v0.2.21 check)
    ax[0].legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.27), ncol=1)
    ax[0].set_title("QX4: purity (efficiency ~1 for all)")
    ax[1].plot(c["n_segments"], c["onebit_cx"], "o-", color=C2, ms=4, label="1-bit filter")
    ax[1].plot(c["n_segments"], c["hhl_cx"], "o-", color=C3, ms=4, label="HHL, 3 clock qubits")
    ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].set_xlabel("segments N"); ax[1].set_ylabel("CX gates")
    ax[1].set_xticks(c["n_segments"].tolist())
    ax[1].set_xticklabels([str(int(v)) for v in c["n_segments"]])
    ax[1].minorticks_off()
    ax[1].set_title("generic Pauli + Lie-Trotter synthesis")
    ax[1].legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.27), ncol=1)
    fig.tight_layout()
    _save(fig, out, "fig_qx4", manifest, [str(hhl_dir)])
    plt.close(fig)


def fig_qgnn(q_dir: Path, out: Path, manifest: dict) -> None:
    plt = _style()
    d = json.loads((q_dir / "summary.json").read_text())
    rows = []
    for key, r in d.items():
        if key == "interaction_network":
            continue
        for model, lab in (("quantum", "quantum circuit"), ("mlp_same_params", "MLP, same #params"), ("logistic", "logistic")):
            rows.append((f"{key} | {lab}", r[model]["auc"], model))
    fig, ax = plt.subplots(figsize=(5.2, 0.28 * len(rows) + 0.9))
    y = np.arange(len(rows))[::-1]
    colr = {"quantum": C1, "mlp_same_params": C2, "logistic": C3}
    for yi, (lab, auc, model) in zip(y, rows):
        ax.plot([auc], [yi], "o", color=colr[model], ms=5)
    ax.axvline(d["interaction_network"]["auc"], color=INK2, lw=1, ls="--")
    ax.text(d["interaction_network"]["auc"], y.max() + 0.6, "full IN", fontsize=7, color=INK2, ha="right")
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=7)
    ax.set_xlabel("edge AUC on held-out tuning events"); ax.set_title("QX5: same inputs, different classifiers")
    fig.tight_layout()
    _save(fig, out, "fig_qx5", manifest, [str(q_dir)])
    plt.close(fig)


def make_all(out: Path, model_dir: Optional[Path] = None, eval_dirs: Optional[List[Path]] = None,
             landscape: Optional[Path] = None, compare: Optional[Path] = None, loop: Optional[Path] = None,
             loop_compare: bool = True, select: Optional[Path] = None, hhl: Optional[Path] = None,
             qgnn: Optional[Path] = None) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    manifest: dict = {}
    if model_dir and eval_dirs:
        fig_training(Path(model_dir), [Path(d) for d in eval_dirs], out, manifest)
    if landscape:
        fig_landscape(Path(landscape), out, manifest)
    if compare:
        fig_compare(Path(compare), out, manifest)
        fig_surrogate(Path(compare), out, manifest, "fig_qx1_surrogate")
    if loop:
        fig_loop_confirm(Path(loop), out, manifest)
        fig_compare_measured(Path(loop), out, manifest)
        fig_surrogate(Path(loop), out, manifest, "fig_qx2_surrogate")
    if select:
        fig_select(Path(select), out, manifest)
    if hhl:
        fig_hhl(Path(hhl), out, manifest)
    if qgnn:
        fig_qgnn(Path(qgnn), out, manifest)
    (out / "figures_manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest

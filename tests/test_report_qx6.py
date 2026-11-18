"""fig_qx6_picks / qx6_pick_table (hsf2.report): the QX6 picks figure is made only from tune-scaling run folders,
never mixes surrogates fitted on different loops, and a later run overrides an earlier one at the same size (the
32-qubit rerun of 13 November 2026)."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hsf2.report import fig_qx6_picks, qx6_pick_table  # noqa: E402


def _run(root: Path, name: str, source: str, y_max: float, sizes: dict) -> Path:
    d = root / name
    d.mkdir(parents=True)
    (d / "spec.json").write_text(json.dumps({"source": source}))
    (d / "surrogate.json").write_text(json.dumps({"y_max": y_max}))
    (d / "summary.json").write_text(json.dumps({"sizes": {str(n): {"predicted_optimum": v[0]} for n, v in sizes.items()}}))
    rows = [{"config": "default", "objective": 0.79}]
    for n, (_, ex, qa) in sizes.items():
        rows += [{"config": f"n{n}/exact", "objective": ex}, {"config": f"n{n}/qaoa", "objective": qa}]
    (d / "validation.json").write_text(json.dumps({"rows": rows}))
    return d


def test_table_override_and_figure(tmp_path):
    a = _run(tmp_path, "a", "runs/02-11-2026-14-42_qx2_loop", 0.9407, {16: (0.96, 0.92, 0.93), 32: (1.0, 0.90, 0.89)})
    b = _run(tmp_path, "b", "runs/02-11-2026-14-42_qx2_loop", 0.9407, {32: (0.965, 0.922, 0.925)})
    t = qx6_pick_table([a, b])
    assert t["n"] == [16, 32] and t["exact"] == [0.92, 0.922] and t["qaoa"] == [0.93, 0.925] and t["pred"][1] == 0.965
    assert t["loop_best"] == 0.9407 and t["source"] == "02-11-2026-14-42_qx2_loop"
    man = {}
    fig_qx6_picks([a, b], tmp_path, man)
    assert (tmp_path / "fig_qx6_picks.pdf").stat().st_size > 1000 and man["fig_qx6_picks"] == [str(a), str(b)]


def test_runs_on_different_loops_are_refused(tmp_path):
    a = _run(tmp_path, "a", "runs/02-11-2026-14-42_qx2_loop", 0.9407, {16: (0.96, 0.92, 0.93)})
    c = _run(tmp_path, "c", "runs/04-11-2026-12-44_qx2_loop_t20", 0.9779, {32: (1.0067, 0.902, 0.898)})
    with pytest.raises(ValueError, match="different loops"):
        qx6_pick_table([a, c])


def _zne_out(root: Path, name: str, rows: dict, arm: str = "cz") -> Path:
    """A minimal hw-zne-analyze output: zne_points.csv and zne_extrapolation.csv for {target: (values, zne)}."""
    d = root / name
    d.mkdir(parents=True)
    pts, ext = ["target,observable,lam,n_2q,shots,value,err,ideal,floor"], ["target,observable,model,value,err"]
    for tgt, (vals, zne) in rows.items():
        for lam, v in zip((1.0, 1.5, 2.0, 3.0), vals):
            pts.append(f"{tgt}[{arm}],F_XEB,{lam},100,4000,{v},0.01,1.0,0.0")
            pts.append(f"{tgt}[{arm}],F_E,{lam},100,4000,{v + 0.2},0.01,1.0,0.0")
        ext += [f"{tgt}[{arm}],F_XEB,exponential,{zne},0.03", f"{tgt}[{arm}],F_XEB,linear,{zne - 0.1},0.02"]
    (d / "zne_points.csv").write_text("\n".join(pts) + "\n")
    (d / "zne_extrapolation.csv").write_text("\n".join(ext) + "\n")
    return d


def test_fold_repeat_table_and_figure(tmp_path):
    """fig_fold_repeat reads only hw-zne-analyze outputs: the exponential extrapolation of the chosen observable and
    folding arm, per piece; a folder without that arm is refused."""
    from hsf2.report import fig_fold_repeat, fold_repeat_table
    a = _zne_out(tmp_path, "a", {"p1": ((0.5, 0.4, 0.3, 0.2), 0.7), "p2": ((0.6, 0.5, 0.4, 0.3), 0.8)})
    b = _zne_out(tmp_path, "b", {"p1": ((0.45, 0.3, 0.2, 0.1), 0.8), "p2": ((0.5, 0.4, 0.3, 0.2), 0.8)})
    t = fold_repeat_table({"x": a})["x"]
    assert set(t) == {"p1", "p2"} and t["p1"]["zne"] == 0.7 and t["p1"]["zne_err"] == 0.03
    assert t["p1"]["lam"].tolist() == [1.0, 1.5, 2.0, 3.0] and t["p2"]["value"].tolist() == [0.6, 0.5, 0.4, 0.3]
    other = _zne_out(tmp_path, "c", {"p1": ((0.5, 0.4, 0.3, 0.2), 0.7)}, arm="all")
    with pytest.raises(ValueError, match="folding arm"):
        fold_repeat_table({"x": other})
    out, man = tmp_path / "figs", {}
    out.mkdir()
    fig_fold_repeat({"dd": a, "twirl": b}, {"dd": a, "twirl": b}, out, man)
    assert (out / "fig_fold_repeat.pdf").stat().st_size > 1000 and len(man["fig_fold_repeat"]) == 4

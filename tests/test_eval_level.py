"""evaluate --level (hit-selection override), fake-track breakdown, edge PR summary, compare-evals.  Synthetic data only;
every folder lives inside the pytest temporary directory (HSF2_WORK / HSF2_DATA / HSF2_MODELS)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """synth data with noise -> S0 model -> evaluate on its own level and on S0N; returns the two output folders."""
    from hsf2 import cli
    mp = pytest.MonkeyPatch()
    tmp = tmp_path_factory.mktemp("evlevel")
    for k, sub in (("HSF2_WORK", ""), ("HSF2_DATA", "data"), ("HSF2_MODELS", "models")):
        mp.setenv(k, str(tmp / sub) if sub else str(tmp))
    try:
        assert cli.main(["synth", "--n-events", "12", "--particles", "150", "--noise-frac", "0.2", "--n-train", "6",
                         "--n-val", "2", "--n-tune", "3", "--n-confirm", "1", "--n-confirm-sets", "1"]) == 0
        assert cli.main(["train", "--data", "latest", "--level", "S0", "--epochs", "2", "--hidden", "16",
                         "--n-iter", "1", "--label", "t"]) == 0
        assert cli.main(["evaluate", "--model", "latest", "--split", "tune"]) == 0
        assert cli.main(["evaluate", "--model", "latest", "--split", "tune", "--level", "S0N"]) == 0
        model = next((tmp / "models").iterdir())
        outs = sorted((model / "outputs").glob("*_evaluate_tune*"), key=lambda p: p.stat().st_mtime_ns)
        # two invocations inside one minute get distinct folders (never-overwrite); pick by recorded level
        by = {json.loads((p / "config.json").read_text())["hits"]["level"]: p for p in outs}
        yield {"S0": by["S0"], "S0N": by["S0N"], "tmp": tmp, "model": model}
    finally:
        mp.undo()


def test_level_override_changes_hits_and_is_recorded(trained):
    a = json.loads((trained["S0"] / "metrics.json").read_text())
    b = json.loads((trained["S0N"] / "metrics.json").read_text())
    assert a["hit_level"] == "S0" and a["model_hit_level"] == "S0"
    assert b["hit_level"] == "S0N" and b["model_hit_level"] == "S0"        # level used and the model's own
    assert b["n_hits"] > a["n_hits"]                                       # the noise hits are in the graph
    assert b["edges_loose_per_event"] > a["edges_loose_per_event"]
    ca = json.loads((trained["S0"] / "config.json").read_text())
    cb = json.loads((trained["S0N"] / "config.json").read_text())
    assert cb["hits"]["keep_noise"] is True and cb["extra"]["level_override"] is True
    assert cb["extra"]["model_hits"]["level"] == "S0" and ca["extra"]["level_override"] is False
    assert ca["events"] == cb["events"]                                    # same events, different hit selection
    # the model on disk is unchanged
    assert json.loads((trained["model"] / "model.json").read_text())["hit_selection"]["level"] == "S0"


def test_metrics_json_has_the_diagnostics(trained):
    m = json.loads((trained["S0N"] / "metrics.json").read_text())
    for k in ("edge_recall", "edge_precision", "edge_pr_auc", "edge_purity_at_recall_95", "edge_auc",
              "edges_loose_per_event", "edges_filtered_per_event", "edge_keep_fraction",
              "t_graph_per_event", "t_gnn_per_event", "t_build_per_event", "hits_per_event"):
        assert k in m, k
    assert 0.0 <= m["edge_keep_fraction"] <= 1.0 and 0.0 <= m["edge_pr_auc"] <= 1.0
    assert m["t_graph_per_event"] > 0.0
    for key in ("pt", "eta"):
        rows = (trained["S0N"] / f"eff_vs_{key}.csv").read_text().splitlines()
        assert rows[0].split(",") == ["lo", "hi", "n", "eff"]
    pe = (trained["S0N"] / "per_event.csv").read_text().splitlines()[0].split(",")
    for c in ("event_id", "eff", "eff_dm", "fake", "dup", "trackml"):
        assert c in pe


def test_fake_breakdown_sums_to_fake_count(trained):
    for lvl in ("S0", "S0N"):
        m = json.loads((trained[lvl] / "metrics.json").read_text())
        assert m["n_fake_noise"] + m["n_fake_mixed"] + m["n_fake_wrong"] == m["n_fake"]
        if m["n_fake"]:
            assert abs(m["fake_noise_frac"] + m["fake_mixed_frac"] + m["fake_wrong_frac"] - 1.0) < 1e-12


def test_fake_breakdown_hand_made_tracks():
    from hsf2.config import TargetConfig
    from hsf2.data.event import EventData
    from hsf2.tracking.metrics import event_counts
    pid = np.array([1, 1, 1, 1, 2, 2, 2, 2, 0, 0, 0, 0])
    n = len(pid)
    hits = {"particle_id": pid, "x": np.arange(n, dtype=float), "y": np.zeros(n), "z": np.zeros(n),
            "layer": np.arange(n), "weight": np.ones(n)}
    parts = {"particle_id": np.array([1, 2]), "pt": np.array([2.0, 2.0]), "eta": np.array([0.0, 0.0]),
             "phi": np.zeros(2), "vz": np.zeros(2)}
    ev = EventData("e", hits, parts, {})
    tracks = [np.array([0, 1, 2, 3]),            # matched (particle 1)
              np.array([8, 9, 10]),              # all noise
              np.array([4, 8, 9, 10]),           # mixed: 1 real + 3 noise
              np.array([0, 4, 1, 5])]            # wrong: two particles, 2 + 2, no noise  -> purity 0.5 = matched at 0.5
    c = event_counts(ev, tracks, TargetConfig(pt_min=1.0, min_hits=3, match_min=0.6))
    assert c["n_fake"] == 3 and (c["n_fake_noise"], c["n_fake_mixed"], c["n_fake_wrong"]) == (1, 1, 1)
    assert c["n_fake_noise"] + c["n_fake_mixed"] + c["n_fake_wrong"] == c["n_fake"]


def test_edge_pr_summary_perfect_and_random():
    from hsf2.tracking.metrics import edge_pr_summary
    y = np.array([1, 1, 1, 0, 0, 0], bool)
    r = edge_pr_summary(y, np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1]))
    assert r["edge_pr_auc"] == pytest.approx(1.0) and r["edge_purity_at_recall_95"] == pytest.approx(1.0)
    r = edge_pr_summary(y, np.array([0.9, 0.8, 0.1, 0.7, 0.6, 0.5]))        # third true edge ranked last
    assert r["edge_purity_at_recall_99"] == pytest.approx(0.5)
    assert np.isnan(edge_pr_summary(np.zeros(3, bool), np.arange(3.0))["edge_pr_auc"])


def test_compare_evals_identical_gives_zero_and_mismatch_is_refused(trained, capsys):
    from hsf2 import cli
    a = trained["S0"]
    assert cli.main(["compare-evals", str(a), str(a), "--n-boot", "200"]) == 0
    out = Path(capsys.readouterr().out.strip().splitlines()[-1])
    assert str(out).startswith(str(trained["tmp"]))                          # never outside the work dir
    res = json.loads((out / "compare.json").read_text())
    assert res["n_events"] == 3
    for name, v in res["metrics"].items():
        assert v["diff"] == 0.0 and v["lo68"] == 0.0 and v["hi68"] == 0.0 and v["stderr"] == 0.0, name
    # S0 vs S0N: same events, runs and gives a finite difference
    assert cli.main(["compare-evals", str(a), str(trained["S0N"]), "--n-boot", "200"]) == 0
    out2 = Path(capsys.readouterr().out.strip().splitlines()[-1])
    r2 = json.loads((out2 / "compare.json").read_text())
    assert np.isfinite(r2["metrics"]["efficiency"]["diff"])
    # different event lists: refused
    import shutil
    bad = trained["tmp"] / "bad_eval"
    shutil.copytree(a, bad)
    pe = (bad / "per_event.csv").read_text().splitlines()
    (bad / "per_event.csv").write_text("\n".join(pe[:-1]) + "\n")            # one event fewer
    with pytest.raises(SystemExit, match="same events"):
        cli.main(["compare-evals", str(a), str(bad)])

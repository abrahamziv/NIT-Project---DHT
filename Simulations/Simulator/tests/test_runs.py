import numpy as np

from ddms import EncoderBank, GaussianShift, LRTEncoder, load_run, save_run
from ddms.sweep import sweep_over_N


def test_save_and_load_round_trip(tmp_path):
    model = GaussianShift(1.0)
    res = sweep_over_N(
        model, lambda n: EncoderBank.identical(LRTEncoder(1.0), n), [1, 2, 4]
    )
    meta = {"model": {"type": "GaussianShift", "mu": 1.0}, "x": "N"}
    run_dir = save_run(res, label="test_run", meta=meta, root=tmp_path)
    assert run_dir.name.endswith("_test_run")
    assert (run_dir / "data.json").exists()

    run = load_run(run_dir)
    assert run["label"] == "test_run"
    assert run["meta"] == meta
    assert set(run["data"]) == set(res)
    for k in res:
        np.testing.assert_array_equal(run["data"][k], np.asarray(res[k]))


def test_round_trip_preserves_floats_exactly(tmp_path):
    res = {"x": np.array([1.0, 2.0]), "v": np.array([-18.492812345678901, np.inf])}
    run_dir = save_run(res, label="floats", root=tmp_path)
    run = load_run(run_dir)
    np.testing.assert_array_equal(run["data"]["v"], res["v"])


def test_two_runs_get_distinct_directories(tmp_path):
    d1 = save_run({"x": [1]}, label="a", root=tmp_path)
    d2 = save_run({"x": [2]}, label="b", root=tmp_path)
    assert d1 != d2
    assert load_run(d1)["data"]["x"] == [1]
    assert load_run(d2)["data"]["x"] == [2]


def test_group_defaults_to_label(tmp_path):
    run_dir = save_run({"x": [1]}, label="solo_run", root=tmp_path)
    date = run_dir.name[:10]
    assert run_dir.parent.name == f"solo_run_{date}"


def test_explicit_group_nests_siblings_together(tmp_path):
    d1 = save_run({"x": [1]}, label="vanilla", group="compare_snrm5", root=tmp_path)
    d2 = save_run({"x": [2]}, label="silence", group="compare_snrm5", root=tmp_path)
    assert d1.parent == d2.parent
    date = d1.name[:10]
    assert d1.parent.name == f"compare_snrm5_{date}"


def test_session_nests_under_a_top_level_bucket(tmp_path):
    run_dir = save_run({"x": [1]}, label="vanilla4", group="cmp", session="M4", root=tmp_path)
    assert run_dir.parent.parent == tmp_path / "M4"


def test_no_session_stays_directly_under_root(tmp_path):
    run_dir = save_run({"x": [1]}, label="solo", root=tmp_path)
    assert run_dir.parent.parent == tmp_path

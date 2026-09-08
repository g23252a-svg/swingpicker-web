"""Prevent alpha labels from crossing the walk-forward training boundary."""
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd

import alpha_engine as ae


def _panel():
    # Missing weekday sessions exercise the exchange-holiday case that a fixed
    # calendar/business-day embargo cannot represent.
    sessions = pd.bdate_range("2026-01-01", "2026-03-31")
    sessions = sessions.difference(pd.to_datetime(["2026-01-26", "2026-02-02"]))
    end_map = pd.Series(sessions, index=sessions).shift(-6)
    rows = []
    rng = np.random.default_rng(24)
    for day in sessions:
        end = end_map.loc[day]
        for i in range(60):
            signal = (i - 30) / 60
            rows.append({
                "_ymd": day.strftime("%Y%m%d"),
                "_label_end_ymd": end.strftime("%Y%m%d") if pd.notna(end) else None,
                "F1": signal,
                "_fwd": signal + rng.normal(0, .1),
                "종목코드": f"{i:06d}",
            })
    panel = pd.DataFrame(rows)
    panel["_y"] = panel.groupby("_ymd")["_fwd"].rank(pct=True)
    return panel


def test_panel_records_label_end_from_price_sessions(tmp_path, monkeypatch):
    sessions = pd.to_datetime([
        "2026-01-23", "2026-01-27", "2026-01-28", "2026-01-29",
        "2026-01-30", "2026-02-03", "2026-02-04", "2026-02-05",
    ])
    codes = [f"{i:06d}" for i in range(30)]
    close = pd.DataFrame(100.0, index=sessions, columns=codes)
    fwd = close.shift(-6) / close.shift(-1) - 1
    monkeypatch.setattr(ae, "_load_ohlcv_panel", lambda _: (close, fwd))
    pd.DataFrame({"종목코드": codes, "종가": 100}).to_csv(
        tmp_path / "recommend_20260123.csv", index=False, encoding="utf-8-sig"
    )

    panel = ae.build_training_panel(str(tmp_path))

    assert len(panel) == 30
    assert panel["_label_end_ymd"].unique().tolist() == ["20260204"]
    assert "_label_end_ymd" not in ae._numeric_feature_cols(panel)


def test_walk_forward_purges_unobservable_labels(monkeypatch):
    panel = _panel()
    # Even an old anchor is forbidden if its outcome is not yet observable.
    panel.loc[panel["_ymd"] == "20260105", "_label_end_ymd"] = "20260310"
    panel.loc[panel["_ymd"] == "20260106", "_label_end_ymd"] = "bad-date"
    calls = []

    def fit_predict(data, features, train_mask, test_mask):
        training = data.loc[train_mask]
        test_month = data.loc[test_mask, "_ymd"].min()[:6]
        assert (training["_label_end_ymd"] < test_month + "01").all()
        assert "20260106" not in set(training["_ymd"])
        if test_month == "202602":
            assert "20260105" not in set(training["_ymd"])
            assert "20260123" not in set(training["_ymd"])
        calls.append(test_month)
        return None, features, data.loc[test_mask, "F1"].to_numpy()

    monkeypatch.setattr(ae, "_fit_predict", fit_predict)
    report = ae.walk_forward_validate(panel, ["F1"], min_train_days=10)

    assert report["ok"]
    assert calls == ["202602", "202603"]
    assert report["validation_scheme"] == ae.VALIDATION_SCHEME
    assert report["label_end_source"] == "ohlcv_label_end"
    assert all(f["train_last_label_end"] < f["test_start"] for f in report["folds"])
    assert all(f["purged_rows"] > 0 for f in report["folds"])


def test_legacy_panel_uses_observed_sessions_and_keeps_unknown_ends_unusable():
    panel = _panel().drop(columns="_label_end_ymd")
    ends, source = ae._validation_label_ends(panel)

    assert source == "observed_panel_sessions_conservative"
    # Jan 26 and Feb 2 are absent: six future sessions end Feb 4, not Feb 2.
    assert ends.loc[panel["_ymd"] == "20260123"].eq(pd.Timestamp("2026-02-04")).all()
    assert ends.loc[panel["_ymd"] == "20260331"].isna().all()


def test_invalid_explicit_label_end_never_falls_back():
    panel = _panel()
    selected = panel["_ymd"] == "20260105"
    panel.loc[selected, "_label_end_ymd"] = "20260105"
    ends, source = ae._validation_label_ends(panel)
    assert source == "ohlcv_label_end"
    assert ends.loc[selected].isna().all()


def test_all_invalid_label_ends_cannot_validate(monkeypatch):
    panel = _panel()
    panel["_label_end_ymd"] = None

    def forbidden_fit(*args):
        raise AssertionError("No observable labels may reach model training")

    monkeypatch.setattr(ae, "_fit_predict", forbidden_fit)
    report = ae.walk_forward_validate(panel, ["F1"], min_train_days=10)
    assert report["ok"] is False
    assert report["validated"] is False
    assert report["folds"] == []


def test_old_unpurged_artifact_cannot_score(tmp_path, monkeypatch):
    import joblib

    (tmp_path / ae.META_PATH).write_text(json.dumps({"validated": True}), encoding="utf-8")
    (tmp_path / ae.MODEL_PATH).write_bytes(b"old artifact")
    loaded = []
    monkeypatch.setattr(joblib, "load", lambda path: loaded.append(path))

    out = ae.score_today(pd.DataFrame({"F1": [1.0]}), str(tmp_path))

    assert not loaded
    assert out["ALPHA_VALIDATED"].eq(0).all()
    assert out["ALPHA_SCORE"].isna().all()
    assert out["ALPHA_WIN_PROB"].isna().all()


def test_revalidated_artifact_can_score(tmp_path, monkeypatch):
    import joblib

    (tmp_path / ae.META_PATH).write_text(json.dumps({
        "validated": True, "validation_scheme": ae.VALIDATION_SCHEME,
        "calibration": [{"decile": 9, "win_rate": .55}],
    }), encoding="utf-8")
    (tmp_path / ae.MODEL_PATH).write_bytes(b"validated artifact")
    model = SimpleNamespace(predict=lambda data: data["F1"].to_numpy())
    monkeypatch.setattr(joblib, "load", lambda _: {"model": model, "features": ["F1"]})

    out = ae.score_today(pd.DataFrame({"F1": [0.2, 0.8]}), str(tmp_path))

    assert out["ALPHA_VALIDATED"].eq(1).all()
    assert out.loc[1, "ALPHA_SCORE"] == 100.0
    assert out.loc[1, "ALPHA_WIN_PROB"] == .55

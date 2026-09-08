"""Behavioral tests of purging, honest probability gates and the UI contract."""
from dataclasses import replace
import json

import numpy as np
import pandas as pd

from services.swing_features import FEATURES
from services.swing_model import POLICY, purged_split, summarize_oos, train_and_predict
from services.swing_service import attach_predictions, enrich_from_snapshot


def sample_panel(n_days=90, codes=30):
    rng = np.random.default_rng(25)
    days = pd.bdate_range("2025-01-02", periods=n_days)
    rows = []
    for i, date in enumerate(days):
        for code in range(codes):
            strength = rng.normal()
            row = {f: rng.normal(scale=.02) for f in FEATURES}
            row.update(date=date, code=f"{code:06d}", label_end=days[i+5] if i+5<len(days) else pd.NaT,
                       close=10000, volume=300000, turnover=3e9, ret_1d=.01 * strength,
                       ma20_distance=strength, relative_volume_20d=1.0,
                       forward_return=.05 if strength > 0 else -.05,
                       trade_return=.05 if strength > 0 else -.05)
            row["target_up"] = float(row["trade_return"] > 0)
            rows.append(row)
    return pd.DataFrame(rows)


def test_purge_uses_completed_label_and_late_availability():
    panel = sample_panel()
    panel["_label_ready"] = panel.label_end
    days = sorted(panel.date.unique())
    policy = replace(POLICY, min_train_days=20, calibration_days=10, min_calibration_days=5)
    # A bar backfilled much later is not a known training label.
    panel.loc[0, "_label_ready"] = days[80]
    train, cal, test = purged_split(panel, days[60], days[69], policy)
    assert 0 not in train.index
    assert train._label_ready.max() < cal.date.min()
    assert cal._label_ready.max() < test.date.min()
    assert train.date.nunique() >= 20 and cal.date.nunique() >= 5


def test_predictable_data_passes_but_unknown_future_labels_stay_out():
    panel = sample_panel()
    policy = replace(POLICY, min_train_days=20, calibration_days=10,
                     min_calibration_days=5, test_block_days=10, max_folds=3, min_oos_days=20)
    asof = panel.date.max().strftime("%Y%m%d")
    picks, report, oos = train_and_predict(panel, asof, policy)
    assert len(picks) == 30
    assert report["validated"] and report["trade_validated"]
    assert report["hit_rate"] > .9
    assert oos.date.max() <= panel.date.max() - pd.Timedelta(days=5)
    assert picks.probability.between(0, 1).all()
    assert all(f["train_label_available"] < f["cal_start"] for f in report["folds"])
    assert all(f["cal_label_available"] < f["test_start"] for f in report["folds"])


def test_ranking_can_be_better_while_all_trades_lose():
    days = pd.bdate_range("2025-01-02", periods=45)
    rows = []
    for date in days:
        for i in range(20):
            rows.append(dict(date=date, code=str(i), target_up=float(i<10),
                             probability=.8 if i<10 else .2, baseline_probability=.5,
                             trade_return=-.01 if i<10 else -.05, forward_return=-.01))
    report = summarize_oos(pd.DataFrame(rows))
    assert not report["trade_validated"]
    assert report["net_return_mean"] < 0
    assert not report["trade_checks"]["net_ci_positive"]


def test_chance_model_is_never_called_validated():
    rng = np.random.default_rng(11)
    rows = pd.DataFrame({"date": np.repeat(pd.bdate_range("2025-01-02", periods=45), 20),
                         "code": [str(i) for i in range(900)],
                         "target_up": rng.integers(0,2,900), "probability": .5,
                         "baseline_probability": .5,"forward_return": 0.,"trade_return": 0.})
    report = summarize_oos(rows)
    assert not report["validated"] and not report["trade_validated"]


def test_unresolved_top_picks_are_never_replaced_by_known_winners():
    rows = []
    for day in pd.bdate_range("2025-01-02", periods=45):
        for i in range(10):
            rows.append(dict(date=day, code=str(i), probability=.9-i*.05,
                baseline_probability=.5, target_up=np.nan if i<5 else 1.,
                trade_return=np.nan if i<5 else .1, forward_return=np.nan if i<5 else .1))
    report = summarize_oos(pd.DataFrame(rows))
    assert report['unresolved_selected'] == 225
    assert report['net_return_mean'] is None
    assert report['hit_rate'] is None
    assert not report['validated'] and not report['trade_validated']
    json.dumps(report, allow_nan=False)


def test_confirmed_stop_is_retained_when_unstopped_price_is_missing():
    panel = sample_panel()
    # A later missing bar cannot hide a confirmed stopped loser.
    panel.loc[panel.trade_return.lt(0), 'forward_return'] = np.nan
    policy = replace(POLICY, min_train_days=20, calibration_days=10,
                     min_calibration_days=5, test_block_days=10, max_folds=3, min_oos_days=20)
    _, _, oos = train_and_predict(panel, panel.date.max().strftime('%Y%m%d'), policy)
    assert oos.trade_return.lt(0).any()
    assert oos.loc[oos.trade_return.lt(0), 'forward_return'].isna().all()


def _df():
    return pd.DataFrame({"종목코드": ["0126Z0","005930"], "기준일": ["20260907"]*2,
                         "PRODUCTION_BUY": [0,1], "켈리_수량": [0,5]})


def _picks():
    return pd.DataFrame({"code":["0126Z0","005930"],"probability":[.75,.7],
                         "rank":[1,2],"score":[100.,50.],"reason":["관찰","관찰"]})


def _report(**kw):
    return dict(version="swing_net_profit_v1",asof="20260907",validated=kw.get("validated",False),
                trade_validated=kw.get("trade_validated",False),oos_days=40, hit_rate=.5,net_return_mean=-.01)


def test_research_scores_cannot_become_probabilities_or_buy_flags():
    data=_df(); before=data.copy(deep=True)
    out=attach_predictions(data,_picks(),_report())
    assert out.SWING_PROB.isna().all()
    assert out.SWING_STATUS.eq("RESEARCH").all()
    assert out.PRODUCTION_BUY.tolist()==[0,1] and out['켈리_수량'].tolist()==[0,5]
    pd.testing.assert_frame_equal(data,before)
    only_accuracy=attach_predictions(data,_picks(),_report(validated=True))
    assert only_accuracy.SWING_PROB.isna().all()
    valid=attach_predictions(data,_picks(),_report(validated=True,trade_validated=True))
    assert valid.SWING_PROB.tolist()==[.75,.7]


def test_snapshot_mismatch_and_corruption_never_promote(tmp_path):
    report=_report(validated=True);report['asof']='20260904'
    pd.testing.assert_frame_equal(attach_predictions(_df(),_picks(),report),_df())
    (tmp_path/'swing_snapshot_latest.json').write_text('{broken')
    pd.testing.assert_frame_equal(enrich_from_snapshot(_df(),tmp_path),_df())
    dup=pd.concat([_picks(),_picks()],ignore_index=True)
    assert attach_predictions(_df(),dup,_report(validated=True)).SWING_PROB.isna().all()


def test_atomic_snapshot_binds_the_report_and_predictions(tmp_path):
    (tmp_path/'swing_snapshot_latest.json').write_text(json.dumps(
        {'predictions':_picks().to_dict('records'),'validation':_report(validated=True,trade_validated=True)}))
    out=enrich_from_snapshot(_df(),tmp_path)
    assert out.SWING_PROB.notna().all()


def test_rerun_cannot_rewrite_first_published_rank(tmp_path, monkeypatch):
    from services import swing_model as model
    monkeypatch.setattr(model, 'load_feature_panel', lambda *a, **k: pd.DataFrame())
    picks = _picks()
    monkeypatch.setattr(model, 'train_and_predict', lambda *a, **k: (picks, _report(), pd.DataFrame()))
    model.run(tmp_path, '20260907')
    first = (tmp_path/'swing_snapshot_20260907.json').read_text()
    picks.loc[0, 'rank'] = 9
    model.run(tmp_path, '20260907')
    assert (tmp_path/'swing_snapshot_20260907.json').read_text() == first
    assert json.loads((tmp_path/'swing_snapshot_latest.json').read_text())['predictions'][0]['rank'] == 9

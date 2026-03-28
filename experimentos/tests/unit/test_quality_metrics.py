import pytest
import numpy as np
import pandas as pd
from recsys.evaluation.quality_metrics import Evaluator


def _preds_df(rows):
    return pd.DataFrame(rows, columns=['user', 'item', 'prediction'])


def _truth_df(rows):
    return pd.DataFrame(rows, columns=['user', 'item', 'true_value'])


def test_calculate_rmse():
    preds = _preds_df([(1, 101, 4.0), (1, 102, 3.0), (2, 101, 2.0)])
    truth = _truth_df([(1, 101, 5.0), (1, 102, 3.0), (2, 101, 1.0)])
    rmse = Evaluator.calculate_rmse(preds, truth)
    expected = np.sqrt(((5-4)**2 + (3-3)**2 + (1-2)**2) / 3)
    assert np.isclose(rmse, expected)


def test_calculate_user_ndcg():
    # Each user needs >= 2 items for ndcg_score to be defined
    preds = _preds_df([(1, 101, 4.0), (1, 102, 3.0), (2, 101, 2.0), (2, 102, 1.5)])
    truth = _truth_df([(1, 101, 5.0), (1, 102, 3.0), (2, 101, 1.0), (2, 102, 4.0)])
    ndcg = Evaluator.calculate_user_ndcg(
        preds.values.tolist(), truth.values.tolist(), k=2
    )
    assert ndcg is not None
    assert 0 <= ndcg <= 1


def test_calculate_f1_global():
    preds = _preds_df([(1, 101, 4.0), (1, 102, 2.0), (2, 101, 3.8)])
    truth = _truth_df([(1, 101, 4.5), (1, 102, 2.0), (2, 101, 3.9)])
    f1 = Evaluator.calculate_f1_global(preds, truth, threshold=3.5)
    assert 0 <= f1 <= 1


def test_empty_predictions():
    preds = _preds_df([])
    truth = _truth_df([])
    result = Evaluator.calculate_rmse(preds, truth)
    assert result is None


def test_mismatched_data():
    preds = _preds_df([(1, 101, 4.0), (1, 102, 3.0)])
    truth = _truth_df([(1, 101, 5.0), (2, 103, 1.0)])
    f1 = Evaluator.calculate_f1_global(preds, truth, threshold=3.5)
    # After inner merge only 1 pair matches; f1 may be 0 or None
    assert f1 is None or 0 <= f1 <= 1


def test_threshold_edge_cases():
    preds = _preds_df([(1, 101, 5.0), (1, 102, 1.0)])
    truth = _truth_df([(1, 101, 5.0), (1, 102, 1.0)])

    f1_high = Evaluator.calculate_f1_global(preds, truth, threshold=4.9)
    assert 0 <= f1_high <= 1

    f1_low = Evaluator.calculate_f1_global(preds, truth, threshold=1.1)
    assert 0 <= f1_low <= 1

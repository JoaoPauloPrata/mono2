import pytest
import numpy as np
import pandas as pd
from src.Metrics.Evaluator import Evaluator
def test_calculate_rmse():
    """
    Testa o cálculo do RMSE com valores conhecidos
    """
    predictions = [
        (1, 101, 4.0),
        (1, 102, 3.0),
        (2, 101, 2.0)
    ]
    
    true_values = [
        (1, 101, 5.0),
        (1, 102, 3.0),
        (2, 101, 1.0)
    ]
    
    rmse = Evaluator.calculate_rmse(predictions, true_values)
    expected_rmse = np.sqrt(((5-4)**2 + (3-3)**2 + (1-2)**2) / 3)
    assert np.isclose(rmse, expected_rmse)

def test_calculate_user_ndcg():
    """
    Testa o cálculo do NDCG por usuário
    """
    predictions = [
        (1, 101, 4.0),
        (1, 102, 3.0),
        (2, 101, 2.0)
    ]
    
    true_values = [
        (1, 101, 5.0),
        (1, 102, 3.0),
        (2, 101, 1.0)
    ]
    
    ndcg = Evaluator.calculate_user_ndcg(predictions, true_values, k=2)
    assert ndcg is not None
    assert 0 <= ndcg <= 1

def test_calculate_f1_global():
    """
    Testa o cálculo do F1 global
    """
    predictions = [
        (1, 101, 4.0),
        (1, 102, 2.0),
        (2, 101, 3.8)
    ]
    
    true_values = [
        (1, 101, 4.5),
        (1, 102, 2.0),
        (2, 101, 3.9)
    ]
    
    f1 = Evaluator.calculate_f1_global(predictions, true_values, threshold=3.5)
    assert 0 <= f1 <= 1

def test_empty_predictions():
    """
    Testa o comportamento com listas vazias
    """
    predictions = []
    true_values = []
    
    with pytest.raises(ValueError):
        Evaluator.calculate_rmse(predictions, true_values)

def test_mismatched_data():
    """
    Testa o comportamento com dados não correspondentes
    """
    predictions = [
        (1, 101, 4.0),
        (1, 102, 3.0)
    ]
    
    true_values = [
        (1, 101, 5.0),
        (2, 103, 1.0)  # usuário e item diferentes
    ]
    
    f1 = Evaluator.calculate_f1_global(predictions, true_values, threshold=3.5)
    assert 0 <= f1 <= 1

def test_threshold_edge_cases():
    """
    Testa casos limite do threshold para F1
    """
    predictions = [
        (1, 101, 5.0),
        (1, 102, 1.0)
    ]
    
    true_values = [
        (1, 101, 5.0),
        (1, 102, 1.0)
    ]
    
    # Teste com threshold extremo alto
    f1_high = Evaluator.calculate_f1_global(predictions, true_values, threshold=4.9)
    assert 0 <= f1_high <= 1
    
    # Teste com threshold extremo baixo
    f1_low = Evaluator.calculate_f1_global(predictions, true_values, threshold=1.1)
    assert 0 <= f1_low <= 1

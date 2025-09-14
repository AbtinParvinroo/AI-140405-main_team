from regression_evaluator import RegressionEvaluator
import numpy as np
import pytest

@pytest.mark.parametrize(
    "y_true, y_pred, n_features",
    [
        (np.array([3.0, -0.5, 2.0, 7.0]), np.array([2.5, 0.0, 2.0, 8.0]), 2),
        (np.array([0, 0, 0, 0]), np.array([0, 0, 0, 0]), 1),
        (np.array([1,2,3,4,5,6,7,8,9,10]), np.array([1.1,1.9,3.0,4.2,5.1,5.9,7.0,7.8,9.2,10.1]), 5),
        (np.array([[1,2],[3,4],[5,6]]), np.array([[0.9,2.1],[3.1,3.9],[5.0,6.2]]), 2)
    ]
)

def test_regression_evaluator_param(y_true, y_pred, n_features):
    evaluator = RegressionEvaluator(y_true, y_pred, n_features=n_features)

    assert isinstance(evaluator.mae(), float)
    assert isinstance(evaluator.mse(), float)
    assert isinstance(evaluator.rmse(), float)
    assert isinstance(evaluator.r2(), float)

    adj_r2 = evaluator.adjusted_r2()
    if adj_r2 is not None:
        assert isinstance(adj_r2, float)
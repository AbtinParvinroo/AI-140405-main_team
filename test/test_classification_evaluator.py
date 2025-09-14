
from classification_evaluator import ClassificationEvaluator
import numpy as np
import pytest

@pytest.mark.parametrize(
    "y_true, y_pred, y_proba, average",
    [
        (np.array([0,1,0,1]), np.array([0,1,1,1]), np.array([0.2,0.8,0.6,0.9]), 'binary'),
        (np.array([0,1,2,1]), np.array([0,1,2,0]), np.array(
            [[0.8,0.1,0.1],
            [0.1,0.7,0.2],
            [0.1,0.1,0.8],
            [0.5,0.3,0.2]]), 'macro'),

        (np.array([1,0,1]), np.array([1,0,0]), None, 'binary')
    ]
)

def test_classification_evaluator(y_true, y_pred, y_proba, average):
    evaluator = ClassificationEvaluator(y_true, y_pred, y_proba=y_proba, average=average)

    assert isinstance(evaluator.accuracy(), float)
    assert isinstance(evaluator.precision(), float)
    assert isinstance(evaluator.recall(), float)
    assert isinstance(evaluator.f1(), float)
    if y_proba is not None:
        roc_auc_val = evaluator.roc_auc()
        assert isinstance(roc_auc_val, float)

    metrics = evaluator.all_metrics()
    assert isinstance(metrics, dict)
    for k, v in metrics.items():
        assert isinstance(v, float)

    summary = evaluator.summary(verbose=False)
    assert summary == metrics
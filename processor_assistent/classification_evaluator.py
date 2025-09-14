from sklearn.metrics import accuracy_score, precision_score, f1_score
from sklearn.metrics import recall_score, roc_auc_score
import numpy as np

class ClassificationEvaluator:
    def __init__(self, y_true:np.ndarray, y_pred:np.ndarray, y_proba:np.ndarray=None, average:str='binary'):
        self.y_true = np.asarray(y_true)
        self.y_pred = np.asarray(y_pred)
        self.y_proba = np.asarray(y_proba) if y_proba is not None else None
        self.average = average
        if self.y_true.shape != self.y_pred.shape:
            raise ValueError("y_true and y_pred must have the same shape")

        if self.y_proba is not None and self.y_proba.shape[0] != self.y_true.shape[0]:
            raise ValueError("y_proba must have the same number of samples as y_true")

    def accuracy(self):
        return accuracy_score(self.y_true, self.y_pred)

    def precision(self):
        return precision_score(self.y_true, self.y_pred, average=self.average, zero_division=0)

    def recall(self):
        return recall_score(self.y_true, self.y_pred, average=self.average, zero_division=0)

    def f1(self):
        return f1_score(self.y_true, self.y_pred, average=self.average, zero_division=0)

    def roc_auc(self):
        if self.y_proba is None:
            raise ValueError("[ERROR] y_proba must be provided for ROC-AUC")

        if self.y_proba.ndim == 1:
            return roc_auc_score(self.y_true, self.y_proba)

        return roc_auc_score(self.y_true, self.y_proba, average=self.average, multi_class='ovo')

    def all_metrics(self):
        metrics = {
            "Accuracy": self.accuracy(),
            "Precision": self.precision(),
            "Recall": self.recall(),
            "F1-Score": self.f1()
        }

        if self.y_proba is not None:
            metrics["ROC-AUC"] = self.roc_auc()

        return metrics

    def summary(self, verbose: bool = True):
        metrics = self.all_metrics()
        if verbose:
            print("[Classification Metrics]")
            for k, v in metrics.items():
                print(f"{k}: {v:.4f}")

        return metrics
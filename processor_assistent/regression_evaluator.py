
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

class RegressionEvaluator:
    def __init__(self, y_true:np.ndarray, y_pred:np.ndarray, n_features:int=None):
        self.y_true = np.asarray(y_true)
        self.y_pred = np.asarray(y_pred)
        self.n_samples = self.y_true.shape[0]
        self.n_features = n_features
        if self.y_true.shape != self.y_pred.shape:
            raise ValueError("y_true and y_pred must have the same shape")

        self.n_outputs = 1 if self.y_true.ndim == 1 else self.y_true.shape[1]

    def mae(self):
        return mean_absolute_error(self.y_true, self.y_pred, multioutput='uniform_average')

    def mse(self):
        return mean_squared_error(self.y_true, self.y_pred, multioutput='uniform_average')

    def rmse(self):
        return np.sqrt(self.mse())

    def r2(self):
        return r2_score(self.y_true, self.y_pred, multioutput='uniform_average')

    def adjusted_r2(self):
        if self.n_features is None:
            raise ValueError("[ERROR] n_features must be provided for Adjusted R²")

        if self.y_true.ndim > 1 and self.y_true.shape[1] > 1:
            return None

        r2 = self.r2()
        adj_r2 = 1 - (1 - r2) * (self.n_samples - 1) / (self.n_samples - self.n_features - 1)
        return adj_r2

    def all_metrics(self):
        metrics = {
            "MAE": self.mae(),
            "MSE": self.mse(),
            "RMSE": self.rmse(),
            "R2": self.r2()
        }

        adj_r2 = self.adjusted_r2()
        if adj_r2 is not None:
            metrics["Adjusted_R2"] = adj_r2

        return metrics

    def summary(self, verbose: bool = True):
        metrics = self.all_metrics()
        if verbose:
            print("[Regression Metrics]")
            for k, v in metrics.items():
                print(f"{k}: {v:.4f}")

        return metrics
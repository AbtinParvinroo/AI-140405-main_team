
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

class CorrelationAnalyzer:
    def __init__(self, df:pd.DataFrame, features:list=None, verbose:bool=False):
        self.df = df.copy()
        self.features = features if features is not None else df.select_dtypes(include=np.number).columns.tolist()
        self.verbose = verbose
        self.corr_matrix = None

    def compute(self, method: str = 'pearson', absolute: bool = False) -> pd.DataFrame:
        if method not in ['pearson', 'spearman', 'kendall']:
            raise ValueError("method must be 'pearson', 'spearman', or 'kendall'")

        numeric_df = self.df[self.features].select_dtypes(include=np.number)
        self.corr_matrix = numeric_df.corr(method=method)
        if absolute:
            self.corr_matrix = self.corr_matrix.abs()

        if self.verbose:
            print(f"[INFO] Correlation computed using {method}{' (absolute)' if absolute else ''}")

        return self.corr_matrix

    def plot_heatmap(self, figsize=(8,6), annot=True, cmap="coolwarm"):
        if self.corr_matrix is None:
            raise ValueError("compute() must be called before plot_heatmap()")

        plt.figure(figsize=figsize)
        sns.heatmap(self.corr_matrix, annot=annot, fmt=".2f", cmap=cmap)
        plt.title("Correlation Heatmap")
        plt.show()

    def top_correlations(self, top_n: int = 10, threshold: float = 0.0, absolute: bool = True) -> pd.Series:
        if self.corr_matrix is None:
            raise ValueError("compute() must be called before top_correlations()")

        corr_mat = self.corr_matrix.copy()
        if absolute:
            corr_mat = corr_mat.abs()

        corr_mat.values[np.tril_indices_from(corr_mat)] = np.nan
        corr_unstack = corr_mat.unstack().dropna()
        corr_unstack = corr_unstack[corr_unstack > threshold]
        return corr_unstack.sort_values(ascending=False).head(top_n)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import category_encoders as ce
import pandas as pd
import numpy as np

class FeatureAnalyze:
    def __init__(self, df:pd.DataFrame, X:np.ndarray, y:np.ndarray):
        self.df = df
        self.X = X
        self.y = y
        self.X_scaled = None
        self.scaler_model = None
        self.pca_model = None

    def scaler(self, mode: str = 'standard'):
        if mode == 'standard':
            scaler = StandardScaler()

        elif mode == 'min_max':
            scaler = MinMaxScaler()

        elif mode == 'robust':
            scaler = RobustScaler()

        else:
            raise ValueError(f"[ERROR] Scaler {mode} not found!")

        self.X_scaled = scaler.fit_transform(self.X)
        self.scaler_model = scaler
        return self.X_scaled

    def train_splitter(self, train_rate: float = 0.8, random_state: int = 42, stratify: bool = False):
        if self.X_scaled is None:
            raise ValueError("[ERROR] You must run scaler() before splitting the data!")

        stratify_y = self.y if stratify else None
        return train_test_split(self.X_scaled, self.y, train_size=train_rate, random_state=random_state, shuffle=True, stratify=stratify_y)

    def variance(self, column: np.ndarray, threshold: list = [0.05, 0.95]):
        col_var = np.var(column)
        status = "fit"
        if col_var >= threshold[1]:
            status = "too high"

        elif col_var <= threshold[0]:
            status = "too low"

        return {"status": status, "variance": col_var}

    def apply_pca(self, n_components: int = 2, return_df: bool = True):
        if self.X_scaled is None:
            raise ValueError("[ERROR] You must run scaler() before applying PCA!")

        self.pca_model = PCA(n_components=n_components)
        X_pca = self.pca_model.fit_transform(self.X_scaled)
        if return_df:
            columns = [f"PC{i+1}" for i in range(n_components)]
            return pd.DataFrame(X_pca, columns=columns), self.pca_model.explained_variance_ratio_

        else:
            return X_pca, self.pca_model.explained_variance_ratio_


class CategoricalEncoder:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.encoders = {}

    def one_hot(self, columns: list):
        for col in columns:
            ohe = pd.get_dummies(self.df[col], prefix=col)
            self.df = pd.concat([self.df.drop(col, axis=1), ohe], axis=1)
            self.encoders[col] = 'OneHot'

        return self.df

    def label(self, columns: list):
        for col in columns:
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col])
            self.encoders[col] = le

        return self.df

    def target(self, columns: list, target: str):
        for col in columns:
            te = ce.TargetEncoder(cols=[col])
            self.df[col] = te.fit_transform(self.df[col], self.df[target])
            self.encoders[col] = te

        return self.df

    def get_encoder(self, column: str):
        return self.encoders.get(column, None)
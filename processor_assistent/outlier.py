from sklearn.neighbors import LocalOutlierFactor
from tensorflow.keras.layers import Input, Dense
from sklearn.ensemble import IsolationForest
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from sklearn.svm import OneClassSVM
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class DataCleaner:
    def __init__(self, df:pd.DataFrame, verbose:bool=False):
        self.df = df.copy()
        self.features = df.columns.tolist()
        self.X = self.df[self.features].values
        self.verbose = verbose

    def detect_autoencoder(self, hidden_size:int=2, epochs:int=20, threshold:float=None) -> np.ndarray:
        input_dim = self.X.shape[1]
        input_layer = Input(shape=(input_dim,))
        encoded = Dense(hidden_size, activation='relu',
            activity_regularizer=regularizers.l1(1e-5))(input_layer)

        decoded = Dense(input_dim, activation='linear')(encoded)
        autoencoder = Model(input_layer, decoded)
        autoencoder.compile(optimizer='adam', loss='mse')
        autoencoder.fit(self.X, self.X, epochs=epochs, verbose=0)
        reconstructions = autoencoder.predict(self.X, verbose=0)
        errors = np.mean(np.square(self.X - reconstructions), axis=1)
        if threshold is None:
            threshold = np.percentile(errors, 95)

        mask = errors > threshold
        if self.verbose:
            logger.debug(f"[AE] Threshold={threshold:.4f}, Outliers={mask.sum()}")

        return np.array(mask)

    def detect_lof(self, contamination: float = 0.05) -> np.ndarray:
        lof = LocalOutlierFactor(n_neighbors=20, contamination=contamination)
        pred = lof.fit_predict(self.X)
        mask = pred == -1
        if self.verbose:
            logger.debug(f"[LOF] Outliers={mask.sum()}")

        return np.array(mask)

    def detect_ocsvm(self, contamination: float = 0.05) -> np.ndarray:
        ocsvm = OneClassSVM(nu=contamination, kernel='rbf', gamma='auto')
        ocsvm.fit(self.X)
        pred = ocsvm.predict(self.X)
        mask = pred == -1
        if self.verbose:
            logger.debug(f"[OCSVM] Outliers={mask.sum()}")

        return np.array(mask)

    def detect_isoforest(self, contamination: float = 0.05) -> np.ndarray:
        iso = IsolationForest(contamination=contamination, random_state=42)
        iso.fit(self.X)
        pred = iso.predict(self.X)
        mask = pred == -1
        if self.verbose:
            logger.debug(f"[IsoForest] Outliers={mask.sum()}")

        return np.array(mask)

    def detect_zscore(self, threshold: float = 3.0) -> np.ndarray:
        z_scores = np.abs((self.X - np.mean(self.X, axis=0)) / np.std(self.X, axis=0))
        mask = (z_scores > threshold).any(axis=1)
        if self.verbose:
            logger.debug(f"[Z-score] Outliers={mask.sum()}")

        return np.array(mask)

    def detect_iqr(self, factor: float = 1.5) -> np.ndarray:
        Q1 = np.percentile(self.X, 25, axis=0)
        Q3 = np.percentile(self.X, 75, axis=0)
        IQR = Q3 - Q1
        mask = ((self.X < (Q1 - factor * IQR)) | (self.X > (Q3 + factor * IQR))).any(axis=1)
        if self.verbose:
            logger.debug(f"[IQR] Outliers={mask.sum()}")

        return np.array(mask)

    def clean(self, detect_method='iqr', replace=None, fill_missing=True) -> pd.DataFrame:
        method_map = {
            'ae': 'detect_autoencoder',
            'lof': 'detect_lof',
            'ocsvm': 'detect_ocsvm',
            'isoforest': 'detect_isoforest',
            'zscore': 'detect_zscore',
            'iqr': 'detect_iqr'
        }

        X_temp = self.df[self.features].copy()
        if fill_missing:
            X_temp = X_temp.fillna(X_temp.mean())

        mask = getattr(self, method_map[detect_method])()

        df_cleaned = self.df.copy()

        if replace in ['mean', 'median']:
            for col in self.features:
                if replace == 'mean':
                    df_cleaned.loc[mask, col] = df_cleaned[col].mean()

                else:
                    df_cleaned.loc[mask, col] = df_cleaned[col].median()
            if fill_missing:
                df_cleaned[self.features] = df_cleaned[self.features].fillna(df_cleaned[self.features].mean())

        elif replace in ['ffill', 'bfill']:
            df_cleaned.loc[mask, self.features] = np.nan
            if replace == 'ffill':
                df_cleaned = df_cleaned.ffill().bfill()

            else:
                df_cleaned = df_cleaned.bfill().ffill()

        elif replace and replace.startswith("percentile:"):
            try:
                p = float(replace.split(":")[1])
                for col in self.features:
                    thresh = np.percentile(df_cleaned[col], p)
                    df_cleaned.loc[mask, col] = thresh

            except Exception as e:
                logger.warning(f"Percentile replace failed: {e}")

            if fill_missing:
                df_cleaned[self.features] = df_cleaned[self.features].fillna(df_cleaned[self.features].mean())

        else:
            df_cleaned = df_cleaned.loc[~mask].dropna(subset=self.features).reset_index(drop=True)

        return df_cleaned
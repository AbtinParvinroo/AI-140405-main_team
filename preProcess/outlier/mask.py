from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Input, Dense
from sklearn.ensemble import IsolationForest
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from sklearn.svm import OneClassSVM
import numpy as np

class OutlierDetector:
    def __init__(self, df, features, verbose=False):
        self.df = df.copy()
        self.features = features
        self.scaler = StandardScaler()
        self.X = self.scaler.fit_transform(self.df[self.features].values)
        self.results = {}
        self.verbose = verbose

    def clear_outliers(self):
        for col in ['outlier_lof', 'outlier_ocsvm', 'outlier_iso', 'outlier_ae']:
            if col in self.df.columns:
                self.df.drop(columns=[col], inplace=True)
        self.results = {}
        if self.verbose:
            print("All previous outlier columns cleared.")

    def detect_lof(self, n_neighbors=20, contamination=0.05):
        lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
        y_pred = lof.fit_predict(self.X)
        self.df['outlier_lof'] = y_pred
        self.results['lof'] = y_pred
        if self.verbose:
            print(f"LOF detected {np.sum(y_pred == -1)} outliers.")

        return self.df[self.df['outlier_lof'] == -1]

    def detect_ocsvm(self, kernel='rbf', nu=0.05, gamma='scale'):
        oc_svm = OneClassSVM(kernel='rbf', nu=0.05, gamma=0.1)
        oc_svm.fit(self.X)
        y_pred = oc_svm.predict(self.X)
        self.df['outlier_ocsvm'] = y_pred
        self.results['ocsvm'] = y_pred
        if self.verbose:
            print(f"One-Class SVM detected {np.sum(y_pred == -1)} outliers.")

        return self.df[self.df['outlier_ocsvm'] == -1]

    def detect_isolation_forest(self, contamination=0.05, random_state=42):
        iso = IsolationForest(contamination=contamination, random_state=random_state)
        iso.fit(self.X)
        y_pred = iso.predict(self.X)
        self.df['outlier_iso'] = y_pred
        self.results['isolation_forest'] = y_pred
        if self.verbose:
            print(f"Isolation Forest detected {np.sum(y_pred == -1)} outliers.")

        return self.df[self.df['outlier_iso'] == -1]

    def _build_autoencoder(self, input_dim, l1_reg=1e-5):
        input_layer = Input(shape=(input_dim,))
        encoded = Dense(64, activation='relu', activity_regularizer=regularizers.l1(l1_reg))(input_layer)
        encoded = Dense(32, activation='relu')(encoded)
        encoded = Dense(16, activation='relu')(encoded)
        decoded = Dense(32, activation='relu')(encoded)
        decoded = Dense(64, activation='relu')(decoded)
        decoded = Dense(input_dim, activation='linear')(decoded)
        autoencoder = Model(inputs=input_layer, outputs=decoded)
        autoencoder.compile(optimizer='adam', loss='mse')
        return autoencoder

    def detect_autoencoder(self, epochs=50, batch_size=32, threshold=None, l1_reg=1e-5):
        input_dim = self.X.shape[1]
        autoencoder = self._build_autoencoder(input_dim, l1_reg)
        if self.verbose:
            print("Training Autoencoder...")

        autoencoder.fit(self.X, self.X, epochs=epochs, batch_size=batch_size, shuffle=True, verbose=self.verbose)
        X_pred = autoencoder.predict(self.X)
        mse = np.mean(np.power(self.X - X_pred, 2), axis=1)
        if threshold is None:
            threshold = np.percentile(mse, 95)
            if self.verbose:
                print(f"Autoencoder threshold set at 95th percentile: {threshold:.6f}")

        outlier_mask = mse > threshold
        self.df['outlier_ae'] = 0
        self.df.loc[outlier_mask, 'outlier_ae'] = 1
        self.results['autoencoder'] = self.df['outlier_ae'].values
        if self.verbose:
            print(f"Autoencoder detected {np.sum(outlier_mask)} outliers.")

        return self.df[self.df['outlier_ae'] == 1]

    def get_summary(self):
        summary = {}
        for method, preds in self.results.items():
            if method == 'autoencoder':
                count = np.sum(preds == 1)

            else:
                count = np.sum(preds == -1)

            summary[method] = count

        return summary

    def replace_outliers(self, method_name, replacement='median'):
        if method_name not in self.results:
            raise ValueError(f"No results found for method '{method_name}'. Run detection first.")

        mask = None
        preds = self.results[method_name]
        if method_name == 'autoencoder':
            mask = preds == 1

        else:
            mask = preds == -1

        if replacement == 'median':
            rep_val = self.df.loc[~mask, self.features].median()

        elif replacement == 'mean':
            rep_val = self.df.loc[~mask, self.features].mean()

        else:
            raise ValueError("Replacement must be 'median' or 'mean'")

        self.df.loc[mask, self.features] = rep_val.values
        if self.verbose:
            print(f"Outliers from '{method_name}' replaced with {replacement}.")

        return self.df
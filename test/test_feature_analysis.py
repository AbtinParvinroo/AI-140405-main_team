from feature_analysis import FeatureAnalyze, CategoricalEncoder
import pandas as pd
import numpy as np
import pytest

@pytest.fixture
def sample_df():
    data = {
        'A': np.random.rand(100),
        'B': np.random.rand(100),
        'C': ['cat', 'dog', 'mouse', 'cat', 'dog']*20,
        'target': np.random.randint(0, 2, 100)
    }

    return pd.DataFrame(data)

def test_scaler_and_split(sample_df):
    X = sample_df[['A','B']].values
    y = sample_df['target'].values
    fa = FeatureAnalyze(sample_df, X, y)
    X_scaled = fa.scaler('standard')
    assert X_scaled.shape == X.shape
    assert np.isclose(X_scaled.mean(), 0, atol=1e-1)
    assert np.isclose(X_scaled.std(), 1, atol=1e-1)

    X_train, X_test, y_train, y_test = fa.train_splitter()
    assert X_train.shape[0] > 0 and X_test.shape[0] > 0

def test_variance(sample_df):
    X = sample_df['A'].values
    fa = FeatureAnalyze(sample_df, sample_df[['A','B']].values, sample_df['target'].values)
    result = fa.variance(X, threshold=[0.01,0.99])
    assert 'status' in result
    assert 'variance' in result

def test_pca(sample_df):
    X = sample_df[['A','B']].values
    fa = FeatureAnalyze(sample_df, X, sample_df['target'].values)
    fa.scaler()
    X_pca_df, var_ratio = fa.apply_pca(n_components=2)
    assert X_pca_df.shape[1] == 2
    assert np.isclose(var_ratio.sum(), sum(var_ratio), atol=1e-6)

def test_categorical_encoder(sample_df):
    ce_instance_ohe = CategoricalEncoder(sample_df.copy())
    df_ohe = ce_instance_ohe.one_hot(['C'])
    assert 'C_cat' in df_ohe.columns

    ce_instance_label = CategoricalEncoder(sample_df.copy())
    df_label = ce_instance_label.label(['C'])
    assert df_label['C'].dtype in [np.int32, np.int64]

    ce_instance_target = CategoricalEncoder(sample_df.copy())
    df_target = ce_instance_target.target(['C'], target='target')
    assert 'C' in df_target.columns
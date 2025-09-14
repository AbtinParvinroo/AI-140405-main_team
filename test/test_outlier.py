from outlier import DataCleaner
import pandas as pd
import numpy as np
import pytest

@pytest.fixture
def sample_df():
    np.random.seed(42)
    df = pd.DataFrame({
        "A": np.random.randn(100),
        "B": np.random.randn(100),
        "C": np.random.randn(100)
    })

    df.loc[0, "A"] = np.nan
    df.loc[1, "B"] = np.nan
    df.loc[2, "C"] = np.nan
    df.loc[3, "A"] = 10
    df.loc[4, "B"] = -10
    return df

def test_outlier_detection_methods(sample_df):
    df_filled = sample_df.fillna(sample_df.mean())
    cleaner = DataCleaner(df_filled, verbose=True)
    methods = ['ae', 'lof', 'ocsvm', 'isoforest', 'zscore', 'iqr']
    method_map = {
        'ae': 'detect_autoencoder',
        'lof': 'detect_lof',
        'ocsvm': 'detect_ocsvm',
        'isoforest': 'detect_isoforest',
        'zscore': 'detect_zscore',
        'iqr': 'detect_iqr'
    }

    for method in methods:
        mask = getattr(cleaner, method_map[method])()
        mask = np.array(mask)
        assert isinstance(mask, np.ndarray)
        assert mask.shape[0] == len(df_filled)
        assert np.isin(mask, [True, False]).all()

@pytest.mark.parametrize("replace_method", ["mean", "median", "ffill", "bfill", "percentile:90"])
def test_clean_replace_methods(sample_df, replace_method):
    cleaner = DataCleaner(sample_df)
    df_cleaned = cleaner.clean(detect_method='iqr', replace=replace_method, fill_missing=True)

    assert df_cleaned.isna().sum().sum() == 0

def test_clean_without_replace(sample_df):
    cleaner = DataCleaner(sample_df)
    df_cleaned = cleaner.clean(detect_method='iqr', replace=None)
    assert df_cleaned.shape[0] <= sample_df.shape[0]
    assert df_cleaned.isna().sum().sum() == 0
from correlation import CorrelationAnalyzer
import pandas as pd
import numpy as np
import pytest

@pytest.fixture
def sample_df():
    np.random.seed(42)
    data = {
        'A': np.random.randn(100),
        'B': np.random.randn(100),
        'C': np.random.randn(100),
        'D': ['text']*100
    }

    data['A'][0] = np.nan
    data['B'][1] = np.nan
    data['C'][2] = np.nan
    return pd.DataFrame(data)

def test_compute_and_top_correlations(sample_df):
    analyzer = CorrelationAnalyzer(sample_df, verbose=True)
    corr_matrix = analyzer.compute(method='pearson', absolute=True)
    assert isinstance(corr_matrix, pd.DataFrame)
    assert corr_matrix.shape[0] == 3

    top_corr = analyzer.top_correlations(top_n=2, threshold=0.0)
    assert isinstance(top_corr, pd.Series)
    assert len(top_corr) <= 3

def test_invalid_method(sample_df):
    analyzer = CorrelationAnalyzer(sample_df)
    with pytest.raises(ValueError):
        analyzer.compute(method='invalid')

def test_plot_heatmap_requires_compute(sample_df):
    analyzer = CorrelationAnalyzer(sample_df)
    import matplotlib
    matplotlib.use('Agg')
    with pytest.raises(ValueError):
        analyzer.plot_heatmap()
from plotter import Plotter
import numpy as np
import matplotlib
import pytest

matplotlib.use('Agg')

@pytest.fixture
def sample_data():
    X = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 3, 5, 7, 11])
    data = np.random.randn(100)
    bar_labels = ['A', 'B', 'C']
    bar_values = [10, 20, 15]
    pie_values = [30, 40, 30]
    pie_labels = ['X', 'Y', 'Z']
    heatmap_data = np.random.rand(5,5)
    return X, y, data, bar_labels, bar_values, pie_values, pie_labels, heatmap_data

def test_line_plotter(sample_data):
    X, y, _, _, _, _, _, _ = sample_data
    plotter = Plotter()
    plotter.line_plotter(X, y)

def test_scatter_plotter(sample_data):
    X, y, _, _, _, _, _, _ = sample_data
    plotter = Plotter()
    plotter.scatter_plotter(X, y)

def test_bar_plotter(sample_data):
    _, _, _, bar_labels, bar_values, _, _, _ = sample_data
    plotter = Plotter()
    plotter.bar_plotter(bar_labels, bar_values)

def test_histogram_plotter(sample_data):
    _, _, data, _, _, _, _, _ = sample_data
    plotter = Plotter()
    plotter.histogram_plotter(data, bins=5)

def test_pie_plotter(sample_data):
    _, _, _, _, _, pie_values, pie_labels, _ = sample_data
    plotter = Plotter()
    plotter.pie_plotter(pie_values, pie_labels)

def test_heatmap_plotter(sample_data):
    _, _, _, _, _, _, _, heatmap_data = sample_data
    plotter = Plotter()
    plotter.heatmap_plotter(heatmap_data)

def test_missing_data_raises(sample_data):
    plotter = Plotter()
    with pytest.raises(ValueError):
        plotter.line_plotter()

    with pytest.raises(ValueError):
        plotter.scatter_plotter()

    with pytest.raises(ValueError):
        plotter.bar_plotter()

    with pytest.raises(ValueError):
        plotter.histogram_plotter()

    with pytest.raises(ValueError):
        plotter.pie_plotter()

    with pytest.raises(ValueError):
        plotter.heatmap_plotter()
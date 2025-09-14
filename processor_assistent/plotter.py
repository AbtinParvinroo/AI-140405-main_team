from typing import Optional, List, Dict
import matplotlib.pyplot as plt
import numpy as np

def grid_handler(grid:bool=False):
    if grid:
        plt.grid()

def label_handler(labels:Optional[Dict[str, str]]=None):
    if labels:
        plt.xlabel(labels.get('xlabel', ''))
        plt.ylabel(labels.get('ylabel', ''))

class Plotter:
    def __init__(self, marker:str='o', line_style:str='-', labels: Optional[Dict[str, str]]=None):
        self.marker = marker
        self.line_style = line_style
        self.labels = labels

    def line_plotter(self, X:np.ndarray=None, y:np.ndarray=None, grid:bool=True):
        if X is None or y is None:
            raise ValueError("X and y must be provided for line_plotter")

        grid_handler(grid)
        label_handler(self.labels)
        plt.plot(X, y, marker=self.marker, ls=self.line_style)
        plt.show()

    def scatter_plotter(self, X:np.ndarray=None, y:np.ndarray=None, grid:bool=True):
        if X is None or y is None:
            raise ValueError("X and y must be provided for scatter_plotter")

        grid_handler(grid)
        label_handler(self.labels)
        plt.scatter(X, y)
        plt.show()

    def bar_plotter(self, x:List[str]=None, y:List[float]=None, grid:bool=True):
        if x is None or y is None:
            raise ValueError("x and y must be provided for bar_plotter")

        grid_handler(grid)
        label_handler(self.labels)
        plt.bar(x, y)
        plt.show()

    def histogram_plotter(self, data:np.ndarray=None, bins:int=10):
        if data is None:
            raise ValueError("data must be provided for histogram_plotter")

        plt.hist(data, bins=bins)
        plt.show()

    def pie_plotter(self, data:List[float]=None, labels:List[str]=None):
        if data is None:
            raise ValueError("data must be provided for pie_plotter")

        plt.pie(data, labels=labels, autopct='%1.1f%%')
        plt.show()

    def heatmap_plotter(self, data:np.ndarray=None):
        if data is None:
            raise ValueError("data must be provided for heatmap_plotter")

        plt.imshow(data, cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.show()
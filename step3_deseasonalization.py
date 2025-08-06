import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL

class TimeSeriesDeseasonalizer:
    def __init__(self, data):
        self.data = data.copy()
    
    def moving_average(self, column, window=12):
       
        print(f"\n Application de la moyenne glissante sur '{column}' avec une fenêtre de {window} périodes.")
        self.data[f'{column}_MA'] = self.data[column].rolling(window=window, center=True).mean()

        # Visualisation
        plt.figure(figsize=(12, 5))
        plt.plot(self.data[column], label='Original')
        plt.plot(self.data[f'{column}_MA'], label='Moyenne glissante', color='orange')
        plt.legend()
        plt.title(f'Moyenne glissante de {column}')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    def stl_decomposition(self, column, period):
        """
        Appliquer la décomposition STL pour isoler la tendance et la saisonnalité.
        """
        print(f"\n▶ Décomposition STL sur '{column}' avec une période de {period}")
        stl = STL(self.data[column], period=period, robust=True)
        result = stl.fit()

        # Visualisation
        result.plot()
        plt.suptitle(f'Décomposition STL de {column}', fontsize=14)
        plt.tight_layout()
        plt.show()
        return result.trend, result.seasonal, result.resid

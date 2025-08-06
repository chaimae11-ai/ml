import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

class StationarityTester:
    def __init__(self, data):
        self.data = data.copy()

    def plot_series(self, column):
        """
        Affiche la série temporelle avec son rolling mean et rolling std.
        """
        plt.figure(figsize=(12, 5))
        plt.plot(self.data[column], label='Original', color='blue')
        plt.plot(self.data[column].rolling(window=12).mean(), label='Moyenne mobile', color='orange')
        plt.plot(self.data[column].rolling(window=12).std(), label='Écart-type mobile', color='green')
        plt.title(f'Visualisation de la stationnarité - {column}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def adf_test(self, column):
        """
        Applique le test de Dickey-Fuller Augmenté (ADF) pour vérifier la stationnarité.
        """
        print(f"\n▶ Test ADF sur la colonne : {column}")
        result = adfuller(self.data[column].dropna())
        
        print(f"ADF Statistic : {result[0]:.4f}")
        print(f"p-value : {result[1]:.4f}")
        print("Valeurs critiques :")
        for key, value in result[4].items():
            print(f"    {key} : {value:.4f}")

        if result[1] <= 0.05:
            print("La série est stationnaire (p-value < 0.05).")
        else:
            print(" La série n'est pas stationnaire (p-value >= 0.05).")

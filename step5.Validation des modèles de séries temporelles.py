import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt


class TimeSeriesValidator:
    def __init__(self, data):
        self.data = data.copy()

    def simple_split_validation(self, column, model):
       
        print(f"\n Validation simple sur : {column}")
        series = self.data[column].dropna()
        train_size = int(len(series) * 0.8)
        train, test = series[:train_size], series[train_size:]

        model.fit(train)
        predictions = model.predict(n_periods=len(test))

        # Évaluation
        mae = mean_absolute_error(test, predictions)
        mse = mean_squared_error(test, predictions)
        print(f"MAE : {mae:.4f} | MSE : {mse:.4f}")

        # Affichage
        plt.figure(figsize=(12, 5))
        plt.plot(train.index, train, label="Train")
        plt.plot(test.index, test, label="Test", color="orange")
        plt.plot(test.index, predictions, label="Prévision", color="green")
        plt.legend()
        plt.title(f"Validation simple - {column}")
        plt.tight_layout()
        plt.grid(True)
        plt.show()

    def cross_validation_sliding(self, column, n_splits=5):
        
        print(f"\n▶ Validation croisée glissante sur : {column}")
        series = self.data[column].dropna().values
        tscv = TimeSeriesSplit(n_splits=n_splits)

        maes, mses = [], []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(series)):
            train, test = series[train_idx], series[test_idx]
            model = self._fit_arima(train)
            preds = model.predict(n_periods=len(test))

            mae = mean_absolute_error(test, preds)
            mse = mean_squared_error(test, preds)

            maes.append(mae)
            mses.append(mse)

            print(f"Fold {fold+1}: MAE={mae:.4f}, MSE={mse:.4f}")

        print("\n Moyenne des scores :")
        print(f"MAE moyen : {np.mean(maes):.4f}")
        print(f"MSE moyen : {np.mean(mses):.4f}")

    def _fit_arima(self, series):
        
        from pmdarima import auto_arima
        model = auto_arima(series, seasonal=False, suppress_warnings=True, stepwise=True)
        return model
Exemple d’utilisation générique :

from step6_validation import TimeSeriesValidator
from pmdarima import auto_arima

validator = TimeSeriesValidator(data)

#  validation simple avec ARIMA
model = auto_arima(data['MaColonne'].dropna(), seasonal=False, suppress_warnings=True)
validator.simple_split_validation(column="MaColonne", model=model)

# Validation croisée glissante
validator.cross_validation_sliding(column="MaColonne", n_splits=5)

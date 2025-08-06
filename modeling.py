import pandas as pd
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import numpy as np

class TimeSeriesModeler:
    def __init__(self, data):
        self.data = data.copy()

    def train_test_split(self, column, train_size=0.8):
        """
        Séparer la série temporelle en jeu d'entraînement et de test.
        """
        series = self.data[column].dropna()
        split_point = int(len(series) * train_size)
        return series[:split_point], series[split_point:]

    def model_arima(self, column):
        """
        Applique auto-ARIMA pour modéliser la série et prédire.
        """
        print(f"\n Modélisation ARIMA sur : {column}")
        train, test = self.train_test_split(column)

        model = auto_arima(train, seasonal=False, stepwise=True, suppress_warnings=True)
        print(model.summary())

        predictions = model.predict(n_periods=len(test))

        # Affichage des résultats
        plt.figure(figsize=(12, 6))
        plt.plot(train.index, train, label='Train')
        plt.plot(test.index, test, label='Test', color='orange')
        plt.plot(test.index, predictions, label='Prévision ARIMA', color='green')
        plt.legend()
        plt.title("Prédictions ARIMA")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        mse = mean_squared_error(test, predictions)
        mae = mean_absolute_error(test, predictions)
        print(f"MAE : {mae:.4f} | MSE : {mse:.4f}")

    def model_lstm(self, column, n_steps=10, epochs=20):
        """
        Applique LSTM sur la série temporelle.
        """
        print(f"\n Modélisation LSTM sur : {column}")
        series = self.data[column].dropna().values.reshape(-1, 1)
        series = series.astype(np.float32)

        # Création des séquences
        X, y = [], []
        for i in range(n_steps, len(series)):
            X.append(series[i - n_steps:i])
            y.append(series[i])
        X, y = np.array(X), np.array(y)

        # Split
        split = int(len(X) * 0.8)
        X_train, y_train = X[:split], y[:split]
        X_test, y_test = X[split:], y[split:]

        # Model
        model = Sequential([
            LSTM(50, activation='relu', input_shape=(n_steps, 1)),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train, y_train, epochs=epochs, verbose=0)

        predictions = model.predict(X_test).flatten()

        # Visualisation
        plt.figure(figsize=(12, 6))
        plt.plot(y_test, label='Réel')
        plt.plot(predictions, label='Prédiction LSTM', color='orange')
        plt.legend()
        plt.title("Prédictions LSTM")
        plt.tight_layout()
        plt.grid(True)
        plt.show()

        print(f"MAE : {mean_absolute_error(y_test, predictions):.4f}")
        print(f"MSE : {mean_squared_error(y_test, predictions):.4f}")

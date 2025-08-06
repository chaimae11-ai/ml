# ml

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_time_series(data, time_column, value_column):
    print("\n" + "="*80)
    print("Visualisation de séries temporelles".center(80))
    print("="*80)

    # Vérification des types
    if time_column not in data.columns and time_column != data.index.name:
        print(f"❌ La colonne '{time_column}' n'existe pas dans les colonnes ou comme index.")
        return

    if value_column not in data.columns:
        print(f"❌ La variable '{value_column}' n'existe pas dans les colonnes.")
        return

    # Tri par date si nécessaire
    if time_column in data.columns:
        data_sorted = data.sort_values(by=time_column).set_index(time_column)
    else:
        data_sorted = data.sort_index()

    # Tracé
    plt.figure(figsize=(12, 6))
    plt.plot(data_sorted.index, data_sorted[value_column], color='tab:blue')
    plt.title(f"Série temporelle : {value_column}", fontsize=14)
    plt.xlabel("Temps")
    plt.ylabel(value_column)
    plt.grid(True)
  
    plt.tight_layout()
    plt.show()

"""File per valutare le prestazioni del modello Random Forest salvato"""

#%% [1] Importazione delle Librerie
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from metrics import plot_learning_curve

#%% [2] Caricamento del Dataset
try:
    df = pd.read_csv('../data/dataset_processed.csv')
    print(f"✅ Dataset caricato con successo: {df.shape[0]} righe e {df.shape[1]} colonne.")
except FileNotFoundError:
    print("❌ Errore: File non trovato. Controlla il percorso del file.")

# %% [3] Suddivisione in Feature e Target (identica al training)
X = df.drop(['id', 'accident_risk','num_reported_accidents'], axis=1)  # Feature
y = df['accident_risk']                                                 # Target

# %% [4] Suddivisione in Set di Addestramento e Test (stesso random_state)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=19)
print(f"✅ Suddivisione completata: {X_train.shape[0]} righe per addestramento, {X_test.shape[0]} righe per test")

# %% [5] Caricamento del Modello Salvato
try:
    best_rfr = joblib.load('random_forest_model.pkl')
    print("✅ Modello Random Forest caricato con successo!")
except FileNotFoundError:
    print("❌ Errore: Modello non trovato. Assicurati che 'random_forest_model.pkl' esista.")



# %% [7] Valutazione del Modello su Test Set
y_pred_test = best_rfr.predict(X_test)
mse_test = mean_squared_error(y_test, y_pred_test)
mae_test = mean_absolute_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)

print("\n" + "="*50)
print("📊 METRICHE SUL TEST SET")
print("="*50)
print(f"MSE:  {mse_test:.6f}")
print(f"MAE:  {mae_test:.6f}")
print(f"R²:   {r2_test:.6f}")

# %% [9] Visualizzazione dei Risultati e Grafici

# Configurazione stile
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 5)

# 1. Grafico Real vs Predicted (Per valutare la precisione visivamente)
plt.subplot(1, 2, 1)
sns.scatterplot(x=y_test, y=y_pred_test, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', lw=2)
plt.title("Real vs Predicted: Accident Risk")
plt.xlabel("Valore Reale")
plt.ylabel("Predizione Modello")

# 2. Distribuzione degli Errori (Residui)
plt.subplot(1, 2, 2)
residuals = y_test - y_pred_test
sns.histplot(residuals, kde=True, color='purple')
plt.title("Distribuzione dei Residui (Errori)")
plt.xlabel("Errore (Reale - Predetto)")

plt.tight_layout()
plt.show()

# 3. Grafico Feature Importance (Fondamentale per la Fase 3 - Ontologia)
plt.figure(figsize=(10, 6))
importances = best_rfr.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(10), palette='viridis')
plt.title("Top 10 Feature più importanti per il Rischio Incidenti")
plt.show()

# Salva l'importanza per usarla nell'ontologia
feature_importance_df.to_csv('../Results/feature_importance.csv', index=False)


# %% [11] Grafico della Learning Curve

# Esegui la funzione sul tuo miglior modello
plot_learning_curve(best_rfr, X_train, y_train, "Random Forest Regressor", 'r2')

# %%

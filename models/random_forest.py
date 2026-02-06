"""In questo File sperimentiamo con la tecnica di Random Forest per la predizione del rischio di incidenti stradali. 
Una volta aver compreso quali sono i migliori iperparametri, salveremo il modello addestrato per usarlo nella fase 3 
(Ontologia)."""

#%% [1] Importazione delle Librerie
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

#%% [2] Caricamento del Dataset
try:
    import os
    os.chdir('/Users/lorenzopeluso/Library/CloudStorage/OneDrive-UniversitàdegliStudidiBari/Documents/Università/TerzoAnno/ICon/SafeDrive')
    df = pd.read_csv('data/dataset_processed.csv')
    print(f"✅ Dataset caricato con successo: {df.shape[0]} righe e {df.shape[1]} colonne.")
    df.head(5)
except FileNotFoundError:
    print("❌ Errore: File non trovato. Controlla il percorso del file.")

# %% [3] Suddivisione in Feature e Target
X = df.drop(['id', 'accident_risk'], axis=1)  # Feature
#Rimuovo id perchè non deve influire nella predizione del rischio
y = df['accident_risk']                       # Target

# %% [4] Suddivisione in Set di Addestramento e Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=19)
print(f"✅ Suddivisione completata: {X_train.shape[0]} righe per addestramento, {X_test.shape[0]} righe per test") 

# %% [5] Addestramento del Modello Random Forest
rfr = RandomForestRegressor(random_state=13)
rfr.fit(X_train, y_train)

print("🚀 Modello Random Forest addestrato con successo!")

# %% [6] Valutazione del Modello

y_pred = rfr.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"✅ Valutazione completata! MSE: {mse:.4f}")
mae = mean_absolute_error(y_test, y_pred)
print(f"✅ Valutazione completata! MAE: {mae:.4f}")

# %% [7] Sperimentiamo con iperparametri tramite la tecnica di Cross Validation
"""Grazie alla tecnica di Cross Validation possiamo sperimantare con gli iperparametri per trovare quelli migliori allo scopo della predizione"""

param_grid = {
    'n_estimators': [100, 200], #numero di alberi nella foresta
    'max_depth': [10, 20],      #profondità massima di ogni albero
    'min_samples_split': [2, 5],    #numero minimo di campioni richiesti per dividere un nodo
    'min_samples_leaf': [1, 2]   #numero minimo di campioni richiesti in un nodo foglia
}

from sklearn.model_selection import GridSearchCV  # noqa: E402

grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=13),
                            param_grid=param_grid,
                            cv=3, #3 fold cross-validation
                            n_jobs=-1, #esegue in parallelo su tutte le CPU disponibili
                            verbose=1) #output minimo
grid_search.fit(X_train, y_train)
print(f"🚀 Grid Search completato! Migliori parametri: {grid_search.best_params_}")

"""Risulta che i migliori parametri trovati sono:
    Migliori parametri: {'max_depth': 10, 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 200}"""
# %% [8] Valutazione del Modello Ottimizzato
best_rfr = grid_search.best_estimator_
y_pred_optimized = best_rfr.predict(X_test)
mse_optimized = mean_squared_error(y_test, y_pred_optimized)
print(f"✅ Valutazione Modello Ottimizzato completata! MSE: {mse_optimized:.4f}")
mae_optimized = mean_absolute_error(y_test, y_pred_optimized)
print(f"✅ Valutazione Modello Ottimizzato completata! MAE: {mae_optimized:.4f}")

#%% [9] Salvataggio del Modello Addestrato
import joblib
joblib.dump(best_rfr, 'models/random_forest_model.pkl')
print("💾 Modello salvato in 'models/random_forest_model.pkl'")

# %% [9] Visualizzazione dei Risultati e Grafici
import matplotlib.pyplot as plt
import seaborn as sns

# Configurazione stile
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 5)

# 1. Grafico Real vs Predicted (Per valutare la precisione visivamente)
plt.subplot(1, 2, 1)
sns.scatterplot(x=y_test, y=y_pred_optimized, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', lw=2)
plt.title("Real vs Predicted: Accident Risk")
plt.xlabel("Valore Reale")
plt.ylabel("Predizione Modello")

# 2. Distribuzione degli Errori (Residui)
plt.subplot(1, 2, 2)
residuals = y_test - y_pred_optimized
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
feature_importance_df.to_csv('results/feature_importance.csv', index=False)

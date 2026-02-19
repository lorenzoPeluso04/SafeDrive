#%% 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import seaborn as sns
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score



def plot_learning_curve(model, X, y, model_name, scoring):
    train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=2, scoring=scoring, n_jobs=-1)

    # Calcola gli errori su addestramento e test
    train_errors = 1 - train_scores
    test_errors = 1 - test_scores

    # Calcola la deviazione standard e la varianza degli errori su addestramento e test
    train_errors_std = np.std(train_errors, axis=1)
    test_errors_std = np.std(test_errors, axis=1)
    train_errors_var = np.var(train_errors, axis=1)
    test_errors_var = np.var(test_errors, axis=1)

    # Stampa i valori numerici della deviazione standard e della varianza
    print(
        f"\033[95m{model_name} - Train Error Std: {train_errors_std[-1]}, Test Error Std: {test_errors_std[-1]}, Train Error Var: {train_errors_var[-1]}, Test Error Var: {test_errors_var[-1]}\033[0m")

    # Calcola gli errori medi su addestramento e test
    mean_train_errors = 1 - np.mean(train_scores, axis=1)
    mean_test_errors = 1 - np.mean(test_scores, axis=1)

    #Visualizza la curva di apprendimento
    plt.figure(figsize=(16, 10))
    plt.plot(train_sizes, mean_train_errors, label='Errore di training', color='green')
    plt.plot(train_sizes, mean_test_errors, label='Errore di testing', color='red')
    plt.title(f'Curva di apprendimento per {model_name}')
    plt.xlabel('Dimensione del training set')
    plt.ylabel('Errore')
    plt.legend()
    plt.show()

#%%
def random_forest_metrics_(model, X, y, test_size=0.2, random_state=19):
    """
    Funzione generalizzata per la valutazione del modello Random Forest.
    Esegue lo split automatico, calcola le metriche di errore e visualizza i grafici.
    """
    
    # 1. Suddivisione automatica in Set di Addestramento e Test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"✅ Split completato: {X_train.shape[0]} righe training, {X_test.shape[0]} righe test.")

    # 2. Valutazione del Modello su Test Set
    y_pred_test = model.predict(X_test)
    
    mse_test = mean_squared_error(y_test, y_pred_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    r2_test = r2_score(y_test, y_pred_test)
    rmse_test = np.sqrt(mse_test)

    print("\n" + "="*50)
    print("📊 METRICHE SUL TEST SET")
    print("="*50)
    print(f"MSE:   {mse_test:.6f}")
    print(f"RMSE:  {rmse_test:.6f}")
    print(f"MAE:   {mae_test:.6f}")
    print(f"R²:    {r2_test:.6f}")

    # 3. Visualizzazione dei Risultati
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(14, 5))

    # Grafico Real vs Predicted
    plt.subplot(1, 2, 1)
    sns.scatterplot(x=y_test, y=y_pred_test, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', lw=2)
    plt.title("Real vs Predicted")
    plt.xlabel("Valore Reale")
    plt.ylabel("Predizione Modello")

    # Distribuzione dei Residui
    plt.subplot(1, 2, 2)
    residuals = y_test - y_pred_test
    sns.histplot(residuals, kde=True, color='purple')
    plt.title("Distribuzione dei Residui (Errori)")
    plt.xlabel("Errore (Reale - Predetto)")

    plt.tight_layout()

    plt.figure(figsize=(10, 6))
    importances = model.feature_importances_
    feature_names = X.columns
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(10), palette='viridis')
    plt.title("Top 10 Feature più importanti per il Rischio Incidenti")
    plt.show()

    plot_learning_curve(model, X_train, y_train, "Random Forest Regressor", 'r2')
    
    # Salva l'importanza per usarla nell'ontologia
    feature_importance_df.to_csv('../Results/feature_importance.csv', index=False)

"""
#%% Plot 
#%% [1] Caricamento del Dataset
try:
    df = pd.read_csv('../data/dataset_processed.csv')
    print(f"✅ Dataset caricato con successo: {df.shape[0]} righe e {df.shape[1]} colonne.")
except FileNotFoundError:
    print("❌ Errore: File non trovato. Controlla il percorso del file.")

# %% [1] Suddivisione in Feature e Target (identica al training)
X = df.drop(['id', 'accident_risk','num_reported_accidents'], axis=1)  # Feature
y = df['accident_risk']                                                 # Target

# %% [2] Caricamento del Modello Salvato
try:
    best_rfr = joblib.load('random_forest_model.pkl')
    print("✅ Modello Random Forest caricato con successo!")
except FileNotFoundError:
    print("❌ Errore: Modello non trovato. Assicurati che 'random_forest_model.pkl' esista.")

#%% [3] Valutazione del Modello su Test Set
random_forest_metrics_(best_rfr, X, y)"""

# %%

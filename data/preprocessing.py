# %% [1] Import delle librerie
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Impostazione estetica per i grafici
sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 100

def plot_accident_risk_distribution(data, column_name='accident_risk'):
    """
    Riproduce il grafico della distribuzione del rischio di incidenti 
    con istogramma, KDE, media e mediana.
    """
    # Calcolo statistiche
    mean_val = data[column_name].mean()
    median_val = data[column_name].median()
    
    # Inizializzazione plot
    plt.figure(figsize=(10, 5))
    
    # Creazione Istogramma + KDE
    sns.histplot(data[column_name], kde=True, stat="density", 
                 color='dodgerblue', bins=40, alpha=0.8, edgecolor='white')
    
    # Aggiunta linee verticali per Media e Mediana
    plt.axvline(mean_val, color='red', linestyle='--', 
                label=f'Media = {mean_val:.3f}')
    plt.axvline(median_val, color='red', linestyle=':', 
                linewidth=2, label=f'Mediana = {median_val:.3f}')
    
    # Personalizzazione estetica
    plt.title('Distribuzione di Accident Risk', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Accident Risk', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend(frameon=True)
    plt.grid(axis='y', alpha=0.3)
    
    # Rimuovere i bordi superiori e destri per un look pulito
    sns.despine(offset=5)
    
    plt.tight_layout()
    plt.show()



# %% [2] Caricamento del Dataset
# Sostituisci 'data/train.csv' con il tuo percorso effettivo
try:
    df = pd.read_csv('data/train.csv')
    print(f"✅ Dataset caricato con successo: {df.shape[0]} righe e {df.shape[1]} colonne.")
except FileNotFoundError:
    print("❌ Errore: File non trovato. Controlla il percorso del file.")

# %% [3] Pulizia Dati (Data Cleaning)
# Gestione valori nulli
if df.isnull().values.any():
    print("⚠️ Rilevati valori nulli. Procedo con l'imputazione/rimozione...")
    df = df.dropna() #Il dataser è grande, eventuali valori nulli possono essere rimossi
    print(f"Nuove dimensioni dopo la rimozione dei nulli: {df.shape}")

# Conversione speed_limit da miglia orarie a km/h
df['speed_limit'] = df['speed_limit'] * 1.60934
print("✅ Speed limit convertito da mph a km/h")

# %% [5] Preprocessing per ML (Encoding)
# 1. Procedo a utilizzare la tecnica di One-Hot Encoding per le variabili categoriche
categorical_cols = ['road_type', "weather", "time_of_day", "lighting"]
df_final = pd.get_dummies(df, columns=categorical_cols) 
#Il metodo get_dummies gestisce automaticamente il One-Hot Encoding associanco 0/1 ad ogni individuo della classe

# Conversione booleani (es. road_signs_present) in 0/1 per i calcoli
bool_cols = df.select_dtypes(include=['bool']).columns
for col in bool_cols:
    df[col] = df[col].astype(int)

# %% [4] Analisi Esplorativa (EDA) - Matrice di Correlazione
# Questo grafico è fondamentale per la tua tesi: mostra cosa causa il rischio.
plt.figure(figsize=(12, 8))
# Selezioniamo solo le colonne numeriche per la correlazione
numeric_df = df.select_dtypes(include=[np.number])
correlation_matrix = numeric_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Matrice di Correlazione: Fattori di Rischio Incidenti")
plt.show()


# %% [6] Salvataggio dei Dati Processati

# Conversione booleani (es. road_signs_present) in 0/1 per i calcoli
bool_cols = df_final.select_dtypes(include=['bool']).columns
for col in bool_cols:
    df_final[col] = df_final[col].astype(int)

df_final.to_csv('data/dataset_processed.csv', index=False)
print("🚀 Fase 1 completata! Dataset salvato in 'data/dataset_processed.csv'")
print(f"Nuove dimensioni del dataset: {df_final.shape}")

"""Di seguito procedo a realizzare il preprocessing specifico per la classificazione del rischio di incidenti stradali."""
# %% [7] Caricamento del Dataset
# Sostituisci 'data/train.csv' con il tuo percorso effettivo
try:
    df = pd.read_csv('data/dataset_processed.csv')
    print(f"✅ Dataset caricato con successo: {df.shape[0]} righe e {df.shape[1]} colonne.")
except FileNotFoundError:
    print("❌ Errore: File non trovato. Controlla il percorso del file.")

# %% [8] Conversione del Target in Classi
# Trasformiamo il rischio continuo in due classi: 1 (Alto Rischio), 0 (Basso Rischio)
df['is_dangerous'] = (df['accident_risk'] >= 0.4).astype(int)

print(f"✅ Target binarizzato: {df['is_dangerous'].value_counts().to_dict()}")
# Rimuoviamo la colonna originale del rischio numerico per non "barare" durante il training
df = df.drop(columns=['accident_risk'])
print(df.head(5))

# %% [9] FEATURE SCALING (Fondamentale per Regressione Logistica)
# Scaliamo le variabili numeriche per avere media 0 e varianza 1
scaler = StandardScaler()
numeric_features = ['speed_limit', 'curvature', 'lane_width'] # Aggiungi altre se presenti

# Verifichiamo che le colonne esistano prima di scalare
existing_numeric = [c for c in numeric_features if c in df.columns]
df[existing_numeric] = scaler.fit_transform(df[existing_numeric])

print(f"✅ Scaling completato su: {existing_numeric}")
print(df.head(5))

# %% [10] Salvataggio
df.to_csv('data/classification_dataset_processed.csv', index=False)
print("🚀 Fase 1 completata! Dataset pronto per Regressione Logistica e Random Forest.")
 #%%
try:
    df = pd.read_csv("data/dataset_processed.csv")
    print(f"✅ Dataset caricato con successo: {df.shape[0]} righe e {df.shape[1]} colonne.")
except FileNotFoundError:
    print("❌ Errore: File non trovato. Controlla il percorso del file.")

plot_accident_risk_distribution(df)
# %%

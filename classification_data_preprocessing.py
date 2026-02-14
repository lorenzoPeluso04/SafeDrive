"""In questo file eseguiamo tutte le operazioni di preprocessing sui dati per eseguire un
problema di classificazione del rischio di incidenti stradali. 
Per rischio >= di 0.5 classifichiamo come 'alto rischio', altrimenti 'basso rischio"""

# %% [1] Import delle librerie 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Impostazione estetica per i grafici
sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 100

# %% [2] Caricamento del Dataset
# Sostituisci 'data/train.csv' con il tuo percorso effettivo
try:
    df = pd.read_csv('data/dataset_processed.csv')
    print(f"✅ Dataset caricato con successo: {df.shape[0]} righe e {df.shape[1]} colonne.")
except FileNotFoundError:
    print("❌ Errore: File non trovato. Controlla il percorso del file.")

# %% [3] Conversione del Target in Classi
# Trasformiamo il rischio continuo in due classi: 1 (Alto Rischio), 0 (Basso Rischio)
df['is_dangerous'] = (df['accident_risk'] >= 0.5).astype(int)

print(f"✅ Target binarizzato: {df['is_dangerous'].value_counts().to_dict()}")
# Rimuoviamo la colonna originale del rischio numerico per non "barare" durante il training
df = df.drop(columns=['accident_risk'])
print(df.head(5))

# %% [4] FEATURE SCALING (Fondamentale per Regressione Logistica)
# Scaliamo le variabili numeriche per avere media 0 e varianza 1
scaler = StandardScaler()
numeric_features = ['speed_limit', 'curvature', 'lane_width'] # Aggiungi altre se presenti

# Verifichiamo che le colonne esistano prima di scalare
existing_numeric = [c for c in numeric_features if c in df.columns]
df[existing_numeric] = scaler.fit_transform(df[existing_numeric])

print(f"✅ Scaling completato su: {existing_numeric}")
print(df.head(5))

# %% [5] Salvataggio
df.to_csv('data/classification_dataset_processed.csv', index=False)
print("🚀 Fase 1 completata! Dataset pronto per Regressione Logistica e Random Forest.")
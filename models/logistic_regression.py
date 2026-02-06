#%% [1] Import delle librerie
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc

from metrics import plot_learning_curve


#%% [2] Caricamento del Dataset
try:
    df = pd.read_csv('data/classification_dataset_processed.csv')
    print(f"✅ Dataset caricato con successo: {df.shape[0]} righe e {df.shape[1]} colonne.")
    #print(df.head(5))
except FileNotFoundError:
    print("❌ Errore: File non trovato. Controlla il percorso del file.")

# %% Grafici per l'analisi esplorativa dei dati

dfo = pd.read_csv('data/dataset_processed.csv')

plt.scatter(dfo['curvature'], dfo['accident_risk'])
plt.xlabel('Curvature')

plt.ylabel('Accident Risk')
plt.title('Scatter Plot: Curvature vs Accident Risk')
plt.grid()
plt.show()

sns.countplot(x='is_dangerous', data=df)

# %% [3] Suddivisione in Feature e Target e Suddivisione in Set di Addestramento e Test

X = df.drop(['id', 'num_reported_accidents', 'is_dangerous'], axis=1)   # Rimuovo id perchè non deve influire nella predizione del rischio, gli altri per non "barare" durante il training
y = df['is_dangerous']        # Target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=11)
print(f"✅ Suddivisione completata: {X_train.shape[0]} righe per addestramento, {X_test.shape[0]} righe per test") 

# %% [4] Addestramento del Modello di Regressione Logistica

model = LogisticRegression()
model.fit(X_train, y_train) 

# %% [5] Valutazione del Modello

y_pred = model.predict(X_test)
print(model.score(X_test, y_test))

print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Matrice di Confusione:")
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Realmente Positivi', 'Realmente Negativi'],
            yticklabels=['Predetti Positivi', 'Predetti Negativi'])
plt.title('Matrice di Confusione')
plt.show()

plot_learning_curve(model, X_train, y_train, "Regressione Logistica", 'accuracy')

# %% [6] Valutazione del Modello con AUC-ROC

fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
auc_score = auc(fpr, tpr)

#plot
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'spazio ROC (AUC = {auc_score:.2f})')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.009])
plt.xlabel('False Positive Rate FPR')
plt.ylabel('True Positive Rate TPR')
plt.title('Spazio ROC')
plt.legend(loc='lower right')
plt.show()


# %%

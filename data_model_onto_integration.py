import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from owlready2 import get_ontology, sync_reasoner_pellet


def carica_dataset(path):
    try:
        df = pd.read_csv(path)
        print(f"✅ Dataset caricato con successo: {df.shape[0]} righe e {df.shape[1]} colonne.")
        return df
    except FileNotFoundError:
        print("❌ Errore: File non trovato. Controlla il percorso del file.")
        return None
    
def preprocessa_dataset(df):
    if df is not None:
        df['speed_limit'] = df['speed_limit'] * 1.60934
        print("✅ Speed limit convertito da mph a km/h")
        categorical_cols = ['road_type', "weather", "time_of_day", "lighting"]
        df = pd.get_dummies(df, columns=categorical_cols) 
        bool_cols = df.select_dtypes(include=['bool']).columns
        for col in bool_cols:
            df[col] = df[col].astype(int)

        df.drop(columns=['num_reported_accidents', 'id'], inplace=True)
        df.to_csv('data/test_regressor_processed.csv', index=False)

        scaler = StandardScaler()
        numeric_features = ['speed_limit', 'curvature', 'lane_width'] # Aggiungi altre se presenti

        # Verifichiamo che le colonne esistano prima di scalare
        existing_numeric = [c for c in numeric_features if c in df.columns]
        df[existing_numeric] = scaler.fit_transform(df[existing_numeric])

        print(f"✅ Scaling completato su: {existing_numeric}")
        df.to_csv('data/test_classifier_processed.csv', index=False)

def carica_modello(path):
    try:
        modello = joblib.load(path)
        print(f"✅ Modello caricato con successo da {path}")
        return modello
    except FileNotFoundError:
        print("❌ Errore: File del modello non trovato.")
        return None

def predizione(modello, X):
    if modello is not None and X is not None:
        predizioni = modello.predict(X)
        print(f"✅ Predizioni generate: {len(predizioni)}")
        return predizioni
    else:
        print("❌ Errore: Modello o dataset non disponibile per la predizione.")
        return None

def popola_ontologia_SafeDrive(record, predizione_classificazione, predizione_regressione):
    pass

def ragiona_ontologia_SafeDrive():
    #with onto:
        #sync_reasoner_pellet(infer_property_values=True, infer_data_property_values=True)
    pass

def stampa_conclusioni_ontologia_SafeDrive():
    pass

        
if __name__ == "__main__":
    """preprocessa_dataset(carica_dataset('data/test.csv'))
    X_class = carica_dataset('data/test_classifier_processed.csv')
    X_reg = carica_dataset('data/test_regressor_processed.csv')
    modello_classificazione = carica_modello('models/logistic_regression_model.pkl')
    modello_regressione = carica_modello('models/random_forest_model.pkl')
    print(predizione(modello_classificazione, X_class[:50]))
    print(predizione(modello_regressione, X_reg[:50]))"""
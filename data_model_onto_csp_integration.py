import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import onto.ontology as owl_onto
from onto.ontology import onto as my_ontology

road_type = {
    'highway': owl_onto.Autostrada,
    'urban': owl_onto.Urbana,
    'rural': owl_onto.Rurale,
}

lighting = {
    "night": owl_onto.Notturno,
    "dim": owl_onto.Penombra,
    "day": owl_onto.Diurno,
}

weather = {
    "clear": owl_onto.Sole,
    "rainy": owl_onto.Pioggia,
    "foggy": owl_onto.Nebbia,
}

time_of_day = {
    "morning": owl_onto.Mattina,
    "afternoon": owl_onto.Pomeriggio,
    "evening": owl_onto.Sera,
}

def carica_dataset(path):
    try:
        df = pd.read_csv(path)
        print(f"✅ Dataset caricato con successo: {df.shape[0]} righe e {df.shape[1]} colonne.")
        return df
    except FileNotFoundError:
        print("❌ Errore: File non tråovato. Controlla il percorso del file.")
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
        return predizioni
    else:
        print("❌ Errore: Modello o dataset non disponibile per la predizione.")
        return None

def popola_ontologia_SafeDrive(record, predizione_classificazione, predizione_regressione):

    te = owl_onto.TrattoStradale(f"TrattoStradale_{record['id']}")
    if record['road_type'] in road_type:
        te.haTipoStrada = road_type.get(record['road_type'])
    
    if record['weather'] in weather:
        te.haCondizioniMeteo = weather.get(record['weather'])

    if record['lighting'] in lighting:
        te.haIlluminazione = lighting.get(record['lighting'])

    #if record['time_of_day'] in time_of_day:
        #te.haMomentoGiorno = time_of_day.get(record['time_of_day'])

    te.haCurvatura = float(record['curvature']) 
    te.haLimiteVelocità = int(record['speed_limit'] * 1.60934)
    te.haSegnaletica = int(record['road_signs_present'])

    te.haPericolo = int(predizione_classificazione) 
    te.haPunteggioPericolo = float(predizione_regressione)

    return te

def ragiona_ontologia_SafeDrive(te):
    with my_ontology:
        owl_onto.sync_reasoner_pellet(infer_property_values=True, infer_data_property_values=True, debug = 0)

    return te

def stampa_conclusioni_ontologia_SafeDrive(t_ontology):

    print(f"\n📍 Analisi: {t_ontology.name}")
    print(f"   Input ML - Score: {t_ontology.haPunteggioPericolo}, Class: {t_ontology.haPericolo}")
    
    # Recuperiamo lo Stato di Sicurezza inferito
    if t_ontology.haStatoSicurezza:
        print(f"   🛡️ STATO RILEVATO: {t_ontology.haStatoSicurezza.name}")
    elif t_ontology.haPericolo == 0 and t_ontology.haPunteggioPericolo > 0.5:
        print("   🛡️ STATO RILEVATO: Conflitto tra i pedittori")
    else:
        print("   🛡️ STATO RILEVATO: Non determinato")

    # Recuperiamo il Tipo di Rischio inferito (se presente)
    rischi = t_ontology.haTipoRischio
    if rischi:
        # Essendo una lista (potrebbero esserci più rischi), li uniamo
        nomi_rischi = ", ".join([r.name for r in rischi])
        print(f"   ⚠️ TIPO RISCHIO: {nomi_rischi}")
    
    # Recuperiamo le Raccomandazioni inferite
    raccomandazioni = t_ontology.haRaccomandazione
    if raccomandazioni:
        print("   💡 RACCOMANDAZIONI SISTEMA:")
        for rac in raccomandazioni:
            print(f"      - {rac.name}")
    else:
        print("      - Guida con prudenza")
        
if __name__ == "__main__":
    #preprocessa_dataset(carica_dataset('data/test.csv'))
    X_class = carica_dataset('data/test_classifier_processed.csv')
    X_reg = carica_dataset('data/test_regressor_processed.csv')
    modello_classificazione = carica_modello('models/logistic_regression_model.pkl')
    modello_regressione = carica_modello('models/random_forest_model.pkl')

    data = carica_dataset('data/test.csv')
    for i in range(len(X_class.iloc[350:650])):
        predizione_classificazione = predizione(modello_classificazione, X_class.iloc[[i]])
        predizione_regressione = predizione(modello_regressione, X_reg.iloc[[i]])
        if predizione_regressione[0] >= 0.5:
            print(f"{predizione_classificazione[0]}, {predizione_regressione[0]:.2f}")
            print(data.iloc[[i]])
            x = popola_ontologia_SafeDrive(data.iloc[i], predizione_classificazione[0], predizione_regressione[0])
            ragiona_ontologia_SafeDrive(x)
            stampa_conclusioni_ontologia_SafeDrive(x)
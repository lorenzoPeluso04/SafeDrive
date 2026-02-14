import data_model_onto_csp_integration as sd #safe drive

if __name__ == "__main__":

    ROSSO = "\033[31m"
    VERDE = "\033[32m"
    RESET = "\033[0m"

    #===============Carichiamo i modelli che abbiamo allenato in precedenza==============#

    classificatore=sd.carica_modello("models/logistic_regression_model.pkl")
    regressore=sd.carica_modello("models/random_forest_model.pkl")

    #===========Carichiamo il dataset: un esempio di strada divisa in segmenti===========#

    data=sd.carica_dataset('data/road_system.csv')

    X_class=sd.preprocessa_dataset(data, add_scaler=True) 
    #processiamo i dati con lo scaler perchè il modello di regressione logistica ottiene una performance migliore

    X_reg=sd.preprocessa_dataset(data, 'data/road_system.csv')

    #========================Effetuiamo le predizioni sul dataset========================#

    predizione_classificazione = sd.predizione(classificatore, X_class)
    predizione_regressione = sd.predizione(regressore, X_reg)

    #==================Popoliamo l'ontologia e avviamo il ragionamento===================#

    for i in range(len(data)):

        #print(f"\nPredizione Classificatore = {predizione_classificazione[i]}, Predizione Regressore = {predizione_regressione[i]:.2f}")
        #print(f"Record Analizato: \n{data.iloc[[i]]}")
        x = sd.popola_ontologia_SafeDrive(data.iloc[i], predizione_classificazione[i], predizione_regressione[i])
        sd.ragiona_ontologia_SafeDrive(x)
        print(f"\nSegmento stradale {i}")
        sd.stampa_conclusioni_ontologia_SafeDrive(x)

    #=====Creiamo e risolviamo il CSP road_planner per ridurre il pericolo in strada=====#

    print(f"\nL'elaborazione potrebbe richiedere qualche secondo, {ROSSO}attendere...{RESET}\n")
    
    lista_limiti = sd.messa_in_sicurezza(data, classificatore) #cspSoft.py riga 81 per ritornare alla verbosità originaria

    print(f"\nSegmenti messi in {VERDE}SICUREZZA{RESET}")
    for i in range(len(data)):
        print(f"\nNuovo limite di velocità del segmento {i} ----> {VERDE}{lista_limiti[i]}{RESET}")
    print("\n")
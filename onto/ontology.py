from owlready2 import get_ontology, Thing, DatatypeProperty, FunctionalProperty, ObjectProperty, AllDifferent, Imp
from owlready2 import sync_reasoner_pellet

onto = get_ontology("http://safedrive.it/ontology.owl")

with onto:

    #---Definiamo le CLASSI dell'ontologia---#

    #Legate alla STRADA

    class Strada(Thing): 
        pass

    class TipoStrada(Thing): 
        pass
    Autostrada = TipoStrada("Autostrada")
    Urbana = TipoStrada("Urbana")
    Rurale = TipoStrada("Rurale")

    AllDifferent([Autostrada, Urbana, Rurale]) # Definisce che gli individui sono distinti

    class TrattoStradale(Strada):
        pass

    #Coindizioni AMBIENTALI

    class Meteo(Thing): 
        pass
    Sole = Meteo("Sole")
    Pioggia = Meteo("Pioggia")
    Nebbia = Meteo("Nebbia")

    AllDifferent([Sole, Pioggia, Nebbia])

    class Illuminazione(Thing): 
        pass
    Diurno = Illuminazione("Diurno")
    Notturno = Illuminazione("Notturno")
    Penombra = Illuminazione("Penombra")

    AllDifferent([Diurno, Notturno, Penombra]) 

    class MomentoGiorno(Thing): 
        pass
    Mattina = MomentoGiorno("Mattina")
    Pomeriggio = MomentoGiorno("Pomeriggio")
    Sera = MomentoGiorno("Sera")

    AllDifferent([Mattina, Pomeriggio, Sera])

    #Stato di SICUREZZA
    class StatoSicurezza(Thing): 
        pass
    Sicuro = StatoSicurezza("Sicuro")
    Pericolo = StatoSicurezza("Pericolo")
    PericoloEstremo = StatoSicurezza("PericoloEstremo")

    AllDifferent([Sicuro, Pericolo, PericoloEstremo]) 
    #Categorie di RISCHIO
    class TipoRischio(Thing): 
        pass
    RischioVisibilità = TipoRischio("RischioVisibilità")
    RischioCurvatura = TipoRischio("RischioCurvatura")
    RischioVelocità = TipoRischio("RischioVelocità")
    RischioLuminosità = TipoRischio("RischioLuminosità")
    RischioComposito = TipoRischio("RischioComposito")

    AllDifferent([RischioVisibilità, RischioCurvatura, RischioVelocità, RischioLuminosità, RischioComposito])
    #Raccomandazioni di GUIDA
    class Raccomandazione(Thing): 
        pass
    Rallentare = Raccomandazione("Rallentare")
    PrestareAttenzione = Raccomandazione("PrestareAttenzione")
    ProtezioneLuminosa = Raccomandazione("ProtezioneLuminosa")
    AumentareDistanzaSicurezza = Raccomandazione("AumentareDistanzaSicurezza")
    Normale = Raccomandazione("Normale")

    #Modelli di ML
    class ModelloML(Thing): 
        pass

    class RegressoreML(ModelloML): 
        pass

    class ClassificatoreML(ModelloML): 
        pass

    RegressioneLogistica = RegressoreML("RegressioneLogistica")
    RandomForest = ClassificatoreML("RandomForest")

    #===============================================================#

    #-------------Definiamo le PROPRIETA' dell'ontologia------------#

    #proprietà che legano oggetti

    class haCondizioniMeteo(ObjectProperty, FunctionalProperty):
        domain = [TrattoStradale]
        range = [Meteo]

    class haIlluminazione(ObjectProperty, FunctionalProperty):
        domain = [TrattoStradale]
        range = [Illuminazione]

    class haMomentoGiorno(ObjectProperty, FunctionalProperty):
        domain = [TrattoStradale]
        range = [MomentoGiorno]
    
    class haStatoSicurezza(ObjectProperty, FunctionalProperty):
        domain = [TrattoStradale]
        range = [StatoSicurezza]

    class haTipoStrada(ObjectProperty, FunctionalProperty):
        domain = [Strada]
        range = [TipoStrada]

    class haRaccomandazione(ObjectProperty): 
        #Una strada può avere più raccomandazioni, quindi non è funzionale
        domain = [TrattoStradale]
        range = [Raccomandazione]

    class haTipoRischio(ObjectProperty):
        domain = [TrattoStradale]
        range = [TipoRischio]
    
    class valutatoDa(ObjectProperty):
        domain = [TrattoStradale]
        range = [ModelloML]

    class adiacente(ObjectProperty):
        domain = [TrattoStradale]
        range = [TrattoStradale]
        symmetric = True #La relazione è simmetrica, se A è adiacente a B, allora B è adiacente a A
    
    class precedente(ObjectProperty):
        domain = [TrattoStradale]
        range = [TrattoStradale]

    class successivo(ObjectProperty):
        domain = [TrattoStradale]
        range = [TrattoStradale]

    #proprietà che legano oggetti a DATI
    
    class haCurvatura(DatatypeProperty, FunctionalProperty):
        domain = [TrattoStradale]
        range = [float]
    
    class haLunghezza(DatatypeProperty, FunctionalProperty):
        domain = [TrattoStradale]
        range = [float]

    class haLimiteVelocità(DatatypeProperty, FunctionalProperty):
        domain = [TrattoStradale]
        range = [int]
    
    class haLimiteVelocitàRaccomandato(DatatypeProperty, FunctionalProperty):
        domain = [TrattoStradale]
        range = [int]
    
    class haSegnaletica(DatatypeProperty, FunctionalProperty):
        #(0/1) 0 = assente, 1 = presente
        domain = [TrattoStradale]
        range = [int]

    #assegnato dal classificatore
    class haPericolo(DatatypeProperty, FunctionalProperty):
        #(0/1) 0 = assente, 1 = presente
        domain = [TrattoStradale]
        range = [int]
    
    #assegnato dal regressore
    class haPunteggioPericolo(DatatypeProperty, FunctionalProperty):
        domain = [TrattoStradale]
        range = [float]

    class haMessaggioUtente(DatatypeProperty, FunctionalProperty):
        domain = [TrattoStradale]
        range = [str]

    ##===============================================================#
    """Ora procediamo a definire le regole di inferenza che ci 
    permettono di dedurre nuove conoscenze a partire dai dati inseriti 
    nell'ontologia"""

    #========In base al rischio fornito dai modelli, assegnamo uno dei 3 livelli di pericolo========#

    ##Nessun Pericolo Rilevato dai Modelli
    regola_normale = Imp()
    regola_normale.set_as_rule("""TrattoStradale(?t),
                                haPericolo(?t, 0), 
                               haPunteggioPericolo(?t, ?p), lessThanOrEqual(?p, 0.5)
                                -> haStatoSicurezza(?t, Sicuro)""")
    
    regola_stato_sicuro = Imp()
    regola_stato_sicuro.set_as_rule("""TrattoStradale(?t), haStatoSicurezza(?t, Sicuro)
                                    -> haRaccomandazione(?t, Normale)""")

    #pericolo rilevato, misuriamo il suo punteggio
    regola_pericolo = Imp()
    regola_pericolo.set_as_rule("""TrattoStradale(?t),
                                haPericolo(?t, 1),
                                haPunteggioPericolo(?t, ?p), greaterThan(?p, 0.5),
                                lessThanOrEqual(?p, 0.8)
                                -> haStatoSicurezza(?t, Pericolo)""")
    
    regola_pericolo_estremo = Imp()
    regola_pericolo_estremo.set_as_rule("""TrattoStradale(?t),
                                        haPericolo(?t, 1),
                                        haPunteggioPericolo(?t, ?p), greaterThan(?p, 0.8) 
                                        -> haStatoSicurezza(?t, PericoloEstremo)""")
    
    #==============In base alle condizioni, associamo un tipo di rischio specifico==============#

    ##Rischio di visibilità

    pericolo_visibilità_notte = Imp()
    pericolo_visibilità_notte.set_as_rule("""TrattoStradale(?t),
                                    haPericolo(?t, 1),
                                    haIlluminazione(?t, Notturno)
                                    -> haTipoRischio(?t, RischioVisibilità)""")
    
    pericolo_visibilità_pioggia = Imp()
    pericolo_visibilità_pioggia.set_as_rule("""TrattoStradale(?t), 
                                    haPericolo(?t, 1),
                                    haCondizioniMeteo(?t, Pioggia)
                                    -> haTipoRischio(?t, RischioVisibilità)""")
    
    pericolo_visibilità_nebbia = Imp()
    pericolo_visibilità_nebbia.set_as_rule("""TrattoStradale(?t), 
                                    haPericolo(?t, 1),
                                    haCondizioniMeteo(?t, Nebbia)
                                    -> haTipoRischio(?t, RischioVisibilità)""")

    #Rischio di curvatura
    pericolo_curvatura = Imp()
    #visto che la curvatura è una feature molto influuente, consideriamo una soglia più bassa
    pericolo_curvatura.set_as_rule("""TrattoStradale(?t),
                                    haPericolo(?t, 1),
                                    haCurvatura(?t, ?c), greaterThan(?c, 0.4)
                                    -> haTipoRischio(?t, RischioCurvatura)""")
    
    #Rischio di velocità
    pericolo_velocità = Imp()
    pericolo_velocità.set_as_rule("""TrattoStradale(?t),
                                    haPericolo(?t, 1),
                                    haLimiteVelocità(?t, ?l), greaterThan(?l, 60)
                                    -> haTipoRischio(?t, RischioVelocità)""")
    
    #Rischio di luminosità
    pericolo_luminosità = Imp()
    pericolo_luminosità.set_as_rule("""TrattoStradale(?t),
                                    haPericolo(?t, 1),
                                    haIlluminazione(?t, Diurno),
                                    haCondizioniMeteo(?t, Sole)
                                    -> haTipoRischio(?t, RischioLuminosità)""")
    
    #========Una volta aver identidificato il tipo di rischio, associamo una raccomandazione========#
    raccomandazione_visibilità = Imp()
    raccomandazione_visibilità.set_as_rule("""TrattoStradale(?t),
                                            haTipoRischio(?t, RischioVisibilità)
                                            -> haRaccomandazione(?t, PrestareAttenzione),
                                            haRaccomandazione(?t, Rallentare),
                                            haRaccomandazione(?t, AumentareDistanzaSicurezza)""")
    
    raccomandazione_curvatura = Imp()
    raccomandazione_curvatura.set_as_rule("""TrattoStradale(?t),
                                            haTipoRischio(?t, RischioCurvatura)
                                            -> haRaccomandazione(?t, Rallentare),
                                            haRaccomandazione(?t, PrestareAttenzione)""")

    raccomandazione_velocità = Imp()
    raccomandazione_velocità.set_as_rule("""TrattoStradale(?t),
                                            haTipoRischio(?t, RischioVelocità)
                                            -> haRaccomandazione(?t, Rallentare),
                                            haRaccomandazione(?t, PrestareAttenzione),
                                            haRaccomandazione(?t, AumentareDistanzaSicurezza)""")
    
    raccomandazione_luminosità = Imp()
    raccomandazione_luminosità.set_as_rule("""TrattoStradale(?t),
                                            haTipoRischio(?t, RischioLuminosità)
                                            -> haRaccomandazione(?t, ProtezioneLuminosa),
                                            haRaccomandazione(?t, AumentareDistanzaSicurezza)""")



    # ... (Tutto il tuo codice precedente) ...

# Importiamo il reasoner specifico per SWRL (Pellet è raccomandato per regole complesse)


def popola_e_ragiona():
    """
    Funzione per simulare l'inserimento dati dai modelli ML 
    e testare le regole di inferenza.
    """
    print("--- Inizio Popolazione ABox ---")

    # --- SCENARIO 1: Strada Sicura ---
    # Il modello ML dice: Basso rischio (0.2), Classificazione 0 (Sicuro)
    t1 = TrattoStradale("Tratto_A1_Autostrada")
    t1.haTipoStrada = Autostrada
    t1.haLunghezza = 2.5
    t1.haLimiteVelocità = 130
    t1.haCurvatura = 0.1            # Curvatura bassa
    t1.haCondizioniMeteo = Sole
    t1.haIlluminazione = Diurno
    
    # Output dei Modelli ML (Input per l'Ontologia)
    t1.haPericolo = 0               # Output Classificatore
    t1.haPunteggioPericolo = 0.2    # Output Regressore
    t1.haMessaggioUtente = "Traffico scorrevole"

    # --- SCENARIO 2: Pericolo Curva Pericolosa ---
    # Il modello ML dice: Alto rischio (0.75), Classificazione 1 (Pericolo)
    t2 = TrattoStradale("Tratto_B3_Statale")
    t2.haTipoStrada = Rurale
    t2.haLunghezza = 1.0
    t2.haLimiteVelocità = 90
    t2.haCurvatura = 0.6            # Curvatura ALTA (> 0.4 scatta la regola)
    t2.haCondizioniMeteo = Sole
    t2.haIlluminazione = Diurno

    # Output dei Modelli ML
    t2.haPericolo = 1               # Output Classificatore
    t2.haPunteggioPericolo = 0.75   # Output Regressore (tra 0.5 e 0.8 -> Pericolo)

    # --- SCENARIO 3: Pericolo Visibilità (Pioggia + Notte) ---
    # Il modello ML dice: Rischio Estremo (0.9), Classificazione 1
    t3 = TrattoStradale("Tratto_C7_Urbana")
    t3.haTipoStrada = Urbana
    t3.haLunghezza = 0.5
    t3.haLimiteVelocità = 50
    t3.haCurvatura = 0.1
    t3.haCondizioniMeteo = Pioggia  # Condizione avversa
    t3.haIlluminazione = Notturno   # Condizione avversa

    # Output dei Modelli ML
    t3.haPericolo = 1               
    t3.haPunteggioPericolo = 0.9    # > 0.8 -> Pericolo Estremo

    print("Dati inseriti. Avvio del Reasoner (Pellet)...")
    
    # AVVIO DEL REASONER
    # infer_property_values=True permette di dedurre le proprietà (es. haRaccomandazione)
    with onto:
        sync_reasoner_pellet(infer_property_values=True, infer_data_property_values=True)

    print("--- Inferenza Completata. Risultati: ---")
    
    # Funzione helper per stampare i risultati
    stampa_report(t1)
    stampa_report(t2)
    stampa_report(t3)

def stampa_report(tratto):
    print(f"\n📍 Analisi: {tratto.name}")
    print(f"   Input ML - Score: {tratto.haPunteggioPericolo}, Class: {tratto.haPericolo}")
    
    # Recuperiamo lo Stato di Sicurezza inferito
    if tratto.haStatoSicurezza:
        print(f"   🛡️ STATO RILEVATO: {tratto.haStatoSicurezza.name}")
    else:
        print("   🛡️ STATO RILEVATO: Non determinato")

    # Recuperiamo il Tipo di Rischio inferito (se presente)
    rischi = tratto.haTipoRischio
    if rischi:
        # Essendo una lista (potrebbero esserci più rischi), li uniamo
        nomi_rischi = ", ".join([r.name for r in rischi])
        print(f"   ⚠️ TIPO RISCHIO: {nomi_rischi}")
    
    # Recuperiamo le Raccomandazioni inferite
    raccomandazioni = tratto.haRaccomandazione
    if raccomandazioni:
        print("   💡 RACCOMANDAZIONI SISTEMA:")
        for rac in raccomandazioni:
            print(f"      - {rac.name}")
    else:
        print("      - Nessuna raccomandazione specifica.")

# Eseguiamo la simulazione
if __name__ == "__main__":
    popola_e_ragiona()
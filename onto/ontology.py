from owlready2 import get_ontology, Thing, DatatypeProperty, FunctionalProperty, ObjectProperty, AllDifferent, Imp

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
    ControllaFariAccesi = Raccomandazione("ControllaFariAccesi")
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

    class èParteDiStrada(ObjectProperty, FunctionalProperty):
        domain = [TrattoStradale]
        range = [Strada]

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
    
    class haLimiteVelocitàLegale(DatatypeProperty, FunctionalProperty):
        domain = [Strada]
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
                               haPunteggioPericolo(?t, ?p), lessThanOrEqual(?p, 0.4)
                                -> haStatoSicurezza(?t, Sicuro)""")
    
    regola_stato_sicuro = Imp()
    regola_stato_sicuro.set_as_rule("""TrattoStradale(?t), haStatoSicurezza(?t, Sicuro)
                                    -> haRaccomandazione(?t, Normale)""")

    #pericolo rilevato, misuriamo il suo punteggio
    regola_pericolo = Imp()
    regola_pericolo.set_as_rule("""TrattoStradale(?t),
                                haPericolo(?t, 1),
                                haPunteggioPericolo(?t, ?p), greaterThan(?p, 0.4),
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
    
    pericolo_curvatura_noML = Imp()
    pericolo_curvatura_noML.set_as_rule("""TrattoStradale(?t), 
                                        haPericolo(?t, 0), haCurvatura(?t, ?c), 
                                        greaterThan(?c, 0.75)
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
                                            haRaccomandazione(?t, AumentareDistanzaSicurezza),
                                           haRaccomandazione(?t, ControllaFariAccesi)""")
    
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
    
    ##========Background Knowlege proveniente dalle regolamentazioni stradali========#

    limite_velocità_autostrada = Imp()
    limite_velocità_autostrada.set_as_rule("""Strada(?s),
                                           haTipoStrada(?s, Autostrada)
                                           -> haLimiteVelocitàLegale(?s, 130)
                                           """)
    
    limite_velocità_urbana = Imp()
    limite_velocità_urbana.set_as_rule("""Strada(?s),
                                           haTipoStrada(?s, Urbana)
                                           -> haLimiteVelocitàLegale(?s, 50)
                                           """)
    
    limite_velocità_rurale = Imp()
    limite_velocità_rurale.set_as_rule("""Strada(?s),
                                           haTipoStrada(?s, Rurale)
                                           -> haLimiteVelocitàLegale(?s, 90)
                                           """)
    
    limite_velocità_ereditato = Imp()
    limite_velocità_ereditato.set_as_rule("""TrattoStradale(?t), 
                                          èParteDiStrada(?t, ?s),
                                          haLimiteVelocitàLegale(?s, ?l)
                                          ->haLimiteVelocitàRaccomandato(?t, ?l)
                                          """)
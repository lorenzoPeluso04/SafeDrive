import sys
import os

# Aggiunge la cartella corrente al path per trovare 'variable', 'cspProblem', ecc.
sys.path.append(os.path.dirname(__file__))

from variable import Variable
from cspProblem import CSP
from cspSoft import SoftConstraint
from cspSoft import DF_branch_and_bound_opt

# 1. Classe per i dati dei segmenti stradali
class SegmentoStradale(object):
    def __init__(self, limite_velocità):
        self.limite_velocità = limite_velocità

    def getLimiteVelocità(self):
        return self.limite_velocità

# 2. Funzione di costo per la differenza di velocità (Fluidità)
# Obiettivo: Minimizzare i cambi bruschi di velocità.
def costo_fluidità(v1, v2):
    diff = abs(v1 - v2)
    # Condizione peggiore (cambio brusco): costo ALTO (100)
    if diff > 20:
        return 100 
    # Condizione migliore (stessa velocità): costo BASSO (prossimo a 0)
    else:
        return diff / 20.0 

# 3. Funzione factory per il vincolo del limite di velocità
def crea_controllo_limite(limite_del_segmento):
    def check(velocità_assegnata):
        # Condizione peggiore (eccesso di velocità): costo ALTO
        if velocità_assegnata > limite_del_segmento:
            return 50 + (velocità_assegnata - limite_del_segmento)
        # Condizione ottimale (rispetto del limite): costo ZERO
        return 0 
    return check

def csp_builder(lista_limiti_segmenti):
    """
    Costruisce il CSP per l'ottimizzazione del traffico.
    """
    dati_stradali = [SegmentoStradale(lim) for lim in lista_limiti_segmenti]
    num_segmenti = len(dati_stradali)
    
    variabili = []
    constraints = []
    
    # Dominio delle velocità possibili
    dominio_velocità = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 130}

    # --- Creazione Variabili e Vincoli Unari ---
    for i in range(num_segmenti):
        nome_var = f"Seg_{i}"
        var = Variable(nome_var, dominio_velocità)
        variabili.append(var)
        
        # Vincolo: non superare il limite del segmento
        funzione_costo = crea_controllo_limite(dati_stradali[i].getLimiteVelocità())
        constraints.append(SoftConstraint([var], funzione_costo, string=f"Limite_Seg_{i}"))

    # --- Creazione Vincoli Binari ---
    for i in range(num_segmenti - 1):
        # Vincolo: fluidità tra segmenti consecutivi
        constraints.append(SoftConstraint([variabili[i], variabili[i+1]], 
                                          costo_fluidità, 
                                          string=f"Fluidità_{i}_{i+1}"))

    # --- Vincolo Globale: Massimizzare la velocità ---
    def costo_lentezza(*args):
        velocità_totale = sum(args)
        # Per massimizzare la velocità in un problema di MINIMIZZAZIONE:
        # la condizione peggiore (andare piano) deve avere un costo ALTO.
        # Più la velocità totale è alta, più il costo scende.
        return 1000 - (velocità_totale / 5.0)

    constraints.append(SoftConstraint(variabili, costo_lentezza, "Efficienza_Globale"))

    return CSP("Traffico Autostradale", variables=set(variabili), constraints=constraints)

if __name__ == "__main__":
    
    
    # 1. Definiamo i limiti dei segmenti (input del problema)
    limiti = [20, 50, 100, 130] 
    
    # 2. Costruiamo il problema CSP
    problema = csp_builder(limiti)

    # 3. Inizializziamo il risolutore Branch and Bound
    # Passiamo il problema e un bound iniziale molto alto (infinito o un numero grande)
    risolutore = DF_branch_and_bound_opt(problema, bound=2000)
    
    # 4. Chiamiamo il metodo optimize() per trovare la soluzione migliore
    soluzione, costo_totale = risolutore.optimize()
    
    # 5. Stampiamo i risultati
    if soluzione:
        print("\n--- SOLUZIONE OTTIMA TROVATA ---")
        # Ordiniamo per nome della variabile per leggibilità
        for var_name in sorted(soluzione.keys(), key=lambda x: x.name):
            print(f"{var_name}: {soluzione[var_name]} km/h")
        print(f"\nCosto totale minimo: {costo_totale:.2f}")
    else:
        print("\nNessuna soluzione trovata sotto il bound specificato.")
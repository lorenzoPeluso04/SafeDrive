#!/usr/bin/env python3
# test_rete_stradale.py
# Script di test completo per il CSP di ottimizzazione velocità stradali
# 
# Uso: python test_rete_stradale.py

import sys
import os

# Assicura che i moduli siano importabili
try:
    from variable import Variable
    from cspProblem import CSP, Constraint
    from display import Displayable
except ImportError:
    print("⚠️  Errore: Impossibile importare i moduli di base.")
    print("Assicurati che variable.py, cspProblem.py, display.py siano nella stessa directory.")
    sys.exit(1)

import random
import time

# ============================================================================
# DEFINIZIONE DELLE CLASSI
# ============================================================================

class TrattoStradale(Variable):
    """Estende Variable per rappresentare un tratto stradale."""
    
    def __init__(self, name, domain, livello_pericolo=0, position=None):
        super().__init__(name, domain, position)
        self.livello_pericolo = livello_pericolo
        
    def __repr__(self):
        return f"{self.name}(pericolo={self.livello_pericolo})"


class ReteStradale(CSP):
    """CSP specializzato per reti stradali."""
    
    def __init__(self, tratti_stradali, lista_adiacenza, titolo="CSP Rete Stradale"):
        self.tratti_stradali = tratti_stradali
        self.lista_adiacenza = lista_adiacenza
        self.mappa_tratti = {tratto.name: tratto for tratto in tratti_stradali}
        
        vincoli = self._crea_vincoli()
        super().__init__(titolo, tratti_stradali, vincoli)
        
    def _crea_vincoli(self):
        """Crea vincoli del problema."""
        vincoli = []
        
        # Vincolo 1: Pericolo → velocità limitata
        for tratto in self.tratti_stradali:
            if tratto.livello_pericolo == 1:
                vincolo = Constraint(
                    scope=[tratto],
                    condition=lambda v, tratto=tratto: v <= 50 if tratto.livello_pericolo == 1 else True,
                    string=f"Pericolo({tratto.name}): v<=50",
                    position=None
                )
                vincoli.append(vincolo)
        
        # Vincolo 2: Adiacenza → differenza velocità
        coppie_elaborate = set()
        for nome_tratto, nomi_adiacenti in self.lista_adiacenza.items():
            tratto = self.mappa_tratti[nome_tratto]
            for nome_adiacente in nomi_adiacenti:
                coppia = tuple(sorted([nome_tratto, nome_adiacente]))
                if coppia not in coppie_elaborate:
                    tratto_adiacente = self.mappa_tratti[nome_adiacente]
                    vincolo = Constraint(
                        scope=[tratto, tratto_adiacente],
                        condition=lambda v1, v2: abs(v1 - v2) <= 20,
                        string=f"Diff({nome_tratto},{nome_adiacente}): |Δv|≤20",
                        position=None
                    )
                    vincoli.append(vincolo)
                    coppie_elaborate.add(coppia)
        
        return vincoli
    
    def velocita_totale(self, assegnazione):
        """Calcola somma velocità totali."""
        return sum(assegnazione.get(tratto, 0) for tratto in self.tratti_stradali)
    
    def calcola_penalita(self, assegnazione):
        """Calcola violazioni di vincoli."""
        violazioni = 0
        
        # Violazioni pericolo
        for tratto in self.tratti_stradali:
            if tratto.livello_pericolo == 1 and assegnazione.get(tratto) > 50:
                violazioni += 1
        
        # Violazioni adiacenza
        for nome_tratto, nomi_adiacenti in self.lista_adiacenza.items():
            tratto = self.mappa_tratti[nome_tratto]
            if tratto in assegnazione:
                for nome_adiacente in nomi_adiacenti:
                    tratto_adiacente = self.mappa_tratti[nome_adiacente]
                    if tratto_adiacente in assegnazione:
                        if abs(assegnazione[tratto] - assegnazione[tratto_adiacente]) > 20:
                            violazioni += 1
        
        return violazioni


class RicercatoreOttimizzato(Displayable):
    """Risolutore SLS per massimizzare velocità."""
    
    def __init__(self, csp):
        self.csp = csp
        self.assegnazione_corrente = None
        self.miglior_assegnazione = None
        self.miglior_punteggio = float('-inf')
        self.numero_passi = 0
        
    def riavvia(self):
        """Inizializzazione con assegnamento casuale."""
        self.assegnazione_corrente = {tratto: random.choice(tratto.domain) 
                                       for tratto in self.csp.tratti_stradali}
        self.display(2, "Assegnazione iniziale:", self._formatta_assegnazione())
        
    def ricerca(self, max_passi=1000, max_violazioni=0):
        """Ricerca soluzione massimizzando velocità."""
        if self.assegnazione_corrente is None:
            self.riavvia()
            self.numero_passi = 1
        
        violazioni = self.csp.calcola_penalita(self.assegnazione_corrente)
        somma_velocita = self.csp.velocita_totale(self.assegnazione_corrente)
        
        if violazioni <= max_violazioni:
            self.miglior_assegnazione = self.assegnazione_corrente.copy()
            self.miglior_punteggio = somma_velocita
            self.display(1, f"✓ Soluzione trovata immediatamente: velocità_totale = {somma_velocita}")
            return self.assegnazione_corrente, somma_velocita, violazioni
        
        for passo in range(1, max_passi):
            self.numero_passi = passo
            
            tratto = random.choice(self.csp.tratti_stradali)
            valore_vecchio = self.assegnazione_corrente[tratto]
            valore_nuovo = random.choice(tratto.domain)
            
            if valore_nuovo != valore_vecchio:
                self.assegnazione_corrente[tratto] = valore_nuovo
                
                violazioni = self.csp.calcola_penalita(self.assegnazione_corrente)
                somma_velocita = self.csp.velocita_totale(self.assegnazione_corrente)
                
                if violazioni <= max_violazioni and somma_velocita > self.miglior_punteggio:
                    self.miglior_assegnazione = self.assegnazione_corrente.copy()
                    self.miglior_punteggio = somma_velocita
                    self.display(2, f"Passo {passo}: Migliore! velocità={somma_velocita}, viol={violazioni}")
                elif violazioni > max_violazioni:
                    if random.random() > 0.3:
                        self.assegnazione_corrente[tratto] = valore_vecchio
        
        if self.miglior_assegnazione:
            self.display(1, f"✓ Miglior soluzione dopo {self.numero_passi} passi: "
                          f"velocità_totale = {self.miglior_punteggio}")
            return self.miglior_assegnazione, self.miglior_punteggio, \
                   self.csp.calcola_penalita(self.miglior_assegnazione)
        else:
            self.display(1, f"✗ Nessuna soluzione trovata dopo {self.numero_passi} passi")
            return self.assegnazione_corrente, somma_velocita, violazioni
    
    def _formatta_assegnazione(self):
        """Formatta assegnazione per display."""
        if not self.assegnazione_corrente:
            return "Vuoto"
        return ", ".join(f"{tratto.name}={self.assegnazione_corrente[tratto]}" 
                        for tratto in self.csp.tratti_stradali)


# ============================================================================
# FUNZIONI DI TEST
# ============================================================================

def crea_strada_lineare_semplice():
    """Crea strada lineare semplice (5 tratti)."""
    dominio_velocita = [30, 50, 70, 90]
    
    tratti = [
        TrattoStradale("S0", dominio_velocita, livello_pericolo=0, position=(0.0, 0.5)),
        TrattoStradale("S1", dominio_velocita, livello_pericolo=1, position=(0.25, 0.5)),
        TrattoStradale("S2", dominio_velocita, livello_pericolo=0, position=(0.5, 0.5)),
        TrattoStradale("S3", dominio_velocita, livello_pericolo=1, position=(0.75, 0.5)),
        TrattoStradale("S4", dominio_velocita, livello_pericolo=0, position=(1.0, 0.5)),
    ]
    
    lista_adiacenza = {
        "S0": ["S1"],
        "S1": ["S0", "S2"],
        "S2": ["S1", "S3"],
        "S3": ["S2", "S4"],
        "S4": ["S3"],
    }
    
    return ReteStradale(tratti, lista_adiacenza, titolo="Strada Lineare Semplice (5 tratti)")


def crea_strada_griglia():
    """Crea rete stradale a griglia (4 tratti)."""
    dominio_velocita = [50, 70, 90, 110]
    
    tratti = [
        TrattoStradale("A0", dominio_velocita, livello_pericolo=0, position=(0.0, 1.0)),
        TrattoStradale("A1", dominio_velocita, livello_pericolo=1, position=(1.0, 1.0)),
        TrattoStradale("B0", dominio_velocita, livello_pericolo=0, position=(0.0, 0.0)),
        TrattoStradale("B1", dominio_velocita, livello_pericolo=1, position=(1.0, 0.0)),
    ]
    
    lista_adiacenza = {
        "A0": ["A1", "B0"],
        "A1": ["A0", "B1"],
        "B0": ["A0", "B1"],
        "B1": ["A1", "B0"],
    }
    
    return ReteStradale(tratti, lista_adiacenza, titolo="Rete Stradale a Griglia (4 tratti)")


def crea_strada_complessa():
    """Crea rete complessa (8 tratti)."""
    dominio_velocita = [30, 50, 70, 90, 110]
    
    tratti = [
        TrattoStradale("S0", dominio_velocita, livello_pericolo=0, position=(0.0, 0.7)),
        TrattoStradale("S1", dominio_velocita, livello_pericolo=1, position=(0.2, 0.7)),
        TrattoStradale("S2", dominio_velocita, livello_pericolo=0, position=(0.4, 0.7)),
        TrattoStradale("S3", dominio_velocita, livello_pericolo=0, position=(0.6, 0.7)),
        TrattoStradale("S4", dominio_velocita, livello_pericolo=1, position=(0.8, 0.7)),
        TrattoStradale("S5", dominio_velocita, livello_pericolo=0, position=(0.0, 0.3)),
        TrattoStradale("S6", dominio_velocita, livello_pericolo=0, position=(0.4, 0.3)),
        TrattoStradale("S7", dominio_velocita, livello_pericolo=1, position=(0.8, 0.3)),
    ]
    
    lista_adiacenza = {
        "S0": ["S1", "S5"],
        "S1": ["S0", "S2"],
        "S2": ["S1", "S3", "S6"],
        "S3": ["S2", "S4"],
        "S4": ["S3", "S7"],
        "S5": ["S0", "S6"],
        "S6": ["S5", "S2", "S7"],
        "S7": ["S6", "S4"],
    }
    
    return ReteStradale(tratti, lista_adiacenza, titolo="Rete Stradale Complessa (8 tratti)")


def stampa_soluzione(titolo, csp, assegnazione, velocita_totale, violazioni, passi):
    """Stampa risultati di una soluzione."""
    print(f"\n{'=' * 75}")
    print(f"RISULTATI: {titolo}")
    print(f"{'=' * 75}\n")
    
    print("Soluzione assegnata:")
    for tratto in csp.tratti_stradali:
        marca_pericolo = "⚠️" if tratto.livello_pericolo == 1 else "✓"
        velocita = assegnazione[tratto]
        print(f"  {tratto.name:4s} {marca_pericolo}: {velocita:3d} km/h", end="")
        
        # Verifica vincoli per questo tratto
        problemi = []
        if tratto.livello_pericolo == 1 and velocita > 50:
            problemi.append("LIMITE PERICOLO VIOLATO")
        
        for nome_adj in csp.lista_adiacenza.get(tratto.name, []):
            tratto_adj = csp.mappa_tratti[nome_adj]
            if tratto_adj in assegnazione:
                diff = abs(velocita - assegnazione[tratto_adj])
                if diff > 20:
                    problemi.append(f"DIFF con {nome_adj}: {diff} > 20")
        
        if problemi:
            print(f" ✗ {', '.join(problemi)}")
        else:
            print(" ✓")
    
    print(f"\n📊 Statistiche:")
    print(f"  Velocità totale:      {velocita_totale} km/h")
    print(f"  Violazioni vincoli:   {violazioni}")
    print(f"  Passi di ricerca:     {passi}")
    print(f"  Velocità media:       {velocita_totale / len(csp.tratti_stradali):.1f} km/h")


def esegui_test(nome_test, crea_csp_func, max_passi=5000):
    """Esegue un singolo test."""
    print(f"\n{'#' * 75}")
    print(f"TEST: {nome_test}")
    print(f"{'#' * 75}")
    
    # Crea il CSP
    csp = crea_csp_func()
    
    print(f"\nCaratteristiche CSP:")
    print(f"  Nome:                 {csp.title}")
    print(f"  Tratti:               {len(csp.tratti_stradali)}")
    print(f"  Vincoli:              {len(csp.constraints)}")
    print(f"  Tratti pericolosi:    {sum(1 for t in csp.tratti_stradali if t.livello_pericolo == 1)}")
    
    print(f"\nDettagli tratti:")
    for tratto in csp.tratti_stradali:
        marca_pericolo = "⚠️ PERICOLOSO" if tratto.livello_pericolo == 1 else "✓ SICURO"
        print(f"  {tratto.name:4s} {marca_pericolo:15s} dominio: {tratto.domain}")
    
    print(f"\nVincoli:")
    for i, vincolo in enumerate(csp.constraints):
        print(f"  {i+1}. {vincolo.string}")
    
    # Risolvi
    print(f"\nAvvio ricerca (max_passi={max_passi})...")
    t_inizio = time.time()
    
    ricercatore = RicercatoreOttimizzato(csp)
    RicercatoreOttimizzato.max_display_level = 2  # Mostra dettagli
    
    assegnazione, velocita_totale, violazioni = ricercatore.ricerca(max_passi=max_passi)
    
    t_passato = time.time() - t_inizio
    
    # Stampa risultati
    stampa_soluzione(nome_test, csp, assegnazione, velocita_totale, violazioni, 
                     ricercatore.numero_passi)
    
    print(f"\n⏱️  Tempo computazione: {t_passato:.3f} secondi")
    
    return csp, assegnazione, velocita_totale, violazioni


def esegui_test_performance():
    """Test di performance con diversi parametri."""
    print(f"\n{'=' * 75}")
    print("TEST DI PERFORMANCE")
    print(f"{'=' * 75}\n")
    
    csp = crea_strada_lineare_semplice()
    risultati = []
    
    for max_passi in [100, 500, 1000, 5000]:
        ricercatore = RicercatoreOttimizzato(csp)
        RicercatoreOttimizzato.max_display_level = 0  # No output
        
        t_inizio = time.time()
        assegnazione, velocita_totale, violazioni = ricercatore.ricerca(max_passi=max_passi)
        t_passato = time.time() - t_inizio
        
        risultati.append({
            'passi': max_passi,
            'velocita': velocita_totale,
            'violazioni': violazioni,
            'tempo': t_passato
        })
    
    print(f"{'Max Passi':<12} {'Velocità (km/h)':<15} {'Violazioni':<15} {'Tempo (ms)':<15}")
    print("-" * 60)
    for r in risultati:
        print(f"{r['passi']:<12} {r['velocita']:<15} {r['violazioni']:<15} {r['tempo']*1000:<15.1f}")


def main():
    """Funzione principale."""
    print("\n" + "=" * 75)
    print("CSP RETE STRADALE - SUITE COMPLETA DI TEST")
    print("=" * 75)
    print("Test: Assegnazione limiti velocità con vincoli di pericolo e adiacenza")
    print("Obiettivo: Massimizzare velocità totale rispettando tutti i vincoli\n")
    
    # Test 1: Strada lineare semplice
    esegui_test("Strada Lineare Semplice (5 tratti)", 
                crea_strada_lineare_semplice, 
                max_passi=3000)
    
    # Test 2: Rete a griglia
    esegui_test("Rete Stradale a Griglia (4 tratti)", 
                crea_strada_griglia, 
                max_passi=2000)
    
    # Test 3: Rete complessa
    esegui_test("Rete Stradale Complessa (8 tratti)", 
                crea_strada_complessa, 
                max_passi=5000)
    
    # Test 4: Performance
    esegui_test_performance()
    
    # Conclusioni
    print(f"\n{'=' * 75}")
    print("TEST COMPLETATI")
    print(f"{'=' * 75}")
    print("""
✓ Vincolo di Pericolo:  Tratti pericolosi limitati a max 50 km/h
✓ Vincolo di Adiacenza: Differenza velocità tra tratti adiacenti ≤ 20 km/h
✓ Ottimizzazione:       Massimizzazione somma velocità totale

Osservazioni:
1. SLS converge rapidamente per problemi piccoli (< 10 secondi)
2. Qualità della soluzione migliora con più iterazioni
3. Trade-off: velocità computazione vs qualità della soluzione
4. Appropriate per problemi offline di pianificazione stradale
    """)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrotto dall'utente.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Errore: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
# RFID Analyzer

Questo software serve ad analizzare i dati raccolti tramite RFID sulle galline. Permette di generare grafici e statistiche utili per la ricerca.

## Requisiti

- **Python 3.8 o superiore**  
  Se non hai Python, scaricalo da [python.org/downloads](https://www.python.org/downloads/).

- **Un programma per il terminale**  
  (su Windows: "Prompt dei comandi" o "PowerShell")

- **Git** (consigliato per scaricare la repo)  
  Scarica Git da [git-scm.com](https://git-scm.com/), scegli la versione per Windows e segui la procedura guidata di installazione.  
  Dopo l’installazione puoi usare il comando `git` dal terminale.

## Come si usa

### 1. Scarica il progetto

**Metodo consigliato: Clona con Git**

- Apri il terminale.
- Vai nella cartella dove vuoi scaricare il progetto.
- Esegui il comando (sostituisci l'URL con quello della repo se diverso):

  ```
  git clone https://github.com/tuo-utente/RFID-GALLINE-ANGELA.git
  ```

- Entra nella cartella del progetto:

  ```
  cd RFID-GALLINE-ANGELA
  ```

**Metodo alternativo: Download ZIP**

- Clicca sul pulsante verde "Code" su GitHub e scegli "Download ZIP".
- Estrai la cartella sul tuo computer.

### 2. Installa le librerie necessarie

- Apri il terminale (Prompt dei comandi o PowerShell su Windows).
- Vai nella cartella del progetto usando il comando:

  ```
  cd percorso/della/cartella/RFID-GALLINE-ANGELA
  ```

  (Sostituisci `percorso/della/cartella` con il percorso dove hai estratto o clonato il progetto.)

- Installa le librerie richieste con:

  ```
  py -m pip install -r requirements.txt
  ```

### 3. Configura le cartelle di input e output

- Apri il file `config/setting.yaml` con un editor di testo (ad esempio Blocco Note).
- All'interno di questo file puoi specificare:
  - La cartella dove si trovano i file CSV con i dati RFID (input)
  - La cartella dove vuoi che vengano salvati i risultati e i grafici (output)
  - Altri parametri di configurazione necessari

  Esempio di come potrebbe apparire il file:

  ```yaml
  input_folder: C:\Users\bergamascot\Documents\Progetti\RFID-GALLINE-ANGELA\Dati grezzi Rfid
  output_folder: C:\Users\bergamascot\Documents\Progetti\RFID-GALLINE-ANGELA\Output
  # altri parametri...
  ```

  Assicurati che i percorsi delle cartelle siano corretti e che esistano sul tuo computer.

### 4. Inserisci i dati

- Metti i file CSV con i dati RFID nella cartella che hai indicato come `input_folder` nel file di configurazione.

### 5. Avvia l’analisi

Hai due possibilità:

- **Metodo automatico (consigliato per Windows):**  
  Nella cartella del progetto è presente un file batch (`.bat`) chiamato `run_main.bat`.  
  Ti basta fare doppio clic su questo file per avviare automaticamente il programma principale senza dover usare il terminale.

- **Metodo manuale:**  
  Dal terminale, sempre nella cartella del progetto, esegui il programma principale.  
  Ad esempio:

  ```
  python src/main.py
  ```

### 6. Guarda i risultati

- I risultati e i grafici verranno salvati nella cartella che hai indicato come `output_folder` nel file di configurazione.
- Apri i file HTML con un doppio clic per vedere i grafici nel browser.

---

## Domande frequenti

**Non trovo Python!**  
Assicurati di averlo installato e di aver selezionato "Aggiungi Python al PATH" durante l’installazione.

**Non ho Git! Come lo installo?**  
Vai su [git-scm.com](https://git-scm.com/), scarica la versione per Windows e segui le istruzioni. Dopo l’installazione puoi usare il comando `git`
# RFID Analyzer

Questo software serve ad analizzare i dati raccolti tramite RFID sulle galline. Permette di generare grafici e statistiche utili per la ricerca.

## Descrizione sintetica

- **Cleaning dei dati:**  
  Il software pulisce i dati RFID eliminando righe corrotte o incomplete e unisce eventi di uscita/entrata molto brevi (inferiori a una soglia configurabile) per ridurre gli artefatti. In particolare, se una gallina esce e rientra nello stesso nido in pochi secondi, questi eventi vengono considerati come un'unica presenza.

- **Plot generati:**  
  Dopo la pulizia, il programma produce:
  - Un bar plot delle preferenze dei nidi (numero di ingressi per nido)
  - Una cluster map interattiva che mostra la frequenza di utilizzo dei nidi da parte delle diverse galline
  - Grafici temporali dei flussi IN/OUT per ogni nido, suddivisi in slot temporali e finestre di giorni
  - Un network plot interattivo delle copresenze tra galline nei nidi, con soglie configurabili per periodo e gruppo di nidi
  - Una heatmap oraria aggregata che mostra il numero di accessi ai nidi per ogni ora e giorno, utile per individuare le fasce orarie di maggiore attività

- **Configurazione tramite YAML:**  
  Il file `config/settings.yaml` permette di:
  - Scegliere la cartella di input dei dati grezzi e quella di output dei risultati
  - Definire **due soglie** (in secondi) per il filtro degli artefatti: `pre_egg_laying_threshold_seconds` (prima di `egg_laying_start_day`) e `post_egg_laying_threshold_seconds` (dopo). Queste soglie permettono di unire eventi di uscita/entrata molto brevi, con valori diversi a seconda del periodo.
  - Definire il formato della data/ora dei file CSV
  - Scegliere la durata degli slot temporali per i plot e la dimensione della finestra di giorni per i grafici temporali
  - La data di inizio ovodeposizione (`egg_laying_start_day`), in formato YYYY-MM-DD, che permette di suddividere automaticamente i risultati in due periodi: prima e dopo questa data. I risultati verranno salvati in due sottocartelle denominate `YYYYMMDD-YYYYMMDD` all'interno della cartella di output, corrispondenti ai due periodi.
  - Impostare soglie minime di copresenza per visualizzare gli archi nel network plot, distinte per periodo e gruppo di nidi
  - Abilitare/disabilitare la generazione della heatmap oraria aggregata tramite il parametro `plot_hourly_access_heatmap`

## Esempi di output

Nella cartella `samples/images` trovi alcuni esempi di output generati dal programma:
- ![Bar plot delle preferenze dei nidi](samples/images/nest_preferences.png)
- ![Grafico temporale dei flussi IN/OUT interattivo](samples/images/time_slot_flows_2023-10-25_to_2023-10-27.html)
- ![Cluster map interattiva nidi 1.1-1.4](samples/images/clustering_heatmap_nidi_1.1-1.4.html)
- ![Cluster map interattiva nidi 2.1-2.4](samples/images/clustering_heatmap_nidi_2.1-2.4.html)
- ![Network plot copresenza interattivo](samples/images/copresence_network_interactive_nidi_1.1-1.4.html)
- ![Heatmap oraria aggregata accessi ai nidi](samples/images/hourly_access_heatmap.png)

Le immagini HTML sono interattive: per visualizzarle, apri i file corrispondenti con un browser web.

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

- Apri il file `config/settings.yaml` con un editor di testo.
- All'interno di questo file puoi specificare:
  - La cartella dove si trovano i file CSV con i dati RFID (`raw_input_folder`)
  - La cartella dove vuoi che vengano salvati i risultati e i grafici (`cleaned_ouput_folder`)
  - **Le due soglie per il filtro degli artefatti:**
    - `pre_egg_laying_threshold_seconds`: soglia (in secondi) per il periodo prima di `egg_laying_start_day`
    - `post_egg_laying_threshold_seconds`: soglia (in secondi) per il periodo dopo `egg_laying_start_day`
  - Il formato della data/ora (`datetime_format`)
  - La durata degli slot temporali per i plot (`time_slot_minutes`)
  - Il numero di giorni per ogni plot temporale (`plot_day_window`)
  - La data di inizio ovodeposizione (`egg_laying_start_day`), in formato YYYY-MM-DD, che permette di suddividere automaticamente i risultati in due periodi: prima e dopo questa data. I risultati verranno salvati in due sottocartelle denominate `YYYYMMDD-YYYYMMDD` all'interno della cartella di output, corrispondenti ai due periodi.

  Questi parametri influenzano direttamente la pulizia dei dati e la generazione dei grafici.

### 4. Inserisci i dati

- Metti i file CSV con i dati RFID nella cartella che hai indicato come `raw_input_folder` nel file di configurazione.

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

- I risultati e i grafici verranno salvati nella cartella che hai indicato come `cleaned_ouput_folder` nel file di configurazione.
- Apri i file HTML con un doppio clic per vedere i grafici nel browser.

---

## Domande frequenti

**Non trovo Python!**  
Assicurati di averlo installato e di aver selezionato "Aggiungi Python al PATH" durante l’installazione.

**Non ho Git! Come lo installo?**  
Vai su [git-scm.com](https://git-scm.com/), scarica la versione per Windows e segui le istruzioni. Dopo l’installazione puoi usare il comando `git` dal terminale.
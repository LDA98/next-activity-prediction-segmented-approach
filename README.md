# Next Activity Prediction - Segmented Approach

Questo progetto implementa un approccio segmentato per la previsione della prossima attività nei processi, utilizzando reti neurali convoluzionali su grafi (DGCNN) e suddividendo i dati in base alla lunghezza dei prefissi.

## 📁 Struttura del progetto

La directory principale è così organizzata:

- `train-nap-segmented-approach/`  
  Contiene gli script per la generazione dei dataset e l’addestramento dei modelli:
  - `0_prefix_generations.py`: genera i dataset di prefissi suddivisi per lunghezza. Richiede in input il percorso dell’event-log in formato `.g`.
  - `train_model.py`: avvia l'addestramento di un modello DGCNN per ciascuna lunghezza di prefisso utilizzando i dataset generati.

- `test-nap-segmented-approach/`  
  Contiene:
  - le sottocartelle `Helpdesk_1/` e `InternationalDeclaretions_1/`, in cui inserire i modelli addestrati da testare.
  - `config.py`: specificare in `MODELS_PATH` il nome della sottocartella da testare.
  - `test_models.py`: esegue il test dei modelli e stampa le metriche finali: `f1_weighted`, `accuracy`, e la matrice di confusione.

- `requirements.txt`: contiene le librerie necessarie per eseguire il progetto.

## 📊 Dataset utilizzati

- **Helpdesk**
- **Internationals Declaretions**

## ⚙️ Requisiti

- Python 3.10  
- Installazione delle dipendenze tramite:

    ```
    pip install -r requirements.txt
    ```
    
## ▶️ Esecuzione

1. Generare i dataset di prefissi con:

    ```
    python train/0_prefix_generations.py
    ```
    
Prima di procedere bisogna inserire il percorso e il nome del processo in formato `.g` che si vuole analizzare. Questo va fatto nel file `0_prefix_generations.py`:

      ```python
      g_path = ''  # Inserire il percorso della cartella contenente il file .g
      g_final_path = join(g_path, '')  # Inserire il nome del file .g
      ```
È possibile modificare nella funzione `find_balanced_cutoff(prefix_counts, threshold_ratio=0.7)` il valore del parametro `threshold_ratio` per definire la lunghezza `l` a partire dalla quale tutte le lunghezze maggiori o uguali saranno aggregate in un unico gruppo.

Inoltre, per uno sviluppo futuro, è possibile specificare manualmente le aggregazioni desiderate nel file `config.py` tramite la variabile `MERGE_LEN`, (il dataset    risultante sarà ordinato per lunghezza).  
Ad esempio:

```
"""
Configurazione che specifica quali lunghezze unire
MERGE_LEN = [
    [2, 3],        # Unire i prefissi di lunghezze 2 e 3
    [4, 5, 6]      # Unire i prefissi di lunghezze 4, 5 e 6
]
Risultato -> {[2, 3], [4, 5, 6], [7], [8+]}
"""
MERGE_LEN = []
```

2. Addestrare i modelli:

    ```
    python train/train_model.py
    ```
    
3.Inserire i modelli ottenuti nella cartella corrispondente all’event-log dentro test-nap-segmented-approach/[`Helpdesk_1/` `InternationalDeclaretions_1/`].

4.Configurare il file test-nap-segmented-approach/config.py impostando il parametro MODELS_PATH.

5.Eseguire il test:

    ```
    python test/test_models.py
    ```

Prima di procedere bisogna inserire il nome completo di estensione del dataset .pt che si vuole testare. Questo va fatto nel file `test_models.py`:

      ```
      dataset_path = join(DATASET_PATH, '') # .pt
      ```

import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader
import numpy as np
from os import listdir
from os.path import join, split
from sklearn.metrics import accuracy_score, f1_score
from config import MODEL, DATASET_PATH, MODELS_PATH
from tqdm import tqdm
from collections import OrderedDict
import matplotlib
matplotlib.use('TkAgg')  # O 'Qt5Agg'
import matplotlib.pyplot as plt
from itertools import product
import pandas as pd
import re


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

class PrefixIGs(InMemoryDataset):
    def __init__(self, ds_path):
        super(PrefixIGs, self).__init__()
        self.data, self.slices = torch.load(ds_path)

def parse_length_range(length_str):
    """Parsa le lunghezze dai nomi dei file e restituisce un elenco di lunghezze gestite."""
    if '+' in length_str:
        parts = length_str.split('+')
        if parts[-1] == "":  
            return [int(parts[0])]  
        return list(map(int, parts))  
    return [int(length_str)]

def load_models(models_dir, dataset):
    """
    Carica i modelli dalla directory, estraendo i parametri dal nome del file in modo robusto.
    """
    models = {}

    # Lista di tutti i file modello
    model_files = [f for f in listdir(models_dir) if f.endswith('model.pt')]
    for model_file in tqdm(model_files, desc="Loading models"):
        try:
            # Parsing robusto usando regex
            dropout = None
            graph_conv_layers = None
            num_neurons = None
            k = None

            # Estrarre parametri tramite regex
            # Esempio: `_0.1_d_`, `_2_gcl_`, `_64_n_`, `_30_k_`
            dropout_match = re.search(r'_(\d+\.\d+)_d_', model_file)  # cerca float in `_d_`
            gcl_match = re.search(r'_(\d+)_gcl_', model_file)  # cerca int in `_gcl_`
            neurons_match = re.search(r'_(\d+)_n_', model_file)  # cerca int in `_n_`
            k_match = re.search(r'_(\d+)_k_', model_file)  # cerca int in `_k_`

            # Risultati del parsing
            if dropout_match:
                dropout = float(dropout_match.group(1))
            if gcl_match:
                graph_conv_layers = int(gcl_match.group(1))
            if neurons_match:
                num_neurons = int(neurons_match.group(1))
            if k_match:
                k = int(k_match.group(1))

            # Verifica che tutti i parametri siano stati estratti
            if not all([dropout, graph_conv_layers, num_neurons, k]):
                raise ValueError(f"Missing parameters in filename: {model_file}")

            # Gestione delle lunghezze
            # `_pl_` indica un range di lunghezze
            if '_pl_' in model_file:
                length_str = model_file.split('_pl_')[1].split('_')[0]
                lengths = parse_length_range(length_str)  # Ottieni tutte le lunghezze per questo modello
            else:
                lengths = [1]  # Modello base con una sola lunghezza

            # Percorso del modello
            model_path = join(models_dir, model_file)

            # Caricamento del modello
            model = MODEL(
                dataset=dataset,
                num_layers=graph_conv_layers,
                dropout=dropout,
                num_neurons=num_neurons,
                k=k
            )
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()

            # Associazione del modello alle lunghezze
            for length in lengths:
                models[length] = model

        except Exception as e:
            print(f"Skipping model due to error in filename {model_file}: {e}")
            continue

    # Ordina il dizionario per chiave crescente e ritorna l'OrderedDict
    models = OrderedDict(sorted(models.items()))
    return models

def plot_confusion_matrix(reals, predictions, dataset_path):
    new_dataset_path = dataset_path.split('.pt')[0]
    with open(f'{new_dataset_path}_activities.txt', 'r') as f:
        activities = []
        for lines in f.readlines():
            lines = lines[:-1]
            activities.append(lines)

    plt.figure(figsize=(15, 15))
    stacked = torch.stack((reals, predictions), dim=1)
    cmt = torch.zeros(len(activities), len(activities), dtype=torch.int64)
    for p in stacked:
        tl, pl = p.tolist()
        cmt[int(tl), int(pl)] = cmt[int(tl), int(pl)] + 1

    plt.imshow(cmt, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    tick_marks = np.arange(len(activities))
    plt.xticks(tick_marks, activities, rotation=45)
    plt.yticks(tick_marks, activities)

    for i, j in product(range(cmt.shape[0]), range(cmt.shape[1])):
        plt.text(j, i, format(cmt[i, j], 'd'), horizontalalignment="center",
                 color="white" if cmt[i, j] > cmt.max() / 2. else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()
    plt.close('all')

def test_models():
    dataset_path = join(DATASET_PATH, '') # .pt
    dataset = PrefixIGs(dataset_path)
    test_dataset = [data for data in dataset if data.set == 'test']
    test_loader = DataLoader(dataset=test_dataset, batch_size=64)

    print(f"Testing..")
    models = load_models(MODELS_PATH, dataset)

    max_length = max(models.keys())  # Lunghezza massima gestita dai modelli
    reals, predicted = torch.tensor([], device=device), torch.tensor([], device=device)
    output_details = []  # List to store prediction details

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing models"):
            batch = batch.to(device)
            prefix_lens = batch.prefix_len.tolist()

            class_pred = torch.tensor([], device=device)
            class_real = batch.y.argmax(dim=1)

            # Trova il modello giusto basato sulla lunghezza del prefisso
            for i, length in enumerate(prefix_lens):
                slenght = length
                if length > max_length:
                    length = max_length  # Usa sempre il modello per la lunghezza massima disponibile

                model = models.get(length, models[1])  # Se non esiste la lunghezza, usa il modello massimo
                graph_data = batch[i]
                pred = model(graph_data)

                class_pred = torch.concat([class_pred, pred.argmax(dim=1)])
                # Save prediction details
                output_details.append({
                    'real': int(class_real[i].item()),
                    'predicted': int(pred.argmax(dim=1).item()),
                    'prefix_length': slenght
                })
                
            reals, predicted = torch.concat([reals, class_real]), torch.concat([predicted, class_pred])

    # Save prediction details to a CSV file
    details_df = pd.DataFrame(output_details)
    details_df.to_csv('prediction_details.csv', index=False)

    # Group predictions by prefix length and calculate weighted F1-score for each group
    prefix_groups = details_df.groupby('prefix_length')
    prefix_f1_results = []

    for prefix_length, group in prefix_groups:
        acc = accuracy_score(group['real'], group['predicted'])
        f1 = f1_score(group['real'], group['predicted'], average='weighted')
        prefix_f1_results.append({'prefix_length': prefix_length, 'accuracy': acc,'f1_score_weighted': f1})

    # Save aggregated prefix F1-score results to a second CSV file
    results_df = pd.DataFrame(prefix_f1_results)
    results_df.to_csv('prefix_f1_results.csv', index=False)

    # Calcola metriche
    accuracy = float(accuracy_score(reals.cpu().numpy(), predicted.cpu().numpy()))
    f1_w = float(f1_score(reals.cpu().numpy(), predicted.cpu().numpy(), average='weighted'))

    print(f"\nFinal Results - Accuracy: {accuracy}, F1-score: {f1_w}\n")
    plot_confusion_matrix(reals.cpu(), predicted.cpu(), dataset_path)

if __name__ == '__main__':
    test_models()

import torch
from torch_geometric.data import Data, InMemoryDataset
from pandas import Series, get_dummies, read_csv
import numpy as np
import networkx as nx
from os.path import join, split
import os
from config import DATASET_PATH, TRAIN_SPLIT, MIN_LEN, MERGE_LEN


def verify_graph(g):
    # almeno MIN_LEN nodi
    if len(g.nodes()) < MIN_LEN:
        return False
    # unica componente connessa
    if nx.number_connected_components(g.to_undirected()) != 1:
        return False
    # qualche nodo non ha l'attività
    if not (all('concept:name' in node for _, node in g.nodes(data=True))):
        return False
    else:
        return True
    
def find_balanced_cutoff(prefix_counts, threshold_ratio=0.7):
    
    # Ottieni il numero di campioni per la lunghezza 2
    length_2_count = prefix_counts.get(2, 0)

    # Se non ci sono campioni di lunghezza 2, non possiamo applicare questa logica
    if length_2_count == 0:
        raise ValueError("Non ci sono campioni con lunghezza 2 nel dataset.")

    # Identifica il cutoff
    for length, count in prefix_counts.items():
        if count < threshold_ratio * length_2_count:
            return length - 1  # Ritorna la lunghezza precedente come cutoff

    return max(prefix_counts.keys())  # Se nessuna lunghezza soddisfa la condizione, ritorna la lunghezza massima

def merge_datasets_by_length(prefix_igs_by_length, merge_config):
    """
    Unisce i sotto-dataset in base alla configurazione fornita.
    """
    merged_prefix_igs = {}

    # Creare un set di lunghezze già unite per evitare duplicazioni
    merged_lengths = set()

    for group in merge_config:
        # Creare una chiave rappresentativa per il gruppo, ad esempio "2+3" per le lunghezze 2 e 3
        group_key = "+".join(map(str, group))
        merged_prefix_igs[group_key] = []

        for length in group:
            if length in prefix_igs_by_length:
                merged_prefix_igs[group_key].extend(prefix_igs_by_length[length])
                merged_lengths.add(length)
        
    # Aggiungere i sotto-dataset non uniti alla configurazione originale
    for length, prefixes in prefix_igs_by_length.items():
        if length not in merged_lengths:
            merged_prefix_igs[str(length)] = prefixes

    return merged_prefix_igs


def generate_prefix_igs(graphs):
    prefix_igs_by_length = {}  # Dizionario per raccogliere prefissi per lunghezza

    for graph_num, graph in enumerate(graphs):
        print(f'Processing graph: {graph_num + 1}/{len(graphs)}')
        prefix_ig_k = nx.DiGraph()
        prefix_ig_k.graph['set'] = graph.graph['set']

        for node in list(graph.nodes()):

            prefix_ig_k.graph['y'] = graph.nodes[node]['concept:name']

            if verify_graph(prefix_ig_k):
                num_nodes = len(prefix_ig_k.nodes())

                if num_nodes not in prefix_igs_by_length:
                    prefix_igs_by_length[num_nodes] = []
                prefix_igs_by_length[num_nodes].append(prefix_ig_k.copy().to_undirected())
                

            attrs = graph.nodes[node]
            prefix_ig_k.add_node(node, **attrs)
            for n in graph.neighbors(node):
                prefix_ig_k.add_edge(n, node)
        
    # Conta il numero di prefissi per ogni lunghezza
    prefix_count_by_length = {length: len(prefix_list) for length, prefix_list in prefix_igs_by_length.items() }

   # Trova il cutoff dinamico per bilanciare il dataset
    cutoff = find_balanced_cutoff(prefix_count_by_length)

    # Raggruppa tutti i prefissi con lunghezza maggiore del cutoff
    prefix_igs_long = []
    for length in list(prefix_igs_by_length.keys()):
        if length > cutoff:
            prefix_igs_long.extend(prefix_igs_by_length[length])
            del prefix_igs_by_length[length]  # Rimuove la chiave dopo averla trasferita

    if prefix_igs_long:
        prefix_igs_by_length[f"{cutoff + 1}+"] = prefix_igs_long

    if MERGE_LEN:
        return merge_datasets_by_length(prefix_igs_by_length, MERGE_LEN)
    else:
        return prefix_igs_by_length


def encode_prefix_igs(prefix_igs, prefix_igs_path, ohe_activities):
    features = ['norm_time', 'trace_time', 'prev_event_time']
    data_list = []
    for index, prefix_ig in enumerate(prefix_igs):
        x1 = []
        print(f'Creating tensor: {index + 1}/{len(prefix_igs)}')
        for node_id in prefix_ig.nodes:
            node = prefix_ig.nodes[node_id]
            node_features = []
            node_features.extend(ohe_activities[node['concept:name']])

            for attr in features:
                if attr in node:
                    node_features.append(float(node[attr]))

            x1.append(node_features)

        x = torch.tensor(x1, dtype=torch.float)
        y = torch.tensor([ohe_activities[prefix_ig.graph['y']]], dtype=torch.float)
        prefix_len = torch.tensor(len(x1), dtype=torch.int)
        adj = nx.to_scipy_sparse_array(prefix_ig)  # prende la matrice di adiacenza del grafo
        adj = adj.tocoo()
        row = torch.from_numpy(adj.row.astype(np.int64)).to(torch.long)
        col = torch.from_numpy(adj.col.astype(np.int64)).to(torch.long)
        edge_index = torch.stack([row, col], dim=0)
        data = Data(x=x, edge_index=edge_index, y=y, prefix_len=prefix_len, set=prefix_ig.graph['set'])
        data_list.append(data)

    data, slices = InMemoryDataset.collate(data_list)
    torch.save((data, slices), prefix_igs_path)


def process(g_final_path):
    def read_dataset(g_final_path, attributes_path):
        g_dataframe = read_csv(g_final_path, header=0, sep=' ')
        graphs, attributes = [], []
        xp_indices = g_dataframe[g_dataframe['e_v'] == 'XP'].index
        for start, end in zip(xp_indices, xp_indices[1:]):
            sub_df = g_dataframe.iloc[start:end]
            for index, row in sub_df.iterrows():

                if row['e_v'] == 'XP':
                    g = nx.DiGraph()

                if row['e_v'] == 'v':
                    node_nr = int(float(row['node1']) - 1)
                    node_attributes = {}
                    to_ignore = ['e_v', 'node1', 'node2']
                    for col_name in sub_df.columns[3:]:
                        if col_name in to_ignore:
                            continue
                        if col_name == 'concept:name':
                            activity = str(row[col_name])
                            node_attributes[col_name] = activity

                            # salvo tutte le Activity, mi serve per dopo quando devo creare il ohe vector
                            if activity not in attributes:
                                attributes.append(activity)
                        else:
                            node_attributes[col_name] = row[col_name]

                    g.add_node(node_nr, **node_attributes)

                elif row['e_v'] == 'e':
                    g.add_edge(int(float(row['node1']) - 1), int(float(row['node2']) - 1))

            if verify_graph(g):
                # salvo il grafo con gli archi invertiti
                graphs.append(g.reverse())

        os.makedirs(os.path.dirname(attributes_path), exist_ok=True)

        with open(attributes_path, 'w') as f:
            for att in attributes:
                f.write(att + '\n')

        num_graphs_train = int(len(graphs) * TRAIN_SPLIT)
        for idx, g in enumerate(graphs):
            g.graph['set'] = 'train' if idx < num_graphs_train else 'test'
        return graphs


    def get_ohe_activities(attributes_path):
        attr = []
        with open(attributes_path, 'r') as f:
            for lines in f.readlines():
                lines = lines[:-1]
                attr.append(lines)  # ricrea la lista degli attributi
        s1 = Series(attr)  # crea una serie come valori le attività
        s2 = get_dummies(s1)  # crea dataframe con tante colonne quante le attività e valori solo 0 e 1
        onedictfeat = {}
        # crea dizionario: chiave=chiave dataframe, valore = dizionario (chiave=colonna dataframe, valore=0 o 1)
        s3 = s2.to_dict()
        for a, b in s3.items():
            onedictfeat[a] = list(b.values())  # nuovo dizionario (valore=lista valori con stessa chiave)
        # print('onedictfeat',onedictfeat)
        return onedictfeat


    g_name = split(g_final_path)[1][:-2]
    attributes_path = join(DATASET_PATH, f'{g_name}_activities.txt')

    graphs = read_dataset(g_final_path, attributes_path)
    ohe_activities = get_ohe_activities(attributes_path)

    prefix_igs_by_length = generate_prefix_igs(graphs)

    for length, prefix_igs in prefix_igs_by_length.items():
        prefix_igs_path = join(DATASET_PATH, f'{g_name}_pl_{length}.pt')
        encode_prefix_igs(prefix_igs, prefix_igs_path, ohe_activities)

if __name__ == '__main__':
    # path del file '.g'
    g_path = ''
    g_final_path = join(g_path, '')
    process(g_final_path)   
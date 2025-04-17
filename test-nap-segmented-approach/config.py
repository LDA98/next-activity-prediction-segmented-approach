from os.path import join, dirname
from os import getcwd, makedirs
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

DIR_PATH = dirname(join(getcwd(), __file__))
DATASET_PATH = join(DIR_PATH, 'dataset')
MODELS_PATH = join(DIR_PATH, 'InternationalDeclarations_1') #   Helpdesk_1  InternationalDeclarations_1
MIN_LEN = 2

"""
Configurazione che specifica quali lunghezze unire
MERGE_LEN = [
    [2, 3],  # Unire i prefissi di lunghezze 2 e 3
    [4, 5, 6]  # Unire i prefissi di lunghezze 4, 5 e 6
]
risultato -> {[2, 3],[4, 5, 6],[7],[8+]}
"""
MERGE_LEN = []

def get_result_path(ds_name, model):
    result_path = join(DIR_PATH, 'NAP', 'results', model, ds_name)
    makedirs(result_path, exist_ok=True)
    return result_path

# TRAINING PARAMETERS
SEED = 42
TRAIN_SPLIT = 0.67
PATIENCE = 20
EPOCH = [100]
BATCH_SIZE = [64]
# LEARNING_RATE = [1e-2, 1e-3, 1e-4]
LEARNING_RATE = [1e-2]
DROPOUT = [0.1]
# DROPOUT = [0.1, 0.2]
# ----------


# MODEL PARAMETERS
MODEL_TYPE = 'DGCNN'  # ['MPGNN', 'DGCNN']
if MODEL_TYPE.upper() == 'DGCNN':
    from DGCNN import DGCNN as MODEL

# il numero di layer convoluzionali
# GRAPH_CONV_LAYERS = [2, 3, 5, 7]
GRAPH_CONV_LAYERS = [2]

# numero di neuroni per layer convoluzionale
NUM_NEURONS = [64]

# il numero di nodi selezionati dal sort pooling
# K_VALUES = [3, 5, 7, 30]
K_VALUES = [3]

GRID_COMBINATIONS = []
for batch_size in BATCH_SIZE:
    for epochs in EPOCH:
        for k in K_VALUES:
            for num_neurons in NUM_NEURONS:
                for graph_conv_layers in GRAPH_CONV_LAYERS:
                    for dropout_rate in DROPOUT:
                        for learning_rate in LEARNING_RATE:
                            if MODEL_TYPE.upper() == 'DGCNN' and k < graph_conv_layers:
                                continue
                            tmp_graph_conv_layer_size = ''
                            data_comb = (f'{learning_rate}_lr_{batch_size}_'
                                         f'bs_{dropout_rate}_d_{graph_conv_layers}_gcl'
                                         f'_{num_neurons}_n_{k}_k')
                            GRID_COMBINATIONS.append({
                                'path': data_comb,
                                'batch_size': batch_size,
                                'epochs': epochs,
                                'learning_rate': learning_rate,
                                'dropout': dropout_rate,
                                'graph_conv_layers': graph_conv_layers,
                                'num_neurons': num_neurons,
                                'k': k})
# ----------

import pandas as pd
from os.path import join, isfile, split
from os import listdir, makedirs
from pandas import DataFrame
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
from itertools import product
import torch
import numpy as np
from config import get_result_path


def _calculate_accuracy(group):
    return accuracy_score(group['y_real'], group['y_estimated'])


def _calculate_f1(group):
    return f1_score(group['y_real'], group['y_estimated'], average='weighted')  # , zero_division=1))


def calculate_metrics_by_prefixes(prefix_results, path, set):
    # metriche dell'epoca al variare della lunghezza del prefisso per il set corrente
    accuracy = prefix_results.groupby('prefix_len').apply(_calculate_accuracy).reset_index(name='accuracy')
    w_f1 = prefix_results.groupby('prefix_len').apply(_calculate_f1).reset_index(name='w_f1')

    result = pd.merge(accuracy, w_f1, on='prefix_len')
    plt.figure(figsize=(15, 6))

    plt.title(f"Metrics varying prefix size on {set} set")
    plt.xlabel(f"Prefix size")
    plt.ylabel(f"Metric")
    plt.xticks([i for i in range(1, len(result['prefix_len']), 2)])
    plt.plot(result['prefix_len'], result['accuracy'], label='Accuracy')
    plt.plot(result['prefix_len'], result['w_f1'], label='Weighted F1')

    # Normalizzazione scala asse y
    min_metric = min(result['accuracy'].min(), result['w_f1'].min())
    max_metric = max(result['accuracy'].max(), result['w_f1'].max())
    if min_metric == max_metric:
        margin = 0.1
    else: margin = 0.05 * (max_metric - min_metric)  # 5% dell'intervallo
    plt.ylim(min_metric - margin, max_metric + margin)  

    plt.tight_layout()
    plt.legend()
    plt.savefig(join(path, f'prefix_metrics_{set}.png'))
    result.to_csv(join(path, f'prefix_metrics_{set}.csv'), index=False)
    plt.close('all')


def plot_combination_metrics(results, path):
    comb_path = split(path)[0]

    # Calcola i limiti comuni per ogni metrica
    acc_min = min(results['train_acc'].min(), results['test_acc'].min())
    acc_max = max(results['train_acc'].max(), results['test_acc'].max())
    if acc_min == acc_max:
        acc_margin = 0.1
    else: acc_margin = 0.05 * (acc_max - acc_min)  # 5% dell'intervallo

    f1_min = min(results['train_f1'].min(), results['test_f1'].min())
    f1_max = max(results['train_f1'].max(), results['test_f1'].max())
    if f1_min == f1_max:
        f1_margin = 0.1
    else: f1_margin = 0.05 * (f1_max - f1_min)  # 5% dell'intervallo

    loss_min = min(results['train_loss'].min(), results['test_loss'].min())
    loss_max = max(results['train_loss'].max(), results['test_loss'].max())
    if loss_min == loss_max:
        loss_margin = 0.1
    else: loss_margin = 0.05 * (loss_max - loss_min)  # 5% dell'intervallo


    # comparing train and test metrics
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(15, 6))
    ax1, ax2, ax3 = axes.flatten()
    ax1.set_title(f"Accuracy on train and test set")
    ax1.set_xlabel(f"Epochs")
    ax1.set_ylabel(f"Accuracy")
    ax1.plot(results['epoch'], results['train_acc'], label='train')
    ax1.plot(results['epoch'], results['test_acc'], label='test')
    ax1.set_ylim(acc_min - acc_margin, acc_max + acc_margin)
    ax1.legend()

    ax2.set_title(f"Weighted F1 on train and test set")
    ax2.set_xlabel(f"Epochs")
    ax2.set_ylabel(f"Weighted F1")
    ax2.plot(results['epoch'], results['train_f1'], label='train')
    ax2.plot(results['epoch'], results['test_f1'], label='test')
    ax2.set_ylim(f1_min - f1_margin, f1_max + f1_margin)
    ax2.legend()

    ax3.set_title(f"Loss on train and test set")
    ax3.set_xlabel(f"Epochs")
    ax3.set_ylabel(f"Loss")
    ax3.plot(results['epoch'], results['train_loss'], label='train')
    ax3.plot(results['epoch'], results['test_loss'], label='test')
    ax3.set_ylim(loss_min - loss_margin, loss_max + loss_margin)
    ax3.legend()

    plt.tight_layout()
    plt.savefig(join(comb_path, f'compare_train_test_metrics.png'))
    plt.close('all')

    # splitting train and test metrics
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 6))
    ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()
    ax1.set_title(f"Accuracy on train set")
    ax1.set_xlabel(f"Epochs")
    ax1.set_ylabel(f"Accuracy")
    ax1.plot(results['epoch'], results['train_acc'])
    ax1.set_ylim(acc_min - acc_margin, acc_max + acc_margin)


    ax4.set_title(f"Accuracy on test set")
    ax4.set_xlabel(f"Epochs")
    ax4.set_ylabel(f"Accuracy")
    ax4.plot(results['epoch'], results['test_acc'])
    ax4.set_ylim(acc_min - acc_margin, acc_max + acc_margin)


    ax2.set_title(f"Weighted F1 on train set")
    ax2.set_xlabel(f"Epochs")
    ax2.set_ylabel(f"Weighted F1")
    ax2.plot(results['epoch'], results['train_f1'])
    ax2.set_ylim(f1_min - f1_margin, f1_max + f1_margin)


    ax5.set_title(f"Weighted F1 on test set")
    ax5.set_xlabel(f"Epochs")
    ax5.set_ylabel(f"Weighted F1")
    ax5.plot(results['epoch'], results['test_f1'])
    ax5.set_ylim(f1_min - f1_margin, f1_max + f1_margin)

    ax3.set_title(f"Loss on train set")
    ax3.set_xlabel(f"Epochs")
    ax3.set_ylabel(f"Loss")
    ax3.plot(results['epoch'], results['train_loss'])
    ax3.set_ylim(loss_min - loss_margin, loss_max + loss_margin)


    ax6.set_title(f"Loss on test set")
    ax6.set_xlabel(f"Epochs")
    ax6.set_ylabel(f"Loss")
    ax6.plot(results['epoch'], results['test_loss'])
    ax6.set_ylim(loss_min - loss_margin, loss_max + loss_margin)


    plt.tight_layout()
    plt.savefig(join(comb_path, f'combination_metrics.png'))
    plt.close('all')


def best_metric_on_set(df, ds_name, model, metric, set):
    path_to_save = get_result_path(ds_name, model)
    makedirs(path_to_save, exist_ok=True)
    best_df = DataFrame()
    on_metric = f'{set}_{metric}'
    for comb in df['combination'].unique().tolist():
        comb_df = df.loc[df['combination'] == comb]
        if metric == 'loss':
            avg_best = comb_df.loc[(comb_df[on_metric] == comb_df[on_metric].min())]
        else:
            avg_best = comb_df.loc[(comb_df[on_metric] == comb_df[on_metric].max())]

        best_df = pd.concat([best_df, avg_best])

    ascending = True if metric == 'loss' else False
    best_df.sort_values(by=[on_metric], ascending=ascending, inplace=True)
    best_df.to_csv(join(path_to_save, f'{ds_name}_best_{on_metric}.csv'), header=True, index=False, sep=',')


def plot_confusion_matrix(results, classes, epoch_path, set):
    plt.figure(figsize=(15, 15))
    reals = torch.tensor(results['y_real'].tolist())
    predictions = torch.tensor(results['y_estimated'].tolist())
    stacked = torch.stack((reals, predictions), dim=1)
    cmt = torch.zeros(len(classes), len(classes), dtype=torch.int64)
    for p in stacked:
        tl, pl = p.tolist()
        cmt[int(tl), int(pl)] = cmt[int(tl), int(pl)] + 1

    plt.imshow(cmt, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    for i, j in product(range(cmt.shape[0]), range(cmt.shape[1])):
        plt.text(j, i, format(cmt[i, j], 'd'), horizontalalignment="center",
                 color="white" if cmt[i, j] > cmt.max() / 2. else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(join(epoch_path, f'confusion_matrix_{set}.png'), pad_inches=5)
    plt.close('all')


def _get_combs_csv(results_path):
    csv_filepaths = []
    runs = list(listdir(results_path))
    for run in runs:
        # results/HD/YYYY-MM-DD_HH_mm_ss
        if run == '.DS_Store' or isfile(join(results_path, run)):
            continue
        combinations = list(listdir(join(results_path, run)))
        for combination in combinations:
            # results/HD/YYYY-MM-DD_HH_mm_ss/comb
            if combination == '.DS_Store':
                continue
            comb_path = join(results_path, run, combination)
            comb_items = list(listdir(comb_path))
            # results/HD/YYYY-MM-DD_HH_mm_ss/comb/epoch
            for comb_item in comb_items:
                if comb_item == '.DS_Store':
                    continue
                # results/HD/YYYY-MM-DD_HH_mm_ss/comb/results_comb.csv
                item_path = join(comb_path, comb_item)
                is_result_csv = isfile(item_path) and comb_item.startswith('results') and comb_item.endswith('.csv')

                if is_result_csv:
                    csv_filepaths.append(item_path)
        return csv_filepaths


def _get_epoch_csv(combination_csv):
    comb_path = split(combination_csv)[0]
    epochs = []
    for epoch in listdir(comb_path):
        epoch_dir = join(comb_path, epoch)
        if not isfile(epoch_dir):
            for item in listdir(epoch_dir):
                if item.endswith('_prefix_results.csv'):
                    epochs.append(join(epoch_dir, item))

    return epochs


def eval_results(dataset_path, model, ds_name):
    if "_pl" in dataset_path:
        new_dataset_path = dataset_path.split("_pl")[0]
    else:
        new_dataset_path = dataset_path.split(".pt")[0]
    with open(f'{new_dataset_path}_activities.txt', 'r') as f:
        activities = []
        for lines in f.readlines():
            lines = lines[:-1]
            activities.append(lines)

    results_path = get_result_path(ds_name, model)
    df_results = pd.DataFrame()
    combinations_csv = _get_combs_csv(results_path)
    for idx, comb_csv in enumerate(combinations_csv):
        print(f'\n** Processing comb: {split(comb_csv)[1]} ({idx}/{len(combinations_csv)})')
        comb_results = pd.read_csv(comb_csv, header=0)
        # metriche per la combinazione corrente al variare dell'epoca, sia train che test
        plot_combination_metrics(comb_results, comb_csv)
        df_results = pd.concat([df_results, comb_results])
        epochs_comb = _get_epoch_csv(comb_csv)
        
        for epoch_comb in sorted(epochs_comb):
            print(f'** Processing epoch: {split(epoch_comb)[1]}')
            # metriche per l'epoca corrente al variare della lunghezza di prefisso, sia train che test
            epoch_results = pd.read_csv(epoch_comb, header=0)
            epoch_path = split(epoch_comb)[0]

            epoch_results_train = epoch_results.loc[epoch_results['set'] == 'train']
            plot_confusion_matrix(epoch_results_train, activities, epoch_path, 'train')
            calculate_metrics_by_prefixes(epoch_results_train, epoch_path, 'train')

            epoch_results_test = epoch_results.loc[epoch_results['set'] == 'test']
            plot_confusion_matrix(epoch_results_test, activities, epoch_path, 'test')
            calculate_metrics_by_prefixes(epoch_results_test, epoch_path, 'test')

    print('\n** Best combinations..')
    best_metric_on_set(df_results, ds_name, model, metric='loss', set='test')

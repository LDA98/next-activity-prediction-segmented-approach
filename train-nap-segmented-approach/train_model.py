import pandas as pd
from time import time
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader
import random
from datetime import datetime
import numpy as np
from os import listdir, makedirs
from os.path import join, split
from sklearn.metrics import accuracy_score, f1_score
from results_evaluation import eval_results
from config import MODEL_TYPE, MODEL, DATASET_PATH, PATIENCE, SEED, GRID_COMBINATIONS, get_result_path

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.enabled = False
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

class PrefixIGs(InMemoryDataset):
    def __init__(self, ds_path):
        super(PrefixIGs, self).__init__()
        self.data, self.slices = torch.load(ds_path)


if __name__ == '__main__':
    print(f'\n**Model: {MODEL_TYPE}**')
    # list of datasets
    datasets = [join(DATASET_PATH, name) for name in listdir(DATASET_PATH) if name.endswith('.pt')]

    for dataset_path in datasets:
        ds_name = split(dataset_path)[1].split('.')[0]
        print(f'**Processing: {ds_name}**')

        result_path = get_result_path(ds_name, MODEL_TYPE)
        run_time = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        run_path = join(result_path, run_time)

        # import dataset
        G = PrefixIGs(dataset_path)

        # train/test split
        train_dataset = [data for data in G if data.set == 'train']
        test_dataset = [data for data in G if data.set == 'test']

        actual_comb, total_combs = 0, len(GRID_COMBINATIONS)
        for comb in GRID_COMBINATIONS:
            # variables
            batch_size, epochs, k, num_neurons = comb['batch_size'], comb['epochs'], comb['k'], comb['num_neurons']
            graph_conv_layers, learning_rate, dropout = comb['graph_conv_layers'], comb['learning_rate'], comb['dropout']

            # path of current combination
            comb_string = comb['path']
            comb_path = join(run_path, comb_string)
            makedirs(comb_path, exist_ok=True)

            # network parameters
            train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size)
            test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size)
            model = MODEL(dataset=G, num_layers=graph_conv_layers, dropout=dropout, num_neurons=num_neurons, k=k)
            model = model.to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            criterion = torch.nn.CrossEntropyLoss()

            results_df = pd.DataFrame(columns=['run', 'combination', 'epoch', 'train_acc', 'test_acc', 'train_f1',
                                               'test_f1', 'train_loss', 'test_loss', 'best_test_loss'])

            print(f'\n** NEW COMBINATION STARTED ({actual_comb + 1}/{total_combs}) **\n'
                  f'** K:{k}, LAYERS: {graph_conv_layers}, D: {dropout}, LR: {learning_rate} **')
            no_improvements, best_test_loss = 0, np.inf
            for epoch in range(epochs):
                prefix_results = []
                epoch_path = join(comb_path, f'{epoch}')
                makedirs(epoch_path, exist_ok=True)
                start_epoch_time = time()

                # train model
                loss_train = 0
                reals, predicted, prefix_pred, prefix_3 = torch.tensor([], device=device), torch.tensor([], device=device), [], []
                model.train()
                for batch in train_loader:
                    batch = batch.to(device)
                    pred = model(batch)
                    class_pred, class_real = pred.argmax(dim=1), batch.y.argmax(dim=1)
                    reals, predicted = torch.concat([reals, class_real]), torch.concat([predicted, class_pred])

                    # train loss
                    loss = criterion(pred, class_real)
                    loss_train += loss.item()

                    # Backpropagation
                    loss.backward()  # compute parameters gradients
                    optimizer.step()  # update parameters
                    optimizer.zero_grad()  # reset the gradients of all parameters

                    # train results
                    prefix_pred.extend(zip(['train'] * len(batch.prefix_len), [epoch] * len(batch.prefix_len),
                                           batch.prefix_len.tolist(), class_pred.tolist(), class_real.tolist())                    )
                # train epoch metrics
                loss_train /= len(train_loader)
                accuracy_train = float(accuracy_score(reals.tolist(), predicted.tolist()))
                f1_w_train = float(f1_score(reals.tolist(), predicted.tolist(), average='weighted'))  # , zero_division=1))
                prefix_results.extend(prefix_pred)

                # test model
                model.eval()
                loss_test = 0
                reals, predicted, prefix_pred = torch.tensor([], device=device), torch.tensor([], device=device), []
                with torch.no_grad():
                    for batch in test_loader:
                        batch = batch.to(device)
                        pred = model(batch)
                        class_pred, class_real = pred.argmax(dim=1), batch.y.argmax(dim=1)
                        reals, predicted = torch.concat([reals, class_real]), torch.concat([predicted, class_pred])

                        # test loss
                        loss = criterion(pred, class_real)
                        loss_test += loss.item()

                        # test results
                        prefix_pred.extend(zip(['test'] * len(batch.prefix_len), [epoch] * len(batch.prefix_len),
                                               batch.prefix_len.tolist(), class_pred.tolist(), class_real.tolist()))

                # test epoch metrics
                loss_test /= len(test_loader)
                accuracy_test = float(accuracy_score(reals.tolist(), predicted.tolist()))
                f1_w_test = float(f1_score(reals.tolist(), predicted.tolist(), average='weighted'))
                prefix_results.extend(prefix_pred)

                epoch_time = time() - start_epoch_time

                # save current epoch results
                results_df.loc[len(results_df)] = [run_time, comb_string, epoch, accuracy_train, accuracy_test,
                                                   f1_w_train, f1_w_test, loss_train, loss_test, str(loss_test < best_test_loss)]

                prefix_results = pd.DataFrame(prefix_results, columns=['set', 'epoch', 'prefix_len', 'y_estimated', 'y_real'])
                prefix_results.to_csv(join(epoch_path, f'{epoch}_prefix_results.csv'), header=True, sep=',', index=False)

                # early stopping
                if loss_test < best_test_loss:
                    no_improvements = 0
                    best_test_loss = loss_test
                    print(f' ** BEST TEST LOSS {round(best_test_loss, 4)} **')
                else:


                    no_improvements += 1

                # save model
                torch.save(model.state_dict(), join(epoch_path, f'{MODEL_TYPE}_{comb_string}_{ds_name}_{epoch}_model.pt'))

                # model summary
                print(f"Epoch: {epoch}/{epochs} | Time: {round(epoch_time, 2)}s "
                      f"| Train loss: {loss_train:.4f} | Test loss: {loss_test:.4f}")

                if no_improvements > PATIENCE:
                    print(f"** EARLY STOPPING AT EPOCH: {epoch} **")
                    break

            print('Combination done!\n')
            results_df.to_csv(join(comb_path, f'results_{comb_string}.csv'), header=True, sep=',', index=False)
            actual_comb += 1
        print('All combinations done!')
        print(f'\n\nPlotting metrics... {dataset_path}    {MODEL_TYPE}    {ds_name} ')
        eval_results(dataset_path, MODEL_TYPE, ds_name)

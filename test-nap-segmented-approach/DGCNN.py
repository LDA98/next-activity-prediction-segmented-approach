from torch.nn import Module, Linear, Conv1d, ModuleList, Dropout
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, SortAggregation


class DGCNN(Module):
    def __init__(self, dataset: object, num_layers: object, dropout: object, num_neurons: object, k: object) -> None:
        """

        :rtype: object
        """
        super(DGCNN, self).__init__()
        self.k = k
        self.conv1 = SAGEConv(in_channels=dataset.num_features, out_channels=num_neurons)

        self.convs = ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(SAGEConv(in_channels=num_neurons, out_channels=num_neurons))
        self.conv1d = Conv1d(in_channels=num_neurons, out_channels=num_neurons, kernel_size=num_layers)
        self.lin1 = Linear(in_features=(num_neurons * (k - num_layers + 1)),
                           out_features=int(num_neurons/2))
        self.dropout = Dropout(p=dropout)
        self.lin2 = Linear(in_features=int(num_neurons/2), out_features=dataset.num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        input_data = self.conv1(x, edge_index)
        x = F.relu(input=input_data)

        for conv in self.convs:
            conv_input = conv(x, edge_index)
            x = F.relu(input=conv_input)
        sort_aggr = SortAggregation(k=self.k)
        x = sort_aggr(x, batch)

        x = x.view(len(x), self.k, -1).permute(dims=[0, 2, 1])
        x = F.relu(self.conv1d(x))
        x = x.view(len(x), -1)
        x = F.relu(self.lin1(x))
        x = self.dropout(x)
        x = self.lin2(x)
        return x

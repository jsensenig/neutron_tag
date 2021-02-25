from typing import List
import uproot
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import neutron_study as ns


class Net(torch.nn.Module):
    def __init__(self, num_features: int, num_classes: int, hidden_channels: int, num_layers: int):
        super(Net, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers

        # Create the conv layers
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(self.num_features, self.hidden_channels))
        for layer in range(self.num_layers-2):
            self.convs.append(GCNConv(self.hidden_channels, self.hidden_channels))
        self.convs.append(GCNConv(self.hidden_channels, self.num_classes))

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr

        for conv in self.convs:
            if conv is self.convs[-1]: # last one
                x = conv(x=x, edge_index=edge_index)
                break
            x = F.relu(conv(x=x, edge_index=edge_index))
            x = F.dropout(x, training=self.training)

        return F.log_softmax(x, dim=1)


def train(model, train_loader, optimizer):
    model.train()
    loss_all = 0
    for data in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, data.y)
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()

    return loss_all / len(train_loader.dataset)


@torch.no_grad()
def test(model, test_loader):
    model.eval()
    correct = 0
    for data in test_loader:
        pred = model(data).max(dim=1)[1]
        correct += pred.eq(data.y).sum().item() / len(data.y)

    return pred, correct / len(test_loader)


def load_data(file_name: str, tree_name: str, features: List[str], position: List[str],
              labels: List[str], df_only: bool = False):
    """ Load data from disk into train and test tensors """

    open_file = uproot.open(file_name)
    tree = open_file[tree_name]
    events = tree.arrays(position + features + labels, library="pd")

    if df_only:
        return events

    return ns.load_dataset(event_df=events, feature_list=features, pos_list=position, cluster='knn', k=4, r=1)
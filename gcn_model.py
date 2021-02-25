from typing import List
import uproot
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import neutron_study as ns


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        num_features = 4
        num_classes = 2
        self.conv1 = GCNConv(num_features, 16)
        self.conv2 = GCNConv(16, 16)
        self.conv3 = GCNConv(16, 16)
        self.conv4 = GCNConv(16, num_classes)
        # self.conv1 = ChebConv(data.num_features, 16, K=2)
        # self.conv2 = ChebConv(16, data.num_features, K=2)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x=x, edge_index=edge_index))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv2(x=x, edge_index=edge_index))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv3(x=x, edge_index=edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv4(x=x, edge_index=edge_index)
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
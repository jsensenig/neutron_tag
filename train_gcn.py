import torch
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv, ChebConv  # noqa

import gcn_model


# label_column = ["reco_all_SpHit_pdg", "reco_all_SpHit_mompdg", "reco_all_SpHit_gmompdg"]
labels = ["reco_all_SpHit_pdg", "reco_all_SpHit_mompdg", "reco_all_SpHit_gmompdg"]
position = ["reco_all_SpHit_X", "reco_all_SpHit_Y", "reco_all_SpHit_Z"]
features = ["reco_all_SpHit_sadc"]
train_file = 'data/sps_hitgmom.root'
test_file = 'data/test_sps_hitgmom.root'

# load data
train_list = gcn_model.load_data(file_name=train_file, tree_name="trkUtil/points;1", features=features, position=position, labels=labels)
test_list = gcn_model.load_data(file_name=test_file, tree_name="trkUtil/points;1", features=features, position=position, labels=labels)

train_loader = DataLoader(dataset=train_list, batch_size=5)
test_loader = DataLoader(dataset=test_list, batch_size=5)

model = gcn_model.Net()
optimizer = torch.optim.Adam([
    dict(params=model.conv1.parameters(), weight_decay=5e-4),
    dict(params=model.conv2.parameters(), weight_decay=5e-4),
    dict(params=model.conv3.parameters(), weight_decay=5e-4),
    dict(params=model.conv4.parameters(), weight_decay=0)
], lr=0.01)  # Only perform weight-decay on first convolution.

best_val_acc = test_acc = 0
for epoch in range(1, 21):
    loss = gcn_model.train(model, train_loader, optimizer)
    train_pred, train_acc = gcn_model.test(model, train_loader)
    test_pred, test_acc = gcn_model.test(model, test_loader)

    print('Epoch: {:03d}, Loss: {:.5f}, Train Acc: {:.5f}, Test Acc: {:.5f}'.
          format(epoch, loss, train_acc, test_acc))

# Save our model
torch.save(model.state_dict(), "models/model.pt")

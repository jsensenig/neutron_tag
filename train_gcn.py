import torch
import matplotlib.pyplot as plt
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv, ChebConv  # noqa

import gcn_model


# label_column = ["reco_all_SpHit_pdg", "reco_all_SpHit_mompdg", "reco_all_SpHit_gmompdg"]
labels = ["reco_all_SpHit_pdg", "reco_all_SpHit_mompdg", "reco_all_SpHit_gmompdg"]
position = ["reco_all_SpHit_X", "reco_all_SpHit_Y", "reco_all_SpHit_Z"]
features = ["reco_all_SpHit_sadc"]
train_file = 'data/sps_rad2.root'
test_file = 'data/sps_rad.root'

# load data
train_list = gcn_model.load_data(file_name=train_file, tree_name="trkUtil/points;1", features=features, position=position, labels=labels)
test_list = gcn_model.load_data(file_name=test_file, tree_name="trkUtil/points;1", features=features, position=position, labels=labels)

train_loader = DataLoader(dataset=train_list, batch_size=5)
test_loader = DataLoader(dataset=test_list, batch_size=5)

model = gcn_model.Net(num_features=4, num_classes=2, hidden_channels=32, num_layers=6)

optimizer = torch.optim.Adam([dict(params=model.convs.parameters(), weight_decay=0.01)], lr=0.01)

epoch_list = []; loss_list = []; train_acc_list = []; test_acc_list = []
best_val_acc = test_acc = 0
for epoch in range(1, 101):
    loss = gcn_model.train(model, train_loader, optimizer)
    train_pred, train_acc = gcn_model.test(model, train_loader)
    test_pred, test_acc = gcn_model.test(model, test_loader)

    print('Epoch: {:03d}, Loss: {:.5f}, Train Acc: {:.5f}, Test Acc: {:.5f}'.
          format(epoch, loss, train_acc, test_acc))

    epoch_list.append(epoch)
    if loss > 1:
        loss_list.append(1.)
    else:
        loss_list.append(loss)
    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)

# Save our model
model_name = 'models/model.pt'
print("Saving model:", model_name)
torch.save(model.state_dict(), model_name)

plt.plot(epoch_list, loss_list)
plt.plot(epoch_list, train_acc_list)
plt.plot(epoch_list, test_acc_list)
plt.xlabel("Epoch")
plt.legend(["Loss", "Train Acc.", "Test Acc."])
plt.show()

import torch
from torch_geometric.data import DataLoader
import numpy as np
import pandas as pd

import neutron_study as ns
import gcn_model

labels = ["reco_all_SpHit_pdg", "reco_all_SpHit_mompdg", "reco_all_SpHit_gmompdg"]
position = ["reco_all_SpHit_X", "reco_all_SpHit_Y", "reco_all_SpHit_Z"]
features = ["reco_all_SpHit_sadc"]
test_file = 'data/test_sps_hitgmom.root'

# load data
test_list = gcn_model.load_data(file_name=test_file, tree_name="trkUtil/points;1", features=features, position=position,
                                labels=labels)
event_df = gcn_model.load_data(file_name=test_file, tree_name="trkUtil/points;1", features=features, position=position,
                                labels=labels, df_only=True)

# Add boolean particle type column
event_df = ns._tag_ancestor(event_df, 2112)

test_loader = DataLoader(dataset=test_list, batch_size=1)
print("Loader", test_loader)

model = gcn_model.Net(num_features=4, num_classes=2, hidden_channels=64, num_layers=4)
optimizer = torch.optim.Adam([dict(params=model.convs.parameters(), weight_decay=0.01)], lr=0.01)


# Load the saved model
model.load_state_dict(torch.load("models/model.pt"))
model.eval()

pred = np.array([])
with torch.no_grad():
    # Create an array of all the predictions
    pred = np.concatenate([model.forward(data).max(dim=1)[1].numpy() for data in test_loader])

print(event_df.columns)

event_df['pred'] = pred

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 7)

print(event_df[labels + ["reco_all_SpHit_sadc", "Isneutron", "pred"]].head(500))

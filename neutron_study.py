from typing import List

import torch
from torch_geometric.data import Data
from torch_cluster import knn_graph, radius_graph

from sklearn.preprocessing import LabelEncoder
import pandas as pd


def _print_label_map(encoder: LabelEncoder):
    ''' Print map between particle PDG code and encoded label'''
    label_map = {}
    i = 0
    for code in encoder.classes_:
        label_map[i] = int(code)
        i += 1
    print("Label map:", label_map)


def _tag_ancestor(df: pd.DataFrame, pdg: int):
    '''
     Add boolean column recording if the mother or grandmother is a given particle
     Args:
          df (pd.Dataframe): Given dataframe
          pdg (int): PDG code you're looking for
     '''
    df['Isneutron'] = (df['reco_all_SpHit_mompdg'] == pdg) | (df['reco_all_SpHit_gmompdg'] == pdg)
    return df


def load_dataset(event_df: pd.DataFrame, feature_list: List[str], pos_list: List[str], cluster: str, k: int, r: float):
    # ADD test_mask=[2708], train_mask=[2708], val_mask=[2708] -- see Planetoid dataset
    """
    Args:
        event_df (panda.DataFrame): Dataframe of data to be used
        feature_list (List[str]): List of strings specifying the features (columns)
    """
    df = event_df
    graph = []

    nevents = len(df.groupby(level='entry'))
    # Truth classification
    df = _tag_ancestor(df, 2112)

    # Each event is a graph
    for evt in range(0, nevents):
        # Make a dataframe for each event
        evt_df = df.loc[evt]
        # Node features x, y, z, SADC, etc
        node_features = torch.FloatTensor(evt_df[pos_list + feature_list].values)

        # Create the edge indices
        if cluster == 'knn': # edges via KNN
            edge_index = knn_graph(x=node_features, k=k)
        elif cluster == 'radius': # edges via distance
            edge_index = radius_graph(x=node_features, r=r)
        else: # edges via simple node index shift by 1 e.g. 0<-->1
            target_nodes = evt_df.index.values[1:]  # 1...M nodes
            source_nodes = evt_df.index.values[:-1]  # 0...M-1 nodes
            edge_index = torch.LongTensor([source_nodes, target_nodes])

        position = torch.FloatTensor(evt_df[pos_list].values)
        edge_attr = position
        # Truth labels
        y = torch.LongTensor(evt_df.Isneutron.values)

        data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, y=y, pos=position)
        graph.append(data)

    return graph

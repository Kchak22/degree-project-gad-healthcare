import pickle
import pandas as pd
import torch
from torch_geometric.data import HeteroData
import numpy as np  

def load_provider_features(filepath="../../data/final_df.pickle"):
    with open(filepath, "rb") as pickle_file:
        df_provider_features = pickle.load(pickle_file)
    providers_dataset = df_provider_features.index.to_list()
    return df_provider_features, providers_dataset

def load_member_features(filepath="../../data/final_members_df.pickle"):
    with open(filepath, "rb") as pickle_file:
        df_member_features = pickle.load(pickle_file)
    members_dataset = df_member_features.index.to_list()
    return df_member_features, members_dataset

def load_claims_data(filepath="../../data/df_descriptions.pickle", members_dataset=None, providers_dataset=None):
    with open(filepath, 'rb') as pickle_file:
        df = pickle.load(pickle_file)
    df_edges = df[["providercode", "membercode", "claimcode"]]
    df_edges = df_edges.groupby(["providercode", "membercode"]).agg({"claimcode": "nunique"}).reset_index()
    if members_dataset is not None and providers_dataset is not None:
        df_edges = df_edges.loc[((df_edges.membercode.isin(members_dataset))
                                 & (df_edges.providercode.isin(providers_dataset)))]
    df_edges.rename(columns={
        "membercode": "member_id",
        "providercode": "provider_id",
        "claimcode": "nbr_claims"
    }, inplace=True)
    return df_edges

def prepare_hetero_data(df_member_features, df_provider_features, df_edges):
    """Construct a PyTorch Geometric HeteroData object from DataFrames."""
    data = HeteroData()
    
    # Add node features
    member_features = torch.tensor(df_member_features.values, dtype=torch.float)
    provider_features = torch.tensor(df_provider_features.values, dtype=torch.float)
    
    data['member'].x = member_features
    data['provider'].x = provider_features
    
    # Add edges
    member_indices = {member: i for i, member in enumerate(df_member_features.index)}
    provider_indices = {provider: i for i, provider in enumerate(df_provider_features.index)}
    
    edge_tuples = []
    edge_attr = []
    
    for _, row in df_edges.iterrows():
        m_idx = member_indices.get(row['member_id'])
        p_idx = provider_indices.get(row['provider_id'])
        
        if m_idx is not None and p_idx is not None:
            edge_tuples.append((p_idx, m_idx))
            edge_attr.append([row['nbr_claims']])
    
    if edge_tuples:
        # Convert to edge index format (2 x num_edges)
        edge_index = torch.tensor(edge_tuples, dtype=torch.long).t()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        
        data['provider', 'to', 'member'].edge_index = edge_index
        data['provider', 'to', 'member'].edge_attr = edge_attr
        
        # Add reverse edges (for convenience with some models)
        reverse_edge_index = torch.stack([edge_index[1], edge_index[0]], dim=0)
        data['member', 'to', 'provider'].edge_index = reverse_edge_index
        data['member', 'to', 'provider'].edge_attr = edge_attr
    
    return data

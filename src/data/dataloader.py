import pickle
import pandas as pd
import torch
from torch_geometric.data import HeteroData
import numpy as np  



original_member_cols = [
    'mean_claim_period', 'median_claim_period', 'min_claim_period',
    'max_claim_period', 'std_claim_period', 'mean_claim_amount',
    'median_claim_amount', 'min_claim_amount', 'max_claim_amount',
    'std_claim_amount', 'avg_provider_claim_period',
    'std_provider_claim_period', 'max_provider_claim_amount',
    'avg_provider_claim_amount', 'unique_providers', 'num_claims',
    'prop_claimtype_not_op', 'prop_claimtype_out-patient',
    'single_interaction_ratio', 'gender_f', 'gender_m',
    'principalcode_dependant', 'principalcode_principal'
]

member_rename_map = {
    # Claim Period Stats (already quite clear, maybe shorten slightly)
    'mean_claim_period': 'claim_period_mean',
    'median_claim_period': 'claim_period_median',
    'min_claim_period': 'claim_period_min',
    'max_claim_period': 'claim_period_max',
    'std_claim_period': 'claim_period_std',
    # Claim Amount Stats (already quite clear)
    'mean_claim_amount': 'claim_amount_mean',
    'median_claim_amount': 'claim_amount_median',
    'min_claim_amount': 'claim_amount_min',
    'max_claim_amount': 'claim_amount_max',
    'std_claim_amount': 'claim_amount_std',
    # Provider-related stats (from member's perspective)
    'avg_provider_claim_period': 'prov_claim_period_avg', # Shorten 'provider', use avg
    'std_provider_claim_period': 'prov_claim_period_std',
    'max_provider_claim_amount': 'prov_claim_amount_max',
    'avg_provider_claim_amount': 'prov_claim_amount_avg',
    # Other interaction stats
    'unique_providers': 'providers_unique_count', # More explicit
    'num_claims': 'claims_count', # More explicit
    'prop_claimtype_not_op': 'claim_prop_inpatient', # Assume 'not_op' = inpatient
    'prop_claimtype_out-patient': 'claim_prop_outpatient', # Remove hyphen
    'single_interaction_ratio': 'interaction_ratio_single', # Slightly reorder, keep if clear
    # Demographic flags
    'gender_f': 'is_female', # Clearer boolean flag
    'gender_m': 'is_male',
    'principalcode_dependant': 'is_dependant', # Clearer boolean flag
    'principalcode_principal': 'is_principal'
}

# --- Provider Feature Renaming ---

original_provider_cols = [
    'one_hot_claim__finalstatus_infrequent_sklearn',
    'one_hot_claim__claimtype_infrequent_sklearn',
    'one_hot_claim__loastatus_infrequent_sklearn',
    'one_hot_claim__roomtype_infrequent_sklearn',
    'one_hot_claim__principalcode_principal',
    'one_hot_claim__casetype_reimbursement', 'one_hot_claim__gender_m',
    'one_hot_provider__providertype_clinic',
    'one_hot_provider__providertype_hospital',
    'one_hot_provider__providertype_infrequent_sklearn',
    'one_hot_provider__providerlevel_level0',
    'one_hot_provider__providerlevel_level1',
    'one_hot_provider__providerlevel_level2',
    'one_hot_provider__accredited_infrequent_sklearn',
    'mean_member_claim_period', 'median_member_claim_period',
    'min_member_claim_period', 'max_member_claim_period',
    'std_member_claim_period', 'mean_claim_period', 'median_claim_period',
    'min_claim_period', 'max_claim_period', 'std_claim_period',
    'mean_member_claim_amount', 'median_member_claim_amount',
    'min_member_claim_amount', 'max_member_claim_amount',
    'std_member_claim_amount', 'mean_claim_amount', 'median_claim_amount',
    'min_claim_amount', 'max_claim_amount', 'std_claim_amount',
    'single_interaction_ratio', 'num_members', 'num_claims',
    'hdbscan_cluster_from_semantics'
]

provider_rename_map = {
    # Claim-related Flags (from provider's claims)
    'one_hot_claim__finalstatus_infrequent_sklearn': 'claim_finalstatus_is_infreq', # Remove prefix, sklearn suffix
    'one_hot_claim__claimtype_infrequent_sklearn': 'claim_type_is_infreq',
    'one_hot_claim__loastatus_infrequent_sklearn': 'claim_loa_status_is_infreq',
    'one_hot_claim__roomtype_infrequent_sklearn': 'claim_roomtype_is_infreq',
    'one_hot_claim__principalcode_principal': 'claim_member_is_principal', # Clarify it's member's status on claim
    'one_hot_claim__casetype_reimbursement': 'claim_is_reimbursement',
    'one_hot_claim__gender_m': 'claim_member_is_male', # Clarify it's member's gender on claim
    # Provider Type/Level Flags
    'one_hot_provider__providertype_clinic': 'is_clinic', # Remove prefix
    'one_hot_provider__providertype_hospital': 'is_hospital',
    'one_hot_provider__providertype_infrequent_sklearn': 'prov_type_is_infreq',
    'one_hot_provider__providerlevel_level0': 'is_level_0',
    'one_hot_provider__providerlevel_level1': 'is_level_1',
    'one_hot_provider__providerlevel_level2': 'is_level_2',
    'one_hot_provider__accredited_infrequent_sklearn': 'accreditation_is_infreq',
    # Stats about Members seen by Provider
    'mean_member_claim_period': 'member_claim_period_mean',
    'median_member_claim_period': 'member_claim_period_median',
    'min_member_claim_period': 'member_claim_period_min',
    'max_member_claim_period': 'member_claim_period_max',
    'std_member_claim_period': 'member_claim_period_std',
    'mean_member_claim_amount': 'member_claim_amount_mean',
    'median_member_claim_amount': 'member_claim_amount_median',
    'min_member_claim_amount': 'member_claim_amount_min',
    'max_member_claim_amount': 'member_claim_amount_max',
    'std_member_claim_amount': 'member_claim_amount_std',
     # Stats about Provider's own claims (overall)
    'mean_claim_period': 'provider_claim_period_mean', # Add prefix to distinguish
    'median_claim_period': 'provider_claim_period_median',
    'min_claim_period': 'provider_claim_period_min',
    'max_claim_period': 'provider_claim_period_max',
    'std_claim_period': 'provider_claim_period_std',
    'mean_claim_amount': 'provider_claim_amount_mean', # Add prefix to distinguish
    'median_claim_amount': 'provider_claim_amount_median',
    'min_claim_amount': 'provider_claim_amount_min',
    'max_claim_amount': 'provider_claim_amount_max',
    'std_claim_amount': 'provider_claim_amount_std',
    # Other Provider Stats
    'single_interaction_ratio': 'interaction_ratio_single', # Keep same as member for consistency? Check context.
    'num_members': 'members_unique_count', # More explicit
    'num_claims': 'provider_claims_count', # Distinguish from member's claims_count
    'hdbscan_cluster_from_semantics': 'semantic_cluster_id' # Simplify
}


def load_provider_features(filepath="../../data/final_df.pickle"):
    with open(filepath, "rb") as pickle_file:
        df_provider_features = pickle.load(pickle_file)
    providers_dataset = df_provider_features.index.to_list()
    df_provider_features.rename(columns=provider_rename_map, inplace=True)
    return df_provider_features, providers_dataset

def load_member_features(filepath="../../data/final_members_df.pickle"):
    with open(filepath, "rb") as pickle_file:
        df_member_features = pickle.load(pickle_file)
    members_dataset = df_member_features.index.to_list()
    df_member_features.rename(columns=member_rename_map, inplace=True)
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

def load_claims_data_with_splitting(filepath="../../data/df_descriptions.pickle", members_dataset=None, providers_dataset=None,
                                   train_split=0.6, val_split=0.2, test_split=0.2):
    with open(filepath, 'rb') as pickle_file:
        df = pickle.load(pickle_file)
    df_edges = df[["providercode", "membercode", "claimcode", "dateavailed"]]
    # Split the dataframe into training, validation and testing sets    
    df_edges = df_edges.sort_values(by='dateavailed', ascending=True)
    train_df = df_edges.iloc[:int(train_split*len(df_edges))]
    val_df = df_edges.iloc[int(train_split*len(df_edges)):int(train_split*len(df_edges)+val_split*len(df_edges))]
    test_df = df_edges.iloc[int(train_split*len(df_edges)+val_split*len(df_edges)):]
    # Now we need to identify the edges that are in the training, validation and testing sets
    train_edges = train_df.groupby(["providercode", "membercode"]).agg({"claimcode": "nunique"}).reset_index()
    val_edges = val_df.groupby(["providercode", "membercode"]).agg({"claimcode": "nunique"}).reset_index()
    test_edges = test_df.groupby(["providercode", "membercode"]).agg({"claimcode": "nunique"}).reset_index()
    # Group by providercode and membercode and count the number of claims
    df_edges = df_edges.groupby(["providercode", "membercode"]).agg({"claimcode": "nunique"}).reset_index()
    if members_dataset is not None and providers_dataset is not None:
        df_edges = df_edges.loc[((df_edges.membercode.isin(members_dataset))
                                 & (df_edges.providercode.isin(providers_dataset)))]
    df_edges.rename(columns={
        "membercode": "member_id",
        "providercode": "provider_id",
        "claimcode": "nbr_claims"
    }, inplace=True)
    train_edges.rename(columns={
        "membercode": "member_id",
        "providercode": "provider_id",
        "claimcode": "nbr_claims"
    }, inplace=True)
    val_edges.rename(columns={
        "membercode": "member_id",
        "providercode": "provider_id",
        "claimcode": "nbr_claims"
    }, inplace=True)
    test_edges.rename(columns={
        "membercode": "member_id",
        "providercode": "provider_id",
        "claimcode": "nbr_claims"
    }, inplace=True)
    return df_edges, train_edges, val_edges, test_edges

def prepare_hetero_data(df_member_features, df_provider_features, df_edges):
    """Construct a PyTorch Geometric HeteroData object from DataFrames."""
    data = HeteroData()
    
    # Identify nodes that are present in the edges
    connected_members = set(df_edges['member_id'])
    connected_providers = set(df_edges['provider_id'])
    
    # Filter features to only include connected nodes
    df_member_features_filtered = df_member_features.loc[df_member_features.index.isin(connected_members)]
    df_provider_features_filtered = df_provider_features.loc[df_provider_features.index.isin(connected_providers)]
    
    # Add node features (only for nodes that are present in the edges)
    member_features = torch.tensor(df_member_features_filtered.values, dtype=torch.float)
    provider_features = torch.tensor(df_provider_features_filtered.values, dtype=torch.float)

    data['member'].x = member_features 
    data['member'].feature_names = df_member_features_filtered.columns.tolist()
    data['provider'].x = provider_features
    data['provider'].feature_names = df_provider_features_filtered.columns.tolist()
    
    # Create new indices for the filtered nodes
    member_indices = {member: i for i, member in enumerate(df_member_features_filtered.index)}
    provider_indices = {provider: i for i, provider in enumerate(df_provider_features_filtered.index)}
    
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


def prepare_hetero_data_with_splitting(df_member_features, df_provider_features, train_edges, val_edges, test_edges):
    """Construct three distinct PyTorch Geometric HeteroData objects from DataFrames based on the splitting of the edges."""
    
    # use the prepare_hetero_data function to create the HeteroData objects
    train_data = prepare_hetero_data(df_member_features, df_provider_features, train_edges)
    val_data = prepare_hetero_data(df_member_features, df_provider_features, val_edges)
    test_data = prepare_hetero_data(df_member_features, df_provider_features, test_edges)
    
    return train_data, val_data, test_data
    
    


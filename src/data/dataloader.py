import pickle
import pandas as pd

def load_provider_features(filepath="final_df.pickle"):
    with open(filepath, "rb") as pickle_file:
        df_provider_features = pickle.load(pickle_file)
    providers_dataset = df_provider_features.index.to_list()
    return df_provider_features, providers_dataset

def load_member_features(filepath="final_members_df.pickle"):
    with open(filepath, "rb") as pickle_file:
        df_member_features = pickle.load(pickle_file)
    members_dataset = df_member_features.index.to_list()
    return df_member_features, members_dataset

def load_claims_data(filepath="df_descriptions.pickle", members_dataset=None, providers_dataset=None):
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

# Optionally, you can add a function here to construct a torch_geometric.data.HeteroData object
# from your DataFrames.

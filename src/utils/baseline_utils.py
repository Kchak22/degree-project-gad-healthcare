import torch
import numpy as np
import pandas as pd
from torch_geometric.data import HeteroData, Data
from torch_geometric.utils import degree, to_networkx, from_networkx
import networkx as nx
import torch.nn.functional as F
from typing import Dict, Tuple

def calculate_structural_features(graph: HeteroData, target_edge_type: Tuple = ('provider', 'to', 'member')) -> Dict[str, torch.Tensor]:
    """
    Calculates basic structural features (degree) for each node type.
    Placeholder for more complex features like clustering coefficient.

    Args:
        graph: The HeteroData object.
        target_edge_type: The primary edge type to consider for degrees.

    Returns:
        Dictionary {node_type: degree_tensor}.
    """
    structural_feats = {}
    src_node, _, dst_node = target_edge_type

    if target_edge_type in graph.edge_index_dict:
        edge_index = graph[target_edge_type].edge_index
        num_src = graph[src_node].num_nodes
        num_dst = graph[dst_node].num_nodes

        # Calculate out-degree for source, in-degree for destination
        deg_src = degree(edge_index[0], num_nodes=num_src)
        deg_dst = degree(edge_index[1], num_nodes=num_dst)

        structural_feats[src_node] = deg_src.float().unsqueeze(1) # Add feature dimension
        structural_feats[dst_node] = deg_dst.float().unsqueeze(1)

        # Add degrees for reverse edge type if it exists
        rev_edge_type = (dst_node, 'to', src_node)
        if rev_edge_type in graph.edge_index_dict:
             edge_index_rev = graph[rev_edge_type].edge_index
             # Calculate out-degree for dst (was src), in-degree for src (was dst)
             deg_dst_out = degree(edge_index_rev[0], num_nodes=num_dst)
             deg_src_in = degree(edge_index_rev[1], num_nodes=num_src)
             # Append these degrees
             structural_feats[dst_node] = torch.cat([structural_feats[dst_node], deg_dst_out.float().unsqueeze(1)], dim=1)
             structural_feats[src_node] = torch.cat([structural_feats[src_node], deg_src_in.float().unsqueeze(1)], dim=1)

    else: # Handle case where target edge type might not exist (e.g., in a subgraph)
        for node_type in graph.node_types:
             structural_feats[node_type] = torch.zeros((graph[node_type].num_nodes, 1),
                                                       dtype=torch.float,
                                                       device=graph[node_type].x.device if hasattr(graph[node_type], 'x') else 'cpu')


    # Placeholder for clustering coefficient (computationally more intensive)
    #try:
    #     G_nx = to_networkx(graph.to_homogeneous(), node_attrs=['x'], edge_attrs=None) # Convert carefully
     #    clustering = nx.clustering(G_nx)
    #     # Need to map clustering values back to original hetero nodes... complex mapping
    # except Exception as e:
    #     print(f"Could not compute clustering coefficient: {e}")

    return structural_feats


def augment_features_for_sklearn(graph: HeteroData, target_edge_type: Tuple = ('provider', 'to', 'member')) -> Dict[str, np.ndarray]:
    """
    Augments node features with structural features (degree) for sklearn models.

    Args:
        graph: The HeteroData object.
        target_edge_type: The primary edge type for calculating degrees.


    Returns:
        Dictionary {node_type: augmented_feature_matrix (numpy)}.
    """
    augmented_features = {}
    structural_feats = calculate_structural_features(graph, target_edge_type)
    device = next(iter(graph.x_dict.values())).device if graph.x_dict else 'cpu' # Get device from data
    print(f"Device for augmentation: {device}")
    for node_type, x in graph.x_dict.items():
        struct_f = structural_feats.get(node_type, None)
        if struct_f is not None:
             struct_f = struct_f.to(device) # Ensure structural feats are on same device
             # Ensure struct_f has same number of rows as x
             if struct_f.shape[0] != x.shape[0]:
                  print(f"Warning: Structural feature row count mismatch for {node_type}. Padding/truncating.")
                  # Simple padding/truncating (adjust logic if needed)
                  target_rows = x.shape[0]
                  current_rows = struct_f.shape[0]
                  if target_rows > current_rows:
                      padding = torch.zeros((target_rows - current_rows, struct_f.shape[1]), device=device)
                      struct_f = torch.cat([struct_f, padding], dim=0)
                  else:
                      struct_f = struct_f[:target_rows, :]

             augmented_x = torch.cat([x, struct_f], dim=1)
        else:
            augmented_x = x # No structural features to add

        augmented_features[node_type] = augmented_x.cpu().numpy()

    # Placeholder for aggregating neighbor features (more complex)

    return augmented_features

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

        # Calculate degree for source and destination nodes
        deg_src = degree(edge_index[0], num_nodes=num_src)
        deg_dst = degree(edge_index[1], num_nodes=num_dst)

        # For bipartite undirected graph, we only need to store the degree once
        structural_feats[src_node] = deg_src.float().unsqueeze(1)  # Add feature dimension
        structural_feats[dst_node] = deg_dst.float().unsqueeze(1)

        # For bipartite undirected graphs, we don't need to add reverse edge degrees
        # since they would be the same as the forward edge degrees
        # The code below is kept but commented out for reference
        
    # Calculate additional structural features for nodes

    # Convert to NetworkX for easier computation of complex metrics
    # Use the built-in to_networkx method instead of manually recreating the graph
    # G = to_networkx(graph, to_undirected=True, node_attrs=None, edge_attrs=None) # Removed: Unused and causes error for HeteroData

    # Calculate degree centrality (normalized by opposite set size)
    src_degree_centrality = torch.zeros(graph[src_node].num_nodes, dtype=torch.float,
                                        device=edge_index.device if target_edge_type in graph.edge_index_dict else 'cpu')
    dst_degree_centrality = torch.zeros(graph[dst_node].num_nodes, dtype=torch.float,
                                        device=edge_index.device if target_edge_type in graph.edge_index_dict else 'cpu')
    
    if target_edge_type in graph.edge_index_dict:
        src_degree_centrality = deg_src.float() / num_dst  # Normalize by destination set size
        dst_degree_centrality = deg_dst.float() / num_src  # Normalize by source set size
    
    # Resource Allocation Index
    src_resource_allocation = torch.zeros(graph[src_node].num_nodes, dtype=torch.float,
                                          device=edge_index.device if target_edge_type in graph.edge_index_dict else 'cpu')
    dst_resource_allocation = torch.zeros(graph[dst_node].num_nodes, dtype=torch.float,
                                          device=edge_index.device if target_edge_type in graph.edge_index_dict else 'cpu')
    
    # Exclusivity
    src_exclusivity = torch.zeros(graph[src_node].num_nodes, dtype=torch.float,
                                  device=edge_index.device if target_edge_type in graph.edge_index_dict else 'cpu')
    dst_exclusivity = torch.zeros(graph[dst_node].num_nodes, dtype=torch.float,
                                  device=edge_index.device if target_edge_type in graph.edge_index_dict else 'cpu')
    
    # Redundancy and Dispersion (more computationally intensive)
    src_redundancy = torch.zeros(graph[src_node].num_nodes, dtype=torch.float,
                                 device=edge_index.device if target_edge_type in graph.edge_index_dict else 'cpu')
    dst_redundancy = torch.zeros(graph[dst_node].num_nodes, dtype=torch.float,
                                 device=edge_index.device if target_edge_type in graph.edge_index_dict else 'cpu')
    
    src_dispersion = torch.zeros(graph[src_node].num_nodes, dtype=torch.float,
                                 device=edge_index.device if target_edge_type in graph.edge_index_dict else 'cpu')
    dst_dispersion = torch.zeros(graph[dst_node].num_nodes, dtype=torch.float,
                                 device=edge_index.device if target_edge_type in graph.edge_index_dict else 'cpu')
    
    if target_edge_type in graph.edge_index_dict:
        # Create adjacency lists for faster computation
        src_to_dst = {}
        dst_to_src = {}
        
        for i in range(edge_index.shape[1]):
            s, d = edge_index[0, i].item(), edge_index[1, i].item()
            if s not in src_to_dst:
                src_to_dst[s] = []
            if d not in dst_to_src:
                dst_to_src[d] = []
            src_to_dst[s].append(d)
            dst_to_src[d].append(s)
        
        # Resource Allocation
        for s in src_to_dst:
            neighbors = src_to_dst[s]
            ra_sum = sum(1.0 / len(dst_to_src.get(d, [])) for d in neighbors if d in dst_to_src)
            src_resource_allocation[s] = ra_sum
        
        for d in dst_to_src:
            neighbors = dst_to_src[d]
            ra_sum = sum(1.0 / len(src_to_dst.get(s, [])) for s in neighbors if s in src_to_dst)
            dst_resource_allocation[d] = ra_sum
        
        # Exclusivity
        for s in src_to_dst:
            neighbors = src_to_dst[s]
            exclusivity_sum = sum(1.0 / len(dst_to_src.get(d, [])) for d in neighbors if d in dst_to_src)
            src_exclusivity[s] = exclusivity_sum / len(neighbors) if neighbors else 0
        
        for d in dst_to_src:
            neighbors = dst_to_src[d]
            exclusivity_sum = sum(1.0 / len(src_to_dst.get(s, [])) for s in neighbors if s in src_to_dst)
            dst_exclusivity[d] = exclusivity_sum / len(neighbors) if neighbors else 0
        
        # Redundancy and Dispersion (simplified versions to maintain O(E²) complexity)
        for s in src_to_dst:
            neighbors = src_to_dst[s]
            if len(neighbors) > 1:
                # Redundancy: count pairs of neighbors that share connections
                redundancy_count = 0
                dispersion_sum = 0
                
                for i, n1 in enumerate(neighbors):
                    for n2 in neighbors[i+1:]:
                        # Check if n1 and n2 share connections other than s
                        n1_connections = set(dst_to_src.get(n1, []))
                        n2_connections = set(dst_to_src.get(n2, []))
                        common = n1_connections.intersection(n2_connections)
                        
                        # Remove the current node from common connections
                        if s in common:
                            common.remove(s)
                        
                        if common:  # They share connections
                            redundancy_count += 1
                        else:  # They don't share connections (dispersed)
                            dispersion_sum += 1
                
                total_pairs = (len(neighbors) * (len(neighbors) - 1)) / 2
                src_redundancy[s] = redundancy_count / total_pairs if total_pairs > 0 else 0
                src_dispersion[s] = dispersion_sum / total_pairs if total_pairs > 0 else 0
        
        for d in dst_to_src:
            neighbors = dst_to_src[d]
            if len(neighbors) > 1:
                # Redundancy: count pairs of neighbors that share connections
                redundancy_count = 0
                dispersion_sum = 0
                
                for i, n1 in enumerate(neighbors):
                    for n2 in neighbors[i+1:]:
                        # Check if n1 and n2 share connections other than d
                        n1_connections = set(src_to_dst.get(n1, []))
                        n2_connections = set(src_to_dst.get(n2, []))
                        common = n1_connections.intersection(n2_connections)
                        
                        # Remove the current node from common connections
                        if d in common:
                            common.remove(d)
                        
                        if common:  # They share connections
                            redundancy_count += 1
                        else:  # They don't share connections (dispersed)
                            dispersion_sum += 1
                
                total_pairs = (len(neighbors) * (len(neighbors) - 1)) / 2
                dst_redundancy[d] = redundancy_count / total_pairs if total_pairs > 0 else 0
                dst_dispersion[d] = dispersion_sum / total_pairs if total_pairs > 0 else 0
    
    # Combine all features
    src_features = torch.stack([
        structural_feats[src_node].squeeze(),
        src_degree_centrality,
        src_resource_allocation,
        src_exclusivity,
        src_redundancy,
        src_dispersion
    ], dim=1)
    
    dst_features = torch.stack([
        structural_feats[dst_node].squeeze(),
        dst_degree_centrality,
        dst_resource_allocation,
        dst_exclusivity,
        dst_redundancy,
        dst_dispersion
    ], dim=1)
    
    # Update structural features dictionary
    structural_feats[src_node] = src_features
    structural_feats[dst_node] = dst_features
    
    # Handle case where target edge type might not exist (e.g., in a subgraph)
    if target_edge_type not in graph.edge_types:
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

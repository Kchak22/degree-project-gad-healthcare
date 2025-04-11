import networkx as nx
import numpy as np
import pandas as pd
from src.utils.eval_utils import compute_evaluation_metrics
from torch_geometric.data import HeteroData
import torch
from typing import Dict, Tuple, Optional


from src.external_repos.OddBall.egonet_extractor import EgonetFeatureExtractor 
from src.external_repos.OddBall.anomaly_detection import (
    StarCliqueAnomalyDetection, HeavyVicinityAnomalyDetection, DominantEdgeAnomalyDetection,
    StarCliqueLOFAnomalyDetection, HeavyVicinityLOFAnomalyDetection, DominantEdgeLOFAnomalyDetection
)
from src.external_repos.OddBall.utils import select_group_nodes, get_node_property_list 



def heterodata_to_nx(data: HeteroData,
                     edge_type: Tuple = ('provider', 'to', 'member'),
                     weight_attr: str = 'edge_attr',
                     use_node_indices: bool = True) -> nx.Graph:
    """
    Converts a HeteroData object into a NetworkX Graph suitable for OddBall.
    Assigns 'group' attribute (0 for src, 1 for dst) to nodes.
    Ensures unique node IDs by offsetting destination node indices.

    Args:
        data: The input HeteroData object.
        edge_type: The edge type to extract ('src', 'rel', 'dst').
        weight_attr: The key for edge weights in data[edge_type]. Defaults to 'edge_attr'.
                     If edge_attr has multiple dimensions, uses the first one.
        use_node_indices: If True, node IDs in the NX graph will be the 0-based
                          indices from HeteroData (offset for dst type).

    Returns:
        A NetworkX Graph.
    """
    G = nx.Graph()
    src_node_type, _, dst_node_type = edge_type

    num_src_nodes = data[src_node_type].num_nodes 
    num_dst_nodes = data[dst_node_type].num_nodes 

    print(f"num_src_nodes: {num_src_nodes}, num_dst_nodes: {num_dst_nodes}")
    src_group_id = 0
    dst_group_id = 1
    dst_node_offset = num_src_nodes # Start destination IDs after source IDs

    # Add source nodes (e.g., providers)
    for i in range(num_src_nodes):
        node_id = i # Use original index as the ID for source nodes
        G.add_node(node_id, group=src_group_id, node_type=src_node_type, original_index=i)

    # Add destination nodes (e.g., members) with offset IDs
    for i in range(num_dst_nodes):
        node_id = i + dst_node_offset # Offset the ID
        G.add_node(node_id, group=dst_group_id, node_type=dst_node_type, original_index=i) # Store original index

    # Add edges using the potentially offset node IDs
    if edge_type in data.edge_types:
        edge_index = data[edge_type].edge_index.cpu().numpy()
        edge_weights = None
        if weight_attr and hasattr(data[edge_type], weight_attr):
            weights_tensor = getattr(data[edge_type], weight_attr)
            if weights_tensor is not None and weights_tensor.numel() > 0: # Check tensor is not empty
                # Use first dimension if multi-dimensional, otherwise use as is
                edge_weights = weights_tensor[:, 0].cpu().numpy() if weights_tensor.ndim > 1 else weights_tensor.cpu().numpy()

        if edge_weights is None:
            # Default weight to 1 if not found or empty
            edge_weights = np.ones(edge_index.shape[1])
        elif len(edge_weights) != edge_index.shape[1]:
             print(f"Warning: Edge weight length ({len(edge_weights)}) mismatch with edge index ({edge_index.shape[1]}). Using default weights=1.")
             edge_weights = np.ones(edge_index.shape[1])


        for i in range(edge_index.shape[1]):
            src_original_idx, dst_original_idx = edge_index[0, i], edge_index[1, i]
            weight = edge_weights[i]

            # Map original indices to potentially offset NetworkX IDs
            nx_src_id = src_original_idx
            nx_dst_id = dst_original_idx + dst_node_offset # Use the offset ID for destination

            # Add edge using the unique NetworkX IDs
            if G.has_node(nx_src_id) and G.has_node(nx_dst_id):
                if G.has_edge(nx_src_id, nx_dst_id):
                    G[nx_src_id][nx_dst_id]['weight'] += weight
                else:
                    G.add_edge(nx_src_id, nx_dst_id, weight=weight)
            # else: # Debugging - should not happen if nodes were added correctly
            #      print(f"Skipping edge ({nx_src_id}, {nx_dst_id}) - Nodes not found.")

    else:
        print(f"Warning: Edge type {edge_type} not found in HeteroData.")

    print(f"Converted HeteroData to nx.Graph: {G.number_of_nodes()} nodes ({num_src_nodes} {src_node_type}, {num_dst_nodes} {dst_node_type}), {G.number_of_edges()} edges.")
    return G


def run_oddball(graph: nx.Graph,
                anomaly_type: str,
                group_to_analyze: int, # 0 for provider, 1 for member
                use_lof: bool = False,
                lof_neighbors: int = 50,
                num_processes: Optional[int] = 1
               ) -> Dict[int, float]:
    """
    Runs the core OddBall anomaly detection logic on a NetworkX graph.
    Maps results back to original node indices.

    Args:
        graph: The input NetworkX graph (nodes must have 'group', 'node_type',
               and 'original_index' attributes from heterodata_to_nx).
        anomaly_type: 'sc', 'hv', or 'de'.
        group_to_analyze: The group ID (0 or 1) to calculate scores for.
        use_lof: Whether to include LOF score.
        lof_neighbors: Number of neighbors for LOF calculation.
        num_processes: Number of processes for feature extraction.

    Returns:
        A dictionary mapping original node indices to anomaly scores for the specified group.
    """
    print(f"Running OddBall: type={anomaly_type}, group={group_to_analyze}, lof={use_lof}")

    if graph.number_of_nodes() == 0:
        print("Oddball Error: Input graph has no nodes.")
        return {}

    # --- Feature Extraction (as before) ---
    extractor = EgonetFeatureExtractor()
    graph_with_features = extractor.get_feature_vector(graph, num_processes)

    # --- Select Anomaly Detection Method (as before) ---
    detector_class = None
    if not use_lof:
        if anomaly_type == "sc": detector_class = StarCliqueAnomalyDetection
        elif anomaly_type == "hv": detector_class = HeavyVicinityAnomalyDetection
        elif anomaly_type == "de": detector_class = DominantEdgeAnomalyDetection
    else:
        if anomaly_type == "sc": detector_class = StarCliqueLOFAnomalyDetection
        elif anomaly_type == "hv": detector_class = HeavyVicinityLOFAnomalyDetection
        elif anomaly_type == "de": detector_class = DominantEdgeLOFAnomalyDetection

    if detector_class is None:
        raise ValueError(f"Invalid anomaly_type: {anomaly_type}")

    print("Instantiating detector...")
    if use_lof:
        try:
            detector = detector_class(graph_with_features, n_neighbors=lof_neighbors, processes=num_processes)
        except TypeError:
            print("Warning: LOF detector might not accept n_neighbors/processes at init. Using defaults.")
            detector = detector_class(graph_with_features) # Fallback
    else:
        detector = detector_class(graph_with_features)

    # --- Run Anomaly Detection (as before) ---
    print(f"Detecting anomalies for group {group_to_analyze}...")
    try:
        # This method calculates and stores 'score' attribute in detector.graph nodes
        model, X, Y, node_ids_in_group = detector.detect_anomalies(group=group_to_analyze)
        print(f"OddBall processed {len(node_ids_in_group)} nodes in group {group_to_analyze}.")
        if len(X) == 0 and len(node_ids_in_group) > 0:
             print("Warning: Oddball processed nodes but filtering resulted in 0 nodes for power law fit. Scores might be unreliable or zero.")

    except Exception as e:
         print(f"Error during OddBall detection: {e}")
         # Attempt to recover scores if partially calculated
         partial_scores = {}
         nodes_data = detector.graph.nodes(data=True)
         for node_id, attrs in nodes_data:
             if attrs.get('group') == group_to_analyze and 'score' in attrs:
                  original_index = attrs.get('original_index')
                  if original_index is not None:
                      partial_scores[original_index] = attrs['score']
         if not partial_scores:
             print("No scores could be extracted after error.")
             num_nodes_in_group = sum(1 for n, d in graph_with_features.nodes(data=True) if d.get('group') == group_to_analyze)
             print(f"Returning zero scores for {num_nodes_in_group} nodes in group {group_to_analyze}.")
             # Return zeros based on original indices
             return {n[1]['original_index']: 0.0 for n in graph_with_features.nodes(data=True) if n[1].get('group') == group_to_analyze}
         else:
              print(f"Extracted {len(partial_scores)} scores despite error.")
              # Continue to map these partial scores back
              node_ids_in_group = list(partial_scores.keys()) # Use recovered original indices
              # Need to create a final dictionary mapping all original indices to scores or 0
              all_nodes_in_group_orig_idx = [n[1]['original_index'] for n in graph_with_features.nodes(data=True) if n[1].get('group') == group_to_analyze]
              final_scores = {orig_idx: partial_scores.get(orig_idx, 0.0) for orig_idx in all_nodes_in_group_orig_idx}
              print(f"Returning {len(final_scores)} scores for group {group_to_analyze} after error recovery.")
              return final_scores


    # --- Extract Scores and Map Back ---
    # Scores are stored as node attributes in detector.graph
    final_scores = {}
    processed_nodes_nx_ids = set(node_ids_in_group) # NetworkX IDs processed by detect_anomalies

    # Iterate through *all* nodes in the original graph belonging to the target group
    target_node_type = detector.graph.nodes[node_ids_in_group[0]]['node_type'] if node_ids_in_group else None # Get node type from first processed node

    all_nodes_of_group = [
        (nx_id, attrs) for nx_id, attrs in detector.graph.nodes(data=True)
        if attrs.get('group') == group_to_analyze
    ]


    score_found_count = 0
    for nx_id, attrs in all_nodes_of_group:
        original_index = attrs.get('original_index')
        if original_index is None:
            print(f"Warning: Node {nx_id} (group {group_to_analyze}) missing 'original_index'. Skipping.")
            continue

        # Get score if the node was processed and has a score attribute
        node_score = attrs.get('score', 0.0) # Default to 0 if score is missing

        # Check if the node ID was actually in the list returned by detect_anomalies
        # This handles cases where detect_anomalies might filter nodes before scoring
        if nx_id not in processed_nodes_nx_ids:
            node_score = 0.0 # Assign 0 if it wasn't in the final list processed by detect_anomalies

        final_scores[original_index] = node_score
        if 'score' in attrs and nx_id in processed_nodes_nx_ids :
            score_found_count+=1


    print(f"Mapped {score_found_count} non-zero scores out of {len(all_nodes_of_group)} total nodes for group {group_to_analyze}.")
    print(f"Returning final score dictionary with {len(final_scores)} entries.")
    return final_scores


def evaluate_oddball_inductive(graph_split: HeteroData,
                               gt_node_labels_split: Dict[str, torch.Tensor],
                               anomaly_type: str,
                               node_type_to_eval: str, # 'provider' or 'member'
                               use_lof: bool = False,
                               target_edge_type: Tuple = ('provider', 'to', 'member'),
                               k_list: list = [50, 100, 200]):
    """Converts graph split, runs OddBall for a specific node type, and evaluates."""
    print(f"\n--- Evaluating OddBall ({anomaly_type}, LOF={use_lof}) for Node Type: {node_type_to_eval} ---")

    # Determine the group ID for OddBall (0=provider, 1=member)
    if node_type_to_eval == 'provider':
        oddball_group = 0
    elif node_type_to_eval == 'member':
        oddball_group = 1
    else:
        raise ValueError("node_type_to_eval must be 'provider' or 'member'")
    
    # Convert HeteroData split to NetworkX Graph
    nx_graph = heterodata_to_nx(graph_split, edge_type=target_edge_type, weight_attr='edge_attr')

    if nx_graph.number_of_nodes() == 0 or nx_graph.number_of_edges() == 0:
         print(f"Skipping Oddball for {node_type_to_eval} - Graph split is empty or has no edges.")
         return {} # Return empty dict if graph is trivial

    # Run OddBall
    # The keys of score_dict should be the original PyG indices
    score_dict = run_oddball(
        graph=nx_graph,
        anomaly_type=anomaly_type,
        group_to_analyze=oddball_group,
        use_lof=use_lof,
        # num_processes=4 # Optional: Adjust processes if needed
    )

    # Prepare scores and labels for evaluation
    if not score_dict:
         print(f"OddBall returned no scores for {node_type_to_eval}.")
         return {}

    gt_labels_tensor = gt_node_labels_split.get(node_type_to_eval)
    if gt_labels_tensor is None:
        print(f"Ground truth labels not found for {node_type_to_eval}.")
        return {}

    num_nodes_in_split = graph_split[node_type_to_eval].num_nodes
    scores_array = np.zeros(num_nodes_in_split)
    processed_indices = 0
    for original_idx, score in score_dict.items():
         if original_idx < num_nodes_in_split: # Check index bounds
             scores_array[original_idx] = score
             processed_indices +=1
         # else: # This shouldn't happen if mapping is correct
         #     print(f"Warning: Score index {original_idx} out of bounds for {node_type_to_eval} (max: {num_nodes_in_split-1}).")
    
    print(f"Mapped {processed_indices} scores to array of size {num_nodes_in_split}.")

    labels_array = gt_labels_tensor.cpu().numpy()

    # Ensure lengths match before evaluation
    if len(scores_array) != len(labels_array):
         print(f"Error: Length mismatch after mapping! Scores: {len(scores_array)}, Labels: {len(labels_array)}")
         # Try resizing scores if it seems like padding issue (use with caution)
         if len(scores_array) < len(labels_array):
             scores_array = np.pad(scores_array, (0, len(labels_array) - len(scores_array)), 'constant')
         else:
             scores_array = scores_array[:len(labels_array)]
         print(f"Attempted resize. New lengths - Scores: {len(scores_array)}, Labels: {len(labels_array)}")


    # Evaluate
    print(f"Computing metrics for {node_type_to_eval}...")
    metrics = compute_evaluation_metrics(scores_array, labels_array, k_list=k_list)

    return metrics
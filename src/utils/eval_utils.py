from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
import numpy as np
import torch
from torch_geometric.data import HeteroData
import torch.nn as nn
from torch_scatter import scatter_add, scatter_mean
from src.utils.train_utils import calculate_node_anomaly_scores
import torch.nn.functional as F

@torch.no_grad()
def evaluate_model_performance(
    model: nn.Module,
    eval_data: HeteroData, # Data split (val or test) with edge_labels (y)
    train_data_struct: HeteroData, # Data providing the graph structure for message passing
    gt_node_labels: dict, # Ground truth node labels (0/1) from injection
    # gt_edge_labels: dict, # Ground truth edge labels (0/1) - Now expected in eval_data[type].y
    lambda_attr: float = 1.0,
    lambda_struct: float = 0.5,
    target_edge_types_eval: list = [('provider', 'to', 'member'), ('member', 'to', 'provider')], # Edges to eval
    k_list=[50, 100, 200]
    ):
    """
    Evaluates the trained model on both node and edge anomaly detection.

    Args:
        model: The trained model instance.
        eval_data: HeteroData for the split (val/test) containing edge labels in .y attribute.
        train_data_struct: HeteroData used for training (provides graph structure for inference).
        gt_node_labels: Dictionary of ground truth NODE anomaly labels.
        lambda_attr: Weight for attribute term in NODE anomaly score.
        lambda_struct: Weight for structural term in NODE anomaly score.
        target_edge_types_eval: List of edge types to evaluate for edge anomalies.
        k_list: List of K values for P@K, R@K (for nodes).

    Returns:
        node_results (dict): Metrics for each node type.
        edge_results (dict): Metrics for each evaluated edge type.
        node_scores (dict): Raw anomaly scores for nodes.
        edge_scores (dict): Raw anomaly logits/scores for edges.
    """
    model.eval()
    device = next(model.parameters()).device
    train_data_struct = train_data_struct.to(device)
    eval_data = eval_data.to(device) # Need eval data on device for edge labels/indices
    gt_node_labels_dev = {k: v.to(device) for k, v in gt_node_labels.items()}

    print("Running forward pass for evaluation...")
    # Forward pass on training graph structure
    x_hat_m, x_hat_p, z_m, z_p = model(train_data_struct)
    x_hat_dict = {'member': x_hat_m, 'provider': x_hat_p}
    x_dict = {'member': train_data_struct['member'].x, 'provider': train_data_struct['provider'].x}
    z_dict = {'member': z_m, 'provider': z_p}

    # --- Node Evaluation ---
    print("Calculating NODE anomaly scores...")
    # For node scoring, use edges present in the *evaluation* split's structure
    # This reflects the edges connected to nodes in that split. Use edge labels from eval_data.y
    node_score_edge_indices = {}
    node_score_edge_labels = {}
    for edge_type in eval_data.edge_types: # Use edges present in the split being evaluated
         if hasattr(eval_data[edge_type], 'edge_index') and hasattr(eval_data[edge_type], 'y'):
              node_score_edge_indices[edge_type] = eval_data[edge_type].edge_index
              node_score_edge_labels[edge_type] = eval_data[edge_type].y # Labels from injection

    node_scores = calculate_node_anomaly_scores_v2(
        x_hat_dict, x_dict, z_dict,
        node_score_edge_indices, # Edges present in the eval split
        node_score_edge_labels,  # Their corresponding labels
        model, lambda_attr, lambda_struct
    )

    print("Computing NODE metrics...")
    node_results = {}
    # Provider Node Metrics
    if 'provider' in node_scores and 'provider' in gt_node_labels_dev:
        scores_p = node_scores['provider'].cpu().numpy()
        labels_p = gt_node_labels_dev['provider'].cpu().numpy()
        node_results['provider'] = compute_evaluation_metrics(scores_p, labels_p, k_list)
    else: node_results['provider'] = {}
    # Member Node Metrics
    if 'member' in node_scores and 'member' in gt_node_labels_dev:
        scores_m = node_scores['member'].cpu().numpy()
        labels_m = gt_node_labels_dev['member'].cpu().numpy()
        node_results['member'] = compute_evaluation_metrics(scores_m, labels_m, k_list)
    else: node_results['member'] = {}

    # --- Edge Evaluation ---
    print("Calculating EDGE anomaly scores/logits...")
    edge_scores = {}
    edge_results = {}
    for edge_type in target_edge_types_eval:
        if edge_type not in eval_data.edge_types:
            print(f"  Skipping edge type {edge_type} for eval (not in data split).")
            continue
        if not hasattr(eval_data[edge_type], 'y') or not hasattr(eval_data[edge_type], 'edge_index'):
            print(f"  Skipping edge type {edge_type} for eval (missing labels 'y' or 'edge_index').")
            continue

        edge_index_eval = eval_data[edge_type].edge_index
        edge_labels_eval = eval_data[edge_type].y.cpu().numpy() # Ground truth for these edges

        if edge_index_eval.numel() == 0:
             print(f"  Skipping edge type {edge_type} (no edges).")
             continue

        # Calculate structural logits for these specific edges
        edge_logits_eval = model.decode_structure(z_dict, edge_index_eval)
        edge_scores_eval = edge_logits_eval.cpu().numpy() # Use logits as anomaly score (higher = more likely normal)
                                                         # Or use 1-sigmoid(logits) if higher score should mean anomalous
        edge_scores[edge_type] = edge_scores_eval

        print(f"  Computing EDGE metrics for {edge_type}...")
        print(f"    Edges: {len(edge_scores_eval)}, Anomalies: {int(np.sum(edge_labels_eval))}")
        # Note: compute_evaluation_metrics expects higher score = higher anomaly likelihood
        # Our logits are higher = more likely *normal*. So we negate logits for metrics.
        edge_results[edge_type] = compute_evaluation_metrics(-edge_scores_eval, edge_labels_eval, k_list=[]) # Don't need P@K for edges usually


    return node_results, edge_results, node_scores, edge_scores

@torch.no_grad()
def evaluate_model_with_scores(model, data_split, train_data_struct, gt_labels,
                               lambda_attr=1.0, lambda_struct=0.5,
                               target_edge_type=('provider', 'to', 'member'),
                               k_list=[50, 100, 200]):
    model.eval()
    device = next(model.parameters()).device
    train_data_struct = train_data_struct.to(device)
    gt_labels_dev = {k: v.to(device) for k, v in gt_labels.items()}

    x_hat_m, x_hat_p, z_m, z_p = model(train_data_struct) # Forward pass on train structure
    x_hat_dict = {'member': x_hat_m, 'provider': x_hat_p}
    x_dict = {'member': train_data_struct['member'].x, 'provider': train_data_struct['provider'].x}
    z_dict = {'member': z_m, 'provider': z_p}

    pos_edge_index_struct = train_data_struct[target_edge_type].edge_index
    anomaly_scores_dict = calculate_node_anomaly_scores(
        x_hat_dict, x_dict, z_dict,
        pos_edge_index_struct, model, lambda_attr, lambda_struct
    )

    results = {}
    # Provider eval
    if 'provider' in anomaly_scores_dict and 'provider' in gt_labels:
        scores_p = anomaly_scores_dict['provider'].cpu().numpy()
        labels_p = gt_labels_dev['provider'].cpu().numpy()
        if len(scores_p) > 0: results['provider'] = compute_evaluation_metrics(scores_p, labels_p, k_list)
        else: results['provider'] = {}
    else: results['provider'] = {}
    # Member eval
    if 'member' in anomaly_scores_dict and 'member' in gt_labels:
        scores_m = anomaly_scores_dict['member'].cpu().numpy()
        labels_m = gt_labels_dev['member'].cpu().numpy()
        if len(scores_m) > 0: results['member'] = compute_evaluation_metrics(scores_m, labels_m, k_list)
        else: results['member'] = {}
    else: results['member'] = {}

    # Return both metrics and the raw scores for plotting
    return results, anomaly_scores_dict

def compute_evaluation_metrics(scores, labels, k_list=[10, 50, 100]):
    """
    Computes AUROC, AP, Precision@K, Recall@K, and Best F1 for given scores and labels.

    Args:
        scores (np.array): Anomaly scores for nodes.
        labels (np.array): Ground truth labels (0=normal, 1=anomaly).
        k_list (list): List of K values for Precision/Recall@K.

    Returns:
        dict: Dictionary containing computed metrics.
    """
    metrics = {}

    # Ensure there are both positive and negative samples for metrics like AUC/AP
    if len(np.unique(labels)) < 2:
        print("Warning: Only one class present in labels. AUC/AP metrics cannot be computed.")
        metrics['AUROC'] = 0.0
        metrics['AP'] = 0.0
    else:
        metrics['AUROC'] = roc_auc_score(labels, scores)
        metrics['AP'] = average_precision_score(labels, scores) # AP is equivalent to AUPRC

    # Calculate Precision@K, Recall@K
    # Sort scores and labels
    desc_score_indices = np.argsort(scores, kind="mergesort")[::-1]
    scores = scores[desc_score_indices]
    labels = labels[desc_score_indices]

    num_anomalies = np.sum(labels)

    if num_anomalies == 0:
        print("Warning: No true anomalies found in labels. Recall@K and F1 will be 0.")
        for k in k_list:
             metrics[f'Precision@{k}'] = 0.0
             metrics[f'Recall@{k}'] = 0.0
        metrics['Best F1'] = 0.0
        metrics['Best F1 Threshold'] = 0.0
    else:
        for k in k_list:
            if len(scores) >= k:
                top_k_labels = labels[:k]
                num_anomalies_in_top_k = np.sum(top_k_labels)
                metrics[f'Precision@{k}'] = num_anomalies_in_top_k / k
                metrics[f'Recall@{k}'] = num_anomalies_in_top_k / num_anomalies
            else: # Handle cases where K is larger than the number of nodes
                metrics[f'Precision@{k}'] = np.sum(labels) / len(labels) if len(labels) > 0 else 0.0
                metrics[f'Recall@{k}'] = 1.0 if np.sum(labels) > 0 else 0.0


        # Calculate best F1 score
        precision, recall, thresholds = precision_recall_curve(labels, scores)
        f1_scores = 2 * recall * precision / (recall + precision + 1e-8) # Add epsilon for stability
        best_f1_idx = np.argmax(f1_scores)
        metrics['Best F1'] = f1_scores[best_f1_idx]
        # Find the threshold corresponding to the best F1 score
        # Note: thresholds array is one element shorter than precision/recall
        metrics['Best F1 Threshold'] = thresholds[min(best_f1_idx, len(thresholds)-1)]


    return metrics


def calculate_node_anomaly_scores_v2(
    x_hat_dict: dict, x_dict: dict, z_dict: dict,
    all_edge_index_dict: dict, # Dict of edge indices per type for structure eval
    all_edge_label_dict: dict, # Dict of edge labels (0/1) per type
    model: nn.Module,
    lambda_attr: float = 1.0, lambda_struct: float = 0.5,
    target_node_types = ['provider', 'member'],
    structural_aggregation: str = 'mean' # 'mean' or 'sum'
    ):
    """
    Calculates node-level anomaly scores considering node attribute reconstruction
    and the structural prediction error of connected edges (using ground truth labels).

    Args:
        x_hat_dict: Reconstructed node features.
        x_dict: Original node features.
        z_dict: Latent node embeddings.
        all_edge_index_dict: Dict {edge_type: edge_index} for edges to evaluate structurally.
        all_edge_label_dict: Dict {edge_type: edge_labels} (0/1) for the edges above.
        model: The trained model instance.
        lambda_attr: Weight for attribute loss component.
        lambda_struct: Weight for structure loss component.
        target_node_types: List of node types to calculate scores for.
        structural_aggregation: How to aggregate edge errors per node ('mean' or 'sum').

    Returns:
        Dictionary {node_type: anomaly_scores_tensor}.
    """
    anomaly_scores = {}
    device = next(model.parameters()).device

    # 1. Attribute Reconstruction Error (Node-wise MSE) - Same as before
    for node_type in target_node_types:
        if node_type in x_hat_dict and node_type in x_dict:
            x_hat = x_hat_dict[node_type]
            x = x_dict[node_type]
            # Ensure tensors are on the same device before calculation
            if x_hat.device != x.device: x = x.to(x_hat.device)

            attr_error = torch.sum((x_hat - x)**2, dim=1) # Sum MSE across features
            anomaly_scores[node_type] = lambda_attr * attr_error
        else:
            # Initialize score tensor if only structural error is computed
             num_nodes = z_dict[node_type].shape[0]
             anomaly_scores[node_type] = torch.zeros(num_nodes, device=device)


    # 2. Structural Reconstruction Error Contribution (Using All Labeled Edges)
    # Initialize structural error tensors
    struct_errors = {nt: torch.zeros(z_dict[nt].size(0), device=device) for nt in target_node_types}
    node_counts = {nt: torch.zeros(z_dict[nt].size(0), device=device) for nt in target_node_types} # For mean aggregation

    for edge_type, edge_index in all_edge_index_dict.items():
        if edge_index.numel() == 0 or edge_type not in all_edge_label_dict:
            continue # Skip if no edges or no labels for this type

        edge_labels = all_edge_label_dict[edge_type].to(device).float() # Ensure float for BCE

        # Calculate edge logits using the model's structure decoder
        edge_logits = model.decode_structure(z_dict, edge_index)

        # Calculate per-edge BCE loss (measures structural prediction error)
        edge_bce_loss = F.binary_cross_entropy_with_logits(edge_logits, edge_labels, reduction='none')

        # Aggregate this loss to the connected nodes
        src_node_type, _, dst_node_type = edge_type
        src_indices, dst_indices = edge_index

        scatter_func = scatter_mean if structural_aggregation == 'mean' else scatter_add

        if src_node_type in target_node_types:
            # Check if scatter_add or scatter_mean is available
            if 'scatter_add' in globals() or 'scatter_mean' in globals():
                struct_errors[src_node_type] = scatter_func(
                    edge_bce_loss, src_indices, dim=0,
                    out=struct_errors[src_node_type], # Use out= for in-place add/mean across types
                    dim_size=struct_errors[src_node_type].size(0)
                )
            else: # Fallback or warning if torch_scatter is missing
                 print(f"Warning: torch_scatter not found, skipping structural error aggregation for {src_node_type} via {edge_type}")

        if dst_node_type in target_node_types:
             if 'scatter_add' in globals() or 'scatter_mean' in globals():
                 struct_errors[dst_node_type] = scatter_func(
                    edge_bce_loss, dst_indices, dim=0,
                    out=struct_errors[dst_node_type],
                    dim_size=struct_errors[dst_node_type].size(0)
                 )
             else:
                  print(f"Warning: torch_scatter not found, skipping structural error aggregation for {dst_node_type} via {edge_type}")


    # Add structural component to the scores
    for node_type in target_node_types:
        anomaly_scores[node_type] += lambda_struct * struct_errors[node_type]

    return anomaly_scores
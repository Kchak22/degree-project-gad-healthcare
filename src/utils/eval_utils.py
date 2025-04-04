import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, f1_score
import numpy as np
import torch
from src.utils.train_utils import *
from torch_geometric.data import HeteroData
import torch.nn as nn
from torch_geometric.utils import to_dense_batch, to_undirected
from torch_geometric.utils import negative_sampling
from torch_geometric.utils import add_self_loops
from torch_geometric.utils import remove_self_loops

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

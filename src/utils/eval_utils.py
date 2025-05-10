from collections import defaultdict
from sklearn.metrics import PrecisionRecallDisplay, roc_auc_score, average_precision_score, precision_recall_curve, f1_score
import numpy as np
import torch
from torch_geometric.data import HeteroData
import torch.nn as nn
from torch_scatter import scatter_add, scatter_mean
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Optional, List
from src.utils.train_utils import *
import warnings # Import warnings


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




#### New function for inductive setting



# Keep this helper function as it is - it correctly compares scores and labels

def compute_evaluation_metrics(scores, labels, k_list=[10, 50, 100]):
    """
    Computes AUROC, AP, Precision@K, Recall@K, and Best F1 for given scores and labels.
    Assumes higher scores indicate higher likelihood of being anomalous.
    ADDED ROBUSTNESS CHECKS.

    Args:
        scores (np.array): Anomaly scores for nodes or edges.
        labels (np.array): Ground truth labels (0=normal, 1=anomaly).
        k_list (list): List of K values for Precision/Recall@K.

    Returns:
        dict: Dictionary containing computed metrics.
    """
    metrics = {}
    if scores is None or labels is None or not isinstance(scores, np.ndarray) or not isinstance(labels, np.ndarray):
        print(f"Warning: Invalid input type for metric computation. Scores type: {type(scores)}, Labels type: {type(labels)}. Returning default metrics.")
        metrics = { 'AUROC': 0.0, 'AP': 0.0, 'Best F1': 0.0, 'Best F1 Threshold': 0.0 }
        for k in k_list:
            metrics[f'Precision@{k}'] = 0.0
            metrics[f'Recall@{k}'] = 0.0
        return metrics

    if len(scores) != len(labels):
        print(f"Warning: Score length ({len(scores)}) != Label length ({len(labels)}). Returning default metrics.")
        # (Same default return structure as above)
        metrics = { 'AUROC': 0.0, 'AP': 0.0, 'Best F1': 0.0, 'Best F1 Threshold': 0.0 }
        for k in k_list:
            metrics[f'Precision@{k}'] = 0.0
            metrics[f'Recall@{k}'] = 0.0
        return metrics

    if len(scores) == 0:
        # print("Warning: Empty scores/labels array. Returning default metrics.")
         # (Same default return structure as above)
        metrics = { 'AUROC': 0.0, 'AP': 0.0, 'Best F1': 0.0, 'Best F1 Threshold': 0.0 }
        for k in k_list:
            metrics[f'Precision@{k}'] = 0.0
            metrics[f'Recall@{k}'] = 0.0
        return metrics


    num_anomalies = np.sum(labels)
    num_items = len(labels)

    # Ensure there are both positive and negative samples for AUC/AP/F1
    if len(np.unique(labels)) < 2:
        # print(f"Warning: Only one class ({np.unique(labels)}) present in labels. AUC/AP/F1 metrics are ill-defined.")
        metrics['AUROC'] = 0.5 # Assign chance level AUC
        metrics['AP'] = float(num_anomalies > 0) # AP is 1 if all are anomalies, 0 otherwise
        # Calculate F1 at a basic threshold (e.g., median) for reference
        try:
            median_score = np.median(scores)
            pred_labels = (scores > median_score).astype(int)
            metrics['Best F1'] = f1_score(labels, pred_labels, zero_division=0)
            metrics['Best F1 Threshold'] = median_score
        except Exception: # Catch any error during median/f1 calc
             metrics['Best F1'] = 0.0
             metrics['Best F1 Threshold'] = 0.0
    else:
        # Calculate AUROC and AP within try-except
        try:
            metrics['AUROC'] = roc_auc_score(labels, scores)
        except ValueError as e:
            print(f"  Warning: ValueError calculating AUROC: {e}. Setting AUROC to 0.0.")
            metrics['AUROC'] = 0.0
        except Exception as e:
             print(f"  Warning: Unexpected error calculating AUROC: {e}. Setting AUROC to 0.0.")
             metrics['AUROC'] = 0.0

        try:
            metrics['AP'] = average_precision_score(labels, scores)
        except ValueError as e:
            print(f"  Warning: ValueError calculating AP: {e}. Setting AP to 0.0.")
            metrics['AP'] = 0.0
        except Exception as e:
            print(f"  Warning: Unexpected error calculating AP: {e}. Setting AP to 0.0.")
            metrics['AP'] = 0.0

        # Calculate Best F1 using Precision-Recall curve
        try:
            # Use warnings context manager to suppress potential UndefinedMetricWarning if needed
            with warnings.catch_warnings():
                warnings.simplefilter("ignore") # Ignore warnings during curve calculation
                precision, recall, thresholds = precision_recall_curve(labels, scores)

            # Handle cases where precision or recall might be zero length or cause division issues
            f1_scores = np.divide(2 * recall * precision, recall + precision, out=np.zeros_like(recall), where=(recall + precision) > 0)

            if len(f1_scores) > 0:
                best_f1_idx = np.argmax(f1_scores)
                metrics['Best F1'] = f1_scores[best_f1_idx]
                # Ensure threshold index is valid
                metrics['Best F1 Threshold'] = thresholds[min(best_f1_idx, len(thresholds) - 1)] if len(thresholds) > 0 else np.median(scores)
            else:
                # Fallback if f1_scores array is empty
                metrics['Best F1'] = 0.0
                metrics['Best F1 Threshold'] = np.median(scores)

        except Exception as e:
            print(f"  Warning: Error calculating Best F1 from PR curve: {e}. Setting Best F1 to 0.0.")
            metrics['Best F1'] = 0.0
            metrics['Best F1 Threshold'] = np.median(scores) # Use median as a fallback threshold


    # Calculate Precision@K, Recall@K
    # Sort scores in descending order to get top K
    # Use stable sort if scores might have ties (mergesort is stable)
    desc_score_indices = np.argsort(scores, kind="mergesort")[::-1]
    labels_sorted = labels[desc_score_indices]

    if num_anomalies == 0:
        # print("Warning: No true anomalies found in labels. Recall@K and F1 will be 0.")
        metrics['Best F1'] = 0.0 # Ensure F1 is 0 if no anomalies
        metrics['Best F1 Threshold'] = np.max(scores) if num_items > 0 else 0.0 # Threshold places everything as normal
        for k in k_list:
            metrics[f'Precision@{k}'] = 0.0
            metrics[f'Recall@{k}'] = 0.0
    else:
        for k in k_list:
            actual_k = min(k, num_items) # Adjust k if it's larger than the number of items
            if actual_k > 0:
                top_k_labels = labels_sorted[:actual_k]
                num_anomalies_in_top_k = np.sum(top_k_labels)
                metrics[f'Precision@{k}'] = num_anomalies_in_top_k / actual_k
                metrics[f'Recall@{k}'] = num_anomalies_in_top_k / num_anomalies
            else:
                metrics[f'Precision@{k}'] = 0.0
                metrics[f'Recall@{k}'] = 0.0

    # Ensure all required keys exist with default values if calculation failed
    metrics.setdefault('AUROC', 0.0)
    metrics.setdefault('AP', 0.0)
    metrics.setdefault('Best F1', 0.0)
    metrics.setdefault('Best F1 Threshold', np.median(scores) if num_items > 0 else 0.0) # Use median as default threshold if needed
    for k in k_list:
        metrics.setdefault(f'Precision@{k}', 0.0)
        metrics.setdefault(f'Recall@{k}', 0.0)

    return metrics


# --- NEW Evaluation Function ---
def evaluate_performance_inductive(node_scores: dict, edge_scores: dict,
                                   gt_node_labels_eval: dict, gt_edge_labels_eval: dict,
                                   k_list: list = [50, 100, 200]):
    """
    Evaluates anomaly detection performance using pre-calculated scores
    against ground truth labels for an evaluation graph (Graph B).

    Args:
        node_scores: Dictionary {node_type: anomaly_scores_tensor} from calculate_anomaly_scores.
        edge_scores: Dictionary {edge_type: edge_anomaly_scores_tensor} from calculate_anomaly_scores.
                     Assumes scores are negated logits (higher = more anomalous).
        gt_node_labels_eval: Ground truth node labels (0/1) for the evaluation graph.
        gt_edge_labels_eval: Ground truth edge labels (0/1) for the evaluation graph.
        k_list: List of K values for P@K, R@K (for nodes).

    Returns:
        results (dict): A dictionary containing evaluation metrics, structured by node/edge type.
                       Example: {'nodes': {'provider': {...metrics...}, 'member': {...metrics...}},
                                 'edges': {('provider','to','member'): {...metrics...}}}
    """
    results = {'nodes': {}, 'edges': {}}
    print("--- Evaluating Performance on Scores ---")

    # Node Evaluation
    print("Evaluating Node Scores...")
    for node_type, scores_tensor in node_scores.items():
        if node_type in gt_node_labels_eval:
            scores_np = scores_tensor.cpu().numpy()
            labels_np = gt_node_labels_eval[node_type].cpu().numpy()

            if len(scores_np) != len(labels_np):
                print(f"Error: Mismatch between scores ({len(scores_np)}) and labels ({len(labels_np)}) for node type {node_type}. Skipping.")
                results['nodes'][node_type] = {}
                continue

            print(f"  Node Type: {node_type} - Items: {len(scores_np)}, Anomalies: {int(np.sum(labels_np))}")
            node_metrics = compute_evaluation_metrics(scores_np, labels_np, k_list)
            results['nodes'][node_type] = node_metrics
        else:
            print(f"Warning: Ground truth node labels not found for type {node_type}. Skipping evaluation.")
            results['nodes'][node_type] = {}

    # Edge Evaluation
    print("Evaluating Edge Scores...")
    for edge_type, scores_np in edge_scores.items():
        if edge_type in gt_edge_labels_eval:
            # Scores are already numpy arrays from calculate_anomaly_scores
            labels_np = gt_edge_labels_eval[edge_type].cpu().numpy()

            if len(scores_np) != len(labels_np):
                print(f"Error: Mismatch between scores ({len(scores_np)}) and labels ({len(labels_np)}) for edge type {edge_type}. Skipping.")
                results['edges'][edge_type] = {}
                continue

            print(f"  Edge Type: {edge_type} - Items: {len(scores_np)}, Anomalies: {int(np.sum(labels_np))}")
            # Use empty k_list for edges usually, unless P@K/R@K for edges is desired
            edge_metrics = compute_evaluation_metrics(scores_np, labels_np, k_list=[])
            results['edges'][str(edge_type)] = edge_metrics # Use string representation for dict key
        else:
            print(f"Warning: Ground truth edge labels not found for type {edge_type}. Skipping evaluation.")
            results['edges'][str(edge_type)] = {}

    print("--- Evaluation Finished ---")
    return results

# MORE EXTENSIVE EVAL FUNCTION (INDUCTIVE)



def plot_metric_comparison_bars(summary_df: pd.DataFrame, metrics_to_plot: List[str] = ['AUROC', 'AP', 'Best F1']):
    """Plots bar charts comparing key metrics across splits and element types."""
    summary_df['Element'] = summary_df['Element'].str.replace('^Edge.*', 'Edge', regex=True)    
    plot_data = summary_df.set_index(['Split', 'Element'])[metrics_to_plot].unstack('Split')
    num_metrics = len(metrics_to_plot)
    if plot_data.empty:
        print("No data to plot for metric comparison.")
        return

    fig, axes = plt.subplots(1, num_metrics, figsize=(6 * num_metrics, 5), sharey=True)
    if num_metrics == 1: axes = [axes] # Ensure axes is iterable

    for i, metric in enumerate(metrics_to_plot):
        plot_data[metric].plot(kind='bar', ax=axes[i], rot=0)
        axes[i].set_title(metric)
        axes[i].set_ylabel("Score")
        axes[i].grid(axis='y', linestyle='--')
        axes[i].legend(title="Split")
        axes[i].set_xlabel("Element Type")

    plt.suptitle("Metric Comparison Across Splits", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, PrecisionRecallDisplay
from typing import Dict, Optional, Tuple, List
import warnings # Added to handle potential single-class warnings gracefully

# Set a visually appealing style
import seaborn as sns
#sns.set_theme(style="whitegrid", palette="deep", font_scale=1.1)

def plot_pr_curves_split(
    scores_dict: Dict,
    gt_node_labels_dict: Dict, # Separate dict for NODE labels
    gt_edge_labels_dict: Dict, # Separate dict for EDGE labels
    split_name: str = 'test',
    figsize: Tuple[int, int] = (18, 5.5), # Slightly increased height
    target_edge_type: Optional[Tuple] = ('provider', 'to', 'member') # Specify edge type to plot, if any
    ):
    """
    Plots PR curves for nodes and a specific edge type for a given data split.

    Args:
        scores_dict: Nested dict {'split': {'nodes': {type: scores}, 'edges': {type: scores}}}.
                     Edge scores should be anomaly scores (higher = more anomalous).
        gt_node_labels_dict: Nested dict {'split': {type: labels}}.
        gt_edge_labels_dict: Nested dict {'split': {type_tuple: labels}}.
        split_name: The name of the split to plot (e.g., 'test', 'val').
        figsize: Figure size.
        target_edge_type: The specific edge type tuple to plot PR curve for. If None, edge plot is skipped.
    """
    if split_name not in scores_dict:
        print(f"Error: Split '{split_name}' not found in scores_dict.")
        return
    if split_name not in gt_node_labels_dict:
        print(f"Warning: Split '{split_name}' not found in gt_node_labels_dict. Node plots might be empty.")
    if split_name not in gt_edge_labels_dict and target_edge_type is not None:
         print(f"Warning: Split '{split_name}' not found in gt_edge_labels_dict. Edge plot might be empty.")

    split_scores = scores_dict[split_name]
    split_node_labels = gt_node_labels_dict.get(split_name, {}) # Use .get for safety
    split_edge_labels = gt_edge_labels_dict.get(split_name, {}) # Use .get for safety

    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True)
    fig.suptitle(f"Precision-Recall Curves ({split_name.capitalize()} Set)", fontsize=16, y=1.02)
    plot_idx = 0
    line_width = 1.8
    grid_style = {'linestyle': ':', 'alpha': 0.7}

    # --- Node PR Curves ---
    for node_type in ['provider', 'member']:
        ax = axes[plot_idx]
        node_scores = split_scores.get('nodes', {}).get(node_type)
        node_labels = split_node_labels.get(node_type) # Get from NODE label dict

        plot_title = f'{node_type.capitalize()} Nodes'
        if node_scores is not None and node_labels is not None:
            # Ensure scores are numpy arrays
            scores_np = node_scores.cpu().numpy() if hasattr(node_scores, 'cpu') else np.array(node_scores)
            labels_np = node_labels.cpu().numpy() if hasattr(node_labels, 'cpu') else np.array(node_labels)

            if len(scores_np) == len(labels_np) and len(scores_np) > 0:
                if len(np.unique(labels_np)) > 1:
                    try:
                        ap = average_precision_score(labels_np, scores_np)
                        prec, rec, _ = precision_recall_curve(labels_np, scores_np)
                        pr_display = PrecisionRecallDisplay(precision=prec, recall=rec) # Removed average_precision=ap for cleaner plot object
                        pr_display.plot(ax=ax, name=f'{node_type.capitalize()} (AP={ap:.3f})', linewidth=line_width)
                        ax.set_title(plot_title)
                    except Exception as e:
                         print(f"Error plotting PR curve for {node_type}: {e}")
                         ax.text(0.5, 0.5, 'Error plotting', ha='center', va='center', transform=ax.transAxes)
                         ax.set_title(f'{plot_title} (Error)')
                else:
                    ax.text(0.5, 0.5, 'Single class\nno PR curve', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'{plot_title} (Single Class)')
            else:
                ax.text(0.5, 0.5, 'Data mismatch\nor empty', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{plot_title} (Data Error)')
        else:
             ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
             ax.set_title(f'{plot_title} (No Data)')

        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.grid(True, **grid_style)
        ax.set_xlim([-0.05, 1.05])
        ax.set_ylim([-0.05, 1.05])
        ax.legend(loc='lower left')
        plot_idx += 1

    # --- Edge PR Curve ---
    ax = axes[plot_idx]
    plot_title_edge = 'Edges (Not Plotted)' # Default title
    if target_edge_type:
        edge_scores = split_scores.get('edges', {}).get(target_edge_type)
        # *** CORRECTED LABEL ACCESS ***
        edge_labels = split_edge_labels.get(target_edge_type) # Get from EDGE label dict
        edge_type_str = str(target_edge_type) # For display
        plot_title_edge = f'Edges: {edge_type_str}'

        if edge_scores is not None and edge_labels is not None:
            scores_np = edge_scores.cpu().numpy() if hasattr(edge_scores, 'cpu') else np.array(edge_scores)
            labels_np = edge_labels.cpu().numpy() if hasattr(edge_labels, 'cpu') else np.array(edge_labels)

            # IMPORTANT: Ensure higher score means MORE anomalous for PR curve
            # If your edge scores are negated logits (lower=more anomalous), negate them back or use 1-sigmoid(logits)
            # Assuming edge_scores ARE anomaly scores (higher = more anomalous)
            # If they are logits (higher = normal), use `-scores_np` in AP/PR calculations.

            if len(scores_np) == len(labels_np) and len(scores_np) > 0:
                if len(np.unique(labels_np)) > 1:
                    try:
                        # Use -scores_np if scores represent normality likelihood (like logits)
                        # Use scores_np if they represent anomaly likelihood (like negated logits)
                        ap = average_precision_score(labels_np, scores_np) # Assuming higher score = anomalous
                        prec, rec, _ = precision_recall_curve(labels_np, scores_np) # Assuming higher score = anomalous
                        pr_display = PrecisionRecallDisplay(precision=prec, recall=rec)
                        pr_display.plot(ax=ax, name=f'Edges (AP={ap:.3f})', linewidth=line_width)
                        ax.set_title(plot_title_edge)
                    except Exception as e:
                         print(f"Error plotting PR curve for edges {edge_type_str}: {e}")
                         ax.text(0.5, 0.5, 'Error plotting', ha='center', va='center', transform=ax.transAxes)
                         ax.set_title(f'{plot_title_edge} (Error)')

                else:
                    ax.text(0.5, 0.5, 'Single class\nno PR curve', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'{plot_title_edge} (Single Class)')
            else:
                 ax.text(0.5, 0.5, 'Data mismatch\nor empty', ha='center', va='center', transform=ax.transAxes)
                 ax.set_title(f'{plot_title_edge} (Data Error)')
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{plot_title_edge} (No Data)')
    else:
         ax.text(0.5, 0.5, 'No target edge type\nspecified', ha='center', va='center', transform=ax.transAxes)
         ax.set_title('Edges (Not Plotted)')


    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.grid(True, **grid_style)
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.legend(loc='lower left')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Dict, Optional, Tuple, List

# Set style (optional, but consistent)
sns.set_theme(style="whitegrid", palette="viridis", font_scale=1.1) # Using viridis for violin

def plot_score_distributions_split(
    scores_dict: Dict,
    gt_node_labels_dict: Dict, # Separate dict for NODE labels
    gt_edge_labels_dict: Dict, # Separate dict for EDGE labels
    split_name: str = 'test',
    figsize: Tuple[int, int] = (18, 5.5),
    target_edge_type: Optional[Tuple] = ('provider', 'to', 'member')
    ):
    """
    Plots score distributions (violin plots) for normal vs anomalous nodes and edges.

    Args:
        scores_dict: Nested dict {'split': {'nodes': {type: scores}, 'edges': {type: scores}}}.
                     Scores should be anomaly scores (higher = more anomalous).
        gt_node_labels_dict: Nested dict {'split': {type: labels}}.
        gt_edge_labels_dict: Nested dict {'split': {type_tuple: labels}}.
        split_name: The name of the split to plot (e.g., 'test', 'val').
        figsize: Figure size.
        target_edge_type: The specific edge type tuple to plot distributions for.
    """
    if split_name not in scores_dict:
        print(f"Error: Split '{split_name}' not found in scores_dict.")
        return
    if split_name not in gt_node_labels_dict:
        print(f"Warning: Split '{split_name}' not found in gt_node_labels_dict. Node plots might be empty.")
    if split_name not in gt_edge_labels_dict and target_edge_type is not None:
         print(f"Warning: Split '{split_name}' not found in gt_edge_labels_dict. Edge plot might be empty.")


    split_scores = scores_dict[split_name]
    split_node_labels = gt_node_labels_dict.get(split_name, {})
    split_edge_labels = gt_edge_labels_dict.get(split_name, {})

    elements_to_plot = []
    # Node Distributions
    for node_type in ['provider', 'member']:
        node_scores = split_scores.get('nodes', {}).get(node_type)
        node_labels = split_node_labels.get(node_type)
        if node_scores is not None and node_labels is not None:
            scores_np = node_scores.cpu().numpy() if hasattr(node_scores, 'cpu') else np.array(node_scores)
            labels_np = node_labels.cpu().numpy() if hasattr(node_labels, 'cpu') else np.array(node_labels)
            if len(scores_np) == len(labels_np) and len(scores_np) > 0:
                elements_to_plot.append({
                    'name': f'{node_type.capitalize()} Nodes',
                    'scores': scores_np,
                    'labels': labels_np
                })

    # Edge Distribution
    if target_edge_type:
        edge_scores = split_scores.get('edges', {}).get(target_edge_type)
        # *** CORRECTED LABEL ACCESS ***
        edge_labels = split_edge_labels.get(target_edge_type)
        edge_type_str = str(target_edge_type)

        if edge_scores is not None and edge_labels is not None:
            scores_np = edge_scores.cpu().numpy() if hasattr(edge_scores, 'cpu') else np.array(edge_scores)
            labels_np = edge_labels.cpu().numpy() if hasattr(edge_labels, 'cpu') else np.array(edge_labels)
            # Assuming higher scores = more anomalous (like negated logits)
            if len(scores_np) == len(labels_np) and len(scores_np) > 0:
                 elements_to_plot.append({
                    'name': f'Edges',
                    'scores': scores_np,
                    'labels': labels_np
                 })

    num_plots = len(elements_to_plot)
    if num_plots == 0:
        print("No valid data found to plot distributions.")
        return

    fig, axes = plt.subplots(1, num_plots, figsize=figsize) # Adjust columns based on actual data
    if num_plots == 1: axes = [axes] # Ensure iterable
    fig.suptitle(f"Anomaly Score Distributions ({split_name.capitalize()} Set)", fontsize=16, y=1.02)
    grid_style = {'linestyle': ':', 'alpha': 0.7}

    # Create plots
    for i, elem_data in enumerate(elements_to_plot):
         ax = axes[i]
         # Filter out potential NaNs which cause violinplot errors
         df_plot = pd.DataFrame({'Score': elem_data['scores'], 'Label': elem_data['labels']}).dropna()

         if not df_plot.empty and df_plot['Label'].nunique() > 0:
             df_plot['Status'] = df_plot['Label'].map({0: 'Normal', 1: 'Anomaly'})
             plot_order = [status for status in ['Normal', 'Anomaly'] if status in df_plot['Status'].unique()] # Ensure correct order

             if len(plot_order) > 0: # Check if there's actually data to plot after filtering
                 try:
                     sns.violinplot(data=df_plot, x='Status', y='Score', ax=ax, order=plot_order, cut=0, inner='quartile', linewidth=1.5) # Added cut=0, inner='quartile'
                     ax.set_title(elem_data['name'])
                     ax.grid(axis='y', **grid_style)
                 except Exception as e:
                     print(f"Error during violin plot for {elem_data['name']}: {e}")
                     ax.text(0.5, 0.5, 'Error plotting', ha='center', va='center', transform=ax.transAxes)
                     ax.set_title(f'{elem_data["name"]} (Plot Error)')

             else: # Data only had NaNs or single class after dropna
                 ax.text(0.5, 0.5, 'Single class or NaN data', ha='center', va='center', transform=ax.transAxes)
                 ax.set_title(f'{elem_data["name"]} (Invalid Data)')
         else:
             ax.text(0.5, 0.5, 'No data or\nsingle class', ha='center', va='center', transform=ax.transAxes)
             ax.set_title(f'{elem_data["name"]} (No Data)')
         ax.set_xlabel("Status")


    # Hide unused axes if fewer than 3 elements plotted originally requested 3 axes
    # for j in range(num_plots, 3):
    #      if j < len(axes): # Check if axis exists before trying to hide
    #           axes[j].set_visible(False)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()



def plot_all_score_distributions(
    scores_dict: Dict[str, Dict], # keys: split_name, values: {'nodes': {type: scores}, 'edges': {type: scores}}
    gt_node_labels_dict: Dict[str, Dict], # keys: split_name, values: {type: labels}
    gt_edge_labels_dict: Dict[str, Dict], # keys: split_name, values: {type_tuple: labels}
    figsize_per_plot: Tuple[int, int] = (6, 5), # Size for EACH individual subplot
    target_edge_type: Optional[Tuple] = ('provider', 'to', 'member')
    ):
    """
    Plots score distributions (violin plots) for normal vs anomalous nodes and edges
    across all available data splits (train, val, test).

    Args:
        scores_dict: Nested dict {'split': {'nodes': {type: scores}, 'edges': {type: scores}}}.
                     Scores should be anomaly scores (higher = more anomalous).
        gt_node_labels_dict: Nested dict {'split': {type: labels}}.
        gt_edge_labels_dict: Nested dict {'split': {type_tuple: labels}}.
        figsize_per_plot: Approximate figure size allocated per subplot (width, height).
                          The total figure size will be calculated based on this.
        target_edge_type: The specific edge type tuple to plot distributions for.
                          If None, edge column will be skipped.
    """
    node_types = ['provider', 'member']
    element_types_to_plot = node_types + ([str(target_edge_type)] if target_edge_type else []) # Use string for lookup later
    element_names = ['Provider Nodes', 'Member Nodes'] + (['Edges'] if target_edge_type else [])

    splits_to_plot = sorted([split for split in scores_dict.keys() if split in gt_node_labels_dict or split in gt_edge_labels_dict])

    if not splits_to_plot:
        print("Error: No valid splits found in scores_dict matching label dicts.")
        return

    num_splits = len(splits_to_plot)
    num_elements = len(element_types_to_plot)

    if num_elements == 0:
        print("Error: No element types specified for plotting (check node_types and target_edge_type).")
        return

    # Calculate total figure size
    total_figsize = (figsize_per_plot[0] * num_elements, figsize_per_plot[1] * num_splits)

    fig, axes = plt.subplots(num_splits, num_elements, figsize=total_figsize, squeeze=False) # Ensure axes is always 2D
    fig.suptitle("Anomaly Score Distributions Across Splits", fontsize=18, y=1.03) # Adjust y if needed

    grid_style = {'linestyle': ':', 'alpha': 0.7}

    for i, split_name in enumerate(splits_to_plot):
        split_scores = scores_dict.get(split_name, {})
        split_node_labels = gt_node_labels_dict.get(split_name, {})
        split_edge_labels = gt_edge_labels_dict.get(split_name, {})

        # --- Data Preparation for the current split ---
        plot_data_for_split = []

        # 1. Node Data
        for node_type in node_types:
            scores = split_scores.get('nodes', {}).get(node_type)
            labels = split_node_labels.get(node_type)
            plot_data_for_split.append({'type': node_type, 'scores': scores, 'labels': labels})

        # 2. Edge Data
        if target_edge_type:
            scores = split_scores.get('edges', {}).get(target_edge_type)
            labels = split_edge_labels.get(target_edge_type)
            plot_data_for_split.append({'type': str(target_edge_type), 'scores': scores, 'labels': labels})


        # --- Plotting for the current split ---
        for j, elem_key in enumerate(element_types_to_plot): # Iterate through columns (provider, member, edge)
            ax = axes[i, j] # Select the correct subplot [row, column]
            data = next((item for item in plot_data_for_split if item['type'] == elem_key), None)
            element_name = element_names[j] # Get the display name

            if data and data['scores'] is not None and data['labels'] is not None:
                scores_np = data['scores'].cpu().numpy() if hasattr(data['scores'], 'cpu') else np.array(data['scores'])
                labels_np = data['labels'].cpu().numpy() if hasattr(data['labels'], 'cpu') else np.array(data['labels'])

                if len(scores_np) == len(labels_np) and len(scores_np) > 0:
                    df_plot = pd.DataFrame({'Score': scores_np, 'Label': labels_np}).dropna()

                    if not df_plot.empty and df_plot['Label'].nunique() > 0:
                        df_plot['Status'] = df_plot['Label'].map({0: 'Normal', 1: 'Anomaly'})
                        plot_order = [status for status in ['Normal', 'Anomaly'] if status in df_plot['Status'].unique()]

                        if len(plot_order) > 0:
                            try:
                                sns.violinplot(data=df_plot, x='Status', y='Score', ax=ax, order=plot_order, cut=0, inner='quartile', linewidth=1.5)
                                ax.grid(axis='y', **grid_style)
                            except Exception as e:
                                print(f"Error during violin plot for {element_name} ({split_name}): {e}")
                                ax.text(0.5, 0.5, 'Error plotting', ha='center', va='center', transform=ax.transAxes)
                                ax.set_title(f'{element_name}\n({split_name.capitalize()}) (Plot Error)') # Add split name to title
                        else:
                            ax.text(0.5, 0.5, 'Single class or NaN data', ha='center', va='center', transform=ax.transAxes)
                            ax.set_title(f'{element_name}\n({split_name.capitalize()}) (Invalid Data)')
                    else:
                        ax.text(0.5, 0.5, 'No data or\nsingle class', ha='center', va='center', transform=ax.transAxes)
                        ax.set_title(f'{element_name}\n({split_name.capitalize()}) (No Data)')
                else:
                     ax.text(0.5, 0.5, 'Empty data or\nmismatched lengths', ha='center', va='center', transform=ax.transAxes)
                     ax.set_title(f'{element_name}\n({split_name.capitalize()}) (Input Error)')

            else: # Data was missing for this element/split combination
                ax.text(0.5, 0.5, 'Data not found', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{element_name}\n({split_name.capitalize()}) (Missing)')

            # Set titles and labels clearly
            if not ax.get_title(): # Set title only if not set by error handling
                 ax.set_title(f'{element_name}\n({split_name.capitalize()})')

            # Clean up labels for clarity (optional - remove inner labels if desired)
            if i < num_splits - 1: # Not the last row
                ax.set_xlabel("")
                ax.set_xticklabels([])
            else:
                 ax.set_xlabel("Status")

            if j > 0: # Not the first column
                ax.set_ylabel("")
            else:
                ax.set_ylabel("Score")


    plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust rect based on suptitle position
    plt.show()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Dict, Tuple, Optional, List
import os

# Set a visually appealing style
sns.set_theme(style="whitegrid", palette="deep", font_scale=1.1)

def plot_anomaly_type_metrics_sorted(
    anomaly_type_df: pd.DataFrame,
    split_name: str = 'test',
    metrics: List[str] = ['Mean Score', 'AUROC', 'AP', 'Best F1'],
    sort_by_metric: str = 'AP', # Metric to sort tags by
    sort_ascending: bool = True, # True: Hardest first (low AP/AUC), False: Easiest first (high Mean Score)
    figsize: Tuple[int, int] = (14, 6), # Adjusted default size
    save_path: Optional[str] = None
):
    """
    Plots metrics broken down by anomaly type for a specific split,
    sorting the anomaly tags based on a chosen performance metric.

    Args:
        anomaly_type_df (pd.DataFrame): DataFrame containing per-tag metrics.
                                        Must include 'Split', 'Node Type', 'Anomaly Tag',
                                        and the specified metrics.
        split_name (str): The split to plot (e.g., 'test').
        metrics (List[str]): List of metric columns to plot.
        sort_by_metric (str): Metric name used to sort the anomaly tags on the plot.
        sort_ascending (bool): Sort order for the tags based on the metric.
                               True (default for AP/AUC/F1) puts worst-performing first.
                               False (default for Mean Score) puts highest-scoring first.
        figsize (Tuple[int, int]): Figure size.
        save_path (Optional[str]): If provided, saves the figure to this path.
    """
    if anomaly_type_df.empty:
        print("No data provided for anomaly type plot.")
        return

    df_split = anomaly_type_df[anomaly_type_df['Split'] == split_name].copy()

    if df_split.empty:
        print(f"No anomaly type data found for split '{split_name}'.")
        return

    # --- Sort Anomaly Tags ---
    if sort_by_metric in df_split.columns:
        # Calculate the average of the sorting metric across node types for each tag
        # Use mean, ignore NaN values if a tag doesn't apply to all node types
        tag_sort_metric = df_split.groupby('Anomaly Tag')[sort_by_metric].mean(numeric_only=True)

        # Sort the tags based on the aggregated metric
        sorted_tags = tag_sort_metric.sort_values(ascending=sort_ascending).index.tolist()
        print(f"Sorting anomaly tags by '{sort_by_metric}' (ascending={sort_ascending}).")
    else:
        print(f"Warning: Metric '{sort_by_metric}' not found for sorting. Using alphabetical tag order.")
        sorted_tags = sorted(df_split['Anomaly Tag'].dropna().unique())

    # --- Plotting ---
    num_metrics = len(metrics)
    # Adjust height based on number of metrics
    fig, axes = plt.subplots(num_metrics, 1, figsize=(figsize[0], figsize[1] * num_metrics / 2), sharex=True)
    if num_metrics == 1: axes = [axes]

    grid_style = {'linestyle': ':', 'alpha': 0.7}

    for i, metric in enumerate(metrics):
        ax = axes[i]
        if metric not in df_split.columns:
             ax.set_title(f"{metric} per Anomaly Type ({split_name.capitalize()}) - Metric Not Found")
             ax.text(0.5, 0.5, f"Metric '{metric}'\nNot Found", ha='center', va='center', transform=ax.transAxes)
             continue

        # Pivot data: Anomaly Tag as index, Node Type as columns
        try:
            pivot_df = pd.pivot_table(df_split, index='Anomaly Tag', columns='Node Type', values=metric, aggfunc='mean')
            # Reindex pivot table rows based on the calculated sort order
            pivot_df = pivot_df.reindex(sorted_tags).dropna(how='all')
        except KeyError:
             ax.set_title(f"{metric} per Anomaly Type ({split_name.capitalize()}) - Pivot Error")
             ax.text(0.5, 0.5, f"Column 'Node Type' missing\nor error during pivot.", ha='center', va='center', transform=ax.transAxes)
             continue
        except Exception as e:
            print(f"Error pivoting/reindexing data for metric '{metric}': {e}")
            ax.set_title(f"{metric} per Anomaly Type ({split_name.capitalize()}) - Data Error")
            ax.text(0.5, 0.5, "Error creating plot data", ha='center', va='center', transform=ax.transAxes)
            continue

        if pivot_df.empty:
            ax.set_title(f"{metric} per Anomaly Type ({split_name.capitalize()}) - No Data")
            ax.text(0.5, 0.5, "No data to plot", ha='center', va='center', transform=ax.transAxes)
            continue

        # Plotting grouped bars
        pivot_df.plot(kind='bar', ax=ax, width=0.8, rot=45) # Use default palette, group by Node Type

        ax.set_title(f"{metric} per Anomaly Type ({split_name.capitalize()})")
        ax.set_ylabel(metric)
        ax.grid(axis='y', **grid_style)
        ax.tick_params(axis='x', labelsize=10)
        # Adjust x-tick label alignment manually for rotated labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')


        ax.legend(title="Node Type", bbox_to_anchor=(1.01, 1), loc='upper left', fontsize='small')

        if i == num_metrics - 1:
            sort_order_text = "Ascending (Worst First)" if sort_ascending else "Descending (Best First)"
            if metric == "Mean Score" and not sort_ascending: # Adjust text for Mean Score default
                sort_order_text = "Descending (Highest Score First)"
            elif metric != "Mean Score" and sort_ascending:
                 sort_order_text = "Ascending (Lowest Score First)"

            ax.set_xlabel(f"Anomaly Tag (Sorted by Avg {sort_by_metric}, {sort_order_text})")
        else:
            ax.set_xlabel("") # Remove x-label from upper plots

    fig.suptitle(f"Anomaly Type Performance ({split_name.capitalize()} Set)", fontsize=16, y=1.02) # Adjust y slightly
    plt.tight_layout(rect=[0, 0.02, 0.88, 0.97]) # Adjust right margin more for potentially longer legend

    if save_path:
        try:
            save_dir = os.path.dirname(save_path)
            if save_dir: os.makedirs(save_dir, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Sorted anomaly type plot saved to {save_path}")
        except Exception as e:
            print(f"Error saving sorted anomaly type plot: {e}")

    plt.show()

def plot_grouped_score_distributions(
    scores_dict: Dict[str, Dict], # keys: split_name ('train', 'val', 'test')
    gt_node_labels_dict: Dict[str, Dict],
    gt_edge_labels_dict: Dict[str, Dict],
    figsize: Tuple[int, int] = (18, 6), # Figure size for the whole row of plots
    target_edge_type: Optional[Tuple] = ('provider', 'to', 'member')
    ):
    """
    Plots score distributions (violin plots) comparing splits (train, val, test)
    for each element type (provider nodes, member nodes, edges) side-by-side.

    Args:
        scores_dict: Nested dict {'split': {'nodes': {type: scores}, 'edges': {type: scores}}}.
        gt_node_labels_dict: Nested dict {'split': {type: labels}}.
        gt_edge_labels_dict: Nested dict {'split': {type_tuple: labels}}.
        figsize: Figure size for the entire figure containing the subplots.
        target_edge_type: The specific edge type tuple to plot distributions for.
    """
    node_types = ['provider', 'member']
    # Store the actual keys used in dictionaries
    element_keys_to_plot = node_types + ([target_edge_type] if target_edge_type else [])
    # User-friendly names for titles
    element_names = ['Provider Nodes', 'Member Nodes'] + (['Edges'] if target_edge_type else [])
    splits = ['train', 'val', 'test'] # Define the order for plotting and legend

    num_elements = len(element_keys_to_plot)
    if num_elements == 0:
        print("Error: No element types specified for plotting.")
        return

    fig, axes = plt.subplots(1, num_elements, figsize=figsize, squeeze=False)
    axes = axes.flatten() # Ensure axes is always a 1D array

    fig.suptitle("Anomaly Score Distributions by Split", fontsize=16, y=1.02)
    grid_style = {'linestyle': ':', 'alpha': 0.7}

    plot_data_found = False # Flag to check if any data was plotted

    for j, elem_key in enumerate(element_keys_to_plot):
        ax = axes[j]
        elem_name = element_names[j]

        # --- Aggregate data across splits for this element type ---
        all_scores = []
        all_labels = []
        all_split_names = []

        is_edge = isinstance(elem_key, tuple) # Differentiate edges (tuple) from nodes (str)

        for split_name in splits:
            scores = None
            labels = None
            # Safely get scores and labels
            if is_edge:
                scores = scores_dict.get(split_name, {}).get('edges', {}).get(elem_key)
                labels = gt_edge_labels_dict.get(split_name, {}).get(elem_key)
            else: # Node type
                scores = scores_dict.get(split_name, {}).get('nodes', {}).get(elem_key)
                labels = gt_node_labels_dict.get(split_name, {}).get(elem_key)

            if scores is not None and labels is not None:
                scores_np = scores.cpu().numpy() if hasattr(scores, 'cpu') else np.array(scores)
                labels_np = labels.cpu().numpy() if hasattr(labels, 'cpu') else np.array(labels)

                if len(scores_np) == len(labels_np) and len(scores_np) > 0:
                    all_scores.extend(scores_np)
                    all_labels.extend(labels_np)
                    all_split_names.extend([split_name] * len(scores_np))
                    plot_data_found = True # Mark that we found some data
                else:
                    print(f"Warning: Mismatched lengths or zero length for {elem_name} in split {split_name}")
            else:
                 print(f"Warning: Missing data for {elem_name} in split {split_name}")


        # --- Create DataFrame for the current element type ---
        if not all_scores: # No data found for this element type at all
            ax.text(0.5, 0.5, 'No data available\nfor any split', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{elem_name}\n(No Data)')
            continue # Skip to the next element plot

        df_element = pd.DataFrame({
            'Score': all_scores,
            'Label': all_labels,
            'Split': all_split_names # Use this column for hue
        })
        df_element = df_element.dropna() # Drop rows where score might be NaN etc.
        df_element['Status'] = df_element['Label'].map({0: 'Normal', 1: 'Anomaly'})

        # --- Check if data is still valid after processing ---
        if df_element.empty or df_element['Status'].nunique() < 1 or df_element['Split'].nunique() < 1:
             ax.text(0.5, 0.5, 'Insufficient valid data\nafter processing', ha='center', va='center', transform=ax.transAxes)
             ax.set_title(f'{elem_name}\n(Invalid Data)')
             continue


        # --- Plot using hue for splits ---
        try:
            sns.violinplot(
                data=df_element,
                x='Status',        # Normal vs Anomaly
                y='Score',
                hue='Split',       # Color by train/val/test
                ax=ax,
                order=['Normal', 'Anomaly'], # Ensure consistent x-axis order
                hue_order=splits,            # Ensure consistent legend/color order
                split=True,        # Show hues side-by-side within each status
                cut=0,
                inner='quartile',
                linewidth=1.2 # Slightly reduced linewidth for split violins
            )
            ax.set_title(elem_name)
            ax.grid(axis='y', **grid_style)
            ax.set_xlabel("Status") # Label x-axis on each plot
            # Remove individual legends if we add a figure legend later
            if ax.get_legend():
                 ax.get_legend().remove()

        except Exception as e:
            print(f"Error during violin plot for {elem_name}: {e}")
            ax.text(0.5, 0.5, 'Error plotting', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{elem_name}\n(Plot Error)')


    # --- Add a single legend for the whole figure ---
    if plot_data_found: # Only add legend if some data was actually plotted
        # Get handles and labels from one of the axes that might have them
        handles, labels = None, None
        for ax in axes:
             h, l = ax.get_legend_handles_labels()
             if h: # Found an axis with legend info
                 handles, labels = h, l
                 break # Use the first one found
        if handles:
            # Place legend outside the plot area to the right
            fig.legend(handles, labels, title='Split', bbox_to_anchor=(1.01, 0.5), loc='center left')


    # Adjust layout - might need fine-tuning depending on legend size
    plt.tight_layout(rect=[0, 0.03, 0.98 if plot_data_found and handles else 1, 0.95]) # Leave space for legend if present
    plt.show()

def evaluate_model_inductively(
    trained_model: nn.Module,
    train_graph: HeteroData,
    val_graph: HeteroData,
    test_graph: HeteroData,
    gt_node_labels: Dict[str, Dict[str, torch.Tensor]],
    gt_edge_labels: Dict[str, Dict[Tuple, torch.Tensor]],
    # --- NEW: Anomaly Tracking Dict ---
    anomaly_tracking_all: Dict[str, Dict[str, Dict[int, List[str]]]],
    # --------------------------------
    device: str = 'mps',
    eval_params: Optional[Dict] = None,
    target_edge_type: Tuple = ('provider', 'to', 'member'),
    plot: bool = True,
    verbose: bool = True
    ) -> Tuple[Dict[str, Dict], pd.DataFrame, pd.DataFrame]: # Added 3rd return value
    """
    Evaluates a pre-trained model on train, validation, and test graph splits.
    Calculates self-supervised anomaly scores, computes metrics (overall and per anomaly type),
    and optionally plots results.

    Args:
        trained_model: The pre-trained GNN model instance.
        train_graph: HeteroData object for the training split.
        val_graph: HeteroData object for the validation split.
        test_graph: HeteroData object for the testing split.
        gt_node_labels: Dict mapping split ('train', 'val', 'test') to node label dicts.
        gt_edge_labels: Dict mapping split ('train', 'val', 'test') to edge label dicts.
        device: Device to run evaluation on.
        eval_params (Optional[Dict]): Dictionary containing parameters for evaluation:
            'k_list' (List[int]): K values for P@K, R@K.
            'lambda_attr' (float): Weight for attribute score term.
            'lambda_struct' (float): Weight for structural score term.
            'k_neg_samples_score' (int): k for negative sampling in structural score calc.
        target_edge_type (Tuple): Primary edge type for structural scoring.
        plot (bool): Whether to generate and display plots.
        verbose (bool): Whether to print progress messages
        anomaly_tracking_all: Dict mapping split ('train', 'val', 'test') to node anomaly tracking dicts..

    Returns:
        all_scores (Dict): {'train': score_dict, 'val': score_dict, 'test': score_dict}
        summary_df (pd.DataFrame): DataFrame containing OVERALL metrics for all splits and types.
        anomaly_type_df (pd.DataFrame): DataFrame containing metrics broken down by ANOMALY TYPE.
    """
    def log(message):
        if verbose:
            print(message)

    log("--- Starting Inductive Model Evaluation (with Anomaly Type Analysis) ---")

    # Default eval params if not provided
    if eval_params is None:
        eval_params = {
            'k_list': [50, 100, 200],
            'lambda_attr': 1.0,
            'lambda_struct': 0.5,
            'k_neg_samples_score': 1
        }
    k_list = eval_params.get('k_list', [50, 100, 200])

    trained_model.to(device) # Ensure model is on correct device
    graphs = {'train': train_graph.to(device), 'val': val_graph.to(device), 'test': test_graph.to(device)}

    # --- 1. Scoring ---
    log("\n--- Scoring Phase ---")
    all_scores = {}
    for split_name, graph_data in graphs.items():
        log(f"Calculating scores for {split_name} split...")
        # (Score calculation logic remains the same)
        from src.utils.train_utils import calculate_anomaly_scores
        node_scores, edge_scores = calculate_anomaly_scores(
            trained_model=trained_model,
            eval_graph_data=graph_data,
            lambda_attr=eval_params['lambda_attr'],
            lambda_struct=eval_params['lambda_struct'],
            target_edge_type=target_edge_type,
            k_neg_samples_struct_score=eval_params.get('k_neg_samples_score', 1)
        )
        all_scores[split_name] = {'nodes': node_scores, 'edges': edge_scores}
    log("--- Scoring Complete ---")

    # --- 2. Evaluation & Anomaly Type Analysis ---
    log("\n--- Evaluation Phase ---")
    all_metrics = {}
    per_tag_results = [] # Store detailed results per anomaly tag

    for split_name in ['train', 'val', 'test']:
        log(f"\nEvaluating performance for {split_name} split...")
        split_results = {'nodes': {}, 'edges': {}}
        current_graph = graphs[split_name]
        current_node_scores = all_scores[split_name]['nodes']
        current_edge_scores = all_scores[split_name]['edges']
        current_gt_nodes = gt_node_labels.get(split_name, {})
        current_gt_edges = gt_edge_labels.get(split_name, {})
        current_tracking = anomaly_tracking_all.get(split_name, {})
        #print("Debug", current_tracking)

        # --- Overall Node Evaluation ---
        for node_type, scores_tensor in current_node_scores.items():
            if node_type in current_gt_nodes:
                 scores_np = scores_tensor.cpu().numpy()
                 labels_np = current_gt_nodes[node_type].cpu().numpy()
                 # Calculate overall metrics for this node type
                 node_metrics = compute_evaluation_metrics(scores_np, labels_np, k_list)
                 split_results['nodes'][node_type] = node_metrics
                 log(f"  Overall Node Metrics ({split_name}, {node_type}): AUROC={node_metrics.get('AUROC', 0):.3f}, AP={node_metrics.get('AP', 0):.3f}")

                 # --- Per Anomaly Type Node Analysis ---
                 node_current_tracking = current_tracking.get('node', {})
                 node_type_tracking = node_current_tracking.get(node_type, {})
                 #print("Debug 2", node_type_tracking)
                 if node_type_tracking: # Proceed only if tracking info exists
                     anomalous_indices = np.where(labels_np == 1)[0]
                     scores_by_tag = defaultdict(list)
                     indices_by_tag = defaultdict(list)

                     # Group scores and indices by tag for anomalous nodes
                     for idx in anomalous_indices:
                          tags = node_type_tracking.get(int(idx), ['unknown']) # Get tags for this node idx
                          for tag in tags:
                              scores_by_tag[tag].append(scores_np[idx])
                              indices_by_tag[tag].append(idx)

                     log(f"    Analyzing {len(scores_by_tag)} unique anomaly tags for {node_type}...")
                     # Calculate metrics per tag
                     for tag, tag_scores_list in scores_by_tag.items():
                          tag_indices = indices_by_tag[tag]
                          count = len(tag_scores_list)
                          mean_score = np.mean(tag_scores_list) if count > 0 else 0
                          median_score = np.median(tag_scores_list) if count > 0 else 0

                          # Create temporary labels: 1 for nodes with this tag, 0 for normals
                          temp_labels = np.zeros_like(labels_np)
                          temp_labels[tag_indices] = 1 # Mark nodes with this tag as positive

                          # Combine scores/labels for nodes with this tag and normal nodes
                          mask_tag_or_normal = np.logical_or(temp_labels == 1, labels_np == 0)
                          scores_subset = scores_np[mask_tag_or_normal]
                          labels_subset = temp_labels[mask_tag_or_normal] # Labels are 1 for tag, 0 for normal

                          # Calculate AUROC/AP for this specific tag vs normals
                          tag_metrics = compute_evaluation_metrics(scores_subset, labels_subset, k_list=[]) # No K needed here

                          per_tag_results.append({
                              'Split': split_name,
                              'Node Type': node_type,
                              'Anomaly Tag': tag,
                              'Count': count,
                              'Mean Score': mean_score,
                              'Median Score': median_score,
                              'AUROC': tag_metrics.get('AUROC', 0.0),
                              'AP': tag_metrics.get('AP', 0.0),
                              'Best F1': tag_metrics.get('Best F1', 0.0) # F1 for tag vs normal
                          })

            else: log(f"  Skipping Node Eval ({split_name}, {node_type}): GT Labels Missing")

        # --- Overall Edge Evaluation ---
        for edge_type, scores_np in current_edge_scores.items():
             edge_type_tuple = eval(edge_type) if isinstance(edge_type, str) and '(' in edge_type else edge_type
             if edge_type_tuple in current_gt_edges:
                  labels_np = current_gt_edges[edge_type_tuple].cpu().numpy()
                  edge_metrics = compute_evaluation_metrics(scores_np, labels_np, k_list=k_list)
                  split_results['edges'][str(edge_type_tuple)] = edge_metrics
                  log(f"  Overall Edge Metrics ({split_name}, {edge_type_tuple}): AUROC={edge_metrics.get('AUROC', 0):.3f}, AP={edge_metrics.get('AP', 0):.3f}")
             else: log(f"  Skipping Edge Eval ({split_name}, {edge_type_tuple}): GT Labels Missing")

        all_metrics[split_name] = split_results
    log("--- Evaluation Complete ---")

    # --- 3. Consolidate Overall Results ---
    log("\n--- Consolidating Overall Results ---")
    # (Consolidation logic remains the same as previous function)
    summary_data = []
    for split_name, results_data in all_metrics.items():
        current_graph = graphs[split_name]
        current_gt_nodes = gt_node_labels.get(split_name, {})
        current_gt_edges = gt_edge_labels.get(split_name, {})
        for node_type, metrics in results_data.get('nodes', {}).items():
            if metrics and node_type in current_graph.node_types:
                num_items = current_graph[node_type].num_nodes
                num_anomalies = int(current_gt_nodes.get(node_type, torch.tensor([])).sum().item())
                perc = (num_anomalies / num_items * 100) if num_items > 0 else 0
                row = {'Split': split_name, 'Element': f'Node ({node_type})', 'Num Items': num_items, 'Num Anomalies': num_anomalies, '% Anomalies': perc}
                row.update(metrics); summary_data.append(row)
        for edge_type_str, metrics in results_data.get('edges', {}).items():
            try: edge_type = eval(edge_type_str)
            except: edge_type = edge_type_str
            if metrics and edge_type in current_graph.edge_types:
                num_items = current_graph[edge_type].num_edges
                num_anomalies = int(current_gt_edges.get(edge_type, torch.tensor([])).sum().item())
                perc = (num_anomalies / num_items * 100) if num_items > 0 else 0
                row = {'Split': split_name, 'Element': f'Edge {edge_type_str}', 'Num Items': num_items, 'Num Anomalies': num_anomalies, '% Anomalies': perc}
                row.update(metrics); summary_data.append(row)

    

    summary_df = pd.DataFrame(summary_data)
    ordered_cols = ['Split', 'Element', 'Num Items', 'Num Anomalies', '% Anomalies',
                   'AUROC', 'AP', 'Best F1', 'Best F1 Threshold'] + \
                   [f'{p}@{k}' for k in k_list for p in ['Precision', 'Recall']]
    summary_df = summary_df.reindex(columns=ordered_cols, fill_value=np.nan)

    # Create DataFrame for per-tag results
    anomaly_type_df = pd.DataFrame(per_tag_results)
    log("--- Results Consolidated ---")

    # --- 4. Plotting (Optional) ---
    if plot:
        log("\n--- Generating Plots ---")
        try:
            plot_metric_comparison_bars(summary_df)
            plot_pr_curves_split(all_scores, gt_node_labels, gt_edge_labels, # Pass both label dicts
                                 split_name='test', target_edge_type=target_edge_type)
            log("  Generating all Score Distributions (All Sets)...")
            plot_all_score_distributions(all_scores, gt_node_labels, gt_edge_labels, # Pass both label dicts
                                            target_edge_type=target_edge_type)
            log("  Generating grouped Score Distributions (All Sets)...")
            plot_grouped_score_distributions(all_scores, gt_node_labels, gt_edge_labels, # Pass both label dicts
                                            target_edge_type=target_edge_type)
            # Plot new anomaly type comparison for test set
            plot_anomaly_type_metrics_sorted(anomaly_type_df, split_name='test')

        except Exception as e:
            log(f"An error occurred during plotting: {e}")


    log("--- Evaluation Function Finished ---")
    # Return scores, overall summary, and per-type summary
    return all_scores, summary_df, anomaly_type_df
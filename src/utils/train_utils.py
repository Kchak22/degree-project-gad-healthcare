import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
from torch_geometric.utils import negative_sampling
from torch_geometric.data import HeteroData
import time
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
import numpy as np
from torch_scatter import scatter_add, scatter_mean
from typing import Optional, Tuple, Dict
from src.utils.eval_utils import *
import os





def calculate_node_anomaly_scores(x_hat_dict, x_dict, z_dict, edge_index, model, lambda_attr=1.0, lambda_struct=0.5):
    """
    Calculates simplified node-level anomaly scores.

    Args:
        x_hat_dict: Dictionary of reconstructed node features.
        x_dict: Dictionary of original node features.
        z_dict: Dictionary of latent node embeddings.
        edge_index: Edge index for the target relation (e.g., provider -> member).
        model: The trained model instance (needed for decode_structure).
        lambda_attr: Weight for attribute loss.
        lambda_struct: Weight for structure loss.

    Returns:
        Dictionary of anomaly scores for each node type.
    """
    anomaly_scores = {}
    device = next(model.parameters()).device

    # 1. Attribute Reconstruction Error (Node-wise MSE)
    for node_type, x_hat in x_hat_dict.items():
        x = x_dict[node_type]
        attr_error = torch.sum((x_hat - x)**2, dim=1) # Sum MSE across features
        anomaly_scores[node_type] = lambda_attr * attr_error

    # 2. Structural Reconstruction Error Contribution (Simplified)
    # Calculate BCE loss contribution for edges connected to each node
    if edge_index.numel() > 0:
        # Calculate positive edge logits and their BCE loss
        pos_logits = model.decode_structure(z_dict, edge_index)
        pos_labels = torch.ones_like(pos_logits)
        # Use sigmoid + binary_cross_entropy instead of BCEWithLogits to get per-edge loss easily
        pos_probs = torch.sigmoid(pos_logits)
        pos_bce = F.binary_cross_entropy(pos_probs, pos_labels, reduction='none') # Per-edge loss

        # Aggregate structural loss contribution to connected nodes
        provider_idx, member_idx = edge_index

        # Use scatter_add to sum losses for each node (needs torch_scatter or PyG equivalent)
        # Fallback to simple loop for clarity, but scatter is much faster
        provider_struct_error = torch.zeros(z_dict['provider'].size(0), device=device)
        member_struct_error = torch.zeros(z_dict['member'].size(0), device=device)

        # Efficient scatter add (requires torch_scatter or recent PyG)
        from torch_scatter import scatter_add
        provider_struct_error = scatter_add(pos_bce, provider_idx, dim=0, dim_size=provider_struct_error.size(0))
        member_struct_error = scatter_add(pos_bce, member_idx, dim=0, dim_size=member_struct_error.size(0))

        # Add structural component to the scores
        if 'provider' in anomaly_scores:
            anomaly_scores['provider'] += lambda_struct * provider_struct_error
        else:
             anomaly_scores['provider'] = lambda_struct * provider_struct_error

        if 'member' in anomaly_scores:
            anomaly_scores['member'] += lambda_struct * member_struct_error
        else:
            anomaly_scores['member'] = lambda_struct * member_struct_error

    return anomaly_scores




def train_model_inductive(model: nn.Module,
                           train_graph: HeteroData,
                           num_epochs: int,
                           optimizer: torch.optim.Optimizer,
                           lambda_attr: float = 1.0,
                           lambda_struct: float = 0.5,
                           k_neg_samples: int = 5,
                           target_edge_type: tuple = ('provider', 'to', 'member'),
                           device: str = 'mps', # Keep default from user
                           log_freq: int = 10,
                           # --- NEW Optional Validation Args ---
                           val_graph: Optional[HeteroData] = None,
                           val_log_freq: Optional[int] = None # Use log_freq if None
                           ) -> Tuple[nn.Module, Dict]: # Return model and history
    """
    Trains the model on a given training graph (Graph A) using self-supervised
    attribute and structure reconstruction loss. Optionally calculates and logs
    validation loss on a separate validation graph.

    Args:
        model: The GNN model instance.
        train_graph: The HeteroData object for training.
        num_epochs: Number of training epochs.
        optimizer: The PyTorch optimizer.
        lambda_attr: Weight for attribute reconstruction loss.
        lambda_struct: Weight for structural reconstruction loss.
        k_neg_samples: Number of negative edge samples per positive edge for TRAIN loss.
        target_edge_type: The primary edge type for structural loss.
        device: The device ('cpu', 'cuda', 'mps') to train on.
        log_freq: How often to print training loss (epochs).
        val_graph (Optional[HeteroData]): An optional validation graph.
        val_log_freq (Optional[int]): How often to calculate and log validation loss.
                                     Defaults to log_freq if None.

    Returns:
        Tuple[nn.Module, Dict]: The trained model and the training history dictionary.
    """
    model.to(device)
    train_graph = train_graph.to(device)
    if val_graph:
        val_graph = val_graph.to(device) # Move validation graph to device if provided

    # Determine validation frequency
    validation_frequency = val_log_freq if val_log_freq is not None else log_freq

    src_node_type, _, dst_node_type = target_edge_type
    num_nodes_src_train = train_graph[src_node_type].num_nodes
    num_nodes_dst_train = train_graph[dst_node_type].num_nodes

    print(f"--- Starting Inductive Training for {num_epochs} epochs ---")
    if val_graph:
        print(f"  Validation will be performed every {validation_frequency} epochs.")
    start_time = time.time()

    # Initialize history dictionary including validation keys
    history = {
        'train_loss': [], 'train_loss_attr': [], 'train_loss_struct': [],
        'val_loss': [], 'val_loss_attr': [], 'val_loss_struct': [],
        'epochs_validated': [] # Track epochs where validation was run
        }

    for epoch in range(num_epochs):
        # --- Training Step ---
        model.train() # Set model to training mode
        optimizer.zero_grad()

        # Forward pass on the entire training graph
        outputs_train = model(train_graph)
        x_hat_dict_train = {'member': outputs_train[0], 'provider': outputs_train[1]}
        z_dict_train = outputs_train[5]

        # 1. Train Attribute Loss
        loss_attr_m_tr = F.mse_loss(x_hat_dict_train['member'], train_graph['member'].x)
        loss_attr_p_tr = F.mse_loss(x_hat_dict_train['provider'], train_graph['provider'].x)
        loss_attr_tr = loss_attr_m_tr + loss_attr_p_tr

        # 2. Train Structural Loss (Self-Supervised)
        loss_struct_tr = torch.tensor(0.0, device=device)
        if target_edge_type in train_graph.edge_index_dict:
            pos_edge_index_tr = train_graph[target_edge_type].edge_index
            num_pos_edges_tr = pos_edge_index_tr.shape[1]

            if num_pos_edges_tr > 0:
                neg_edge_index_tr = negative_sampling(
                    edge_index=pos_edge_index_tr,
                    num_nodes=(num_nodes_src_train, num_nodes_dst_train),
                    num_neg_samples=num_pos_edges_tr * k_neg_samples,
                    method='sparse'
                ).to(device)

                pos_logits_tr = model.decode_structure(z_dict_train, pos_edge_index_tr)
                neg_logits_tr = model.decode_structure(z_dict_train, neg_edge_index_tr)

                logits_tr = torch.cat([pos_logits_tr, neg_logits_tr], dim=0)
                labels_tr = torch.cat([
                    torch.ones_like(pos_logits_tr),
                    torch.zeros_like(neg_logits_tr)
                ], dim=0)
                loss_struct_tr = F.binary_cross_entropy_with_logits(logits_tr, labels_tr)

        # Total Training Loss
        total_loss_tr = lambda_attr * loss_attr_tr + lambda_struct * loss_struct_tr

        total_loss_tr.backward()
        optimizer.step()

        # Store training history for this epoch
        history['train_loss'].append(total_loss_tr.item())
        history['train_loss_attr'].append(loss_attr_tr.item())
        history['train_loss_struct'].append(loss_struct_tr.item())

        # --- Optional Validation Step ---
        run_validation = val_graph is not None and ((epoch + 1) % validation_frequency == 0 or epoch == num_epochs - 1)

        log_message = f"Epoch {epoch+1}/{num_epochs} | Train Loss: {total_loss_tr.item():.4f} " \
                      f"(Attr: {loss_attr_tr.item():.4f}, Struct: {loss_struct_tr.item():.4f})"

        if run_validation:
            model.eval() # Set model to evaluation mode
            with torch.no_grad():
                # Forward pass on the validation graph
                outputs_val = model(val_graph)
                x_hat_dict_val = {'member': outputs_val[0], 'provider': outputs_val[1]}
                z_dict_val = outputs_val[5]

                # 1. Validation Attribute Loss
                loss_attr_m_val = torch.tensor(0.0, device=device)
                loss_attr_p_val = torch.tensor(0.0, device=device)
                # Check if node types exist in val graph before calculating loss
                if 'member' in val_graph.node_types and val_graph['member'].num_nodes > 0:
                     loss_attr_m_val = F.mse_loss(x_hat_dict_val['member'], val_graph['member'].x)
                if 'provider' in val_graph.node_types and val_graph['provider'].num_nodes > 0:
                     loss_attr_p_val = F.mse_loss(x_hat_dict_val['provider'], val_graph['provider'].x)
                loss_attr_val = loss_attr_m_val + loss_attr_p_val

                # 2. Validation Structural Loss
                loss_struct_val = torch.tensor(0.0, device=device)
                if target_edge_type in val_graph.edge_index_dict:
                    pos_edge_index_val = val_graph[target_edge_type].edge_index
                    num_pos_edges_val = pos_edge_index_val.shape[1]

                    if num_pos_edges_val > 0:
                        # Negative sampling within the validation graph context
                        num_nodes_src_val = val_graph[src_node_type].num_nodes
                        num_nodes_dst_val = val_graph[dst_node_type].num_nodes

                        # Only sample if nodes exist in val graph
                        if num_nodes_src_val > 0 and num_nodes_dst_val > 0:
                             neg_edge_index_val = negative_sampling(
                                 edge_index=pos_edge_index_val,
                                 num_nodes=(num_nodes_src_val, num_nodes_dst_val),
                                 # Use k=1 for validation loss efficiency maybe? Or same k? Let's use same k for consistency.
                                 num_neg_samples=num_pos_edges_val * k_neg_samples,
                                 method='sparse'
                             ).to(device)

                             pos_logits_val = model.decode_structure(z_dict_val, pos_edge_index_val)
                             neg_logits_val = model.decode_structure(z_dict_val, neg_edge_index_val)

                             logits_val = torch.cat([pos_logits_val, neg_logits_val], dim=0)
                             labels_val = torch.cat([
                                 torch.ones_like(pos_logits_val),
                                 torch.zeros_like(neg_logits_val)
                             ], dim=0)
                             loss_struct_val = F.binary_cross_entropy_with_logits(logits_val, labels_val)

                # Total Validation Loss
                total_loss_val = lambda_attr * loss_attr_val + lambda_struct * loss_struct_val

                # Store validation history
                history['val_loss'].append(total_loss_val.item())
                history['val_loss_attr'].append(loss_attr_val.item())
                history['val_loss_struct'].append(loss_struct_val.item())
                history['epochs_validated'].append(epoch + 1) # Store epoch number

                log_message += f" | Val Loss: {total_loss_val.item():.4f} " \
                               f"(Attr: {loss_attr_val.item():.4f}, Struct: {loss_struct_val.item():.4f})"

        # Log training/validation info for this epoch
        if (epoch + 1) % log_freq == 0 or epoch == num_epochs - 1:
            print(log_message)


    end_time = time.time()
    print(f"--- Training Finished in {end_time - start_time:.2f} seconds ---")

    # --- Plotting ---
    num_plots = 2 # Train Loss, Val Loss (if available)
    fig, axes = plt.subplots(num_plots, 1, figsize=(10, 5 * num_plots), sharex=True)

    # Plot Training Loss
    axes[0].plot(history['train_loss'], label='Train Total Loss', alpha=0.8)
    axes[0].plot(history['train_loss_attr'], label='Train Attr Loss', linestyle='--', alpha=0.6)
    axes[0].plot(history['train_loss_struct'], label='Train Struct Loss', linestyle=':', alpha=0.6)
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss Components')
    axes[0].legend()
    axes[0].grid(True)

    # Plot Validation Loss (if available)
    if history['val_loss']: # Check if validation was run
        val_epochs = history['epochs_validated']
        axes[1].plot(val_epochs, history['val_loss'], label='Validation Total Loss', marker='o', linestyle='-', markersize=4)
        axes[1].plot(val_epochs, history['val_loss_attr'], label='Val Attr Loss', marker='x', linestyle='--', markersize=4)
        axes[1].plot(val_epochs, history['val_loss_struct'], label='Val Struct Loss', marker='+', linestyle=':', markersize=4)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Validation Loss Components')
        axes[1].legend()
        axes[1].grid(True)
    else:
        # Hide the second subplot if no validation data
        fig.delaxes(axes[1])
        fig.set_size_inches(10, 5) # Adjust figure size

    plt.tight_layout()
    plt.show()

    return model, history



import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
from torch_geometric.utils import negative_sampling
from torch_geometric.data import HeteroData
import time
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, f1_score, PrecisionRecallDisplay
import numpy as np
import pandas as pd
import seaborn as sns
from collections import defaultdict
import warnings
import os # For path operations
from typing import Optional, Tuple, Dict, List

# Ensure torch_scatter functions are available or handle the import error
try:
    from torch_scatter import scatter_add, scatter_mean
except ImportError:
    print("torch_scatter not found. Structural score aggregation might be limited or slow.")
    # Define dummy functions or raise error if essential
    def scatter_add(*args, **kwargs): raise ImportError("torch_scatter.scatter_add required")
    def scatter_mean(*args, **kwargs): raise ImportError("torch_scatter.scatter_mean required")

# ======================================================================
# == Assume calculate_anomaly_scores & compute_evaluation_metrics    ==
# == are defined correctly elsewhere and imported/available.         ==
# == Their definitions are NOT included here for brevity, but        ==
# == they are required for this training function to work.           ==
# ======================================================================
# Example Signatures (replace with your actual functions):
# def calculate_anomaly_scores(trained_model, eval_graph_data, lambda_attr, lambda_struct, target_edge_type, k_neg_samples_struct_score) -> Tuple[Dict, Dict]: ...
# def compute_evaluation_metrics(scores, labels, k_list) -> Dict: ...
# ======================================================================


def train_model_inductive_with_metrics(
                           model: nn.Module,
                           train_graph: HeteroData,
                           num_epochs: int,
                           optimizer: torch.optim.Optimizer,
                           lambda_attr: float = 1.0,
                           lambda_struct: float = 0.5,
                           k_neg_samples: int = 5,
                           target_edge_type: tuple = ('provider', 'to', 'member'),
                           device: str = 'cpu', # Default to CPU if not specified
                           log_freq: int = 10,
                           # --- Validation Args ---
                           val_graph: Optional[HeteroData] = None,
                           gt_node_labels_val: Optional[Dict[str, torch.Tensor]] = None,
                           gt_edge_labels_val: Optional[Dict[Tuple, torch.Tensor]] = None,
                           val_log_freq: Optional[int] = None,
                           # --- Validation Scoring Params ---
                           val_lambda_attr: float = 1.0,
                           val_lambda_struct: float = 0.5,
                           val_k_neg_samples_score: int = 1,
                           # --- Node Metrics Params ---
                           node_k_list: List[int] = [50, 100, 200],
                           # --- Early Stopping/Model Saving ---
                           early_stopping_metric: str = 'AP',
                           early_stopping_element: str = 'Avg AP', # ('provider','member','Avg AP', or str(edge_type))
                           patience: int = 10,
                           save_best_model_path: Optional[str] = None # Full file path
                           ) -> Tuple[nn.Module, Dict, Optional[str]]:
    """
    Trains the model inductively, monitors validation loss AND full metrics
    (nodes & edges), and implements early stopping based on a chosen metric.

    Requires the `calculate_anomaly_scores` and `compute_evaluation_metrics`
    functions to be defined and available in the scope.

    Args:
        model: The GNN model instance (e.g., BipartiteGraphAutoEncoder).
        train_graph: The HeteroData object for training.
        num_epochs: Number of training epochs.
        optimizer: The PyTorch optimizer.
        lambda_attr: Weight for training attribute reconstruction loss.
        lambda_struct: Weight for training structural reconstruction loss.
        k_neg_samples: Number of negative edge samples per positive edge for TRAIN loss.
        target_edge_type: The primary edge type tuple (src, rel, dst) for structural loss/eval.
        device: The device ('cpu', 'cuda', 'mps') to train on.
        log_freq: How often to print training/validation logs (epochs).
        val_graph (Optional[HeteroData]): The validation graph. Required for validation steps.
        gt_node_labels_val (Optional[Dict]): Ground truth node labels {type: tensor} for validation.
                                             Required for node metric calculation and node/Avg AP early stopping.
        gt_edge_labels_val (Optional[Dict]): Ground truth edge labels {type_tuple: tensor} for validation.
                                             Required for edge metric calculation and edge early stopping.
        val_log_freq (Optional[int]): How often to calculate and log validation performance. Defaults to log_freq.
        val_lambda_attr (float): lambda_attr used for validation scoring.
        val_lambda_struct (float): lambda_struct used for validation scoring.
        val_k_neg_samples_score (int): k for negative sampling during structural validation scoring.
        node_k_list (List[int]): K values for node P@K/R@K calculation in history.
        early_stopping_metric (str): Metric to monitor ('AUROC', 'AP', 'Best F1').
        early_stopping_element (str): Element type ('provider', 'member', 'Avg AP', or edge type string like "('provider', 'to', 'member')").
        patience (int): Number of validation checks without improvement before stopping.
        save_best_model_path (Optional[str]): If provided, saves the best model state dict to this file path.

    Returns:
        Tuple[nn.Module, Dict, Optional[str]]:
            - The model (last state or best loaded state if saved).
            - The training history dictionary containing losses and metrics per epoch/validated epoch.
            - Path to the saved best model state_dict (or None).
    """
    # --- Setup ---
    model.to(device)
    train_graph = train_graph.to(device)
    validation_active = False # Will be set to True if validation can run with metrics
    avg_ap_possible = False

    if val_graph:
        val_graph = val_graph.to(device)
        if gt_node_labels_val:
             gt_node_labels_val = {k: v.to(device) for k, v in gt_node_labels_val.items()}
             avg_ap_possible = 'provider' in gt_node_labels_val and 'member' in gt_node_labels_val
        if gt_edge_labels_val:
            gt_edge_labels_val = {k: v.to(device) for k, v in gt_edge_labels_val.items()}

        # Determine if early stopping/metric validation is possible
        if early_stopping_element == 'Avg AP':
            validation_active = avg_ap_possible
        elif early_stopping_element in ['provider', 'member']:
            validation_active = gt_node_labels_val is not None and early_stopping_element in gt_node_labels_val
        else: # Assume edge type
            try: # Safely check if the element is a valid edge type with labels
                edge_type_key = eval(early_stopping_element) if isinstance(early_stopping_element, str) and '(' in early_stopping_element else early_stopping_element
                validation_active = gt_edge_labels_val is not None and edge_type_key in gt_edge_labels_val
            except:
                validation_active = False # Invalid element string

    validation_frequency = val_log_freq if val_log_freq is not None else log_freq
    if not val_graph:
        print("Warning: No validation graph provided. Validation steps will be skipped.")
    elif not validation_active:
         print(f"Warning: Validation graph provided, but ground truth labels for monitoring element '{early_stopping_element}' are missing. Validation metrics and early stopping disabled.")

    src_node_type, _, dst_node_type = target_edge_type
    num_nodes_src_train = train_graph[src_node_type].num_nodes
    num_nodes_dst_train = train_graph[dst_node_type].num_nodes

    print(f"--- Starting Inductive Training for {num_epochs} epochs ---")
    if validation_active:
        print(f"  Validation loss & full metrics calculated every {validation_frequency} epochs.")
        print(f"  Monitoring Validation '{early_stopping_element}' '{early_stopping_metric}' for Early Stopping.")
        print(f"  Early stopping patience: {patience} validation checks.")
        if save_best_model_path:
            print(f"  Best model will be saved to: {save_best_model_path}")
    elif val_graph:
         print(f"  Validation loss calculated every {validation_frequency} epochs (GT labels missing/invalid for metrics/early stopping).")

    start_time = time.time()

    # Initialize history dictionary
    history = {
        'train_loss': [], 'train_loss_attr': [], 'train_loss_struct': [],
        'val_loss': [], 'val_loss_attr': [], 'val_loss_struct': [],
        'epochs_validated': [],
        'val_provider_AUROC': [], 'val_provider_AP': [], 'val_provider_Best F1': [],
        'val_member_AUROC': [], 'val_member_AP': [], 'val_member_Best F1': [],
        'val_Avg_AP': [],
        'val_edge_AUROC': [], 'val_edge_AP': [], 'val_edge_Best F1': [],
        'best_val_metric_value': -np.inf, # Use -inf for maximization metrics
        'best_epoch': -1
    }
    # Add P@K, R@K keys
    for k in node_k_list:
        history[f'val_provider_Precision@{k}'] = []
        history[f'val_provider_Recall@{k}'] = []
        history[f'val_member_Precision@{k}'] = []
        history[f'val_member_Recall@{k}'] = []

    best_val_metric = -np.inf
    epochs_no_improve = 0
    saved_best_model_filepath = None

    # --- Training Loop ---
    for epoch in range(num_epochs):
        # --- Training Step ---
        model.train()
        optimizer.zero_grad()

        # Forward pass on the entire training graph
        # Assumes model returns: x_hat_m, x_hat_p, z_m, z_p, edge_hat_dict, z_dict
        # Adjust indices if your model returns something different
        try:
             outputs_train = model(train_graph)
             x_hat_dict_train = {'member': outputs_train[0], 'provider': outputs_train[1]}
             z_dict_train = outputs_train[5] # Expects z_dict = {'provider': z_p, 'member': z_m}
        except Exception as e:
             print(f"Error during model forward pass in training (epoch {epoch+1}): {e}")
             # Option: re-raise, break, or continue with dummy values if possible
             break # Stop training if forward pass fails

        # 1. Train Attribute Loss
        loss_attr_tr = torch.tensor(0.0, device=device)
        try:
            # Check node types and feature existence before calculating loss
            if 'member' in train_graph.node_types and hasattr(train_graph['member'], 'x') and train_graph['member'].num_nodes > 0:
                if x_hat_dict_train['member'].shape == train_graph['member'].x.shape:
                     loss_attr_tr += F.mse_loss(x_hat_dict_train['member'], train_graph['member'].x)
                else: print(f"Warning (Train Epoch {epoch+1}): Shape mismatch for member features.")
            if 'provider' in train_graph.node_types and hasattr(train_graph['provider'], 'x') and train_graph['provider'].num_nodes > 0:
                 if x_hat_dict_train['provider'].shape == train_graph['provider'].x.shape:
                     loss_attr_tr += F.mse_loss(x_hat_dict_train['provider'], train_graph['provider'].x)
                 else: print(f"Warning (Train Epoch {epoch+1}): Shape mismatch for provider features.")
        except Exception as e:
            print(f"Error calculating training attribute loss (epoch {epoch+1}): {e}")
            loss_attr_tr = torch.tensor(torch.nan, device=device) # Mark as NaN


        # 2. Train Structural Loss (Self-Supervised)
        loss_struct_tr = torch.tensor(0.0, device=device)
        if target_edge_type in train_graph.edge_index_dict:
            pos_edge_index_tr = train_graph[target_edge_type].edge_index
            num_pos_edges_tr = pos_edge_index_tr.shape[1]

            if num_pos_edges_tr > 0 and num_nodes_src_train > 0 and num_nodes_dst_train > 0:
                try:
                    neg_edge_index_tr = negative_sampling(
                        edge_index=pos_edge_index_tr,
                        num_nodes=(num_nodes_src_train, num_nodes_dst_train),
                        num_neg_samples=num_pos_edges_tr * k_neg_samples,
                        method='sparse' # 'sparse' is generally recommended
                    ).to(device)

                    pos_logits_tr = model.decode_structure(z_dict_train, pos_edge_index_tr)
                    neg_logits_tr = model.decode_structure(z_dict_train, neg_edge_index_tr)

                    logits_tr = torch.cat([pos_logits_tr, neg_logits_tr], dim=0)
                    labels_tr = torch.cat([
                        torch.ones_like(pos_logits_tr),
                        torch.zeros_like(neg_logits_tr)
                    ], dim=0)
                    loss_struct_tr = F.binary_cross_entropy_with_logits(logits_tr, labels_tr)
                except Exception as e:
                    print(f"Error calculating training structural loss (epoch {epoch+1}): {e}")
                    loss_struct_tr = torch.tensor(torch.nan, device=device) # Mark as NaN


        # Total Training Loss
        total_loss_tr = lambda_attr * loss_attr_tr + lambda_struct * loss_struct_tr

        # Backward pass and optimization step (handle potential NaN loss)
        if not torch.isnan(total_loss_tr):
             try:
                 total_loss_tr.backward()
                 optimizer.step()
             except Exception as e:
                  print(f"Error during backward/optimizer step (epoch {epoch+1}): {e}")
                  # Potentially break or skip epoch depending on severity
        else:
             print(f"Warning: NaN loss detected in training (epoch {epoch+1}). Skipping backward/step.")


        # Store training history for this epoch
        history['train_loss'].append(total_loss_tr.item())
        history['train_loss_attr'].append(loss_attr_tr.item())
        history['train_loss_struct'].append(loss_struct_tr.item())

        # --- Optional Validation Step ---
        run_validation_this_epoch = val_graph is not None and ((epoch + 1) % validation_frequency == 0 or epoch == num_epochs - 1)

        log_message_train = f"Epoch {epoch+1}/{num_epochs} | Train Loss: {total_loss_tr.item():.4f}"
        log_message_val = ""

        if run_validation_this_epoch:
            model.eval() # Set model to evaluation mode
            with torch.no_grad():
                # --- Calculate Validation Loss ---
                try:
                    outputs_val = model(val_graph)
                    x_hat_dict_val = {'member': outputs_val[0], 'provider': outputs_val[1]}
                    z_dict_val = outputs_val[5] # Assuming z_dict structure is correct
                except Exception as e:
                     print(f"Error during model forward pass in validation (epoch {epoch+1}): {e}")
                     # Skip validation if forward pass fails
                     continue

                # Val Attribute Loss
                loss_attr_val = torch.tensor(0.0, device=device)
                if 'member' in val_graph.node_types and hasattr(val_graph['member'], 'x') and val_graph['member'].num_nodes > 0:
                     if x_hat_dict_val['member'].shape == val_graph['member'].x.shape:
                          loss_attr_val += F.mse_loss(x_hat_dict_val['member'], val_graph['member'].x)
                     else: print(f"Warning (Val Epoch {epoch+1}): Shape mismatch for member features.")
                if 'provider' in val_graph.node_types and hasattr(val_graph['provider'], 'x') and val_graph['provider'].num_nodes > 0:
                     if x_hat_dict_val['provider'].shape == val_graph['provider'].x.shape:
                          loss_attr_val += F.mse_loss(x_hat_dict_val['provider'], val_graph['provider'].x)
                     else: print(f"Warning (Val Epoch {epoch+1}): Shape mismatch for provider features.")

                # Val Structural Loss (using same k as training for consistency in loss value)
                loss_struct_val = torch.tensor(0.0, device=device)
                if target_edge_type in val_graph.edge_index_dict:
                    pos_edge_index_val = val_graph[target_edge_type].edge_index
                    num_pos_edges_val = pos_edge_index_val.shape[1]
                    # Get node counts from val graph
                    num_nodes_src_val = val_graph[src_node_type].num_nodes if src_node_type in val_graph.node_types else 0
                    num_nodes_dst_val = val_graph[dst_node_type].num_nodes if dst_node_type in val_graph.node_types else 0

                    if num_pos_edges_val > 0 and num_nodes_src_val > 0 and num_nodes_dst_val > 0:
                         try:
                              neg_edge_index_val = negative_sampling(
                                  edge_index=pos_edge_index_val,
                                  num_nodes=(num_nodes_src_val, num_nodes_dst_val),
                                  num_neg_samples=num_pos_edges_val * k_neg_samples, # Use training k for loss calc
                                  method='sparse'
                              ).to(device)
                              pos_logits_val = model.decode_structure(z_dict_val, pos_edge_index_val)
                              neg_logits_val = model.decode_structure(z_dict_val, neg_edge_index_val)
                              logits_val = torch.cat([pos_logits_val, neg_logits_val], dim=0)
                              labels_val = torch.cat([torch.ones_like(pos_logits_val), torch.zeros_like(neg_logits_val)], dim=0)
                              loss_struct_val = F.binary_cross_entropy_with_logits(logits_val, labels_val)
                         except Exception as e:
                              print(f"Error calculating validation structural loss (epoch {epoch+1}): {e}")
                              loss_struct_val = torch.tensor(torch.nan, device=device)

                total_loss_val = val_lambda_attr * loss_attr_val + val_lambda_struct * loss_struct_val # Use val lambdas

                # Store validation loss history
                history['val_loss'].append(total_loss_val.item())
                history['val_loss_attr'].append(loss_attr_val.item())
                history['val_loss_struct'].append(loss_struct_val.item())
                history['epochs_validated'].append(epoch + 1)
                log_message_val += f" | Val Loss: {total_loss_val.item():.4f}"

                # --- Calculate Validation METRICS (if labels provided) ---
                current_metric_for_stopping = -np.inf # Default for maximization
                ap_p_val, ap_m_val = np.nan, np.nan # For Avg AP calculation

                if validation_active: # Only calculate metrics if GT labels for monitored element exist
                    # Calculate validation node scores using self-supervised method
                    try:
                        # Use calculate_anomaly_scores (ensure it's defined/imported)
                        val_node_scores, val_edge_scores = calculate_anomaly_scores(
                            trained_model=model, # Current state
                            eval_graph_data=val_graph,
                            lambda_attr=val_lambda_attr, # Use lambdas specific to validation scoring
                            lambda_struct=val_lambda_struct,
                            target_edge_type=target_edge_type,
                            k_neg_samples_struct_score=val_k_neg_samples_score # Use k specific to val scoring
                        )
                    except Exception as e:
                        print(f"Error calculating validation anomaly scores (epoch {epoch+1}): {e}")
                        val_node_scores, val_edge_scores = {}, {} # Prevent further errors

                    # Node Metrics
                    for node_type in ['provider', 'member']:
                        scores_tensor = val_node_scores.get(node_type)
                        labels_tensor = gt_node_labels_val.get(node_type)

                        if scores_tensor is not None and labels_tensor is not None:
                            try:
                                scores_np = scores_tensor.cpu().numpy()
                                labels_np = labels_tensor.cpu().numpy()
                                metrics = compute_evaluation_metrics(scores_np, labels_np, node_k_list) # Ensure function handles potential errors
                                for metric_name, value in metrics.items():
                                    history_key = f'val_{node_type}_{metric_name}'
                                    if history_key in history: history[history_key].append(value)

                                # Track AP for averaging
                                if node_type == 'provider': ap_p_val = metrics.get('AP', np.nan)
                                if node_type == 'member': ap_m_val = metrics.get('AP', np.nan)

                                log_message_val += f" | Val {node_type[0].upper()} AP:{metrics.get('AP',np.nan):.3f}"
                                if early_stopping_element == node_type:
                                    current_metric_for_stopping = metrics.get(early_stopping_metric, -np.inf)
                            except Exception as e:
                                print(f"Error computing node metrics for {node_type} (epoch {epoch+1}): {e}")
                                # Append NaNs if metrics calculation fails
                                for metric_name in ['AUROC', 'AP', 'Best F1'] + [f'{p}@{k}' for k in node_k_list for p in ['Precision', 'Recall']]:
                                     history_key = f'val_{node_type}_{metric_name}'
                                     if history_key in history: history[history_key].append(np.nan)

                        else:
                             # Append NaNs if labels or scores missing
                             for metric_name in ['AUROC', 'AP', 'Best F1'] + [f'{p}@{k}' for k in node_k_list for p in ['Precision', 'Recall']]:
                                 history_key = f'val_{node_type}_{metric_name}'
                                 if history_key in history: history[history_key].append(np.nan)

                    # Calculate and Store Average AP
                    avg_ap_val = np.nanmean([ap_p_val, ap_m_val]) if not (np.isnan(ap_p_val) and np.isnan(ap_m_val)) else np.nan
                    history['val_Avg_AP'].append(avg_ap_val)
                    if early_stopping_element == 'Avg AP':
                        current_metric_for_stopping = avg_ap_val if not np.isnan(avg_ap_val) else -np.inf


                    # Edge Metrics
                    target_edge_str = str(target_edge_type)
                    edge_scores_np = val_edge_scores.get(target_edge_type) # Should be numpy array from calc_scores
                    edge_labels_tensor = gt_edge_labels_val.get(target_edge_type)

                    if edge_scores_np is not None and edge_labels_tensor is not None:
                        try:
                            edge_labels_np = edge_labels_tensor.cpu().numpy()
                            # Note: calculate_anomaly_scores returns negated logits for edges.
                            # compute_evaluation_metrics assumes higher score = more anomalous.
                            edge_metrics = compute_evaluation_metrics(edge_scores_np, edge_labels_np, k_list=[])
                            history['val_edge_AUROC'].append(edge_metrics.get('AUROC', 0.0))
                            history['val_edge_AP'].append(edge_metrics.get('AP', 0.0))
                            history['val_edge_Best F1'].append(edge_metrics.get('Best F1', 0.0))
                            log_message_val += f" | Val Edge AP:{edge_metrics.get('AP',np.nan):.3f}"
                            # Check early stopping for edge
                            if early_stopping_element == target_edge_str: # Check if monitoring edge
                                current_metric_for_stopping = edge_metrics.get(early_stopping_metric, -np.inf)
                        except Exception as e:
                             print(f"Error computing edge metrics for {target_edge_type} (epoch {epoch+1}): {e}")
                             history['val_edge_AUROC'].append(np.nan); history['val_edge_AP'].append(np.nan); history['val_edge_Best F1'].append(np.nan)
                    else:
                        # Append NaNs if labels or scores missing
                        history['val_edge_AUROC'].append(np.nan)
                        history['val_edge_AP'].append(np.nan)
                        history['val_edge_Best F1'].append(np.nan)


                    # --- Early Stopping Check ---
                    # Use >= to favor later epochs slightly in case of ties
                    if current_metric_for_stopping >= best_val_metric:
                        best_val_metric = current_metric_for_stopping
                        history['best_val_metric_value'] = best_val_metric
                        history['best_epoch'] = epoch + 1
                        epochs_no_improve = 0
                        if save_best_model_path:
                            try:
                                # Ensure directory exists before saving
                                save_dir = os.path.dirname(save_best_model_path)
                                if save_dir: os.makedirs(save_dir, exist_ok=True) # Check if path has dir part
                                torch.save(model.state_dict(), save_best_model_path)
                                saved_best_model_filepath = save_best_model_path
                                log_message_val += " *" # Indicate improvement/saving
                            except Exception as e:
                                print(f"Warning: Failed to save best model to {save_best_model_path}: {e}")
                                saved_best_model_filepath = None # Reset if save failed
                    else:
                        epochs_no_improve += 1

                    if epochs_no_improve >= patience:
                        print(f"\nEarly stopping triggered at epoch {epoch+1} after {patience} checks ({patience * validation_frequency} epochs) without improvement "
                              f"on validation '{early_stopping_element}' '{early_stopping_metric}'.")
                        break # Exit the training loop

        # Log combined message (print every log_freq or if validation happened)
        if (epoch + 1) % log_freq == 0 or epoch == num_epochs - 1 or run_validation_this_epoch:
             print(log_message_train + log_message_val)

        # Check if loop should break due to early stopping
        if epochs_no_improve >= patience and validation_active:
            break

    # --- Post-Training ---
    end_time = time.time()
    print(f"--- Training Finished in {end_time - start_time:.2f} seconds ---")
    if history['best_epoch'] != -1:
        print(f"Best validation '{early_stopping_element}' '{early_stopping_metric}': {history['best_val_metric_value']:.4f} at epoch {history['best_epoch']}")
        if saved_best_model_filepath:
            print(f"Best model state_dict saved to: {saved_best_model_filepath}")
    elif validation_active:
         print("No improvement detected during validation based on monitored metric.")

    # Load the best model state if it was saved
    if saved_best_model_filepath and os.path.exists(saved_best_model_filepath):
        print(f"Loading best model state from epoch {history['best_epoch']} ({saved_best_model_filepath})")
        try:
            # Ensure map_location handles device properly
            model.load_state_dict(torch.load(saved_best_model_filepath, map_location=torch.device(device)))
            model.to(device) # Ensure model is on the correct device after loading
        except Exception as e:
            print(f"Warning: Failed to load best model state from {saved_best_model_filepath}: {e}")
            saved_best_model_filepath = None # Indicate loading failed
    elif save_best_model_path:
         print("Warning: Best model path provided but no model was saved (or loading failed). Returning last model state.")



    return model, history, saved_best_model_filepath

@torch.no_grad()
def calculate_anomaly_scores(
    trained_model: nn.Module,
    eval_graph_data: HeteroData,
    lambda_attr: float = 1.0,
    lambda_struct: float = 0.5,
    target_edge_type: tuple = ('provider', 'to', 'member'),
    k_neg_samples_struct_score: int = 1 # Number of neg samples per node for scoring
    ):
    """
    Calculates node and edge anomaly scores for an evaluation graph using a
    trained model in a purely self-supervised manner.

    Args:
        trained_model: The already trained GNN model instance.
        eval_graph_data: The HeteroData object to evaluate (e.g., Graph B).
        lambda_attr: Weight for the attribute reconstruction error component.
        lambda_struct: Weight for the structural reconstruction error component.
        target_edge_type: The primary edge type for structure evaluation.
        k_neg_samples_struct_score: Number of negative edges to sample per positive
                                     edge for calculating the structural score component.
                                     Set to 0 to only use positive edge error.

    Returns:
        node_scores (dict): Dictionary {node_type: anomaly_scores_tensor}.
        edge_scores (dict): Dictionary {edge_type: edge_anomaly_scores_tensor (negated logits)}.
                            Contains scores only for the target_edge_type.
    """
    trained_model.eval()
    device = next(trained_model.parameters()).device
    eval_graph_data = eval_graph_data.to(device)

    src_node_type, _, dst_node_type = target_edge_type
    num_nodes_src = eval_graph_data[src_node_type].num_nodes
    num_nodes_dst = eval_graph_data[dst_node_type].num_nodes

    print("Calculating self-supervised anomaly scores...")

    # Run forward pass on the evaluation graph
    outputs = trained_model(eval_graph_data)
    x_hat_dict = {'member': outputs[0], 'provider': outputs[1]}
    z_dict = outputs[5]
    x_dict = eval_graph_data.x_dict # Original features from the eval graph

    node_scores = {}
    edge_scores = {}

    # 1. Calculate Attribute Reconstruction Error (Node-wise MSE)
    for node_type in eval_graph_data.node_types:
        if node_type in x_hat_dict and node_type in x_dict:
            # Ensure features exist and dimensions match before calculating loss
            if x_hat_dict[node_type].shape[0] == x_dict[node_type].shape[0] and \
               x_hat_dict[node_type].shape[1] == x_dict[node_type].shape[1]:
                attr_error = F.mse_loss(x_hat_dict[node_type], x_dict[node_type], reduction='none').sum(dim=1)
                node_scores[node_type] = lambda_attr * attr_error
            else:
                 print(f"Warning: Feature shape mismatch for node type {node_type} in eval graph. Skipping attribute score.")
                 node_scores[node_type] = torch.zeros(eval_graph_data[node_type].num_nodes, device=device) # Initialize score tensor

        else:
            # Initialize score tensor if node type doesn't have features or wasn't reconstructed
            if node_type in eval_graph_data.node_types:
                 node_scores[node_type] = torch.zeros(eval_graph_data[node_type].num_nodes, device=device)


    # 2. Calculate Structural Reconstruction Error Contribution (Self-Supervised)
    struct_errors = {nt: torch.zeros(eval_graph_data[nt].num_nodes, device=device)
                     for nt in eval_graph_data.node_types}

    if target_edge_type in eval_graph_data.edge_index_dict:
        pos_edge_index = eval_graph_data[target_edge_type].edge_index
        num_pos_edges = pos_edge_index.shape[1]

        if num_pos_edges > 0:
            # --- Positive Edge Errors ---
            pos_logits = trained_model.decode_structure(z_dict, pos_edge_index)
            # Loss for existing edges (how well does the model predict they should exist?)
            pos_bce_loss = F.binary_cross_entropy_with_logits(
                pos_logits, torch.ones_like(pos_logits), reduction='none'
            )

            # Aggregate positive loss to nodes using scatter_mean
            src_indices, dst_indices = pos_edge_index
            struct_errors[src_node_type] = scatter_mean(
                pos_bce_loss, src_indices, dim=0,
                out=struct_errors[src_node_type], # In-place add to existing zeros
                dim_size=num_nodes_src
            )
            struct_errors[dst_node_type] = scatter_mean(
                pos_bce_loss, dst_indices, dim=0,
                out=struct_errors[dst_node_type],
                dim_size=num_nodes_dst
            )

            # --- Negative Edge Errors (Optional but Recommended) ---
            if k_neg_samples_struct_score > 0:
                neg_edge_index = negative_sampling(
                    edge_index=pos_edge_index,
                    num_nodes=(num_nodes_src, num_nodes_dst),
                    num_neg_samples=num_pos_edges * k_neg_samples_struct_score, # Sample negatives
                    method='sparse'
                ).to(device)

                neg_logits = trained_model.decode_structure(z_dict, neg_edge_index)
                # Loss for non-existing edges (how well does the model predict they shouldn't exist?)
                neg_bce_loss = F.binary_cross_entropy_with_logits(
                    neg_logits, torch.zeros_like(neg_logits), reduction='none'
                )

                # Aggregate negative loss to nodes using scatter_mean
                # Note: A node involved in many negative samples might get a higher aggregated score here.
                neg_src_indices, neg_dst_indices = neg_edge_index
                struct_errors[src_node_type] = scatter_mean(
                    neg_bce_loss, neg_src_indices, dim=0,
                    out=struct_errors[src_node_type], # Add to the positive error aggregate
                    dim_size=num_nodes_src
                )
                struct_errors[dst_node_type] = scatter_mean(
                    neg_bce_loss, neg_dst_indices, dim=0,
                    out=struct_errors[dst_node_type],
                    dim_size=num_nodes_dst
                )
        else:
             print(f"Warning: No edges found for target type {target_edge_type} in eval graph. Structural score will be zero.")

        # Add structural component to the final node scores
        for node_type in eval_graph_data.node_types:
            if node_type in node_scores: # Check if node type was processed for attr score
                 node_scores[node_type] += lambda_struct * struct_errors[node_type]


        # 3. Calculate Edge Anomaly Scores (Negated Logits for target_edge_type)
        all_logits = trained_model.decode_structure(z_dict, pos_edge_index)
        # Higher score = more anomalous. Since higher logits = more normal, we negate.
        edge_scores[target_edge_type] = -all_logits.cpu().numpy()

    else:
        print(f"Warning: Target edge type {target_edge_type} not found in eval_graph_data.")


    print("Anomaly score calculation finished.")
    return node_scores, edge_scores
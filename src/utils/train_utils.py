# --- Training Function based on Report ---
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

def train_and_validate_model(model, train_data, val_data, gt_labels, num_epochs, optimizer,
                             lambda_attr=1.0, lambda_struct=0.5, k_neg_samples=5,
                             target_edge_type=('provider', 'to', 'member'), device='cpu',
                             eval_freq=10, plot=True):
    """
    Trains and validates the BipartiteGraphAutoEncoder_ReportBased model.
    Includes loss monitoring and basic anomaly detection metric calculation on validation set.
    """
    model.to(device) # Ensure model is on the correct device
    train_data = train_data.to(device)
    val_data = val_data.to(device)
    # Move ground truth labels to device
    gt_labels = {k: v.to(device) for k, v in gt_labels.items()}


    history = {'train_loss': [], 'train_loss_attr': [], 'train_loss_struct': [],
               'val_loss': [], 'val_loss_attr': [], 'val_loss_struct': [],
               'val_auc_provider': [], 'val_ap_provider': [],
               'val_auc_member': [], 'val_ap_member': []}

    # Get positive edge indices for training and validation structural loss
    pos_edge_index_train = train_data[target_edge_type].edge_index
    num_pos_edges_train = pos_edge_index_train.shape[1]

    # Edges for validation structural loss calculation (usually from val_data.edge_label_index)
    # Need both positive and negative edges from the split for validation loss
    val_pos_edge_index = val_data[target_edge_type].edge_label_index[:, val_data[target_edge_type].edge_label == 1]
    val_neg_edge_index = val_data[target_edge_type].edge_label_index[:, val_data[target_edge_type].edge_label == 0]
    num_pos_edges_val = val_pos_edge_index.shape[1]

    num_nodes_provider = train_data['provider'].num_nodes # Assuming node counts are same across splits
    num_nodes_member = train_data['member'].num_nodes

    print(f"Starting training/validation for {num_epochs} epochs...")
    print(f"Train positive edges: {num_pos_edges_train}, Val positive edges: {num_pos_edges_val}")
    print(f"Eval Freq: {eval_freq}")

    start_time = time.time()
    for epoch in range(num_epochs):
        # --- Training Phase ---
        model.train()
        optimizer.zero_grad()

        # Forward pass on training data (using train edges for message passing)
        x_hat_m, x_hat_p, z_m, z_p = model(train_data)
        z_dict_train = {'member': z_m, 'provider': z_p}

        # Train Attribute Loss
        loss_attr_m_tr = F.mse_loss(x_hat_m, train_data['member'].x)
        loss_attr_p_tr = F.mse_loss(x_hat_p, train_data['provider'].x)
        loss_attr_tr = loss_attr_m_tr + loss_attr_p_tr

        # Train Structural Loss (with negative sampling)
        loss_struct_tr = torch.tensor(0.0, device=device)
        if num_pos_edges_train > 0:
            neg_edge_index_train = negative_sampling(
                edge_index=pos_edge_index_train,
                num_nodes=(num_nodes_provider, num_nodes_member),
                num_neg_samples=num_pos_edges_train * k_neg_samples,
                method='sparse'
            )
            pos_logits_tr = model.decode_structure(z_dict_train, pos_edge_index_train)
            neg_logits_tr = model.decode_structure(z_dict_train, neg_edge_index_train)
            logits_tr = torch.cat([pos_logits_tr, neg_logits_tr], dim=0)
            labels_tr = torch.cat([torch.ones_like(pos_logits_tr), torch.zeros_like(neg_logits_tr)], dim=0)
            loss_struct_tr = F.binary_cross_entropy_with_logits(logits_tr, labels_tr)

        total_loss_tr = lambda_attr * loss_attr_tr + lambda_struct * loss_struct_tr
        total_loss_tr.backward()
        optimizer.step()

        history['train_loss'].append(total_loss_tr.item())
        history['train_loss_attr'].append(loss_attr_tr.item())
        history['train_loss_struct'].append(loss_struct_tr.item())

        # --- Validation Phase ---
        if (epoch + 1) % eval_freq == 0 or epoch == num_epochs - 1:
            model.eval()
            with torch.no_grad():
                # Forward pass on validation data (using train edges for message passing)
                # Important: Use train_data for message passing structure, evaluate on val edges
                x_hat_m_val, x_hat_p_val, z_m_val, z_p_val = model(train_data) # Pass train data for consistent structure
                z_dict_val = {'member': z_m_val, 'provider': z_p_val}

                # Val Attribute Loss (on all nodes, using reconstructions based on train graph structure)
                loss_attr_m_val = F.mse_loss(x_hat_m_val, train_data['member'].x) # Compare against original features
                loss_attr_p_val = F.mse_loss(x_hat_p_val, train_data['provider'].x)
                loss_attr_val = loss_attr_m_val + loss_attr_p_val

                # Val Structural Loss (using pos/neg edges from val_data split)
                loss_struct_val = torch.tensor(0.0, device=device)
                if num_pos_edges_val > 0 or val_neg_edge_index.shape[1] > 0:
                    pos_logits_val = model.decode_structure(z_dict_val, val_pos_edge_index)
                    neg_logits_val = model.decode_structure(z_dict_val, val_neg_edge_index)
                    logits_val = torch.cat([pos_logits_val, neg_logits_val], dim=0)
                    labels_val = torch.cat([torch.ones_like(pos_logits_val), torch.zeros_like(neg_logits_val)], dim=0)
                    loss_struct_val = F.binary_cross_entropy_with_logits(logits_val, labels_val)

                total_loss_val = lambda_attr * loss_attr_val + lambda_struct * loss_struct_val
                history['val_loss'].append(total_loss_val.item())
                history['val_loss_attr'].append(loss_attr_val.item())
                history['val_loss_struct'].append(loss_struct_val.item())

                # --- Anomaly Detection Metrics ---
                # Calculate node anomaly scores based on the validation forward pass
                anomaly_scores_dict = calculate_node_anomaly_scores(
                    {'member': x_hat_m_val, 'provider': x_hat_p_val}, # Reconstructions
                    {'member': train_data['member'].x, 'provider': train_data['provider'].x}, # Originals
                    z_dict_val, # Latent embeddings
                    val_pos_edge_index, # Use positive validation edges for structure part of score
                    model,
                    lambda_attr,
                    lambda_struct
                )

                auc_provider, ap_provider, auc_member, ap_member = 0.0, 0.0, 0.0, 0.0
                if 'provider' in anomaly_scores_dict and 'provider' in gt_labels:
                    scores_p = anomaly_scores_dict['provider'].cpu().numpy()
                    labels_p = gt_labels['provider'].cpu().numpy()
                    if len(np.unique(labels_p)) > 1: # Check if both classes exist
                       auc_provider = roc_auc_score(labels_p, scores_p)
                       ap_provider = average_precision_score(labels_p, scores_p)

                if 'member' in anomaly_scores_dict and 'member' in gt_labels:
                    scores_m = anomaly_scores_dict['member'].cpu().numpy()
                    labels_m = gt_labels['member'].cpu().numpy()
                    if len(np.unique(labels_m)) > 1: # Check if both classes exist
                       auc_member = roc_auc_score(labels_m, scores_m)
                       ap_member = average_precision_score(labels_m, scores_m)

                history['val_auc_provider'].append(auc_provider)
                history['val_ap_provider'].append(ap_provider)
                history['val_auc_member'].append(auc_member)
                history['val_ap_member'].append(ap_member)

                print(f"Epoch {epoch+1}/{num_epochs} | "
                      f"Train Loss: {total_loss_tr.item():.4f} | "
                      f"Val Loss: {total_loss_val.item():.4f} | "
                      f"Val AUC(P/M): {auc_provider:.4f}/{auc_member:.4f} | "
                      f"Val AP(P/M): {ap_provider:.4f}/{ap_member:.4f}")
        elif (epoch + 1) % 10 == 0: # Print training loss less frequently if not validating
             print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {total_loss_tr.item():.4f}")


    end_time = time.time()
    print(f"Training & Validation finished in {end_time - start_time:.2f} seconds.")

    # Plotting
    if plot:
        num_plots = 3 # Train/Val Loss, Attr Loss, Struct Loss + Metrics
        fig, axes = plt.subplots(num_plots, 1, figsize=(10, 5 * num_plots), sharex=True)

        # Plot Total Loss
        axes[0].plot(history['train_loss'], label='Train Total Loss')
        val_epochs = range(eval_freq - 1, num_epochs, eval_freq) # X-axis for validation points
        axes[0].plot(val_epochs, history['val_loss'], label='Validation Total Loss', marker='o')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Total Loss')
        axes[0].legend()
        axes[0].grid(True)

        # Plot Loss Components
        axes[1].plot(history['train_loss_attr'], label='Train Attr Loss', linestyle='--')
        axes[1].plot(history['train_loss_struct'], label='Train Struct Loss', linestyle=':')
        axes[1].plot(val_epochs, history['val_loss_attr'], label='Val Attr Loss', marker='x')
        axes[1].plot(val_epochs, history['val_loss_struct'], label='Val Struct Loss', marker='+')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Loss Components')
        axes[1].legend()
        axes[1].grid(True)

        # Plot AUC/AP Metrics
        axes[2].plot(val_epochs, history['val_auc_provider'], label='Val AUC Provider', marker='s')
        axes[2].plot(val_epochs, history['val_ap_provider'], label='Val AP Provider', marker='d')
        axes[2].plot(val_epochs, history['val_auc_member'], label='Val AUC Member', marker='^')
        axes[2].plot(val_epochs, history['val_ap_member'], label='Val AP Member', marker='v')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Metric Score')
        axes[2].set_title('Validation Anomaly Detection Metrics')
        axes[2].legend()
        axes[2].grid(True)
        axes[2].set_ylim(bottom=0) # Metrics are usually >= 0

        plt.tight_layout()
        plt.show()

    return history


# New functions for inductive self-supervised training

import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
from torch_geometric.utils import negative_sampling
from torch_geometric.data import HeteroData
import time
from torch_scatter import scatter_mean, scatter_add # Assuming you need these elsewhere
from typing import Optional, Tuple, Dict # Import necessary types
import numpy as np # Keep for potential future use if needed

# Assuming BipartiteGraphAutoEncoder_ReportBased is importable or defined above

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
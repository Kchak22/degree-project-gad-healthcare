# --- Training Function based on Report ---
import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
from torch_geometric.utils import negative_sampling
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

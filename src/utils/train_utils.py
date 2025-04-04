# --- Training Function based on Report ---
import torch
import torch.nn.functional as F
import torch.nn as nn   
import matplotlib.pyplot as plt
from torch_geometric.utils import negative_sampling
from torch_geometric.data import HeteroData
from torch_geometric.utils import to_dense_batch, to_undirected
from torch_geometric.utils import add_self_loops
from torch_geometric.utils import remove_self_loops
from torch_geometric.utils import to_undirected 

def train_model_report_based(model, data, num_epochs, optimizer,
                              lambda_attr=1.0, lambda_struct=0.5, k_neg_samples=5,
                              target_edge_type=('provider', 'to', 'member'), plot=True):
    """
    Trains the BipartiteGraphAutoEncoder_ReportBased model.

    Args:
        model: The BipartiteGraphAutoEncoder_ReportBased instance.
        data: The HeteroData object containing graph data.
        num_epochs: Number of training epochs.
        optimizer: PyTorch optimizer instance.
        lambda_attr: Weight for attribute reconstruction loss.
        lambda_struct: Weight for structural reconstruction loss.
        k_neg_samples: Number of negative samples per positive edge.
        target_edge_type: The edge type for structure reconstruction.
        plot: Whether to plot the loss curve.
    """
    model.train()
    loss_history = []
    device = next(model.parameters()).device # Get device model is on
    data = data.to(device) # Ensure data is on the same device

    num_nodes_provider = data['provider'].num_nodes
    num_nodes_member = data['member'].num_nodes

    print(f"Starting training for {num_epochs} epochs...")
    print(f"Lambda Attr: {lambda_attr}, Lambda Struct: {lambda_struct}, K Neg Samples: {k_neg_samples}")
    print(f"Provider nodes: {num_nodes_provider}, Member nodes: {num_nodes_member}")
    
    pos_edge_index = data[target_edge_type].edge_index
    print(f"Positive edges ('{target_edge_type[0]}' to '{target_edge_type[2]}'): {pos_edge_index.shape[1]}")


    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # Forward pass: Get reconstructed features and latent embeddings
        x_hat_m, x_hat_p, z_m, z_p = model(data)
        z_dict = {'member': z_m, 'provider': z_p} # Dictionary for structure decoder

        # 1. Attribute Reconstruction Loss (L_attr - Eq 3.19)
        loss_attr_m = F.mse_loss(x_hat_m, data['member'].x)
        loss_attr_p = F.mse_loss(x_hat_p, data['provider'].x)
        loss_attr = loss_attr_m + loss_attr_p # Or average if preferred

        # 2. Structural Reconstruction Loss (L_struct - Eq 3.20)
        # --- Negative Sampling (Eq 3.17) ---
        # Ensure negative sampling is done correctly for bipartite graphs
        # Need total number of nodes for sampling space if using default PyG negative_sampling
        # However, for bipartite, sampling should respect types.
        # Sample members for a given provider (that are not connected)
        # Sample providers for a given member (that are not connected)
        
        # Simpler approach: Use PyG's negative_sampling which samples random pairs,
        # potentially creating provider-provider or member-member edges if not careful.
        # A more correct bipartite negative sampling might be needed for strictness.
        # For now, let's use the standard one, assuming it samples indices correctly
        # across the union of nodes, which might not be ideal.
        
        # *Corrected Bipartite Negative Sampling Idea*
        # Sample k negative destinations (members) for each source (provider) in positive edges
        neg_edge_index = negative_sampling(
            edge_index=pos_edge_index,
            num_nodes=(num_nodes_provider, num_nodes_member), # Specify node counts per type
            num_neg_samples=pos_edge_index.size(1) * k_neg_samples # Sample k per positive edge
            # method='sparse' # Or 'binary' - sparse is usually faster for large graphs
        )
        # negative_sampling returns shape [2, num_neg_samples], potentially needing adjustment if k is not 1.
        # If num_neg_samples is specified as num_pos * k, it should return the right number.

        # --- Calculate Logits for Positive and Negative Edges ---
        pos_logits = model.decode_structure(z_dict, pos_edge_index)
        neg_logits = model.decode_structure(z_dict, neg_edge_index)

        # --- Combine Logits and Create Labels ---
        logits = torch.cat([pos_logits, neg_logits], dim=0)
        pos_labels = torch.ones_like(pos_logits)
        neg_labels = torch.zeros_like(neg_logits)
        labels = torch.cat([pos_labels, neg_labels], dim=0)

        # --- Calculate BCE Loss (Eq 3.20) ---
        # Use BCEWithLogitsLoss for numerical stability
        loss_struct = F.binary_cross_entropy_with_logits(logits, labels)

        # 3. Total Loss (Eq 3.18)
        total_loss = lambda_attr * loss_attr + lambda_struct * loss_struct

        # Backward pass and optimization
        total_loss.backward()
        optimizer.step()

        loss_history.append(total_loss.item())
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss.item():.4f} '
                  f'(Attr: {loss_attr.item():.4f}, Struct: {loss_struct.item():.4f})')

    print("Training finished.")

    # Plotting the loss
    if plot:
        plt.figure(figsize=(10, 5))
        plt.plot(loss_history, label='Total Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Epochs')
        plt.legend()
        plt.grid(True)
        plt.show()

    return loss_history

from sklearn.metrics import roc_auc_score, average_precision_score
import time


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



from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, f1_score
import numpy as np
import torch

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


# %% [markdown]
# ### Main Evaluation Function

# %%
import pandas as pd # For displaying results nicely

@torch.no_grad() # Disable gradient calculations for evaluation
def evaluate_model(model, data_split, train_data_struct, gt_labels,
                    lambda_attr=1.0, lambda_struct=0.5,
                    target_edge_type=('provider', 'to', 'member'),
                    k_list=[50, 100, 200]): # Define K values
    """
    Evaluates the trained model on a given data split (validation or test).

    Args:
        model: The trained model instance.
        data_split: The HeteroData object for the split (e.g., val_data, test_data).
                    Contains edge_label and edge_label_index for structural eval if needed,
                    but node evaluation primarily uses gt_labels.
        train_data_struct: The HeteroData object used for training (provides graph structure
                           for message passing during inference).
        gt_labels: Dictionary of ground truth anomaly labels {'provider': tensor, 'member': tensor}.
        lambda_attr: Weight used for attribute term in anomaly score.
        lambda_struct: Weight used for structural term in anomaly score.
        target_edge_type: The edge type used for structural score component.
        k_list: List of K values for P@K, R@K.

    Returns:
        dict: A dictionary containing evaluation metrics for provider and member nodes.
    """
    model.eval() # Set model to evaluation mode
    device = next(model.parameters()).device

    # Ensure necessary data is on the correct device
    train_data_struct = train_data_struct.to(device)
    # data_split = data_split.to(device) # Not strictly needed if only using train_data_struct for forward pass
    gt_labels_dev = {k: v.to(device) for k, v in gt_labels.items()}


    print("Running forward pass for evaluation...")
    # Perform forward pass using the *training graph structure*
    # to get reconstructions and embeddings consistent with training context.
    x_hat_m, x_hat_p, z_m, z_p = model(train_data_struct)

    x_hat_dict = {'member': x_hat_m, 'provider': x_hat_p}
    x_dict = {'member': train_data_struct['member'].x, 'provider': train_data_struct['provider'].x}
    z_dict = {'member': z_m, 'provider': z_p}

    print("Calculating anomaly scores...")
    # Calculate anomaly scores for all nodes
    # Use the positive edges from the training structure for score calculation consistency
    pos_edge_index_struct = train_data_struct[target_edge_type].edge_index
    anomaly_scores_dict = calculate_node_anomaly_scores(
        x_hat_dict, x_dict, z_dict,
        pos_edge_index_struct, # Use train edges for structural component of score
        model, lambda_attr, lambda_struct
    )

    results = {}
    print("Computing metrics...")
    # Evaluate Provider nodes
    if 'provider' in anomaly_scores_dict and 'provider' in gt_labels:
        scores_p = anomaly_scores_dict['provider'].cpu().numpy()
        labels_p = gt_labels_dev['provider'].cpu().numpy() # Use gt_labels from injection
        print(f"  Providers: {len(scores_p)} nodes, {int(np.sum(labels_p))} anomalies.")
        if len(scores_p) > 0:
             results['provider'] = compute_evaluation_metrics(scores_p, labels_p, k_list)
        else:
             results['provider'] = {}
    else:
        print("  Skipping provider metrics (scores or labels missing).")
        results['provider'] = {}

    # Evaluate Member nodes
    if 'member' in anomaly_scores_dict and 'member' in gt_labels:
        scores_m = anomaly_scores_dict['member'].cpu().numpy()
        labels_m = gt_labels_dev['member'].cpu().numpy() # Use gt_labels from injection
        print(f"  Members: {len(scores_m)} nodes, {int(np.sum(labels_m))} anomalies.")
        if len(scores_m) > 0:
            results['member'] = compute_evaluation_metrics(scores_m, labels_m, k_list)
        else:
            results['member'] = {}
    else:
         print("  Skipping member metrics (scores or labels missing).")
         results['member'] = {}

    return results

from torch_scatter import scatter_add, scatter_mean # Use scatter_mean for average error per node



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
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import torch
from torch_geometric.utils import degree
from collections import Counter
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from torch_geometric.data import HeteroData # Import HeteroData

def analyze_injected_anomalies(data: HeteroData, gt_node_labels: dict, anomaly_tracking: dict,
                               original_data: HeteroData = None, # Optional: for feature comparison
                               gt_edge_labels: dict = None, # Optional: for edge stats
                               plot_features: bool = True, pca_components: int = 2):
    """
    Analyzes the characteristics of injected anomalies in the HeteroData object,
    compatible with the output of `inject_simplified_anomalies`.

    Args:
        data: The HeteroData object *after* anomaly injection.
        gt_node_labels: Dictionary of ground truth node labels {'provider': tensor, 'member': tensor}.
        anomaly_tracking: Dictionary tracking anomaly types per node.
        original_data: (Optional) The HeteroData object *before* injection for comparison.
        gt_edge_labels: (Optional) Dictionary of ground truth edge labels per edge type.
        plot_features: Whether to generate PCA/t-SNE plots of features.
        pca_components: Number of components for PCA plot.
    """
    print("--- Anomaly Injection Analysis ---")

    results = {}

    # --- Node Analysis ---
    for node_type in data.node_types:
        print(f"\n--- Analyzing Node Type: {node_type} ---")
        if node_type not in gt_node_labels:
            print("  No ground truth node labels found.")
            continue

        labels = gt_node_labels[node_type].cpu().numpy()
        num_nodes = len(labels)
        num_anomalies = int(labels.sum())
        perc_anomalies = (num_anomalies / num_nodes) * 100 if num_nodes > 0 else 0

        print(f"  Total Nodes: {num_nodes}")
        print(f"  Anomalous Nodes: {num_anomalies} ({perc_anomalies:.2f}%)")

        results[node_type] = {'total_nodes': num_nodes, 'anomalous_nodes': num_anomalies}

        if num_anomalies > 0:
            # Anomaly Type Distribution (using the types set by inject_simplified_anomalies)
            type_counter = Counter()
            anom_indices = np.where(labels == 1)[0]
            node_specific_tracking = anomaly_tracking.get(node_type, {}) # Get tracking for this node type

            for idx in anom_indices:
                 if idx in node_specific_tracking:
                      type_counter.update(list(node_specific_tracking[idx]))
                 else:
                      type_counter.update(['unknown']) # If node is labeled anomalous but not tracked

            print("  Anomaly Type Component Counts (Node Level):")
            if not type_counter:
                 print("    No specific anomaly types tracked for anomalous nodes.")
            else:
                 for type_name, count in type_counter.most_common():
                     print(f"    - {type_name}: {count}")
            results[node_type]['anomaly_type_components'] = dict(type_counter) # Store detailed components

            # Also count primary injection types ('structural', 'attribute', 'combined')
            primary_type_counter = Counter()
            for idx in anom_indices:
                 if idx in node_specific_tracking:
                     # Check for the primary types assigned during injection
                     if 'structural' in node_specific_tracking[idx]: primary_type_counter['structural'] += 1
                     if 'attribute' in node_specific_tracking[idx]: primary_type_counter['attribute'] += 1
                     if 'combined' in node_specific_tracking[idx]: primary_type_counter['combined'] += 1
            print("  Primary Injection Type Counts (Node Level):")
            for type_name, count in primary_type_counter.most_common():
                 print(f"    - {type_name}: {count}")
            results[node_type]['primary_anomaly_types'] = dict(primary_type_counter)


            # Degree Analysis
            node_degrees = {}
            num_nodes_dict = {nt: data[nt].num_nodes for nt in data.node_types} # Get counts for degree calc

            for edge_type in data.edge_types:
                src, _, dst = edge_type
                if not hasattr(data[edge_type], 'edge_index'): continue # Skip if edge type has no index
                edge_index = data[edge_type].edge_index

                if node_type == src: # Out-degree
                    num_src_nodes = num_nodes_dict[src]
                    # Check if num_src_nodes is valid before calling degree
                    if num_src_nodes > 0:
                         deg = degree(edge_index[0], num_nodes=num_src_nodes)
                         node_degrees[f'out_{dst}'] = deg.cpu().numpy()
                if node_type == dst: # In-degree
                     num_dst_nodes = num_nodes_dict[dst]
                     # Check if num_dst_nodes is valid
                     if num_dst_nodes > 0:
                          deg = degree(edge_index[1], num_nodes=num_dst_nodes)
                          node_degrees[f'in_{src}'] = deg.cpu().numpy()


            if node_degrees:
                 print("  Average Degree (Normal vs Anomalous):")
                 degree_stats = {}
                 for deg_type, degrees in node_degrees.items():
                      # Ensure degrees array matches labels length (should if num_nodes was correct)
                      if len(degrees) != len(labels):
                           print(f"    Warning: Degree array length ({len(degrees)}) mismatch with labels ({len(labels)}) for {deg_type}. Skipping.")
                           continue
                      avg_deg_normal = degrees[labels == 0].mean() if (labels == 0).sum() > 0 else 0
                      avg_deg_anom = degrees[labels == 1].mean() # Already checked num_anomalies > 0
                      print(f"    - Avg. Degree ({deg_type}): Normal={avg_deg_normal:.2f}, Anomalous={avg_deg_anom:.2f}")
                      degree_stats[deg_type] = {'normal': avg_deg_normal, 'anomalous': avg_deg_anom}
                 results[node_type]['degree_stats'] = degree_stats


            # Feature Analysis Plot
            if plot_features and hasattr(data[node_type], 'x') and data[node_type].x.numel() > 0:
                print(f"  Generating Feature Plot for {node_type}...")
                features = data[node_type].x.cpu().numpy()
                if features.shape[0] != len(labels):
                     print(f"    Warning: Feature shape {features.shape} mismatch with labels length {len(labels)}. Skipping plot.")
                else:
                    pca_comps_actual = min(pca_components, features.shape[1], features.shape[0]) # Adjust components if needed
                    if pca_comps_actual < 1:
                         print("    Cannot perform PCA with < 1 component.")
                    elif pca_comps_actual == 1:
                         print(f"    Applying PCA ({pca_comps_actual} component)...")
                         pca = PCA(n_components=pca_comps_actual, random_state=42)
                         features_reduced = pca.fit_transform(features)
                         plt.figure(figsize=(8, 2))
                         sns.histplot(data=pd.DataFrame({'PC1': features_reduced[:, 0], 'Anomaly': labels}), x='PC1', hue='Anomaly', palette='coolwarm', kde=True)
                         plt.title(f'{node_type.capitalize()} Features PCA (1 Component)')
                         plt.xlabel('Principal Component 1')
                         plt.grid(True)
                         plt.show()
                    else: # pca_components >= 2
                        print(f"    Applying PCA ({pca_comps_actual} components)...")
                        pca = PCA(n_components=pca_comps_actual, random_state=42)
                        features_reduced = pca.fit_transform(features)
                        plt.figure(figsize=(8, 6))
                        scatter = plt.scatter(features_reduced[:, 0], features_reduced[:, 1], c=labels, cmap='coolwarm', alpha=0.6, s=10)
                        plt.title(f'{node_type.capitalize()} Features PCA ({pca_comps_actual} Components)')
                        plt.xlabel('Principal Component 1')
                        plt.ylabel('Principal Component 2')
                        plt.legend(handles=scatter.legend_elements()[0], labels=['Normal', 'Anomaly'])
                        plt.grid(True)
                        plt.show()

    # --- Edge Analysis (Optional) ---
    if gt_edge_labels:
        print("\n--- Analyzing Edges ---")
        edge_results = {}
        for edge_type, edge_labels in gt_edge_labels.items():
            if edge_type not in data.edge_types:
                print(f"  Edge type {edge_type} not found in modified data, skipping.")
                continue

            labels_np = edge_labels.cpu().numpy()
            num_edges = len(labels_np)
            num_anomalies = int(labels_np.sum())
            perc_anomalies = (num_anomalies / num_edges) * 100 if num_edges > 0 else 0

            print(f"  Edge Type: {edge_type}")
            print(f"    Total Edges: {num_edges}")
            print(f"    Anomalous Edges (Injected): {num_anomalies} ({perc_anomalies:.2f}%)")
            edge_results[str(edge_type)] = {'total_edges': num_edges, 'anomalous_edges': num_anomalies}
        results['edges'] = edge_results


    print("\n--- Analysis Complete ---")
    # Combine results into a more structured dictionary or DataFrame
    summary_dict = {}
    for node_type, stats in results.items():
        if node_type != 'edges':
             summary_dict[f"{node_type}_total"] = stats.get('total_nodes', 0)
             summary_dict[f"{node_type}_anomalous"] = stats.get('anomalous_nodes', 0)
    if 'edges' in results:
         for edge_type_str, stats in results['edges'].items():
              summary_dict[f"{edge_type_str}_total"] = stats.get('total_edges', 0)
              summary_dict[f"{edge_type_str}_anomalous"] = stats.get('anomalous_edges', 0)

    return pd.Series(summary_dict).to_frame('Count') # Return summary as DataFrame
from pyvis.network import Network
import networkx as nx
from torch_geometric.utils import to_networkx
import random

def visualize_anomaly_graph(data, gt_labels=None, anomaly_tracking=None, sample_size=100, filename='anomaly_graph.html'):
    """
    Visualize the healthcare graph with anomalies highlighted.
    
    Args:
        data: HeteroData object with the graph data
        gt_labels: Dictionary mapping node type to ground truth labels
        anomaly_tracking: Dictionary mapping node type to anomaly type tracking
        sample_size: Number of nodes to sample for visualization
        filename: Output HTML file to save the visualization
    """
 
    # Create a NetworkX graph
    G = nx.Graph()
    
    # Get nodes to visualize (sample a subset if graph is large)
    provider_indices = range(data['provider'].num_nodes)
    member_indices = range(data['member'].num_nodes)
    
    if data['provider'].num_nodes + data['member'].num_nodes > sample_size * 2:
        # Sample nodes, ensuring we include anomalies
        if gt_labels is not None:
            # Get anomalous indices
            provider_anomaly_indices = gt_labels['provider'].nonzero().squeeze().tolist()
            member_anomaly_indices = gt_labels['member'].nonzero().squeeze().tolist()
            
            # Convert to list if single element
            if not isinstance(provider_anomaly_indices, list):
                provider_anomaly_indices = [provider_anomaly_indices]
            if not isinstance(member_anomaly_indices, list):
                member_anomaly_indices = [member_anomaly_indices]
                
            # Sample additional normal nodes
            provider_normal_indices = [i for i in range(data['provider'].num_nodes) 
                                     if i not in provider_anomaly_indices]
            member_normal_indices = [i for i in range(data['member'].num_nodes) 
                                   if i not in member_anomaly_indices]
            
            # Combine anomalous nodes with a sample of normal nodes
            sampled_provider_normal = random.sample(
                provider_normal_indices, 
                min(sample_size - len(provider_anomaly_indices), len(provider_normal_indices))
            )
            sampled_member_normal = random.sample(
                member_normal_indices, 
                min(sample_size - len(member_anomaly_indices), len(member_normal_indices))
            )
            
            provider_indices = provider_anomaly_indices + sampled_provider_normal
            member_indices = member_anomaly_indices + sampled_member_normal
        else:
            # Random sampling
            provider_indices = random.sample(range(data['provider'].num_nodes), 
                                           min(sample_size, data['provider'].num_nodes))
            member_indices = random.sample(range(data['member'].num_nodes), 
                                         min(sample_size, data['member'].num_nodes))
    
    # Add provider nodes
    for i in provider_indices:
        node_type = 'provider'
        node_id = f"{node_type}_{i}"
        
        # Set node properties
        color = "#66CCFF"  # Default blue for providers
        title = f"Provider {i}"
        is_anomaly = False
        
        if gt_labels is not None and gt_labels[node_type][i] == 1:
            color = "#FF6666"  # Red for anomalies
            is_anomaly = True
            
            # Add anomaly type information to the title
            if anomaly_tracking and i in anomaly_tracking[node_type]:
                anomaly_types = ', '.join(anomaly_tracking[node_type][i])
                title += f"\nAnomaly Types: {anomaly_types}"
        
        G.add_node(node_id, label=f"P{i}", color=color, title=title, 
                  group=node_type, is_anomaly=is_anomaly)
    
    # Add member nodes
    for i in member_indices:
        node_type = 'member'
        node_id = f"{node_type}_{i}"
        
        # Set node properties
        color = "#99CC99"  # Default green for members
        title = f"Member {i}"
        is_anomaly = False
        
        if gt_labels is not None and gt_labels[node_type][i] == 1:
            color = "#FFCC66"  # Orange for anomalies
            is_anomaly = True
            
            # Add anomaly type information to the title
            if anomaly_tracking and i in anomaly_tracking[node_type]:
                anomaly_types = ', '.join(anomaly_tracking[node_type][i])
                title += f"\nAnomaly Types: {anomaly_types}"
        
        G.add_node(node_id, label=f"M{i}", color=color, title=title, 
                  group=node_type, is_anomaly=is_anomaly)
    
    # Add edges
    edge_index = data['provider', 'to', 'member'].edge_index
    edge_attr = data['provider', 'to', 'member'].edge_attr
    
    provider_set = set(f"provider_{i}" for i in provider_indices)
    member_set = set(f"member_{i}" for i in member_indices)
    
    for i in range(edge_index.size(1)):
        provider_idx = edge_index[0, i].item()
        member_idx = edge_index[1, i].item()
        
        provider_id = f"provider_{provider_idx}"
        member_id = f"member_{member_idx}"
        
        # Only add edges between nodes that are in our visualization
        if provider_id in provider_set and member_id in member_set:
            # Get edge weight (number of claims)
            weight = edge_attr[i].item()
            
            # Determine edge properties
            width = max(1, min(5, weight / 10))  # Scale edge width, but cap it
            title = f"Claims: {weight:.1f}"
            
            # Check if either node is anomalous
            provider_anomaly = gt_labels is not None and gt_labels['provider'][provider_idx] == 1
            member_anomaly = gt_labels is not None and gt_labels['member'][member_idx] == 1
            
            color = "#666666"  # Default gray
            if provider_anomaly or member_anomaly:
                color = "#FF9999"  # Pink for edges connected to anomalies
                
            G.add_edge(provider_id, member_id, width=width, color=color, title=title, value=weight)
    
    # Create a PyVis network from the NetworkX graph
    net = Network(height="800px", width="100%", notebook=True, directed=False)
    
    # Configure physics for better layout
    net.barnes_hut(spring_length=200, spring_strength=0.01, damping=0.09)
    
    # Add the graph
    net.from_nx(G)
    
    # Add legend as node
    net.add_node("legend_normal_provider", label="Normal Provider", color="#66CCFF", shape="box", size=15)
    net.add_node("legend_anomaly_provider", label="Anomalous Provider", color="#FF6666", shape="box", size=15)
    net.add_node("legend_normal_member", label="Normal Member", color="#99CC99", shape="box", size=15)
    net.add_node("legend_anomaly_member", label="Anomalous Member", color="#FFCC66", shape="box", size=15)
    
    # Fix legend positions
    for i, node_id in enumerate(["legend_normal_provider", "legend_anomaly_provider", 
                                "legend_normal_member", "legend_anomaly_member"]):
        net.get_node(node_id)['fixed'] = True
        net.get_node(node_id)['physics'] = False
        net.get_node(node_id)['x'] = -500
        net.get_node(node_id)['y'] = -300 + i * 50
    
    # Set options
    net.set_options("""
    {
      "nodes": {
        "shape": "dot",
        "size": 15,
        "font": {
          "size": 15
        }
      },
      "edges": {
        "smooth": {
          "enabled": false
        },
        "arrows": {
          "to": {
            "enabled": false
          }
        }
      },
      "physics": {
        "stabilization": {
          "iterations": 100
        }
      }
    }
    """)
    
    # Save and show
    net.save_graph(filename)
    
    return net


def visualize_anomaly_scores(results, anomaly_tracking=None, top_n=50):
    """
    Visualize anomaly scores from model results, highlighting different anomaly types.
    
    Args:
        results: Dictionary with anomaly detection results from evaluate_anomaly_detection
        anomaly_tracking: Dictionary with anomaly type information
        top_n: Number of top anomalies to visualize
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    for node_type in results:
        scores = results[node_type]['scores']
        labels = results[node_type]['labels']
        
        # Create a dataframe for easier manipulation
        df = pd.DataFrame({
            'node_idx': range(len(scores)),
            'anomaly_score': scores,
            'is_anomaly': labels
        })
        
        # Sort by anomaly score in descending order
        df_sorted = df.sort_values('anomaly_score', ascending=False).reset_index(drop=True)
        
        # Take top N for visualization
        df_top = df_sorted.head(top_n)
        
        # Add anomaly type information if available
        if anomaly_tracking:
            df_top['anomaly_type'] = df_top['node_idx'].apply(
                lambda idx: '/'.join(anomaly_tracking[node_type].get(idx, ['normal'])) 
                if idx in anomaly_tracking[node_type] else 'normal'
            )
        else:
            df_top['anomaly_type'] = df_top['is_anomaly'].apply(
                lambda x: 'anomaly' if x == 1 else 'normal'
            )
        
        # Create the plot
        plt.figure(figsize=(14, 8))
        
        # Use different colors for different anomaly types
        colors = []
        for _, row in df_top.iterrows():
            if 'normal' in row['anomaly_type']:
                colors.append('blue')
            elif 'structural' in row['anomaly_type'] and 'feature' in row['anomaly_type']:
                colors.append('purple')  # Mixed type
            elif 'structural' in row['anomaly_type']:
                colors.append('green')
            elif 'feature' in row['anomaly_type']:
                colors.append('red')
            elif 'healthcare' in row['anomaly_type']:
                colors.append('orange')
            else:
                colors.append('gray')
        
        # Plot top anomaly scores
        plt.bar(range(len(df_top)), df_top['anomaly_score'], color=colors)
        
        # Add a horizontal line at the detection threshold
        if len(df) > 100:
            threshold = np.percentile(scores, 95)  # 95th percentile
            plt.axhline(y=threshold, color='r', linestyle='--', alpha=0.5, 
                       label=f'95th Percentile Threshold: {threshold:.2f}')
        
        # Customize the plot
        plt.title(f'Top {top_n} Anomaly Scores for {node_type.capitalize()} Nodes')
        plt.xlabel('Rank')
        plt.ylabel('Anomaly Score')
        plt.xticks(range(len(df_top)), [f"{idx} ({row['anomaly_type']})" for idx, row in df_top.iterrows()], 
                  rotation=90, fontsize=8)
        
        # Add legend for anomaly types
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Normal'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Structural'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Feature'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', markersize=10, label='Mixed'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='Healthcare')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.show()
        
        # Also create a ROC curve
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(labels, scores)
        
        plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, marker='.')
        plt.title(f'ROC Curve for {node_type.capitalize()} Nodes (AUC: {results[node_type]["auc"]:.4f})')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.grid(True)
        plt.plot([0, 1], [0, 1], 'r--')
        plt.tight_layout()
        plt.show()

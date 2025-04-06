from pyvis.network import Network
import networkx as nx
import random
import torch
from torch_geometric.data import HeteroData



def visualize_anomalous_subgraph(data: HeteroData, gt_node_labels: dict, anomaly_tracking: dict,
                                 num_anomalous_providers: int = 10,
                                 num_anomalous_members: int = 20,
                                 context_neighbors: int = 2, # How many normal neighbors to show per anomaly
                                 filename='anomalous_subgraph.html'):
    """
    Visualizes a subgraph centered around anomalous nodes.

    Args:
        data: HeteroData object (ideally after anomaly injection).
        gt_node_labels: Dictionary of ground truth node labels (0/1).
        anomaly_tracking: Dictionary with anomaly types for anomalous nodes.
        num_anomalous_providers: Number of anomalous providers to sample.
        num_anomalous_members: Number of anomalous members to sample.
        context_neighbors: Max number of normal neighbors to include per sampled anomaly.
        filename: Output HTML filename.
    """
    print("Visualizing anomalous subgraph...")
    provider_labels = gt_node_labels.get('provider', torch.tensor([]))
    member_labels = gt_node_labels.get('member', torch.tensor([]))

    anom_provider_indices = torch.where(provider_labels == 1)[0].tolist()
    anom_member_indices = torch.where(member_labels == 1)[0].tolist()
    normal_provider_indices = torch.where(provider_labels == 0)[0].tolist()
    normal_member_indices = torch.where(member_labels == 0)[0].tolist()

    # Sample anomalous nodes
    sampled_anom_providers = random.sample(anom_provider_indices, min(num_anomalous_providers, len(anom_provider_indices)))
    sampled_anom_members = random.sample(anom_member_indices, min(num_anomalous_members, len(anom_member_indices)))

    nodes_to_include = {'provider': set(sampled_anom_providers), 'member': set(sampled_anom_members)}
    edge_index_ptm = data['provider', 'to', 'member'].edge_index.cpu()

    # Add context neighbors
    neighbors_to_add = {'provider': set(), 'member': set()}
    edges_in_subgraph = set()

    # Neighbors of anomalous providers
    for p_idx in sampled_anom_providers:
        connected_members = edge_index_ptm[1, edge_index_ptm[0] == p_idx].tolist()
        edges_in_subgraph.update([(p_idx, m) for m in connected_members])
        normal_neighbors = [m for m in connected_members if m in normal_member_indices]
        neighbors_to_add['member'].update(random.sample(normal_neighbors, min(context_neighbors, len(normal_neighbors))))
        neighbors_to_add['member'].update([m for m in connected_members if m in sampled_anom_members]) # Ensure connected anomalies are included

    # Neighbors of anomalous members
    for m_idx in sampled_anom_members:
        connected_providers = edge_index_ptm[0, edge_index_ptm[1] == m_idx].tolist()
        edges_in_subgraph.update([(p, m_idx) for p in connected_providers])
        normal_neighbors = [p for p in connected_providers if p in normal_provider_indices]
        neighbors_to_add['provider'].update(random.sample(normal_neighbors, min(context_neighbors, len(normal_neighbors))))
        neighbors_to_add['provider'].update([p for p in connected_providers if p in sampled_anom_providers]) # Ensure connected anomalies are included

    nodes_to_include['provider'].update(neighbors_to_add['provider'])
    nodes_to_include['member'].update(neighbors_to_add['member'])

    # Build NetworkX Graph
    G = nx.Graph()
    node_colors = {}
    node_sizes = {}

    # Node color/size/tooltip logic
    for node_type, indices in nodes_to_include.items():
        labels = gt_node_labels[node_type]
        tracking = anomaly_tracking.get(node_type, {})
        for i in indices:
            node_id = f"{node_type}_{i}"
            is_anomalous = labels[i].item() == 1
            color = "red" if is_anomalous else ("blue" if node_type == 'provider' else "green")
            size = 20 if is_anomalous else 10
            extra = ""
            if is_anomalous and i in tracking:
                extra = f"\nAnomalies: {', '.join(tracking[i])}"
            title = f"{node_type.capitalize()} {i}{extra}"
            G.add_node(node_id, label=f"{node_type[0].upper()}{i}", title=title, group=node_type)
            node_colors[node_id] = color
            node_sizes[node_id] = size

    # Add edges connecting included nodes
    edge_attr_ptm = getattr(data['provider', 'to', 'member'], 'edge_attr', None)
    edge_labels_ptm = getattr(data['provider', 'to', 'member'], 'y', None) # Get edge labels if stored in 'y'

    edge_colors = {}
    for i in range(edge_index_ptm.shape[1]):
        p_idx, m_idx = edge_index_ptm[:, i].tolist()
        if p_idx in nodes_to_include['provider'] and m_idx in nodes_to_include['member']:
            p_node_id = f"provider_{p_idx}"
            m_node_id = f"member_{m_idx}"
            edge_id = (p_node_id, m_node_id)

            edge_title = ""
            if edge_attr_ptm is not None:
                edge_title += f"Attr: {edge_attr_ptm[i].item():.2f}"
            is_edge_anomalous = edge_labels_ptm is not None and edge_labels_ptm[i].item() == 1

            color = "magenta" # Default edge color
            if is_edge_anomalous:
                color = "red" # Highlight injected edges
                edge_title += " (Injected Anomalous Edge)"
            elif gt_node_labels['provider'][p_idx] == 1 and gt_node_labels['member'][m_idx] == 1:
                color = "orange" # Edge between two anomalous nodes (likely part of block)


            G.add_edge(p_node_id, m_node_id, title=edge_title, color=color, width=2 if is_edge_anomalous else 1)
            # We store color directly in edge attributes for pyvis

    # Create Pyvis Network
    net = Network(height="700px", width="100%", notebook=True, directed=False) # Use directed=True if needed
    net.from_nx(G)

    # Apply node styles (size might need explicit setting)
    for node in net.nodes:
        node['color'] = node_colors[node['id']]
        node['size'] = node_sizes[node['id']]

    # Apply edge styles (color is directly set above)

    print(f"Saving visualization to {filename}")
    net.save_graph(filename)
    return net

def visualize_graph_with_anomaly_info(
    data: HeteroData,
    gt_node_labels: dict,
    anomaly_tracking: dict,
    gt_edge_labels: dict = None, # Pass the dict returned by injection
    target_edge_type = ('provider', 'to', 'member'), # Edge type to visualize
    sample_size_provider: int = 50, # How many providers to start sampling from
    max_members_per_provider: int = 5, # Max members shown per sampled provider
    filename='general_anomaly_graph_viz.html'
    ):
    """
    Visualize a sampled bipartite subgraph highlighting node and edge anomalies.

    Args:
        data (HeteroData): The graph data (potentially modified by injection).
                           Expects node features implicitly and edge features optionally.
                           Can also check for edge labels stored as data[edge_type].y.
        gt_node_labels (dict): Node anomaly labels (0/1) per type.
        anomaly_tracking (dict): Detailed anomaly type info per anomalous node.
        gt_edge_labels (dict, optional): Edge anomaly labels (0/1) per type.
                                           If None, tries to use data[edge_type].y.
        target_edge_type (tuple): The primary edge type ('src', 'rel', 'dst') to visualize.
        sample_size_provider (int): Number of provider nodes to sample initially.
        max_members_per_provider (int): Max members shown per provider for density control.
        filename (str): Output HTML filename.

    Returns:
        pyvis.network.Network: The generated pyvis network object.
    """
    print(f"Generating visualization for {filename}...")

    # --- Data Extraction ---
    provider_labels = gt_node_labels.get('provider', torch.tensor([]))
    member_labels = gt_node_labels.get('member', torch.tensor([]))
    provider_tracking = anomaly_tracking.get('provider', {})
    member_tracking = anomaly_tracking.get('member', {})

    if target_edge_type not in data.edge_types:
        print(f"Warning: Target edge type {target_edge_type} not found in data. No edges will be shown.")
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = None
        edge_y = None
    else:
        edge_index = data[target_edge_type].edge_index.cpu()
        edge_attr = getattr(data[target_edge_type], 'edge_attr', None)
        if edge_attr is not None: edge_attr = edge_attr.cpu()

        # Get edge labels either from the passed dict or the data object
        edge_y = None
        if gt_edge_labels and target_edge_type in gt_edge_labels:
            edge_y = gt_edge_labels[target_edge_type].cpu()
        elif hasattr(data[target_edge_type], 'y'):
            edge_y = data[target_edge_type].y.cpu()
            print("Using edge labels stored in data[edge_type].y")

        if edge_y is not None and edge_y.shape[0] != edge_index.shape[1]:
             print(f"Warning: Edge label length ({len(edge_y)}) mismatch with edge index length ({edge_index.shape[1]}). Edge highlighting might be incorrect.")
             # Decide how to handle: maybe disable edge highlighting?
             # edge_y = None # Option: Disable if lengths mismatch

    num_providers = data['provider'].num_nodes
    num_members = data['member'].num_nodes

    # --- Sampling Logic (Same as original simple_visualize_graph) ---
    provider_indices_sampled = random.sample(range(num_providers), min(sample_size_provider, num_providers))
    member_indices_connected = set()
    for i in range(edge_index.shape[1]):
        p_idx, m_idx = edge_index[:, i].tolist()
        if p_idx in provider_indices_sampled:
            member_indices_connected.add(m_idx)

    # Limit members based on max_members_per_provider for clarity
    member_indices_sampled = list(member_indices_connected)
    random.shuffle(member_indices_sampled)
    max_total_members = sample_size_provider * max_members_per_provider
    member_indices_sampled = member_indices_sampled[:min(len(member_indices_sampled), max_total_members)]

    sampled_nodes = {'provider': set(provider_indices_sampled), 'member': set(member_indices_sampled)}
    print(f"Sampled {len(sampled_nodes['provider'])} providers and {len(sampled_nodes['member'])} members.")

    # --- Build NetworkX Graph ---
    G = nx.Graph() # Use Graph for undirected layout, edge color indicates type/anomaly
    node_styles = {} # Store color and size

    # Add Provider Nodes
    for p_idx in sampled_nodes['provider']:
        node_id = f"provider_{p_idx}"
        is_anomalous = provider_labels[p_idx].item() == 1
        color = "red" if is_anomalous else "blue"
        size = 15 if is_anomalous else 10
        types_str = ', '.join(provider_tracking.get(p_idx, []))
        title = f"Provider {p_idx}" + (f"\nAnomalies: {types_str}" if is_anomalous else "\nStatus: Normal")
        G.add_node(node_id, label=f"P{p_idx}", title=title, group='provider', color=color, size=size)
        node_styles[node_id] = {'color': color, 'size': size}

    # Add Member Nodes
    for m_idx in sampled_nodes['member']:
        node_id = f"member_{m_idx}"
        is_anomalous = member_labels[m_idx].item() == 1
        color = "red" if is_anomalous else "green" # Changed normal member color
        size = 15 if is_anomalous else 10
        types_str = ', '.join(member_tracking.get(m_idx, []))
        title = f"Member {m_idx}" + (f"\nAnomalies: {types_str}" if is_anomalous else "\nStatus: Normal")
        G.add_node(node_id, label=f"M{m_idx}", title=title, group='member', color=color, size=size)
        node_styles[node_id] = {'color': color, 'size': size}

    # Add Edges (within the sampled nodes) with anomaly highlighting
    edge_count = 0
    for i in range(edge_index.shape[1]):
        p_idx, m_idx = edge_index[:, i].tolist()

        # Check if both endpoints are in our sampled sets
        if p_idx in sampled_nodes['provider'] and m_idx in sampled_nodes['member']:
            p_node_id = f"provider_{p_idx}"
            m_node_id = f"member_{m_idx}"

            # Determine edge style based on node and edge labels
            edge_color = "grey" # Default for normal edge between normal nodes
            edge_width = 1
            edge_title = ""

            is_edge_anomalous = edge_y is not None and i < len(edge_y) and edge_y[i].item() == 1
            is_provider_anom = provider_labels[p_idx].item() == 1
            is_member_anom = member_labels[m_idx].item() == 1

            if is_edge_anomalous:
                edge_color = "red" # Explicitly injected edge anomaly
                edge_width = 2.5
                edge_title = "(Injected Edge Anomaly)"
            elif is_provider_anom and is_member_anom:
                 edge_color = "orange" # Connects two anomalous nodes (likely part of structural block)
                 edge_width = 1.5
            elif is_provider_anom or is_member_anom:
                 edge_color = "lightcoral" # Connects one anomalous node

            # Add edge attribute info to title
            if edge_attr is not None and i < edge_attr.shape[0]:
                edge_title += f" Attr: {edge_attr[i].cpu().numpy()}" # Show full attribute vector or specific element

            # Add edge with style attributes
            G.add_edge(p_node_id, m_node_id, title=edge_title.strip(), color=edge_color, width=edge_width)
            edge_count += 1

    print(f"Added {edge_count} edges to the visualization.")

    # --- Create Pyvis Network ---
    net = Network(height="700px", width="100%", notebook=True, directed=False) # Set directed=True if needed
    net.from_nx(G)

    # Optional: Ensure styles are applied (Pyvis usually takes 'color' and 'size' from nx)
    # for node in net.nodes:
    #     styles = node_styles.get(node['id'], {})
    #     node.update(styles) # Apply color and size

    net.show_buttons(filter_=['physics']) # Add physics toggle button for layout control
    print(f"Saving visualization to {filename}")
    net.save_graph(filename)

    return net
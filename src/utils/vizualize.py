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


import torch
import random
import numpy as np
import pandas as pd
from torch_geometric.data import HeteroData
from pyvis.network import Network
from collections import defaultdict
from typing import Tuple, List, Dict, Optional, Set, Any

# Assuming the helper function _parse_scenario_tags is available from the previous step
def _parse_scenario_tags(tags: List[str]) -> List[str]:
    """Helper to extract base scenario names from tracking tags."""
    base_scenarios = set()
    for tag in tags:
        if tag == "Combined":
            continue
        if '/' in tag: tag = tag.split('/', 1)[1]
        if tag.endswith("_target"): tag = tag[:-7]
        elif tag.endswith("_partner"): tag = tag[:-8]
        base_scenarios.add(tag)
    return sorted(list(base_scenarios))

def visualize_anomaly_sample(
    graph: HeteroData,
    gt_node_labels: Dict[str, torch.Tensor],
    gt_edge_labels_dict: Dict[Tuple, torch.Tensor],
    anomaly_tracking: Dict[str, Dict],
    num_instances_per_scenario: int = 2,
    num_normal_nodes_per_type: int = 20, # << INCREASED DEFAULT
    neighborhood_hops: int = 1,
    provider_node_type: str = 'provider',
    member_node_type: str = 'member',
    target_edge_type: Tuple = ('provider', 'to', 'member'),
    output_filename: str = "anomaly_sample_visualization.html",
    show_buttons: bool = True,
    notebook: bool = False,
    height: str = "800px",
    width: str = "100%"
    ) -> None:
    """
    Visualizes a sample of the graph highlighting injected anomalies using Pyvis,
    with distinct shapes for provider/member nodes, including full structural
    anomaly groups, and more normal nodes.

    Args:
        graph (HeteroData): The potentially modified graph data.
        gt_node_labels (Dict[str, Tensor]): Ground truth node labels.
        gt_edge_labels_dict (Dict[Tuple, Tensor]): Ground truth edge labels.
        anomaly_tracking (Dict[str, Dict]): Detailed anomaly tracking info.
        num_instances_per_scenario (int): Max structural *instances* or attribute
                                          *nodes* per scenario to seed the sample.
        num_normal_nodes_per_type (int): Number of normal nodes of each type.
        neighborhood_hops (int): Hops to expand around the initial sample.
        provider_node_type (str): The exact name used for provider nodes.
        member_node_type (str): The exact name used for member nodes.
        target_edge_type (Tuple): Primary edge type.
        output_filename (str): Output HTML file name.
        show_buttons (bool): Show Pyvis physics/filter buttons.
        notebook (bool): True if rendering directly in a Jupyter notebook.
        height (str): Height of the visualization canvas.
        width (str): Width of the visualization canvas.
    """
    print(f"Generating anomaly visualization (k={neighborhood_hops})...")

    # --- 1. Identify Anomalous Nodes per Scenario ---
    scenario_nodes = defaultdict(list) # {scenario_name: [(ntype, idx), ...]}
    node_types_present = list(graph.node_types)
    if provider_node_type not in node_types_present: print(f"Warning: provider_node_type '{provider_node_type}' not found.")
    if member_node_type not in node_types_present: print(f"Warning: member_node_type '{member_node_type}' not found.")
    device = next(iter(gt_node_labels.values())).device

    if 'node' in anomaly_tracking:
        for ntype, nodes in anomaly_tracking['node'].items():
             if ntype not in gt_node_labels: continue
             labels = gt_node_labels[ntype]
             for idx, tags in nodes.items():
                 if idx < len(labels) and labels[idx] == 1:
                     base_scenarios = _parse_scenario_tags(tags)
                     for sc in base_scenarios:
                         scenario_nodes[sc].append((ntype, idx))

    # Identify multi-node structural scenarios (adjust names if needed)
    multi_node_structural_scenarios = {'collusion_ring', 'over_referral_clique', 'member_clique'}

    # --- 2. Select Core Nodes for Visualization ---
    core_nodes_to_visualize: Set[Tuple[str, int]] = set()
    sampled_structural_nodes = set() # Track nodes sampled as part of structural groups

    # Sample anomalous nodes/instances per scenario
    print(f"  Sampling up to {num_instances_per_scenario} instances/nodes per scenario:")
    sorted_scenarios = sorted(scenario_nodes.keys()) # Process consistently
    for scenario in sorted_scenarios:
        nodes = scenario_nodes[scenario]
        is_multi_node_struct = scenario in multi_node_structural_scenarios

        sampled_count = 0
        sampled_nodes_for_this_scenario = set()
        
        # Shuffle nodes to get random instances if sampling less than available
        random.shuffle(nodes)

        for ntype, idx in nodes:
            node_tuple = (ntype, idx)
            
            # If it's multi-node, check if we've already included this node via another sampled instance
            if is_multi_node_struct and node_tuple in sampled_structural_nodes:
                continue
                
            # If not multi-node, just check if we reached the limit for this scenario
            if not is_multi_node_struct and sampled_count >= num_instances_per_scenario:
                continue

            # Add the sampled node
            sampled_nodes_for_this_scenario.add(node_tuple)
            sampled_count += 1

            # If it's a multi-node structural scenario, find and add ALL nodes associated with this scenario tag
            # (This assumes all nodes tagged with the same structural scenario name belong to the same 'conceptual' instance for visualization purposes,
            # which might group multiple small injected instances together visually if num_instances_per_scenario > 1)
            if is_multi_node_struct:
                print(f"    - Structural Scenario '{scenario}': Including all nodes for instance containing {node_tuple}.")
                all_instance_nodes = scenario_nodes[scenario] # Get all nodes originally tagged
                sampled_nodes_for_this_scenario.update(all_instance_nodes)
                sampled_structural_nodes.update(all_instance_nodes) # Mark them all as processed
                # Stop sampling more instances for this *specific* scenario once one instance is fully added
                # Or adjust if you want exactly num_instances_per_scenario distinct *components*
                break # Break inner loop once one full structural instance component is added

            # If it's an attribute scenario, stop if limit reached
            if not is_multi_node_struct and sampled_count >= num_instances_per_scenario:
                 break


        if sampled_nodes_for_this_scenario:
            core_nodes_to_visualize.update(sampled_nodes_for_this_scenario)
            print(f"    - {scenario}: Added {len(sampled_nodes_for_this_scenario)} nodes (including full structural instances if applicable).")


    # Sample normal nodes (Increased default)
    print(f"  Sampling up to {num_normal_nodes_per_type} normal nodes per type:")
    for ntype in node_types_present:
        if ntype in gt_node_labels:
            labels = gt_node_labels[ntype]
            normal_indices = torch.where(labels == 0)[0].tolist()
            num_to_sample = min(num_normal_nodes_per_type, len(normal_indices))
            if num_to_sample > 0:
                sampled_indices = random.sample(normal_indices, num_to_sample)
                core_nodes_to_visualize.update([(ntype, idx) for idx in sampled_indices])
                print(f"    - {ntype}: Sampled {len(sampled_indices)} normal nodes.")

    print(f"  Initial core sample size: {len(core_nodes_to_visualize)} nodes.")

    # --- 3. Expand Neighborhood ---
    # (Neighborhood expansion logic remains the same)
    print(f"  Expanding neighborhood by {neighborhood_hops} hops...")
    all_nodes_to_visualize = core_nodes_to_visualize.copy()
    current_frontier = core_nodes_to_visualize.copy()

    for hop in range(neighborhood_hops):
        new_neighbors = set()
        print(f"    Hop {hop + 1}: Frontier size = {len(current_frontier)}")
        node_indices_by_type = defaultdict(list)
        for ntype, idx in current_frontier:
            node_indices_by_type[ntype].append(idx)

        for etype in graph.edge_types:
            if etype not in graph.edge_types: continue
            src_type, _, dst_type = etype
            edge_index = graph[etype].edge_index.to('cpu')

            if src_type in node_indices_by_type and len(node_indices_by_type[src_type]) > 0:
                 src_nodes_tensor = torch.tensor(node_indices_by_type[src_type], dtype=torch.long)
                 mask_src = torch.isin(edge_index[0], src_nodes_tensor)
                 dst_neighbors = edge_index[1, mask_src].tolist()
                 new_neighbors.update([(dst_type, n_idx) for n_idx in dst_neighbors])

            if dst_type in node_indices_by_type and len(node_indices_by_type[dst_type]) > 0:
                 dst_nodes_tensor = torch.tensor(node_indices_by_type[dst_type], dtype=torch.long)
                 mask_dst = torch.isin(edge_index[1], dst_nodes_tensor)
                 src_neighbors = edge_index[0, mask_dst].tolist()
                 new_neighbors.update([(src_type, n_idx) for n_idx in src_neighbors])

        current_frontier = new_neighbors - all_nodes_to_visualize
        all_nodes_to_visualize.update(new_neighbors)
        if not current_frontier:
             print(f"    No new neighbors found at hop {hop + 1}. Stopping expansion.")
             break

    print(f"  Final node sample size after expansion: {len(all_nodes_to_visualize)} nodes.")

    # --- 4. Identify Edges within the Sample ---
    # (Edge identification logic remains the same)
    edges_to_visualize = {}
    print("  Identifying edges within the node sample...")
    for etype in graph.edge_types:
        if etype not in graph.edge_types: continue
        src_type, _, dst_type = etype
        edge_index = graph[etype].edge_index.to('cpu')
        num_edges_etype = edge_index.shape[1]

        for i in range(num_edges_etype):
             u, v = edge_index[0, i].item(), edge_index[1, i].item()
             u_node = (src_type, u)
             v_node = (dst_type, v)
             if u_node in all_nodes_to_visualize and v_node in all_nodes_to_visualize:
                 edges_to_visualize[(etype, i)] = {'u': u_node, 'v': v_node}
    print(f"  Found {len(edges_to_visualize)} edges within the sample.")

    # --- 5. Create Pyvis Network ---
    net = Network(height=height, width=width, notebook=notebook, heading='Anomaly Sample Visualization', directed=True)

    # Define node properties
    NODE_COLORS = {provider_node_type: "#DC143C", member_node_type: "#1E90FF"} # Crimson Red, Dodger Blue
    NODE_SHAPES = {provider_node_type: "square", member_node_type: "dot"}
    DEFAULT_NODE_COLOR = "#C0C0C0"
    DEFAULT_NODE_SHAPE = "ellipse"
    ANOMALOUS_BORDER_COLOR = "#FFD700" # Gold border for anomalous
    NORMAL_BORDER_COLOR = "#666666" # Darker grey border for normal
    NODE_SIZE_NORMAL = 15
    NODE_SIZE_ANOMALOUS = 25

    # Edge colors
    EDGE_COLOR_NORMAL = "#CCCCCC" # Light grey
    EDGE_COLOR_ANOMALOUS = "#FF4500" # OrangeRed

    added_node_ids = set()
    node_id_map = {}

    # --- 6. Add Nodes to Pyvis ---
    print("  Adding nodes to visualization...")
    for ntype, idx in all_nodes_to_visualize:
        node_pyvis_id = f"{ntype}_{idx}"
        if node_pyvis_id in added_node_ids: continue

        is_anomalous = bool(ntype in gt_node_labels and idx < len(gt_node_labels[ntype]) and gt_node_labels[ntype][idx] == 1)
        node_color = NODE_COLORS.get(ntype, DEFAULT_NODE_COLOR)
        node_shape = NODE_SHAPES.get(ntype, DEFAULT_NODE_SHAPE)
        node_size = NODE_SIZE_ANOMALOUS if is_anomalous else NODE_SIZE_NORMAL
        border_color = ANOMALOUS_BORDER_COLOR if is_anomalous else NORMAL_BORDER_COLOR

        # Construct label and title
        label_parts = [node_pyvis_id]
        title_parts = [f"ID: {node_pyvis_id}"]
        if is_anomalous:
            #label_parts.append("!") # Indicator
            title_parts.append("Status: Anomalous")
            if 'node' in anomaly_tracking and ntype in anomaly_tracking['node'] and idx in anomaly_tracking['node'][ntype]:
                tags = anomaly_tracking['node'][ntype][idx]
                scenarios = _parse_scenario_tags(tags)
                types = set()
                if "Combined" in tags: types.add("Combined")
                if any(t.startswith("Structural/") for t in tags): types.add("Structural")
                if any(t.startswith("Attribute/") for t in tags): types.add("Attribute")
                type_str = "/".join(sorted(list(types))) if types else "Unknown"
                scenario_str = ", ".join(scenarios)
                #label_parts.append(f"({type_str[:3]})")
                title_parts.append(f"Type: {type_str}")
                title_parts.append(f"Scenarios: {scenario_str}")
            else:
                title_parts.append("Type: Unknown (Not in tracking)")
        else:
            title_parts.append("Status: Normal")

        final_label = " ".join(label_parts)
        final_title = "\n".join(title_parts)

        net.add_node(node_pyvis_id, label=final_label, title=final_title,
                       color=node_color, # Base color shows type
                       borderWidth=3 if is_anomalous else 1.5, # Thicker border if anomalous
                       borderColor=border_color, # Border color shows status
                       shape=node_shape,
                       size=node_size
                       )
        added_node_ids.add(node_pyvis_id)
        node_id_map[(ntype, idx)] = node_pyvis_id

    # --- 7. Add Edges to Pyvis ---
    print("  Adding edges to visualization...")
    for (etype, edge_idx), edge_info in edges_to_visualize.items():
        u_node, v_node = edge_info['u'], edge_info['v']
        u_id, v_id = node_id_map.get(u_node), node_id_map.get(v_node)

        if u_id and v_id:
            is_anomalous = bool(etype in gt_edge_labels_dict and edge_idx < len(gt_edge_labels_dict[etype]) and gt_edge_labels_dict[etype][edge_idx] == 1)
            edge_color = EDGE_COLOR_ANOMALOUS if is_anomalous else EDGE_COLOR_NORMAL
            edge_width = 2.5 if is_anomalous else 1
            edge_title = f"Type: {etype}\nIndex: {edge_idx}\nStatus: {'Anomalous' if is_anomalous else 'Normal'}"
            if hasattr(graph[etype], 'edge_attr') and graph[etype].edge_attr is not None:
                 attrs = graph[etype].edge_attr[edge_idx].tolist()
                 if len(attrs) < 5: edge_title += f"\nAttrs: {attrs}"

            net.add_edge(u_id, v_id, title=edge_title, color=edge_color, width=edge_width)

    # --- 8. Configure and Save/Show ---
    if show_buttons:
        net.show_buttons(filter_=['physics', 'nodes', 'edges'])

    try:
        if notebook:
            net.show(output_filename)
            print(f"Visualization displayed in notebook (also saved to {output_filename})")
        else:
            net.save_graph(output_filename)
            print(f"Visualization saved to {output_filename}")
    except Exception as e:
         print(f"Error during Pyvis generation/saving: {e}")
         print("Try installing requirements: pip install pyvis pandas")



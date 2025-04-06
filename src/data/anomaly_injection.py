import pandas as pd
import torch
from torch_geometric.data import HeteroData
import numpy as np
import random
from collections import defaultdict
import math  # For ceiling division
from typing import Optional, Tuple, List, Dict, Union # Added for type hinting

def inject_combined_anomalies_relative(data: HeteroData, num_injections: int = 20,
                                        node_perturb_scale: float = 2.0, # Scale factor relative to std dev
                                        edge_perturb_scale: float = 3.0, # Scale factor relative to edge attr std dev
                                        always_perturb_nodes: bool = True): # Ensure nodes always get attribute anomaly
    """
    Injects combined topological and *relative* attribute anomalies.

    Changes from original:
    - Node feature noise scale is relative to feature standard deviation.
    - Edge attribute noise scale is relative to edge attribute standard deviation.
    - Base edge value uses mean of existing edge attributes.
    - Option to always apply node feature perturbation to anomalous nodes.
    """
    modified_data = data.clone() # Use copy() method for HeteroData

    num_providers = modified_data['provider'].num_nodes
    num_members = modified_data['member'].num_nodes

    gt_labels = {
        'provider': torch.zeros(num_providers, dtype=torch.long),
        'member': torch.zeros(num_members, dtype=torch.long)
    }
    # Use defaultdict for easier aggregation
    anomaly_tracking = {
        'provider': defaultdict(set),
        'member': defaultdict(set)
    }

    target_edge_type = ('provider', 'to', 'member') # Assuming this is the primary edge type
    reverse_edge_type = ('member', 'to', 'provider')

    # --- Calculate Feature Statistics for Relative Noise ---
    feature_std_dev = {}
    for node_type in data.node_types:
        if hasattr(data[node_type], 'x') and data[node_type].x.numel() > 0:
            feature_std_dev[node_type] = data[node_type].x.std(dim=0)
            # Handle features with zero std dev (e.g., constants) to avoid dividing by zero
            feature_std_dev[node_type][feature_std_dev[node_type] == 0] = 1.0
        else:
            feature_std_dev[node_type] = None


    # --- Calculate Edge Attribute Statistics ---
    edge_attr_mean = 1.0 # Default fallback
    edge_attr_std = 1.0  # Default fallback
    if target_edge_type in data.edge_types and hasattr(data[target_edge_type], 'edge_attr'):
       original_edge_attr = data[target_edge_type].edge_attr
       if original_edge_attr is not None and original_edge_attr.numel() > 0:
           edge_attr_mean = original_edge_attr.mean(dim=0)
           edge_attr_std = original_edge_attr.std(dim=0)
           # Handle zero std dev
           edge_attr_std[edge_attr_std == 0] = 1.0
           print(f"Edge Attr Stats: Mean={edge_attr_mean.item():.2f}, Std={edge_attr_std.item():.2f}")


    for inj in range(num_injections):
        size_provider = random.randint(2, min(20, num_providers))
        size_member = random.randint(2, min(20, num_members))

        providers_anom = np.random.choice(num_providers, size=size_provider, replace=False)
        members_anom = np.random.choice(num_members, size=size_member, replace=False)

        for p in providers_anom:
            gt_labels['provider'][p] = 1
            anomaly_tracking['provider'][p].add('structural')
        for m in members_anom:
            gt_labels['member'][m] = 1
            anomaly_tracking['member'][m].add('structural')

        full_block = random.random() < 0.5
        if full_block:
            new_edges = [(p, m) for p in providers_anom for m in members_anom]
            block_type = 'full_dense'
        else:
            frac = random.uniform(0.2, 1.0)
            possible_edges = [(p, m) for p in providers_anom for m in members_anom]
            num_edges_to_select = max(1, int(frac * len(possible_edges)))
            new_edges = random.sample(possible_edges, num_edges_to_select)
            block_type = 'partial_dense'

        existing_edge_index = modified_data[target_edge_type].edge_index
        existing_edges_set = set(tuple(map(int, x)) for x in existing_edge_index.t().tolist())


        new_edge_list = []
        new_edge_attr_list = []
        for (p, m) in new_edges:
            if (p, m) in existing_edges_set:
                continue

            technique = random.choice(['outside', 'scaled'])
            # Use relative scale based on edge attr std dev
            c_edge = random.uniform(edge_perturb_scale - 1.0, edge_perturb_scale + 1.0) * edge_attr_std
            base_val = edge_attr_mean # Use mean as base

            if technique == 'outside':
                # Push value significantly away from mean
                new_val = base_val + c_edge * np.sign(np.random.randn()) # Add or subtract scaled std dev
                anomaly_type_edge = 'feature_outside'
            else: # 'scaled'
                # Add Gaussian noise with std dev relative to original std dev
                noise = torch.normal(mean=0.0, std=c_edge)
                new_val = base_val + noise
                anomaly_type_edge = 'feature_scaled'

            # Ensure edge attributes have the correct shape (e.g., [1] if edge_dim=1)
            if isinstance(new_val, torch.Tensor):
                 new_val_tensor = new_val.clone().detach().reshape(base_val.shape) # Ensure same shape as mean
            else: # Handle scalar case if edge_attr_mean was scalar
                 new_val_tensor = torch.tensor([new_val], dtype=original_edge_attr.dtype)


            new_edge_list.append((p, m))
            new_edge_attr_list.append(new_val_tensor) # Store tensor directly

            anomaly_tracking['provider'][p].add(anomaly_type_edge)
            anomaly_tracking['member'][m].add(anomaly_type_edge)

        if new_edge_list:
            new_edge_tensor = torch.tensor(new_edge_list, dtype=torch.long).t()
            # Stack the list of tensors correctly
            new_edge_attr_tensor = torch.stack(new_edge_attr_list, dim=0)

            modified_data[target_edge_type].edge_index = torch.cat(
                [modified_data[target_edge_type].edge_index, new_edge_tensor], dim=1
            )
            modified_data[target_edge_type].edge_attr = torch.cat(
                [modified_data[target_edge_type].edge_attr, new_edge_attr_tensor], dim=0
            )

            if reverse_edge_type in modified_data.edge_types:
                reverse_new_edges = torch.stack([new_edge_tensor[1], new_edge_tensor[0]], dim=0)
                modified_data[reverse_edge_type].edge_index = torch.cat(
                    [modified_data[reverse_edge_type].edge_index, reverse_new_edges], dim=1
                )
                # Add corresponding attributes to reverse edges if they exist
                if hasattr(modified_data[reverse_edge_type], 'edge_attr'):
                     modified_data[reverse_edge_type].edge_attr = torch.cat(
                           [modified_data[reverse_edge_type].edge_attr, new_edge_attr_tensor], dim=0
                     )


        # Node feature anomalies (relative noise)
        if always_perturb_nodes or random.random() < 0.5:
            for node_type, nodes in [('provider', providers_anom), ('member', members_anom)]:
                 if feature_std_dev.get(node_type) is None: continue # Skip if no features

                 std_devs = feature_std_dev[node_type].to(modified_data[node_type].x.device)

                 for idx in nodes:
                    x = modified_data[node_type].x[idx]
                    num_features = x.size(0)
                    num_to_perturb = max(1, int(0.3 * num_features))
                    perturb_indices = np.random.choice(num_features, num_to_perturb, replace=False)

                    # Calculate noise relative to standard deviation of each feature
                    noise_scale = random.uniform(node_perturb_scale - 0.5, node_perturb_scale + 0.5)
                    noise = torch.normal(mean=0.0, std=std_devs[perturb_indices] * noise_scale)
                    noise = noise.to(x.device) # Ensure noise is on the same device

                    # Add noise
                    x_perturbed = x.clone()
                    x_perturbed[perturb_indices] += noise
                    modified_data[node_type].x[idx] = x_perturbed # Update data

                    anomaly_tracking[node_type][idx].add('feature_node_relative') # New tag

        for p in providers_anom:
            anomaly_tracking['provider'][p].add(block_type)
        for m in members_anom:
            anomaly_tracking['member'][m].add(block_type)


    # Clean up tracking dict (remove empty sets for non-anomalous nodes)
    # Convert back from defaultdict if needed, though usually not necessary
    final_anomaly_tracking = {
         'provider': {i: types for i, types in anomaly_tracking['provider'].items() if gt_labels['provider'][i] == 1},
         'member': {i: types for i, types in anomaly_tracking['member'].items() if gt_labels['member'][i] == 1}
     }


    return modified_data, gt_labels, final_anomaly_tracking




import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
import numpy as np
import random
from collections import defaultdict
import math
from typing import Optional, Tuple, List, Dict, Union, Literal # Added Literal

# Helper function for statistical attribute anomaly (outside interval)
def perturb_statistical_outside(feature_vector, mean, std, c):
    """Perturbs features to be outside mean +/- c*std."""
    noise = torch.normal(mean=0.0, std=c * std)
    # Push value away from mean
    perturbed_val = mean + torch.sign(torch.randn_like(noise)) * torch.abs(noise)
    # Simple approach: replace original with perturbed
    # More robust: check if original was already outside, if so maybe push further?
    # Keep it simple for now.
    return perturbed_val

# Helper function for statistical attribute anomaly (scaled Gaussian)
def perturb_statistical_scaled(feature_vector, std, c):
    """Adds scaled Gaussian noise N(0, (c*std)^2)."""
    noise = torch.normal(mean=0.0, std=c * std)
    return feature_vector + noise

def inject_custom_anomalies(
    data: HeteroData,
    # --- Injection Budgets ---
    num_structural: int = 5,
    num_node_attr: int = 10,
    num_edge_attr: int = 10,
    num_combined: int = 5,
    # --- Structural Params (Type 1 & 4) ---
    struct_min_nodes_u: int = 2,
    struct_max_nodes_u: int = 10,
    struct_min_nodes_v: int = 2,
    struct_max_nodes_v: int = 10,
    struct_partial_block_frac_range: Tuple[float, float] = (0.5, 1.0), # Defaulting to denser partial blocks
    # --- Node Attribute Params (Type 2 & 4) ---
    node_attr_method: Literal['statistical_outside', 'statistical_scaled', 'swap'] = 'swap',
    node_attr_perturb_frac: float = 0.3, # Fraction of features to perturb for statistical methods
    node_attr_stat_c_range: Tuple[float, float] = (2.0, 4.0), # 'c' for statistical methods
    node_attr_swap_pool_size: int = 50, # 'k' candidates for swap method
    node_attr_swap_metric: Literal['euclidean', 'cosine'] = 'euclidean', # Distance for swap
    # --- Edge Attribute Params (Type 3 & 4) ---
    edge_attr_c_range: Tuple[float, float] = (2.0, 4.0), # 'c' for modifying claim counts (deviation from mean)
    edge_attr_direction: Literal['high', 'low', 'both'] = 'high', # Target high, low, or either claim count anomaly
    # --- Combined Params (Type 4) ---
    combined_node_attr_prob: float = 0.75, # Probability of adding node anomalies in combined blocks
    # --- General ---
    seed: Optional[int] = None
    ) -> Tuple[HeteroData, Dict[str, torch.Tensor], Dict[Tuple, torch.Tensor], Dict[str, Dict]]:
    """
    Injects synthetic anomalies based on GraphBEAN/DOMINANT principles.

    Injects four types:
    1. Structural: Dense bipartite blocks.
    2. Node Attribute: Modifies node features (statistical or swap).
    3. Edge Attribute: Modifies existing edge features (claim counts).
    4. Combined: Structural block + Edge attribute mods + Optional Node attribute mods.

    Args:
        data (HeteroData): Original graph data. Needs 'provider', 'member' node types
                           and edge type ('provider', 'to', 'member').
                           Node features 'x' needed for node/combined injections.
                           Edge features 'edge_attr' needed for edge/combined injections.
        num_structural, num_node_attr, num_edge_attr, num_combined (int): Number of
            injection events for each anomaly type.
        struct_*: Parameters controlling structural block creation.
        node_attr_*: Parameters controlling node attribute modifications.
                      `node_attr_method` chooses the technique.
        edge_attr_*: Parameters controlling edge attribute modifications.
                      `edge_attr_direction` controls if high/low values are targeted.
        combined_node_attr_prob (float): Probability of applying node attribute
                                         modification within a combined anomaly block.
        seed (Optional[int]): Random seed for reproducibility.

    Returns:
        modified_data (HeteroData): Graph with injected anomalies.
        gt_node_labels (Dict[str, Tensor]): Binary labels for anomalous nodes.
        gt_edge_labels (Dict[Tuple, Tensor]): Binary labels for anomalous edges
                                               (structurally added or attribute modified).
        anomaly_tracking (Dict[str, Dict]): Detailed anomaly types per node/edge index.
                                            Uses tuples like ('edge', edge_type, edge_idx)
                                            as keys for edge tracking.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    print("--- Injecting Custom Anomalies (GraphBEAN/DOMINANT Inspired) ---")
    print(f"Budgets: Struct={num_structural}, NodeAttr={num_node_attr}, EdgeAttr={num_edge_attr}, Combined={num_combined}")

    modified_data = data.clone() # Use deepcopy for safety
    node_type_u, node_type_v = 'provider', 'member'
    target_edge_type = (node_type_u, 'to', node_type_v)
    reverse_edge_type = (node_type_v, 'to', node_type_u)

    # --- Input Checks ---
    if node_type_u not in modified_data.node_types or node_type_v not in modified_data.node_types:
         raise ValueError("Data must contain node types 'provider' and 'member'.")
    if target_edge_type not in modified_data.edge_types:
         raise ValueError(f"Data must contain edge type {target_edge_type}.")
    if (num_node_attr > 0 or (num_combined > 0 and combined_node_attr_prob > 0)):
        if not hasattr(modified_data[node_type_u], 'x') or not hasattr(modified_data[node_type_v], 'x'):
            print("Warning: Node features 'x' missing. Skipping Node Attribute and Combined(Node) injections.")
            num_node_attr = 0; num_combined = 0 # Avoid errors later
    if num_edge_attr > 0 or num_combined > 0:
         if not hasattr(modified_data[target_edge_type], 'edge_attr'):
             print("Warning: Edge features 'edge_attr' missing for target type. Skipping Edge Attribute and Combined(Edge) injections.")
             num_edge_attr = 0; # Skip edge part of combined later

    device = modified_data.x_dict[node_type_u].device if hasattr(modified_data[node_type_u], 'x') else modified_data.edge_index_dict[target_edge_type].device


    num_nodes_u = modified_data[node_type_u].num_nodes
    num_nodes_v = modified_data[node_type_v].num_nodes
    original_num_edges = {et: modified_data[et].edge_index.shape[1] for et in modified_data.edge_types}

    # --- Initialize Ground Truth & Tracking ---
    gt_node_labels = {nt: torch.zeros(modified_data[nt].num_nodes, dtype=torch.long, device=device) for nt in modified_data.node_types}
    # Edge labels: Start with zeros for existing edges. Will grow.
    gt_edge_labels_dict = {et: torch.zeros(original_num_edges[et], dtype=torch.long, device=device) for et in modified_data.edge_types}
    anomaly_tracking = {nt: defaultdict(list) for nt in modified_data.node_types} # Store lists of tags
    anomaly_tracking['edge'] = defaultdict(list) # For tracking specific edges

    # Keep track of modified elements to avoid re-modifying within the same type loop
    nodes_modified_struct = {nt: set() for nt in modified_data.node_types}
    nodes_modified_attr = {nt: set() for nt in modified_data.node_types}
    edges_modified_attr = {et: set() for et in modified_data.edge_types} # Stores original indices

    nodes_involved_in_swaps = {nt: set() for nt in modified_data.node_types} # Track nodes swapped in this run

    # --- Calculate Statistics ---
    node_stats = {}
    if num_node_attr > 0 or (num_combined > 0 and combined_node_attr_prob > 0):
        for nt in [node_type_u, node_type_v]:
             if hasattr(modified_data[nt], 'x'):
                 x = modified_data[nt].x
                 mean = x.mean(dim=0)
                 std = x.std(dim=0)
                 std[std == 0] = 1.0 # Avoid NaN/Inf
                 node_stats[nt] = {'mean': mean.to(device), 'std': std.to(device)}

    edge_stats = {}
    if num_edge_attr > 0 or num_combined > 0:
        if hasattr(modified_data[target_edge_type], 'edge_attr'):
            ea = modified_data[target_edge_type].edge_attr
            mean = ea.mean(dim=0)
            std = ea.std(dim=0)
            std[std == 0] = 1.0
            edge_stats = {'mean': mean.to(device), 'std': std.to(device)}
            edge_attr_dtype = ea.dtype
            edge_attr_shape = list(mean.shape) if mean.numel() > 1 else [1] # Store shape
        else: # Handle case where attributes might be added later
            edge_stats = {'mean': torch.tensor([1.0], device=device), 'std': torch.tensor([1.0], device=device)}
            edge_attr_dtype = torch.float
            edge_attr_shape = [1]
            print("Using default edge stats mean=1, std=1.")

    # --- Store newly added edges temporarily ---
    new_edges_all_types = {et: [] for et in [target_edge_type, reverse_edge_type] if et in modified_data.edge_types}
    new_edge_attrs_all_types = {et: [] for et in [target_edge_type, reverse_edge_type] if et in modified_data.edge_types}


    # === Injection Loops ===

    # --- Type 1: Structural Anomalies ---
    print(f"\nInjecting {num_structural} Structural Anomalies...")
    for i in range(num_structural):
        # Select block nodes
        size_u = random.randint(struct_min_nodes_u, min(struct_max_nodes_u, num_nodes_u)) if num_nodes_u >= struct_min_nodes_u else 0
        size_v = random.randint(struct_min_nodes_v, min(struct_max_nodes_v, num_nodes_v)) if num_nodes_v >= struct_min_nodes_v else 0
        if size_u < struct_min_nodes_u or size_v < struct_min_nodes_v: continue

        nodes_u = np.random.choice(num_nodes_u, size=size_u, replace=False)
        nodes_v = np.random.choice(num_nodes_v, size=size_v, replace=False)

        # Generate block edges (full or partial)
        is_partial = random.random() > 0.5
        block_type_tag = f"{'partial' if is_partial else 'full'}_dense_block_struct_only"
        possible_edges = [(u, v) for u in nodes_u for v in nodes_v]
        if is_partial:
            frac = random.uniform(struct_partial_block_frac_range[0], struct_partial_block_frac_range[1])
            edges_to_consider = random.sample(possible_edges, max(1, int(frac * len(possible_edges))))
        else:
            edges_to_consider = possible_edges

        # Filter existing edges from the *current* state of the graph before adding attributes
        current_edge_index = modified_data[target_edge_type].edge_index
        current_existing_set = set(tuple(map(int, x)) for x in current_edge_index.t().tolist())
        new_edges_struct = [edge for edge in edges_to_consider if tuple(edge) not in current_existing_set]

        if not new_edges_struct: continue

        # Assign default/mean attributes
        default_attr_val = edge_stats.get('mean', torch.tensor([1.0], device=device))
        new_edge_attrs_struct = [default_attr_val.clone().detach().reshape(edge_attr_shape).to(edge_attr_dtype) for _ in new_edges_struct]

        # Mark nodes and track
        for u in nodes_u:
            gt_node_labels[node_type_u][u] = 1
            anomaly_tracking[node_type_u][u].append(block_type_tag)
            nodes_modified_struct[node_type_u].add(u)
        for v in nodes_v:
            gt_node_labels[node_type_v][v] = 1
            anomaly_tracking[node_type_v][v].append(block_type_tag)
            nodes_modified_struct[node_type_v].add(v)

        # Store edges and attributes to be added later
        new_edges_all_types[target_edge_type].extend(new_edges_struct)
        new_edge_attrs_all_types[target_edge_type].extend(new_edge_attrs_struct)
        # Handle reverse edges storage
        if reverse_edge_type in new_edges_all_types:
            new_edges_all_types[reverse_edge_type].extend([(v, u) for u, v in new_edges_struct])
            new_edge_attrs_all_types[reverse_edge_type].extend(new_edge_attrs_struct) # Assuming symmetric attrs for reverse

    # --- Type 2: Node Attribute Anomalies ---
    print(f"\nInjecting {num_node_attr} Node Attribute Anomalies ({node_attr_method})...")
    for i in range(num_node_attr):
        # Choose node type proportionally (or randomly if preferred)
        node_type = random.choices([node_type_u, node_type_v], weights=[num_nodes_u, num_nodes_v], k=1)[0]
        num_nodes = modified_data[node_type].num_nodes
        if num_nodes == 0 or node_type not in node_stats: continue # Skip if no nodes or no stats/features

        # Select target node (prefer nodes not already modified by this type)
        available_targets = list(set(range(num_nodes)) - nodes_modified_attr[node_type])
        if not available_targets: available_targets = list(range(num_nodes)) # Fallback if all modified
        if not available_targets: continue
        target_idx = random.choice(available_targets)

        tag_list = [] # Specific tags for this anomaly

        if node_attr_method == 'swap':
            # Find swap partner
            potential_pool_indices = list(set(range(num_nodes)) - {target_idx}) # Exclude self
            if not potential_pool_indices: continue # Need at least one other node
            current_pool_size = min(node_attr_swap_pool_size, len(potential_pool_indices))
            pool_indices = np.random.choice(potential_pool_indices, size=current_pool_size, replace=False)

            target_feat = modified_data[node_type].x[target_idx].unsqueeze(0)
            pool_feats = modified_data[node_type].x[pool_indices]

            if node_attr_swap_metric == 'euclidean': distances = torch.cdist(target_feat, pool_feats, p=2.0).squeeze(0)
            elif node_attr_swap_metric == 'cosine': distances = 1.0 - F.cosine_similarity(target_feat, pool_feats, dim=1)
            else: raise ValueError("Invalid node_attr_swap_metric.")

            distances = torch.nan_to_num(distances, nan=-torch.inf)
            if distances.numel() == 0 or torch.all(torch.isinf(distances)): continue

            # Find most distant partner
            most_distant_pool_idx = torch.argmax(distances).item()
            swap_partner_idx = pool_indices[most_distant_pool_idx]

            # Perform swap
            feat_target_original = modified_data[node_type].x[target_idx].clone()
            feat_swap_original = modified_data[node_type].x[swap_partner_idx].clone()
            modified_data[node_type].x[target_idx] = feat_swap_original
            modified_data[node_type].x[swap_partner_idx] = feat_target_original

            # Mark both and track
            gt_node_labels[node_type][target_idx] = 1
            gt_node_labels[node_type][swap_partner_idx] = 1
            tag_list.append('node_attr_swap_target')
            anomaly_tracking[node_type][swap_partner_idx].append('node_attr_swap_partner')
            nodes_modified_attr[node_type].add(target_idx)
            nodes_modified_attr[node_type].add(swap_partner_idx)

        elif node_attr_method in ['statistical_outside', 'statistical_scaled']:
            # Statistical perturbation
            x = modified_data[node_type].x[target_idx]
            num_features = x.size(0)
            if num_features == 0: continue
            num_to_perturb = max(1, int(node_attr_perturb_frac * num_features))
            perturb_indices = np.random.choice(num_features, num_to_perturb, replace=False)

            mean_f = node_stats[node_type]['mean'][perturb_indices]
            std_f = node_stats[node_type]['std'][perturb_indices]
            c = random.uniform(node_attr_stat_c_range[0], node_attr_stat_c_range[1])
            x_perturbed = x.clone()

            if node_attr_method == 'statistical_outside':
                 perturbed_vals = perturb_statistical_outside(x[perturb_indices], mean_f, std_f, c)
                 tag_list.append('node_attr_stat_outside')
            else: # statistical_scaled
                 perturbed_vals = perturb_statistical_scaled(x[perturb_indices], std_f, c)
                 tag_list.append('node_attr_stat_scaled')

            x_perturbed[perturb_indices] = perturbed_vals.to(x.dtype)
            modified_data[node_type].x[target_idx] = x_perturbed

            # Mark target and track
            gt_node_labels[node_type][target_idx] = 1
            nodes_modified_attr[node_type].add(target_idx)

        else:
            raise ValueError(f"Unknown node_attr_method: {node_attr_method}")

        # Add common tag
        anomaly_tracking[node_type][target_idx].extend(tag_list)


    # --- Type 3: Edge Attribute Anomalies ---
    print(f"\nInjecting {num_edge_attr} Edge Attribute Anomalies...")
    current_edge_attr = modified_data.get(target_edge_type, {}).get('edge_attr')
    if num_edge_attr > 0 and current_edge_attr is not None and edge_stats:
        num_existing_edges = original_num_edges[target_edge_type]
        if num_existing_edges > 0:
            available_edge_indices = list(set(range(num_existing_edges)) - edges_modified_attr[target_edge_type])
            num_to_inject_edge = min(num_edge_attr, len(available_edge_indices))

            if num_to_inject_edge > 0:
                selected_edge_indices = random.sample(available_edge_indices, num_to_inject_edge)

                for edge_idx in selected_edge_indices:
                    c = random.uniform(edge_attr_c_range[0], edge_attr_c_range[1])
                    mean_ea = edge_stats['mean']
                    std_ea = edge_stats['std']
                    original_val = current_edge_attr[edge_idx]

                    # Modify value based on direction
                    if edge_attr_direction == 'high':
                        new_val = mean_ea + c * std_ea
                    elif edge_attr_direction == 'low':
                        new_val = mean_ea - c * std_ea
                        # Optional: clamp at zero or some minimum if needed
                        # new_val = torch.clamp(new_val, min=0)
                    else: # 'both'
                        new_val = mean_ea + random.choice([-1, 1]) * c * std_ea

                    # Ensure shape and type match original
                    modified_data[target_edge_type].edge_attr[edge_idx] = new_val.reshape(edge_attr_shape).to(edge_attr_dtype)

                    # Mark edge as anomalous in the label tensor (use original index)
                    gt_edge_labels_dict[target_edge_type][edge_idx] = 1
                    # Track edge anomaly by its original index
                    anomaly_tracking['edge'][(target_edge_type, edge_idx)].append('edge_attr_claims')
                    edges_modified_attr[target_edge_type].add(edge_idx)

                    # Also track for reverse edge if it exists and shares attributes
                    if reverse_edge_type in modified_data.edge_types and hasattr(modified_data[reverse_edge_type], 'edge_attr'):
                         # Assuming edge_attr might be shared or symmetrically modified
                         # This requires careful handling if attrs aren't shared!
                         # For simplicity, assume modification applies to both directions' index 'edge_idx'
                         # This might be incorrect if reverse edge attributes are independent
                         if edge_idx < original_num_edges.get(reverse_edge_type, 0):
                              modified_data[reverse_edge_type].edge_attr[edge_idx] = new_val.reshape(edge_attr_shape).to(edge_attr_dtype)
                              gt_edge_labels_dict[reverse_edge_type][edge_idx] = 1
                              anomaly_tracking['edge'][(reverse_edge_type, edge_idx)].append('edge_attr_claims')
                              edges_modified_attr[reverse_edge_type].add(edge_idx)


            else: print("  No available existing edges to modify for Type 3.")
        else: print("  Skipping Type 3: No existing edges.")
    elif num_edge_attr > 0: print("  Skipping Type 3: No edge attributes found.")


    # --- Type 4: Combined Anomalies ---
    print(f"\nInjecting {num_combined} Combined Anomalies...")
    for i in range(num_combined):
        # 1. Inject Structure (similar to Type 1)
        size_u = random.randint(struct_min_nodes_u, min(struct_max_nodes_u, num_nodes_u)) if num_nodes_u >= struct_min_nodes_u else 0
        size_v = random.randint(struct_min_nodes_v, min(struct_max_nodes_v, num_nodes_v)) if num_nodes_v >= struct_min_nodes_v else 0
        if size_u < struct_min_nodes_u or size_v < struct_min_nodes_v: continue

        nodes_u = np.random.choice(num_nodes_u, size=size_u, replace=False)
        nodes_v = np.random.choice(num_nodes_v, size=size_v, replace=False)

        is_partial = random.random() > 0.5
        block_type_tag = f"{'partial' if is_partial else 'full'}_dense_block_combined"
        possible_edges = [(u, v) for u in nodes_u for v in nodes_v]
        if is_partial:
            frac = random.uniform(struct_partial_block_frac_range[0], struct_partial_block_frac_range[1])
            edges_to_consider = random.sample(possible_edges, max(1, int(frac * len(possible_edges))))
        else: edges_to_consider = possible_edges

        current_edge_index = modified_data[target_edge_type].edge_index
        current_existing_set = set(tuple(map(int, x)) for x in current_edge_index.t().tolist())
        new_edges_combined = [edge for edge in edges_to_consider if tuple(edge) not in current_existing_set]

        if not new_edges_combined: continue

        # Mark nodes and track structural part
        for u in nodes_u:
            gt_node_labels[node_type_u][u] = 1; anomaly_tracking[node_type_u][u].append(block_type_tag)
            nodes_modified_struct[node_type_u].add(u) # Track struct modification
        for v in nodes_v:
            gt_node_labels[node_type_v][v] = 1; anomaly_tracking[node_type_v][v].append(block_type_tag)
            nodes_modified_struct[node_type_v].add(v)

        # 2. Inject Edge Attribute Anomalies on *new* edges
        new_edge_attrs_combined = []
        if edge_stats: # Proceed only if stats were calculated
            num_new_edges = len(new_edges_combined)
            c = random.uniform(edge_attr_c_range[0], edge_attr_c_range[1])
            mean_ea, std_ea = edge_stats['mean'], edge_stats['std']

            for u_idx, v_idx in new_edges_combined:
                # Use same logic as Type 3 but apply to new edges
                if edge_attr_direction == 'high': new_val = mean_ea + c * std_ea
                elif edge_attr_direction == 'low': new_val = mean_ea - c * std_ea # Optional clamping needed?
                else: new_val = mean_ea + random.choice([-1, 1]) * c * std_ea

                new_edge_attrs_combined.append(new_val.reshape(edge_attr_shape).to(edge_attr_dtype))
                # Track edge anomaly contribution to nodes
                anomaly_tracking[node_type_u][u_idx].append('combined_edge_attr')
                anomaly_tracking[node_type_v][v_idx].append('combined_edge_attr')
        else: # No edge stats, use default attribute
             default_attr_val = torch.tensor([1.0], device=device)
             new_edge_attrs_combined = [default_attr_val.clone().detach().reshape(edge_attr_shape).to(edge_attr_dtype) for _ in new_edges_combined]


        # 3. Optionally Inject Node Attribute Anomalies
        if random.random() < combined_node_attr_prob:
            for node_type, nodes_in_block in [(node_type_u, nodes_u), (node_type_v, nodes_v)]:
                if node_type not in node_stats: continue # Skip if no features/stats

                for target_idx in nodes_in_block:
                    # Use selected node_attr_method
                    tag_list = []
                    if node_attr_method == 'swap':
                         # Find swap partner (ensure partner is not also in the current combined block for realism?)
                         # Pool excludes target and nodes *already swapped this run*, and potentially nodes within the same block
                         potential_pool_indices = list(set(range(modified_data[node_type].num_nodes)) - {target_idx} - nodes_involved_in_swaps[node_type] - set(nodes_u if node_type==node_type_u else nodes_v))
                         if not potential_pool_indices: continue
                         current_pool_size = min(node_attr_swap_pool_size, len(potential_pool_indices))
                         pool_indices = np.random.choice(potential_pool_indices, size=current_pool_size, replace=False)

                         target_feat = modified_data[node_type].x[target_idx].unsqueeze(0)
                         pool_feats = modified_data[node_type].x[pool_indices]

                         if node_attr_swap_metric == 'euclidean': distances = torch.cdist(target_feat, pool_feats, p=2.0).squeeze(0)
                         elif node_attr_swap_metric == 'cosine': distances = 1.0 - F.cosine_similarity(target_feat, pool_feats, dim=1)
                         else: raise ValueError("Invalid node_attr_swap_metric.")
                         distances = torch.nan_to_num(distances, nan=-torch.inf)
                         if distances.numel() == 0 or torch.all(torch.isinf(distances)): continue

                         most_distant_pool_idx = torch.argmax(distances).item()
                         swap_partner_idx = pool_indices[most_distant_pool_idx]

                         if swap_partner_idx in nodes_involved_in_swaps[node_type]: continue

                         feat_target_original = modified_data[node_type].x[target_idx].clone()
                         feat_swap_original = modified_data[node_type].x[swap_partner_idx].clone()
                         modified_data[node_type].x[target_idx] = feat_swap_original
                         modified_data[node_type].x[swap_partner_idx] = feat_target_original

                         # Mark partner (target already marked by struct), track both
                         gt_node_labels[node_type][swap_partner_idx] = 1
                         tag_list.append('combined_node_attr_swap_target')
                         anomaly_tracking[node_type][swap_partner_idx].append('combined_node_attr_swap_partner')
                         nodes_involved_in_swaps[node_type].add(target_idx) # Add target here too
                         nodes_involved_in_swaps[node_type].add(swap_partner_idx)

                    elif node_attr_method in ['statistical_outside', 'statistical_scaled']:
                        x = modified_data[node_type].x[target_idx]
                        num_features = x.size(0)
                        if num_features == 0: continue
                        num_to_perturb = max(1, int(node_attr_perturb_frac * num_features))
                        perturb_indices = np.random.choice(num_features, num_to_perturb, replace=False)
                        mean_f = node_stats[node_type]['mean'][perturb_indices]; std_f = node_stats[node_type]['std'][perturb_indices]
                        c = random.uniform(node_attr_stat_c_range[0], node_attr_stat_c_range[1])
                        x_perturbed = x.clone()
                        if node_attr_method == 'statistical_outside':
                             perturbed_vals = perturb_statistical_outside(x[perturb_indices], mean_f, std_f, c)
                             tag_list.append('combined_node_attr_stat_outside')
                        else:
                             perturbed_vals = perturb_statistical_scaled(x[perturb_indices], std_f, c)
                             tag_list.append('combined_node_attr_stat_scaled')
                        x_perturbed[perturb_indices] = perturbed_vals.to(x.dtype)
                        modified_data[node_type].x[target_idx] = x_perturbed
                        # Node already marked by struct, just add tag and track modification
                        nodes_modified_attr[node_type].add(target_idx) # Track attr modification

                    anomaly_tracking[node_type][target_idx].extend(tag_list) # Add specific tag

        # Store combined edges and attributes
        new_edges_all_types[target_edge_type].extend(new_edges_combined)
        new_edge_attrs_all_types[target_edge_type].extend(new_edge_attrs_combined)
        if reverse_edge_type in new_edges_all_types:
             new_edges_all_types[reverse_edge_type].extend([(v, u) for u, v in new_edges_combined])
             new_edge_attrs_all_types[reverse_edge_type].extend(new_edge_attrs_combined)


    # === Final Graph Update ===
    print("\nConsolidating graph modifications...")
    for edge_type in new_edges_all_types:
        if not new_edges_all_types[edge_type]: continue # Skip if no new edges for this type

        new_edges_tensor = torch.tensor(new_edges_all_types[edge_type], dtype=torch.long, device=device).t()
        new_attrs_tensor = torch.stack(new_edge_attrs_all_types[edge_type], dim=0).to(device)

        # Append Edges
        modified_data[edge_type].edge_index = torch.cat([
            modified_data[edge_type].edge_index, new_edges_tensor
        ], dim=1)

        # Append Attributes (handle potential initial None)
        if hasattr(modified_data[edge_type], 'edge_attr') and modified_data[edge_type].edge_attr is not None:
            modified_data[edge_type].edge_attr = torch.cat([
                modified_data[edge_type].edge_attr, new_attrs_tensor
            ], dim=0)
        elif new_attrs_tensor.shape[0] > 0:
             # If original was None, but we added some, need to pad original part
             num_orig = original_num_edges[edge_type]
             # Check if Type 3 modified existing attrs - if so, modified_data[et].edge_attr exists
             if hasattr(modified_data[edge_type], 'edge_attr') and modified_data[edge_type].edge_attr is not None and modified_data[edge_type].edge_attr.shape[0] == num_orig:
                 modified_data[edge_type].edge_attr = torch.cat([modified_data[edge_type].edge_attr, new_attrs_tensor], dim=0)
             else: # original was None and not modified by Type 3
                 print(f"Initializing edge_attr for {edge_type} before appending new attributes.")
                 # Create placeholder for original attrs if they were None
                 placeholder_attrs = torch.full((num_orig,) + tuple(edge_attr_shape), fill_value=0.0, dtype=edge_attr_dtype, device=device) # Or use mean?
                 modified_data[edge_type].edge_attr = torch.cat([placeholder_attrs, new_attrs_tensor], dim=0)


        # Append Edge Labels
        new_edge_labels = torch.ones(new_edges_tensor.shape[1], dtype=torch.long, device=device)
        gt_edge_labels_dict[edge_type] = torch.cat([
            gt_edge_labels_dict[edge_type], new_edge_labels
        ])

    # Ensure final shapes match and add .y attribute
    for edge_type, final_labels in gt_edge_labels_dict.items():
        if edge_type in modified_data.edge_types:
             current_num_edges = modified_data[edge_type].edge_index.shape[1]
             if len(final_labels) != current_num_edges:
                  print(f"ERROR FINAL: Mismatch gt_edge_labels ({len(final_labels)}) vs final edge_index ({current_num_edges}) for {edge_type}.")
                  # Attempt resize if possible (e.g., if only added edges)
                  if len(final_labels) < current_num_edges and len(final_labels) == original_num_edges[edge_type]:
                      print("Attempting to fix label length by padding with 1s for added edges...")
                      padding = torch.ones(current_num_edges - len(final_labels), dtype=torch.long, device=device)
                      gt_edge_labels_dict[edge_type] = torch.cat([final_labels, padding])
                      modified_data[edge_type].y = gt_edge_labels_dict[edge_type]
                  else:
                       print("Cannot fix label length mismatch.")
                       # Assign empty tensor or raise error? For now, assign potentially wrong labels
                       modified_data[edge_type].y = final_labels[:current_num_edges] if len(final_labels) > current_num_edges else torch.cat([final_labels, torch.zeros(current_num_edges-len(final_labels), dtype=torch.long, device=device)])


             else:
                 modified_data[edge_type].y = final_labels
        else:
            print(f"Warning: Edge type {edge_type} not found in final modified_data keys.")

    # Cleanup anomaly tracking dict
    final_anomaly_tracking = {}
    for nt, tracking_dict in anomaly_tracking.items():
        if nt == 'edge': continue
        filtered_tracking = {idx: list(set(tags)) for idx, tags in tracking_dict.items() if gt_node_labels[nt][idx] == 1}
        if filtered_tracking: final_anomaly_tracking[nt] = filtered_tracking

    # Format edge tracking (using original indices for Type 3, need mapping for added edges)
    # This part is complex to map back perfectly. Let's simplify: provide node tracking only for now.
    # Edge tracking requires mapping added edge indices back to their injection event.
    # Or we can return gt_edge_labels_dict separately.

    final_anom_nodes = sum(gt.sum().item() for gt in gt_node_labels.values())
    final_anom_edges = sum(gt.sum().item() for gt in gt_edge_labels_dict.values())
    print(f"Anomaly injection finished. Total anomalous nodes: {final_anom_nodes}, Total anomalous edges: {final_anom_edges}")

    # Return gt_edge_labels_dict directly as it matches the final graph structure
    return modified_data, gt_node_labels, gt_edge_labels_dict, final_anomaly_tracking

# Keep the original function signature for external calls
def inject_simplified_anomalies(data: HeteroData,
                                num_total_injections: int = 30, # Keep original arg for backward compatibility if ratio is not used
                                node_perturb_scale: float = 2.5,
                                edge_perturb_scale: float = 3.0,
                                anomaly_ratio: Optional[float] = None): # Add ratio internally
    """
    Injects simplified anomalies: Structural Only, Attribute Only, Combined.
    Also returns ground truth labels for injected edges.

    Anomalies can be specified either by the total number of injection events
    (`num_total_injections`) or by a desired ratio (`anomaly_ratio`) relative
    to the total number of nodes. If `anomaly_ratio` is provided, it takes precedence.

    Args:
        data: The original HeteroData object.
        num_total_injections (int, optional): Total number of injection events
            to perform *if* `anomaly_ratio` is None. Defaults to 30.
        node_perturb_scale (float): Scale factor for node feature noise relative
            to std dev. Defaults to 2.5.
        edge_perturb_scale (float): Scale factor for edge attribute noise relative
            to std dev. Defaults to 3.0.
        anomaly_ratio (Optional[float], optional): The desired ratio of injection
            events relative to the total number of nodes (providers + members).
            If specified, this overrides `num_total_injections`.
            For example, 0.01 means injection events equal to 1% of total nodes.
            Defaults to None.

    Returns:
        modified_data (HeteroData): HeteroData with anomalies.
        gt_node_labels (Dict[str, Tensor]): Dict {'provider': tensor, 'member': tensor}
                                            node labels (0/1).
        gt_edge_labels (Dict[Tuple, Tensor]): Dict {('src', 'rel', 'dst'): tensor}
                                              edge labels (0/1). Indices match the
                                              edge_index in modified_data.
        anomaly_tracking (Dict[str, Dict]): Dict detailing anomaly types per node.
    """
    modified_data = data.clone()
    target_edge_type = ('provider', 'to', 'member')
    reverse_edge_type = ('member', 'to', 'provider')

    if target_edge_type not in modified_data.edge_types:
        raise ValueError(f"Target edge type {target_edge_type} not found in data.")

    num_providers = modified_data['provider'].num_nodes
    num_members = modified_data['member'].num_nodes

    # --- Determine the number of injection events ---
    _num_total_injections_arg = num_total_injections # Keep track of the original argument

    if anomaly_ratio is not None:
        if not 0 < anomaly_ratio <= 1: # Allow up to 100% ratio
             raise ValueError("'anomaly_ratio' must be between 0 (exclusive) and 1 (inclusive).")
        total_nodes = num_providers + num_members
        if total_nodes == 0:
             actual_num_injections = 0
             print("Warning: Graph has 0 nodes. No anomalies will be injected.")
        else:
            # Calculate based on ratio of total nodes
            actual_num_injections = max(1, int(total_nodes * anomaly_ratio))
        print(f"Anomaly ratio {anomaly_ratio:.4f} specified. Calculated {actual_num_injections} injection events based on {total_nodes} total nodes.")
    else:
        actual_num_injections = _num_total_injections_arg
        if actual_num_injections is None or actual_num_injections < 0:
             raise ValueError("If 'anomaly_ratio' is not provided, 'num_total_injections' must be a non-negative integer.")
        print(f"Number of injection events {actual_num_injections} specified directly.")

    if actual_num_injections == 0:
         print("Warning: Number of injections is 0. Returning original data structure with empty ground truth additions.")
         # Initialize outputs correctly for the zero-injection case
         gt_node_labels = {nt: torch.zeros(n, dtype=torch.long) for nt, n in [('provider', num_providers), ('member', num_members)]}
         gt_edge_labels = {}
         for edge_type in modified_data.edge_types:
              num_edges = modified_data[edge_type].edge_index.shape[1]
              gt_edge_labels[edge_type] = torch.zeros(num_edges, dtype=torch.long)
              if hasattr(modified_data[edge_type], 'y'): # Check if y exists before assigning
                    modified_data[edge_type].y = gt_edge_labels[edge_type]
              else: # Add y if it doesn't exist
                  setattr(modified_data[edge_type], 'y', gt_edge_labels[edge_type])


         return modified_data, gt_node_labels, gt_edge_labels, {'provider': {}, 'member': {}}


    # --- Initialize Ground Truth (moved after num_injection calculation) ---
    gt_node_labels = {
        'provider': torch.zeros(num_providers, dtype=torch.long),
        'member': torch.zeros(num_members, dtype=torch.long)
    }
    gt_edge_labels = {}
    original_edge_indices = {} # Keep track to ensure we label original edges correctly
    for edge_type in modified_data.edge_types:
         num_edges = modified_data[edge_type].edge_index.shape[1]
         gt_edge_labels[edge_type] = torch.zeros(num_edges, dtype=torch.long)
         original_edge_indices[edge_type] = modified_data[edge_type].edge_index.clone()

    anomaly_tracking = {
        'provider': defaultdict(set),
        'member': defaultdict(set)
    }

    # --- Calculate Feature Statistics ---
    feature_std_dev = {}
    for node_type in data.node_types:
        if hasattr(data[node_type], 'x') and data[node_type].x.numel() > 0:
            std = data[node_type].x.std(dim=0)
            std[std == 0] = 1.0 # Avoid division by zero
            feature_std_dev[node_type] = std
        else:
            feature_std_dev[node_type] = None

    edge_attr_mean = torch.tensor([1.0]) # Default scalar tensor
    edge_attr_std = torch.tensor([1.0]) # Default scalar tensor
    attr_dtype = torch.float # Default dtype
    base_shape = [1] # Default shape for edge attr

    original_edge_attr = getattr(data[target_edge_type], 'edge_attr', None)
    if original_edge_attr is not None and original_edge_attr.numel() > 0:
        edge_attr_mean = original_edge_attr.mean(dim=0)
        std = original_edge_attr.std(dim=0)
        std[std == 0] = 1.0
        edge_attr_std = std
        attr_dtype = original_edge_attr.dtype
        base_shape = edge_attr_mean.shape # Use actual shape
        print(f"Edge Attr Stats: Mean={edge_attr_mean}, Std={edge_attr_std}")
    else:
        print("No edge attributes found or empty. Using default mean=1, std=1.")


    # --- Divide Injections into 3 Types ---
    num_per_type = math.ceil(actual_num_injections / 3.0)
    # Ensure correct distribution even if actual_num_injections is not divisible by 3
    injection_types = (['structural'] * num_per_type +
                       ['attribute'] * num_per_type +
                       ['combined'] * num_per_type)[:actual_num_injections] # Slice to exact number
    random.shuffle(injection_types)

    print(f"Injecting {actual_num_injections} anomalies:")
    print(f"  Structural: {injection_types.count('structural')}")
    print(f"  Attribute: {injection_types.count('attribute')}")
    print(f"  Combined: {injection_types.count('combined')}")

    # Keep track of edges added in this function call
    newly_added_edges_indices = {et: [] for et in modified_data.edge_types if et == target_edge_type or et == reverse_edge_type}

    # --- Injection Loop ---
    for injection_idx, injection_type in enumerate(injection_types):
        # --- Select Nodes ---
        # Use min(20, ...) to prevent selecting too many nodes if graph is small
        # Ensure we don't select more nodes than available
        max_p_size = min(20, num_providers) if num_providers > 1 else num_providers
        max_m_size = min(20, num_members) if num_members > 1 else num_members
        if max_p_size < 2 or max_m_size < 2:
            print(f"  Skipping injection {injection_idx+1} ({injection_type}): Not enough nodes to form a block (providers: {num_providers}, members: {num_members}).")
            continue

        size_provider = random.randint(2, max_p_size)
        size_member = random.randint(2, max_m_size)

        # Ensure we don't sample more unique nodes than exist
        providers_anom = np.random.choice(num_providers, size=min(size_provider, num_providers), replace=False)
        members_anom = np.random.choice(num_members, size=min(size_member, num_members), replace=False)

        # --- Mark Nodes ---
        for p in providers_anom:
            gt_node_labels['provider'][p] = 1
            anomaly_tracking['provider'][p].add(injection_type)
        for m in members_anom:
            gt_node_labels['member'][m] = 1
            anomaly_tracking['member'][m].add(injection_type)

        # --- Apply Anomaly Specifics ---
        new_edges_current_injection = []

        # == Structural or Combined Anomalies ==
        if injection_type == 'structural' or injection_type == 'combined':
            for p in providers_anom: anomaly_tracking['provider'][p].add('structural_block')
            for m in members_anom: anomaly_tracking['member'][m].add('structural_block')

            # Add Dense Block Edges
            full_block = random.random() < 0.5
            if full_block:
                edges_to_add = [(p, m) for p in providers_anom for m in members_anom]
            else:
                frac = random.uniform(0.2, 1.0)
                possible_edges = [(p, m) for p in providers_anom for m in members_anom]
                num_edges_to_select = max(1, int(frac * len(possible_edges)))
                edges_to_add = random.sample(possible_edges, num_edges_to_select)

            current_existing_edge_index = modified_data[target_edge_type].edge_index
            current_existing_set = set(tuple(map(int, x)) for x in current_existing_edge_index.t().tolist())

            new_edge_list = []
            new_edge_attr_list = []


            for (p, m) in edges_to_add:
                if (p, m) in current_existing_set: continue # Skip existing edges

                new_edges_current_injection.append((p, m))

                if injection_type == 'combined':
                    technique = random.choice(['outside', 'scaled'])
                    c_edge = random.uniform(edge_perturb_scale - 1.0, edge_perturb_scale + 1.0) * edge_attr_std
                    base_val = edge_attr_mean

                    if technique == 'outside':
                        new_val = base_val + c_edge * np.sign(np.random.randn())
                        anomaly_tracking['provider'][p].add('edge_feature_outside')
                        anomaly_tracking['member'][m].add('edge_feature_outside')
                    else: # 'scaled'
                        noise = torch.normal(mean=torch.zeros_like(c_edge), std=c_edge) # Match shape
                        new_val = base_val + noise
                        anomaly_tracking['provider'][p].add('edge_feature_scaled')
                        anomaly_tracking['member'][m].add('edge_feature_scaled')

                    new_val_tensor = new_val.clone().detach().reshape(base_shape).to(attr_dtype)
                    new_edge_attr_list.append(new_val_tensor)
                else: # Structural only - add edge with *normal* attributes
                    # Ensure the normal attribute has the correct shape and type
                    normal_attr = edge_attr_mean.clone().detach().reshape(base_shape).to(attr_dtype)
                    new_edge_attr_list.append(normal_attr)

                new_edge_list.append((p, m))

            # Append new edges and their labels
            if new_edge_list:
                start_idx = modified_data[target_edge_type].edge_index.shape[1]
                new_edge_tensor = torch.tensor(new_edge_list, dtype=torch.long).t()
                # Only stack if list is not empty
                new_edge_attr_tensor = torch.stack(new_edge_attr_list, dim=0) if new_edge_attr_list else torch.empty((0,) + tuple(base_shape), dtype=attr_dtype)

                modified_data[target_edge_type].edge_index = torch.cat(
                    [modified_data[target_edge_type].edge_index, new_edge_tensor], dim=1
                )
                # Only concat attributes if they existed originally or if new ones were created
                if original_edge_attr is not None or new_edge_attr_tensor.shape[0] > 0:
                    # Ensure original attributes exist before concatenating
                    if not hasattr(modified_data[target_edge_type], 'edge_attr') or modified_data[target_edge_type].edge_attr is None:
                         # If original was None but we are adding some, initialize with empty tensor of correct shape
                         if original_edge_attr is None:
                            original_num_edges = gt_edge_labels[target_edge_type].shape[0] # Num original edges
                            modified_data[target_edge_type].edge_attr = torch.empty((original_num_edges,) + tuple(base_shape), dtype=attr_dtype)
                            print(f"Warning: Original edge_attr was None for {target_edge_type}, initializing.")
                         else: # Should not happen if original_edge_attr was not None, but safety check
                             modified_data[target_edge_type].edge_attr = original_edge_attr

                    # Now concatenate safely
                    modified_data[target_edge_type].edge_attr = torch.cat(
                        [modified_data[target_edge_type].edge_attr, new_edge_attr_tensor], dim=0
                    )

                # Add labels (1 for anomalous) for the new edges
                new_edge_labels = torch.ones(new_edge_tensor.shape[1], dtype=torch.long)
                gt_edge_labels[target_edge_type] = torch.cat(
                     [gt_edge_labels[target_edge_type], new_edge_labels]
                )
                newly_added_edges_indices[target_edge_type].extend(list(range(start_idx, start_idx + len(new_edge_list))))


                # Handle reverse edges
                if reverse_edge_type in modified_data.edge_types:
                    start_idx_rev = modified_data[reverse_edge_type].edge_index.shape[1]
                    reverse_new_edges = torch.stack([new_edge_tensor[1], new_edge_tensor[0]], dim=0)
                    modified_data[reverse_edge_type].edge_index = torch.cat(
                        [modified_data[reverse_edge_type].edge_index, reverse_new_edges], dim=1
                    )

                    # Handle reverse edge attributes similarly
                    if hasattr(data[reverse_edge_type], 'edge_attr') and data[reverse_edge_type].edge_attr is not None or new_edge_attr_tensor.shape[0] > 0:
                         original_reverse_attr = getattr(data[reverse_edge_type], 'edge_attr', None)
                         if not hasattr(modified_data[reverse_edge_type], 'edge_attr') or modified_data[reverse_edge_type].edge_attr is None:
                             if original_reverse_attr is None:
                                 original_num_rev_edges = gt_edge_labels[reverse_edge_type].shape[0]
                                 modified_data[reverse_edge_type].edge_attr = torch.empty((original_num_rev_edges,) + tuple(base_shape), dtype=attr_dtype)
                                 print(f"Warning: Original edge_attr was None for {reverse_edge_type}, initializing.")
                             else:
                                modified_data[reverse_edge_type].edge_attr = original_reverse_attr

                         modified_data[reverse_edge_type].edge_attr = torch.cat(
                               [modified_data[reverse_edge_type].edge_attr, new_edge_attr_tensor], dim=0
                         )

                    # Add labels for reverse edges too
                    gt_edge_labels[reverse_edge_type] = torch.cat(
                        [gt_edge_labels[reverse_edge_type], new_edge_labels]
                    )
                    newly_added_edges_indices[reverse_edge_type].extend(list(range(start_idx_rev, start_idx_rev + len(new_edge_list))))

        # == Attribute or Combined Anomalies ==
        if injection_type == 'attribute' or injection_type == 'combined':
            # Node feature perturbation
            for node_type, nodes in [('provider', providers_anom), ('member', members_anom)]:
                 if feature_std_dev.get(node_type) is None: continue

                 # Ensure features exist for this node type in modified_data
                 if not hasattr(modified_data[node_type], 'x') or modified_data[node_type].x is None:
                      print(f"Warning: Node type {node_type} has no features 'x'. Skipping attribute anomaly injection.")
                      continue

                 std_devs = feature_std_dev[node_type].to(modified_data[node_type].x.device)

                 for idx in nodes:
                     x = modified_data[node_type].x[idx]
                     num_features = x.size(0)
                     if num_features == 0: continue # Skip if node has no features

                     num_to_perturb = max(1, int(0.3 * num_features)) # Perturb 30%
                     perturb_indices = np.random.choice(num_features, num_to_perturb, replace=False)

                     noise_scale = random.uniform(node_perturb_scale - 0.5, node_perturb_scale + 0.5)
                     # Ensure std_devs used here corresponds to the selected features
                     # Make sure noise has same shape as selected features' std dev
                     noise = torch.normal(mean=torch.zeros_like(std_devs[perturb_indices]),
                                          std=std_devs[perturb_indices] * noise_scale)
                     noise = noise.to(x.device) # Ensure noise is on the same device

                     x_perturbed = x.clone()
                     x_perturbed[perturb_indices] += noise
                     modified_data[node_type].x[idx] = x_perturbed
                     anomaly_tracking[node_type][idx].add('node_feature_relative')

    # --- Final Cleanup ---
    final_anomaly_tracking = {
         nt: dict(at) for nt, at in anomaly_tracking.items() if at # Convert defaultdict back, remove empty node types
    }
    # Filter tracking to only include nodes marked as anomalous
    if 'provider' in final_anomaly_tracking:
        final_anomaly_tracking['provider'] = {i: types for i, types in final_anomaly_tracking['provider'].items() if gt_node_labels['provider'][i] == 1}
    if 'member' in final_anomaly_tracking:
        final_anomaly_tracking['member'] = {i: types for i, types in final_anomaly_tracking['member'].items() if gt_node_labels['member'][i] == 1}


    # Add edge labels to the data object itself for convenience ('y' attribute)
    # Ensure the shapes match after adding new edges
    for edge_type, labels in gt_edge_labels.items():
        if edge_type in modified_data.edge_types:
             current_num_edges = modified_data[edge_type].edge_index.shape[1]
             if len(labels) != current_num_edges:
                  print(f"Error: Mismatch between gt_edge_labels length ({len(labels)}) and final edge_index length ({current_num_edges}) for {edge_type}. Check injection logic.")
                  # Handle error appropriately - perhaps raise exception or skip assignment
                  # For now, let's print error and skip assigning 'y' for this type
                  continue
             # Only assign if 'y' doesn't exist or if it needs updating
             if not hasattr(modified_data[edge_type], 'y') or modified_data[edge_type].y.shape[0] != current_num_edges:
                  modified_data[edge_type].y = labels

    print("Anomaly injection finished.")
    # Note: gt_edge_labels dict is also returned separately
    return modified_data, gt_node_labels, gt_edge_labels, final_anomaly_tracking

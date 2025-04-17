import torch
import random
import numpy as np
from torch_geometric.data import HeteroData
from collections import defaultdict
from typing import Tuple, List, Dict, Optional, Literal, Any
import torch.nn.functional as F

# === Configuration for Scenarios ===

# --- Default Scenario Parameters ---
DEFAULT_PARAMS = {
    # Structural - Providers
    "k_p_clique": 3,
    "p_link_clique": 0.95,
    # Structural - Members
    "k_m_clique": 15,
    "p_link_member_clique": 0.95,
    "k_providers_theft_touring": 10,
    "mode_theft_touring": 'predefined', # or 'degree_based'
    # Structural - Both
    "k_p_collusion": 3,
    "k_m_collusion": 10,
    "collusion_partial_block_frac_range": (0.7, 1.0), # Denser default
    # Attribute - General
    "swap_pool_size": 50,
    "swap_metric": 'euclidean',
    # Attribute Tweaking
    "attr_perturb_std_factor_range": (3.0, 6.0), # How many std devs away for numeric attrs
    "attr_perturb_frac": 0.3 # Fraction of features to perturb if multiple exist
}

# --- Scenario Definitions ---
# Structure: scenario_name: {type: 'structural'/'attribute', node_type: 'provider'/'member'/'both', params: [list_of_param_names_from_DEFAULT_PARAMS], features: [optional_list_of_features_needed]}
SCENARIOS = {
    # --- Structural Scenarios ---
    "over_referral_clique": {
        "type": "structural",
        "node_type": "provider",
        "params": ["k_p_clique", "p_link_clique"],
        "description": "Group of providers referring patients primarily amongst themselves.",
        "requires_degree": True # Needs degree info for selection
    },
    "member_clique": {
        "type": "structural",
        "node_type": "member",
        "params": ["k_m_clique", "p_link_member_clique"],
        "description": "Group of members seeing the exact same provider network.",
         "requires_degree": False
    },
    "identity_theft_doctor_touring": {
        "type": "structural",
        "node_type": "member",
        "params": ["k_providers_theft_touring", "mode_theft_touring"],
        "description": "Single member accessing an unusually high number of distinct providers.",
        "requires_degree": True # If mode is 'degree_based'
    },
    "collusion_ring": {
        "type": "structural",
        "node_type": "both", # Involves both node types directly
        "params": ["k_p_collusion", "k_m_collusion", "collusion_partial_block_frac_range"],
        "description": "Dense interaction block between specific providers and members.",
         "requires_degree": False
    },
    # --- Attribute Scenarios ---
    "identity_swap": {
        "type": "attribute",
        "node_type": "both", # Can apply to either provider or member
        "params": ["swap_pool_size", "swap_metric"],
        "description": "Swap feature vector with a dissimilar node.",
        "requires_features": True # Needs node features 'x'
    },
    "provider_over_reimbursement": {
        "type": "attribute",
        "node_type": "provider",
        "params": ["attr_perturb_std_factor_range"], # Uses generic perturbation factor
        "features": ["claim_is_reimbursement"], # Specific feature target
        "description": "Provider submitting unusually high proportion of reimbursement claims.",
        "requires_features": True
    },
    "member_over_consulting": {
        "type": "attribute",
        "node_type": "member",
        "params": ["attr_perturb_std_factor_range"],
        "features": ["claim_period_mean", "claim_period_median"], # Specific feature targets
        "description": "Member seeking consultations/services far more frequently than normal.",
        "requires_features": True
    },
    # --- Placeholder for more scenarios ---
    # "provider_upcoding": { ... }
    # "member_excessive_claims": { ... }
}


# === Helper Functions ===

def _compute_graph_stats(data: HeteroData, node_features_to_stat: Dict[str, List[str]], device: torch.device, edge_attr_name: str = 'edge_attr') -> Dict[str, Any]:
    """Computes necessary statistics from the graph."""
    print("Computing graph statistics...")
    stats = {'node': defaultdict(dict), 'edge': defaultdict(dict)}
    # device is now passed as an argument

    # Node Degrees
    for ntype in data.node_types:
        stats['node'][ntype]['degree'] = defaultdict(int)
        # Ensure total_degree tensor is created on the correct device
        total_degree = torch.zeros(data[ntype].num_nodes, dtype=torch.long, device=device)
        for etype in data.edge_types:
            # Ensure edge indices are on the correct device before using them
            edge_index = data[etype].edge_index.to(device)
            if ntype == etype[0]: # Source node
                idx, counts = torch.unique(edge_index[0], return_counts=True)
                 # Ensure idx and counts are on the correct device if they aren't already
                idx, counts = idx.to(device), counts.to(device)
                total_degree.scatter_add_(0, idx, counts) # Use scatter_add_ for safe inplace addition
            if ntype == etype[2]: # Destination node
                idx, counts = torch.unique(edge_index[1], return_counts=True)
                idx, counts = idx.to(device), counts.to(device)
                total_degree.scatter_add_(0, idx, counts)

        stats['node'][ntype]['degree']['values'] = total_degree
        if total_degree.numel() > 0:
            # Ensure calculations happen on the correct device
            total_degree_float = total_degree.float()
            stats['node'][ntype]['degree']['mean'] = total_degree_float.mean().item()
            stats['node'][ntype]['degree']['std'] = total_degree_float.std().item()
            stats['node'][ntype]['degree']['p90'] = torch.quantile(total_degree_float, 0.9).item() if total_degree.numel() > 0 else 0
            stats['node'][ntype]['degree']['max'] = total_degree.max().item() if total_degree.numel() > 0 else 0
        else:
             stats['node'][ntype]['degree']['mean'] = 0
             stats['node'][ntype]['degree']['std'] = 1
             stats['node'][ntype]['degree']['p90'] = 0
             stats['node'][ntype]['degree']['max'] = 0
        print(f"  Node Type '{ntype}': Found {data[ntype].num_nodes} nodes. Degree (mean={stats['node'][ntype]['degree']['mean']:.2f}, std={stats['node'][ntype]['degree']['std']:.2f}, p90={stats['node'][ntype]['degree']['p90']:.2f})")


    # Node Features
    for ntype, features in node_features_to_stat.items():
        if ntype in data.node_types and hasattr(data[ntype], 'x') and data[ntype].x is not None:
            # Ensure node features are on the correct device
            x = data[ntype].x.to(device)
            stats['node'][ntype]['features'] = {}
            stats['node'][ntype]['features']['all_means'] = x.mean(dim=0)
            stats['node'][ntype]['features']['all_stds'] = x.std(dim=0)
            # Handle std=0
            stats['node'][ntype]['features']['all_stds'][stats['node'][ntype]['features']['all_stds'] == 0] = 1.0
            stats['node'][ntype]['features']['map'] = {} # Store index of each named feature

            print(f"  Computing stats for {ntype} features on device: {x.device}") # Debug print
            # --- THIS PART NEEDS USER INPUT: Mapping feature names to indices ---
            print(f"  Warning: Node feature statistics computed, but mapping from names {features} to indices in 'x' is assumed sequential.")
            if hasattr(data[ntype], 'feature_names'): # Ideal case: if feature names are stored
                feature_map = {name: i for i, name in enumerate(data[ntype].feature_names)}
            else: # Fallback: Assume sequential order based on user's list
                feature_map = {name: i for i, name in enumerate(features)} # Simplified assumption

            for i, feat_name in enumerate(features):
                 if feat_name in feature_map:
                    feat_idx = feature_map[feat_name]
                    if feat_idx < x.shape[1]:
                        stats['node'][ntype]['features']['map'][feat_name] = feat_idx
                        # Ensure stats tensors used here are on the correct device (they should be)
                        mean_val = stats['node'][ntype]['features']['all_means'][feat_idx].item()
                        std_val = stats['node'][ntype]['features']['all_stds'][feat_idx].item()
                        print(f"    Feature '{feat_name}' (Index {feat_idx}): Mean={mean_val:.2f}, Std={std_val:.2f}")
                    else:
                        print(f"    Warning: Feature '{feat_name}' index {feat_idx} out of bounds for {ntype} features (shape {x.shape}). Skipping stats.")
                 else:
                    print(f"    Warning: Feature '{feat_name}' not found in assumed feature map for {ntype}. Skipping stats.")

        else:
            print(f"  Node Type '{ntype}': No features 'x' found or requested. Skipping feature stats.")

    # Edge Attributes (assuming single attribute for now)
    target_edge_type = ('provider', 'to', 'member') # Hardcoded for now
    if target_edge_type in data.edge_types and hasattr(data[target_edge_type], edge_attr_name) and data[target_edge_type][edge_attr_name] is not None:
        # Ensure edge attributes are on the correct device
        ea = data[target_edge_type][edge_attr_name].to(device)
        if ea.numel() > 0:
             # Assuming edge_attr is [N_edges, 1] or [N_edges] for claim count
             ea_flat = ea.flatten().float() # This should be on the correct device
             stats['edge'][target_edge_type]['mean'] = ea_flat.mean().item()
             stats['edge'][target_edge_type]['std'] = ea_flat.std().item()
             stats['edge'][target_edge_type]['p99'] = torch.quantile(ea_flat, 0.99).item()
             stats['edge'][target_edge_type]['p01'] = torch.quantile(ea_flat, 0.01).item()
             stats['edge'][target_edge_type]['dtype'] = ea.dtype
             stats['edge'][target_edge_type]['shape'] = list(ea.shape[1:]) if ea.dim() > 1 else [1]
             # Handle std=0
             if stats['edge'][target_edge_type]['std'] == 0:
                 stats['edge'][target_edge_type]['std'] = 1.0
             print(f"  Edge Type '{target_edge_type}': Found {ea.shape[0]} edges. Attribute '{edge_attr_name}' (mean={stats['edge'][target_edge_type]['mean']:.2f}, std={stats['edge'][target_edge_type]['std']:.2f}, p99={stats['edge'][target_edge_type]['p99']:.2f}, p01={stats['edge'][target_edge_type]['p01']:.2f})")
        else:
            print(f"  Edge Type '{target_edge_type}': Attribute '{edge_attr_name}' is empty. Using default stats.")
            stats['edge'][target_edge_type]['mean'] = 1.0
            stats['edge'][target_edge_type]['std'] = 1.0
            stats['edge'][target_edge_type]['p99'] = 5.0 # Default guess
            stats['edge'][target_edge_type]['p01'] = 0.0 # Default guess
            stats['edge'][target_edge_type]['dtype'] = torch.float
            stats['edge'][target_edge_type]['shape'] = [1]
    else:
        print(f"  Edge Type '{target_edge_type}': No attribute '{edge_attr_name}' found. Using default stats.")
        stats['edge'][target_edge_type]['mean'] = 1.0
        stats['edge'][target_edge_type]['std'] = 1.0
        stats['edge'][target_edge_type]['p99'] = 5.0 # Default guess
        stats['edge'][target_edge_type]['p01'] = 0.0 # Default guess
        stats['edge'][target_edge_type]['dtype'] = torch.float
        stats['edge'][target_edge_type]['shape'] = [1]

    print("Statistics computation finished.")
    return stats

def _select_nodes_for_anomaly(num_nodes: int, percentage: float) -> List[int]:
    """Selects a percentage of node indices without replacement."""
    num_to_select = int(num_nodes * percentage)
    if num_to_select > num_nodes:
        num_to_select = num_nodes
    if num_nodes <= 0: # Avoid error with range(0)
        return []
    return random.sample(range(num_nodes), num_to_select)

def _partition_nodes(selected_nodes: List[int], lambda_structural: float) -> Tuple[List[int], List[int]]:
    """Partitions selected nodes into structural and attribute pools."""
    num_structural = int(len(selected_nodes) * lambda_structural)
    random.shuffle(selected_nodes)
    structural_nodes = selected_nodes[:num_structural]
    attribute_nodes = selected_nodes[num_structural:]
    return structural_nodes, attribute_nodes

def _update_tracking(anomaly_tracking, scope, node_type, node_idx, tags):
    """Helper to add tags to the tracking dictionary."""
    if not isinstance(tags, list):
        tags = [tags]
    if scope == 'node':
        anomaly_tracking['node'][node_type][node_idx].extend(tags)
    elif scope == 'edge':
        # key is (edge_type, edge_index_in_final_tensor)
        # This helper might need refinement if used for edges added mid-process
        edge_key = (node_type, node_idx) # Assuming node_type holds edge_type, node_idx holds edge_idx
        anomaly_tracking['edge'][edge_key].extend(tags)
    else: # Both nodes of an edge
         u_type, v_type = node_type # Expecting a tuple like ('provider', 'member')
         u_idx, v_idx = node_idx # Expecting a tuple like (10, 25)
         anomaly_tracking['node'][u_type][u_idx].extend(tags)
         anomaly_tracking['node'][v_type][v_idx].extend(tags)


def _add_edges_and_update_labels(
    modified_data: HeteroData,
    gt_edge_labels_dict: Dict[Tuple, torch.Tensor],
    edge_type: Tuple,
    new_edges: List[Tuple[int, int]],
    device: torch.device, # <-- No default, moved BEFORE defaults
    new_edge_attrs: Optional[List[torch.Tensor]] = None # <-- Has default, now at the end
    ) -> int:
    if not new_edges:
        return 0

    num_added = len(new_edges)
    # Ensure new tensors are created on the correct device
    new_edges_tensor = torch.tensor(new_edges, dtype=torch.long, device=device).t()

    # Ensure original edge index is on the correct device before concatenation
    orig_edge_index = modified_data[edge_type].edge_index.to(device)
    modified_data[edge_type].edge_index = torch.cat([
        orig_edge_index, new_edges_tensor
    ], dim=1)

    # Append Attributes
    if new_edge_attrs:
        # Ensure attributes are stacked correctly and on the right device
        new_attrs_tensor = torch.stack(new_edge_attrs).to(device)
        if hasattr(modified_data[edge_type], 'edge_attr') and modified_data[edge_type].edge_attr is not None:
             # Ensure original attributes are on the correct device
             orig_attrs = modified_data[edge_type].edge_attr.to(device)
             # Shape adjustment logic remains the same
             if orig_attrs.dim() == 1 and new_attrs_tensor.dim() == 2 and new_attrs_tensor.shape[1] == 1:
                 orig_attrs = orig_attrs.unsqueeze(1)
             elif orig_attrs.dim() == 2 and orig_attrs.shape[1] == 1 and new_attrs_tensor.dim() == 1:
                 new_attrs_tensor = new_attrs_tensor.unsqueeze(1)

             if orig_attrs.shape[1:] == new_attrs_tensor.shape[1:]:
                 modified_data[edge_type].edge_attr = torch.cat([orig_attrs, new_attrs_tensor], dim=0)
             else:
                  print(f"Warning: Shape mismatch appending edge attributes for {edge_type}. Original: {orig_attrs.shape}, New: {new_attrs_tensor.shape}. Skipping append.")

        elif num_added > 0: # Original was None or didn't exist
             num_orig_edges = modified_data[edge_type].edge_index.shape[1] - num_added
             if num_orig_edges > 0:
                 print(f"Warning: Initializing edge_attr for {edge_type} before appending. Original edges get 0-value attrs.")
                 placeholder_shape = (num_orig_edges,) + tuple(new_attrs_tensor.shape[1:])
                 # Create placeholder on the correct device
                 placeholder_attrs = torch.zeros(placeholder_shape, dtype=new_attrs_tensor.dtype, device=device)
                 modified_data[edge_type].edge_attr = torch.cat([placeholder_attrs, new_attrs_tensor], dim=0)
             else:
                 modified_data[edge_type].edge_attr = new_attrs_tensor

    # Append Edge Labels
    # Ensure original labels are on the correct device
    orig_labels = gt_edge_labels_dict[edge_type].to(device)
    # Create new labels on the correct device
    new_edge_labels = torch.ones(num_added, dtype=torch.long, device=device)
    gt_edge_labels_dict[edge_type] = torch.cat([
        orig_labels, new_edge_labels
    ])

    return num_added

# === Scenario Injection Functions ===

def _inject_structural_scenario(
    scenario_name: str,
    scenario_config: Dict,
    modified_data: HeteroData,
    gt_node_labels: Dict[str, torch.Tensor],
    anomaly_tracking: Dict,
    nodes_involved: Dict[str, List[int]], # Nodes pre-selected for this instance
    stats: Dict,
    params: Dict,
    device: torch.device, # Explicitly require device
    target_edge_type: Tuple = ('provider', 'to', 'member'),
    reverse_edge_type: Optional[Tuple] = ('member', 'to', 'provider')
) -> Tuple[List[Tuple], List[torch.Tensor]]: # Returns (new_edges_fwd, new_attrs_fwd)
    """Injects a structural anomaly based on the scenario."""
    print(f"    Injecting Structural Scenario: {scenario_name}")
    provider_type, member_type = target_edge_type[0], target_edge_type[2]
    new_edges_fwd = []
    new_edge_attrs_fwd = []
    nodes_labelled = defaultdict(list)

    # --- Default Edge Attribute ---
    edge_stat = stats['edge'].get(target_edge_type, {})
    # Create tensor on the correct device
    default_attr_val = torch.tensor([edge_stat.get('mean', 1.0)], device=device, dtype=edge_stat.get('dtype', torch.float))
    default_attr_shape = edge_stat.get('shape', [1])

    # --- Scenario Logic ---
    if scenario_name == "over_referral_clique":
        providers = nodes_involved[provider_type]
        k_p = len(providers)
        p_link = params['p_link_clique']
        k_m_target = 5
        if modified_data[member_type].num_nodes < k_m_target: return [], []
        target_members = random.sample(range(modified_data[member_type].num_nodes), k_m_target)

        possible_edges = [(p, m) for p in providers for m in target_members]
        edges_to_add_indices = [i for i, _ in enumerate(possible_edges) if random.random() < p_link]
        edges_to_add = [possible_edges[i] for i in edges_to_add_indices]

        # Ensure edge index is on the correct device for checking existing edges
        current_edge_index = modified_data[target_edge_type].edge_index.to(device)
        current_existing_set = set(tuple(map(int, x)) for x in current_edge_index.t().tolist())
        new_edges_struct = [edge for edge in edges_to_add if tuple(edge) not in current_existing_set]

        if not new_edges_struct: return [], []

        new_edges_fwd.extend(new_edges_struct)
        attr_val = default_attr_val.clone().detach().reshape(default_attr_shape)
        # Ensure attribute list contains tensors (already done by default_attr_val)
        new_edge_attrs_fwd.extend([attr_val] * len(new_edges_struct))

        tag = f"Structural/{scenario_name}"
        for p_idx in providers:
            if gt_node_labels[provider_type][p_idx] == 0: nodes_labelled[provider_type].append(p_idx)
            gt_node_labels[provider_type][p_idx] = 1 # In-place modification ok if tensor is on device
            _update_tracking(anomaly_tracking, 'node', provider_type, p_idx, tag)

    elif scenario_name == "member_clique":
        members = nodes_involved[member_type]
        k_m = len(members)
        p_link = params['p_link_member_clique']
        k_p_target = 3
        if modified_data[provider_type].num_nodes < k_p_target: return [], []
        target_providers = random.sample(range(modified_data[provider_type].num_nodes), k_p_target)

        possible_edges = [(p, m) for p in target_providers for m in members]
        edges_to_add_indices = [i for i, _ in enumerate(possible_edges) if random.random() < p_link]
        edges_to_add = [possible_edges[i] for i in edges_to_add_indices]

        current_edge_index = modified_data[target_edge_type].edge_index.to(device)
        current_existing_set = set(tuple(map(int, x)) for x in current_edge_index.t().tolist())
        new_edges_struct = [edge for edge in edges_to_add if tuple(edge) not in current_existing_set]

        if not new_edges_struct: return [], []

        new_edges_fwd.extend(new_edges_struct)
        # Create tensor on the correct device
        attr_val = torch.tensor([1.0], device=device, dtype=edge_stat.get('dtype', torch.float)).reshape(default_attr_shape)
        new_edge_attrs_fwd.extend([attr_val] * len(new_edges_struct))

        tag = f"Structural/{scenario_name}"
        for m_idx in members:
             if gt_node_labels[member_type][m_idx] == 0: nodes_labelled[member_type].append(m_idx)
             gt_node_labels[member_type][m_idx] = 1
             _update_tracking(anomaly_tracking, 'node', member_type, m_idx, tag)

    elif scenario_name == "identity_theft_doctor_touring":
        member = nodes_involved[member_type][0]
        num_providers = params['k_providers_theft_touring']
        if modified_data[provider_type].num_nodes < num_providers: return [], []
        target_providers = random.sample(range(modified_data[provider_type].num_nodes), num_providers)
        edges_to_add = [(p, member) for p in target_providers]

        current_edge_index = modified_data[target_edge_type].edge_index.to(device)
        current_existing_set = set(tuple(map(int, x)) for x in current_edge_index.t().tolist())
        new_edges_struct = [edge for edge in edges_to_add if tuple(edge) not in current_existing_set]

        if not new_edges_struct: return [], []

        new_edges_fwd.extend(new_edges_struct)
        mean_claims = edge_stat.get('mean', 1.0)
        # Create tensors on the correct device
        low_med_attrs = [torch.tensor([random.uniform(1.0, max(1.1, mean_claims))], device=device, dtype=edge_stat.get('dtype', torch.float)).reshape(default_attr_shape) for _ in new_edges_struct]
        new_edge_attrs_fwd.extend(low_med_attrs)

        tag = f"Structural/{scenario_name}"
        if gt_node_labels[member_type][member] == 0: nodes_labelled[member_type].append(member)
        gt_node_labels[member_type][member] = 1
        _update_tracking(anomaly_tracking, 'node', member_type, member, tag)

    elif scenario_name == "collusion_ring":
        # Assuming nodes_involved has been populated correctly for both types by planning step
        providers = nodes_involved.get(provider_type, [])
        members = nodes_involved.get(member_type, [])
        if not providers or not members:
             print(f"  Skipping {scenario_name}: Missing required nodes ({provider_type}={len(providers)}, {member_type}={len(members)}). Check planning logic.")
             return [],[]

        k_p, k_m = len(providers), len(members)
        frac_range = params['collusion_partial_block_frac_range']
        is_partial = random.random() < 0.5
        possible_edges = [(p, m) for p in providers for m in members]

        if is_partial and len(possible_edges) > 0:
            frac = random.uniform(frac_range[0], frac_range[1])
            edges_to_consider = random.sample(possible_edges, max(1, int(frac * len(possible_edges))))
        else:
            edges_to_consider = possible_edges

        current_edge_index = modified_data[target_edge_type].edge_index.to(device)
        current_existing_set = set(tuple(map(int, x)) for x in current_edge_index.t().tolist())
        new_edges_struct = [edge for edge in edges_to_consider if tuple(edge) not in current_existing_set]

        if not new_edges_struct: return [], []

        new_edges_fwd.extend(new_edges_struct)
        attr_val = default_attr_val.clone().detach().reshape(default_attr_shape)
        new_edge_attrs_fwd.extend([attr_val] * len(new_edges_struct))

        tag = f"Structural/{scenario_name}"
        for p_idx in providers:
            if gt_node_labels[provider_type][p_idx] == 0: nodes_labelled[provider_type].append(p_idx)
            gt_node_labels[provider_type][p_idx] = 1
            _update_tracking(anomaly_tracking, 'node', provider_type, p_idx, tag)
        for m_idx in members:
            if gt_node_labels[member_type][m_idx] == 0: nodes_labelled[member_type].append(m_idx)
            gt_node_labels[member_type][m_idx] = 1
            _update_tracking(anomaly_tracking, 'node', member_type, m_idx, tag)

    else:
        print(f"Warning: Unknown structural scenario '{scenario_name}'. Skipping.")
        return [], []

    print(f"      Added {len(new_edges_fwd)} new edges for {scenario_name}.")
    return new_edges_fwd, new_edge_attrs_fwd


def _inject_attribute_scenario(
    scenario_name: str,
    scenario_config: Dict,
    modified_data: HeteroData,
    gt_node_labels: Dict[str, torch.Tensor],
    anomaly_tracking: Dict,
    target_node_type: str,
    target_node_idx: int,
    stats: Dict,
    params: Dict,
    device: torch.device # Explicitly require device
) -> bool: # Returns True if modification occurred
    """Injects an attribute anomaly based on the scenario."""
    print(f"    Injecting Attribute Scenario: {scenario_name} on {target_node_type} {target_node_idx}")
    modified = False
    tag_base = f"Attribute/{scenario_name}"

    # Check if features exist and move to device
    if not hasattr(modified_data[target_node_type], 'x') or modified_data[target_node_type].x is None:
         print(f"      Skipping {scenario_name}: Node type '{target_node_type}' has no features 'x'.")
         return False
    node_x = modified_data[target_node_type].x.to(device) # Ensure features are on device

    if target_node_type not in stats['node'] or 'features' not in stats['node'][target_node_type]:
        print(f"      Skipping {scenario_name}: Node type '{target_node_type}' has no precomputed feature statistics.")
        return False

    node_stats = stats['node'][target_node_type]['features']
    # Ensure stats tensors are on device
    all_means = node_stats['all_means'].to(device)
    all_stds = node_stats['all_stds'].to(device)

    original_features = node_x[target_node_idx].clone()
    perturbed_features = original_features.clone()

    if scenario_name == "identity_swap":
        pool_size = params['swap_pool_size']
        metric = params['swap_metric']
        num_nodes = node_x.shape[0] # Use shape from device tensor

        potential_pool_indices = list(set(range(num_nodes)) - {target_node_idx})
        if not potential_pool_indices: return False
        current_pool_size = min(pool_size, len(potential_pool_indices))
        pool_indices = np.random.choice(potential_pool_indices, size=current_pool_size, replace=False)
        pool_indices_tensor = torch.tensor(pool_indices, device=device) # Move indices to device

        target_feat = perturbed_features.unsqueeze(0)
        pool_feats = node_x[pool_indices_tensor] # Index with device tensor

        # Ensure distance calculations happen on the correct device
        if metric == 'euclidean': distances = torch.cdist(target_feat, pool_feats, p=2.0).squeeze(0)
        elif metric == 'cosine': distances = 1.0 - F.cosine_similarity(target_feat, pool_feats, dim=1)
        else: raise ValueError("Invalid node_attr_swap_metric.")

        distances = torch.nan_to_num(distances, nan=-torch.inf)
        if distances.numel() == 0 or torch.all(torch.isinf(distances)): return False

        most_distant_pool_idx = torch.argmax(distances).item()
        # Use original numpy indices to get the final python int index
        swap_partner_idx = pool_indices[most_distant_pool_idx]

        # Perform swap using integer indices on the device tensor
        feat_swap_original = node_x[swap_partner_idx].clone()
        modified_data[target_node_type].x[target_node_idx] = feat_swap_original
        modified_data[target_node_type].x[swap_partner_idx] = original_features

        # Mark both and track
        gt_node_labels[target_node_type][target_node_idx] = 1
        gt_node_labels[target_node_type][swap_partner_idx] = 1
        _update_tracking(anomaly_tracking, 'node', target_node_type, target_node_idx, tag_base + "_target")
        _update_tracking(anomaly_tracking, 'node', target_node_type, swap_partner_idx, tag_base + "_partner")
        modified = True

    else: # General attribute perturbation logic
        feature_names_to_perturb = scenario_config.get('features', [])
        if not feature_names_to_perturb:
             num_features = perturbed_features.size(0)
             num_to_perturb = max(1, int(params['attr_perturb_frac'] * num_features))
             feature_indices = np.random.choice(num_features, num_to_perturb, replace=False).tolist()
             print(f"      Perturbing {num_to_perturb} random features for {scenario_name}.")
        else:
            feature_indices = []
            for fname in feature_names_to_perturb:
                if fname in node_stats['map']:
                    feature_indices.append(node_stats['map'][fname])
                else:
                    print(f"      Warning: Feature '{fname}' for scenario {scenario_name} not found in stats map for {target_node_type}. Skipping.")
            if not feature_indices: return False
            print(f"      Perturbing features {feature_names_to_perturb} (indices {feature_indices}) for {scenario_name}.")

        # Ensure indices tensor is on device if used for indexing stats
        feature_indices_tensor = torch.tensor(feature_indices, device=device, dtype=torch.long)

        # --- Perturbation Logic ---
        c_low, c_high = params['attr_perturb_std_factor_range']
        c = random.uniform(c_low, c_high)
        # Index stats tensors with device tensor
        means = all_means[feature_indices_tensor]
        stds = all_stds[feature_indices_tensor]

        # Apply scenario-specific logic
        if scenario_name == "provider_over_reimbursement":
             # Create perturbation on device
            perturbed_features[feature_indices_tensor] = torch.tensor(1.0, device=device, dtype=perturbed_features.dtype)
        elif scenario_name == "member_over_consulting":
            new_vals = means - c * stds
            new_vals = torch.clamp(new_vals, min=0.0)
            perturbed_features[feature_indices_tensor] = new_vals
        else:
            # Create noise on device
            direction = torch.sign(torch.randn_like(means, device=device))
            noise_magnitude = c * stds
            new_vals = means + direction * noise_magnitude
            perturbed_features[feature_indices_tensor] = new_vals

        # Apply perturbation (use integer index for direct assignment)
        modified_data[target_node_type].x[target_node_idx] = perturbed_features.to(original_features.dtype)

        # Mark target and track
        gt_node_labels[target_node_type][target_node_idx] = 1
        _update_tracking(anomaly_tracking, 'node', target_node_type, target_node_idx, tag_base)
        modified = True

    return modified


def _inject_edge_attribute_anomalies(
    modified_data: HeteroData,
    gt_edge_labels_dict: Dict[Tuple, torch.Tensor],
    anomaly_tracking: Dict,
    target_edge_type: Tuple,
    num_to_inject: int,
    stats: Dict,
    params: Dict, # Contains edge_attr_c_range, edge_attr_direction
    device: torch.device # Explicitly require device
    ):
    """Injects attribute anomalies on existing edges."""
    if num_to_inject <= 0: return
    if target_edge_type not in modified_data.edge_types or not hasattr(modified_data[target_edge_type], 'edge_attr') or modified_data[target_edge_type].edge_attr is None:
        print(f"  Skipping edge attribute injection: No edge attributes found for {target_edge_type}.")
        return

    edge_stat = stats['edge'].get(target_edge_type, {})
    if not edge_stat:
         print(f"  Skipping edge attribute injection: No stats found for {target_edge_type}.")
         return

    # Ensure attributes and labels are on the correct device
    current_edge_attr = modified_data[target_edge_type].edge_attr.to(device)
    current_edge_labels = gt_edge_labels_dict[target_edge_type].to(device)
    num_existing_edges = current_edge_attr.shape[0]

    # Find edges on the correct device
    available_edge_indices_tensor = torch.where(current_edge_labels == 0)[0]
    available_edge_indices = available_edge_indices_tensor.tolist() # Convert to list for sampling

    if not available_edge_indices:
        print("  No non-anomalous edges available for attribute injection.")
        return

    num_can_inject = min(num_to_inject, len(available_edge_indices))
    selected_edge_indices = random.sample(available_edge_indices, num_can_inject)
    print(f"  Injecting {num_can_inject} edge attribute anomalies on {target_edge_type}...")

    c_low, c_high = params.get('edge_attr_c_range', (2.0, 4.0))
    direction = params.get('edge_attr_direction', 'high')
    mean_ea = edge_stat['mean']
    std_ea = edge_stat['std']
    p99 = edge_stat['p99']
    p01 = edge_stat['p01']
    dtype = edge_stat['dtype']
    shape = edge_stat['shape']

    modified_count = 0
    for edge_idx in selected_edge_indices: # Use integer index for assignment
        c = random.uniform(c_low, c_high)

        # Calculate new value (scalar calculation is fine)
        if direction == 'high':
            new_val = max(p99, mean_ea + c * std_ea)
        elif direction == 'low':
            new_val = min(p01, mean_ea - c * std_ea)
            new_val = max(0.0, new_val)
        else:
             if random.random() < 0.5:
                 new_val = max(p99, mean_ea + c * std_ea)
             else:
                 new_val = min(p01, mean_ea - c * std_ea)
                 new_val = max(0.0, new_val)

        # Create tensor for assignment on the correct device
        new_val_tensor = torch.tensor([new_val], device=device).reshape(shape).to(dtype)
        # Assign using integer index
        modified_data[target_edge_type].edge_attr[edge_idx] = new_val_tensor

        # Mark edge as anomalous (use integer index)
        gt_edge_labels_dict[target_edge_type][edge_idx] = 1
        tag = "Attribute/AbnormalClaimVolume"
        _update_tracking(anomaly_tracking, 'edge', target_edge_type, edge_idx, tag)
        modified_count += 1

    print(f"    Modified attributes of {modified_count} existing edges.")


# === Main Injection Function ===

def inject_scenario_anomalies(
    data: HeteroData,
    # --- Injection Budgets ---
    p_node_anomalies: float = 0.05, # Percentage of nodes per type to select
    p_edge_anomalies: float = 0.02, # Target percentage of *total* edges to be anomalous
    lambda_structural: float = 0.5, # Proportion of selected nodes for structural scenarios
    # --- Scenario Parameters (can override defaults) ---
    scenario_params: Optional[Dict[str, Any]] = None,
    # --- Features & Types ---
    provider_node_type: str = 'provider',
    member_node_type: str = 'member',
    target_edge_type: Tuple[str, str, str] = ('provider', 'to', 'member'),
    # --- General ---
    seed: Optional[int] = None
    ) -> Tuple[HeteroData, Dict[str, torch.Tensor], Dict[Tuple, torch.Tensor], Dict[str, Dict]]:
    """
    Injects synthetic anomalies based on predefined scenarios.

    Args:
        data (HeteroData): Original graph data. Assumed to be on CPU initially.
        p_node_anomalies (float): Percentage of nodes of *each type* to target for anomalies.
        p_edge_anomalies (float): Target percentage of *total* final edges to be anomalous.
        lambda_structural (float): Fraction of targeted nodes to receive structural anomalies (rest get attribute).
        scenario_params (Optional[Dict]): Dictionary to override default scenario parameters.
        provider_node_type (str): Name of the provider node type.
        member_node_type (str): Name of the member node type.
        target_edge_type (Tuple): The primary edge type (provider, to, member).
        seed (Optional[int]): Random seed.

    Returns:
        modified_data (HeteroData): Graph with injected anomalies (on the determined device).
        gt_node_labels (Dict[str, Tensor]): Binary labels for anomalous nodes (on the determined device).
        gt_edge_labels (Dict[Tuple, Tensor]): Binary labels for anomalous edges (on the determined device).
        anomaly_tracking (Dict[str, Dict]): Detailed anomaly tags per node/edge index.
    """
    # === Seeding ===
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        # Seed MPS if available and using torch >= 1.12 (approx)
        if hasattr(torch, 'mps') and torch.backends.mps.is_available():
             try:
                 torch.mps.manual_seed(seed)
             except AttributeError: # Older torch versions might not have torch.mps.manual_seed
                 print("Warning: torch.mps.manual_seed not found. MPS seeding skipped.")
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # === Device Setup ===
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print(f"--- Injecting Scenario-Based Anomalies on Device: {device} ---")
    print(f"Node Anomaly Budget: {p_node_anomalies*100:.1f}% per type")
    print(f"Edge Anomaly Target: {p_edge_anomalies*100:.1f}% of total edges")
    print(f"Structural/Attribute Split (Lambda): {lambda_structural*100:.1f}% / {(1-lambda_structural)*100:.1f}%")

    modified_data = data.clone()
    # Move the entire data object structure to the target device
    modified_data.to(device)

    reverse_edge_type = (target_edge_type[2], target_edge_type[1], target_edge_type[0])
    if reverse_edge_type not in modified_data.edge_types:
        reverse_edge_type = None
        print("  No reverse edge type found.")

    # === Parameter Setup ===
    params = DEFAULT_PARAMS.copy()
    if scenario_params:
        params.update(scenario_params)

    # === Feature Identification ===
    node_features_needed = defaultdict(list)
    # Logic for identifying features remains the same...
    for sc_name, sc_config in SCENARIOS.items():
        if sc_config['type'] == 'attribute' and sc_config.get('requires_features', False):
             features = sc_config.get('features')
             sc_node_type = sc_config['node_type']
             node_types_to_check = [provider_node_type, member_node_type] if sc_node_type == 'both' else [sc_node_type]
             for ntype in node_types_to_check:
                 if features: node_features_needed[ntype].extend(features)
                 elif not node_features_needed[ntype]: node_features_needed[ntype].append("__ANY__")
    # Logic for handling unique features remains the same...
    for ntype in node_features_needed:
        unique_features = sorted(list(set(node_features_needed[ntype])))
        if len(unique_features) > 1 and "__ANY__" in unique_features: unique_features.remove("__ANY__")
        if unique_features == ["__ANY__"]:
             print(f"  Warning: Attribute scenarios need features for {ntype}, but no specific feature names provided.")
             node_features_needed[ntype] = []
        else: node_features_needed[ntype] = unique_features

    # === Compute Stats (Pass Device) ===
    stats = _compute_graph_stats(modified_data, node_features_needed, device)

    # === Initialize Ground Truth & Tracking (Create on Device) ===
    gt_node_labels = {nt: torch.zeros(modified_data[nt].num_nodes, dtype=torch.long, device=device)
                      for nt in modified_data.node_types}
    gt_edge_labels_dict = {et: torch.zeros(modified_data[et].edge_index.shape[1], dtype=torch.long, device=device)
                           for et in modified_data.edge_types}
    anomaly_tracking = {'node': defaultdict(lambda: defaultdict(list)), 'edge': defaultdict(list)}
    edges_added_count = 0

    # === Node Anomaly Injection Loop ===
    print("\n--- Phase 1: Node Anomaly Injection ---")
    nodes_marked_primary = {nt: set() for nt in modified_data.node_types}

    # --- Refined Planning Logic for Cross-Type Scenarios ---
    # Plan structural anomalies first, especially cross-type ones
    structural_plan = []
    attribute_plan = [] # Plan attribute later

    # Select all nodes first
    all_selected_nodes = {}
    structural_pools = {}
    attribute_pools = {}
    for node_type in [provider_node_type, member_node_type]:
         num_nodes = modified_data[node_type].num_nodes
         if num_nodes == 0: continue
         selected_indices = _select_nodes_for_anomaly(num_nodes, p_node_anomalies)
         all_selected_nodes[node_type] = selected_indices
         s_nodes, a_nodes = _partition_nodes(selected_indices, lambda_structural)
         structural_pools[node_type] = set(s_nodes)
         attribute_pools[node_type] = set(a_nodes)
         print(f"  {node_type}: Selected {len(selected_indices)}, Struct Pool {len(s_nodes)}, Attr Pool {len(a_nodes)}")


    # Plan Collusion Rings first if requested
    collusion_sc_name = "collusion_ring"
    if collusion_sc_name in SCENARIOS:
        sc_config = SCENARIOS[collusion_sc_name]
        req_p = params['k_p_collusion']
        req_m = params['k_m_collusion']
        pool_p = structural_pools.get(provider_node_type, set())
        pool_m = structural_pools.get(member_node_type, set())

        # Try to plan instances as long as nodes are available
        while len(pool_p) >= req_p and len(pool_m) >= req_m:
            chosen_p = random.sample(list(pool_p), req_p)
            chosen_m = random.sample(list(pool_m), req_m)
            instance_nodes = {provider_node_type: chosen_p, member_node_type: chosen_m}
            print(f"    Planning instance: {collusion_sc_name} involving {instance_nodes}")
            structural_plan.append({'name': collusion_sc_name, 'nodes': instance_nodes})
            pool_p.difference_update(chosen_p)
            pool_m.difference_update(chosen_m)


    # Plan other structural scenarios per node type
    for node_type in [provider_node_type, member_node_type]:
        pool = structural_pools.get(node_type, set())
        if not pool: continue

        # Iterate through applicable structural scenarios (excluding collusion)
        applicable_struct_scenarios = [
            s_name for s_name, s_config in SCENARIOS.items()
            if s_config['type'] == 'structural' and s_name != "collusion_ring" and
               (s_config['node_type'] == node_type) # Only single-type or both handled earlier
        ]
        random.shuffle(applicable_struct_scenarios) # Randomize order

        for sc_name in applicable_struct_scenarios:
            sc_config = SCENARIOS[sc_name]
            required_count = 0
            if sc_name == "over_referral_clique": required_count = params['k_p_clique']
            elif sc_name == "member_clique": required_count = params['k_m_clique']
            elif sc_name == "identity_theft_doctor_touring": required_count = 1
            else: continue

            # Try to plan instances
            while len(pool) >= required_count:
                 candidates = list(pool)
                 # Apply degree constraints if needed
                 if sc_name == "over_referral_clique" and stats['node'][node_type]['degree']:
                     p90_degree = stats['node'][node_type]['degree']['p90']
                     degrees = stats['node'][node_type]['degree']['values']
                     candidates = [n for n in candidates if degrees[n] <= p90_degree]

                 if len(candidates) >= required_count:
                     chosen = random.sample(candidates, required_count)
                     instance_nodes = {node_type: chosen}
                     print(f"    Planning instance: {sc_name} involving {instance_nodes}")
                     structural_plan.append({'name': sc_name, 'nodes': instance_nodes})
                     pool.difference_update(chosen)
                 else:
                     break # Not enough valid candidates left in the pool


    # Plan Attribute Scenarios (simpler - assign one per node in pool)
    for node_type in [provider_node_type, member_node_type]:
        pool = attribute_pools.get(node_type, set())
        if not pool: continue

        applicable_attr_scenarios = [
            s_name for s_name, s_config in SCENARIOS.items()
            if s_config['type'] == 'attribute' and (s_config['node_type'] == node_type or s_config['node_type'] == 'both')
        ]
        if not applicable_attr_scenarios: continue

        for node_idx in pool:
            chosen_sc_name = random.choice(applicable_attr_scenarios)
            instance_nodes = {node_type: [node_idx]}
            attribute_plan.append({'name': chosen_sc_name, 'nodes': instance_nodes})
            # Don't remove from pool here, just plan one per node


    # --- Execute Planned Scenarios ---
    print(f"  Executing {len(structural_plan)} structural and {len(attribute_plan)} attribute planned instances...")
    processed_nodes_struct = defaultdict(set)
    processed_nodes_attr = defaultdict(set)
    new_edges_struct_all = []
    new_attrs_struct_all = []

    # 1. Execute Structural Scenarios
    for instance in structural_plan:
        sc_name = instance['name']
        sc_nodes = instance['nodes'] # Dict: {ntype: [indices]}
        sc_config = SCENARIOS[sc_name]

        # Inject scenario (pass device)
        new_edges_fwd, new_attrs_fwd = _inject_structural_scenario(
            sc_name, sc_config, modified_data, gt_node_labels, anomaly_tracking,
            sc_nodes, stats, params, device, target_edge_type, reverse_edge_type
        )
        new_edges_struct_all.extend(new_edges_fwd)
        new_attrs_struct_all.extend(new_attrs_fwd)

        # Mark nodes as processed for this type
        for ntype, indices in sc_nodes.items():
            processed_nodes_struct[ntype].update(indices)
            nodes_marked_primary[ntype].update(indices)

    # 2. Execute Attribute Scenarios
    for instance in attribute_plan:
         sc_name = instance['name']
         sc_nodes = instance['nodes']
         sc_config = SCENARIOS[sc_name]
         # Should only be one node_type and one index here based on planning
         node_type = list(sc_nodes.keys())[0]
         node_idx = sc_nodes[node_type][0]

         # Inject scenario (pass device)
         modified = _inject_attribute_scenario(
             sc_name, sc_config, modified_data, gt_node_labels, anomaly_tracking,
             node_type, node_idx, stats, params, device
         )
         if modified:
             processed_nodes_attr[node_type].add(node_idx)
             nodes_marked_primary[node_type].add(node_idx)


    # 3. Apply Combined Anomalies
    print(f"  Checking for Combined Anomalies...")
    combined_count = 0
    nodes_already_combined = defaultdict(set)

    # Nodes that got structural -> 50% chance add attribute
    for node_type, indices in processed_nodes_struct.items():
        applicable_attr_scenarios = [
            s_name for s_name, s_config in SCENARIOS.items()
            if s_config['type'] == 'attribute' and (s_config['node_type'] == node_type or s_config['node_type'] == 'both')
            and s_config.get('requires_features', False)
        ]
        if not applicable_attr_scenarios: continue

        for node_idx in indices:
            if node_idx not in nodes_already_combined[node_type] and random.random() < 0.5:
                 chosen_attr_sc_name = random.choice(applicable_attr_scenarios)
                 chosen_attr_sc_config = SCENARIOS[chosen_attr_sc_name]
                 print(f"      Adding combined attribute '{chosen_attr_sc_name}' to struct {node_type} {node_idx}")
                 modified = _inject_attribute_scenario(
                     chosen_attr_sc_name, chosen_attr_sc_config, modified_data, gt_node_labels,
                     anomaly_tracking, node_type, node_idx, stats, params, device
                 )
                 if modified:
                     _update_tracking(anomaly_tracking, 'node', node_type, node_idx, "Combined")
                     nodes_already_combined[node_type].add(node_idx)
                     combined_count += 1

    # Nodes that got attribute -> 50% chance add structural (if applicable - harder)
    # --- TODO: Implement logic for Attr -> Struct combined if desired ---
    # This would involve picking an applicable structural scenario and potentially
    # finding *other* nodes to form the structure with. Complex. Skipping for now.


    # --- Add Structurally Added Edges (Pass Device) ---
    num_added = _add_edges_and_update_labels(
    modified_data, gt_edge_labels_dict, target_edge_type,
    new_edges_struct_all, device, new_attrs_struct_all 
    )
    edges_added_count += num_added
    # Add reverse edges if applicable
    if reverse_edge_type and num_added > 0:
        new_rev_edges = [(v, u) for u, v in new_edges_struct_all]
        _add_edges_and_update_labels(
        modified_data, gt_edge_labels_dict, reverse_edge_type,
        new_rev_edges, device, new_attrs_struct_all # <--- Correct order
    )


    # === Edge Anomaly Injection (Attribute Tweaking - Pass Device) ===
    print("\n--- Phase 2: Edge Attribute Anomaly Injection ---")
    # Ensure edge index is on device for count
    total_current_edges = modified_data[target_edge_type].edge_index.shape[1] if target_edge_type in modified_data.edge_types else 0
    target_total_anom_edges = int(total_current_edges * p_edge_anomalies)
    # Ensure labels are on device for sum
    current_anom_edges = gt_edge_labels_dict[target_edge_type].sum().item() if target_edge_type in gt_edge_labels_dict else 0
    remaining_edge_budget = max(0, target_total_anom_edges - current_anom_edges)

    print(f"  Target anomalous edges: {target_total_anom_edges} ({p_edge_anomalies*100:.1f}% of {total_current_edges})")
    print(f"  Edges already anomalous (structurally added approx): {current_anom_edges}")
    print(f"  Remaining budget for edge attribute tweaking: {remaining_edge_budget}")

    if target_edge_type in modified_data.edge_types: # Check type exists before injecting
        _inject_edge_attribute_anomalies(
            modified_data, gt_edge_labels_dict, anomaly_tracking,
            target_edge_type, remaining_edge_budget, stats, params, device
            )
    # Add reverse edge logic if needed, passing device


    # === Final Graph Update & Cleanup ===
    print("\nConsolidating graph modifications...")
    final_anomaly_tracking = {'node': {}, 'edge': {}}

    # Finalize edge labels and assign to '.y' (ensure tensors are on device)
    for edge_type, final_labels in gt_edge_labels_dict.items():
        if edge_type in modified_data.edge_types:
            current_num_edges = modified_data[edge_type].edge_index.shape[1]
            labels_on_device = final_labels.to(device) # Ensure labels are on device
            if len(labels_on_device) != current_num_edges:
                 print(f"ERROR FINAL: Mismatch gt_edge_labels ({len(labels_on_device)}) vs final edge_index ({current_num_edges}) for {edge_type}.")
                 if len(labels_on_device) < current_num_edges:
                      # Create padding on device
                      padding = torch.zeros(current_num_edges - len(labels_on_device), dtype=torch.long, device=device)
                      gt_edge_labels_dict[edge_type] = torch.cat([labels_on_device, padding])
                      print(f"      Resized labels for {edge_type} with padding.")
                 else:
                      gt_edge_labels_dict[edge_type] = labels_on_device[:current_num_edges]
                      print(f"      Truncated labels for {edge_type}.")
            else:
                 gt_edge_labels_dict[edge_type] = labels_on_device # Assign device tensor back

            modified_data[edge_type].y = gt_edge_labels_dict[edge_type] # Assign final device tensor
        else:
            print(f"Warning: Edge type {edge_type} not found in final modified_data keys.")

    # Format node tracking (label check uses device tensor)
    for ntype, nodes in anomaly_tracking['node'].items():
        final_anomaly_tracking['node'][ntype] = {
            idx: sorted(list(set(tags))) for idx, tags in nodes.items() if gt_node_labels[ntype][idx] == 1
        }
    # Format edge tracking (label check uses device tensor)
    for (etype, idx), tags in anomaly_tracking['edge'].items():
         if etype in modified_data.edge_types and idx < modified_data[etype].edge_index.shape[1]:
             if gt_edge_labels_dict[etype][idx] == 1:
                 final_anomaly_tracking['edge'][(etype, idx)] = sorted(list(set(tags)))

    # Final summary
    final_anom_nodes = sum(gt.sum().item() for gt in gt_node_labels.values())
    final_anom_edges = sum(gt.sum().item() for gt in gt_edge_labels_dict.values() if gt is not None)
    total_nodes_sum = sum(data[nt].num_nodes for nt in data.node_types) # Use original data for total count
    print(f"\nAnomaly injection finished.")
    print(f"  Total nodes: {total_nodes_sum}")
    if total_nodes_sum > 0:
        print(f"  Total anomalous nodes: {final_anom_nodes} ({final_anom_nodes / total_nodes_sum * 100:.2f}%)")
    else:
        print(f"  Total anomalous nodes: 0")
    if target_edge_type in modified_data.edge_types:
        final_edge_count = modified_data[target_edge_type].edge_index.shape[1]
        final_anom_edge_count = gt_edge_labels_dict[target_edge_type].sum().item()
        print(f"  Total final edges ({target_edge_type}): {final_edge_count}")
        if final_edge_count > 0:
            print(f"  Total anomalous edges ({target_edge_type}): {final_anom_edge_count} ({final_anom_edge_count / final_edge_count * 100:.2f}%)")
        else:
            print(f"  Total anomalous edges ({target_edge_type}): 0")


    # Return data/labels (they are already on the target device)
    return modified_data, gt_node_labels, gt_edge_labels_dict, final_anomaly_tracking


# === Example Usage (Placeholder) ===
if __name__ == '__main__':

    # --- Device Setup for Example ---
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Example Usage using device: {device}")

    # --- Create Dummy Data (Create directly on target device) ---
    data = HeteroData()
    data['provider'].x = torch.randn(50, 16, device=device)
    data['member'].x = torch.randn(500, 23, device=device)
    member_feature_names = [
       'mean_claim_period', 'median_claim_period', 'min_claim_period',
       'max_claim_period', 'std_claim_period', 'mean_claim_amount',
       'median_claim_amount', 'min_claim_amount', 'max_claim_amount',
       'std_claim_amount', 'avg_provider_claim_period',
       'std_provider_claim_period', 'max_provider_claim_amount',
       'avg_provider_claim_amount', 'unique_providers', 'num_claims',
       'prop_claimtype_not_op', 'prop_claimtype_out-patient',
       'single_interaction_ratio', 'gender_f', 'gender_m',
       'principalcode_dependant', 'principalcode_principal'
    ]
    data['member'].feature_names = member_feature_names
    provider_feature_names = [f'prov_feat_{i}' for i in range(16)]
    reimbursement_idx = 5
    data['provider'].x[:, reimbursement_idx] = torch.randint(0, 2, (50,), device=device).float()
    provider_feature_names[reimbursement_idx] = 'one_hot_claim__casetype_reimbursement'
    data['provider'].feature_names = provider_feature_names

    edge_index_pm = torch.randint(0, 50, (2, 1000), device=device)
    edge_index_pm[1] = torch.randint(0, 500, (1000,), device=device)
    data['provider', 'to', 'member'].edge_index = edge_index_pm
    data['provider', 'to', 'member'].edge_attr = torch.randint(1, 15, (1000, 1), device=device).float()

    # data is now created on the target device

    print("Original Data:\n", data)
    print(f"Original data is on device: {data['provider'].x.device}") # Check one tensor

    # --- Inject Anomalies ---
    custom_params = {
        "k_m_clique": 20,
        "attr_perturb_std_factor_range": (4.0, 7.0)
    }

    # The function now handles device internally and returns results on that device
    modified_graph, node_labels, edge_labels, tracking = inject_scenario_anomalies(
        data, # Pass data (already on device or will be moved inside)
        p_node_anomalies=0.1,
        p_edge_anomalies=0.05,
        lambda_structural=0.6, # Adjusted lambda to 0.6 for example
        scenario_params=custom_params,
        seed=42
    )

    print("\n--- Injection Results ---")
    print("Modified Graph Structure:\n", modified_graph)
    print(f"Modified data is on device: {modified_graph['provider'].x.device}") # Check one tensor
    print("\nAnomalous Node Counts:")
    for ntype, labels in node_labels.items():
        print(f"  {ntype}: {labels.sum().item()} / {labels.numel()} (Device: {labels.device})")
    print("\nAnomalous Edge Counts:")
    for etype, labels in edge_labels.items():
        if etype in modified_graph.edge_types:
             print(f"  {etype}: {labels.sum().item()} / {labels.numel()} (Device: {labels.device})")


# === List of Points Still Requiring User Input/Refinement ===
# (List remains the same as previous version)
print("\n\n===== TODO / Refinements Needed =====")
print("1.  **Node Feature Name Mapping:** The `_compute_graph_stats` function assumes a sequential mapping or relies on `data[ntype].feature_names`. Provide the exact mapping between the feature names used in `SCENARIOS` (e.g., 'mean_claim_period') and their column index in `data[node_type].x`. Storing names in `data[ntype].feature_names` is the best practice.")
print("2.  **Complete Attribute Scenarios:** Fill in the `SCENARIOS` dictionary with all desired attribute scenarios, including their specific `features` list.")
print("3.  **Minimum Claim Period:** Define the realistic minimum value for 'mean_claim_period'/'median_claim_period' in `_inject_attribute_scenario` (used for clamping in 'member_over_consulting'). Is it 0, 1, or something else?")
print("4.  **Structural Scenario Targets:** Clarify how target nodes are selected in structural scenarios:")
print("    - `over_referral_clique`: How are the `k_m_target` members chosen (randomly, based on overlap, etc.)?")
print("    - `member_clique`: How are the `k_p_target` providers chosen?")
print("5.  **Collusion Ring Planning Refinement:** The improved planning logic handles collusion rings better, but review if the 'while loop' approach to plan multiple instances is desired or if a fixed number based on budget is better.")
print("6.  **Structural Edge Labelling:** Review the edge labelling logic for structural scenarios (e.g., 'over_referral_clique', 'member_clique'). Is labelling *only* the newly added edges sufficient? (Current code labels only *new* edges).")
print("7.  **Combined Anomaly (Attr -> Struct):** The current implementation only adds an attribute anomaly on top of a structural one. Define if/how the reverse should work.")
print("8.  **Reverse Edge Handling:** Decide how anomalies should affect reverse edges (`member` -> `provider`). Are attributes shared? Should structural anomalies be mirrored? Should attribute tweaks be applied independently?")
print("9.  **Edge Attribute Perturbation Detail:** The final edge attribute tweak pushes values beyond P99/P01 or uses `mean +/- c*std`. Refine this if more specific distributions (e.g., Poisson for counts) or perturbation methods are desired.")
print("10. **Degree Constraint Usage:** Currently, only 'over_referral_clique' uses degree constraints (avoiding high-degree). Review if other scenarios need similar constraints during node selection.")
print("11. **Identity Swap Partner Tracking:** The 'Combined' logic needs a way to know which node was the partner in an 'identity_swap' if the partner also needs the 'Combined' tag (requires modification to `_inject_attribute_scenario` to return partner info).")
print("12. **Placeholder Features ('__ANY__'):** Decide which features should be used for stats/distance if a scenario requires features generically (like swap) but doesn't list specific ones.")
print("======================================")

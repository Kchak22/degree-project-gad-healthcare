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
    "max_degree_percentile_clique_prov": 0.9, # Max degree for providers in clique
    "max_degree_percentile_clique_member_target": 0.9, # Max degree for members targeted by prov clique
    # Structural - Members
    "k_m_clique": 15,
    "p_link_member_clique": 0.95,
    "max_degree_percentile_member_clique_prov_target": 0.9, # Max degree for providers targeted by member clique
    "k_providers_theft_touring": 10,
    "mode_theft_touring": 'predefined', # or 'degree_based'
    "max_degree_percentile_theft_touring_prov_target": 0.75, # Prefer lower-degree providers for touring
    # Structural - Both
    "k_p_collusion": 3,
    "k_m_collusion": 10,
    "collusion_partial_block_frac_range": (0.7, 1.0),
    "max_degree_percentile_collusion": 0.9, # Max degree for nodes selected for collusion ring
    # Attribute - General
    "swap_pool_size": 50,
    "swap_metric": 'euclidean',
    # Attribute Tweaking
    "attr_perturb_std_factor_range": (3.0, 6.0), # Used if percentile stats unavailable
    "attr_perturb_percentiles": (0.01, 0.99), # Target low/high percentiles for perturbation
    "attr_perturb_frac": 0.3,
    # Edge Attribute Tweaking
    "edge_attr_method": "poisson", # 'poisson' or 'extreme_quantile'
    "edge_attr_poisson_scale_range": (3.0, 6.0), # Multiplier for mean in poisson lambda
    "edge_attr_direction": "high", # For non-poisson methods
    "edge_attr_c_range": (3.0, 5.0) # For non-poisson methods (std factor)
}

# --- Scenario Definitions ---
# Feature names MUST match the keys used in the stats dictionary (after potential renaming)
SCENARIOS = {
    # --- Structural Scenarios ---
    "over_referral_clique": {
        "type": "structural",
        "node_type": "provider",
        "params": ["k_p_clique", "p_link_clique", "max_degree_percentile_clique_prov", "max_degree_percentile_clique_member_target"],
        "description": "Group of providers referring patients primarily amongst themselves.",
        "requires_degree": True
    },
    "member_clique": {
        "type": "structural",
        "node_type": "member",
        "params": ["k_m_clique", "p_link_member_clique", "max_degree_percentile_member_clique_prov_target"],
        "description": "Group of members seeing the exact same provider network.",
         "requires_degree": True # For selecting target providers
    },
    "identity_theft_doctor_touring": {
        "type": "structural",
        "node_type": "member",
        "params": ["k_providers_theft_touring", "mode_theft_touring", "max_degree_percentile_theft_touring_prov_target"],
        "description": "Single member accessing an unusually high number of distinct providers.",
        "requires_degree": True # For selecting target providers
    },
    "collusion_ring": {
        "type": "structural",
        "node_type": "both",
        "params": ["k_p_collusion", "k_m_collusion", "collusion_partial_block_frac_range", "max_degree_percentile_collusion"],
        "description": "Dense interaction block between specific providers and members.",
         "requires_degree": True
    },
    # --- Attribute Scenarios ---
    "identity_swap": {
        "type": "attribute",
        "node_type": "both",
        "params": ["swap_pool_size", "swap_metric"],
        "description": "Swap feature vector with a dissimilar node.",
        "requires_features": True # Needs generic features 'x'
    },
    # --- Provider Attribute Scenarios ---
    "provider_overbilling": {
        "type": "attribute",
        "node_type": "provider",
        "params": ["attr_perturb_percentiles", "attr_perturb_std_factor_range"],
        "features": [ # Features to increase significantly
            'provider_claim_amount_mean', 'provider_claim_amount_median', 'provider_claim_amount_max', 'provider_claim_amount_std',
            'member_claim_amount_mean', 'member_claim_amount_median', 'member_claim_amount_max', 'member_claim_amount_std'
            ],
        "perturb_direction": "high", # Push values higher
        "description": "Provider submitting or associated with unusually high claim amounts.",
        "requires_features": True
    },
     "provider_nonaccredited_highfreq": {
        "type": "attribute",
        "node_type": "provider",
        "params": ["attr_perturb_percentiles", "attr_perturb_std_factor_range"],
        "features": { # Feature-specific modifications
             "accreditation_is_infreq": 0.0, # Set to 0 (meaning NOT infrequent -> accredited? CHECK LOGIC) or flip existing? Assuming 0 = accredited
             "member_claim_period_mean": "low", # Decrease period -> higher freq
             "member_claim_period_median": "low",
             "provider_claim_period_mean": "low",
             "provider_claim_period_median": "low",
             "interaction_ratio_single": "high" # Increase single interactions
         },
        "description": "Provider possibly non-accredited with unusually high claim frequency.",
        "requires_features": True
    },
    "provider_over_reimbursement": { # Keep original one too
        "type": "attribute",
        "node_type": "provider",
        "params": ["attr_perturb_percentiles"],
        "features": {"claim_is_reimbursement": 1.0}, # Target specific value
        "description": "Provider submitting unusually high proportion of reimbursement claims.",
        "requires_features": True
    },
    # --- Member Attribute Scenarios ---
    "member_over_consulting": {
        "type": "attribute",
        "node_type": "member",
        "params": ["attr_perturb_percentiles", "attr_perturb_std_factor_range"],
        "features": [ # Features to decrease significantly
            'claim_period_mean', 'claim_period_median', 'claim_period_min',
            'prov_claim_period_avg', 'prov_claim_period_std' # Assuming lower std means more regular frequent visits
            ],
        "perturb_direction": "low", # Push values lower
        "description": "Member seeking consultations/services far more frequently than normal.",
        "requires_features": True
    },
    "member_high_reimbursements": {
        "type": "attribute",
        "node_type": "member",
        "params": ["attr_perturb_percentiles", "attr_perturb_std_factor_range"],
         "features": [ # Features to increase significantly
            'claim_amount_mean', 'claim_amount_median', 'claim_amount_max', 'claim_amount_std',
            'interaction_ratio_single', 'providers_unique_count'
            ],
        "perturb_direction": "high", # Push values higher
        "description": "Member associated with high claim amounts, potentially seeing many providers.",
        "requires_features": True
    },
}


# === Helper Functions ===

def _get_percentiles(tensor: torch.Tensor, percentiles: Tuple[float, float] = (0.01, 0.99)) -> Tuple[Optional[float], Optional[float]]:
    """Safely compute low/high percentiles of a tensor."""
    if tensor is None or tensor.numel() == 0:
        return None, None
    try:
        # Ensure float for quantile calculation
        tensor_float = tensor.flatten().float()
        if tensor_float.numel() == 0: # Check after flattening
            return None, None
        p_low = torch.quantile(tensor_float, percentiles[0]).item()
        p_high = torch.quantile(tensor_float, percentiles[1]).item()
        return p_low, p_high
    except Exception as e:
        print(f"Warning: Could not compute percentiles: {e}")
        return None, None

def _compute_graph_stats(data: HeteroData, node_features_to_stat: Dict[str, List[str]], device: torch.device, edge_attr_name: str = 'edge_attr') -> Dict[str, Any]:
    """Computes necessary statistics from the graph."""
    print("Computing graph statistics...")
    stats = {'node': defaultdict(dict), 'edge': defaultdict(dict)}

    # Node Degrees & Percentiles
    for ntype in data.node_types:
        stats['node'][ntype]['degree'] = defaultdict(lambda: None) # Use None default
        total_degree = torch.zeros(data[ntype].num_nodes, dtype=torch.long, device=device)
        # ... (degree calculation logic - Ensure scatter_add_ is used correctly) ...
        for etype in data.edge_types:
            edge_index = data[etype].edge_index.to(device)
            if data[ntype].num_nodes > 0: # Check nodes exist before indexing
                if etype[0] == ntype:
                    if edge_index.shape[1] > 0: # Check edges exist
                        idx, counts = torch.unique(edge_index[0], return_counts=True)
                        idx, counts = idx.to(device), counts.to(device)
                        total_degree.scatter_add_(0, idx, counts)
                if etype[2] == ntype:
                    if edge_index.shape[1] > 0: # Check edges exist
                        idx, counts = torch.unique(edge_index[1], return_counts=True)
                        idx, counts = idx.to(device), counts.to(device)
                        total_degree.scatter_add_(0, idx, counts)


        stats['node'][ntype]['degree']['values'] = total_degree
        if total_degree.numel() > 0:
            total_degree_float = total_degree.float()
            stats['node'][ntype]['degree']['mean'] = total_degree_float.mean().item()
            stats['node'][ntype]['degree']['std'] = total_degree_float.std().item()
            # Compute various percentiles
            p_low, p_high = _get_percentiles(total_degree_float, (0.1, 0.9)) # e.g., P10, P90
            stats['node'][ntype]['degree']['p10'] = p_low
            stats['node'][ntype]['degree']['p90'] = p_high
            stats['node'][ntype]['degree']['max'] = total_degree.max().item() if total_degree.numel() > 0 else 0
            print(f"  Node Type '{ntype}': {data[ntype].num_nodes} nodes. Degree (mean={stats['node'][ntype]['degree']['mean']:.2f}, std={stats['node'][ntype]['degree']['std']:.2f}, p10={p_low:.2f}, p90={p_high:.2f})")
        else:
             stats['node'][ntype]['degree']['mean'] = 0.0
             stats['node'][ntype]['degree']['std'] = 1.0
             print(f"  Node Type '{ntype}': {data[ntype].num_nodes} nodes. No degrees calculated.")


    # Node Features & Percentiles
    for ntype, features in node_features_to_stat.items():
        if ntype in data.node_types and hasattr(data[ntype], 'x') and data[ntype].x is not None:
            x = data[ntype].x.to(device)
            if x.numel() == 0:
                print(f"  Node Type '{ntype}': Features 'x' found but empty. Skipping feature stats.")
                continue

            stats['node'][ntype]['features'] = {}
            stats['node'][ntype]['features']['all_means'] = x.mean(dim=0)
            stats['node'][ntype]['features']['all_stds'] = x.std(dim=0)
            stats['node'][ntype]['features']['all_p01'] = torch.full_like(stats['node'][ntype]['features']['all_means'], float('nan'))
            stats['node'][ntype]['features']['all_p99'] = torch.full_like(stats['node'][ntype]['features']['all_means'], float('nan'))

            # Compute percentiles per feature
            for feat_idx in range(x.shape[1]):
                 p01, p99 = _get_percentiles(x[:, feat_idx], (0.01, 0.99))
                 if p01 is not None: stats['node'][ntype]['features']['all_p01'][feat_idx] = p01
                 if p99 is not None: stats['node'][ntype]['features']['all_p99'][feat_idx] = p99

            stats['node'][ntype]['features']['all_stds'][stats['node'][ntype]['features']['all_stds'] == 0] = 1.0
            stats['node'][ntype]['features']['map'] = {}

            # --- Feature name mapping ---
            feature_map = {}
            if hasattr(data[ntype], 'feature_names'):
                feature_map = {name: i for i, name in enumerate(data[ntype].feature_names)}
                print(f"  Using provided feature names for {ntype}.")
            elif features and features != ["__ANY__"]:
                 # Fallback only if features are explicitly listed for stats
                feature_map = {name: i for i, name in enumerate(features)}
                print(f"  Warning: Assuming sequential feature names for {ntype}: {features}")

            stats['node'][ntype]['features']['name_list'] = list(feature_map.keys()) # Store names used

            print(f"  Computing stats for {ntype} features on device: {x.device}")
            for feat_name, feat_idx in feature_map.items():
                 if feat_idx < x.shape[1]:
                    stats['node'][ntype]['features']['map'][feat_name] = feat_idx
                    mean_val = stats['node'][ntype]['features']['all_means'][feat_idx].item()
                    std_val = stats['node'][ntype]['features']['all_stds'][feat_idx].item()
                    p01_val = stats['node'][ntype]['features']['all_p01'][feat_idx].item()
                    p99_val = stats['node'][ntype]['features']['all_p99'][feat_idx].item()
                    print(f"    Feature '{feat_name}' (Idx {feat_idx}): Mean={mean_val:.2f}, Std={std_val:.2f}, P01={p01_val:.2f}, P99={p99_val:.2f}")
                 else:
                    print(f"    Warning: Feature '{feat_name}' index {feat_idx} out of bounds for {ntype} features (shape {x.shape}). Skipping stats.")

        else:
            print(f"  Node Type '{ntype}': No features 'x' found or requested. Skipping feature stats.")

    # Edge Attributes & Percentiles
    target_edge_type = ('provider', 'to', 'member')
    if target_edge_type in data.edge_types and hasattr(data[target_edge_type], edge_attr_name) and data[target_edge_type][edge_attr_name] is not None:
        ea = data[target_edge_type][edge_attr_name].to(device)
        if ea.numel() > 0:
             ea_flat = ea.flatten().float()
             stats['edge'][target_edge_type] = {} # Initialize dict
             stats['edge'][target_edge_type]['mean'] = ea_flat.mean().item()
             stats['edge'][target_edge_type]['std'] = ea_flat.std().item()
             # Compute percentiles for edge attributes
             p01, p99 = _get_percentiles(ea_flat, (0.01, 0.99))
             stats['edge'][target_edge_type]['p01'] = p01
             stats['edge'][target_edge_type]['p99'] = p99
             stats['edge'][target_edge_type]['dtype'] = ea.dtype
             stats['edge'][target_edge_type]['shape'] = list(ea.shape[1:]) if ea.dim() > 1 else [1]
             if stats['edge'][target_edge_type]['std'] == 0:
                 stats['edge'][target_edge_type]['std'] = 1.0
             print(f"  Edge Type '{target_edge_type}': {ea.shape[0]} edges. Attr '{edge_attr_name}' (mean={stats['edge'][target_edge_type]['mean']:.2f}, std={stats['edge'][target_edge_type]['std']:.2f}, p01={p01:.2f}, p99={p99:.2f})")
        else:
             stats['edge'][target_edge_type] = {'mean': 1.0, 'std': 1.0, 'p01': 0.0, 'p99': 5.0, 'dtype': torch.float, 'shape': [1]}
             print(f"  Edge Type '{target_edge_type}': Attr '{edge_attr_name}' empty. Using defaults.")
    else:
        stats['edge'][target_edge_type] = {'mean': 1.0, 'std': 1.0, 'p01': 0.0, 'p99': 5.0, 'dtype': torch.float, 'shape': [1]}
        print(f"  Edge Type '{target_edge_type}': No attr '{edge_attr_name}'. Using defaults.")

    print("Statistics computation finished.")
    return stats

# ... (keep _select_nodes_for_anomaly, _partition_nodes, _update_tracking) ...
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
        edge_key = (node_type, node_idx) # node_type holds edge_type, node_idx holds edge_idx
        anomaly_tracking['edge'][edge_key].extend(tags)
    else: # Both nodes of an edge
         u_type, v_type = node_type
         u_idx, v_idx = node_idx
         anomaly_tracking['node'][u_type][u_idx].extend(tags)
         anomaly_tracking['node'][v_type][v_idx].extend(tags)


def _add_edges_and_update_labels(
    modified_data: HeteroData,
    gt_edge_labels_dict: Dict[Tuple, torch.Tensor],
    edge_type: Tuple,
    new_edges: List[Tuple[int, int]],
    device: torch.device,
    new_edge_attrs: Optional[List[torch.Tensor]] = None,
    label_existing: bool = False, # New flag for Point 6
    nodes_involved: Optional[List[int]] = None # Nodes whose existing edges should be labeled
    ) -> int:
    """Adds new edges, attributes, and updates edge labels. Returns number added."""
    num_added = 0
    start_index_new_edges = modified_data[edge_type].edge_index.shape[1] # Index where new edges will start

    if new_edges:
        num_added = len(new_edges)
        new_edges_tensor = torch.tensor(new_edges, dtype=torch.long, device=device).t()
        orig_edge_index = modified_data[edge_type].edge_index.to(device)
        modified_data[edge_type].edge_index = torch.cat([orig_edge_index, new_edges_tensor], dim=1)

        # Append Attributes (logic unchanged, device handled inside)
        if new_edge_attrs:
            new_attrs_tensor = torch.stack(new_edge_attrs).to(device)
            if hasattr(modified_data[edge_type], 'edge_attr') and modified_data[edge_type].edge_attr is not None:
                 orig_attrs = modified_data[edge_type].edge_attr.to(device)
                 if orig_attrs.dim() == 1 and new_attrs_tensor.dim() == 2 and new_attrs_tensor.shape[1] == 1:
                     orig_attrs = orig_attrs.unsqueeze(1)
                 elif orig_attrs.dim() == 2 and orig_attrs.shape[1] == 1 and new_attrs_tensor.dim() == 1:
                     new_attrs_tensor = new_attrs_tensor.unsqueeze(1)

                 if orig_attrs.shape[1:] == new_attrs_tensor.shape[1:]:
                     modified_data[edge_type].edge_attr = torch.cat([orig_attrs, new_attrs_tensor], dim=0)
                 else:
                      print(f"Warning: Shape mismatch appending edge attributes for {edge_type}. Original: {orig_attrs.shape}, New: {new_attrs_tensor.shape}. Skipping append.")

            elif num_added > 0:
                 num_orig_edges = modified_data[edge_type].edge_index.shape[1] - num_added
                 if num_orig_edges > 0:
                     print(f"Warning: Initializing edge_attr for {edge_type} before appending. Original edges get 0-value attrs.")
                     placeholder_shape = (num_orig_edges,) + tuple(new_attrs_tensor.shape[1:])
                     placeholder_attrs = torch.zeros(placeholder_shape, dtype=new_attrs_tensor.dtype, device=device)
                     modified_data[edge_type].edge_attr = torch.cat([placeholder_attrs, new_attrs_tensor], dim=0)
                 else:
                     modified_data[edge_type].edge_attr = new_attrs_tensor

        # Append Edge Labels for NEW edges
        orig_labels = gt_edge_labels_dict[edge_type].to(device)
        new_edge_labels = torch.ones(num_added, dtype=torch.long, device=device)
        gt_edge_labels_dict[edge_type] = torch.cat([orig_labels, new_edge_labels])

    # --- Label Existing Edges (Point 6) ---
    if label_existing and nodes_involved:
        print(f"      Labeling existing edges connected to {len(nodes_involved)} nodes for {edge_type}...")
        current_edge_index = modified_data[edge_type].edge_index # Already on device from above/start
        current_labels = gt_edge_labels_dict[edge_type] # Already on device from above/start
        num_edges_total = current_edge_index.shape[1]

        nodes_involved_tensor = torch.tensor(nodes_involved, device=device)

        # Find indices of edges connected to the specified nodes (check both source and dest)
        # Consider only edges *before* the newly added ones
        mask_u = torch.isin(current_edge_index[0,:start_index_new_edges], nodes_involved_tensor)
        mask_v = torch.isin(current_edge_index[1,:start_index_new_edges], nodes_involved_tensor)
        existing_edge_indices_to_label = torch.where(mask_u | mask_v)[0]

        if existing_edge_indices_to_label.numel() > 0:
            print(f"        Found {existing_edge_indices_to_label.numel()} existing edges to label.")
            current_labels[existing_edge_indices_to_label] = 1 # Label them as anomalous
            gt_edge_labels_dict[edge_type] = current_labels # Update the dictionary entry
        else:
            print("        No existing edges found involving these nodes.")

    return num_added


# --- Helper to find neighbors ---
def _find_neighbors(edge_index: torch.Tensor, nodes: List[int], direction: int = 0) -> Dict[int, List[int]]:
    """Finds neighbors for a list of nodes."""
    neighbors = defaultdict(list)
    nodes_tensor = torch.tensor(nodes, device=edge_index.device)
    
    # Find edges where the source/target is one of the nodes
    mask = torch.isin(edge_index[direction], nodes_tensor)
    relevant_edges = edge_index[:, mask]

    # Get neighbors
    source_nodes = relevant_edges[direction].tolist()
    neighbor_nodes = relevant_edges[1 - direction].tolist()

    for node, neighbor in zip(source_nodes, neighbor_nodes):
        neighbors[node].append(neighbor)
        
    return neighbors

def _find_shared_neighbors(edge_index: torch.Tensor, nodes1: List[int], nodes2: List[int]) -> List[int]:
    """Finds neighbors shared between two sets of nodes (e.g., members seen by providers)."""
    # Find neighbors for each node in nodes1
    neighbors1 = _find_neighbors(edge_index, nodes1, direction=0) # providers -> members
    
    # Aggregate all unique neighbors seen by nodes1
    all_neighbors1 = set()
    for node1 in nodes1:
        all_neighbors1.update(neighbors1.get(node1, []))
        
    if not all_neighbors1:
        return []

    # Check which of nodes2 are among the shared neighbors
    shared = [node2 for node2 in nodes2 if node2 in all_neighbors1]
    
    return shared


def _select_target_nodes_with_constraints(
    num_to_select: int,
    candidate_pool: List[int],
    degree_values: torch.Tensor,
    degree_percentile_threshold: Optional[float] = None,
    stats: Dict = {}, # Pass full stats if needed for percentile calculation
    device: torch.device = 'cpu'
) -> List[int]:
    """Selects target nodes randomly from a pool, applying degree constraints."""
    if not candidate_pool:
        return []

    valid_candidates = list(candidate_pool) # Start with all candidates

    # Apply degree constraint
    if degree_percentile_threshold is not None and degree_values is not None:
        if degree_values.numel() > 0:
            try:
                # Ensure calculation on correct device
                threshold_value = torch.quantile(degree_values.float().to(device), degree_percentile_threshold).item()
                # Filter candidates based on degree threshold
                valid_candidates = [n for n in valid_candidates if degree_values[n].item() <= threshold_value]
                print(f"      Applied degree constraint (<= P{int(degree_percentile_threshold*100)} = {threshold_value:.2f}), {len(valid_candidates)} candidates remain.")
            except Exception as e:
                 print(f"      Warning: Could not apply degree constraint: {e}")
        else:
            print("      Warning: Cannot apply degree constraint, degree values tensor is empty.")


    if not valid_candidates:
        print("      Warning: No valid candidates remain after applying constraints.")
        return []

    # Sample from the valid candidates
    num_can_select = min(num_to_select, len(valid_candidates))
    return random.sample(valid_candidates, num_can_select)


def _inject_structural_scenario(
    scenario_name: str,
    scenario_config: Dict,
    modified_data: HeteroData,
    gt_node_labels: Dict[str, torch.Tensor],
    gt_edge_labels_dict: Dict[Tuple, torch.Tensor], # Pass edge labels for update
    anomaly_tracking: Dict,
    nodes_involved: Dict[str, List[int]],
    stats: Dict,
    params: Dict,
    device: torch.device,
    target_edge_type: Tuple = ('provider', 'to', 'member'),
    reverse_edge_type: Optional[Tuple] = ('member', 'to', 'provider')
) -> Tuple[List[Tuple], List[torch.Tensor]]:
    """Injects a structural anomaly, including degree constraints and overlap logic."""
    print(f"    Injecting Structural Scenario: {scenario_name}")
    provider_type, member_type = target_edge_type[0], target_edge_type[2]
    new_edges_fwd = []
    new_edge_attrs_fwd = []
    nodes_labelled = defaultdict(list)
    all_involved_nodes = [] # For labelling existing edges

    # Default Edge Attribute
    edge_stat = stats['edge'].get(target_edge_type, {})
    default_attr_val = torch.tensor([edge_stat.get('mean', 1.0)], device=device, dtype=edge_stat.get('dtype', torch.float))
    default_attr_shape = edge_stat.get('shape', [1])

    # Scenario Logic
    if scenario_name == "over_referral_clique":
        providers = nodes_involved[provider_type]
        all_involved_nodes.extend(providers) # Mark providers involved
        k_p = len(providers)
        p_link = params['p_link_clique']
        k_m_target_overlap = 5 # Example: Target 5 members based on overlap/random
        deg_percentile = params['max_degree_percentile_clique_member_target']

        # --- Point 4: Select target members based on overlap/random ---
        target_members = []
        # Find existing members connected to the clique providers
        edge_index_pm = modified_data[target_edge_type].edge_index.to(device)
        
        provider_neighbors = _find_neighbors(edge_index_pm, providers, direction=0) #direction 0 means provider -> member
        
        shared_member_candidates = set()
        for p in providers:
            shared_member_candidates.update(provider_neighbors.get(p,[]))
        
        shared_members = list(shared_member_candidates)
        print(f"      Found {len(shared_members)} existing members connected to clique providers.")

        # Select from shared members first, applying degree constraint
        if shared_members:
            member_degrees = stats['node'][member_type]['degree']['values']
            selected_shared = _select_target_nodes_with_constraints(
                k_m_target_overlap, shared_members, member_degrees, deg_percentile, stats, device
            )
            target_members.extend(selected_shared)
            print(f"        Selected {len(selected_shared)} members from overlap.")

        # If not enough, select remaining randomly
        num_needed = k_m_target_overlap - len(target_members)
        if num_needed > 0:
            all_member_indices = list(range(modified_data[member_type].num_nodes))
            # Exclude already selected members and potentially the clique members themselves if different type
            candidate_pool = list(set(all_member_indices) - set(target_members))
            member_degrees = stats['node'][member_type]['degree']['values']
            selected_random = _select_target_nodes_with_constraints(
                num_needed, candidate_pool, member_degrees, deg_percentile, stats, device
            )
            target_members.extend(selected_random)
            print(f"        Selected {len(selected_random)} additional members randomly.")

        if not target_members: return [], []
        all_involved_nodes.extend(target_members) # Also mark target members involved

        # --- Create edges ---
        possible_edges = [(p, m) for p in providers for m in target_members]
        edges_to_add_indices = [i for i, _ in enumerate(possible_edges) if random.random() < p_link]
        edges_to_add = [possible_edges[i] for i in edges_to_add_indices]
        current_existing_set = set(tuple(map(int, x)) for x in edge_index_pm.t().tolist())
        new_edges_struct = [edge for edge in edges_to_add if tuple(edge) not in current_existing_set]

        if not new_edges_struct: return [], []
        new_edges_fwd.extend(new_edges_struct)
        attr_val = default_attr_val.clone().detach().reshape(default_attr_shape)
        new_edge_attrs_fwd.extend([attr_val] * len(new_edges_struct))

        # Label nodes and existing edges
        tag = f"Structural/{scenario_name}"
        for p_idx in providers:
            gt_node_labels[provider_type][p_idx] = 1
            _update_tracking(anomaly_tracking, 'node', provider_type, p_idx, tag)
        # Label existing edges involving the involved nodes
        _add_edges_and_update_labels(modified_data, gt_edge_labels_dict, target_edge_type, [], device, label_existing=False, nodes_involved=all_involved_nodes)


    elif scenario_name == "member_clique":
        members = nodes_involved[member_type]
        all_involved_nodes.extend(members)
        k_m = len(members)
        p_link = params['p_link_member_clique']
        k_p_target_overlap = 3 # Example target number of providers
        deg_percentile = params['max_degree_percentile_member_clique_prov_target']

        # --- Point 4: Select target providers based on overlap/random ---
        target_providers = []
        edge_index_pm = modified_data[target_edge_type].edge_index.to(device)
        
        # Find existing providers connected to the clique members
        member_neighbors = _find_neighbors(edge_index_pm, members, direction=1) # direction 1 means member -> provider

        shared_provider_candidates = set()
        for m in members:
            shared_provider_candidates.update(member_neighbors.get(m, []))

        shared_providers = list(shared_provider_candidates)
        print(f"      Found {len(shared_providers)} existing providers connected to clique members.")

        if shared_providers:
             provider_degrees = stats['node'][provider_type]['degree']['values']
             selected_shared = _select_target_nodes_with_constraints(
                 k_p_target_overlap, shared_providers, provider_degrees, deg_percentile, stats, device
             )
             target_providers.extend(selected_shared)
             print(f"        Selected {len(selected_shared)} providers from overlap.")

        num_needed = k_p_target_overlap - len(target_providers)
        if num_needed > 0:
             all_provider_indices = list(range(modified_data[provider_type].num_nodes))
             candidate_pool = list(set(all_provider_indices) - set(target_providers))
             provider_degrees = stats['node'][provider_type]['degree']['values']
             selected_random = _select_target_nodes_with_constraints(
                 num_needed, candidate_pool, provider_degrees, deg_percentile, stats, device
             )
             target_providers.extend(selected_random)
             print(f"        Selected {len(selected_random)} additional providers randomly.")


        if not target_providers: return [], []
        all_involved_nodes.extend(target_providers)

        # --- Create Edges ---
        possible_edges = [(p, m) for p in target_providers for m in members]
        edges_to_add_indices = [i for i, _ in enumerate(possible_edges) if random.random() < p_link]
        edges_to_add = [possible_edges[i] for i in edges_to_add_indices]
        current_existing_set = set(tuple(map(int, x)) for x in edge_index_pm.t().tolist())
        new_edges_struct = [edge for edge in edges_to_add if tuple(edge) not in current_existing_set]

        if not new_edges_struct: return [], []
        new_edges_fwd.extend(new_edges_struct)
        attr_val = torch.tensor([1.0], device=device, dtype=edge_stat.get('dtype', torch.float)).reshape(default_attr_shape)
        new_edge_attrs_fwd.extend([attr_val] * len(new_edges_struct))

        # Label nodes and existing edges
        tag = f"Structural/{scenario_name}"
        for m_idx in members:
             gt_node_labels[member_type][m_idx] = 1
             _update_tracking(anomaly_tracking, 'node', member_type, m_idx, tag)
        _add_edges_and_update_labels(modified_data, gt_edge_labels_dict, target_edge_type, [], device, label_existing=False, nodes_involved=all_involved_nodes)


    elif scenario_name == "identity_theft_doctor_touring":
        member = nodes_involved[member_type][0]
        all_involved_nodes.append(member)
        k_providers = params['k_providers_theft_touring']
        deg_percentile = params['max_degree_percentile_theft_touring_prov_target']

        # Select target providers with degree constraint
        all_provider_indices = list(range(modified_data[provider_type].num_nodes))
        provider_degrees = stats['node'][provider_type]['degree']['values']
        target_providers = _select_target_nodes_with_constraints(
            k_providers, all_provider_indices, provider_degrees, deg_percentile, stats, device
        )

        if not target_providers: return [], []
        all_involved_nodes.extend(target_providers)

        # Create Edges
        edges_to_add = [(p, member) for p in target_providers]
        edge_index_pm = modified_data[target_edge_type].edge_index.to(device)
        current_existing_set = set(tuple(map(int, x)) for x in edge_index_pm.t().tolist())
        new_edges_struct = [edge for edge in edges_to_add if tuple(edge) not in current_existing_set]

        if not new_edges_struct: return [], []
        new_edges_fwd.extend(new_edges_struct)
        mean_claims = edge_stat.get('mean', 1.0)
        low_med_attrs = [torch.tensor([random.uniform(1.0, max(1.1, mean_claims))], device=device, dtype=edge_stat.get('dtype', torch.float)).reshape(default_attr_shape) for _ in new_edges_struct]
        new_edge_attrs_fwd.extend(low_med_attrs)

        # Label node and existing edges
        tag = f"Structural/{scenario_name}"
        gt_node_labels[member_type][member] = 1
        _update_tracking(anomaly_tracking, 'node', member_type, member, tag)
        _add_edges_and_update_labels(modified_data, gt_edge_labels_dict, target_edge_type, [], device, label_existing=False, nodes_involved=all_involved_nodes)


    elif scenario_name == "collusion_ring":
        providers = nodes_involved.get(provider_type, [])
        members = nodes_involved.get(member_type, [])
        if not providers or not members: return [], []
        all_involved_nodes.extend(providers)
        all_involved_nodes.extend(members)

        k_p, k_m = len(providers), len(members)
        frac_range = params['collusion_partial_block_frac_range']
        is_partial = random.random() < 0.5
        possible_edges = [(p, m) for p in providers for m in members]

        if is_partial and len(possible_edges) > 0:
            frac = random.uniform(frac_range[0], frac_range[1])
            edges_to_consider = random.sample(possible_edges, max(1, int(frac * len(possible_edges))))
        else: edges_to_consider = possible_edges

        edge_index_pm = modified_data[target_edge_type].edge_index.to(device)
        current_existing_set = set(tuple(map(int, x)) for x in edge_index_pm.t().tolist())
        new_edges_struct = [edge for edge in edges_to_consider if tuple(edge) not in current_existing_set]

        if not new_edges_struct: return [], []
        new_edges_fwd.extend(new_edges_struct)
        attr_val = default_attr_val.clone().detach().reshape(default_attr_shape)
        new_edge_attrs_fwd.extend([attr_val] * len(new_edges_struct))

        # Label nodes and existing edges
        tag = f"Structural/{scenario_name}"
        for p_idx in providers:
            gt_node_labels[provider_type][p_idx] = 1
            _update_tracking(anomaly_tracking, 'node', provider_type, p_idx, tag)
        for m_idx in members:
            gt_node_labels[member_type][m_idx] = 1
            _update_tracking(anomaly_tracking, 'node', member_type, m_idx, tag)
        _add_edges_and_update_labels(modified_data, gt_edge_labels_dict, target_edge_type, [], device, label_existing=False, nodes_involved=all_involved_nodes)

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
    device: torch.device
) -> Tuple[bool, Optional[int]]: # Point 11: Return (modified, swap_partner_idx)
    """Injects an attribute anomaly based on the scenario."""
    print(f"    Injecting Attribute Scenario: {scenario_name} on {target_node_type} {target_node_idx}")
    modified = False
    swap_partner_idx_out = None # For Point 11
    tag_base = f"Attribute/{scenario_name}"

    if not hasattr(modified_data[target_node_type], 'x') or modified_data[target_node_type].x is None:
         print(f"      Skipping {scenario_name}: Node type '{target_node_type}' has no features 'x'.")
         return False, None
    node_x = modified_data[target_node_type].x.to(device)

    if target_node_type not in stats['node'] or 'features' not in stats['node'][target_node_type]:
        print(f"      Skipping {scenario_name}: Node type '{target_node_type}' has no precomputed feature statistics.")
        return False, None

    node_stats = stats['node'][target_node_type]['features']
    all_means = node_stats['all_means'].to(device)
    all_stds = node_stats['all_stds'].to(device)
    all_p01 = node_stats['all_p01'].to(device)
    all_p99 = node_stats['all_p99'].to(device)

    original_features = node_x[target_node_idx].clone()
    perturbed_features = original_features.clone()

    if scenario_name == "identity_swap":
        pool_size = params['swap_pool_size']
        metric = params['swap_metric']
        num_nodes = node_x.shape[0]
        potential_pool_indices = list(set(range(num_nodes)) - {target_node_idx})
        if not potential_pool_indices: return False, None
        current_pool_size = min(pool_size, len(potential_pool_indices))
        pool_indices = np.random.choice(potential_pool_indices, size=current_pool_size, replace=False)
        pool_indices_tensor = torch.tensor(pool_indices, device=device)

        target_feat = perturbed_features.unsqueeze(0)
        pool_feats = node_x[pool_indices_tensor]

        if metric == 'euclidean': distances = torch.cdist(target_feat, pool_feats, p=2.0).squeeze(0)
        elif metric == 'cosine': distances = 1.0 - F.cosine_similarity(target_feat, pool_feats, dim=1)
        else: raise ValueError("Invalid node_attr_swap_metric.")
        distances = torch.nan_to_num(distances, nan=-torch.inf)
        if distances.numel() == 0 or torch.all(torch.isinf(distances)): return False, None

        most_distant_pool_idx = torch.argmax(distances).item()
        swap_partner_idx = pool_indices[most_distant_pool_idx]
        swap_partner_idx_out = swap_partner_idx # For return value

        feat_swap_original = node_x[swap_partner_idx].clone()
        modified_data[target_node_type].x[target_node_idx] = feat_swap_original
        modified_data[target_node_type].x[swap_partner_idx] = original_features

        gt_node_labels[target_node_type][target_node_idx] = 1
        gt_node_labels[target_node_type][swap_partner_idx] = 1
        _update_tracking(anomaly_tracking, 'node', target_node_type, target_node_idx, tag_base + "_target")
        _update_tracking(anomaly_tracking, 'node', target_node_type, swap_partner_idx, tag_base + "_partner")
        modified = True

    else: # General attribute perturbation logic
        # --- Determine features and indices ---
        feature_specs = scenario_config.get('features', [])
        perturb_direction = scenario_config.get('perturb_direction') # 'high', 'low', or None
        feature_indices = []
        target_values = {} # For scenarios specifying exact target values {idx: val}
        direction_map = {} # For scenarios specifying direction per feature {idx: 'high'/'low'}

        if isinstance(feature_specs, list): # List of names, use global direction
            for fname in feature_specs:
                if fname in node_stats['map']:
                    idx = node_stats['map'][fname]
                    feature_indices.append(idx)
                    if perturb_direction: direction_map[idx] = perturb_direction
                else: print(f"Warning: Feature '{fname}' not found. Skipping.")
        elif isinstance(feature_specs, dict): # Dict specifying values or directions
             for fname, direction_or_value in feature_specs.items():
                 if fname in node_stats['map']:
                     idx = node_stats['map'][fname]
                     feature_indices.append(idx)
                     if isinstance(direction_or_value, str) and direction_or_value in ['high', 'low']:
                         direction_map[idx] = direction_or_value
                     elif isinstance(direction_or_value, (float, int)):
                         target_values[idx] = direction_or_value
                     else: print(f"Warning: Invalid spec '{direction_or_value}' for feature '{fname}'.")
                 else: print(f"Warning: Feature '{fname}' not found. Skipping.")
        else: # Fallback: perturb random fraction
             num_features = perturbed_features.size(0)
             num_to_perturb = max(1, int(params['attr_perturb_frac'] * num_features))
             feature_indices = np.random.choice(num_features, num_to_perturb, replace=False).tolist()
             print(f"      Perturbing {num_to_perturb} random features for {scenario_name}.")
             # Use default direction (away from mean) if none specified globally
             if not perturb_direction: perturb_direction = 'away'


        if not feature_indices: return False, None
        print(f"      Perturbing indices {feature_indices} for {scenario_name}.")
        feature_indices_tensor = torch.tensor(feature_indices, device=device, dtype=torch.long)

        # --- Perturbation Logic (Point 3) ---
        p_low_target, p_high_target = params['attr_perturb_percentiles']
        c_low, c_high = params['attr_perturb_std_factor_range']
        c = random.uniform(c_low, c_high)

        for idx in feature_indices:
            idx_tensor = torch.tensor([idx], device=device, dtype=torch.long) # Tensor for indexing stats
            mean = all_means[idx_tensor]
            std = all_stds[idx_tensor]
            p01 = all_p01[idx_tensor]
            p99 = all_p99[idx_tensor]

            # Determine target direction for this specific feature
            current_direction = direction_map.get(idx, perturb_direction) # Use specific, then global

            if idx in target_values: # Set specific value
                 perturbed_features[idx] = torch.tensor(target_values[idx], device=device, dtype=perturbed_features.dtype)
                 modified = True
            elif current_direction == 'high':
                 target_val = p99 if not torch.isnan(p99) else mean + c * std
                 perturbed_features[idx] = torch.max(target_val, mean + c * std) # Ensure significantly higher
                 modified = True
            elif current_direction == 'low':
                 target_val = p01 if not torch.isnan(p01) else mean - c * std
                 perturbed_features[idx] = torch.min(target_val, mean - c * std) # Ensure significantly lower
                 # Clamping for specific features like period
                 if scenario_name == "member_over_consulting" or scenario_name == "provider_nonaccredited_highfreq":
                     perturbed_features[idx] = torch.clamp(perturbed_features[idx], min=0.0)
                 modified = True
            elif current_direction == 'away': # Default: push away from mean
                noise_direction = torch.sign(torch.randn(1, device=device))
                noise_magnitude = c * std
                perturbed_features[idx] = mean + noise_direction * noise_magnitude
                modified = True
            # Else: No direction specified for this feature, skip? Or use default 'away'? Let's skip.

        if modified:
             modified_data[target_node_type].x[target_node_idx] = perturbed_features.to(original_features.dtype)
             gt_node_labels[target_node_type][target_node_idx] = 1
             _update_tracking(anomaly_tracking, 'node', target_node_type, target_node_idx, tag_base)

    return modified, swap_partner_idx_out


def _inject_edge_attribute_anomalies(
    modified_data: HeteroData,
    gt_edge_labels_dict: Dict[Tuple, torch.Tensor],
    anomaly_tracking: Dict,
    target_edge_type: Tuple,
    num_to_inject: int,
    stats: Dict,
    params: Dict,
    device: torch.device
    ):
    """Injects attribute anomalies on existing edges using specified method."""
    if num_to_inject <= 0: return
    if target_edge_type not in modified_data.edge_types or not hasattr(modified_data[target_edge_type], 'edge_attr') or modified_data[target_edge_type].edge_attr is None:
        print(f"  Skipping edge attribute injection: No edge attributes found for {target_edge_type}.")
        return

    edge_stat = stats['edge'].get(target_edge_type, {})
    if not edge_stat:
         print(f"  Skipping edge attribute injection: No stats found for {target_edge_type}.")
         return

    current_edge_attr = modified_data[target_edge_type].edge_attr.to(device)
    current_edge_labels = gt_edge_labels_dict[target_edge_type].to(device)

    available_edge_indices_tensor = torch.where(current_edge_labels == 0)[0]
    available_edge_indices = available_edge_indices_tensor.tolist()

    if not available_edge_indices:
        print("  No non-anomalous edges available for attribute injection.")
        return

    num_can_inject = min(num_to_inject, len(available_edge_indices))
    selected_edge_indices = random.sample(available_edge_indices, num_can_inject)

    method = params.get('edge_attr_method', 'poisson') # Default to poisson
    print(f"  Injecting {num_can_inject} edge attribute anomalies on {target_edge_type} using method: {method}...")

    dtype = edge_stat['dtype']
    shape = edge_stat['shape']
    mean_ea = edge_stat.get('mean', 1.0)
    std_ea = edge_stat.get('std', 1.0)
    p01 = edge_stat.get('p01', 0.0)
    p99 = edge_stat.get('p99', max(5.0, mean_ea + 3*std_ea)) # Ensure p99 is reasonably high

    modified_count = 0
    for edge_idx in selected_edge_indices:
        c = random.uniform(params['edge_attr_c_range'][0], params['edge_attr_c_range'][1])
        poisson_scale = random.uniform(params['edge_attr_poisson_scale_range'][0], params['edge_attr_poisson_scale_range'][1])

        new_val = None
        if method == 'poisson':
            # Point 9: Poisson method - Sample from Poisson(lambda) where lambda is extreme
            target_lambda_high = max(1.0, mean_ea + poisson_scale * std_ea) # Ensure lambda >= 1 for non-zero samples
            target_lambda_low = max(0.0, mean_ea - poisson_scale * std_ea) # Allow zero lambda
            # Decide high/low randomly or based on a param? Let's do random for now.
            target_lambda = target_lambda_high if random.random() < 0.5 else target_lambda_low
            # Ensure lambda isn't NaN
            if np.isnan(target_lambda): target_lambda = p99 if random.random() < 0.5 else p01

            sampled_val = np.random.poisson(target_lambda)
            new_val = float(sampled_val) # Convert numpy float
            tag = "Attribute/AbnormalClaimVolume_Poisson"

        elif method == 'extreme_quantile':
            direction = params.get('edge_attr_direction', 'high')
            tag = "Attribute/AbnormalClaimVolume_Extreme"
            if direction == 'high':
                new_val = max(p99, mean_ea + c * std_ea)
            elif direction == 'low':
                new_val = min(p01, mean_ea - c * std_ea)
                new_val = max(0.0, new_val) # Clamp at zero
            else: # 'both'
                 if random.random() < 0.5:
                     new_val = max(p99, mean_ea + c * std_ea)
                 else:
                     new_val = min(p01, mean_ea - c * std_ea)
                     new_val = max(0.0, new_val)
        else:
            print(f"Warning: Unknown edge_attr_method '{method}'. Skipping edge {edge_idx}.")
            continue

        # Ensure new_val is not None before proceeding
        if new_val is not None:
            # Round if original dtype is integer, clamp negatives
            if torch.is_floating_point(torch.tensor(0, dtype=dtype)):
                new_val = max(0.0, new_val) # Clamp float
            else:
                new_val = max(0, round(new_val)) # Round and clamp integer

            new_val_tensor = torch.tensor([new_val], device=device).reshape(shape).to(dtype)
            modified_data[target_edge_type].edge_attr[edge_idx] = new_val_tensor
            gt_edge_labels_dict[target_edge_type][edge_idx] = 1
            _update_tracking(anomaly_tracking, 'edge', target_edge_type, edge_idx, tag)
            modified_count += 1

    print(f"    Modified attributes of {modified_count} existing edges.")


# === Main Injection Function ===

def inject_scenario_anomalies(
    data: HeteroData,
    p_node_anomalies: float = 0.05,
    p_edge_anomalies: float = 0.02,
    lambda_structural: float = 0.5,
    num_structural_instances: Optional[int] = None, # Point 5: Alternative budget
    num_attribute_instances: Optional[int] = None, # Point 5: Alternative budget
    scenario_params: Optional[Dict[str, Any]] = None,
    provider_node_type: str = 'provider',
    member_node_type: str = 'member',
    target_edge_type: Tuple[str, str, str] = ('provider', 'to', 'member'),
    seed: Optional[int] = None
    ) -> Tuple[HeteroData, Dict[str, torch.Tensor], Dict[Tuple, torch.Tensor], Dict[str, Dict]]:
    """Injects synthetic anomalies based on predefined scenarios."""

    # === Seeding ===
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if hasattr(torch, 'mps') and torch.backends.mps.is_available():
             try: torch.mps.manual_seed(seed)
             except AttributeError: print("Warning: torch.mps.manual_seed not found.")
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    # === Device Setup ===
    if torch.cuda.is_available(): device = torch.device('cuda')
    elif torch.backends.mps.is_available(): device = torch.device('mps')
    else: device = torch.device('cpu')
    print(f"--- Injecting Scenario-Based Anomalies on Device: {device} ---")

    # === Parameter & Data Setup ===
    modified_data = data.clone().to(device)
    params = DEFAULT_PARAMS.copy()
    if scenario_params: params.update(scenario_params)
    reverse_edge_type = (target_edge_type[2], target_edge_type[1], target_edge_type[0])
    if reverse_edge_type not in modified_data.edge_types: reverse_edge_type = None

    # === Feature Identification (Point 1 & 12) ===
    node_features_needed = defaultdict(list)
    all_feature_names = defaultdict(list) # Store all available feature names per type

    for ntype in [provider_node_type, member_node_type]:
        if hasattr(data[ntype], 'feature_names') and data[ntype].feature_names:
             all_feature_names[ntype] = data[ntype].feature_names
        elif hasattr(data[ntype], 'x') and data[ntype].x is not None:
             # Fallback: assume sequential if no names provided
             num_feats = data[ntype].x.shape[1]
             all_feature_names[ntype] = [f"{ntype}_feat_{i}" for i in range(num_feats)]
             print(f"Warning: No feature names found for {ntype}. Assuming generic names: {all_feature_names[ntype][:3]}...")


    for sc_name, sc_config in SCENARIOS.items():
        if sc_config['type'] == 'attribute' and sc_config.get('requires_features', False):
             features_spec = sc_config.get('features')
             sc_node_type = sc_config['node_type']
             node_types_to_check = [provider_node_type, member_node_type] if sc_node_type == 'both' else [sc_node_type]

             for ntype in node_types_to_check:
                 required_features = []
                 if isinstance(features_spec, list): required_features = features_spec
                 elif isinstance(features_spec, dict): required_features = list(features_spec.keys())
                 elif sc_name == 'identity_swap': # Needs all features for distance
                      required_features = ["__ALL__"]

                 # Check if features exist and add to needed list
                 valid_features = []
                 for fname in required_features:
                     if fname == "__ALL__":
                          valid_features.extend(all_feature_names[ntype])
                          # Ensure __ALL__ is added if needed later for stats key check
                          if "__ALL__" not in node_features_needed[ntype]:
                              node_features_needed[ntype].append("__ALL__")
                     elif fname in all_feature_names[ntype]:
                          valid_features.append(fname)
                     else:
                          print(f"Warning: Feature '{fname}' required by {sc_name} not found in available features for {ntype}.")

                 node_features_needed[ntype].extend(valid_features)


    # Finalize feature list for stats
    final_node_features_for_stats = defaultdict(list)
    for ntype, features in node_features_needed.items():
         unique_features = sorted(list(set(f for f in features if f != "__ALL__")))
         # If __ALL__ was needed (e.g., for swap), but we have specific features,
         # we compute stats only for specific ones. Swap will use all features but maybe only mean/std.
         final_node_features_for_stats[ntype] = unique_features
         # If only __ALL__ is present, compute stats for everything available
         if not unique_features and "__ALL__" in features and all_feature_names[ntype]:
              final_node_features_for_stats[ntype] = all_feature_names[ntype]
              print(f"Computing stats for ALL available features for {ntype} due to generic requirement.")


    # === Compute Stats ===
    stats = _compute_graph_stats(modified_data, final_node_features_for_stats, device)
    # Store all feature names in stats for swap/generic use
    for ntype, fnames in all_feature_names.items():
        if ntype in stats['node'] and 'features' in stats['node'][ntype]:
             stats['node'][ntype]['features']['all_feature_names'] = fnames
        elif ntype in stats['node']: # Ensure features dict exists even if no specific stats computed
             stats['node'][ntype]['features'] = {'all_feature_names': fnames}


    # === Initialize GT & Tracking ===
    gt_node_labels = {nt: torch.zeros(modified_data[nt].num_nodes, dtype=torch.long, device=device) for nt in modified_data.node_types}
    gt_edge_labels_dict = {et: torch.zeros(modified_data[et].edge_index.shape[1], dtype=torch.long, device=device) for et in modified_data.edge_types}
    anomaly_tracking = {'node': defaultdict(lambda: defaultdict(list)), 'edge': defaultdict(list)}

    # === Node Anomaly Planning (Point 5 & User Question Refinement) ===
    print("\n--- Phase 1: Node Anomaly Planning & Injection ---")
    structural_pools = {}
    attribute_pools = {}
    total_nodes_selected = 0

    # 1. Select initial node pools based on percentage
    for node_type in [provider_node_type, member_node_type]:
         num_nodes = modified_data[node_type].num_nodes
         if num_nodes == 0: continue
         selected_indices = _select_nodes_for_anomaly(num_nodes, p_node_anomalies)
         total_nodes_selected += len(selected_indices)
         s_nodes, a_nodes = _partition_nodes(selected_indices, lambda_structural)
         structural_pools[node_type] = set(s_nodes)
         attribute_pools[node_type] = set(a_nodes)
         print(f"  {node_type}: Selected {len(selected_indices)}, Struct Pool {len(s_nodes)}, Attr Pool {len(a_nodes)}")

    # 2. Plan number of instances (Point 5) - Simplified approach: target ~even distribution
    # Calculate target instances based on pools (can be refined with explicit num_structural_instances etc.)
    target_struct_instances = max(1, int(sum(len(p) for p in structural_pools.values()) / 5)) # Rough guess: avg 5 nodes per struct instance?
    target_attr_instances = sum(len(p) for p in attribute_pools.values()) # Each node gets one attribute instance

    structural_plan = [] # List of {'name': sc_name, 'nodes': {ntype: [idx]}}
    attribute_plan = []

    # --- Plan Structural Instances ---
    # Prioritize based on complexity or budget split? Let's try to distribute somewhat evenly.
    possible_struct_scenarios = {
        s_name: s_config for s_name, s_config in SCENARIOS.items() if s_config['type'] == 'structural'
    }
    planned_struct_count = 0
    available_struct_pools = {k: v.copy() for k,v in structural_pools.items()} # Work on copies

    struct_sc_names = list(possible_struct_scenarios.keys())
    random.shuffle(struct_sc_names)

    while planned_struct_count < target_struct_instances and any(available_struct_pools.values()):
        made_progress = False
        for sc_name in struct_sc_names: # Cycle through scenario types
            if planned_struct_count >= target_struct_instances: break
            sc_config = possible_struct_scenarios[sc_name]
            sc_node_type = sc_config['node_type']

            # Determine node requirements and check pools
            required_nodes = defaultdict(int)
            deg_percentiles = {} # {ntype: percentile_param_name}
            if sc_name == "collusion_ring":
                required_nodes[provider_node_type] = params['k_p_collusion']
                required_nodes[member_node_type] = params['k_m_collusion']
                deg_percentiles[provider_node_type] = params['max_degree_percentile_collusion']
                deg_percentiles[member_node_type] = params['max_degree_percentile_collusion']
            elif sc_name == "over_referral_clique":
                 required_nodes[provider_node_type] = params['k_p_clique']
                 deg_percentiles[provider_node_type] = params['max_degree_percentile_clique_prov']
            elif sc_name == "member_clique":
                 required_nodes[member_node_type] = params['k_m_clique']
                 # Degree constraint applies to target providers, handled inside injection function
            elif sc_name == "identity_theft_doctor_touring":
                 required_nodes[member_node_type] = 1
                 # Degree constraint applies to target providers, handled inside injection function

            # Check if enough nodes are available in the *current* pools
            can_plan_instance = True
            candidate_nodes = {}
            for ntype, req_count in required_nodes.items():
                 pool = available_struct_pools.get(ntype, set())
                 # Apply degree constraint for selection *from the pool*
                 deg_percentile = deg_percentiles.get(ntype)
                 valid_pool_nodes = list(pool)
                 if deg_percentile is not None and ntype in stats['node'] and stats['node'][ntype]['degree']['values'] is not None:
                      degrees = stats['node'][ntype]['degree']['values']
                      threshold_val = torch.quantile(degrees.float().to(device), deg_percentile).item()
                      valid_pool_nodes = [n for n in valid_pool_nodes if degrees[n].item() <= threshold_val]

                 if len(valid_pool_nodes) >= req_count:
                      candidate_nodes[ntype] = valid_pool_nodes
                 else:
                      can_plan_instance = False; break

            # If possible, plan one instance and reserve nodes
            if can_plan_instance:
                 instance_nodes = {}
                 nodes_to_reserve = set()
                 for ntype, candidates in candidate_nodes.items():
                      req_count = required_nodes[ntype]
                      chosen = random.sample(candidates, req_count)
                      instance_nodes[ntype] = chosen
                      nodes_to_reserve.update(chosen)

                 # Remove reserved nodes from available pools
                 for ntype, indices in instance_nodes.items():
                      available_struct_pools[ntype].difference_update(indices)

                 print(f"    Planning instance: {sc_name} involving {instance_nodes}")
                 structural_plan.append({'name': sc_name, 'nodes': instance_nodes})
                 planned_struct_count += 1
                 made_progress = True

        if not made_progress: # Avoid infinite loop if no scenario can be planned
             print("    No further structural scenarios can be planned with remaining nodes.")
             break


    # --- Plan Attribute Instances ---
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
            attribute_plan.append({'name': chosen_sc_name, 'nodes': {node_type: [node_idx]}})


    # === Execute Planned Scenarios ===
    print(f"\n  Executing {len(structural_plan)} structural and {len(attribute_plan)} attribute planned instances...")
    processed_nodes_struct = defaultdict(set)
    processed_nodes_attr = defaultdict(set)
    edges_added_struct = [] # Store tuples (fwd_edges, fwd_attrs)

    # 1. Execute Structural Scenarios
    for instance in structural_plan:
        sc_name = instance['name']
        sc_nodes = instance['nodes']
        sc_config = SCENARIOS[sc_name]

        # Pass edge labels dict for potential modification of existing edge labels
        new_edges_fwd, new_attrs_fwd = _inject_structural_scenario(
            sc_name, sc_config, modified_data, gt_node_labels, gt_edge_labels_dict,
            anomaly_tracking, sc_nodes, stats, params, device,
            target_edge_type, reverse_edge_type
        )
        if new_edges_fwd:
             edges_added_struct.append((new_edges_fwd, new_attrs_fwd))

        for ntype, indices in sc_nodes.items():
            processed_nodes_struct[ntype].update(indices)

    # --- Add Structurally Added Edges Now ---
    # This ensures edge indices are updated before attribute scenarios might use them (unlikely but safer)
    print("  Adding structurally generated edges to the graph...")
    total_edges_added_struct = 0
    all_new_edges_fwd = []
    all_new_attrs_fwd = []
    for edges_fwd, attrs_fwd in edges_added_struct:
        all_new_edges_fwd.extend(edges_fwd)
        all_new_attrs_fwd.extend(attrs_fwd)

    num_added_fwd = _add_edges_and_update_labels(
        modified_data, gt_edge_labels_dict, target_edge_type,
        all_new_edges_fwd, device, all_new_attrs_fwd,
        label_existing=False # Labelling of existing edges happened inside _inject_structural
    )
    total_edges_added_struct += num_added_fwd
    # Handle reverse edges if needed (Point 8: Assuming shared attributes, just add labels)
    if reverse_edge_type and num_added_fwd > 0:
        num_orig_rev = gt_edge_labels_dict.get(reverse_edge_type, torch.empty(0)).shape[0]
        num_total_rev = modified_data.get(reverse_edge_type, {}).get('edge_index', torch.empty(2,0)).shape[1]

        if reverse_edge_type in gt_edge_labels_dict:
             print(f"    Updating labels for reverse edge type {reverse_edge_type} for {num_added_fwd} added edges.")
             rev_labels = gt_edge_labels_dict[reverse_edge_type].to(device)
             # Check if reverse edge list was also updated in structure (assume yes if forward was)
             if num_total_rev == num_orig_rev + num_added_fwd:
                 new_rev_labels = torch.ones(num_added_fwd, dtype=torch.long, device=device)
                 gt_edge_labels_dict[reverse_edge_type] = torch.cat([rev_labels, new_rev_labels])
             else:
                  print(f"    Warning: Size mismatch when adding reverse labels for {reverse_edge_type}. Expected {num_orig_rev + num_added_fwd}, got {num_total_rev}. Skipping reverse label update.")
        else:
             print(f"    Warning: Reverse edge type {reverse_edge_type} not found in gt_edge_labels_dict. Cannot update labels.")

    print(f"  Added {total_edges_added_struct} edges from structural anomalies.")


    # 2. Execute Attribute Scenarios
    swap_partners = {} # Track swap partners {target_idx: partner_idx}
    for instance in attribute_plan:
         sc_name = instance['name']
         sc_nodes = instance['nodes']
         sc_config = SCENARIOS[sc_name]
         node_type = list(sc_nodes.keys())[0]
         node_idx = sc_nodes[node_type][0]

         # Point 11: Get swap partner info
         modified, swap_partner_idx = _inject_attribute_scenario(
             sc_name, sc_config, modified_data, gt_node_labels, anomaly_tracking,
             node_type, node_idx, stats, params, device
         )
         if modified:
             processed_nodes_attr[node_type].add(node_idx)
             if swap_partner_idx is not None:
                  # Mark swap partner as processed by attribute injection as well
                  processed_nodes_attr[node_type].add(swap_partner_idx)
                  swap_partners[node_idx] = swap_partner_idx


    # 3. Apply Combined Anomalies
    print(f"\n  Checking for Combined Anomalies (Struct -> Attribute)...")
    combined_count = 0
    nodes_already_combined = defaultdict(set)

    for node_type, indices in processed_nodes_struct.items():
        applicable_attr_scenarios = [
            s_name for s_name, s_config in SCENARIOS.items()
            if s_config['type'] == 'attribute' and (s_config['node_type'] == node_type or s_config['node_type'] == 'both')
            and s_config.get('requires_features', False)
        ]
        if not applicable_attr_scenarios: continue

        for node_idx in indices:
            # Check if node ALSO got a primary attribute anomaly (e.g., from swap partner)
            # or if it was already made combined
            if node_idx in processed_nodes_attr[node_type] or node_idx in nodes_already_combined[node_type]:
                continue

            if random.random() < 0.5:
                 chosen_attr_sc_name = random.choice(applicable_attr_scenarios)
                 chosen_attr_sc_config = SCENARIOS[chosen_attr_sc_name]
                 print(f"      Adding combined attribute '{chosen_attr_sc_name}' to struct {node_type} {node_idx}")

                 modified, swap_partner_idx = _inject_attribute_scenario(
                     chosen_attr_sc_name, chosen_attr_sc_config, modified_data, gt_node_labels,
                     anomaly_tracking, node_type, node_idx, stats, params, device
                 )
                 if modified:
                     _update_tracking(anomaly_tracking, 'node', node_type, node_idx, "Combined")
                     nodes_already_combined[node_type].add(node_idx)
                     combined_count += 1
                     # Point 11: If combined involved a swap, mark the partner too
                     if swap_partner_idx is not None and swap_partner_idx not in processed_nodes_struct.get(node_type, set()):
                          print(f"      Marking swap partner {node_type} {swap_partner_idx} also as Combined.")
                          _update_tracking(anomaly_tracking, 'node', node_type, swap_partner_idx, "Combined")
                          nodes_already_combined[node_type].add(swap_partner_idx)


    # === Edge Anomaly Injection (Attribute Tweaking) ===
    print("\n--- Phase 2: Edge Attribute Anomaly Injection ---")
    final_edge_count = modified_data[target_edge_type].edge_index.shape[1] if target_edge_type in modified_data.edge_types else 0
    target_total_anom_edges = int(final_edge_count * p_edge_anomalies)
    # Recalculate current anomalous edges accurately
    current_anom_edges = gt_edge_labels_dict[target_edge_type].sum().item() if target_edge_type in gt_edge_labels_dict else 0
    remaining_edge_budget = max(0, target_total_anom_edges - current_anom_edges)

    print(f"  Target anomalous edges: {target_total_anom_edges} ({p_edge_anomalies*100:.1f}% of {final_edge_count})")
    print(f"  Edges already anomalous (structural + existing labeled): {current_anom_edges}")
    print(f"  Remaining budget for edge attribute tweaking: {remaining_edge_budget}")

    if target_edge_type in modified_data.edge_types:
        _inject_edge_attribute_anomalies(
            modified_data, gt_edge_labels_dict, anomaly_tracking,
            target_edge_type, remaining_edge_budget, stats, params, device
            )
        # Point 8: Update reverse labels for edges modified here
        if reverse_edge_type and reverse_edge_type in gt_edge_labels_dict:
             # Find indices modified in the forward direction during this step
             # This requires tracking which indices were selected in _inject_edge_attribute_anomalies
             # Or, less efficiently, re-check labels against original state (difficult)
             # Simplification: Assume the labeling reflects modifications; just ensure consistency.
             # If forward label is 1, ensure reverse label is 1 (if symmetrical)
             # This needs mapping between forward/reverse edge indices if not identical!
             print(f"    Skipping reverse edge label update for attribute tweaks due to index mapping complexity.")


    # === Final Graph Update & Cleanup ===
    print("\nConsolidating graph modifications...")
    # (Final label assignment and tracking formatting logic remains largely the same)
    # ...
    final_anomaly_tracking = {'node': {}, 'edge': {}}
    for edge_type, final_labels in gt_edge_labels_dict.items():
        if edge_type in modified_data.edge_types:
            current_num_edges = modified_data[edge_type].edge_index.shape[1]
            labels_on_device = final_labels.to(device)
            if len(labels_on_device) != current_num_edges:
                 print(f"ERROR FINAL: Mismatch gt_edge_labels ({len(labels_on_device)}) vs final edge_index ({current_num_edges}) for {edge_type}.")
                 if len(labels_on_device) < current_num_edges:
                      padding = torch.zeros(current_num_edges - len(labels_on_device), dtype=torch.long, device=device)
                      gt_edge_labels_dict[edge_type] = torch.cat([labels_on_device, padding])
                      print(f"      Resized labels for {edge_type} with padding.")
                 else:
                      gt_edge_labels_dict[edge_type] = labels_on_device[:current_num_edges]
                      print(f"      Truncated labels for {edge_type}.")
            else:
                 gt_edge_labels_dict[edge_type] = labels_on_device
            modified_data[edge_type].y = gt_edge_labels_dict[edge_type]
        else: pass # Ignore labels for edge types not present

    for ntype, nodes in anomaly_tracking['node'].items():
        if ntype in gt_node_labels: # Check node type exists
            final_anomaly_tracking['node'][ntype] = {
                idx: sorted(list(set(tags))) for idx, tags in nodes.items() if gt_node_labels[ntype][idx] == 1
            }
    for (etype, idx), tags in anomaly_tracking['edge'].items():
         if etype in modified_data.edge_types and idx < modified_data[etype].edge_index.shape[1]:
             if etype in gt_edge_labels_dict and gt_edge_labels_dict[etype][idx] == 1:
                 final_anomaly_tracking['edge'][(etype, idx)] = sorted(list(set(tags)))


    # Final summary
    final_anom_nodes = sum(gt.sum().item() for gt in gt_node_labels.values())
    final_anom_edges = sum(gt.sum().item() for gt in gt_edge_labels_dict.values() if gt is not None)
    total_nodes_sum = sum(data[nt].num_nodes for nt in data.node_types)
    print(f"\nAnomaly injection finished.")
    print(f"  Total nodes: {total_nodes_sum}")
    if total_nodes_sum > 0: print(f"  Total anomalous nodes: {final_anom_nodes} ({final_anom_nodes / total_nodes_sum * 100:.2f}%)")
    else: print(f"  Total anomalous nodes: 0")

    if target_edge_type in modified_data.edge_types:
        final_edge_count = modified_data[target_edge_type].edge_index.shape[1]
        final_anom_edge_count = gt_edge_labels_dict.get(target_edge_type, torch.empty(0)).sum().item()
        print(f"  Total final edges ({target_edge_type}): {final_edge_count}")
        if final_edge_count > 0: print(f"  Total anomalous edges ({target_edge_type}): {final_anom_edge_count} ({final_anom_edge_count / final_edge_count * 100:.2f}%)")
        else: print(f"  Total anomalous edges ({target_edge_type}): 0")


    return modified_data, gt_node_labels, gt_edge_labels_dict, final_anomaly_tracking



import pandas as pd
import torch
from collections import defaultdict
from typing import Tuple, List, Dict, Optional, Literal, Any
from torch_geometric.data import HeteroData # Import if not already imported

def _parse_scenario_tags(tags: List[str]) -> List[str]:
    """Helper to extract base scenario names from tracking tags."""
    base_scenarios = set()
    for tag in tags:
        if tag == "Combined":
            continue
        # Remove prefixes like Structural/, Attribute/
        if '/' in tag:
            tag = tag.split('/', 1)[1]
        # Remove suffixes like _target, _partner
        if tag.endswith("_target"):
            tag = tag[:-7]
        elif tag.endswith("_partner"):
             tag = tag[:-8]
        # Special handling for edge attribute tags if needed
        # e.g., if tag was "Attribute/AbnormalClaimVolume_Poisson" -> "AbnormalClaimVolume_Poisson"
        base_scenarios.add(tag)
    return sorted(list(base_scenarios))

def summarize_injected_anomalies(
    modified_data: HeteroData,
    gt_node_labels: Dict[str, torch.Tensor],
    gt_edge_labels_dict: Dict[Tuple, torch.Tensor],
    anomaly_tracking: Dict[str, Dict],
    target_edge_type: Tuple = ('provider', 'to', 'member') # To focus edge summary
    ):
    """
    Provides a detailed summary of injected anomalies.

    Args:
        modified_data (HeteroData): Graph with injected anomalies.
        gt_node_labels (Dict[str, Tensor]): Binary labels for anomalous nodes.
        gt_edge_labels_dict (Dict[Tuple, Tensor]): Binary labels for anomalous edges.
        anomaly_tracking (Dict[str, Dict]): Detailed anomaly tags per node/edge index.
        target_edge_type (Tuple): The primary edge type to focus summary on.
    """
    print("\n--- Anomaly Injection Summary ---")

    # === 1. Overall Counts ===
    print("\n1. Overall Anomaly Counts:")
    summary_data = []
    total_nodes = 0
    total_anom_nodes = 0
    for ntype, labels in gt_node_labels.items():
        num_nodes = labels.numel()
        num_anom = labels.sum().item()
        percent = (num_anom / num_nodes * 100) if num_nodes > 0 else 0
        summary_data.append({
            "Element Type": "Node",
            "Specific Type": ntype,
            "Total Count": num_nodes,
            "Anomalous Count": num_anom,
            "Percentage (%)": f"{percent:.2f}"
        })
        total_nodes += num_nodes
        total_anom_nodes += num_anom

    total_edges = 0
    total_anom_edges = 0
    for etype, labels in gt_edge_labels_dict.items():
         # Use final edge count from modified_data
        num_edges = modified_data[etype].edge_index.shape[1] if etype in modified_data.edge_types else 0
        num_anom = labels.sum().item()
        percent = (num_anom / num_edges * 100) if num_edges > 0 else 0
        summary_data.append({
            "Element Type": "Edge",
            "Specific Type": str(etype),
            "Total Count": num_edges,
            "Anomalous Count": num_anom,
            "Percentage (%)": f"{percent:.2f}"
        })
        # Only add target edge type to overall summary counts if desired
        if etype == target_edge_type:
            total_edges += num_edges
            total_anom_edges += num_anom

    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    print(f"\nTotal Nodes: {total_nodes}, Total Anomalous Nodes: {total_anom_nodes} ({total_anom_nodes/total_nodes*100:.2f}%)")
    print(f"Total Target Edges {target_edge_type}: {total_edges}, Total Anomalous Target Edges: {total_anom_edges} ({total_anom_edges/total_edges*100:.2f}%)")


    # === 2. Process Tracking Data ===
    node_details = defaultdict(lambda: {"type": set(), "scenarios": set()}) # {ntype: {idx: {"type":{S/A/C}, "scenarios":{sc1, sc2}}}}
    edge_details = defaultdict(lambda: {"type": set(), "scenarios": set()}) # {(etype, idx): {"type":{S/A}, "scenarios":{sc1}}}

    # Process Nodes
    if 'node' in anomaly_tracking:
        for ntype, nodes in anomaly_tracking['node'].items():
             if ntype not in gt_node_labels: continue # Skip if node type not in labels
             for idx, tags in nodes.items():
                 if gt_node_labels[ntype][idx] == 1: # Ensure node is actually anomalous
                     is_combined = "Combined" in tags
                     base_scenarios = _parse_scenario_tags(tags)
                     node_details[(ntype, idx)]["scenarios"].update(base_scenarios)

                     if is_combined:
                         node_details[(ntype, idx)]["type"].add("Combined")
                     # Determine primary type based on remaining tags
                     is_structural = any(t.startswith("Structural/") for t in tags)
                     is_attribute = any(t.startswith("Attribute/") for t in tags)

                     if not is_combined:
                         if is_structural: node_details[(ntype, idx)]["type"].add("Structural")
                         if is_attribute: node_details[(ntype, idx)]["type"].add("Attribute")
                         # If somehow only a partner tag exists? Default to attribute maybe?
                         if not is_structural and not is_attribute and base_scenarios:
                              node_details[(ntype, idx)]["type"].add("Attribute") # Default assumption

    # Process Edges
    edge_attr_scenarios = set() # Track scenarios that explicitly tag edges
    if 'edge' in anomaly_tracking:
        for edge_key, tags in anomaly_tracking['edge'].items():
            etype, idx = edge_key
            if etype not in gt_edge_labels_dict: continue
             # Ensure index is valid and edge is anomalous
            if idx < gt_edge_labels_dict[etype].shape[0] and gt_edge_labels_dict[etype][idx] == 1:
                base_scenarios = _parse_scenario_tags(tags)
                edge_details[edge_key]["scenarios"].update(base_scenarios)
                # Assume any tagged edge was due to an attribute modification
                edge_details[edge_key]["type"].add("Attribute")
                edge_attr_scenarios.update(base_scenarios)

    # === 3. Structural/Attribute/Combined Breakdown ===
    print("\n2. Anomaly Type Breakdown (Structural/Attribute/Combined):")
    type_summary_data = []
    node_type_counts = defaultdict(lambda: defaultdict(int)) # {ntype: {type: count}}
    for (ntype, idx), details in node_details.items():
         # Prioritize Combined -> Structural -> Attribute
         if "Combined" in details["type"]: node_type_counts[ntype]["Combined"] += 1
         elif "Structural" in details["type"]: node_type_counts[ntype]["Structural"] += 1
         elif "Attribute" in details["type"]: node_type_counts[ntype]["Attribute"] += 1
         else: node_type_counts[ntype]["Unknown"] += 1 # Should ideally be 0

    for ntype, counts in node_type_counts.items():
         total_anom = gt_node_labels[ntype].sum().item()
         type_summary_data.append({
             "Element": f"Node ({ntype})",
             "Structural": counts.get("Structural", 0),
             "Attribute": counts.get("Attribute", 0),
             "Combined": counts.get("Combined", 0),
             "Total Anom": total_anom
         })

    # Edge Type Breakdown (Simpler: Attribute vs. Implied Structural)
    # Count Attribute edges explicitly tagged
    edge_type_counts = defaultdict(lambda: defaultdict(int))
    for edge_key, details in edge_details.items():
        etype, idx = edge_key
        if "Attribute" in details["type"]:
            edge_type_counts[etype]["Attribute"] += 1

    # Calculate implied structural edges
    for etype, labels in gt_edge_labels_dict.items():
        if etype != target_edge_type: continue # Focus on target type for this summary
        total_anom = labels.sum().item()
        attr_count = edge_type_counts[etype].get("Attribute", 0)
        struct_count = total_anom - attr_count # Remainder assumed structural (added or labeled via nodes)
        edge_type_counts[etype]["Structural (Implied)"] = struct_count
        edge_type_counts[etype]["Total Anom"] = total_anom

        type_summary_data.append({
             "Element": f"Edge {etype}",
             "Structural": struct_count,
             "Attribute": attr_count,
             "Combined": 0, # Edges aren't marked combined in this scheme
             "Total Anom": total_anom
         })


    type_summary_df = pd.DataFrame(type_summary_data)
    print(type_summary_df.to_string(index=False))

    # === 4. Detailed Scenario Breakdown ===
    print("\n3. Scenario Instance Counts (Nodes):")
    node_scenario_counts = defaultdict(lambda: defaultdict(int)) # {ntype: {scenario_name: count}}
    for (ntype, idx), details in node_details.items():
         for scenario in details["scenarios"]:
             node_scenario_counts[ntype][scenario] += 1

    if node_scenario_counts:
         node_scenario_df = pd.DataFrame(node_scenario_counts).fillna(0).astype(int)
         node_scenario_df = node_scenario_df.sort_index() # Sort scenarios alphabetically
         print(node_scenario_df)
    else:
         print("  No node scenario details found in tracking.")


    print("\n4. Scenario Instance Counts (Edges - Attribute Only):")
    # Only counts edges explicitly tagged by attribute scenarios
    edge_scenario_counts = defaultdict(lambda: defaultdict(int)) # {etype: {scenario_name: count}}
    for (etype, idx), details in edge_details.items():
         if etype != target_edge_type: continue # Focus summary
         for scenario in details["scenarios"]:
             edge_scenario_counts[etype][scenario] += 1

    if edge_scenario_counts:
         edge_scenario_df = pd.DataFrame(edge_scenario_counts).fillna(0).astype(int)
         edge_scenario_df = edge_scenario_df.sort_index()
         print(edge_scenario_df)
    else:
         print(f"  No edge attribute scenario details found in tracking for {target_edge_type}.")

    print("\n--- End of Summary ---")


import pandas as pd
import torch
from collections import defaultdict
from typing import Tuple, List, Dict, Optional, Literal, Any
from torch_geometric.data import HeteroData

# Assuming the helper functions _parse_scenario_tags and format_percent
# are defined in the same scope or imported correctly.

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

def format_percent(value, total):
    """Helper to format percentage strings, handling zero total."""
    if total is not None and total > 0:
        return f"{(value / total * 100):.1f}%"
    elif value == 0 and total == 0:
         return "0.0%" # Or N/A
    else:
        return "N/A" # Indicate invalid percentage if total is 0 but value isn't

def compare_anomaly_splits(splits_data: Dict[str, Tuple[HeteroData, Dict, Dict, Dict]],
                           target_edge_type: Tuple = ('provider', 'to', 'member')):
    """
    Compares anomaly injection results across different graph splits (train, val, test).

    Args:
        splits_data (Dict): A dictionary where keys are split names (e.g., 'train', 'val', 'test')
                            and values are tuples containing:
                            (modified_graph, gt_node_labels, gt_edge_labels_dict, anomaly_tracking)
                            for that split.
        target_edge_type (Tuple): The primary edge type to focus edge summaries on.
    """
    print("\n--- Comparing Anomaly Injection Across Splits ---")

    summary_rows = []
    scenario_details = defaultdict(lambda: defaultdict(lambda: defaultdict(int))) # {scenario: {split: {element_type: count}}}
    type_details = defaultdict(lambda: defaultdict(lambda: defaultdict(int))) # {split: {element_type: {type: count}}}
    totals = defaultdict(lambda: defaultdict(int)) # {split: {element_type: total_count/total_anom_count}}

    all_scenarios = set()
    all_node_types = set()
    all_edge_types = set()

    # === 1. Gather Data for Each Split ===
    for split_name, split_outputs in splits_data.items():
        print(f"\nProcessing split: {split_name}")
        graph, node_labels, edge_labels, tracking = split_outputs

        # --- Overall Totals ---
        split_total_nodes = 0
        split_anom_nodes = 0
        for ntype, labels in node_labels.items():
            all_node_types.add(ntype)
            count = labels.numel()
            anom_count = labels.sum().item()
            totals[split_name][f'nodes_{ntype}_total'] = count
            totals[split_name][f'nodes_{ntype}_anom'] = anom_count
            split_total_nodes += count
            split_anom_nodes += anom_count
        totals[split_name]['nodes_total'] = split_total_nodes
        totals[split_name]['nodes_anom'] = split_anom_nodes

        split_total_target_edges = 0
        split_anom_target_edges = 0
        for etype, labels in edge_labels.items():
            all_edge_types.add(etype)
            count = graph.get(etype, {}).get('num_edges', 0)
            if count == 0 and etype in graph.edge_types:
                 count = graph[etype].edge_index.shape[1]
            anom_count = labels.sum().item()
            totals[split_name][f'edges_{etype}_total'] = count
            totals[split_name][f'edges_{etype}_anom'] = anom_count
            if etype == target_edge_type:
                split_total_target_edges += count
                split_anom_target_edges += anom_count
        totals[split_name][f'edges_{target_edge_type}_total'] = split_total_target_edges
        totals[split_name][f'edges_{target_edge_type}_anom'] = split_anom_target_edges

        # --- Process Tracking for Types and Scenarios ---
        # Nodes
        if 'node' in tracking:
            for ntype, nodes in tracking['node'].items():
                if ntype not in node_labels: continue
                for idx, tags in nodes.items():
                    if idx < node_labels[ntype].shape[0] and node_labels[ntype][idx] == 1:
                        base_scenarios = _parse_scenario_tags(tags)
                        all_scenarios.update(base_scenarios)
                        is_combined = "Combined" in tags

                        # Scenario Counts
                        for sc in base_scenarios:
                            scenario_details[sc][split_name][ntype] += 1

                        # Type Counts
                        element_key = f"Node ({ntype})"
                        if is_combined:
                            type_details[split_name][element_key]["Combined"] += 1
                        else:
                            is_structural = any(t.startswith("Structural/") for t in tags)
                            is_attribute = any(t.startswith("Attribute/") for t in tags)
                            if is_structural: type_details[split_name][element_key]["Structural"] += 1
                            if is_attribute: type_details[split_name][element_key]["Attribute"] += 1
                            if not is_structural and not is_attribute and base_scenarios:
                                type_details[split_name][element_key]["Attribute"] += 1 # Default

        # Edges (focus on target type for detailed breakdown)
        if 'edge' in tracking:
            element_key = f"Edge {target_edge_type}"
            for edge_key, tags in tracking['edge'].items():
                etype, idx = edge_key
                if etype != target_edge_type or etype not in edge_labels: continue
                if idx < edge_labels[etype].shape[0] and edge_labels[etype][idx] == 1:
                    base_scenarios = _parse_scenario_tags(tags)
                    all_scenarios.update(base_scenarios)

                    # Scenario Counts
                    for sc in base_scenarios:
                         # Use a generic 'edge' key for scenarios affecting edges
                        scenario_details[sc][split_name][f'Edge {etype}'] += 1

                    # Type Counts (Attribute)
                    is_attribute = any(t.startswith("Attribute/") for t in tags)
                    if is_attribute:
                        type_details[split_name][element_key]["Attribute"] += 1

            # Calculate Implied Structural Edge Count for Target Type
            total_anom_target = totals[split_name][f'edges_{target_edge_type}_anom']
            attr_count_target = type_details[split_name][element_key].get("Attribute", 0)
            struct_count_target = total_anom_target - attr_count_target
            type_details[split_name][element_key]["Structural (Implied)"] = struct_count_target

    # === 2. Build Summary DataFrame ===
    print("\n--- Summary DataFrame: Overall Proportions and Type Breakdown ---")
    summary_rows = []
    split_names = list(splits_data.keys())

    for split in split_names:
        row = {'Split': split}
        # Overall Node Stats
        row['Nodes Anom'] = totals[split]['nodes_anom']
        row['Nodes Anom (%)'] = format_percent(totals[split]['nodes_anom'], totals[split]['nodes_total'])
        # Overall Edge Stats (Target Type)
        target_edge_key = f'edges_{target_edge_type}_anom'
        target_edge_total_key = f'edges_{target_edge_type}_total'
        row['Target Edges Anom'] = totals[split].get(target_edge_key, 0)
        row['Target Edges Anom (%)'] = format_percent(totals[split].get(target_edge_key, 0), totals[split].get(target_edge_total_key, 0))

        # Type Breakdown - Nodes
        for ntype in sorted(list(all_node_types)):
            element_key = f"Node ({ntype})"
            total_anom_type = totals[split].get(f'nodes_{ntype}_anom', 0)
            struct_count = type_details[split][element_key].get("Structural", 0)
            attr_count = type_details[split][element_key].get("Attribute", 0)
            comb_count = type_details[split][element_key].get("Combined", 0)
            row[f'{ntype} Struct (%)'] = format_percent(struct_count, total_anom_type)
            row[f'{ntype} Attr (%)'] = format_percent(attr_count, total_anom_type)
            row[f'{ntype} Comb (%)'] = format_percent(comb_count, total_anom_type)

        # Type Breakdown - Edges (Target Type)
        element_key = f"Edge {target_edge_type}"
        total_anom_type = totals[split].get(target_edge_key, 0)
        struct_count = type_details[split][element_key].get("Structural (Implied)", 0)
        attr_count = type_details[split][element_key].get("Attribute", 0)
        row['Edge Struct (%)'] = format_percent(struct_count, total_anom_type)
        row['Edge Attr (%)'] = format_percent(attr_count, total_anom_type)

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    # Reorder columns for clarity
    ordered_cols = ['Split', 'Nodes Anom', 'Nodes Anom (%)', 'Target Edges Anom', 'Target Edges Anom (%)']
    for ntype in sorted(list(all_node_types)):
        ordered_cols.extend([f'{ntype} Struct (%)', f'{ntype} Attr (%)', f'{ntype} Comb (%)'])
    ordered_cols.extend(['Edge Struct (%)', 'Edge Attr (%)'])
    # Filter out columns that might not exist if a node type had 0 anomalies
    ordered_cols = [col for col in ordered_cols if col in summary_df.columns]
    summary_df = summary_df[ordered_cols]

    print(summary_df.to_string(index=False))


    # === 3. Build Detailed Scenario DataFrame ===
    print("\n--- Detailed DataFrame: Scenario Counts and Proportions ---")
    detailed_rows = []
    sorted_scenarios = sorted(list(all_scenarios))
    sorted_node_types = sorted(list(all_node_types))

    header_row = {'Category': 'Scenario'}
    count_columns = []
    percent_columns = []

    for split in split_names:
        for ntype in sorted_node_types:
            col_name = f"{split}_{ntype}"
            header_row[col_name] = 0 # Initialize counts
            count_columns.append(col_name)
        # Add edge column for target type
        edge_col_name = f"{split}_Edge_{target_edge_type}"
        header_row[edge_col_name] = 0
        count_columns.append(edge_col_name)

    for scenario in sorted_scenarios:
        row = {'Category': scenario}
        for split in split_names:
            # Node counts/percentages
            for ntype in sorted_node_types:
                col_name = f"{split}_{ntype}"
                count = scenario_details[scenario][split].get(ntype, 0)
                row[col_name] = count
                total_anom = totals[split].get(f'nodes_{ntype}_anom', 0)
                row[f"{col_name}_(%)"] = format_percent(count, total_anom)
            # Edge counts/percentages
            edge_col_name = f"{split}_Edge_{target_edge_type}"
            edge_count_key = f"Edge {target_edge_type}" # Key used in scenario_details
            edge_count = scenario_details[scenario][split].get(edge_count_key, 0)
            row[edge_col_name] = edge_count
            total_anom_edges = totals[split].get(f'edges_{target_edge_type}_anom', 0)
            row[f"{edge_col_name}_(%)"] = format_percent(edge_count, total_anom_edges)
        detailed_rows.append(row)

    detailed_df = pd.DataFrame(detailed_rows)

    # Order columns: Split1_Node1, Split1_Node1_%, Split1_Node2, Split1_Node2_%, Split1_Edge, Split1_Edge_%, Split2_Node1, ...
    final_detailed_cols = ['Category']
    for split in split_names:
        for ntype in sorted_node_types:
            col_name = f"{split}_{ntype}"
            if col_name in detailed_df.columns:
                final_detailed_cols.append(col_name)
                final_detailed_cols.append(f"{col_name}_(%)")
        # Add edge column
        edge_col_name = f"{split}_Edge_{target_edge_type}"
        if edge_col_name in detailed_df.columns:
             final_detailed_cols.append(edge_col_name)
             final_detailed_cols.append(f"{edge_col_name}_(%)")

    detailed_df = detailed_df[final_detailed_cols]
    print(detailed_df.to_string(index=False))
    
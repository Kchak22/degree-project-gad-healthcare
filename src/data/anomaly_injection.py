import pandas as pd
import torch
from torch_geometric.data import HeteroData
import numpy as np
import random
from collections import defaultdict
import math  # For ceiling division

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




def inject_simplified_anomalies(data: HeteroData, num_total_injections: int = 30,
                                node_perturb_scale: float = 2.5,
                                edge_perturb_scale: float = 3.0):
    """
    Injects simplified anomalies: Structural Only, Attribute Only, Combined.
    Also returns ground truth labels for injected edges.

    Args:
        data: The original HeteroData object.
        num_total_injections: Total number of injection events to perform.
        node_perturb_scale: Scale factor for node feature noise relative to std dev.
        edge_perturb_scale: Scale factor for edge attribute noise relative to std dev.

    Returns:
        modified_data: HeteroData with anomalies.
        gt_node_labels: Dict {'provider': tensor, 'member': tensor} node labels (0/1).
        gt_edge_labels: Dict {('src', 'rel', 'dst'): tensor} edge labels (0/1).
                        Indices match the edge_index in modified_data.
        anomaly_tracking: Dict detailing anomaly types per node.
    """
    modified_data = data.clone()
    target_edge_type = ('provider', 'to', 'member')
    reverse_edge_type = ('member', 'to', 'provider')

    if target_edge_type not in modified_data.edge_types:
        raise ValueError(f"Target edge type {target_edge_type} not found in data.")

    num_providers = modified_data['provider'].num_nodes
    num_members = modified_data['member'].num_nodes

    # --- Initialize Ground Truth ---
    gt_node_labels = {
        'provider': torch.zeros(num_providers, dtype=torch.long),
        'member': torch.zeros(num_members, dtype=torch.long)
    }
    # Initialize edge labels for existing edges as normal (0)
    gt_edge_labels = {}
    original_edge_indices = {}
    for edge_type in modified_data.edge_types:
         num_edges = modified_data[edge_type].edge_index.shape[1]
         gt_edge_labels[edge_type] = torch.zeros(num_edges, dtype=torch.long)
         original_edge_indices[edge_type] = modified_data[edge_type].edge_index.clone() # Store original indices

    anomaly_tracking = {
        'provider': defaultdict(set),
        'member': defaultdict(set)
    }

    # --- Calculate Feature Statistics (as before) ---
    feature_std_dev = {}
    for node_type in data.node_types:
        if hasattr(data[node_type], 'x') and data[node_type].x.numel() > 0:
            std = data[node_type].x.std(dim=0)
            std[std == 0] = 1.0
            feature_std_dev[node_type] = std
        else: feature_std_dev[node_type] = None

    edge_attr_mean, edge_attr_std = 1.0, 1.0
    original_edge_attr = getattr(data[target_edge_type], 'edge_attr', None)
    if original_edge_attr is not None and original_edge_attr.numel() > 0:
        edge_attr_mean = original_edge_attr.mean(dim=0)
        std = original_edge_attr.std(dim=0)
        std[std == 0] = 1.0
        edge_attr_std = std

    # --- Divide Injections into 3 Types ---
    num_per_type = math.ceil(num_total_injections / 3.0)
    injection_types = (['structural'] * num_per_type +
                       ['attribute'] * num_per_type +
                       ['combined'] * num_per_type)[:num_total_injections] # Ensure exact total number
    random.shuffle(injection_types)

    print(f"Injecting {num_total_injections} anomalies:")
    print(f"  Structural: {injection_types.count('structural')}")
    print(f"  Attribute: {injection_types.count('attribute')}")
    print(f"  Combined: {injection_types.count('combined')}")

    newly_added_edges_info = {target_edge_type: [], reverse_edge_type: []}

    for injection_type in injection_types:
        # --- Select Nodes ---
        # Make selection independent of type for attribute-only
        if injection_type == 'attribute':
            # Select random nodes without forcing structural links
             size_provider = random.randint(2, min(20, num_providers))
             size_member = random.randint(2, min(20, num_members))
             providers_anom = np.random.choice(num_providers, size=size_provider, replace=False)
             members_anom = np.random.choice(num_members, size=size_member, replace=False)
        else: # Structural or Combined - select nodes for block
            size_provider = random.randint(2, min(20, num_providers))
            size_member = random.randint(2, min(20, num_members))
            providers_anom = np.random.choice(num_providers, size=size_provider, replace=False)
            members_anom = np.random.choice(num_members, size=size_member, replace=False)

        # --- Mark Nodes (Common for all types involving selected nodes) ---
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
            anomaly_tracking['provider'][p].add('structural_block') # More specific tag
            anomaly_tracking['member'][m].add('structural_block')

            # Add Dense Block Edges
            full_block = random.random() < 0.5 # Or other logic
            if full_block:
                edges_to_add = [(p, m) for p in providers_anom for m in members_anom]
            else:
                frac = random.uniform(0.2, 1.0)
                possible_edges = [(p, m) for p in providers_anom for m in members_anom]
                num_edges_to_select = max(1, int(frac * len(possible_edges)))
                edges_to_add = random.sample(possible_edges, num_edges_to_select)

            # Filter existing edges
            current_existing_edge_index = modified_data[target_edge_type].edge_index
            current_existing_set = set(tuple(map(int, x)) for x in current_existing_edge_index.t().tolist())

            new_edge_list = []
            new_edge_attr_list = []
            base_shape = edge_attr_mean.shape if isinstance(edge_attr_mean, torch.Tensor) else [1]
            attr_dtype = original_edge_attr.dtype if original_edge_attr is not None else torch.float

            for (p, m) in edges_to_add:
                if (p, m) in current_existing_set: continue

                new_edges_current_injection.append((p,m)) # Track added edges

                if injection_type == 'combined':
                    # Inject edge attribute anomaly ONLY for combined type
                    technique = random.choice(['outside', 'scaled'])
                    c_edge = random.uniform(edge_perturb_scale - 1.0, edge_perturb_scale + 1.0) * edge_attr_std
                    base_val = edge_attr_mean

                    if technique == 'outside':
                        new_val = base_val + c_edge * np.sign(np.random.randn())
                        anomaly_tracking['provider'][p].add('edge_feature_outside')
                        anomaly_tracking['member'][m].add('edge_feature_outside')
                    else: # 'scaled'
                        noise = torch.normal(mean=0.0, std=c_edge)
                        new_val = base_val + noise
                        anomaly_tracking['provider'][p].add('edge_feature_scaled')
                        anomaly_tracking['member'][m].add('edge_feature_scaled')

                    # Ensure correct shape/type
                    new_val_tensor = new_val.clone().detach().reshape(base_shape).to(attr_dtype) if isinstance(new_val, torch.Tensor) else torch.tensor([new_val], dtype=attr_dtype)
                    new_edge_attr_list.append(new_val_tensor)

                else: # Structural only - add edge with *normal* attributes
                     new_edge_attr_list.append(edge_attr_mean.clone().detach().reshape(base_shape).to(attr_dtype)) # Use mean value

                new_edge_list.append((p,m))

            # Append new edges to graph (if any)
            if new_edge_list:
                start_idx = modified_data[target_edge_type].edge_index.shape[1] # Index before adding
                new_edge_tensor = torch.tensor(new_edge_list, dtype=torch.long).t()
                new_edge_attr_tensor = torch.stack(new_edge_attr_list, dim=0)

                modified_data[target_edge_type].edge_index = torch.cat(
                    [modified_data[target_edge_type].edge_index, new_edge_tensor], dim=1
                )
                modified_data[target_edge_type].edge_attr = torch.cat(
                    [modified_data[target_edge_type].edge_attr, new_edge_attr_tensor], dim=0
                )
                # Add labels (1 for anomalous) for the new edges
                new_edge_labels = torch.ones(new_edge_tensor.shape[1], dtype=torch.long)
                gt_edge_labels[target_edge_type] = torch.cat(
                     [gt_edge_labels[target_edge_type], new_edge_labels]
                )
                newly_added_edges_info[target_edge_type].extend(list(range(start_idx, start_idx + len(new_edge_list))))


                # Handle reverse edges
                if reverse_edge_type in modified_data.edge_types:
                    start_idx_rev = modified_data[reverse_edge_type].edge_index.shape[1]
                    reverse_new_edges = torch.stack([new_edge_tensor[1], new_edge_tensor[0]], dim=0)
                    modified_data[reverse_edge_type].edge_index = torch.cat(
                        [modified_data[reverse_edge_type].edge_index, reverse_new_edges], dim=1
                    )
                    if hasattr(modified_data[reverse_edge_type], 'edge_attr'):
                         modified_data[reverse_edge_type].edge_attr = torch.cat(
                              [modified_data[reverse_edge_type].edge_attr, new_edge_attr_tensor], dim=0
                         )
                    # Add labels for reverse edges too
                    gt_edge_labels[reverse_edge_type] = torch.cat(
                        [gt_edge_labels[reverse_edge_type], new_edge_labels]
                    )
                    newly_added_edges_info[reverse_edge_type].extend(list(range(start_idx_rev, start_idx_rev + len(new_edge_list))))


        # == Attribute or Combined Anomalies ==
        if injection_type == 'attribute' or injection_type == 'combined':
            # Node feature perturbation
            for node_type, nodes in [('provider', providers_anom), ('member', members_anom)]:
                 if feature_std_dev.get(node_type) is None: continue
                 std_devs = feature_std_dev[node_type].to(modified_data[node_type].x.device)

                 for idx in nodes:
                     # Check if node already has feature anomaly from another injection if needed
                     x = modified_data[node_type].x[idx]
                     num_features = x.size(0)
                     num_to_perturb = max(1, int(0.3 * num_features)) # Perturb 30%
                     perturb_indices = np.random.choice(num_features, num_to_perturb, replace=False)

                     noise_scale = random.uniform(node_perturb_scale - 0.5, node_perturb_scale + 0.5)
                     # Ensure std_devs used here corresponds to the selected features
                     noise = torch.normal(mean=0.0, std=std_devs[perturb_indices] * noise_scale)
                     noise = noise.to(x.device)

                     x_perturbed = x.clone()
                     x_perturbed[perturb_indices] += noise
                     modified_data[node_type].x[idx] = x_perturbed
                     anomaly_tracking[node_type][idx].add('node_feature_relative')

    # --- Final Cleanup ---
    final_anomaly_tracking = {
         nt: dict(at) for nt, at in anomaly_tracking.items() # Convert defaultdict back
    }
    final_anomaly_tracking['provider'] = {i: types for i, types in final_anomaly_tracking['provider'].items() if gt_node_labels['provider'][i] == 1}
    final_anomaly_tracking['member'] = {i: types for i, types in final_anomaly_tracking['member'].items() if gt_node_labels['member'][i] == 1}


    # Add edge labels to the data object itself for convenience
    for edge_type, labels in gt_edge_labels.items():
        if edge_type in modified_data.edge_types:
             modified_data[edge_type].y = labels # Store edge labels in 'y' attribute

    print("Anomaly injection finished.")
    # Note: gt_edge_labels dict is also returned separately
    return modified_data, gt_node_labels, gt_edge_labels, final_anomaly_tracking
import pandas as pd
import torch
from torch_geometric.data import HeteroData
import numpy as np 

def inject_anomalies(data, node_type, percentage=0.05, scale=5.0):
    """Inject anomalies by scaling features of randomly selected nodes."""

    num_nodes = data[node_type].num_nodes
    num_anomalies = int(percentage * num_nodes)
    
    # Randomly select nodes to make anomalous
    anomaly_indices = np.random.choice(num_nodes, num_anomalies, replace=False)
    
    # Store original values for later evaluation
    original_x = data[node_type].x.clone()
    
    # Create a copy to avoid modifying the original
    modified_data = data.clone()
    
    # Modify features to create anomalies
    modified_data[node_type].x[anomaly_indices] *= scale
    
    # Create ground truth labels (1 for anomalies)
    gt_labels = torch.zeros(num_nodes, dtype=torch.long)
    gt_labels[anomaly_indices] = 1
    
    return modified_data, gt_labels, original_x

def inject_structural_anomalies(data, percentage=0.03, anomaly_types=None):
    """
    Inject structural anomalies into the graph.
    
    Types of structural anomalies:
    1. Dense subgraph (fraud ring) - providers and members forming unusual connections
    2. Isolated high-activity nodes - providers with unusually high claim volumes
    3. Bipartite cliques - perfect bipartite subgraphs where all members connect to all providers
    4. Hub nodes - providers connected to an unusually large number of members
    """
    # Create a copy to avoid modifying the original
    modified_data = data.clone()
    
    # Default anomaly types if not specified
    if anomaly_types is None:
        anomaly_types = ['fraud_ring', 'isolated_node', 'bipartite_clique', 'hub']
    
    # Track which nodes are anomalous
    num_members = modified_data['member'].num_nodes
    num_providers = modified_data['provider'].num_nodes
    member_anomaly_mask = torch.zeros(num_members, dtype=torch.bool)
    provider_anomaly_mask = torch.zeros(num_providers, dtype=torch.bool)
    
    # Get existing edges as a set for quick lookup
    edge_index = modified_data['provider', 'to', 'member'].edge_index
    existing_edges = set([(edge_index[0, i].item(), edge_index[1, i].item()) 
                          for i in range(edge_index.size(1))])
    
    new_edges = []
    new_edge_attr = []
    
    # 1. Create a small fraud ring (dense subgraph)
    if 'fraud_ring' in anomaly_types:
        ring_size_providers = max(2, int(num_providers * percentage * 0.5))
        ring_size_members = max(5, int(num_members * percentage * 0.5))
        
        # Select random providers and members for the fraud ring
        ring_providers = np.random.choice(num_providers, ring_size_providers, replace=False)
        ring_members = np.random.choice(num_members, ring_size_members, replace=False)
        
        # Mark them as anomalous
        provider_anomaly_mask[ring_providers] = True
        member_anomaly_mask[ring_members] = True
        
        # Create dense connections between the fraud ring members
        for p in ring_providers:
            for m in ring_members:
                if (p, m) not in existing_edges:
                    new_edges.append((p, m))
                    # Create unusually high number of claims
                    new_edge_attr.append([float(np.random.randint(20, 50))])
    
    # 2. Create isolated high-activity nodes
    if 'isolated_node' in anomaly_types:
        num_isolated = max(1, int(num_providers * percentage * 0.3))
        # Choose providers that aren't already in fraud ring
        available_providers = np.array([i for i in range(num_providers) 
                                       if i not in provider_anomaly_mask.nonzero().squeeze()])
        
        if len(available_providers) >= num_isolated:
            isolated_providers = np.random.choice(available_providers, num_isolated, replace=False)
            provider_anomaly_mask[isolated_providers] = True
            
            # For each isolated provider, connect to many random members with high claim counts
            for p in isolated_providers:
                target_members = np.random.choice(num_members, 
                                                 min(int(num_members * 0.1), 30), 
                                                 replace=False)
                for m in target_members:
                    if (p, m) not in existing_edges:
                        new_edges.append((p, m))
                        new_edge_attr.append([float(np.random.randint(15, 40))])
    
    # 3. Create bipartite cliques
    if 'bipartite_clique' in anomaly_types:
        clique_size_p = max(2, int(num_providers * percentage * 0.2))
        clique_size_m = max(2, int(num_members * percentage * 0.2))
        
        # Select nodes for the clique not already anomalous
        available_providers = np.array([i for i in range(num_providers) 
                                       if i not in provider_anomaly_mask.nonzero().squeeze()])
        available_members = np.array([i for i in range(num_members) 
                                    if i not in member_anomaly_mask.nonzero().squeeze()])
        
        if len(available_providers) >= clique_size_p and len(available_members) >= clique_size_m:
            clique_providers = np.random.choice(available_providers, clique_size_p, replace=False)
            clique_members = np.random.choice(available_members, clique_size_m, replace=False)
            
            provider_anomaly_mask[clique_providers] = True
            member_anomaly_mask[clique_members] = True
            
            # Create perfect bipartite subgraph
            for p in clique_providers:
                for m in clique_members:
                    if (p, m) not in existing_edges:
                        new_edges.append((p, m))
                        new_edge_attr.append([float(np.random.randint(5, 15))])
    
    # 4. Create hub nodes (providers with many connections)
    if 'hub' in anomaly_types:
        num_hubs = max(1, int(num_providers * percentage * 0.2))
        available_providers = np.array([i for i in range(num_providers) 
                                       if i not in provider_anomaly_mask.nonzero().squeeze()])
        
        if len(available_providers) >= num_hubs:
            hub_providers = np.random.choice(available_providers, num_hubs, replace=False)
            provider_anomaly_mask[hub_providers] = True
            
            # Connect each hub to many members
            for p in hub_providers:
                # Unusually high number of connections
                hub_connections = min(int(num_members * 0.15), 50)
                target_members = np.random.choice(num_members, hub_connections, replace=False)
                
                for m in target_members:
                    if (p, m) not in existing_edges:
                        new_edges.append((p, m))
                        new_edge_attr.append([float(np.random.randint(1, 10))])
    
    # Add the new edges to the graph
    if new_edges:
        new_edges_tensor = torch.tensor(new_edges, dtype=torch.long).t()
        new_edge_attr_tensor = torch.tensor(new_edge_attr, dtype=torch.float)
        
        # Concatenate with existing edges
        modified_data['provider', 'to', 'member'].edge_index = torch.cat(
            [modified_data['provider', 'to', 'member'].edge_index, new_edges_tensor], dim=1
        )
        modified_data['provider', 'to', 'member'].edge_attr = torch.cat(
            [modified_data['provider', 'to', 'member'].edge_attr, new_edge_attr_tensor], dim=0
        )
        
        # Also update reverse edges
        reverse_new_edges = torch.stack([new_edges_tensor[1], new_edges_tensor[0]], dim=0)
        modified_data['member', 'to', 'provider'].edge_index = torch.cat(
            [modified_data['member', 'to', 'provider'].edge_index, reverse_new_edges], dim=1
        )
        modified_data['member', 'to', 'provider'].edge_attr = torch.cat(
            [modified_data['member', 'to', 'provider'].edge_attr, new_edge_attr_tensor], dim=0
        )
    
    # Create ground truth labels (1 for anomalies)
    member_gt_labels = torch.zeros(num_members, dtype=torch.long)
    member_gt_labels[member_anomaly_mask] = 1
    
    provider_gt_labels = torch.zeros(num_providers, dtype=torch.long)
    provider_gt_labels[provider_anomaly_mask] = 1
    
    return modified_data, {'member': member_gt_labels, 'provider': provider_gt_labels}

def inject_feature_anomalies(data, percentage=0.05, anomaly_types=None):
    """
    Inject feature-based anomalies:
    
    Types of feature anomalies:
    1. Outlier values: extremely high values for certain features
    2. Inconsistent correlations: modify related features in inconsistent ways
    3. Pattern breaking: change multiple features to break normal patterns
    4. Specialty-specific changes: modify features based on provider specialties
    """
    # Create a copy to avoid modifying the original
    modified_data = data.clone()
    
    if anomaly_types is None:
        anomaly_types = ['outliers', 'correlations', 'patterns', 'specialty']
    
    num_members = modified_data['member'].num_nodes
    num_providers = modified_data['provider'].num_nodes
    
    member_anomaly_indices = set()
    provider_anomaly_indices = set()
    
    # 1. Outlier values
    if 'outliers' in anomaly_types:
        # For providers
        num_provider_outliers = max(1, int(num_providers * percentage * 0.4))
        outlier_indices = np.random.choice(num_providers, num_provider_outliers, replace=False)
        provider_anomaly_indices.update(outlier_indices.tolist())
        
        for idx in outlier_indices:
            # Get random feature indices (columns)
            feature_dims = modified_data['provider'].x.size(1)
            anomaly_features = np.random.choice(
                feature_dims, size=max(1, int(feature_dims * 0.3)), replace=False
            )
            
            # Scale up these features significantly
            scale_factor = torch.tensor(
                [np.random.uniform(5, 10) if i in anomaly_features else 1.0 
                 for i in range(feature_dims)], 
                dtype=torch.float
            )
            modified_data['provider'].x[idx] *= scale_factor
        
        # For members
        num_member_outliers = max(1, int(num_members * percentage * 0.4))
        outlier_indices = np.random.choice(num_members, num_member_outliers, replace=False)
        member_anomaly_indices.update(outlier_indices.tolist())
        
        for idx in outlier_indices:
            # Get random feature indices
            feature_dims = modified_data['member'].x.size(1)
            anomaly_features = np.random.choice(
                feature_dims, size=max(1, int(feature_dims * 0.3)), replace=False
            )
            
            # Scale up these features
            scale_factor = torch.tensor(
                [np.random.uniform(5, 10) if i in anomaly_features else 1.0 
                 for i in range(feature_dims)], 
                dtype=torch.float
            )
            modified_data['member'].x[idx] *= scale_factor
    
    # 2. Inconsistent correlations (assuming certain features are correlated)
    if 'correlations' in anomaly_types and modified_data['provider'].x.size(1) > 3:
        num_anomalies = max(1, int(num_providers * percentage * 0.3))
        correlation_indices = np.random.choice(
            [i for i in range(num_providers) if i not in provider_anomaly_indices],
            num_anomalies, 
            replace=False
        )
        provider_anomaly_indices.update(correlation_indices.tolist())
        
        for idx in correlation_indices:
            # Assuming first few features might be correlated (like different types of claims)
            # Increase one feature while decreasing another
            feature1, feature2 = np.random.choice(range(3), 2, replace=False)
            
            # Store original values to preserve scale
            orig_val1 = modified_data['provider'].x[idx, feature1].item()
            orig_val2 = modified_data['provider'].x[idx, feature2].item()
            
            # Create inconsistency
            modified_data['provider'].x[idx, feature1] = orig_val1 * np.random.uniform(3, 5)
            modified_data['provider'].x[idx, feature2] = orig_val2 * np.random.uniform(0.1, 0.5)
    
    # 3. Pattern breaking (changing multiple features in unusual ways)
    if 'patterns' in anomaly_types:
        num_anomalies = max(1, int(num_providers * percentage * 0.2))
        pattern_indices = np.random.choice(
            [i for i in range(num_providers) if i not in provider_anomaly_indices],
            num_anomalies, 
            replace=False
        )
        provider_anomaly_indices.update(pattern_indices.tolist())
        
        for idx in pattern_indices:
            feature_dims = modified_data['provider'].x.size(1)
            # Modify a large portion of features
            num_features_to_modify = max(3, int(feature_dims * 0.7))
            features_to_modify = np.random.choice(feature_dims, num_features_to_modify, replace=False)
            
            # Apply a combination of operations to break patterns
            for feat in features_to_modify:
                op_type = np.random.choice(['multiply', 'invert', 'zero', 'extreme'])
                if op_type == 'multiply':
                    modified_data['provider'].x[idx, feat] *= np.random.uniform(2, 4)
                elif op_type == 'invert':
                    # Invert the value within reasonable bounds
                    mean_val = modified_data['provider'].x[:, feat].mean().item()
                    if modified_data['provider'].x[idx, feat].item() > mean_val:
                        modified_data['provider'].x[idx, feat] = mean_val * np.random.uniform(0.2, 0.5)
                    else:
                        modified_data['provider'].x[idx, feat] = mean_val * np.random.uniform(1.5, 2.5)
                elif op_type == 'zero':
                    modified_data['provider'].x[idx, feat] = 0.0
                elif op_type == 'extreme':
                    modified_data['provider'].x[idx, feat] = modified_data['provider'].x[:, feat].max() * np.random.uniform(1.5, 3.0)
    
    # Create ground truth labels
    member_gt_labels = torch.zeros(num_members, dtype=torch.long)
    member_gt_labels[list(member_anomaly_indices)] = 1
    
    provider_gt_labels = torch.zeros(num_providers, dtype=torch.long)
    provider_gt_labels[list(provider_anomaly_indices)] = 1
    
    return modified_data, {'member': member_gt_labels, 'provider': provider_gt_labels}

def inject_mixed_anomalies(data, percentages=None, methods=None):
    """
    Inject a mix of different anomaly types using multiple methods.
    
    Args:
        data: The original HeteroData object
        percentages: Dict specifying percentage for each method
        methods: List of methods to use
    
    Returns:
        Modified data and ground truth labels
    """
    if percentages is None:
        percentages = {
            'structural': 0.03,
            'feature': 0.04,
            'contextual': 0.02
        }
    
    if methods is None:
        methods = ['structural', 'feature']
        
    # Make a copy of the original data
    modified_data = data.clone()
    
    # Initialize ground truth label dictionaries
    member_gt_labels = torch.zeros(modified_data['member'].num_nodes, dtype=torch.long)
    provider_gt_labels = torch.zeros(modified_data['provider'].num_nodes, dtype=torch.long)
    gt_labels = {'member': member_gt_labels, 'provider': provider_gt_labels}
    
    # Apply each method
    if 'structural' in methods:
        modified_data, structural_labels = inject_structural_anomalies(
            modified_data, percentage=percentages.get('structural', 0.03)
        )
        
        # Merge labels
        gt_labels['member'] = torch.max(gt_labels['member'], structural_labels['member'])
        gt_labels['provider'] = torch.max(gt_labels['provider'], structural_labels['provider'])
    
    if 'feature' in methods:
        modified_data, feature_labels = inject_feature_anomalies(
            modified_data, percentage=percentages.get('feature', 0.04)
        )
        
        # Merge labels
        gt_labels['member'] = torch.max(gt_labels['member'], feature_labels['member'])
        gt_labels['provider'] = torch.max(gt_labels['provider'], feature_labels['provider'])
    
    # Return the modified data and combined labels
    return modified_data, gt_labels

def inject_healthcare_fraud_patterns(data, df_provider_features, df_member_features, percentage=0.04):
    """
    Inject realistic healthcare fraud patterns based on domain knowledge
    
    Types of patterns:
    1. Upcoding - providers charging for more expensive procedures
    2. Phantom billing - billing for services not rendered
    3. Unbundling - charging separately for bundled services
    4. Patient-provider collusion - suspicious patterns in member-provider connections
    5. Specialty mismatch - providers billing outside their specialty
    """
    modified_data = data.clone()
    
    num_members = modified_data['member'].num_nodes
    num_providers = modified_data['provider'].num_nodes
    
    member_anomaly_indices = set()
    provider_anomaly_indices = set()
    
    # 1. Upcoding - modify specific features related to claim values
    num_upcoding = max(1, int(num_providers * percentage * 0.3))
    upcoding_indices = np.random.choice(num_providers, num_upcoding, replace=False)
    provider_anomaly_indices.update(upcoding_indices.tolist())
    
    # Assuming certain columns represent claim amounts or procedure codes
    # You'll need to adapt this based on your actual feature meanings
    for idx in upcoding_indices:
        # Find claim amount features (adapt to your dataset)
        claim_feature_indices = []
        for i, col in enumerate(df_provider_features.columns):
            if any(term in col.lower() for term in ['charge', 'amount', 'payment', 'cost', 'fee']):
                claim_feature_indices.append(i)
                
        if claim_feature_indices:
            # Increase these features substantially
            for feat_idx in claim_feature_indices:
                modified_data['provider'].x[idx, feat_idx] *= np.random.uniform(2.0, 4.0)
    
    # 2. Phantom billing (providers with many claims but low actual services)
    num_phantom = max(1, int(num_providers * percentage * 0.2))
    phantom_indices = np.random.choice(
        [i for i in range(num_providers) if i not in provider_anomaly_indices],
        num_phantom,
        replace=False
    )
    provider_anomaly_indices.update(phantom_indices.tolist())
    
    for idx in phantom_indices:
        # Increase claim counts but decrease service-related metrics
        service_feature_indices = []
        count_feature_indices = []
        
        for i, col in enumerate(df_provider_features.columns):
            if any(term in col.lower() for term in ['count', 'claims', 'quantity']):
                count_feature_indices.append(i)
            elif any(term in col.lower() for term in ['service', 'visit', 'procedure', 'treatment']):
                service_feature_indices.append(i)
        
        # Modify features
        for feat_idx in count_feature_indices:
            modified_data['provider'].x[idx, feat_idx] *= np.random.uniform(1.5, 3.0)
        
        for feat_idx in service_feature_indices:
            if modified_data['provider'].x[idx, feat_idx] > 0:
                modified_data['provider'].x[idx, feat_idx] *= np.random.uniform(0.3, 0.7)
    
    # 3. Patient-provider collusion - modify both provider and member
    num_collusion_pairs = max(1, int(min(num_providers, num_members) * percentage * 0.2))
    
    # Get existing edges to find connected pairs
    edge_index = modified_data['provider', 'to', 'member'].edge_index
    edge_pairs = [(edge_index[0, i].item(), edge_index[1, i].item()) for i in range(edge_index.size(1))]
    
    # Select random existing connections
    if edge_pairs and len(edge_pairs) >= num_collusion_pairs:
        selected_pairs = np.random.choice(len(edge_pairs), num_collusion_pairs, replace=False)
        
        for idx in selected_pairs:
            provider_idx, member_idx = edge_pairs[idx]
            
            # Mark both as anomalous
            provider_anomaly_indices.add(provider_idx)
            member_anomaly_indices.add(member_idx)
            
            # Modify both provider and member features to show suspicious patterns
            # For example, increase claim-related features for both
            for i in range(modified_data['provider'].x.size(1)):
                if np.random.random() < 0.4:  # Modify some features
                    modified_data['provider'].x[provider_idx, i] *= np.random.uniform(1.5, 2.5)
                    
            for i in range(modified_data['member'].x.size(1)):
                if np.random.random() < 0.4:
                    modified_data['member'].x[member_idx, i] *= np.random.uniform(1.5, 2.5)
            
            # Also modify the edge attributes connecting them (increase claim count)
            for i in range(edge_index.size(1)):
                if edge_index[0, i].item() == provider_idx and edge_index[1, i].item() == member_idx:
                    modified_data['provider', 'to', 'member'].edge_attr[i] *= np.random.uniform(2.0, 5.0)
                    # Also update the reverse edge
                    for j in range(modified_data['member', 'to', 'provider'].edge_index.size(1)):
                        if (modified_data['member', 'to', 'provider'].edge_index[0, j].item() == member_idx and
                            modified_data['member', 'to', 'provider'].edge_index[1, j].item() == provider_idx):
                            modified_data['member', 'to', 'provider'].edge_attr[j] = modified_data['provider', 'to', 'member'].edge_attr[i]
                            break
    
    # Create ground truth labels
    member_gt_labels = torch.zeros(num_members, dtype=torch.long)
    member_gt_labels[list(member_anomaly_indices)] = 1
    
    provider_gt_labels = torch.zeros(num_providers, dtype=torch.long)
    provider_gt_labels[list(provider_anomaly_indices)] = 1
    
    return modified_data, {'member': member_gt_labels, 'provider': provider_gt_labels}


def inject_mixed_anomalies_with_tracking(data, percentages=None, methods=None):
    """
    Inject a mix of different anomaly types and track which node has which type of anomaly.
    
    Returns:
        - Modified data
        - Ground truth labels
        - Dictionary mapping each anomalous node to its anomaly type(s)
    """
    if percentages is None:
        percentages = {
            'structural': 0.03,
            'feature': 0.04,
            'healthcare': 0.03
        }
    
    if methods is None:
        methods = ['structural', 'feature', 'healthcare']
        
    # Make a copy of the original data
    modified_data = data.clone()
    
    # Initialize ground truth label dictionaries
    member_gt_labels = torch.zeros(modified_data['member'].num_nodes, dtype=torch.long)
    provider_gt_labels = torch.zeros(modified_data['provider'].num_nodes, dtype=torch.long)
    gt_labels = {'member': member_gt_labels, 'provider': provider_gt_labels}
    
    # Initialize tracking dictionaries - maps node index to set of anomaly types
    anomaly_tracking = {
        'member': {idx: set() for idx in range(modified_data['member'].num_nodes)},
        'provider': {idx: set() for idx in range(modified_data['provider'].num_nodes)}
    }
    
    # For structural anomalies, also track subtypes
    structural_subtypes = {
        'member': {},
        'provider': {}
    }
    
    # Apply each method
    if 'structural' in methods:
        modified_data, structural_labels, structural_types = inject_structural_anomalies_with_tracking(
            modified_data, percentage=percentages.get('structural', 0.03)
        )
        
        # Merge labels
        gt_labels['member'] = torch.max(gt_labels['member'], structural_labels['member'])
        gt_labels['provider'] = torch.max(gt_labels['provider'], structural_labels['provider'])
        
        # Track anomaly types
        for node_type in ['member', 'provider']:
            for idx in range(len(structural_labels[node_type])):
                if structural_labels[node_type][idx] == 1:
                    anomaly_tracking[node_type][idx].add('structural')
                    
                    # Also store the structural subtype if available
                    if idx in structural_types[node_type]:
                        structural_subtypes[node_type][idx] = structural_types[node_type][idx]
    
    if 'feature' in methods:
        modified_data, feature_labels, feature_subtypes = inject_feature_anomalies_with_tracking(
            modified_data, percentage=percentages.get('feature', 0.04)
        )
        
        # Merge labels
        gt_labels['member'] = torch.max(gt_labels['member'], feature_labels['member'])
        gt_labels['provider'] = torch.max(gt_labels['provider'], feature_labels['provider'])
        
        # Track anomaly types
        for node_type in ['member', 'provider']:
            for idx in range(len(feature_labels[node_type])):
                if feature_labels[node_type][idx] == 1:
                    anomaly_tracking[node_type][idx].add('feature')
                    # Store feature subtype
                    if idx in feature_subtypes[node_type]:
                        anomaly_tracking[node_type][idx].add(f"feature_{feature_subtypes[node_type][idx]}")
    
    if 'healthcare' in methods:
        modified_data, healthcare_labels, healthcare_subtypes = inject_healthcare_fraud_patterns_with_tracking(
            modified_data, df_provider_features, df_member_features, 
            percentage=percentages.get('healthcare', 0.03)
        )
        
        # Merge labels
        gt_labels['member'] = torch.max(gt_labels['member'], healthcare_labels['member'])
        gt_labels['provider'] = torch.max(gt_labels['provider'], healthcare_labels['provider'])
        
        # Track anomaly types
        for node_type in ['member', 'provider']:
            for idx in range(len(healthcare_labels[node_type])):
                if healthcare_labels[node_type][idx] == 1:
                    anomaly_tracking[node_type][idx].add('healthcare')
                    # Store healthcare fraud subtype
                    if idx in healthcare_subtypes[node_type]:
                        anomaly_tracking[node_type][idx].add(f"healthcare_{healthcare_subtypes[node_type][idx]}")
    
    # Create final tracking dictionary with only anomalous nodes
    final_tracking = {
        'member': {idx: types for idx, types in anomaly_tracking['member'].items() 
                 if gt_labels['member'][idx] == 1},
        'provider': {idx: types for idx, types in anomaly_tracking['provider'].items() 
                   if gt_labels['provider'][idx] == 1}
    }
    
    # Add structural subtypes to final tracking
    for node_type in ['member', 'provider']:
        for idx, subtype in structural_subtypes[node_type].items():
            if idx in final_tracking[node_type]:
                final_tracking[node_type][idx].add(f"structural_{subtype}")
    
    # Return the modified data, combined labels, and anomaly tracking
    return modified_data, gt_labels, final_tracking

def inject_feature_anomalies_with_tracking(data, percentage=0.05, anomaly_types=None):
    """
    Inject feature-based anomalies with tracking of anomaly subtypes.
    
    Types of feature anomalies:
    1. Outlier values: extremely high values for certain features
    2. Inconsistent correlations: modify related features in inconsistent ways
    3. Pattern breaking: change multiple features to break normal patterns
    4. Specialty-specific changes: modify features based on provider specialties
    
    Returns:
        - Modified data
        - Ground truth labels
        - Dictionary tracking which nodes have which anomaly subtypes
    """
    # Create a copy to avoid modifying the original
    modified_data = data.clone()
    
    if anomaly_types is None:
        anomaly_types = ['outliers', 'correlations', 'patterns']
    
    num_members = modified_data['member'].num_nodes
    num_providers = modified_data['provider'].num_nodes
    
    member_anomaly_indices = set()
    provider_anomaly_indices = set()
    
    # Track anomaly subtypes
    member_anomaly_types = {}  # Maps node idx -> anomaly subtype
    provider_anomaly_types = {}
    
    # 1. Outlier values
    if 'outliers' in anomaly_types:
        # For providers
        num_provider_outliers = max(1, int(num_providers * percentage * 0.4))
        outlier_indices = np.random.choice(num_providers, num_provider_outliers, replace=False)
        provider_anomaly_indices.update(outlier_indices.tolist())
        
        for idx in outlier_indices:
            # Get random feature indices (columns)
            feature_dims = modified_data['provider'].x.size(1)
            anomaly_features = np.random.choice(
                feature_dims, size=max(1, int(feature_dims * 0.3)), replace=False
            )
            
            # Scale up these features significantly
            scale_factor = torch.tensor(
                [np.random.uniform(5, 10) if i in anomaly_features else 1.0 
                 for i in range(feature_dims)], 
                dtype=torch.float
            )
            modified_data['provider'].x[idx] *= scale_factor
            
            # Track anomaly subtype
            provider_anomaly_types[idx] = 'outliers'
        
        # For members
        num_member_outliers = max(1, int(num_members * percentage * 0.4))
        outlier_indices = np.random.choice(num_members, num_member_outliers, replace=False)
        member_anomaly_indices.update(outlier_indices.tolist())
        
        for idx in outlier_indices:
            # Get random feature indices
            feature_dims = modified_data['member'].x.size(1)
            anomaly_features = np.random.choice(
                feature_dims, size=max(1, int(feature_dims * 0.3)), replace=False
            )
            
            # Scale up these features
            scale_factor = torch.tensor(
                [np.random.uniform(5, 10) if i in anomaly_features else 1.0 
                 for i in range(feature_dims)], 
                dtype=torch.float
            )
            modified_data['member'].x[idx] *= scale_factor
            
            # Track anomaly subtype
            member_anomaly_types[idx] = 'outliers'
    
    # 2. Inconsistent correlations (assuming certain features are correlated)
    if 'correlations' in anomaly_types and modified_data['provider'].x.size(1) > 3:
        num_anomalies = max(1, int(num_providers * percentage * 0.3))
        correlation_indices = np.random.choice(
            [i for i in range(num_providers) if i not in provider_anomaly_indices],
            num_anomalies, 
            replace=False
        )
        provider_anomaly_indices.update(correlation_indices.tolist())
        
        for idx in correlation_indices:
            # Assuming first few features might be correlated (like different types of claims)
            # Increase one feature while decreasing another
            feature1, feature2 = np.random.choice(range(3), 2, replace=False)
            
            # Store original values to preserve scale
            orig_val1 = modified_data['provider'].x[idx, feature1].item()
            orig_val2 = modified_data['provider'].x[idx, feature2].item()
            
            # Create inconsistency
            modified_data['provider'].x[idx, feature1] = orig_val1 * np.random.uniform(3, 5)
            modified_data['provider'].x[idx, feature2] = orig_val2 * np.random.uniform(0.1, 0.5)
            
            # Track anomaly subtype
            provider_anomaly_types[idx] = 'correlations'
    
    # 3. Pattern breaking (changing multiple features in unusual ways)
    if 'patterns' in anomaly_types:
        num_anomalies = max(1, int(num_providers * percentage * 0.2))
        pattern_indices = np.random.choice(
            [i for i in range(num_providers) if i not in provider_anomaly_indices],
            num_anomalies, 
            replace=False
        )
        provider_anomaly_indices.update(pattern_indices.tolist())
        
        for idx in pattern_indices:
            feature_dims = modified_data['provider'].x.size(1)
            # Modify a large portion of features
            num_features_to_modify = max(3, int(feature_dims * 0.7))
            features_to_modify = np.random.choice(feature_dims, num_features_to_modify, replace=False)
            
            # Apply a combination of operations to break patterns
            for feat in features_to_modify:
                op_type = np.random.choice(['multiply', 'invert', 'zero', 'extreme'])
                if op_type == 'multiply':
                    modified_data['provider'].x[idx, feat] *= np.random.uniform(2, 4)
                elif op_type == 'invert':
                    # Invert the value within reasonable bounds
                    mean_val = modified_data['provider'].x[:, feat].mean().item()
                    if modified_data['provider'].x[idx, feat].item() > mean_val:
                        modified_data['provider'].x[idx, feat] = mean_val * np.random.uniform(0.2, 0.5)
                    else:
                        modified_data['provider'].x[idx, feat] = mean_val * np.random.uniform(1.5, 2.5)
                elif op_type == 'zero':
                    modified_data['provider'].x[idx, feat] = 0.0
                elif op_type == 'extreme':
                    modified_data['provider'].x[idx, feat] = modified_data['provider'].x[:, feat].max() * np.random.uniform(1.5, 3.0)
            
            # Track anomaly subtype
            provider_anomaly_types[idx] = 'patterns'
    
    # Create ground truth labels
    member_gt_labels = torch.zeros(num_members, dtype=torch.long)
    member_gt_labels[list(member_anomaly_indices)] = 1
    
    provider_gt_labels = torch.zeros(num_providers, dtype=torch.long)
    provider_gt_labels[list(provider_anomaly_indices)] = 1
    
    # Prepare anomaly tracking
    anomaly_types_tracking = {
        'member': member_anomaly_types,
        'provider': provider_anomaly_types
    }
    
    return modified_data, {'member': member_gt_labels, 'provider': provider_gt_labels}, anomaly_types_tracking


def inject_structural_anomalies_with_tracking(data, percentage=0.03, anomaly_types=None):
    """
    Inject structural anomalies and track which nodes belong to which anomaly subtype.
    """
    # Create a copy to avoid modifying the original
    modified_data = data.clone()
    
    # Default anomaly types if not specified
    if anomaly_types is None:
        anomaly_types = ['fraud_ring', 'isolated_node', 'bipartite_clique', 'hub']
    
    # Track which nodes are anomalous
    num_members = modified_data['member'].num_nodes
    num_providers = modified_data['provider'].num_nodes
    member_anomaly_mask = torch.zeros(num_members, dtype=torch.bool)
    provider_anomaly_mask = torch.zeros(num_providers, dtype=torch.bool)
    
    # Track anomaly types for each node
    member_anomaly_types = {}  # Maps node idx -> anomaly subtype
    provider_anomaly_types = {}
    
    # Get existing edges as a set for quick lookup
    edge_index = modified_data['provider', 'to', 'member'].edge_index
    existing_edges = set([(edge_index[0, i].item(), edge_index[1, i].item()) 
                          for i in range(edge_index.size(1))])
    
    new_edges = []
    new_edge_attr = []
    
    # 1. Create a small fraud ring (dense subgraph)
    if 'fraud_ring' in anomaly_types:
        ring_size_providers = max(2, int(num_providers * percentage * 0.5))
        ring_size_members = max(5, int(num_members * percentage * 0.5))
        
        # Select random providers and members for the fraud ring
        ring_providers = np.random.choice(num_providers, ring_size_providers, replace=False)
        ring_members = np.random.choice(num_members, ring_size_members, replace=False)
        
        # Mark them as anomalous
        provider_anomaly_mask[ring_providers] = True
        member_anomaly_mask[ring_members] = True
        
        # Track anomaly type
        for p in ring_providers:
            provider_anomaly_types[p] = 'fraud_ring'
        for m in ring_members:
            member_anomaly_types[m] = 'fraud_ring'
        
        # Create dense connections between the fraud ring members
        for p in ring_providers:
            for m in ring_members:
                if (p, m) not in existing_edges:
                    new_edges.append((p, m))
                    # Create unusually high number of claims
                    new_edge_attr.append([float(np.random.randint(20, 50))])
    
    # 2. Create isolated high-activity nodes
    if 'isolated_node' in anomaly_types:
        num_isolated = max(1, int(num_providers * percentage * 0.3))
        # Choose providers that aren't already in fraud ring
        available_providers = np.array([i for i in range(num_providers) 
                                       if i not in provider_anomaly_mask.nonzero().squeeze()])
        
        if len(available_providers) >= num_isolated:
            isolated_providers = np.random.choice(available_providers, num_isolated, replace=False)
            provider_anomaly_mask[isolated_providers] = True
            
            # Track anomaly type
            for p in isolated_providers:
                provider_anomaly_types[p] = 'isolated_node'
            
            # For each isolated provider, connect to many random members with high claim counts
            for p in isolated_providers:
                target_members = np.random.choice(num_members, 
                                                 min(int(num_members * 0.1), 30), 
                                                 replace=False)
                for m in target_members:
                    if (p, m) not in existing_edges:
                        new_edges.append((p, m))
                        new_edge_attr.append([float(np.random.randint(15, 40))])
    
    # Complete the other structural anomaly types with tracking...
    # [code for bipartite_clique and hub with tracking similar to above]
    
    # Add the new edges to the graph
    if new_edges:
        new_edges_tensor = torch.tensor(new_edges, dtype=torch.long).t()
        new_edge_attr_tensor = torch.tensor(new_edge_attr, dtype=torch.float)
        
        # Concatenate with existing edges
        modified_data['provider', 'to', 'member'].edge_index = torch.cat(
            [modified_data['provider', 'to', 'member'].edge_index, new_edges_tensor], dim=1
        )
        modified_data['provider', 'to', 'member'].edge_attr = torch.cat(
            [modified_data['provider', 'to', 'member'].edge_attr, new_edge_attr_tensor], dim=0
        )
        
        # Also update reverse edges
        reverse_new_edges = torch.stack([new_edges_tensor[1], new_edges_tensor[0]], dim=0)
        modified_data['member', 'to', 'provider'].edge_index = torch.cat(
            [modified_data['member', 'to', 'provider'].edge_index, reverse_new_edges], dim=1
        )
        modified_data['member', 'to', 'provider'].edge_attr = torch.cat(
            [modified_data['member', 'to', 'provider'].edge_attr, new_edge_attr_tensor], dim=0
        )
    
    # Create ground truth labels (1 for anomalies)
    member_gt_labels = torch.zeros(num_members, dtype=torch.long)
    member_gt_labels[member_anomaly_mask] = 1
    
    provider_gt_labels = torch.zeros(num_providers, dtype=torch.long)
    provider_gt_labels[provider_anomaly_mask] = 1
    
    # Prepare anomaly tracking
    anomaly_types = {
        'member': member_anomaly_types,
        'provider': provider_anomaly_types
    }
    
    return modified_data, {'member': member_gt_labels, 'provider': provider_gt_labels}, anomaly_types

def inject_healthcare_fraud_patterns_with_tracking(data, df_provider_features, df_member_features, percentage=0.04):
    """
    Inject realistic healthcare fraud patterns with tracking of anomaly subtypes.
    
    Types of patterns:
    1. Upcoding - providers charging for more expensive procedures
    2. Phantom billing - billing for services not rendered
    3. Unbundling - charging separately for bundled services
    4. Patient-provider collusion - suspicious patterns in member-provider connections
    5. Specialty mismatch - providers billing outside their specialty
    
    Returns:
        - Modified data
        - Ground truth labels
        - Dictionary tracking which nodes have which anomaly subtypes
    """
    modified_data = data.clone()
    
    num_members = modified_data['member'].num_nodes
    num_providers = modified_data['provider'].num_nodes
    
    member_anomaly_indices = set()
    provider_anomaly_indices = set()
    
    # Track anomaly subtypes
    member_anomaly_types = {}  # Maps node idx -> anomaly subtype
    provider_anomaly_types = {}
    
    # 1. Upcoding - modify specific features related to claim values
    num_upcoding = max(1, int(num_providers * percentage * 0.3))
    upcoding_indices = np.random.choice(num_providers, num_upcoding, replace=False)
    provider_anomaly_indices.update(upcoding_indices.tolist())
    
    # Assuming certain columns represent claim amounts or procedure codes
    # You'll need to adapt this based on your actual feature meanings
    for idx in upcoding_indices:
        # Find claim amount features (adapt to your dataset)
        claim_feature_indices = []
        for i, col in enumerate(df_provider_features.columns):
            if any(term in col.lower() for term in ['charge', 'amount', 'payment', 'cost', 'fee']):
                claim_feature_indices.append(i)
                
        if claim_feature_indices:
            # Increase these features substantially
            for feat_idx in claim_feature_indices:
                modified_data['provider'].x[idx, feat_idx] *= np.random.uniform(2.0, 4.0)
        else:
            # If we can't identify specific columns, just modify random features
            feature_dims = modified_data['provider'].x.size(1)
            random_features = np.random.choice(feature_dims, size=max(1, int(feature_dims * 0.2)), replace=False)
            for feat_idx in random_features:
                modified_data['provider'].x[idx, feat_idx] *= np.random.uniform(2.0, 4.0)
        
        # Track anomaly subtype
        provider_anomaly_types[idx] = 'upcoding'
    
    # 2. Phantom billing (providers with many claims but low actual services)
    num_phantom = max(1, int(num_providers * percentage * 0.2))
    phantom_indices = np.random.choice(
        [i for i in range(num_providers) if i not in provider_anomaly_indices],
        num_phantom,
        replace=False
    )
    provider_anomaly_indices.update(phantom_indices.tolist())
    
    for idx in phantom_indices:
        # Increase claim counts but decrease service-related metrics
        service_feature_indices = []
        count_feature_indices = []
        
        for i, col in enumerate(df_provider_features.columns):
            if any(term in col.lower() for term in ['count', 'claims', 'quantity']):
                count_feature_indices.append(i)
            elif any(term in col.lower() for term in ['service', 'visit', 'procedure', 'treatment']):
                service_feature_indices.append(i)
        
        # Modify features
        if count_feature_indices:
            for feat_idx in count_feature_indices:
                modified_data['provider'].x[idx, feat_idx] *= np.random.uniform(1.5, 3.0)
        
        if service_feature_indices:
            for feat_idx in service_feature_indices:
                if modified_data['provider'].x[idx, feat_idx] > 0:
                    modified_data['provider'].x[idx, feat_idx] *= np.random.uniform(0.3, 0.7)
        
        if not count_feature_indices and not service_feature_indices:
            # If we can't identify specific columns, create generic phantom billing pattern
            feature_dims = modified_data['provider'].x.size(1)
            # Increase first half of features (representing claims/billings)
            for i in range(min(3, feature_dims // 2)):
                modified_data['provider'].x[idx, i] *= np.random.uniform(1.5, 3.0)
            # Decrease second half of features (representing actual services)
            for i in range(feature_dims // 2, min(feature_dims, feature_dims // 2 + 3)):
                if modified_data['provider'].x[idx, i] > 0:
                    modified_data['provider'].x[idx, i] *= np.random.uniform(0.3, 0.7)
        
        # Track anomaly subtype
        provider_anomaly_types[idx] = 'phantom_billing'
    
    # 3. Patient-provider collusion - modify both provider and member
    num_collusion_pairs = max(1, int(min(num_providers, num_members) * percentage * 0.2))
    
    # Get existing edges to find connected pairs
    edge_index = modified_data['provider', 'to', 'member'].edge_index
    edge_pairs = [(edge_index[0, i].item(), edge_index[1, i].item()) for i in range(edge_index.size(1))]
    
    # Select random existing connections
    if edge_pairs and len(edge_pairs) >= num_collusion_pairs:
        selected_pairs = np.random.choice(len(edge_pairs), num_collusion_pairs, replace=False)
        
        for idx in selected_pairs:
            provider_idx, member_idx = edge_pairs[idx]
            
            # Mark both as anomalous
            provider_anomaly_indices.add(provider_idx)
            member_anomaly_indices.add(member_idx)
            
            # Modify both provider and member features to show suspicious patterns
            # For example, increase claim-related features for both
            for i in range(modified_data['provider'].x.size(1)):
                if np.random.random() < 0.4:  # Modify some features
                    modified_data['provider'].x[provider_idx, i] *= np.random.uniform(1.5, 2.5)
                    
            for i in range(modified_data['member'].x.size(1)):
                if np.random.random() < 0.4:
                    modified_data['member'].x[member_idx, i] *= np.random.uniform(1.5, 2.5)
            
            # Track anomaly subtype
            provider_anomaly_types[provider_idx] = 'collusion'
            member_anomaly_types[member_idx] = 'collusion'
            
            # Also modify the edge attributes connecting them (increase claim count)
            for i in range(edge_index.size(1)):
                if edge_index[0, i].item() == provider_idx and edge_index[1, i].item() == member_idx:
                    modified_data['provider', 'to', 'member'].edge_attr[i] *= np.random.uniform(2.0, 5.0)
                    # Also update the reverse edge
                    for j in range(modified_data['member', 'to', 'provider'].edge_index.size(1)):
                        if (modified_data['member', 'to', 'provider'].edge_index[0, j].item() == member_idx and
                            modified_data['member', 'to', 'provider'].edge_index[1, j].item() == provider_idx):
                            modified_data['member', 'to', 'provider'].edge_attr[j] = modified_data['provider', 'to', 'member'].edge_attr[i]
                            break
    
    # Create ground truth labels
    member_gt_labels = torch.zeros(num_members, dtype=torch.long)
    member_gt_labels[list(member_anomaly_indices)] = 1
    
    provider_gt_labels = torch.zeros(num_providers, dtype=torch.long)
    provider_gt_labels[list(provider_anomaly_indices)] = 1
    
    # Prepare anomaly tracking
    anomaly_types_tracking = {
        'member': member_anomaly_types,
        'provider': provider_anomaly_types
    }
    
    return modified_data, {'member': member_gt_labels, 'provider': provider_gt_labels}, anomaly_types_tracking

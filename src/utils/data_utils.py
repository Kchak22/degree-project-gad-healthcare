# Split into training/validation/test using RandomNodeSplit
import torch
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData
from typing import Tuple, Dict

def split_graph_nodes_inductive(
    data: HeteroData,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    add_dummy_key: bool = True
    ) -> Tuple[HeteroData, HeteroData, HeteroData]:
    """
    Splits a HeteroData object into train, validation, and test subgraphs
    based on node-level splits.

    Note: This creates separate graph objects. Edges between nodes assigned
    to different splits in the original graph will be lost in the subgraphs.
    This is an approximation for inductive evaluation.

    Args:
        data (HeteroData): The input heterogeneous graph data.
        val_ratio (float): The proportion of nodes to use for validation.
        test_ratio (float): The proportion of nodes to use for testing.
        add_dummy_key (bool): If True, adds a temporary key to all node types
                              to ensure RandomNodeSplit processes them.

    Returns:
        Tuple[HeteroData, HeteroData, HeteroData]: train_graph, val_graph, test_graph
    """
    data_copy = data.clone() # Work on a copy to not modify original

    if data_copy.num_nodes == 0:
        print("Warning: Input graph has 0 nodes. Returning empty graphs.")
        return data_copy.clone(), data_copy.clone(), data_copy.clone()

    split_key = 'y' # Default key used by RandomNodeSplit
    temp_key_added = False

    if add_dummy_key:
        split_key = '_split_node_key_' # Use a temporary key name
        temp_key_added = True
        print(f"Adding temporary key '{split_key}' to node types for splitting...")
        for node_type in data_copy.node_types:
            # Add a simple range tensor; its content doesn't matter, only its presence
            if data_copy[node_type].num_nodes > 0:
                 data_copy[node_type][split_key] = torch.arange(data_copy[node_type].num_nodes)
            else:
                 # Handle node types with 0 nodes if they exist
                 data_copy[node_type][split_key] = torch.tensor([], dtype=torch.long)


    print("Applying RandomNodeSplit...")
    # Use 'random' split to specify exact ratios/counts for val/test
    # 'split="random"' requires num_val and num_test.
    # It doesn't use num_train_per_class in this mode if num_val/num_test are set.
    node_splitter = T.RandomNodeSplit(
        split='train_rest', # Use train_rest: Defines val/test, train is the rest
        num_val=val_ratio,  # Can be float (ratio) or int (count)
        num_test=test_ratio,# Can be float (ratio) or int (count)
        key=split_key      # Use the (potentially temporary) key
    )

    # RandomNodeSplit adds masks in-place (or returns the object with masks)
    data_with_masks = node_splitter(data_copy)
    print("RandomNodeSplit applied.")


    # --- Create subgraph dictionaries from masks ---
    train_nodes_dict: Dict[str, torch.Tensor] = {}
    val_nodes_dict: Dict[str, torch.Tensor] = {}
    test_nodes_dict: Dict[str, torch.Tensor] = {}

    found_mask = False
    for node_type in data_with_masks.node_types:
        if hasattr(data_with_masks[node_type], 'train_mask'):
            found_mask = True
            # Keep nodes where the mask is True
            train_nodes_dict[node_type] = data_with_masks[node_type].train_mask
            val_nodes_dict[node_type] = data_with_masks[node_type].val_mask
            test_nodes_dict[node_type] = data_with_masks[node_type].test_mask
            print(f"  Masks found for {node_type}: "
                  f"Train={train_nodes_dict[node_type].sum().item()}, "
                  f"Val={val_nodes_dict[node_type].sum().item()}, "
                  f"Test={test_nodes_dict[node_type].sum().item()}")
        # else:
            # If a node type didn't have the key, it won't have masks.
            # Decide how to handle: include all nodes or none?
            # Including all might be desired if only splitting specific types.
            # For splitting *all* types, the dummy key approach is needed.
            # print(f"  No masks found for {node_type}. Keeping all nodes in all splits.")
            # Keep all nodes if masks are missing (adjust if needed)
            # num_nodes = data_with_masks[node_type].num_nodes
            # train_nodes_dict[node_type] = torch.ones(num_nodes, dtype=torch.bool)
            # val_nodes_dict[node_type] = torch.ones(num_nodes, dtype=torch.bool)
            # test_nodes_dict[node_type] = torch.ones(num_nodes, dtype=torch.bool)


    if not found_mask:
         print("Error: No train/val/test masks were added by RandomNodeSplit.")
         print("Ensure the 'key' used exists in the node types you want to split,")
         print("or use 'add_dummy_key=True'. Returning original graph for all splits.")
         # Clean up temporary key if added
         if temp_key_added:
            for node_type in data_copy.node_types:
                if hasattr(data_copy[node_type], split_key):
                    del data_copy[node_type][split_key]
         return data.clone(), data.clone(), data.clone()


    # --- Create subgraph objects ---
    print("Creating subgraph objects...")
    # Important: Pass the original data *without masks* to subgraph,
    # as the dict contains the boolean masks or indices.
    original_data_for_subgraph = data # Use the very original data if keys were added to copy

    train_graph = original_data_for_subgraph.subgraph(train_nodes_dict)
    val_graph = original_data_for_subgraph.subgraph(val_nodes_dict)
    test_graph = original_data_for_subgraph.subgraph(test_nodes_dict)
    print("Subgraph objects created.")

    # --- Clean up temporary key if it was added ---
    # We clean the copy used for splitting, not the original or the subgraphs
    if temp_key_added:
        print(f"Removing temporary key '{split_key}'...")
        for node_type in data_copy.node_types:
            if hasattr(data_copy[node_type], split_key):
                 del data_copy[node_type][split_key] # Remove from the copy
        # Also remove from the original if it somehow got added (shouldn't have)
        # for node_type in data.node_types:
        #      if hasattr(data[node_type], split_key):
        #          del data[node_type][split_key]


    print(f"Train Graph: {train_graph}")
    print(f"Validation Graph: {val_graph}")
    print(f"Test Graph: {test_graph}")

    return train_graph, val_graph, test_graph
    
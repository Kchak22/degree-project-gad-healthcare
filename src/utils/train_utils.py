import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch_geometric.transforms import RandomNodeSplit

def train_model(model, data, num_epochs=100, learning_rate=0.001, weight_decay=1e-5,
                 lambda_attr=1.0, lambda_struct=1.0, plot=True):
    device = data['member'].x.device
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Apply RandomNodeSplit for automatic splitting
    node_split = RandomNodeSplit(num_val=0.1, num_test=0.1)(data)
    
    train_mask_member = node_split['member'].train_mask.to(device)
    val_mask_member = node_split['member'].val_mask.to(device)
    test_mask_member = node_split['member'].test_mask.to(device)
    
    train_mask_provider = node_split['provider'].train_mask.to(device)
    val_mask_provider = node_split['provider'].val_mask.to(device)
    test_mask_provider = node_split['provider'].test_mask.to(device)
    
    # Lists to track losses
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        if hasattr(model, 'compute_loss'):
            loss = model.compute_loss(node_split, lambda_attr=lambda_attr, lambda_struct=lambda_struct)
        else:
            x_hat_member, x_hat_provider, edge_logits = model(node_split)
            
            loss_attr_member = F.mse_loss(
                x_hat_member[train_mask_member],
                node_split['member'].x[train_mask_member]
            )
            loss_attr_provider = F.mse_loss(
                x_hat_provider[train_mask_provider],
                node_split['provider'].x[train_mask_provider]
            )
            loss_attr = loss_attr_member + loss_attr_provider
            
            edge_target = torch.ones_like(edge_logits)
            loss_struct = F.binary_cross_entropy_with_logits(edge_logits, edge_target)
            
            loss = lambda_attr * loss_attr + lambda_struct * loss_struct
        
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        
        # Validation step
        model.eval()
        with torch.no_grad():
            if hasattr(model, 'compute_loss'):
                val_loss = model.compute_loss(node_split, lambda_attr=lambda_attr, lambda_struct=lambda_struct)
            else:
                x_hat_m_val, x_hat_p_val, edge_logits_val = model(node_split)
                val_attr_m = F.mse_loss(
                    x_hat_m_val[val_mask_member],
                    node_split['member'].x[val_mask_member]
                )
                val_attr_p = F.mse_loss(
                    x_hat_p_val[val_mask_provider],
                    node_split['provider'].x[val_mask_provider]
                )
                val_attr_loss = val_attr_m + val_attr_p
                val_struct_loss = F.binary_cross_entropy_with_logits(
                    edge_logits_val, torch.ones_like(edge_logits_val)
                )
                val_loss = lambda_attr * val_attr_loss + lambda_struct * val_struct_loss
        
        val_losses.append(val_loss.item())
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch+1:03d} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f}")
    
    # Plot losses
    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.title('Training and Validation Losses')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
    
    return train_losses, val_losses


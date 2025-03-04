import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

def train_model(model, data, num_epochs=100, learning_rate=0.001, weight_decay=1e-5,
             lambda_attr=1.0, lambda_struct=1.0, plot=True):
    device = data['member'].x.device
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Create train/val/test masks for 'member' nodes:
    N_member = data['member'].num_nodes
    perm_member = torch.randperm(N_member)
    train_mask_member = torch.zeros(N_member, dtype=torch.bool, device=device)
    val_mask_member = torch.zeros(N_member, dtype=torch.bool, device=device)
    test_mask_member = torch.zeros(N_member, dtype=torch.bool, device=device)
    train_mask_member[perm_member[:int(0.8 * N_member)]] = True
    val_mask_member[perm_member[int(0.8 * N_member):int(0.9 * N_member)]] = True
    test_mask_member[perm_member[int(0.9 * N_member):]] = True
    data['member'].train_mask = train_mask_member
    data['member'].val_mask = val_mask_member
    data['member'].test_mask = test_mask_member

    # Similarly for 'provider' nodes:
    N_provider = data['provider'].num_nodes
    perm_provider = torch.randperm(N_provider)
    train_mask_provider = torch.zeros(N_provider, dtype=torch.bool, device=device)
    val_mask_provider = torch.zeros(N_provider, dtype=torch.bool, device=device)
    test_mask_provider = torch.zeros(N_provider, dtype=torch.bool, device=device)
    train_mask_provider[perm_provider[:int(0.8 * N_provider)]] = True
    val_mask_provider[perm_provider[int(0.8 * N_provider):int(0.9 * N_provider)]] = True
    test_mask_provider[perm_provider[int(0.9 * N_provider):]] = True
    data['provider'].train_mask = train_mask_provider
    data['provider'].val_mask = val_mask_provider
    data['provider'].test_mask = test_mask_provider

    # Lists to track losses
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass (handle different model interfaces)
        if hasattr(model, 'compute_loss'):
            # Your custom model
            loss = model.compute_loss(data, lambda_attr=lambda_attr, lambda_struct=lambda_struct)
        else:
            # Baseline models
            x_hat_member, x_hat_provider, edge_logits = model(data)
            
            # Attribute reconstruction loss on training masks:
            loss_attr_member = F.mse_loss(
                x_hat_member[data['member'].train_mask],
                data['member'].x[data['member'].train_mask]
            )
            loss_attr_provider = F.mse_loss(
                x_hat_provider[data['provider'].train_mask],
                data['provider'].x[data['provider'].train_mask]
            )
            loss_attr = loss_attr_member + loss_attr_provider
            
            # Structural reconstruction loss:
            edge_target = torch.ones_like(edge_logits)
            loss_struct = F.binary_cross_entropy_with_logits(edge_logits, edge_target)
            
            loss = lambda_attr * loss_attr + lambda_struct * loss_struct
        
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        
        # Validation step:
        model.eval()
        with torch.no_grad():
            if hasattr(model, 'compute_loss'):
                val_loss = model.compute_loss(data, lambda_attr=lambda_attr, lambda_struct=lambda_struct)
            else:
                x_hat_m_val, x_hat_p_val, edge_logits_val = model(data)
                val_attr_m = F.mse_loss(
                    x_hat_m_val[data['member'].val_mask],
                    data['member'].x[data['member'].val_mask]
                )
                val_attr_p = F.mse_loss(
                    x_hat_p_val[data['provider'].val_mask],
                    data['provider'].x[data['provider'].val_mask]
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

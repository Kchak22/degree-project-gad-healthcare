import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from src.models.main_model import BipartiteGraphAutoEncoder

def train_model(model, data, num_epochs=100, learning_rate=0.001, weight_decay=1e-5,
                lambda_attr=1.0, lambda_struct=1.0):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Create train/val/test masks for 'member' nodes:
    N_member = data['member'].num_nodes
    perm_member = torch.randperm(N_member)
    train_mask_member = torch.zeros(N_member, dtype=torch.bool)
    val_mask_member = torch.zeros(N_member, dtype=torch.bool)
    test_mask_member = torch.zeros(N_member, dtype=torch.bool)
    train_mask_member[perm_member[:int(0.8 * N_member)]] = True
    val_mask_member[perm_member[int(0.8 * N_member):int(0.9 * N_member)]] = True
    test_mask_member[perm_member[int(0.9 * N_member):]] = True
    data['member'].train_mask = train_mask_member
    data['member'].val_mask = val_mask_member
    data['member'].test_mask = test_mask_member

    # Similarly for 'provider' nodes:
    N_provider = data['provider'].num_nodes
    perm_provider = torch.randperm(N_provider)
    train_mask_provider = torch.zeros(N_provider, dtype=torch.bool)
    val_mask_provider = torch.zeros(N_provider, dtype=torch.bool)
    test_mask_provider = torch.zeros(N_provider, dtype=torch.bool)
    train_mask_provider[perm_provider[:int(0.8 * N_provider)]] = True
    val_mask_provider[perm_provider[int(0.8 * N_provider):int(0.9 * N_provider)]] = True
    test_mask_provider[perm_provider[int(0.9 * N_provider):]] = True
    data['provider'].train_mask = train_mask_provider
    data['provider'].val_mask = val_mask_provider
    data['provider'].test_mask = test_mask_provider

    # Lists to track losses
    train_total_losses = []
    train_attr_losses = []
    train_struct_losses = []
    val_total_losses = []
    val_attr_losses = []
    val_struct_losses = []

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        
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
        
        total_loss = lambda_attr * loss_attr + lambda_struct * loss_struct
        total_loss.backward()
        optimizer.step()
        
        train_total_losses.append(total_loss.item())
        train_attr_losses.append(loss_attr.item())
        train_struct_losses.append(loss_struct.item())
        
        # Validation step:
        model.eval()
        with torch.no_grad():
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
            val_total = lambda_attr * val_attr_loss + lambda_struct * val_struct_loss
        
        val_total_losses.append(val_total.item())
        val_attr_losses.append(val_attr_loss.item())
        val_struct_losses.append(val_struct_loss.item())
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch+1:03d} | "
                  f"Train Total: {total_loss.item():.4f} (A: {loss_attr.item():.4f}, S: {loss_struct.item():.4f}) | "
                  f"Val Total: {val_total.item():.4f} (A: {val_attr_loss.item():.4f}, S: {val_struct_loss.item():.4f})")
    
    # Plot training and validation losses:
    plt.figure(figsize=(10, 6))
    plt.plot(train_total_losses, label='Train Total')
    plt.plot(val_total_losses, label='Val Total')
    plt.title('Total Loss')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(train_attr_losses, label='Train Attr')
    plt.plot(val_attr_losses, label='Val Attr')
    plt.title('Attribute Loss')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(train_struct_losses, label='Train Struct')
    plt.plot(val_struct_losses, label='Val Struct')
    plt.title('Structure Loss')
    plt.legend()
    plt.show()

# If running this file directly, you could include:
if __name__ == "__main__":
    # Import or construct your HeteroData object here (e.g., by calling functions from dataloader.py)
    # For example:
    # from torch_geometric.data import HeteroData
    # data = HeteroData()
    # ... (populate data['member'] and data['provider'] with x, edge_index_dict, edge_attr_dict, etc.)
    
    # Set hyperparameters based on your dataset dimensions:
    in_dim_member = 128      # Example dimension (change accordingly)
    in_dim_provider = 128    # Example dimension (change accordingly)
    edge_dim = 1             # For 'nbr_claims' as a scalar
    hidden_dim = 64
    out_dim = 32
    num_layers = 2

    model = BipartiteGraphAutoEncoder(
        in_dim_member=in_dim_member,
        in_dim_provider=in_dim_provider,
        edge_dim=edge_dim,
        hidden_dim_member=hidden_dim,
        hidden_dim_provider=hidden_dim,
        out_dim=out_dim,
        num_layers=num_layers
    )
    
    # data should be prepared (e.g., using your data loader functions) before training.
    # For demonstration, assume `data` is ready.
    # train_model(model, data)

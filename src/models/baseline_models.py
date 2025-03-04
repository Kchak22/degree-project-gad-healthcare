import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, InnerProductDecoder
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA
import numpy as np

# MLPAutoencoder, SklearnBaseline, GCNAutoencoder, GATAutoencoder,
# SAGEAutoencoder, SplitAnomalyDetector

#################################################
# 1. Non-Graph Methods
#################################################

class MLPAutoencoder(nn.Module):
    """Simple MLP-based autoencoder that ignores graph structure."""
    def __init__(self, member_dim, provider_dim, hidden_dim, latent_dim, dropout=0.5):
        super(MLPAutoencoder, self).__init__()
        
        # Separate encoders for members and providers
        self.member_encoder = nn.Sequential(
            nn.Linear(member_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        self.provider_encoder = nn.Sequential(
            nn.Linear(provider_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # Separate decoders for members and providers
        self.member_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, member_dim)
        )
        
        self.provider_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, provider_dim)
        )
        
    def forward(self, data):
        # Process members and providers separately
        z_member = self.member_encoder(data['member'].x)
        z_provider = self.provider_encoder(data['provider'].x)
        
        x_hat_member = self.member_decoder(z_member)
        x_hat_provider = self.provider_decoder(z_provider)
        
        # For compatibility with your training function
        provider_idx, member_idx = data['provider', 'to', 'member'].edge_index
        edge_logits = torch.zeros(provider_idx.size(0), device=provider_idx.device)
        
        return x_hat_member, x_hat_provider, edge_logits
        
    def compute_anomaly_scores(self, data):
        x_hat_member, x_hat_provider, _ = self(data)
        
        # Compute anomaly scores as reconstruction errors
        member_scores = F.mse_loss(x_hat_member, data['member'].x, reduction='none').sum(dim=1)
        provider_scores = F.mse_loss(x_hat_provider, data['provider'].x, reduction='none').sum(dim=1)
        
        return {
            'member': member_scores,
            'provider': provider_scores
        }
    

class SklearnBaseline:
    """Wrapper for sklearn-based anomaly detection methods."""
    def __init__(self, method='iforest', **kwargs):
        if method == 'iforest':
            self.member_model = IsolationForest(**kwargs)
            self.provider_model = IsolationForest(**kwargs)
        elif method == 'ocsvm':
            self.member_model = OneClassSVM(**kwargs)
            self.provider_model = OneClassSVM(**kwargs)
        elif method == 'pca':
            self.member_model = PCA(n_components=kwargs.get('n_components', 0.95))
            self.provider_model = PCA(n_components=kwargs.get('n_components', 0.95))
        self.method = method
        
    def fit(self, data):
        self.member_model.fit(data['member'].x.detach().cpu().numpy())
        self.provider_model.fit(data['provider'].x.detach().cpu().numpy())
        return self
    
    # Adding these methods for compatibility
    def eval(self):
        pass
        
    def train(self):
        pass
    
    def compute_anomaly_scores(self, data):
        member_x = data['member'].x.detach().cpu().numpy().astype(np.float32)  # Convert to float32
        provider_x = data['provider'].x.detach().cpu().numpy().astype(np.float32)  # Convert to float32
        
        if self.method in ['iforest', 'ocsvm']:
            # Convert decision function to anomaly score (higher = more anomalous)
            member_scores = -self.member_model.decision_function(member_x).astype(np.float32)
            provider_scores = -self.provider_model.decision_function(provider_x).astype(np.float32)
        elif self.method == 'pca':
            # Reconstruction error as anomaly score
            member_proj = self.member_model.transform(member_x)
            member_recon = self.member_model.inverse_transform(member_proj)
            member_scores = np.sum((member_x - member_recon) ** 2, axis=1).astype(np.float32)
            
            provider_proj = self.provider_model.transform(provider_x)
            provider_recon = self.provider_model.inverse_transform(provider_proj)
            provider_scores = np.sum((provider_x - provider_recon) ** 2, axis=1).astype(np.float32)
        
        # Ensure we're using float32 before sending to device
        return {
            'member': torch.tensor(member_scores, dtype=torch.float32, device=data['member'].x.device),
            'provider': torch.tensor(provider_scores, dtype=torch.float32, device=data['provider'].x.device)
        }



#################################################
# 2. Simpler Graph Models
#################################################

class GCNEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, latent_dim, dropout=0.5):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, latent_dim)
        self.dropout = dropout
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class GCNAutoencoder(nn.Module):
    """Standard GCN-based autoencoder (not bipartite-specific)."""
    def __init__(self, in_dim, hidden_dim, latent_dim, dropout=0.5):
        super(GCNAutoencoder, self).__init__()
        self.encoder = GCNEncoder(in_dim, hidden_dim, latent_dim, dropout)
        self.decoder = nn.Linear(latent_dim, in_dim)
        
    def forward(self, data):
        # We need to create a unified graph for standard GCN
        # Combine features by padding if different dimensions
        max_dim = max(data['member'].x.size(1), data['provider'].x.size(1))
        
        x_member = F.pad(data['member'].x, (0, max_dim - data['member'].x.size(1)))
        x_provider = F.pad(data['provider'].x, (0, max_dim - data['provider'].x.size(1)))
        
        x = torch.cat([x_member, x_provider], dim=0)
        
        # Create bidirectional edge indices from the member-provider edges
        edge_index = data['provider', 'to', 'member'].edge_index
        edge_index_reverse = torch.stack([edge_index[1], edge_index[0]], dim=0)
        
        # Add offset to provider indices (they come after members)
        edge_index_reverse[0, :] += data['member'].num_nodes
        edge_index[1, :] += data['member'].num_nodes
        
        # Combine edges
        combined_edge_index = torch.cat([edge_index, edge_index_reverse], dim=1)
        
        # Encode
        z = self.encoder(x, combined_edge_index)
        
        # Decode
        x_hat = self.decoder(z)
        
        # Split back to member and provider
        x_hat_member_full = x_hat[:data['member'].num_nodes]
        x_hat_provider_full = x_hat[data['member'].num_nodes:]
        
        # Use only the dimensions that exist in the original data
        member_dim = data['member'].x.size(1)
        provider_dim = data['provider'].x.size(1)
        
        x_hat_member = x_hat_member_full[:, :member_dim]
        x_hat_provider = x_hat_provider_full[:, :provider_dim]
        
        # For compatibility: fake edge_logits for structure prediction
        provider_idx, member_idx = data['provider', 'to', 'member'].edge_index
        edge_logits = torch.zeros(provider_idx.size(0), device=provider_idx.device)
        
        return x_hat_member, x_hat_provider, edge_logits

    
    def compute_anomaly_scores(self, data):
        x_hat_member, x_hat_provider, _ = self(data)
        
        # Use only the dimensions that exist in the original data
        member_dim = data['member'].x.size(1)
        provider_dim = data['provider'].x.size(1)
        
        x_hat_member = x_hat_member[:, :member_dim]
        x_hat_provider = x_hat_provider[:, :provider_dim]
        
        # Compute anomaly scores as reconstruction errors
        member_scores = F.mse_loss(x_hat_member, data['member'].x, reduction='none').sum(dim=1)
        provider_scores = F.mse_loss(x_hat_provider, data['provider'].x, reduction='none').sum(dim=1)
        
        return {
            'member': member_scores,
            'provider': provider_scores
        }

class GATAutoencoder(nn.Module):
    """GAT-based autoencoder with separate dimensions for member and provider nodes."""
    def __init__(self, in_dim_member, in_dim_provider, hidden_dim, latent_dim, heads=4, dropout=0.5):
        super(GATAutoencoder, self).__init__()
        
        # GAT encoders for each node type with correct dimensions
        self.member_conv1 = GATConv(in_dim_member, hidden_dim // heads, heads=heads, dropout=dropout)
        self.provider_conv1 = GATConv(in_dim_provider, hidden_dim // heads, heads=heads, dropout=dropout)
        
        self.member_conv2 = GATConv(hidden_dim, latent_dim, heads=1, dropout=dropout)
        self.provider_conv2 = GATConv(hidden_dim, latent_dim, heads=1, dropout=dropout)
        
        # Decoders with correct dimensions
        self.member_decoder = nn.Linear(latent_dim, in_dim_member)
        self.provider_decoder = nn.Linear(latent_dim, in_dim_provider)
        
        self.dropout = dropout
        
    def forward(self, data):
        # Create member-to-member and provider-to-provider edge indices 
        # (self-loops for simplicity)
        member_self_loops = torch.stack([
            torch.arange(data['member'].num_nodes, device=data['member'].x.device),
            torch.arange(data['member'].num_nodes, device=data['member'].x.device)
        ], dim=0)
        
        provider_self_loops = torch.stack([
            torch.arange(data['provider'].num_nodes, device=data['provider'].x.device),
            torch.arange(data['provider'].num_nodes, device=data['provider'].x.device)
        ], dim=0)
        
        # Get bipartite edges
        edge_index = data['provider', 'to', 'member'].edge_index
        edge_index_reverse = torch.stack([edge_index[1], edge_index[0] + data['member'].num_nodes], dim=0)
        member_to_provider = torch.stack([edge_index[0], edge_index[1] + data['member'].num_nodes], dim=0)
        
        # Perform convolutions for members
        member_edge_index = torch.cat([member_self_loops, member_to_provider], dim=1)
        x_member = F.elu(self.member_conv1(data['member'].x, member_edge_index))
        x_member = F.dropout(x_member, p=self.dropout, training=self.training)
        z_member = self.member_conv2(x_member, member_edge_index)
        
        # Perform convolutions for providers
        provider_edge_index = torch.cat([provider_self_loops, edge_index_reverse], dim=1)
        provider_edge_index[1, provider_self_loops.size(1):] -= data['member'].num_nodes
        x_provider = F.elu(self.provider_conv1(data['provider'].x, provider_edge_index))
        x_provider = F.dropout(x_provider, p=self.dropout, training=self.training)
        z_provider = self.provider_conv2(x_provider, provider_edge_index)
        
        # Decode
        x_hat_member = self.member_decoder(z_member)
        x_hat_provider = self.provider_decoder(z_provider)
        
        # Edge prediction
        provider_idx, member_idx = data['provider', 'to', 'member'].edge_index
        edge_logits = (z_provider[provider_idx] * z_member[member_idx]).sum(dim=1)
        
        return x_hat_member, x_hat_provider, edge_logits
        
    def compute_anomaly_scores(self, data):
        x_hat_member, x_hat_provider, edge_logits = self(data)
        
        # Attribute anomaly scores
        member_scores = F.mse_loss(x_hat_member, data['member'].x, reduction='none').sum(dim=1)
        provider_scores = F.mse_loss(x_hat_provider, data['provider'].x, reduction='none').sum(dim=1)
        
        return {
            'member': member_scores,
            'provider': provider_scores
        }

class SAGEAutoencoder(nn.Module):
    """GraphSAGE-based autoencoder."""
    def __init__(self, in_dim, hidden_dim, latent_dim, dropout=0.5):
        super(SAGEAutoencoder, self).__init__()
        
        # GraphSAGE encoders
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, latent_dim)
        
        # Decoders
        self.decoder = nn.Linear(latent_dim, in_dim)
        
        self.dropout = dropout
        
    def forward(self, data):
        # Similar to GCN approach - unified graph
        max_dim = max(data['member'].x.size(1), data['provider'].x.size(1))
        
        x_member = F.pad(data['member'].x, (0, max_dim - data['member'].x.size(1)))
        x_provider = F.pad(data['provider'].x, (0, max_dim - data['provider'].x.size(1)))
        
        x = torch.cat([x_member, x_provider], dim=0)
        
        # Create bidirectional edge indices
        edge_index = data['provider', 'to', 'member'].edge_index
        edge_index_reverse = torch.stack([edge_index[1], edge_index[0]], dim=0)
        
        # Add offset to provider indices
        edge_index_reverse[0, :] += data['member'].num_nodes
        edge_index[1, :] += data['member'].num_nodes
        
        # Combine edges
        combined_edge_index = torch.cat([edge_index, edge_index_reverse], dim=1)
        
        # Encode with GraphSAGE
        x = F.relu(self.conv1(x, combined_edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        z = self.conv2(x, combined_edge_index)
        
        # Decode
        x_hat = self.decoder(z)
        
        # Split back to member and provider
        x_hat_member = x_hat[:data['member'].num_nodes]
        x_hat_provider = x_hat[data['member'].num_nodes:]
        
        # For compatibility: edge prediction
        provider_idx, member_idx = data['provider', 'to', 'member'].edge_index
        member_z = z[:data['member'].num_nodes]
        provider_z = z[data['member'].num_nodes:]
        edge_logits = (provider_z[provider_idx] * member_z[member_idx]).sum(dim=1)
        
        return x_hat_member, x_hat_provider, edge_logits
    
    def compute_anomaly_scores(self, data):
        x_hat_member, x_hat_provider, _ = self(data)
        
        # Use only the dimensions that exist in the original data
        member_dim = data['member'].x.size(1)
        provider_dim = data['provider'].x.size(1)
        
        x_hat_member = x_hat_member[:, :member_dim]
        x_hat_provider = x_hat_provider[:, :provider_dim]
        
        # Compute anomaly scores
        member_scores = F.mse_loss(x_hat_member, data['member'].x, reduction='none').sum(dim=1)
        provider_scores = F.mse_loss(x_hat_provider, data['provider'].x, reduction='none').sum(dim=1)
        
        return {
            'member': member_scores,
            'provider': provider_scores
        }

#################################################
# 3. Split Attribute and Structure Models
#################################################

class SplitAnomalyDetector:
    """Combines separate structure and attribute anomaly detection methods."""
    def __init__(self, attr_model, struct_model, alpha=0.5):
        self.attr_model = attr_model
        self.struct_model = struct_model
        self.alpha = alpha  # Weight for combining scores
        
    def fit(self, data):
        if hasattr(self.attr_model, 'fit'):
            self.attr_model.fit(data)
        if hasattr(self.struct_model, 'fit'):
            self.struct_model.fit(data)
        return self
    
    def compute_anomaly_scores(self, data):
        attr_scores = self.attr_model.compute_anomaly_scores(data)
        struct_scores = self.struct_model.compute_anomaly_scores(data)
        
        # Normalize scores between 0 and 1
        for node_type in attr_scores:
            min_val = attr_scores[node_type].min()
            max_val = attr_scores[node_type].max()
            if max_val > min_val:
                attr_scores[node_type] = (attr_scores[node_type] - min_val) / (max_val - min_val)
            
            min_val = struct_scores[node_type].min()
            max_val = struct_scores[node_type].max()
            if max_val > min_val:
                struct_scores[node_type] = (struct_scores[node_type] - min_val) / (max_val - min_val)
        
        # Combine scores
        combined_scores = {}
        for node_type in attr_scores:
            combined_scores[node_type] = self.alpha * attr_scores[node_type] + (1 - self.alpha) * struct_scores[node_type]
            
        return combined_scores

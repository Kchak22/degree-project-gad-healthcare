import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, HeteroConv
from torch_geometric.utils import softmax, negative_sampling
from torch_geometric.data import HeteroData # Assuming HeteroData is used
import matplotlib.pyplot as plt # For plotting loss

# Custom Bipartite Attention Layer (Assuming this is correct as per your previous code)
class BipartiteAttentionConv(MessagePassing):
    def __init__(self, in_channels_src, in_channels_dst, edge_dim, out_channels):
        super(BipartiteAttentionConv, self).__init__(aggr='add', node_dim=0) # node_dim=0 for heterogeneous graphs
        self.lin_src = nn.Linear(in_channels_src, out_channels, bias=False)
        self.lin_dst = nn.Linear(in_channels_dst, out_channels, bias=False)
        self.lin_edge = nn.Linear(edge_dim, out_channels, bias=False)
        # Attention mechanism parameters
        self.att_src = nn.Parameter(torch.Tensor(1, out_channels))
        self.att_dst = nn.Parameter(torch.Tensor(1, out_channels))
        self.att_edge = nn.Parameter(torch.Tensor(1, out_channels)) # Added attention for edge features influence

        # Removed the single self.att parameter for more granular attention learning
        # self.att = nn.Parameter(torch.Tensor(2 * out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_src.weight)
        nn.init.xavier_uniform_(self.lin_dst.weight)
        nn.init.xavier_uniform_(self.lin_edge.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        nn.init.xavier_uniform_(self.att_edge)

    def forward(self, x, edge_index, edge_attr=None, size=None):
        # x: Tuple (x_src, x_dst)
        x_src, x_dst = x
        
        # Project features before propagation
        x_src_proj = self.lin_src(x_src) # Used in attention calculation
        x_dst_proj = self.lin_dst(x_dst) # Used in attention calculation and final update

        # Project edge features
        if edge_attr is None:
             # Handle case with no edge attributes if necessary, maybe default to zeros
             edge_attr = torch.zeros((edge_index.size(1), self.lin_edge.in_features),
                                    device=edge_index.device)
        edge_attr_proj = self.lin_edge(edge_attr) # Used in attention and message

        # Propagate messages
        # size argument is crucial for bipartite graphs to know node counts
        out = self.propagate(edge_index, x=(x_src_proj, x_dst_proj), edge_attr=edge_attr_proj, size=size)

        # Add residual connection and apply activation (consistent with report Eq 3.6)
        # Note: Report adds residual h_i^(l) BEFORE ELU, common practice is after
        # Residual added to the destination node features x_dst
        out = out + x_dst # Residual connection (using original x_dst, not projected)
        
        return out # ELU/Dropout applied in the main model's forward pass

    def message(self, x_j, x_i, edge_attr, index, ptr, size_i):
         # x_i is the destination node feature (projected), x_j is the source node feature (projected)
         # edge_attr is the projected edge feature

        # Calculate attention scores (aligning with Eq 3.3 concept)
        # Use projected features for attention calculation
        alpha_src = (x_j * self.att_src).sum(dim=-1)
        alpha_dst = (x_i * self.att_dst).sum(dim=-1)
        alpha_edge = (edge_attr * self.att_edge).sum(dim=-1) # Edge influence on attention

        # Combine attention components and apply LeakyReLU
        alpha = alpha_src + alpha_dst + alpha_edge
        alpha = F.leaky_relu(alpha)

        # Normalize attention scores
        alpha = softmax(alpha, index, ptr, size_i)

        # Calculate message (aligning with Eq 3.4)
        # Message = attention_weight * (source_node_feature + transformed_edge_feature)
        message = alpha.unsqueeze(-1) * (x_j + edge_attr)
        return message

    # Update function removed - aggregation happens via aggr='add', residual in forward


# Modular Attribute Decoder (Keep as is)
class AttributeDecoder(nn.Module):
    def __init__(self, in_dim, out_dim, num_layers=2, hidden_dim=None, dropout=0.5):
        super(AttributeDecoder, self).__init__()
        if hidden_dim is None:
            hidden_dim = in_dim
        layers = []
        if num_layers == 1:
            layers.append(nn.Linear(in_dim, out_dim))
        else:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU()) # Report doesn't specify ReLU here, but common practice
            layers.append(nn.Dropout(dropout))
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(hidden_dim, out_dim))
        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.decoder(x)

# --- Bipartite Graph Autoencoder based on Report ---
class BipartiteGraphAutoEncoder_ReportBased(nn.Module):
    def __init__(self, in_dim_member, in_dim_provider, edge_dim,
                 hidden_dim, # Simplified to one hidden_dim for intermediate layers
                 latent_dim, # Output dimension 'out_dim' renamed to 'latent_dim'
                 num_conv_layers=2, # Renamed num_layers to num_conv_layers
                 num_dec_layers=2,
                 dropout=0.5):
        super(BipartiteGraphAutoEncoder_ReportBased, self).__init__()
        self.dropout_rate = dropout # Renamed dropout to dropout_rate

        # 1. Feature Projection (Encoder - Step 1, Eq 3.7, 3.8)
        # Use different hidden dims if desired, but simplifying for now
        self.proj_member = nn.Linear(in_dim_member, hidden_dim)
        self.proj_provider = nn.Linear(in_dim_provider, hidden_dim)

        # 2. Convolution layers (Encoder - Step 2, Eq 3.9, 3.10)
        self.convs = nn.ModuleList()
        for i in range(num_conv_layers):
            # Determine input dimension for this layer
            current_hidden_dim = hidden_dim # Input for all conv layers is hidden_dim after projection
            
            conv_member = BipartiteAttentionConv(
                in_channels_src=current_hidden_dim, # Provider embeds
                in_channels_dst=current_hidden_dim, # Member embeds
                edge_dim=edge_dim,
                out_channels=current_hidden_dim # Output dimension remains hidden_dim
            )
            conv_provider = BipartiteAttentionConv(
                in_channels_src=current_hidden_dim, # Member embeds
                in_channels_dst=current_hidden_dim, # Provider embeds
                edge_dim=edge_dim,
                out_channels=current_hidden_dim # Output dimension remains hidden_dim
            )
            conv = HeteroConv({
                ('provider', 'to', 'member'): conv_member,
                ('member', 'to', 'provider'): conv_provider,
            }, aggr='sum') # Aggregation for HeteroConv layer itself
            self.convs.append(conv)

        # 3. Final Transformation to Latent Space (Encoder - Step 4, Eq 3.12, 3.13)
        self.final_member = nn.Linear(hidden_dim, latent_dim)
        self.final_provider = nn.Linear(hidden_dim, latent_dim)

        # 4. Feature Decoders (Eq 3.14, 3.15)
        self.dec_member = AttributeDecoder(latent_dim, in_dim_member, num_dec_layers, hidden_dim=hidden_dim, dropout=dropout)
        self.dec_provider = AttributeDecoder(latent_dim, in_dim_provider, num_dec_layers, hidden_dim=hidden_dim, dropout=dropout)

        # Structure Decoder is implicit via inner product (Eq 3.16)

    def encode(self, data: HeteroData):
        x_dict = data.x_dict
        edge_index_dict = data.edge_index_dict
        edge_attr_dict = data.edge_attr_dict # Assuming edge_attr is present

        # 1. Feature Projection & Initial Activation/Dropout
        x_dict = {
            'member': F.elu(self.proj_member(x_dict['member'])),
            'provider': F.elu(self.proj_provider(x_dict['provider']))
        }
        x_dict = {k: F.dropout(v, p=self.dropout_rate, training=self.training)
                  for k, v in x_dict.items()}

        # 2. Message Passing Layers
        for conv in self.convs:
             # Pass necessary edge attributes
             edge_attr_provider_to_member = edge_attr_dict.get(('provider', 'to', 'member'), None)
             edge_attr_member_to_provider = edge_attr_dict.get(('member', 'to', 'provider'), None)
             
             # HeteroConv needs edge_attr per edge type if the conv layer uses it
             conv_edge_attr_dict = {}
             if edge_attr_provider_to_member is not None:
                 conv_edge_attr_dict[('provider', 'to', 'member')] = edge_attr_provider_to_member
             if edge_attr_member_to_provider is not None:
                 conv_edge_attr_dict[('member', 'to', 'provider')] = edge_attr_member_to_provider

             x_dict = conv(x_dict, edge_index_dict, edge_attr_dict=conv_edge_attr_dict)
             
             # Activation and Dropout after each conv layer (Eq 3.6, 3.11)
             x_dict = {k: F.elu(v) for k, v in x_dict.items()}
             x_dict = {k: F.dropout(v, p=self.dropout_rate, training=self.training)
                       for k, v in x_dict.items()}

        # 3. Final Transformation to Latent Space
        z_dict = {
            'member': self.final_member(x_dict['member']),
            'provider': self.final_provider(x_dict['provider'])
        }

        return z_dict # Return latent representations

    def decode_features(self, z_dict):
        # 4. Feature Reconstruction
        x_hat_member = self.dec_member(z_dict['member'])
        x_hat_provider = self.dec_provider(z_dict['provider'])
        return x_hat_member, x_hat_provider

    def decode_structure(self, z_dict, edge_index):
        # 5. Structure Reconstruction (Inner product for specific edges)
        # Assumes edge_index is for ('provider', 'to', 'member')
        row, col = edge_index
        z_provider = z_dict['provider'][row]
        z_member = z_dict['member'][col]
        
        # Compute dot product (Eq 3.16, before sigmoid)
        edge_logits = (z_provider * z_member).sum(dim=-1)
        return edge_logits

    def forward(self, data: HeteroData):
        # Encode to get latent representations
        z_dict = self.encode(data)
        
        # Decode features
        x_hat_member, x_hat_provider = self.decode_features(z_dict)
        
        # Note: Structure decoding (calculating edge logits) is done in the loss function
        # as it requires sampling positive and negative edges.
        # We return latent embeddings needed for structure loss.
        
        return x_hat_member, x_hat_provider, z_dict['member'], z_dict['provider']


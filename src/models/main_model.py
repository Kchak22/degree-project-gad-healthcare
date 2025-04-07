import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, HeteroConv
from torch_geometric.utils import softmax, negative_sampling
from torch_geometric.data import HeteroData
import matplotlib.pyplot as plt

class BipartiteAttentionConv(MessagePassing):
    def __init__(self, in_channels_src, in_channels_dst, edge_dim, out_channels):
        super(BipartiteAttentionConv, self).__init__(aggr='add', node_dim=0)
        self.lin_src = nn.Linear(in_channels_src, out_channels, bias=False)
        self.lin_dst = nn.Linear(in_channels_dst, out_channels, bias=False)
        # Adjust lin_edge input dim if edge_dim is 0
        self.lin_edge = nn.Linear(max(edge_dim, 1), out_channels, bias=False) # Use max(edge_dim, 1)
        self.att_src = nn.Parameter(torch.Tensor(1, out_channels))
        self.att_dst = nn.Parameter(torch.Tensor(1, out_channels))
        self.att_edge = nn.Parameter(torch.Tensor(1, out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_src.weight)
        nn.init.xavier_uniform_(self.lin_dst.weight)
        nn.init.xavier_uniform_(self.lin_edge.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        nn.init.xavier_uniform_(self.att_edge)

    def forward(self, x, edge_index, edge_attr=None, size=None):
        x_src, x_dst = x
        x_src_proj = self.lin_src(x_src)
        x_dst_proj = self.lin_dst(x_dst)

        # Handle missing or zero-dim edge_attr
        if edge_attr is None or self.lin_edge.in_features == 1 and edge_attr.shape[1] == 0:
             edge_attr_processed = torch.zeros((edge_index.size(1), max(self.lin_edge.in_features, 1)),
                                    device=edge_index.device)
             if self.lin_edge.in_features > 0 and edge_attr is not None: # If edge_attr exists but model expects dim=1
                 # This case needs careful handling based on expectation.
                 # Assuming we default to zeros if edge_dim=0 was specified at init but attrs exist.
                 pass # edge_attr_processed remains zeros
             elif edge_attr is not None:
                 edge_attr_processed = edge_attr # Use provided if dims match

        elif edge_attr.shape[1] != self.lin_edge.in_features:
             # If edge_dim > 0 specified, but edge_attr is missing/different
             print(f"Warning: Edge attributes shape mismatch. Expected {self.lin_edge.in_features}, got {edge_attr.shape[1]}. Using zeros.")
             edge_attr_processed = torch.zeros((edge_index.size(1), self.lin_edge.in_features),
                                    device=edge_index.device)
        else:
            edge_attr_processed = edge_attr

        edge_attr_proj = self.lin_edge(edge_attr_processed)
        out = self.propagate(edge_index, x=(x_src_proj, x_dst_proj), edge_attr=edge_attr_proj, size=size)
        out = out + x_dst # Residual connection
        return out

    def message(self, x_j, x_i, edge_attr, index, ptr, size_i):
        alpha_src = (x_j * self.att_src).sum(dim=-1)
        alpha_dst = (x_i * self.att_dst).sum(dim=-1)
        alpha_edge = (edge_attr * self.att_edge).sum(dim=-1)
        alpha = alpha_src + alpha_dst + alpha_edge
        alpha = F.leaky_relu(alpha)
        alpha = softmax(alpha, index, ptr, size_i)
        message = alpha.unsqueeze(-1) * (x_j + edge_attr) # Combine source and edge info in message
        return message

# --- AttributeDecoder remains the same ---
class AttributeDecoder(nn.Module):
    def __init__(self, in_dim, out_dim, num_layers=2, hidden_dim=None, dropout=0.5):
        super(AttributeDecoder, self).__init__()
        if hidden_dim is None:
            hidden_dim = in_dim # Default hidden to input if not specified
        layers = []
        current_dim = in_dim
        if num_layers == 1:
            layers.append(nn.Linear(current_dim, out_dim))
        else:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(current_dim, hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(hidden_dim, out_dim))
        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.decoder(x)

# --- Bipartite Graph Autoencoder with Edge Decoder ---
class BipartiteGraphAutoEncoder_ReportBased(nn.Module):
    def __init__(self, in_dim_member, in_dim_provider, edge_dim,
                 hidden_dim, latent_dim,
                 num_conv_layers=2, num_dec_layers=2, dropout=0.5):
        super(BipartiteGraphAutoEncoder_ReportBased, self).__init__()
        self.dropout_rate = dropout
        self.edge_dim = edge_dim # Store edge_dim

        # Encoder parts
        self.proj_member = nn.Linear(in_dim_member, hidden_dim)
        self.proj_provider = nn.Linear(in_dim_provider, hidden_dim)
        self.convs = nn.ModuleList()
        for _ in range(num_conv_layers):
            conv = HeteroConv({
                ('provider', 'to', 'member'): BipartiteAttentionConv(hidden_dim, hidden_dim, max(edge_dim, 1), hidden_dim),
                ('member', 'to', 'provider'): BipartiteAttentionConv(hidden_dim, hidden_dim, max(edge_dim, 1), hidden_dim),
            }, aggr='sum')
            self.convs.append(conv)
        self.final_member = nn.Linear(hidden_dim, latent_dim)
        self.final_provider = nn.Linear(hidden_dim, latent_dim)

        # Feature Decoders
        self.dec_member = AttributeDecoder(latent_dim, in_dim_member, num_dec_layers, hidden_dim, dropout)
        self.dec_provider = AttributeDecoder(latent_dim, in_dim_provider, num_dec_layers, hidden_dim, dropout)

        # <<< --- ADDED: Edge Feature Decoder --- >>>
        # Takes concatenated latent features of connected nodes
        # Only add if edge_dim > 0
        if self.edge_dim > 0:
            self.dec_edge = AttributeDecoder(latent_dim * 2, edge_dim, num_dec_layers, hidden_dim, dropout)
        else:
            self.dec_edge = None # No edge decoder needed if no edge features

    def encode(self, data: HeteroData):
        x_dict = data.x_dict
        edge_index_dict = data.edge_index_dict
        # Use getattr to safely get edge_attr_dict, default to empty dict
        edge_attr_dict = getattr(data, 'edge_attr_dict', {})

        x_dict = {
            'member': F.elu(self.proj_member(x_dict['member'])),
            'provider': F.elu(self.proj_provider(x_dict['provider']))
        }
        x_dict = {k: F.dropout(v, p=self.dropout_rate, training=self.training) for k, v in x_dict.items()}

        for conv in self.convs:
            # Prepare edge attributes for HeteroConv correctly
            conv_edge_attr_dict = {}
            for edge_type, edge_idx in edge_index_dict.items():
                # Retrieve attributes if they exist for this edge type
                edge_attr = edge_attr_dict.get(edge_type, None)
                if edge_attr is not None:
                    conv_edge_attr_dict[edge_type] = edge_attr
                # If edge_dim=0, BipartiteAttentionConv will handle None/zeros

            x_dict = conv(x_dict, edge_index_dict, edge_attr_dict=conv_edge_attr_dict)
            x_dict = {k: F.elu(v) for k, v in x_dict.items()}
            x_dict = {k: F.dropout(v, p=self.dropout_rate, training=self.training) for k, v in x_dict.items()}

        z_dict = {
            'member': self.final_member(x_dict['member']),
            'provider': self.final_provider(x_dict['provider'])
        }
        return z_dict

    def decode_features(self, z_dict):
        x_hat_member = self.dec_member(z_dict['member'])
        x_hat_provider = self.dec_provider(z_dict['provider'])
        return x_hat_member, x_hat_provider

    def decode_structure(self, z_dict, edge_index):
        row, col = edge_index
        z_provider = z_dict['provider'][row]
        z_member = z_dict['member'][col]
        edge_logits = (z_provider * z_member).sum(dim=-1)
        return edge_logits

    def decode_edge_attributes(self, z_dict, edge_index):
        """Decodes edge attributes for given edges based on node embeddings."""
        if self.dec_edge is None:
            # Return None or zeros if no edge features are expected
            return None # Or maybe torch.zeros(edge_index.shape[1], 0)?

        row, col = edge_index # Assuming provider -> member edge_index
        z_provider = z_dict['provider'][row]
        z_member = z_dict['member'][col]

        # Concatenate node embeddings to form input for edge decoder
        z_edge_input = torch.cat([z_provider, z_member], dim=-1)
        edge_attr_hat = self.dec_edge(z_edge_input)
        return edge_attr_hat

    def forward(self, data: HeteroData):
        z_dict = self.encode(data)
        x_hat_member, x_hat_provider = self.decode_features(z_dict)

        # <<< --- MODIFIED: Optionally decode edge attributes --- >>>
        # Decode attributes only for the primary edge type for efficiency, if needed later
        target_edge_type = ('provider', 'to', 'member')
        edge_attr_hat = None
        if target_edge_type in data.edge_index_dict and self.dec_edge is not None:
             edge_index = data[target_edge_type].edge_index
             edge_attr_hat = self.decode_edge_attributes(z_dict, edge_index)

        # Return latent embeddings and feature reconstructions.
        # Return edge reconstructions separately if needed.
        return x_hat_member, x_hat_provider, z_dict['member'], z_dict['provider'], edge_attr_hat, z_dict # Return full z_dict
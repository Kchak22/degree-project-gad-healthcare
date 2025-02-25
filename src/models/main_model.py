import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, HeteroConv
from torch_geometric.utils import softmax

# Custom Bipartite Attention Layer 
class BipartiteAttentionConv(MessagePassing):
    def __init__(self, in_channels_src, in_channels_dst, edge_dim, out_channels):
        super(BipartiteAttentionConv, self).__init__(aggr='add')
        self.lin_src = nn.Linear(in_channels_src, out_channels, bias=False)
        self.lin_dst = nn.Linear(in_channels_dst, out_channels, bias=False)
        self.lin_edge = nn.Linear(edge_dim, out_channels, bias=False)
        self.att = nn.Parameter(torch.Tensor(2 * out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_src.weight)
        nn.init.xavier_uniform_(self.lin_dst.weight)
        nn.init.xavier_uniform_(self.lin_edge.weight)
        nn.init.xavier_uniform_(self.att.unsqueeze(0))

    def forward(self, x, edge_index, edge_attr=None):
        x_src, x_dst = x
        if edge_attr is None:
            edge_attr = torch.zeros((edge_index.size(1), self.lin_edge.in_features),
                                    device=x_src.device)
        x_src_trans = self.lin_src(x_src)
        x_dst_trans = self.lin_dst(x_dst)
        edge_attr_trans = self.lin_edge(edge_attr)
        return self.propagate(edge_index, x=(x_src_trans, x_dst_trans), edge_attr=edge_attr_trans)

    def message(self, x_i, x_j, edge_attr, index):
        a_input = torch.cat([x_i, x_j], dim=-1)
        alpha = F.leaky_relu((a_input * self.att).sum(dim=-1))
        alpha = softmax(alpha, index)
        return alpha.view(-1, 1) * (x_j + edge_attr)

    def update(self, aggr_out, x):
        return F.elu(aggr_out + x[1])

# Modular Attribute Decoder 
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
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(hidden_dim, out_dim))
        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.decoder(x)

# Improved Bipartite Graph Autoencoder
class BipartiteGraphAutoEncoder(nn.Module):
    def __init__(self, in_dim_member, in_dim_provider, edge_dim,
                 hidden_dim_member, hidden_dim_provider, out_dim,
                 num_layers=2, dropout=0.5, num_dec_layers=2):
        super(BipartiteGraphAutoEncoder, self).__init__()
        self.dropout = dropout

        # Feature projectors
        self.self_member = nn.Linear(in_dim_member, hidden_dim_member)
        self.self_provider = nn.Linear(in_dim_provider, hidden_dim_provider)

        # Convolution layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv_member = BipartiteAttentionConv(
                in_channels_src=hidden_dim_provider,
                in_channels_dst=hidden_dim_member,
                edge_dim=edge_dim,
                out_channels=hidden_dim_member
            )
            conv_provider = BipartiteAttentionConv(
                in_channels_src=hidden_dim_member,
                in_channels_dst=hidden_dim_provider,
                edge_dim=edge_dim,
                out_channels=hidden_dim_provider
            )
            conv = HeteroConv({
                ('provider', 'to', 'member'): conv_member,
                ('member', 'to', 'provider'): conv_provider,
            }, aggr='sum')
            self.convs.append(conv)

        # Structure Decoder (NEW)
        self.struct_decoder = nn.Linear(out_dim, out_dim)
        
        # Final projections and attribute decoders
        self.final_member = nn.Linear(hidden_dim_member, out_dim)
        self.final_provider = nn.Linear(hidden_dim_provider, out_dim)
        self.dec_member = AttributeDecoder(hidden_dim_member, in_dim_member, num_dec_layers, dropout=dropout)
        self.dec_provider = AttributeDecoder(hidden_dim_provider, in_dim_provider, num_dec_layers, dropout=dropout)

    def forward(self, data):
        x_dict = {
            'member': F.elu(self.self_member(data['member'].x)),
            'provider': F.elu(self.self_provider(data['provider'].x))
        }

        x_dict = {k: F.dropout(v, p=self.dropout, training=self.training) 
                  for k, v in x_dict.items()}

        # Message passing
        for conv in self.convs:
            x_dict = conv(x_dict, data.edge_index_dict, data.edge_attr_dict)
            x_dict = {k: F.elu(v) for k, v in x_dict.items()}
            x_dict = {k: F.dropout(v, p=self.dropout, training=self.training) 
                      for k, v in x_dict.items()}

        # Feature reconstruction
        x_hat_member = self.dec_member(x_dict['member'])
        x_hat_provider = self.dec_provider(x_dict['provider'])

        # Structure Decoding (Embeddings)
        z_member = self.final_member(x_dict['member'])
        z_provider = self.final_provider(x_dict['provider'])

        # Edge logits computation (NEW integration)
        provider_idx, member_idx = data['provider', 'to', 'member'].edge_index
        edge_logits = (z_provider[provider_idx] * z_member[member_idx]).sum(dim=1)

        return x_hat_member, x_hat_provider, edge_logits

    def compute_loss(self, data, lambda_attr=1.0, lambda_struct=1.0):
        x_hat_m, x_hat_p, edge_logits = self.forward(data)
        
        # Attribute reconstruction
        loss_attr = (
            F.mse_loss(x_hat_m, data['member'].x) + 
            F.mse_loss(x_hat_p, data['provider'].x)
        )
        
        # Structural reconstruction (BCEWithLogitsLoss)
        edge_target = torch.ones_like(edge_logits)
        loss_struct = F.binary_cross_entropy_with_logits(edge_logits, edge_target)
        
        return lambda_attr * loss_attr + lambda_struct * loss_struct

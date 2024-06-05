import torch.nn as nn
import numpy as np
from torch_geometric.nn import GATConv
import torch

import time
from torch import optim
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GATConv, GATv2Conv, PNAConv
import torchvision
from sklearn import metrics
import os
from pathlib import Path
from torch.nn import Embedding, Linear

device = 'cuda' if torch.cuda.is_available() else 'cpu'

import torch
from torch_geometric.nn import to_hetero, Linear


class ItemTransformerEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, heads, dropout, dim_mlp):
        super().__init__()
        # Assume that the input features dimensions are known (in_channels)
        self.encoder = GATTransformerEncoder(dim=in_channels, heads=heads, dim_edge=5,
                                             dropout=dropout, dim_mlp=dim_mlp, depth=2)

    def forward(self, x, edge_index):
        # Prepare data for transformer input
        # data = (x, edge_index, None)  # No edge attributes
        return self.encoder(x, edge_index)

class UserTransformerEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, heads, dropout, dim_mlp):
        super().__init__()
        self.encoder1 = GATTransformerEncoder(dim=in_channels, heads=heads, dim_edge=5,
                                             dropout=dropout, dim_mlp=dim_mlp, depth=1)
        self.encoder2 = GATTransformerEncoderHeterogenous(in_dim=(in_channels, in_channels),hid_dim=in_channels, out_dim=out_channels, heads=heads, dim_edge=5,
                                             dropout=dropout, dim_mlp=dim_mlp, depth=1)
        self.encoder3 = GATTransformerEncoderHeterogenous(in_dim=(in_channels, in_channels),hid_dim=in_channels, out_dim=out_channels, heads=heads, dim_edge=5,
                                             dropout=dropout, dim_mlp=dim_mlp, depth=1)

    def forward(self, x_dict, edge_index_dict):
        product_x = self.encoder1(
            x_dict['product'],
            edge_index_dict[('product', 'metapath_0', 'product')],
        )

        customer_x = self.encoder2(
            (x_dict['product'], x_dict['customer']),
            edge_index_dict[('product', 'rev_buys', 'customer')],
        )

        customer_x = self.encoder3(
            (product_x, customer_x),
            edge_index_dict[('product', 'rev_buys', 'customer')],
        )
        return customer_x
    
class EdgeDecoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.lin1 = Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)

    def forward(self, z_src, z_dst, edge_label_index):
        row, col = edge_label_index
        z = torch.cat([z_src[row], z_dst[col]], dim=-1)

        z = self.lin1(z).relu()
        z = self.lin2(z)
        return z

class MetaTransformerGat(torch.nn.Module):
    def __init__(self, num_customers, hidden_channels, out_channels):
        super().__init__()
        self.customer_emb = Linear(5, hidden_channels)
        self.product_emb = Linear(9, hidden_channels)
        self.customer_encoder = UserTransformerEncoder(hidden_channels, out_channels, heads=4, dropout=0.1, dim_mlp=hidden_channels)
        self.item_encoder = ItemTransformerEncoder(hidden_channels, out_channels, heads=4, dropout=0.1, dim_mlp=hidden_channels)
        self.decoder = EdgeDecoder(out_channels)

    def forward(self, x_dict, edge_index_dict, edge_label_index, *args, **kwargs):
        x_dict['customer'] = self.customer_emb(x_dict['customer'])
        x_dict['product'] = self.product_emb(x_dict['product'])
        z_dict = {
            'customer': self.customer_encoder(x_dict, edge_index_dict),
            'product': self.item_encoder(
                x_dict['product'],
                edge_index_dict[('product', 'metapath_0', 'product')],
            )}
        return self.decoder(z_dict['customer'], z_dict['product'], edge_label_index)


class ResidualConnection(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PostAttentionMultiHeadProjection(nn.Module):
    def __init__(self, dim, heads, fn):
        super().__init__()
        self.linear = nn.Linear(dim * heads, dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.linear(self.fn(x, **kwargs))


class GraphEncoderMLP(nn.Module):

    def __init__(self, in_dim, hid_dim, out_dim, dropout=0.2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim, out_dim),
            nn.Dropout(dropout)
        )

    def forward(self, data):
        return self.mlp(data)


class PreLayerNorm(nn.Module):
    def __init__(self, layer_dim, func):
        super().__init__()
        self.func = func
        self.l_norm = nn.LayerNorm(layer_dim)

    def forward(self, data, **kwargs):
        return self.func(self.l_norm(data), **kwargs)


class GATTransformerEncoderHeterogenous(nn.Module):

    def __init__(self, in_dim, hid_dim, out_dim, heads, dim_edge, dropout, dim_mlp, depth=4):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                ResidualConnection(
                    PreLayerNorm(hid_dim, PostAttentionMultiHeadProjection(dim=hid_dim, heads=heads,
                                                                       fn=GATv2Conv(in_channels=in_dim, out_channels=out_dim,
                                                                                    heads=heads, add_self_loops=False,
                                                                                    edge_dim=dim_edge,
                                                                                    dropout=dropout)))),
                ResidualConnection(
                    PreLayerNorm(hid_dim, GraphEncoderMLP(in_dim=hid_dim, hid_dim=dim_mlp, out_dim=out_dim, dropout=dropout)))
            ]))

    # def forward(self, data):
    def forward(self, x, edge_index):
        for att, ff in self.layers:
            x = att(x=x, edge_index=edge_index)
            x = ff(x=x)
        return x


class GATTransformerEncoder(nn.Module):

    def __init__(self, dim, heads, dim_edge, dropout, dim_mlp, depth=4):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                ResidualConnection(
                    PreLayerNorm(dim, PostAttentionMultiHeadProjection(dim=dim, heads=heads,
                                                                       fn=GATv2Conv(in_channels=dim, out_channels=dim,
                                                                                    heads=heads, add_self_loops=False,
                                                                                    edge_dim=dim_edge,
                                                                                    dropout=dropout)))),
                ResidualConnection(
                    PreLayerNorm(dim, GraphEncoderMLP(in_dim=dim, hid_dim=dim_mlp, out_dim=dim, dropout=dropout)))
            ]))

    # def forward(self, data):
    def forward(self, x, edge_index):
        # x, edge_index, edge_attr = data
        # print(x)
        # print(x.shape)
        # print(edge_index)
        # print(edge_index.shape)
        for att, ff in self.layers:
            # x = att(x=x, edge_index=edge_index, edge_attr=edge_attr)
            x = att(x=x, edge_index=edge_index)
            # x = att(x=x, edge_index=edge_index)
            x = ff(x=x)
        return x


class PNATransformerEncoder(nn.Module):

    def __init__(self, dim=None, dim_edge=None, dropout=0.0, dim_mlp=None, deg=None, aggregators=None, scalers=None, towers=4, depth=4,
                 divide_input=False, pre_layers=1, post_layers=1):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.aggregators = aggregators or ['mean', 'min', 'max', 'std']
        self.scalers = scalers or ['identity', 'amplification', 'attenuation']
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                ResidualConnection(
                    PreLayerNorm(dim, PNAConv(in_channels=dim, out_channels=dim,
                                              aggregators=self.aggregators,
                                              scalers=self.scalers,
                                              towers=towers,
                                              edge_dim=dim_edge,
                                              dropout=dropout,
                                              deg=deg,
                                              divide_input=divide_input,
                                              pre_layers=pre_layers,
                                              post_layers=post_layers))),
                ResidualConnection(
                    PreLayerNorm(dim, GraphEncoderMLP(in_dim=dim, hid_dim=dim_mlp, out_dim=dim, dropout=dropout)))
            ]))

    # def forward(self, data):
    def forward(self, x, edge_index):
        # x, edge_index, edge_attr = data
        # x, edge_index, edge_attr = data
        for att, ff in self.layers:
            x = att(x=x, edge_index=edge_index)
            x = ff(x=x)
        return x
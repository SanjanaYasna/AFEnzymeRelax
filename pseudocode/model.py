import os
import os.path as osp
from math import pi as PI
from math import sqrt
from typing import Callable, Union

import numpy as np
import torch
from torch.nn import Embedding, Linear, Dropout
from torch_scatter import scatter
from torch_sparse import SparseTensor

from torch_geometric.data import Dataset, download_url
from torch_geometric.data.makedirs import makedirs
from torch_geometric.nn import radius_graph
from torch_geometric.nn.inits import glorot_orthogonal
from torch_geometric.nn import global_mean_pool
from torch.nn import LazyLinear
from torch.nn import Linear


"""
Embedds the nodes in the graph 

Inputs: 
one_hot_dim: number of nodes/one-hot encodings
hidden_channels: number of hidden channels
"""
class NodeEmbeddingBlock(torch.nn.Module):
    def __init__(self, one_hot_dim, hidden_channels):
        super().__init__()
        #self.emb = Embedding(num_atoms, hidden_channels)
        self.emb = Linear(one_hot_dim, hidden_channels)
        self.lin_angle_geom = Linear(3, hidden_channels)
        self.lin_angle_final = LazyLinear(hidden_channels)
        self.lin_rbf = LazyLinear(hidden_channels * 2)
        self.lin = Linear(3 * hidden_channels, hidden_channels)
        self.act = torch.nn.ReLU()
        self.reset_parameters()

    def reset_parameters(self):
        #self.emb.weight.data.uniform_(-sqrt(3), sqrt(3))
        self.lin_rbf.reset_parameters()
        self.lin.reset_parameters()
        
    #distance-based RBF, 6 functions by default (change D_count to change number of functions)
    def rbf(D):
        # Distance radial basis function
        D_min, D_max, D_count = 0., 20., 6
        D_mu = torch.linspace(D_min, D_max, D_count).to(D.device)
        D_mu = D_mu[None,:]
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        RBF = torch.exp(-((D_expand - D_mu) / D_sigma)**2)
        return RBF
    """
    Inputs:
    x: node features
    angle_geom: angle geometry 3 value planar features
    pos: node positions (CA_coords)
    Output:
    Node embeddings tensor (num_nodes, hidden_channels*4) 
    """
    def forward(self, x, angle_geom, pos):
        #get the rbf transformation
        init_xyz = pos.reshape(len(pos), 1, 3)
        rbf_feat = rbf(torch.cdist(init_xyz[:,0,:], init_xyz[:,0,:]))
        #embed one hot encodings
        x = self.emb(x) 
        angle_geom = self.act(self.lin_angle_geom(angle_geom))
        #flatten the last two dimensions together of angle_Geom
        angle_geom = self.lin_angle_final(angle_geom.view(angle_geom.size(0), -1))
        #flatten last two dimensions of rbf
        rbf = rbf_feat.reshape(rbf_feat.size(0), -1)
        rbf = self.lin_rbf(rbf)
        """
        Dimensions: 
        x = (num_nodes, hidden_channels)
        angle_geom = (num_nodes, hidden_channels)
        rbf = (num_nodes, hidden_channels * 2)
        """
        embs = self.act(torch.cat([x, angle_geom, rbf], dim=-1))
        
        return embs
    
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
    
    def forward(self, x: torch.Tensor, ca_coords: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        '''
        triplet set = triplets(edge_index, num_nodes)
        #calculate distances
        distances = torch.norm(ca_coords[triplet_set[2]] - ca_coords[triplet_set[3]], dim=1)
        #calculate angles
        angles = torch.acos((torch.sum((ca_coords[triplet_set[2]] - 
        ca_coords[triplet_set[3]]) * (ca_coords[triplet_set[4]] - 
        ca_coords[triplet_set[3]]), dim=1) / (distances * distances)))
        
        #apply rbf transofrmation
        rbf = self.rbf(distances)
        #embeding block of nodes from rbf
        x = self.lin(x, rbf, edge_triplets)
        #interaction block form DimeNet
        '''
        return self.lin(x)
    def 
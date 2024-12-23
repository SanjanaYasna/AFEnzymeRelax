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
from torch_geometric.nn.models.dimenet import SphericalBasisLayer, BesselBasisLayer
from dimenetpp_adjusted import InteractionBlock
"""
Embedds the nodes in the graph 

Inputs: 
one_hot_dim: number of nodes/one-hot encodings
hidden_channels: number of hidden channels
"""
class NodeEmbeddingBlock(torch.nn.Module):
    def __init__(self, one_hot_dim, hidden_channels):
        super().__init__()
        self.emb = Linear(one_hot_dim, hidden_channels)
        self.lin_angle_geom = Linear(3, hidden_channels)
        self.lin_angle_final = LazyLinear(hidden_channels)
        self.lin_rbf = LazyLinear(hidden_channels * 2)
        self.lin = Linear(3 * hidden_channels, hidden_channels)
        self.act = torch.nn.ReLU()
        self.reset_parameters()
        self.rbf = self.rbf

    def reset_parameters(self):
        #self.emb.weight.data.uniform_(-sqrt(3), sqrt(3))
        self.lin_rbf.reset_parameters()
        self.lin.reset_parameters()
        
    #distance-based RBF, 6 functions by default (change D_count to change number of functions)
    def rbf(self, D):
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
    ca_coords: node positions (CA_coords)
    
    Example extraction that should work:
    node_one_hot = torch.tensor([att["x"] for node, att in graph.nodes(data=True)])
    pos = torch.tensor([att["ca_coords"] for node, att in graph.nodes(data=True)])
    edge_index = torch.LongTensor(list(graph.edges)).t().contiguous()
    
    Output:
    Node embeddings tensor (num_nodes, hidden_channels*4) 
    """
    def forward(self, x, angle_geom, pos):
        #get the rbf transformation
        init_xyz = pos.reshape(len(pos), 1, 3)
        rbf_feat = self.rbf(torch.cdist(init_xyz[:,0,:], init_xyz[:,0,:]))
        #embed one hot encodings
        #error: expected scalar type Long but found Float...
        x = x.float()

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
        embs = self.act(torch.cat((x, angle_geom, rbf), -1))
        return embs
    
class InitialInteraction(torch.nn.Module):
    def __init__(self, hidden_channels, one_hot_dim = 22,
                num_spherical=7, 
                num_radial=6, 
                cutoff = 5.0,
                envelope_exponent=5,
                num_before_skip=1,
                num_after_skip=2, 
                #num_output_layers=3, 
                num_bilinear= 2,
                interaction_layers = 3,
                act = torch.nn.ReLU()):
        super(InitialInteraction, self).__init__()
        #calls the node embedding module to embedd the nodes to have overall dimension of embedding hidden_channels*4
        self.node_embeddings = NodeEmbeddingBlock(one_hot_dim, hidden_channels)
        self.rbf_emb = BesselBasisLayer(
            num_radial=num_radial, cutoff=cutoff, envelope_exponent=envelope_exponent)
        self.sbf_layer = SphericalBasisLayer(num_spherical = num_spherical,
            num_radial=num_radial,
            cutoff = cutoff,
            envelope_exponent=envelope_exponent)
        #interaction block needs to know num_nodes to output its [num_nodes, hidden_channels] tensor without issues with einsum
        self.interaction_block =  InteractionBlock(
            hidden_channels= hidden_channels*4,
            num_bilinear= num_bilinear,
            num_spherical= num_spherical,
            num_radial=num_radial,
            num_before_skip= num_before_skip,
            num_after_skip= num_after_skip,
            act= act
            )
        self.interaction_layers = interaction_layers
        self.act = act
        self.dropout = Dropout(0.1)
        
    #returns indices along rows and columsn, triplet distances and angles, etc...lots of stuff
    def triplets(self, edge_index, num_nodes):
        # edge_index = torch.tensor(list(edge_index)).t().contiguous()
        row, col = edge_index  # j->i
        value = torch.arange(row.size(0), device=row.device)
        adj_t = SparseTensor(row=col, col=row, value=value,
                                sparse_sizes=(num_nodes, num_nodes))
        adj_t_row = adj_t[row]
        num_triplets = adj_t_row.set_value(None).sum(dim=1).to(torch.long)

        # Node indices (k->j->i) for triplets.
        idx_i = col.repeat_interleave(num_triplets)
        idx_j = row.repeat_interleave(num_triplets)
        idx_k = adj_t_row.storage.col()
        mask = idx_i != idx_k  # Remove i == k triplets.
        idx_i, idx_j, idx_k = idx_i[mask], idx_j[mask], idx_k[mask]

        # Edge indices (k-j, j->i) for triplets.
        idx_kj = adj_t_row.storage.value()[mask]
        idx_ji = adj_t_row.storage.row()[mask]
        return col, row, idx_i, idx_j, idx_k, idx_kj, idx_ji
    
    """
    Same input as NodeEmbeddingBlock.forward
    Inputs:
    x: node features #will result in x_orig, but x itself used below is just the node positions relative to each other in sequence, so a set of scalars... 
    angle_geom: angle geometry 3 value planar features
    pos: node positions (CA_coords)
    edge_index: should be converted as  torch.LongTensor(list(graph.edges)).t().contiguous()
    
    Example conversions that should work from networkx graph:
    node_one_hot (as x) = torch.tensor([att["x"] for node, att in graph.nodes(data=True)])
    pos = torch.tensor([att["ca_coords"] for node, att in graph.nodes(data=True)])
    edge_index = torch.LongTensor(list(graph.edges)).t().contiguous()
    angle_geom = torch.tensor([att["angle_geom"] for node, att in graph.nodes(data=True)])
    """
    
    #take in list of torch tensors 
    #after running through entire set, run forward and backwards after
    #batch != parallel
    #can apply scan and distribute functions on what must be universally distributed... 
    
    def forward(self, x: torch.Tensor, angle_geom: torch.Tensor, ca_coords: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
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
        #get node embeddings
        x_orig = self.node_embeddings(x, angle_geom, ca_coords)
        #get triplet pairs and indices
        i, j, idx_i, idx_j, idx_k, idx_kj, idx_ji = self.triplets(
                    edge_index, len(x_orig))
        
        # Calculate distances.
        dist = (ca_coords[i] - ca_coords[j]).pow(2).sum(dim=-1).sqrt()
        # Calculate angles.
        pos_i = ca_coords[idx_i]
        pos_ji, pos_ki = ca_coords[idx_j] - pos_i, ca_coords[idx_k] - pos_i
        a = (pos_ji * pos_ki).sum(dim=-1)
        b = torch.cross(pos_ji, pos_ki).norm(dim=-1)
        angle = torch.atan2(b, a)

        #get rbf and sbf embeddings 
        rbf = self.rbf_emb(dist)
        sbf = self.sbf_layer(dist, angle, idx_kj)
        
        P = x_orig
        #iterate interaction_layer times
        for _ in range(self.interaction_layers):
            interaction_weights = self.interaction_block(
                x_orig, rbf, sbf, idx_kj,  i)
            P += interaction_weights
        P = self.dropout(self.act(P))
        #return node embeddings from result of their interactions
        return  P
import torch
from torch.nn import Sigmoid
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn import conv
from torch import Tensor
from torch_geometric.utils import k_hop_subgraph

from torch_geometric.typing import OptTensor
from typing import Callable, Optional, Tuple, Union
from torch_geometric.utils import softmax
from torch.nn import ReLU
from utils.rpn_conv_only import GraphRPNConvOnly
from utils.relational_module import NodeEmbeddingBlock

class MainModel(torch.nn.Module): 
    def __init__(self, 
                # needed for node embedding block, intitial interaction, and rpn
                hidden_channels, 
                #one_hot_dim for node embedding block
                one_hot_dim =23,
                #used in graphrpn
                num_classes = 1, 
                conv_layers = 5,
                k = 2,
                grad_cam = True,
                perturbed = True
                ):
        super(MainModel, self).__init__()
        #initial interaction encompasses node-embedding block
        self.perturbed = perturbed
        self.node_embedding = NodeEmbeddingBlock(one_hot_dim, hidden_channels)
        
        self.rpn = GraphRPNConvOnly(hidden_dim=hidden_channels*4, #INITIAL INTERACTION OUTPUTS 4*HIDDEN_CHANNELS, SO RPN HAS THIS VALUE AS A RESULT 
                            num_classes=num_classes,
                            k=k,
                            conv_layers=conv_layers,
                            grad_cam=grad_cam
                            )
    def forward(self, x: torch.Tensor, angle_geom: torch.Tensor, ca_coords: torch.Tensor, edge_index: torch.Tensor, batch: list):
        x = self.node_embedding(x, angle_geom, ca_coords)
        print(x.shape)
        node_scores, node_list, func_probability, x = self.rpn(x, edge_index, batch)
        return node_scores, node_list, func_probability, x
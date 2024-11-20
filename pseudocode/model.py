import torch
from torch.nn import Sigmoid
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn import conv
from torch import Tensor
from torch_geometric.utils import k_hop_subgraph

from torch_geometric.typing import OptTensor
from typing import Callable, Optional, Tuple, Union
from torch_geometric.utils import softmax

from gcn_net import GraphRPN


class MainModel(torch.nn.Module): 
    def __init__(self, 
                 # needed for node embedding block
                 one_hot_dim,
                 hidden_channels, 
                 #2 required graphrpn params
                input_dim, 
                 hidden_dim, 
                 # initial interactoins (using adjusted dimenetpp)
                 num_nodes,
                 num_spherical=7, 
                 num_radial=6, 
                 cutoff = 5.0,
                 envelope_exponent=5,
                 num_before_skip=1,
                 num_after_skip=2, 
                 num_bilinear= 2,
                 interaction_layers = 3,
                 act = torch.nn.ReLU(),
                 #used in graphrpn
                 num_classes = 1, 
                 k = 2,
                 grad_cam = True
                 ):
        super(MainModel, self).__init__()
    pass
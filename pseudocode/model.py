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
from relational_module import InitialInteraction
from CL import OutputPred
class MainModel(torch.nn.Module): 
    def __init__(self, 
                # needed for node embedding block, intitial interaction, and rpn
                hidden_channels, 
                #one_hot_dim for node embedding block
                one_hot_dim =22,
                #initial interaction block params
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
                grad_cam = True,
                #for output_pred, num unique classses default 8 
                pred_classes =8,
                perturbed = True
                ):
        super(MainModel, self).__init__()
        #initial interaction encompasses node-embedding block
        self.perturbed = perturbed
        self.initial_interaction = InitialInteraction(hidden_channels, one_hot_dim = one_hot_dim,
                                    num_spherical=num_spherical, 
                                    num_radial=num_radial, 
                                    cutoff = cutoff,
                                    envelope_exponent=envelope_exponent,
                                    num_before_skip=num_before_skip,
                                    num_after_skip=num_after_skip, 
                                    num_bilinear= num_bilinear,
                                    interaction_layers = interaction_layers,
                                    act = act)
        
        self.rpn = GraphRPN(hidden_dim=hidden_channels*4, #INITIAL INTERACTION OUTPUTS 4*HIDDEN_CHANNELS, SO RPN HAS THIS VALUE AS A RESULT 
                            num_classes=num_classes,
                            k=k,
                            grad_cam=grad_cam
                            )
        
        self.OutputPred = OutputPred(hidden_dim=hidden_channels*4, num_classes = pred_classes)
    def forward(self, x: torch.Tensor, angle_geom: torch.Tensor, ca_coords: torch.Tensor, edge_index: torch.Tensor, batch: list):
        x = self.initial_interaction(x, angle_geom, ca_coords, edge_index)
        node_list, func_probability, x = self.rpn(x, edge_index, batch)
        return node_list, func_probability, x
        # if self.perturbed:
        #     #you get series of predictions for each of the pred_classes
        #     x, x2, pred = self.OutputPred(x, batch, perturbed = True)
        #     return x, x2, pred, func_probability, node_list
        # else:
        #     x =  self.OutputPred(x, batch, perturbed = False)
        #     return x, func_probability, node_list
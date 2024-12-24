import torch
from torch.nn import Sigmoid
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn import conv
from torch import Tensor
from torch_geometric.utils import k_hop_subgraph
import numpy as np
from torch_geometric.typing import OptTensor
from typing import Callable, Optional, Tuple, Union
from torch_geometric.utils import softmax
from utils.gcn_net_blocks import FunctionalResiduePredUnit, CustomNodeDropPoolingLayer

"""Convolutional layers, meant to go with the version of model without dimenet. Main distinction is that
Here, layers and k are separate parameters since now we're liekly using more layers than the number of subgraph hops we'd want to do"""
class GraphRPNConvOnly(torch.nn.Module):
    """Graph RPN Model: A GNN model with NO PRUNING and a functionality prediction unit (that is just attention based)
    for ego labels """
    #CURRENTLY THERE IS NO DOWNSCALING NOR UPSCALING IN HIDDEN DIMENSIONS FOR GCN AND GAT
    def __init__(self, hidden_dim, 
                 num_classes = 1, 
                 conv_layers = 5,
                 k = 2,
                 grad_cam = True):
        """_summary_

        Args:
            k (_type_): number of layers of GCNs. 2 has been found sufficient prior to GAT layer
            input_dim (_type_): batch size 
            hidden_dim (_type_): dimension of each node embedding 
            num_classes (_type_): for now, just functional or not? so just 1 by default
        """
        super(GraphRPNConvOnly, self).__init__()
        self.k_layer_gcn = torch.nn.ModuleList(
            [GCNConv(hidden_dim, hidden_dim) for i in range(conv_layers)]
        )
        self.functional_residue_prediction_unit = FunctionalResiduePredUnit(k, hidden_dim, hidden_dim, num_classes)
        #GAT is used as a final predictoin unit for functionality per node
        self.functionality_prediction_unit = GATConv(hidden_dim, num_classes)
        self.sigmoid = Sigmoid()
        self.grad_cam = grad_cam
#NOTE: RECONSIDER FLOAT CONVERSIONS (TRY T0 KEEP TO ONE KIND OF FLOAT)
    def forward(self, x, edge_index, batch = None):
        """_summary_

        Args:
            data (_type_): _description_


        Returns:
            node_scores_list = list of node scores for each ego graph
            node_list = nodes as part of ego graph that are predicted
            func_probability = probability of functionality, this aught to be visualized iwth grad-cam
            x = final node embeddings after all layers of GCN
        """
        #x_orig is cloned to avoid in-place operation errors when putting it through gcn
        x_orig = x.detach().clone()
        # Apply GCN layers
        for gcn in self.k_layer_gcn:
            #x = x.to(torch.float32)
            
            x = gcn(
                x=x, edge_index=edge_index
            )  #### TODO: DOES NOT SUPPORT BATCHING NEED TO CHANGE (now, single samples...batched later)
            # print("printing shape of x after gcn layer", x.shape)
            
            
        # # Graph functional label prediction 
        node_scores, node_list = self.functional_residue_prediction_unit(
            x, x_orig, edge_index, batch
        )
        #TODO: IMPLEMENT GRAD-CAM BETWEEN THE TWO TO VISUALIZE THE ATTENTION FROM PENULTIMATE LAYER OF GCN
        if self.grad_cam:
            pass
        
        
        functionality_logits = self.functionality_prediction_unit(x, edge_index)
        
        func_probability = self.sigmoid(functionality_logits)
        #the idea is by controlling the loss of hte node list in corretly identifying the funcitonal regions, you'd have better loss in your attention layer helping the functionallity logits identify ego labels
        #BCE loss between node_list and data.y (where there is a match)
        #BCE loss between func_probaibilty nad data.ego_label 
        return node_scores, node_list, func_probability, x
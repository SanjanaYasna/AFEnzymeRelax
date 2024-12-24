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

"""
Currently no node-drop prenumbral cone attention implemented. Right now, 
NO NODES ARE DROPPED (a bit of a misnomer since future work will maybe drop it)
Just a regular pooling layer for now. May as well even just call a run-of-the-mill
GCN pooling layer instead.
"""
class CustomNodeDropPoolingLayer(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        GNN: torch.nn.Module,  # The GNN layer class should be passed during initialization
        **kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        # Initialize the GNN layer for computing node scores
        self.gnn = GNN(in_channels, 1, **kwargs)
        self.reset_parameters()

    def reset_parameters(self):
        self.gnn.reset_parameters()

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        batch: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor], Optional[Tensor], Tensor, Tensor]:
        if batch is None:
            batch = edge_index.new_zeros(x.size(0), dtype=torch.long)

        # Use the GNN to compute raw scores for each node
        scores = self.gnn(x, edge_index)

        # Apply softmax on the scores within each graph in the batch
        scores = softmax(scores, batch)

        # For this example, no nodes are actually dropped; all are retained.
        # In a practical implementation, you might apply a threshold or selection mechanism here.
        perm = torch.arange(x.size(0), dtype=torch.long, device=x.device)

        # Update `x`, `edge_index`, and potentially `edge_attr` if necessary
        # Here, we assume no actual pooling (dropping of nodes) is performed,
        # so `x`, `edge_index`, and `edge_attr` remain unchanged.
        # TODO: ADD DROPOUT WITH PRUNING RUNS IN FUTURE

        # Return the modified x, edge_index, optional edge_attr, batch vector, perm, and scores
        return x, edge_index, edge_attr, batch, perm, scores

class FunctionalResiduePredUnit(torch.nn.Module):
    """Pruning unit to process subgraph anchors (ego graphs) and predict nodes 

    Returns:
        torch (List[Tensors], List[Tensors]): list of node score tensors for each ego graph, and nodes as part of ego graph for one batch
    """

    def __init__(self, k, input_dim, hidden_dim, output_dim):
        """_summary_
        hidden_dim = 128 for example
        input_dim = 1280 (esm_embedding size)
        output_dim = 1 (for functional prediction)
        Args:
            k (_type_):
            input_dim (_type_): _description_
            output_dim (_type_): _description_
        """
        super(FunctionalResiduePredUnit, self).__init__()
        self.k = k
        self.conv1 = GATConv(hidden_dim, input_dim, heads=1)

#possibiltiy alternative pooling with SAG?

        #currently has NO node drop
        self.pool_layer = CustomNodeDropPoolingLayer(
            input_dim,
            GNN=GATConv,
            # nonlinearity=torch.nn.LeakyReLU(0.1),
        )

        self.linear = torch.nn.Linear(input_dim, output_dim)
        self.sigmoid = Sigmoid()

#SUBGRAPHS NUMBERED FROM 0!!!
    def forward(self, x, x_orig, edge_index, batch):
        """features used are original esm embeddings of each of the

        Args:
            x (_type_): _description_
            edge_index (_type_): _description_
            batch (_type_): _description_

        Returns:
            _type_: _description_
        """
        x = self.conv1(
            x, edge_index
        )  # INCREasing dimension from hidden dim to input dim
        subgraph_list = []
        # subgraph_batch_list = []
        ego_nodes = []
        # Iterate over all unique graphs in the batch
        for batch_num in batch.unique():
            # Select nodes belonging to the current graph
            nodes = (batch == batch_num).nonzero().squeeze()
            for node in nodes:
                # print("Anchor node", node)
                # Extract the k-hop subgraph for the current node

                subset, edge_index_sub, _, _ = k_hop_subgraph(
                    node_idx=node.item(),
                    num_hops=self.k,
                    edge_index=edge_index,
                    relabel_nodes=True,
                    num_nodes=None if batch is None else batch.size(0),
                )
                # Create subgraph batch
                subgraph_batch = torch.zeros(
                    subset.size(0), dtype=torch.long, device=edge_index.device
                )

                # Get node features for the subgraph
                x_sub = x_orig[subset]
                x_sub, edge_index_sub, _, subgraph_batch, perm, score_sub = (
                    self.pool_layer(
                        x=x_sub, edge_index=edge_index_sub, batch=subgraph_batch
                    )
                )
                #result is a collection of anchors that are believed to have the most possible fucntional residues 
                #score_sub = self.sigmoid(score_sub)
                ego_nodes.append(subset[perm])  # original nodes
                # Store processed subgraphs
                assert len(score_sub) == len(
                    perm
                )  ## node scores length should be equal to number of nodes in subgraph
                assert len(perm) == len(subgraph_batch)
                subgraph_list.append(score_sub)
        #         subgraph_batch_list.append(subgraph_batch)
        # x_batched, batch_mapping = to_dense_batch(torch.cat(subgraph_list, dim=0), batch=torch.cat(subgraph_batch_list, dim=0))
        return (
            subgraph_list,
            ego_nodes
        )  # returns list of variable sized tensors with node scores for each ego graph, and nodes as part of ego graph for one batch

"""
TODO : Implement GRAD-CAM layer visualizations  + output functionality scores per node
TODO : add a heal-inspired transformer component after the message passing layers 
"""
class GraphRPN(torch.nn.Module):
    """Graph RPN Model: A GNN model with NO PRUNING and a functionality prediction unit (that is just attention based)
    for ego labels """
    #CURRENTLY THERE IS NO DOWNSCALING NOR UPSCALING IN HIDDEN DIMENSIONS FOR GCN AND GAT
    def __init__(self, hidden_dim, 
                 num_classes = 1, 
                 k = 2,
                 grad_cam = True):
        """_summary_

        Args:
            k (_type_): number of layers of GCNs. 2 has been found sufficient prior to GAT layer
            input_dim (_type_): batch size 
            hidden_dim (_type_): dimension of each node embedding 
            num_classes (_type_): for now, just functional or not? so just 1 by default
        """
        super(GraphRPN, self).__init__()
        self.k_layer_gcn = torch.nn.ModuleList(
            [GCNConv(hidden_dim, hidden_dim) for i in range(k)]
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
            x = x.to(torch.float32)
            
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
        
        #get node_score means among the anchor node indices
        score_mean = torch.zeros(len(x))
        node_list_counts = torch.zeros(len(x))
        for list in node_list:
            node_list_counts[list] += 1
        for score, list in zip(node_scores, node_list):
            score_mean[list] += score.squeeze()
        score_mean = torch.div(score_mean, node_list_counts) 
        
        functionality_logits = self.functionality_prediction_unit(x, edge_index)
        
        func_probability = self.sigmoid(functionality_logits)
        #the idea is by controlling the loss of hte node list in corretly identifying the funcitonal regions, you'd have better loss in your attention layer helping the functionallity logits identify ego labels
        #BCE loss between node_list and data.y (where there is a match)
        #BCE loss between func_probaibilty nad data.ego_label 
        return score_mean, node_scores, node_list, func_probability, x
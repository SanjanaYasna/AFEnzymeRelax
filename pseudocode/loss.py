import torch
from torch.nn.functional import binary_cross_entropy as bce_loss

#node list is indexed from 0, and so should target_res
#so now, see where the node list indices differ from ground truth indices
def custom_bce_loss_node_list(node_list, target_res):
    #get all unique tensor indices in node_list and make that a new tensor of length target_res
    #get the indices of the unique nodes in node_list
    #make equivlaent tensor from all these subgraph predictions
    equivalent = torch.zeros(len(target_res))
    #fill with ones at total_unique_preds as indices
    for val in node_list:
        equivalent[val] = 1
    target_res = target_res.float()
    mse = bce_loss(equivalent, target_res, reduction="mean")
    return mse
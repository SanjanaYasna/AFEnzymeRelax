import torch
from torch.nn.functional import binary_cross_entropy as bce_loss
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix
from sklearn.metrics import f1_score as sklearn_f1_score
import numpy as np
from torch.nn import functional as F
#node list is indexed from 0, and so should target_res
#so now, see where the node list indices differ from ground truth indices
# pred_nodes= list(compress(node_scores, target_res))
# pred_nodes = [x for xs in pred_nodes for x in xs]
def custom_bce_loss_node_list(node_scores, node_list, target_res):
    mse_losses = []
    for pred_tensor, node_tensor in zip(node_scores, node_list):
        target_subset = target_res[node_tensor]
        target_subset = target_subset.unsqueeze(1)
        # pred_tensor = pred_tensor.squeeze(-1)
        mse = bce_loss(pred_tensor, target_subset, reduction="mean")
        mse_losses.append(mse)
    # Calculate the average BCE over all tensors in the batch
    batch_loss = torch.mean(torch.stack(mse_losses))
    return batch_loss

def bce_loss_weighted_node_list(node_scores, node_list, target_res, weight):
    mse_losses = []
    for pred_tensor, node_tensor in zip(node_scores, node_list):
        target_subset = target_res[node_tensor]
        target_subset = target_subset.unsqueeze(1)
        # pred_tensor = pred_tensor.squeeze(-1)
        mse = weighted_bce(input = pred_tensor, target = target_subset, weight = weight)
        if mse == 0:
            continue
        mse_losses.append(mse)
    # Calculate the average BCE over all tensors in the batch
    batch_loss = torch.mean(torch.stack(mse_losses))
    
    print("batch loss", batch_loss)
    return batch_loss

def focal_loss_node_list(node_scores, 
                         node_list, 
                         target_res, 
                         reduction, 
                         alpha=0.25, gamma=2):
    mse_losses = []
    for pred_tensor, node_tensor in zip(node_scores, node_list):
        target_subset = target_res[node_tensor]
        target_subset = target_subset.unsqueeze(1)
        mse = focal_loss(input = pred_tensor, 
                        target = target_subset, 
                        reduction = reduction, 
                        alpha = alpha, 
                        gamma = gamma)
        mse_losses.append(mse)
    # Calculate the average BCE over all tensors in the batch
    batch_loss = torch.mean(torch.stack(mse_losses))
    print("batch loss", batch_loss)
    return batch_loss

def compute_f1_score(pred_list, node_list, data_y):
    """
    Computes the F1 score for the entire batch.

    Args:
        pred_list (List[torch.Tensor]): List of node score tensors for each ego graph.
        node_list (List[torch.Tensor]): List of nodes as part of ego graph for one batch.
        data_y (torch.Tensor): Ground truth for entire batch.

    Returns:
        float: F1 score for the entire batch.
    """
    y_true = []
    y_pred = []
    for pred_tensor, node_tensor in zip(pred_list, node_list):
        gt_tensor = data_y[node_tensor]
        pred_tensor = pred_tensor > 0.5

        y_true.extend(gt_tensor.cpu().numpy().tolist())
        y_pred.extend(pred_tensor.cpu().numpy().tolist())

    # Calculate F1 score using true positives, false positives, and false negatives
    f1 = sklearn_f1_score(y_true, y_pred, average="binary")

    return f1

def compute_confusion_matrix(pred_list, node_list, data_y):
    """
    Computes the confusion matrix for the entire batch.

    Args:
        pred_list (List[torch.Tensor]): List of node score tensors for ONLY true positive anchors.
        node_list (List[torch.Tensor]): List of nodes as part of ego graphs that are true positive anchors.
        data_y (torch.Tensor): Ground truth for entire batch.
        func_proba (torch.Tensor): Functional probability for each ego graph/anchor.

    Returns:
        np.array, np.array: confusion matrix for pruning and anchor classification
    """
    y_true = []
    y_pred = []
    for pred_tensor, node_tensor in zip(pred_list, node_list):
        gt_tensor = data_y[node_tensor]
        pred_tensor = pred_tensor > 0.5
        y_true.extend(gt_tensor.cpu().numpy().tolist())
        y_pred.extend(pred_tensor.cpu().numpy().tolist())
    confusion_matrix_pruning = sklearn_confusion_matrix(y_true, y_pred, labels=[0, 1])
    #divide confusion matrix by length of y_true
    confusion_matrix_pruning = confusion_matrix_pruning
    return confusion_matrix_pruning

"""A more ambitious anomaly GAL implementation (softmax activation preferred? even with binary case...)"""


"""SIMPLE OPTIONS"""

"""Focal loss... binary case only for now
where:
    pt = sigmoid(x) for binary classification, and softmax(x) for multi-class classification
    alpha = balancing parameter, default to 1, the balance between positive and negative samples
    gamma = focusing parameter, default to 2, the degree of focusing, 0 means no focusing
    Still not entirely sure about good params
"""
def focal_loss(input, target, reduction,  alpha=0.25, gamma=2):
    ce_loss = F.binary_cross_entropy_with_logits(input, target, reduction="none")
    #in rpns, focal loss typically normalized by number of ground truth anchors (if done on all anchors instead of heuristically sampled ones, like here)
    num_assigned_anchors = torch.sum(input)
    pt = torch.exp(-ce_loss)
    focal_loss = alpha * (1 - pt) ** gamma * ce_loss
    if reduction == "none":
        return focal_loss
    if reduction == "mean_pos":
        return focal_loss.sum() / num_assigned_anchors
    if reduction == "mean":
        return focal_loss.mean()
    else: #assume sum by default
        return focal_loss.sum()

"""weighted cross entropy, idfk
    weight typically is ratio of anomaly to normal nodes, in this case,
    data.y weight: 0.03
    ego label weight: 0.0142
"""
def weighted_bce(input, target, weight):
    eps = 1e-3
    loss = torch.sum(-torch.mean(target * weight * torch.log(input + eps)
                                + (1 - target) * torch.log(1 - input + eps)))
    #if loss is inf or -inf, return none
    if loss == float('inf') or loss == float('-inf'):
        return 0
    return loss



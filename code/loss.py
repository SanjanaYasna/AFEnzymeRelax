import torch
from torch.nn.functional import binary_cross_entropy as bce_loss
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix
from sklearn.metrics import f1_score as sklearn_f1_score
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
    confusion_matrix_pruning = sklearn_confusion_matrix(y_true, y_pred)
    #divide confusion matrix by length of y_true
    confusion_matrix_pruning = confusion_matrix_pruning / len(y_true)
    return confusion_matrix_pruning
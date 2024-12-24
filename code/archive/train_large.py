import numpy as np
import torch
from torch.nn.functional import binary_cross_entropy as bce_loss
from torch.optim import SGD, lr_scheduler
from torch_geometric.loader import DataLoader
from utils.dataloader import ProteinDataLoader
from utils.loss import custom_bce_loss_node_list, compute_confusion_matrix
from model import MainModel
from sklearn.metrics import f1_score as sklearn_f1_score
from torch.utils.tensorboard import SummaryWriter
#TODO : USE DATA PARALLEL (and look into allreduce sgd for efficiency)
#from torch.nn.parallel import DistributedDataParallel as DDP
#PARAMS TO CHANGE FOR TRAINING
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 64
num_epochs = 100
model = MainModel(8)
optimizer = SGD(model.parameters(), lr=0.1)
scheduler = lr_scheduler.CyclicLR(optimizer, base_lr=0.01, max_lr=0.1)
writer = SummaryWriter(f"/kuhpc/work/slusky/syasna_sta/func_pred/AFEnzymeRelax/full_rpn_runs/GRPN_region_pred_only")

# #data parellel option TODO
# if torch.cuda.device_count() > 1:
#     print("Let's use", torch.cuda.device_count(), "GPUks!")
#     # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
#     model = torch.nn.DataParallel(model)
model.to(device)

#TODO MODIFY AS NEEDED
dataloader = ProteinDataLoader("/kuhpc/scratch/slusky/syasna_sta/swissprot_protein_data/wildtype_pts"
                            )
testloader = ProteinDataLoader("/kuhpc/scratch/slusky/syasna_sta/swissprot_protein_data/wildtype_pts_test")
train_loader = DataLoader(dataloader, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(testloader, batch_size=batch_size, shuffle=True)
torch.autograd.set_detect_anomaly(True)

#loss files 
loss_file = open("/kuhpc/work/slusky/syasna_sta/func_pred/AFEnzymeRelax/full_rpn_runs/losses.txt", "w")

test_file = open("/kuhpc/work/slusky/syasna_sta/func_pred/AFEnzymeRelax/full_rpn_runs/test_losses.txt", "w")

for epoch in range(num_epochs+1):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    model.train(True)
    losses = []
    ego_losses = []
    data_y_losses = []
    confusion_matrix_sum = np.zeros((2, 2))
    for i, data in enumerate(train_loader):
        optimizer.zero_grad()
        score_mean, node_scores, node_list, func_probability, x = model.forward(data.x,
                                                                                data.angle_geom, 
                                                                                data.pos, 
                                                                                data.edge_index, 
                                                                                data.batch)
        target_res = data.y
        ego_labels = data.label_graphs
        #DATA.Y LOSS
        data_y_loss = custom_bce_loss_node_list(node_scores, node_list, target_res)

        #DATA.EGO_LABELS LOSS
        func_probability = torch.squeeze(func_probability)
        ego_label_loss = bce_loss(func_probability, ego_labels, reduction="mean")
        
        #total loss is sum of the two
        loss = (data_y_loss + ego_label_loss)
        losses.append(loss.item())
        ego_losses.append(ego_label_loss.item())
        data_y_losses.append(data_y_loss.item())
        with torch.no_grad():
            #f1 and confusion matrix metrics
            confusion_matrix = compute_confusion_matrix(node_scores, node_list, data.y)
            f1_data_y = sklearn_f1_score(
                data.y.cpu().numpy(),
                score_mean.detach().cpu().numpy() > 0.5, average="binary")
        #updates
        loss.backward()
        optimizer.step()
        
        #batch-level writer updates
        writer.add_scalar(
            "batch total loss", loss.item(), epoch * len(train_loader) + i
        )
        #add to confusion matrix
        confusion_matrix_sum += confusion_matrix
        writer.add_scalar(
            "batch f1 score of data.y", f1_data_y, epoch * len(train_loader) + i
        )
    #write to losses file
    loss_avg = sum(losses) / len(losses)
    ego_avg = sum(ego_losses) / len(ego_losses)
    data_y_avg = sum(data_y_losses) / len(data_y_losses)
    confusion_matrix_sum = confusion_matrix_sum / len(train_loader)
    loss_file.write(f"Epoch {epoch}\n")
    loss_file.write(f"Data y loss: {data_y_avg}\n")
    loss_file.write(f"Ego label loss: {ego_avg}\n")
    loss_file.write(f"total loss: {loss_avg}\n")
    loss_file.write(f"confusion matrix: {confusion_matrix_sum}\n")
    loss_file.flush()
    #log in writer
    writer.add_scalar("data y loss", data_y_avg, epoch)
    writer.add_scalar("ego label loss", ego_avg, epoch)
    writer.add_scalar("total loss", loss_avg, epoch)
    
    #save model weights every 10th epoch
    if epoch % 10 == 0:
        torch.save(model.state_dict(), f"/kuhpc/work/slusky/syasna_sta/func_pred/AFEnzymeRelax/full_rpn_runs/GRPN_region_pred_only/epoch_{epoch}.pt")
    #TESTING
    if epoch %20 ==0:
        model.eval()
        losses = []
        ego_losses = []
        data_y_losses = []
        confusion_matrix_sum = np.zeros((2, 2))
        for i, data in enumerate(test_loader):
            score_mean, node_scores, node_list, func_probability, x = model.forward(data.x,
                                                                                data.angle_geom, 
                                                                                data.pos, 
                                                                                data.edge_index, 
                                                                                data.batch)
            target_res = data.y
            ego_labels = data.label_graphs
            #DATA.Y LOSS
            data_y_loss = custom_bce_loss_node_list(node_scores, node_list, target_res)
            #DATA.EGO_LABELS LOSS
            func_probability = torch.squeeze(func_probability)
            ego_label_loss = bce_loss(func_probability, ego_labels, reduction="mean")
            
            #total loss is sum of the two
            test_loss = (data_y_loss + ego_label_loss)
            losses.append(test_loss.item())
            ego_losses.append(ego_label_loss.item())
            data_y_losses.append(data_y_loss.item())
            with torch.no_grad():
                #f1 and confusion matrix metrics
                confusion_matrix = compute_confusion_matrix(node_scores, node_list, data.y)
                f1_data_y = sklearn_f1_score(
                    data.y.cpu().numpy(),
                    score_mean.detach().cpu().numpy() > 0.5, average="binary")
        
            confusion_matrix_sum += confusion_matrix
        
        loss_avg = sum(losses) / len(losses)
        ego_avg = sum(ego_losses) / len(ego_losses)
        data_y_avg = sum(data_y_losses) / len(data_y_losses)
        confusion_matrix_sum = confusion_matrix_sum / len(train_loader)
        test_file.write(f"Test after Epoch {epoch}\n")
        test_file.write(f"Data y loss: {data_y_avg}\n")
        test_file.write(f"Ego label loss: {ego_avg}\n")
        test_file.write(f"total loss: {loss_avg}\n")
        test_file.write(f"confusion matrix: {confusion_matrix_sum}\n")
        test_file.flush()
loss_file.close()
test_file.close()
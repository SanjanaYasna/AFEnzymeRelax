import numpy as np
import torch
from torch.nn.functional import binary_cross_entropy as bce_loss
from torch.optim import Adam
from torch_geometric.loader import DataLoader
import sys
sys.path.insert(0,"/kuhpc/work/slusky/syasna_sta/func_pred/code/")
from utils.dataloader import ProteinDataLoader
from utils.loss import custom_bce_loss_node_list, compute_confusion_matrix
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix
from model import MainModel
from sklearn.metrics import f1_score as sklearn_f1_score
from torch.utils.tensorboard import SummaryWriter
#TODO : USE DATA PARALLEL (and look into allreduce sgd for efficiency)
#from torch.nn.parallel import DistributedDataParallel as DDP
#PARAMS TO CHANGE FOR TRAINING
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 32
num_epochs = 100
model = MainModel(8)

optimizer = Adam(model.parameters(), lr=0.001)

writer = SummaryWriter(f"/kuhpc/scratch/slusky/syasna_sta/rpn_metrics/GRPN_adam_dimenet/tensorboard")

# #data parellel option TODO
# if torch.cuda.device_count() > 1:
#     print("Let's use", torch.cuda.device_count(), "GPUks!")
#     # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
#     model = torch.nn.DataParallel(model)

# model.load_state_dict(torch.load("/kuhpc/work/slusky/syasna_sta/func_pred/AFEnzymeRelax/rpn_runs/GRPN_region_pred_only/epoch_0.pt", weights_only=True))
model.to(device)

#TODO MODIFY AS NEEDED
dataloader = ProteinDataLoader("/kuhpc/scratch/slusky/syasna_sta/swissprot_protein_data/wildtype_pts"
                            )
testloader = ProteinDataLoader("/kuhpc/scratch/slusky/syasna_sta/swissprot_protein_data/wildtype_pts_test")
train_loader = DataLoader(dataloader, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(testloader, batch_size=batch_size, shuffle=True)
torch.autograd.set_detect_anomaly(True)

#loss files 
loss_file = open("/kuhpc/work/slusky/syasna_sta/func_pred/code/GRPN_adam_dimenet/losses.csv", "a")

loss_file.write("epoch,data_y_loss,label_graph_loss,total_loss,TN_y,FP_y,FN_y,TP_y,TN_ego,FP_ego,FN_ego,TP_ego\n")
# test_file = open("/kuhpc/work/slusky/syasna_sta/func_pred/AFEnzymeRelax/rpn_runs/test_losses.txt", "w")

for epoch in range(0, num_epochs+1):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    model.train(True)
    losses = []
    ego_losses = []
    data_y_losses = []
    confusion_matrix_sum = np.zeros((2, 2))
    confusion_matrix_sum_ego_label = np.zeros((2, 2))
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
        with torch.no_grad(): #computations on cpu (seems ok so far)
            #f1 and confusion matrix metrics
            confusion_matrix = compute_confusion_matrix(node_scores, node_list, data.y) 
            ego_label_confusion_matrix = sklearn_confusion_matrix(ego_labels.detach().cpu().numpy(), func_probability.detach().cpu().numpy() > 0.5, labels=[0, 1])
        #updates
        loss.backward()
        optimizer.step()
        #add to confusion matrix
        confusion_matrix_sum += confusion_matrix
        confusion_matrix_sum_ego_label += ego_label_confusion_matrix
    #write to losses file
    loss_avg = sum(losses) / len(losses)
    ego_avg = sum(ego_losses) / len(ego_losses)
    data_y_avg = sum(data_y_losses) / len(data_y_losses)
    loss_file.write(f"{epoch},{data_y_avg},{ego_avg},{loss_avg}")
    #confusion matrix of data_y
    loss_file.write(f",{confusion_matrix_sum[0, 0]},{confusion_matrix_sum[0, 1]},{confusion_matrix_sum[1, 0]},{confusion_matrix_sum[1, 1]}")
    #confusion matrix of ego label
    loss_file.write(f",{confusion_matrix_sum_ego_label[0, 0]},{confusion_matrix_sum_ego_label[0, 1]},{confusion_matrix_sum_ego_label[1, 0]},{confusion_matrix_sum_ego_label[1, 1]}\n")
    loss_file.flush()
    #log in writer
    writer.add_scalar("data y loss", data_y_avg, epoch)
    writer.add_scalar("ego label loss", ego_avg, epoch)
    writer.add_scalar("total loss", loss_avg, epoch)
    
    #save model weights every epoch
    torch.save(model.state_dict(), f"/kuhpc/scratch/slusky/syasna_sta/rpn_metrics/GRPN_adam_dimenet/weights/epoch_{epoch}.pt")
    
    
    # #TESTING
    # if epoch %20 ==0:
    #     model.eval()
    #     losses = []
    #     ego_losses = []
    #     data_y_losses = []
    #     confusion_matrix_sum = np.zeros((2, 2))
    #     for i, data in enumerate(test_loader):
    #         score_mean, node_scores, node_list, func_probability, x = model.forward(data.x,
    #                                                                             data.angle_geom, 
    #                                                                             data.pos, 
    #                                                                             data.edge_index, 
    #                                                                             data.batch)
    #         target_res = data.y
    #         ego_labels = data.label_graphs
    #         #DATA.Y LOSS
    #         data_y_loss = custom_bce_loss_node_list(node_scores, node_list, target_res)
    #         #DATA.EGO_LABELS LOSS
    #         func_probability = torch.squeeze(func_probability)
    #         ego_label_loss = bce_loss(func_probability, ego_labels, reduction="mean")
    
    #         #total loss is sum of the two
    #         test_loss = (data_y_loss + ego_label_loss)
    #         losses.append(test_loss.item())
    #         ego_losses.append(ego_label_loss.item())
    #         data_y_losses.append(data_y_loss.item())
    #         with torch.no_grad():
    #             #f1 and confusion matrix metrics
    #             confusion_matrix = compute_confusion_matrix(node_scores, node_list, data.y) 
    #
    #         confusion_matrix_sum += confusion_matrix
    #     loss_avg = sum(losses) / len(losses)
    #     ego_avg = sum(ego_losses) / len(ego_losses)
    #     data_y_avg = sum(data_y_losses) / len(data_y_losses)
    #     confusion_matrix_sum = confusion_matrix_sum
    #     test_file.write(f"Test after Epoch {epoch}\n")
    #     test_file.write(f"Data y loss: {data_y_avg}\n")
    #     test_file.write(f"Ego label loss: {ego_avg}\n")
    #     test_file.write(f"total loss: {loss_avg}\n")
    #     test_file.write(f"confusion matrix: {confusion_matrix_sum}\n")
    #     test_file.flush()
loss_file.close()
# test_file.close()
import numpy as np
import torch
from torch.optim import Adam
from torch_geometric.loader import DataLoader
import sys
sys.path.insert(0,"/kuhpc/work/slusky/syasna_sta/func_pred/code/")
# from utils.dataloader import ProteinDataLoader
from utils.loss import focal_loss_node_list, compute_confusion_matrix, focal_loss
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix
from model import MainModel
from sklearn.metrics import f1_score as sklearn_f1_score
from torch.utils.tensorboard import SummaryWriter
import os
    
#TODO : USE DATA PARALLEL (and look into allreduce sgd for efficiency)
#from torch.nn.parallel import DistributedDataParallel as DDP
#PARAMS TO CHANGE FOR TRAINING
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 32
num_epochs = 10
model = MainModel(8)

class ProteinDataLoader(DataLoader):
    def __init__(self, pt_path):
        self.files = list(os.listdir(pt_path))[:1000]
        self.pt_path = pt_path 
        super(ProteinDataLoader, self).__init__(self)
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        #get file name at csv idx
        file_name = self.files[idx]
        file_path = self.pt_path + "/" + file_name
        return torch.load(file_path)

"""dir params"""
loss_dir = "/kuhpc/work/slusky/syasna_sta/func_pred/code/gconv_focal_loss/red_combos_cont.csv"


optimizer = Adam(model.parameters(), lr=0.001)

# writer = SummaryWriter(tb_logdir)

# model.load_state_dict(torch.load("/kuhpc/work/slusky/syasna_sta/func_pred/AFEnzymeRelax/rpn_runs/GRPN_region_pred_only/epoch_0.pt", weights_only=True))
model.to(device)

#TODO MODIFY AS NEEDED
dataloader = ProteinDataLoader("/kuhpc/scratch/slusky/syasna_sta/swissprot_protein_data/wildtype_pts"
                            )
testloader = ProteinDataLoader("/kuhpc/scratch/slusky/syasna_sta/swissprot_protein_data/wildtype_pts_test")
train_loader = DataLoader(dataloader, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(testloader, batch_size=batch_size, shuffle=True)
torch.autograd.set_detect_anomaly(True)

alphas = [0.25] #assumed to be best alpha value from flimsy stability runs

#gamma kept to 2 for now
reduction_combos = [
   # ["mean_pos", "mean"],
    ["mean", "mean_pos"],
    ["mean", "mean"],
    ["mean_pos", "mean_pos"]
]

#loss files 
loss_file = open(loss_dir, "a")
eps = 1e-6
loss_file.write("epoch,data_y_loss,label_graph_loss,total_loss,TN_y,FP_y,FN_y,TP_y,TN_ego,FP_ego,FN_ego,TP_ego\n")
# test_file = open("/kuhpc/work/slusky/syasna_sta/func_pred/AFEnzymeRelax/rpn_runs/test_losses.txt", "w")

for alpha in alphas:  
    for red_combo in reduction_combos:  
        loss_file.write(f"alpha: {alpha}, reduction: {red_combo}\n")
        for epoch in range(0, num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")
            model.train(True)
            losses = []
            ego_losses = []
            data_y_losses = []
            confusion_matrix_sum = np.zeros((2, 2))
            confusion_matrix_sum_ego_label = np.zeros((2, 2))
            for i, data in enumerate(train_loader):
                optimizer.zero_grad()
                node_scores, node_list, func_probability, x = model.forward(data.x,
                                                                                        data.angle_geom, 
                                                                                        data.pos, 
                                                                                        data.edge_index, 
                                                                                        data.batch)
                target_res = data.y
                ego_labels = data.label_graphs
                #DATA.Y LOSS
                data_y_loss = focal_loss_node_list(node_scores, node_list, target_res, reduction=red_combo[0],
                                                alpha = alpha)

                #DATA.EGO_LABELS LOSS
                func_probability = torch.squeeze(func_probability)
                ego_label_loss = focal_loss(func_probability, ego_labels, reduction = red_combo[1], alpha = alpha)
                
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
        # #write f1 precision recall scores from confusion matrix
        # f1 = 2 * confusion_matrix_sum[1, 1] / (2 * confusion_matrix_sum[1, 1] + confusion_matrix_sum[0, 1] + confusion_matrix_sum[1, 0])
        # precision = confusion_matrix_sum[1, 1] / (confusion_matrix_sum[1, 1] + confusion_matrix_sum[0, 1])
        # recall = confusion_matrix_sum[1, 1] / (confusion_matrix_sum[1, 1] + confusion_matrix_sum[1, 0])
        # loss_file.write(f"Data.y F1: {f1}, Precision: {precision}, Recall: {recall}\n")
loss_file.close()
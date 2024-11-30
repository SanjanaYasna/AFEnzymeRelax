import torch
from torch.nn.functional import binary_cross_entropy as bce_loss
from torch.optim import SGD, lr_scheduler
from torch.nn.functional import binary_cross_entropy as bce_loss
from sklearn.metrics import accuracy_score as sklearn_accuracy_score
from torch_geometric.loader import DataLoader
from itertools import compress
from dataloader import ProteinDataLoader
from loss import custom_bce_loss_node_list
from model import MainModel

#PARAMS TO CHANGE FOR TRAINING
batch_size = 5
num_epochs = 100
model = MainModel(8)
optimizer = SGD(model.parameters(), lr=0.1)
scheduler = lr_scheduler.CyclicLR(optimizer, base_lr=0.05, max_lr=0.1)
device = "cuda" if torch.cuda.is_available() else "cpu"


#TODO CREATE A TESTING DATASET TOO
dataloader = ProteinDataLoader('/kuhpc/scratch/slusky/syasna_sta/swissprot_protein_data/pdb_pts', 
                            '/kuhpc/work/slusky/syasna_sta/func_pred/AFEnzymeRelax/data_stats/possible_tsvs/final_protein_dataset.csv'
                            )
train_loader = DataLoader(dataloader, batch_size=1, shuffle=True)

#CHANGE THIS

data = next(iter(train_loader))
data = data.to(device)
print(data.label_graphs)

node_list, func_probability, x = model(data.x, data.angle_geom, data.pos, data.edge_index, data.batch)
print("functionality probs", func_probability.shape)
print("target labels", data.y.shape)
target_res = data.y
#pred scores list 
#we are more concerned about subgraphs missing lots of functional residues, 
# labelling right ones is hopefully improved by func_probability's loss
#DATA.Y LOSS
pred_nodes= list(compress(node_list, target_res))
pred_nodes = [x for xs in pred_nodes for x in xs]
print(data.protein_id)
data_y_loss = custom_bce_loss_node_list(pred_nodes, target_res)
print(data_y_loss)

#DATA.EGO_LABELS LOSS
ego_labels = data.label_graphs
print(ego_labels)








    
    
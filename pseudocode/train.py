import torch
from torch.nn.functional import binary_cross_entropy as bce_loss
from torch.optim import SGD, lr_scheduler

from sklearn.metrics import accuracy_score as sklearn_accuracy_score
from torch_geometric.loader import DataLoader

from dataloader import ProteinDataLoader

from model import MainModel

#PARAMS TO CHANGE FOR TRAINING
batch_size = 5
num_epochs = 100
optimizer = SGD(model.parameters(), lr=0.1)
scheduler = lr_scheduler.CyclicLR(optimizer, base_lr=0.05, max_lr=0.1)


#TODO CREATE A TESTING DATASET TOO
dataloader = ProteinDataLoader('/kuhpc/scratch/slusky/syasna_sta/swissprot_protein_data/pdb_pts', 
                               '/kuhpc/work/slusky/syasna_sta/func_pred/AFEnzymeRelax/data_stats/possible_tsvs/final_protein_dataset.csv'
                               )
train_loader = DataLoader(dataloader, batch_size=3, shuffle=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
#CHANGE THIS
model = MainModel(8, batch_size, len(data.x))




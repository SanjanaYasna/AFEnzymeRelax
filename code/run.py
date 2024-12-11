import torch
from torch_geometric.loader import  DataLoader
import numpy as np
import pandas as pd
import os
from relational_module import InitialInteraction
import re
from torch_geometric.data import Data
from torch.nn import LazyLinear
from torch.nn import Linear
from model import MainModel

#TODO JUST MAKE A MAP OF ENTRY TO PT FILE PATH
class ProteinDataLoader(DataLoader):
    def __init__(self, pt_path, csv_path):
        self.files = list(os.listdir(pt_path))
        self.pt_path = pt_path
        self.csv = pd.read_csv(csv_path)    
        super(ProteinDataLoader, self).__init__(self)
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        #get file name at csv idx
        file_name = self.csv.iloc[idx, 0]
        r = re.compile(f'{file_name}*')
        file_path = self.pt_path + "/" + list(filter(r.match, self.files))[0]
        return torch.load(file_path)
    
dataloader = ProteinDataLoader('/kuhpc/scratch/slusky/syasna_sta/swissprot_protein_data/pdb_pts', 
                               '/kuhpc/work/slusky/syasna_sta/func_pred/AFEnzymeRelax/data_stats/possible_tsvs/final_protein_dataset.csv'
                               )
batch_size = 3
train_loader = DataLoader(dataloader, batch_size=2, shuffle=False)

for data in train_loader:
    break
model = MainModel(8, batch_size, len(data.x))
res = model(data.x, data.angle_geom, data.pos, data.edge_index, data.batch)
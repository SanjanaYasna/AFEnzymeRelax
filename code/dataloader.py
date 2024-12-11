#make dataloader class for item retrieval
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
from struc_feat import create_protein_graph, ego_label_set  
from relational_module import InitialInteraction
import re
from torch_geometric.data import Data

#Put in an out_dir to get an outpath for pt files that are put in

class RetrieveData(Dataset):
    def __init__(self, root, hidden_channels, out_dir = None):
        self.root = root 
        #get full path of each data
        self.data_list = [root + "/" + file for file in os.listdir(root)]
        self.hidden_channels = hidden_channels
        self.out_dir = out_dir
    def __len__(self):
        return len(self.data_list)
    """"
    Returns following attrs:
    edge_index: edge index
    label_graphs: label graphs (ego/self labels)
    x: node embeddings
    y: node labels
    #TODO: DSSP labels 
    #TODO: kinetics? look at scannet... 
    """
    def __getitem__(self, idx):
        file = self.data_list[idx]
        functional_nodes = [9, 25]
        graph  = create_protein_graph(file, functional_nodes)
        label_graphs = ego_label_set(graph, functional_nodes)
        node_one_hot = torch.tensor([att["x"] for node, att in graph.nodes(data=True)])
        pos = torch.tensor([att["ca_coords"] for node, att in graph.nodes(data=True)])
        edge_index = torch.LongTensor(list(graph.edges)).t().contiguous()
        angle_geom = torch.tensor([att["angle_geom"] for node, att in graph.nodes(data=True)])
        # initial_embedding = InitialInteraction(self.hidden_channels, len(graph.nodes))
        # node_embs = initial_embedding(node_one_hot, angle_geom, pos, edge_index)
        data = Data(edge_index = edge_index,  
                    label_graphs = label_graphs, 
                    x = node_one_hot, 
                    pos = pos,
                    angle_geom = angle_geom,
                    y = torch.tensor([att["y"] for node, att in graph.nodes(data=True)]),
                    )
        # data = {"edge_index": edge_index,  "label_graphs": label_graphs, "x": node_embs, "y": torch.tensor([att["y"] for node, att in graph.nodes(data=True)])}
        #save data if out dir specified
        if self.out_dir:
            file_name = os.path.basename(file).split(".")[0]
            torch.save(data, self.out_dir + "/" + file_name + ".pt")
        return data
    
#dataloader for pt data...

#TODO JUST MAKE A MAP OF ENTRY TO PT FILE PATH
class ProteinDataLoader(DataLoader):
    def __init__(self, pt_path):
        self.files = list(os.listdir(pt_path))
        self.pt_path = pt_path 
        super(ProteinDataLoader, self).__init__(self)
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        #get file name at csv idx
        file_name = self.files[idx]
        file_path = self.pt_path + "/" + file_name
        return torch.load(file_path)
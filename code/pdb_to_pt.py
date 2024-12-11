import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
from struc_feat import create_protein_graph, ego_label_set  
import os
from torch_geometric.data import Data
import re
import ast
from multiprocessing import Pool
from joblib import Parallel, delayed
from Bio.PDB import PDBParser


#TODO ADD EC NUMBER AS LABEL TOO
def generate_geometric_object(id, active_sites, binding_sites, ec_nums):
    #find the file name from file set that contains id
    r = re.compile(f'{id}*')

    file_name = list(filter(r.match, file_list))
    try:
        if file_name and not os.path.exists(f'/kuhpc/scratch/slusky/syasna_sta/swissprot_protein_data/pdb_pts/{file_name[0]}.pt'):
            file_name = file_name[0]
            #get the graph object
            pdb_path = f'/kuhpc/scratch/slusky/syasna_sta/swissprot_protein_data/pdbs/{file_name}'
            ec_nums = list(ec_nums)
            functional_nodes = active_sites
            functional_nodes.extend(binding_sites)
            #for some unexplained fucking reason, it's like this...do they need their own kind of parser, or what's going on? why is it like this?
            parser = PDBParser()
            protein = parser.get_structure(id, pdb_path)
            protein = protein[0]
            
            #create graph object with functional node labels
            graph  = create_protein_graph(pdb_path, functional_nodes, protein)
            #get the data.y labels  and the one-hot encoding of each node
            label_graphs = ego_label_set(graph, functional_nodes)
            #one_hot embeddings are the problematic ones...
            try:
                node_one_hot = torch.tensor([att["x"] for node, att in graph.nodes(data=True) if node is not None])
            except:
                node_one_hot = [att["x"] for node, att in graph.nodes(data=True)]
                #filter out none values
                node_one_hot = [node for node in node_one_hot if node is not None]
                node_one_hot = torch.tensor(node_one_hot)
            pos = torch.tensor([att["ca_coords"] for node, att in graph.nodes(data=True)])
            edge_index = torch.LongTensor(list(graph.edges)).t().contiguous()
            angle_geom = torch.tensor([att["angle_geom"] for node, att in graph.nodes(data=True)])
            #properties
            data = Data(
                protein_id = id,
                ec_number = ec_nums,
                edge_index = edge_index,  
                label_graphs = label_graphs, 
                x = node_one_hot, 
                pos = pos,
                angle_geom = angle_geom,
                y = torch.tensor([att["y"] for node, att in graph.nodes(data=True)]),
            )
            torch.save(data, out_dir + "/" + file_name + ".pt")
    except:
        print(f"Could not get pt for {id}")

if __name__ == '__main__':
    #convert tsv to csv
    alphafold = pd.read_csv("/kuhpc/work/slusky/syasna_sta/func_pred/AFEnzymeRelax/data_stats/possible_tsvs/final_protein_dataset.csv")
    #get pool object with 30 processes
    ids = list(alphafold['Entry'])
    ec_nums = list(alphafold['EC_Shortened'])
    ec_nums = [ast.literal_eval(num) for num in ec_nums]
    active_sites = list(alphafold['Active site'])
    active_sites = [ast.literal_eval(site) for site in active_sites]
    binding_sites = list(alphafold['Binding site'])
    binding_sites = [ast.literal_eval(site) for site in binding_sites]
    file_set = set(os.listdir("/kuhpc/scratch/slusky/syasna_sta/swissprot_protein_data/pdbs"))
    
    out_dir = '/kuhpc/scratch/slusky/syasna_sta/swissprot_protein_data/pdb_pts'
    file_list = list(file_set)
    
    #need to streamline these thre alues
    
    Parallel(n_jobs=-1, backend="threading")(delayed(generate_geometric_object)(
        ids[i], active_sites[i], binding_sites[i], ec_nums[i]) 
                                        for i in range(len(ids)))
                                        #     now do circular range from half of len ids to around, and this is worst case only arond 1/5th done
                                        #   for i in range(len(ids)))
                                        # for i in reversed(range(len(ids))))
    

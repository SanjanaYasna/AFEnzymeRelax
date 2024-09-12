import numpy as np
import torch
from torch.utils.data import Dataset
from Utils import struc_feat
#import from atomRefine cal_covalent and bond and 
from Utils.pair_handler import cal_covalent
import os,re

#take heavy_atoms.pkl from atomRefine


"""
    Current use: desired conversion of afdb pre-relaxed files
 """
class DataConversion(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(DataConversion, self).__init__(root, transform=None,
                pre_transform=None)
        #TO DO: possible pre-relax transformation protocol?
        
    def __len__(self):
        return self.data_len
    
    @property
    def raw_relaxed_afdb_files(self):
        return [filename for filename in os.scandir(self.root + "/relax_afdb") if "_relaxed_0001.pdb" in filename.name]
    @property
    def raw_pre_relax_files(self):
        return [filename for filename in os.scandir(self.root + "/alphafoldSwiss") ]
    
        """
        Input: relax and pre_relax paths
        Node features:
        atom_position
        
        
        """
    def __get_node_feat__(self,  seq_len):
        # node feat
        node_feat = {}
        for file in self.raw_pre_relax_files:
            pre_relax_file  = file
            id = os.path.splitext(os.path.basename(file))[0]
            relax_file = self.root + "/alphafoldSwiss/" + id + "_relaxed_0001.pdb"
           # id_relax = id = os.path.splitext(os.path.basename(file))[0].replace("_relaxed_0001", "")
            # atom_emb
            #get residue sequence and length of amino acid sequence and feed it into the embedding extractor 
            residue_sequence, seq_len = struc_feat.get_sequence(pre_relax_file)
            atom_embs, atom_xyz, atom_relax_xyz, atom_nums, ca_pose= struc_feat.get_atom_emb(pre_relax_file, relax_file, [1, seq_len], residue_sequence)
            
            #get residue embeddings to concatenate iwth atom emb
            residue_onehot = struc_feat.residue_onehot(residue_sequence)
            #add residue encoding to atom emb
            for i in range(len(atom_embs)):
                #assign a seq emb to every atom of that residue
                seq_emb = np.tile(residue_onehot[i], (len(atom_embs[i]), 1))
                atom_emb = atom_embs[i]
                atom_embs[i] = np.concatenate((seq_emb, atom_emb[:, :]), axis=1)
            node_feat['atom_emb'] = {
                'embedding': atom_embs, #one_hot_atom_id +  one_hot_residue_id
                'atom_pos': atom_xyz,
                'atom_relax_pos': atom_relax_xyz,
                'atom_nums': atom_nums,
                'residue_sequence': residue_sequence,
                'CA_lst': ca_pose,
            }
        return node_feat
        # for file in self.raw_relaxed_afdb_files:
        #     id = os.path.splitext(os.path.basename(file))[0].replace("_relaxed_0001", "")
        
        

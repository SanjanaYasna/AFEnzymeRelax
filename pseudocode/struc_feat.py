import re
from typing import OrderedDict
import warnings
import pandas as pd
import numpy as np
from Bio import PDB
import torch

# from Bio.PDB import PDBList
import Bio.PDB.PDBParser as PDBParser
import Bio.PDB.PDBList as PDBList
import networkx as nx
# import torch_geometric
# import torch_geometric.utils
import pickle 
import os
import torch_geometric
from torch_geometric.data import DataLoader
from CL import OutputPred
from gcn_net import GraphRPN
from torch_geometric.data.batch import Batch
warnings.simplefilter("ignore")
warnings.filterwarnings("ignore")


__MYPATH__ = os.path.split(os.path.realpath(__file__))[0]

heavy_atoms = pickle.load(open("/Users/robsonlab/Teetly/AFEnzymeRelax/Utils/heavy_atoms.pkl", "rb"))
#atom_dict = {'C':0, 'N':1, 'O':2, 'S':3}
atom_dict = {"N":0, "CA":1, "C":2, "O":3, "CB":4, "OG":5, "CG":6, "CD1":7, "CD2":8, "CE1":9, "CE2":10, "CZ":11, 
             "OD1":12, "ND2":13, "CG1":14, "CG2":15, "CD":16, "CE":17, "NZ":18, "OD2":19, "OE1":20, "NE2":21, 
             "OE2":22, "OH":23, "NE":24, "NH1":25, "NH2":26, "OG1":27, "SD":28, "ND1":29, "SG":30, "NE1":31, 
             "CE3":32, "CZ2":33, "CZ3":34, "CH2":35, "OXT":36}

AA_NAME_MAP = OrderedDict((
    ("CYS", "C"),
    ("ASP", "D"),
    ("SER", "S"),
    ("GLN", "Q"),
    ("LYS", "K"),
    ("ILE", "I"),
    ("PRO", "P"),
    ("THR", "T"),
    ("PHE", "F"),
    ("ASN", "N"),
    ("GLY", "G"),
    ("HIS", "H"),
    ("LEU", "L"),
    ("ARG", "R"),
    ("TRP", "W"),
    ("ALA", "A"),
    ("VAL", "V"),
    ("GLU", "E"),
    ("TYR", "Y"),
    ("MET", "M"),
    ("XAA", "X"),
    ("TER", "*"))
)
AA_NAME_MAP_INDICES = {v: k for k, v in enumerate(AA_NAME_MAP.values())}
#derived from scannet
atom_type_mass = {'C': 12, 'CA': 12, 'CB': 12, 'CD': 12, 'CD1': 12, 
     'CD2': 12, 'CE': 12, 'CE1': 12, 'CE2': 12, 'CE3': 12, 
     'CG': 12, 'CG1': 12, 'CG2': 12, 'CH2': 12, 'CZ': 12, 
     'CZ2': 12, 'CZ3': 12, 'N': 14, 'ND1': 14, 'ND2': 14, 
     'NE': 14, 'NE1': 14, 'NE2': 14, 'NH1': 14, 'NH2': 14, 
     'NZ': 14, 'O': 16, 'OD1': 16, 'OD2': 16, 'OE1': 16, 
     'OE2': 16, 'OG': 16, 'OG1': 16, 'OH': 16, 'OXT': 16, 
     'SD': 32, 'SE': 32, 'SG': 32}

parser = PDBParser(QUIET = True)

def compute_contacts(coords, node_labels, threshold=6.9, binarize=True):
    """Compute the pairwise contacts.

    Here we define a contact as the C-alpha atom
    of 2 amino acids being within 9 Angstrom of each other.

    Args:
        coords (_type_): array of shape (num_residues, 3)
        node_labels (_type_): _description_
        threshold (float, optional): distance threshold to consider a contact. Defaults to 6.9.
        binarize (bool, optional): _description_. Defaults to True.

    Returns:
        contacts (pd.DataFrame): Dataframe of shape (num_residues, num_residues)
                                containing contacts or distances between residues
    """

    num_residues = coords.shape[0]
    contacts = np.zeros((num_residues, num_residues))

    for i in range(num_residues):
        for j in range(i + 1, num_residues):  # Skip self and already computed
            distance = np.linalg.norm(coords[i] - coords[j])
            if binarize:
                if distance <= threshold:
                    contacts[i, j] = 1
                    contacts[j, i] = 1  # The matrix is symmetric
            else:
                contacts[i, j] = distance
                contacts[j, i] = distance

    return contacts


def load_pdb(pdb_path):
    """For a given protein pdb file and extract the sequence and coordinates.

    Returns:
        node_labels (List[str]): label for each node to be considered while constructing graph
                                format: "chain_position_residue"
        embed_dict (dict): dictionary containing embeddings for each residue
        coords (np.array): array of shape (num_residues, 3) containing the centroid of each residue
    """
    #read protein structure with parser
    protein = parser.get_structure("protein", pdb_path)
    protein = protein[0]

    # sequence = []
    coords = []
    # chain_ids = []
    node_embeddings = []
    node_labels = []
    lrf = []
    for chain in protein:
        # if chain.id != chain_id:
        #   continue
        residue_number = []
        current_sequence = []
        for residue in chain:
            #get CA coord of residue
            ca_coord = residue["CA"].coord
            #call the lrf function
            lrf.append(set_lframe(residue["N"].coord, residue["CA"].coord, residue["C"].coord, res_range=None))
            coords.append(ca_coord)
            try: 
                resname = AA_NAME_MAP[residue.resname]
            except Exception as e:
                resname = "X"
            finally: 
                resname = "X"
            #get one-hot encoding of residue
            res_one_hot = np.zeros(len(AA_NAME_MAP))
            res_one_hot[AA_NAME_MAP_INDICES[resname]] = 1
            node_embeddings.append(res_one_hot)
            # sequence.append(resname)
            current_sequence.append(resname)
            residue_number.append(
                "".join([str(x) for x in residue.get_id()]).strip()
            )
            
        # print(chain.id, residue_number)
        # assert len(residue_number) == len(current_sequence) == len(embed_values)
        new_node_labels = [
            f"{chain.id}_{n}_{s}" for n, s in zip(residue_number, current_sequence)
        ]
        
        node_labels += new_node_labels
        assert len(node_labels) == len(coords)
    return node_labels, np.array(coords), np.array(lrf), np.array(node_embeddings)

"""
AtomRefine LRF snipp;et, per-desidue x y z axis vectors for exhibiting very basic planar geometry"""
def set_lframe(N_coord, Ca_coord, C_coord, res_range=None):
    '''
    Agrs:
        structure_file full path
        atom_xyz : np.array, shape (n,3)
        atom_nums : list of num atoms per residue, shape (n,)
    '''
    # set local frame for each residue in pdb
    pdict = dict()
    pdict['N'] = N_coord
    pdict['Ca'] = Ca_coord
    pdict['C'] = C_coord
    
    # recreate Cb given N,Ca,C
    ca = -0.58273431
    cb = 0.56802827
    cc = -0.54067466
    
    b = pdict['Ca'] - pdict['N']
    c = pdict['C'] - pdict['Ca']
    a = b * c
    #TODO: MAKE BETTER REACHABLE WITHOUT SEVERE PRECISION LOSS...
    pdict['Cb'] = ca * a + cb * b + cc * c

    # local frame is the C-Ca-N plane
    z = pdict['Cb'] - pdict['Ca']
    z = z / np.linalg.norm(z)
    x = np.cross(pdict['Ca']-pdict['N'], z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    y = y / np.linalg.norm(y)

    xyz = np.stack([x,y,z])
    return xyz
    
    
"""
Scannet-inspired residue frame encoding. NOT USED currently"""
#@njit(cache=True, parallel=False)
def _get_aa_frameCloud_triplet_sidechain(atom_coordinates, ca_coord, atom_ids, verbose=True):
    L = len(atom_coordinates)
    aa_triplets = list()
    count = 0
    for l in range(L):
        atom_coordinate = atom_coordinates[l]
        atom_id = atom_ids[l]
        natoms = len(atom_id)
        center = 1 * count
        count += 1
        if count > 1:
            previous = aa_triplets[-1][0]
        else:
            # Need to place another virtual Calpha.
            virtual_calpha_coordinate = 2 * ca_coord - atom_coordinates[1][0]
            previous = 1 * count
            count += 1

        sidechain_CoM = np.zeros(3, dtype=np.float32)
        sidechain_mass = 0.
        for n in range(natoms):
            if not atom_id[n] in [0, 1, 17, 26, 34]:
                mass = atom_type_mass[atom_id[n]]
                sidechain_CoM += mass * atom_coordinate[n]
                sidechain_mass += mass
        if sidechain_mass > 0:
            sidechain_CoM /= sidechain_mass
        else:  # Usually case of Glycin
            #'''
            #TO CHANGE FOR NEXT NETWORK ITERATION... I used the wrong nitrogen when I rewrote the function...
            if l>0:
                if (0 in atom_id) & (1 in atom_id) & (17 in atom_ids[l-1]):  # If C,N,Calpha are here, place virtual CoM
                    sidechain_CoM = 3 * atom_coordinate[atom_id.index(1)] - atom_coordinates[l-1][atom_ids[l-1].index(17)] - \
                                    atom_coordinate[atom_id.index(0)]
                else:
                    if verbose:
                        print('Warning, pathological amino acid missing side chain and backbone', l)
                    sidechain_CoM = atom_coordinate[-1]
            else:
                if verbose:
                    print('Warning, pathological amino acid missing side chain and backbone', l)
                sidechain_CoM = atom_coordinate[-1]

        next = 1 * count
        count += 1
        aa_triplets.append((center, previous, next))
    return  aa_triplets


def create_protein_graph(pdb_path, active_and_binding_site_residues):
    """For a given protein pdb_id, extract all functional site annotations and create a graph where the contact map is
    the adjancency matrix, nodes are labelled according to "chain_position_residue" and functionality of nodes is depicted.

    Args:
        PDB_ID (str): PDB ID of the protein which is to be converted to a graph
        df (pd.DataFrame): Dataframe containing the PDBSite annotations

    Returns:
        protein_graph (nx.Graph): graph with attributes such as functionality, pdb_site_id, length of functional site,
                                edges are the contacts between residues
    """

    node_labels, coords, lrfs, residue_one_hot = load_pdb(pdb_path)
    contacts = compute_contacts(coords, node_labels)
    # plt.imshow(contacts)
    # print(node_labels, info_dict, coords)
    assert contacts.shape[0] == len(node_labels)
    protein_graph = nx.from_numpy_matrix(contacts)
    # protein_graph.edges
    nx.set_node_attributes(
        protein_graph, name="y", values=0
    )  # 0 for non-functional, 1 for functional
    nx.set_node_attributes(protein_graph, name="x", values="None")
    nx.set_node_attributes(protein_graph, name="dssp", values="None")
    nx.set_node_attributes(protein_graph, name="ca_coords", values="None")
    nx.set_node_attributes(protein_graph, name="angle_geom", values="None")
    # nx.set_node_attributes(protein_graph, name="coords", values="None")
   

    ##### groupby protein structure, set all nodes to none
    ##### select functional sites, put label function, subtype: header
    #take in the active and binding site residues and label their respective nodes as y = 1
    for res_num in active_and_binding_site_residues:
        #index for that residue number
        protein_graph.nodes[res_num]["y"] = 1
    #add in ca_coords
    for i in range(len(coords)):
        protein_graph.nodes[i]["ca_coords"] = coords[i]
    #set local lrfs
    for i in range(len(lrfs)):
        protein_graph.nodes[i]["angle_geom"] = lrfs[i]
    #set residue one-hot encodings
    for i in range(len(residue_one_hot)):
        protein_graph.nodes[i]["x"] = residue_one_hot[i]
    return protein_graph
    
    '''TO DO: DSSP assignment + possible SASA (
        https://github.com/jertubiana/ScanNet/blob/main/preprocessing/PDB_processing.py#L235
        Scannet line ~238, estimate relative SASA from DSSP
    )'''

#NEEDS TESTING
#TODO: add dssp assignment to this factoring of anchor regions 
"""Adds functionality labels for the ego graph anchor centered at each functional node IF 

    >90% of a functional site (what is within 10 distance of that node) is within the ego graph
    NOTE: "distance" networkx weight metric is used to guage the 10 upper limit. So this isn't necessarily gonig to extract
    all atoms that are 10 angstroms apart, but atoms that fall within 10
    of each other in the binary edges of the graph, which are by default 1 between nodes that are 10 angstroms apart by CA. 

    Args:
        protein_graph (nx.Graph): takes in protein graph processed by create_protein_graph()
        radius = 2 (int): radius of ego graph
        overlap_ratio_cutoff = 0.7 (float): ratio of functional nodes within 10 angstroms of a functional node that must be in ego graph for ego label to be 1

    Returns:
        label_graphs (dict): for each (ground truth = 1) functional node, what are the ground truth graph pdbsites
        graph is directly edited in-place to include ego_label attribute
    """
def ego_label_set(graph: nx.Graph, sites: list, radius = 2, overlap_ratio_cutoff = 0.9):
    ego_label = {node: 0 for node, att in graph.nodes(data=True)}
    label_graphs = (
        {}
    )  # for each (ground truth = 1) functional node what are the ground truth graph pdbsites
    functional_nodes = [
        node for node, att in graph.nodes(data=True) if att["y"] == 1
    ]
    for functional_node in functional_nodes:
        ego_subgraph = nx.ego_graph(graph, functional_node, radius=radius)
        #count number of functional nodes in ego subgraph
        func_subgraph_nodes = len([
            node for node, att in ego_subgraph.nodes(data=True) if att["y"] == 1
        ])
        #now find number of functional nodes within 10 angstroms of functional node
        #extract site nodes numbers, and compute distances
        total_func_nodes_ten_apart = len([
            node for node in sites if nx.shortest_path_length(graph, source=functional_node, target=node, weight='distance') <= 10
        ])
        #if this functional node has at least [overlap_ratio_cutoff] of the functional nodes within 10 angstroms of it, give it an ego label of 1 and add it
        #to label_graphs dictionary
        if func_subgraph_nodes / total_func_nodes_ten_apart > overlap_ratio_cutoff:
            ego_label[functional_node] = 1
            label_graphs[functional_node] = ego_subgraph
    nx.set_node_attributes(graph, ego_label, "ego_label")
    return label_graphs

# def get_a_batch(batch_size):
#     pdb_path = "/Users/robsonlab/Teetly/AFEnzymeRelax/test/relax/A0A009IHW8_relaxed_0001.pdb"
#     root = "/Users/robsonlab/Teetly/AFEnzymeRelax/test/relax/"
#     dataset = RetrieveData(root, 8)
#     batch_size = batch_size
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
#     data = next(iter(dataloader))
#     return data


#default execution
if __name__ == "__main__":
    root = "/Users/robsonlab/Teetly/AFEnzymeRelax/test/relax/"
    from dataloader import ProteinDataLoader
    load_pt = ProteinDataLoader(root = "/Users/robsonlab/Teetly/AFEnzymeRelax/test/data_pts")
    loader = DataLoader(load_pt, batch_size = 5)
    data = next(iter(loader))
    #this is just for the sake of trying out the contrastive learning shit
    x, edge_index, batch = data.x, data.edge_index, data.batch
    
    embed_dim = data.x.shape[1]
    in_progress_rpn = GraphRPN(32, 32)
    node_scores, ego_preds, func_scores, nodes = in_progress_rpn(x, edge_index, batch)
    pred_layer = OutputPred(32, 8)
    x, x2, pred = pred_layer(nodes, data.batch)
    
    # for i in range(len(dataset)):
    #     data = dataset[i]
    #     torch.save(data, root + f"graph_{i}.pt")
    # batch_size = 3
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # data = next(iter(dataloader))
    
    #add in a csv active and binding site list processor later on, or name these pdb files accordingly
    #ASSUME THAT THESE SITE LABELS ARE FROM THE COUNT OF 1 (NOT FROM 0)
    # functional_nodes = [8, 13]
    # graph  = create_protein_graph(pdb_path, functional_nodes)
    # label_graphs = ego_label_set(graph, functional_nodes)
    # node_one_hot = torch.tensor([att["x"] for node, att in graph.nodes(data=True)])
    # pos = torch.tensor([att["ca_coords"] for node, att in graph.nodes(data=True)])
    # edge_index = torch.LongTensor(list(graph.edges)).t().contiguous()
    # angle_geom = torch.tensor([att["angle_geom"] for node, att in graph.nodes(data=True)])
    # #can x be from model import NodeEmbeddingBlock 
    # from relational_module import InitialInteraction
    # try_model = InitialInteraction(8, len(node_one_hot))
    # out = try_model(node_one_hot, angle_geom, pos, edge_index)
    #below import will be deleted if this integration was tested successful, which is why imports are here...
    
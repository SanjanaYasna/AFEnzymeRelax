import re
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

warnings.simplefilter("ignore")
warnings.filterwarnings("ignore")

__MYPATH__ = os.path.split(os.path.realpath(__file__))[0]

heavy_atoms = pickle.load(open("/Users/robsonlab/Teetly/AFEnzymeRelax/Utils/heavy_atoms.pkl", "rb"))
#atom_dict = {'C':0, 'N':1, 'O':2, 'S':3}
atom_dict = {"N":0, "CA":1, "C":2, "O":3, "CB":4, "OG":5, "CG":6, "CD1":7, "CD2":8, "CE1":9, "CE2":10, "CZ":11, 
             "OD1":12, "ND2":13, "CG1":14, "CG2":15, "CD":16, "CE":17, "NZ":18, "OD2":19, "OE1":20, "NE2":21, 
             "OE2":22, "OH":23, "NE":24, "NH1":25, "NH2":26, "OG1":27, "SD":28, "ND1":29, "SG":30, "NE1":31, 
             "CE3":32, "CZ2":33, "CZ3":34, "CH2":35, "OXT":36}

AA_NAME_MAP = {
    "CYS": "C",
    "ASP": "D",
    "SER": "S",
    "GLN": "Q",
    "LYS": "K",
    "ILE": "I",
    "PRO": "P",
    "THR": "T",
    "PHE": "F",
    "ASN": "N",
    "GLY": "G",
    "HIS": "H",
    "LEU": "L",
    "ARG": "R",
    "TRP": "W",
    "TER": "*",
    "ALA": "A",
    "VAL": "V",
    "GLU": "E",
    "TYR": "Y",
    "MET": "M",
    "XAA": "X",
}

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
    embed_dict = {}
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
            resname = AA_NAME_MAP[residue.resname]
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
    return node_labels, np.array(coords), np.array(lrf)

"""
AtomRefine LRF:"""
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

    node_labels, coords, lrfs= load_pdb(pdb_path)
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
    
    '''TO DO: DSSP assignment + possible SASA (
        https://github.com/jertubiana/ScanNet/blob/main/preprocessing/PDB_processing.py#L235
        Scannet line ~238, estimate relative SASA from DSSP
    )'''
    # groups_df = df.groupby("pdb_id")
    # functional_nodes = groups_df.get_group(PDB_ID).node_labels.values
    # headers = groups_df.get_group(PDB_ID)["y"].values

    # func_node_attr = {}
    # for header, pdbsite_id, lensite_num, nodes in zip(
    #     headers, pdbsite_ids, lensite, functional_nodes
    # ):
    #     for node in nodes:
    #         if node not in func_node_attr:
    #             func_node_attr[node] = {
    #                 "y": 1,
    #                 "subtype": [header],
    #                 "pdbsiteid": [pdbsite_id],
    #                 "lensite": [lensite_num],
    #                 # "coords": coords[node_labels.index(node)],
    #             }
    #         else:
    #             print("common node found")
    #             func_node_attr[node]["subtype"].append(header)
    #             func_node_attr[node]["pdbsiteid"].append(pdbsite_id)
    #             func_node_attr[node]["lensite"].append(lensite_num)
    #             print(func_node_attr[node])

    # nx.set_node_attributes(protein_graph, func_node_attr)
    # return protein_graph

#default execution
if __name__ == "__main__":
    pdb_path = "/Users/robsonlab/Teetly/AFEnzymeRelax/test/relax/A0A009IHW8_relaxed_0001.pdb"
    #add in a csv active and binding site list processor later on, or name these pdb files accordingly
    #ASSUME THAT THESE SITE LABELS ARE FROM THE COUNT OF 1 (NOT FROM 0)
    functional_nodes = [8, 13]
    create_protein_graph(pdb_path, functional_nodes)
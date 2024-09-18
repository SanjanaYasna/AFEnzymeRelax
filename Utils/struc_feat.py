import os, pickle
import Bio.PDB
import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB import Superimposer
from Bio.PDB.Atom import *
from Bio.PDB.Residue import *
from Bio.PDB.Chain import *
from Bio.PDB.Model import *
from Bio.PDB.Structure import *
# from Bio.PDB.Vector import *
from Bio.PDB.Entity import*
import math
from Bio.PDB.Atom import get_vector 

from Bio import SeqIO
from Bio.PDB.DSSP import DSSP
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

__MYPATH__ = os.path.split(os.path.realpath(__file__))[0]

heavy_atoms = pickle.load(open(__MYPATH__+"/heavy_atoms.pkl", "rb"))
#atom_dict = {'C':0, 'N':1, 'O':2, 'S':3}
atom_dict = {"N":0, "CA":1, "C":2, "O":3, "CB":4, "OG":5, "CG":6, "CD1":7, "CD2":8, "CE1":9, "CE2":10, "CZ":11, 
             "OD1":12, "ND2":13, "CG1":14, "CG2":15, "CD":16, "CE":17, "NZ":18, "OD2":19, "OE1":20, "NE2":21, 
             "OE2":22, "OH":23, "NE":24, "NH1":25, "NH2":26, "OG1":27, "SD":28, "ND1":29, "SG":30, "NE1":31, 
             "CE3":32, "CZ2":33, "CZ3":34, "CH2":35, "OXT":36}

RESIDUE_TYPES = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y','X']
parser = Bio.PDB.PDBParser(QUIET = True)


"""
Inspired by anglesrefine (https://github.com/Cao-Labs/AnglesRefine/blob/main/PDB2Angles.py)
Extract backbone angles and plane per residue.
Should be done by both pre-relaxed and relaxed structures
Input: file path
Output:
    6 Angles: phi, psi, omega,  CA_C_N_angle, N_CA_C_angle, C_N_Ca_angle
    3 bond lengths centered around residue angle frame: Ca-C, N-Ca, C_prev-N
"""
#TODO figure out a universal geometry builder. Perhaps from PeptideBuilder import Geometry , geometry objects?
"""
3 bond lengths centered around residue angle frame: Ca-C, N-Ca, C_prev-N"""
def get_backbone_angles_and_bond_lengths(file): 
    #load structure
    structure = parser.get_structure("input", file)
    C_prev = None
    N_prev = None
    CA_prev = None
    #initialize arrays to store angles and planes
    residue_angles = {
        'phi': [],
        'psi': [],
        'omega': [],
        'planes': [],
        'CA_C_N_angle': [],
        'N_CA_C_angle': [],
        'C_N_Ca_angle': [],
        'CA_N_length': [],
        'CA_C_length': [],
        'N_C_length': []
    }
    #iterate over residues
    for res in structure.get_residues():
        #check if this is the first residue
        if C_prev is None:
            C_prev = res['C'].get_vector()
            N_prev = res['N'].get_vector()
            CA_prev = res['CA'].get_vector()
            continue
        #if it isn't we can get all the bond lengths and angles:
        else:
            try:
                print(res['CA'].get_vector())
                #get current residue vectors
                N = res['N'].get_vector()
                CA = res['CA'].get_vector()
                C = res['C'].get_vector()
                O = res['O'].get_vector() #just keep if needed
                print(C, N, CA, O)
            except:
                continue
            
            #calculate dihedral angles of curr residue using the reference atoms of previous C, N, CA
            residue_angles['phi'].append(math.radians(Bio.PDB.calc_dihedral(C_prev, N, CA, C)))
            residue_angles['psi'].append(math.radians(Bio.PDB.calc_dihedral(N_prev, CA_prev, C, N)))
            residue_angles['omega'].append(math.radians(Bio.PDB.calc_dihedral(CA_prev, C_prev, N, CA)))
            #calculate the plane
            residue_angles['planes'].append([N, CA, C])
            
            
            #calculate three angles among N, CA, and C
            CA_C_N_angle =math.radians(Bio.PDB.calc_angle(CA_prev, C_prev, N))
            N_CA_C_angle = math.radians(Bio.PDB.calc_angle(N, CA, C))
            C_N_Ca_angle = math.radians(Bio.PDB.calc_angle(C_prev, N, CA))   
            residue_angles['CA_C_N_angle'].append(CA_C_N_angle)
            residue_angles['N_CA_C_angle'].append(N_CA_C_angle)
            residue_angles['C_N_Ca_angle'].append(C_N_Ca_angle)
            
            #get the 3 bond lengths 
            CA_N_length = CA - N
            CA_C_length = CA - C
            N_C_length = N - C_prev #this is the peptide bond, so between two residue
            residue_angles['CA_N_length'].append(CA_N_length)
            residue_angles['CA_C_length'].append(CA_C_length)
            residue_angles['N_C_length'].append(N_C_length)
        
            
            #update the prev values by these new ones 
            C_prev = C
            N_prev = N
            CA_prev = CA
            
        #convert to numpy arrays
        phi_angles = np.vstack(residue_angles['phi'])
        psi_angles = np.vstack(residue_angles['psi'])
        omega_angles = np.vstack(residue_angles['omega'])
        planes = np.vstack(residue_angles['planes'])
        CA_C_N_angle = np.vstack(residue_angles['CA_C_N_angle'])
        N_CA_C_angle = np.vstack(residue_angles['N_CA_C_angle'])
        C_N_Ca_angle = np.vstack(residue_angles['C_N_Ca_angle'])
        CA_N_length = np.vstack(residue_angles['CA_N_length'])
        CA_C_length = np.vstack(residue_angles['CA_C_length'])
        N_C_length = np.vstack(residue_angles['N_C_length'])
        
        return phi_angles, psi_angles, omega_angles, planes, CA_C_N_angle, N_CA_C_angle, C_N_Ca_angle, CA_N_length, CA_C_length, N_C_length


"""
Return: residue sequence, len of residue sequence 
"""
def get_sequence(file):
    for record in SeqIO.parse(file, "pdb-atom"):
        return str(record.seq).upper(), len(record.seq)
    
    """
input: paths to relaxed and pre-relaxed pdb files, and the nubmer of residues in the protein
returns: atom_embs, atom_xyz, atom_relax_xyz, atom_nums, ca_pose
If "relax" not in name, assume it's pre-relax aspect of structure
    """
def get_atom_emb(pre_relax_file, relax_file, num_residues, residue_sequence):
    #structure dfs
    structure = parser.get_structure("input", pre_relax_file)
    structure_target = parser.get_structure("target", relax_file)

    #initialize tensors for atom embeddings and xyz, whihc will be output
    atom_embs = [-1 for _ in range(num_residues[1])]
    atom_xyz = [-1 for _ in range(num_residues[1])]
    atom_relax_xyz = [-1 for _ in range(num_residues[1])]
    #alpha carbons in pre_relax tracked
    ca_pose = []
    #get the atom_pos arrays
    # + a few other residue and atom counts
    for res, res_relax in zip(structure.get_residues(), structure_target.get_residues()):
        if res.id[1] > num_residues[1]  or res.id[1] < 0:
            continue
        atom_pos_pre_relax, one_hot = [] , []
        atom_pos_relax = []
        _residue = res.get_resname()
        atom_nums = []
        #get atom position and one hot encodings for the atoms in the residue
        for _atom in heavy_atoms[_residue]['atoms']:
            #get coord and append to atom_pos_pre_relax
            atom_pos_pre_relax.append(res[_atom].coord)
            atom_pos_relax.append(res_relax[_atom].coord)
            _onehot = np.zeros(len(atom_dict))
            _onehot[atom_dict[_atom]] = 1 #one-hot encoding is pre_relax
            one_hot.append(_onehot)
            atom_nums.append(len(atom_pos_pre_relax))
        ca_coords = res['CA'].coord
        ca_pose.append(ca_coords)
        #create atom emb:  one_hot
        atom_emb =  one_hot #np.concatenate([atom_pos_pre_relax, one_hot], axis=1)
        atom_embs[res.id[1] -1] = np.array(atom_emb).astype(np.float16)
        atom_xyz[res.id[1] -1] = np.array(atom_pos_pre_relax).astype(np.float16)
    return atom_embs, atom_xyz, atom_relax_xyz, atom_nums, ca_pose
        
def residue_onehot(residue_sequence):
    one_hot = np.zeros((len(residue_sequence), len(RESIDUE_TYPES)))
    for i, residue in enumerate(residue_sequence):
        if residue not in RESIDUE_TYPES: residue = 'X'
        one_hot[i, RESIDUE_TYPES.index(residue)] = 1
    return one_hot
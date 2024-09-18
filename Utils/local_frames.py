
import os, pickle
import numpy as np
import Bio.PDB
from Bio import SeqIO
from Bio.PDB.DSSP import DSSP
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")
pdb_parser = Bio.PDB.PDBParser(QUIET = True)

"""
ATOMIC LEVEL
Implementation of ScanNet-inspired reference frame
"""
#TODO 


"""
AtomRefine residue local coordinate system/frame implementation:
RESIDUE LEVEL

"""
def set_lframe(structure_file, atom_xyz, atom_nums, res_range=None):
    '''
    Agrs:
        structure_file full path
        atom_xyz : np.array, shape (n,3)
        atom_nums : list of num atoms per residue, shape (n,)
    '''
    # load pdb file
    structure = pdb_parser.get_structure("tmp_stru", structure_file)
    residues = [_ for _ in structure.get_residues()]

    # the residue num
    res_num = res_range[1]-res_range[0]+1

    # set local frame for each residue in pdb
    pdict = dict()
    pdict['N'] = np.stack([np.array(residue['N'].coord) for residue in residues])
    pdict['Ca'] = np.stack([np.array(residue['CA'].coord) for residue in residues])
    pdict['C'] = np.stack([np.array(residue['C'].coord) for residue in residues])

    # recreate Cb given N,Ca,C
    ca = -0.58273431
    cb = 0.56802827
    cc = -0.54067466
    
    b = pdict['Ca'] - pdict['N']
    c = pdict['C'] - pdict['Ca']
    a = np.cross(b, c)
    #TODO: MAKE BETTER REACHABLE WITHOUT SEVERE PRECISION LOSS...
    pdict['Cb'] = ca * a + cb * b + cc * c

    # local frame
    z = pdict['Cb'] - pdict['Ca']
    z /= np.linalg.norm(z, axis=-1)[:,None]
    x = np.cross(pdict['Ca']-pdict['N'], z)
    x /= np.linalg.norm(x, axis=-1)[:,None]
    y = np.cross(z, x)
    y /= np.linalg.norm(y, axis=-1)[:,None]

    xyz = np.stack([x,y,z])

    pdict['lfr'] = np.transpose(xyz, [1,0,2])
    
    start, end, j = 0, 0, 0
    atom_idx = [-1 for _ in range(atom_xyz.shape[0])]
    for i in range(len(atom_nums)):
        start = end
        end += atom_nums[i]
        atom_idx[start:end] = [j]*atom_nums[i]
        j = j+1
        
    
    p = np.zeros((atom_xyz.shape[0], atom_xyz.shape[0],3))
    q = np.zeros((atom_xyz.shape[0], atom_xyz.shape[0],3))
    k = np.zeros((atom_xyz.shape[0], atom_xyz.shape[0],3))
    t = np.zeros((atom_xyz.shape[0], atom_xyz.shape[0],3))
    for i in range(atom_xyz.shape[0]):
        res_idx = atom_idx[i]
        for j in range(atom_xyz.shape[0]):
            p[i,j,:] = np.matmul(pdict['lfr'][res_idx],atom_xyz[j]-atom_xyz[i])
            q[i,j,:] = np.matmul(pdict['lfr'][atom_idx[i]],pdict['lfr'][atom_idx[j]][0])
            k[i,j,:] = np.matmul(pdict['lfr'][atom_idx[i]],pdict['lfr'][atom_idx[j]][1])
            t[i,j,:] = np.matmul(pdict['lfr'][atom_idx[i]],pdict['lfr'][atom_idx[j]][2])
    
    return p,q,k,t
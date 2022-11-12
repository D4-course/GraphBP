"""
Main code for evaluation
"""

import pickle
import os
from rdkit.Chem import Draw

from rdkit.Chem.rdForceFieldHelpers import UFFOptimizeMolecule
from rdkit import Chem
from rdkit.Chem import AllChem

import numpy as np
import scipy as sp
from utils import BondAdder


# config
SAVE_MOL = True
UFF = True
UFF_W_REC = False  # UFF in the context of binding site
SAVE_SDF_BEFORE_UFF = False
SAVE_SDF = True
DATA_ROOT = './data/crossdock2020'

PATH = './trained_model'
EPOCH = 33

all_mols_dict_path = os.PATH.join(PATH, f'{EPOCH}_mols.mol_dict')


def check_chemical_validity(mol):
    """
    Checks the chemical validity of the mol object. Existing mol object is
    not modified. Radicals pass this test.

    Args:
        mol: Rdkit mol object

    :rtype:
        :class:`bool`, True if chemically valid, False otherwise
    """

    smile = Chem.MolToSmiles(mol, isomericSmiles=True)
    molecule = Chem.MolFromSmiles(smile)  # implicitly performs sanitization
    if molecule:
        return True
    return False


def rd_mol_to_sdf(rd_molecule, sdf_filepath, kekulize=False, name=''):
    """
    Function to get sdf from rd_molecule
    """
    writer = Chem.SDWriter(sdf_filepath)
    writer.SetKekulize(kekulize)
    if name:
        rd_molecule.SetProp('_Name', name)
    writer.write(rd_molecule)
    writer.close()


def get_rd_atom_res_id(rd_atom):
    '''
    Return an object that uniquely
    identifies the residue that the
    atom belongs to in a given PDB.
    '''
    res_info = rd_atom.GetPDBResidueInfo()
    return (
        res_info.GetChainId(),
        res_info.GetResidueNumber()
    )


def get_pocket(lig_mol, rec_molecule, max_dist=8):
    """
    Function to get pocket
    """
    lig_coords = lig_mol.GetConformer().GetPositions()
    rec_coords = rec_molecule.GetConformer().GetPositions()
    dist = sp.spatial.distance.cdist(lig_coords, rec_coords)

    # indexes of atoms in rec_mol that are
    #   within max_dist of an atom in lig_mol
    pocket_atom_idxs = set(np.nonzero((dist < max_dist))[1])

    # determine pocket residues
    pocket_res_ids = set()
    for atom_index in pocket_atom_idxs:
        atom = rec_molecule.GetAtomWithIdx(int(atom_index))
        res_id = get_rd_atom_res_id(atom)
        pocket_res_ids.add(res_id)

    # copy mol and delete atoms
    pkt_mol = rec_molecule
    pkt_mol = Chem.RWMol(pkt_mol)
    for atom in list(pkt_mol.GetAtoms()):
        res_id = get_rd_atom_res_id(atom)
        if res_id not in pocket_res_ids:
            pkt_mol.RemoveAtom(atom.GetIdx())

    Chem.SanitizeMol(pkt_mol)
    return pkt_mol


with open(all_mols_dict_path, 'rb') as f:
    all_mols_dict = pickle.load(f)

bond_adder = BondAdder()


all_results_dict = {}
os.makedirs(
    os.PATH.join(
        PATH,
        'gen_mols' +
        '_epoch_' +
        str(EPOCH) +
        '/'),
    exist_ok=True)

GLOBAL_INDEX = 0
global_index_to_rec_src = {}
global_index_to_ref_lig_src = {}
NUM_VALID = 0
for index in all_mols_dict:
    # print(index)
    mol_dicts = all_mols_dict[index]
    for num_atom in mol_dicts:
        if isinstance(num_atom,int):
            mol_dicts_w_num_atom = mol_dicts[num_atom]
            num_mol_w_num_atom = len(mol_dicts_w_num_atom['_atomic_numbers'])
            for j in range(num_mol_w_num_atom):
                GLOBAL_INDEX += 1

                # Add bonds
                atomic_numbers = mol_dicts_w_num_atom['_atomic_numbers'][j]
                positions = mol_dicts_w_num_atom['_positions'][j]
                rd_mol, ob_mol = bond_adder.make_mol(atomic_numbers, positions)

                # check validity
                if check_chemical_validity(rd_mol):
                    NUM_VALID += 1
                print('Valid molecules:', NUM_VALID)

                rd_mol = Chem.AddHs(rd_mol, explicitOnly=True, addCoords=True)
                if SAVE_SDF_BEFORE_UFF:
                    sdf_file = os.PATH.join(
                        PATH, 'gen_mols' + '_epoch_' + str(EPOCH) + '/' +
                        str(GLOBAL_INDEX) + '_beforeuff.sdf')
                    rd_mol_to_sdf(rd_mol, sdf_file)
                    print('Saving' + str(sdf_file))

                # ### UFF minimization
                if UFF:
                    try:
                        # print(rd_mol.GetConformer().GetPositions())
                        UFFOptimizeMolecule(rd_mol)
                        print("Performing UFF...")
                        # print(rd_mol.GetConformer().GetPositions())
                    except BaseException:
                        print('Skip UFF...')
                        # pass

                if UFF_W_REC:
                    # try:
                    # print(rd_mol.GetConformer().GetPositions())
                    # print(rd_mol.GetConformer().GetPositions().shape)
                    rd_mol = Chem.RWMol(rd_mol)
                    rec_mol = Chem.MolFromPDBFile(
                        os.PATH.join(
                            DATA_ROOT,
                            mol_dicts['rec_src']),
                        sanitize=True)
                    rec_mol = get_pocket(rd_mol, rec_mol)

                    uff_mol = Chem.CombineMols(rec_mol, rd_mol)

                    # print(uff_mol.GetConformer().GetPositions()[:-rd_mol.GetNumAtoms()])
                    # print(uff_mol.GetConformer().GetPositions()[:-rd_mol.GetNumAtoms()].shape)

                    try:
                        Chem.SanitizeMol(uff_mol)
                    except Chem.AtomValenceException:
                        print('Invalid valence')
                    except (Chem.AtomKekulizeException, Chem.KekulizeException):
                        print('Failed to kekulize')
                    try:
                        # UFFOptimizeMolecule(uff_mol)
                        UFF = AllChem.UFFGetMoleculeForceField(
                            uff_mol, confId=0, ignoreInterfragInteractions=False)
                        UFF.Initialize()
                        # E_init = UFF.CalcEnergy()
                        for i in range(
                                rec_mol.GetNumAtoms()):  # Fix the rec atoms
                            UFF.AddFixedPoint(i)
                        CONVERGED = False
                        N_ITERS = 200
                        N_TRIES = 2
                        while N_TRIES > 0 and not CONVERGED:
                            print('.', end='', flush=True)
                            CONVERGED = not UFF.Minimize(maxIts=N_ITERS)
                            N_TRIES -= 1
                        print(flush=True)
                        # E_final = UFF.CalcEnergy()
                        print("Performed UFF with binding site...")
                    except BaseException:
                        print('Skip UFF...')
                    coords = uff_mol.GetConformer().GetPositions()
                    rd_conf = rd_mol.GetConformer()
                    for i, xyz in enumerate(coords[-rd_mol.GetNumAtoms():]):
                        rd_conf.SetAtomPosition(i, xyz)
                    # print(rd_mol.GetConformer().GetPositions())
                    # print(rd_mol.GetConformer().GetPositions().shape)
                    # print(uff_mol.GetConformer().GetPositions()[:-rd_mol.GetNumAtoms()])
                    # print(uff_mol.GetConformer().GetPositions()[:-rd_mol.GetNumAtoms()].shape)
                    # print(E_init, E_final)

                if SAVE_SDF:

                    ###
                    try:
                        rd_mol = Chem.RemoveHs(rd_mol)
                        print("Remove H atoms before saving mol...")
                    except BaseException:
                        print("Cannot remove H atoms...")

                    sdf_file = os.PATH.join(
                        PATH, 'gen_mols' + '_epoch_' + str(EPOCH) + '/' +
                        str(GLOBAL_INDEX) + '.sdf')
                    rd_mol_to_sdf(rd_mol, sdf_file)
                    print('Saving' + str(sdf_file))
                    global_index_to_rec_src[GLOBAL_INDEX] = mol_dicts['rec_src']
                    global_index_to_ref_lig_src[GLOBAL_INDEX] = mol_dicts['lig_src']

                if SAVE_MOL:
                    try:
                        img_path = os.PATH.join(
                            PATH, 'gen_mols' + '_epoch_' + str(EPOCH) + '/' +
                            str(GLOBAL_INDEX) + '.png')
                        img = Draw.MolsToGridImage([rd_mol])
                        img.save(img_path)
                        print('Saving' + str(img_path))
                    except BaseException:
                        pass
                print('------------------------------------------------')
        else:
            continue

if SAVE_SDF:
    print('Saving dicts...')
    with open(os.PATH.join(PATH, \
        'gen_mols_epoch_{}/global_index_to_rec_src.dict').format(EPOCH), 'wb') as f:
        pickle.dump(global_index_to_rec_src, f)
    with open(os.PATH.join(PATH, \
        'gen_mols_epoch_{}/global_index_to_ref_lig_src.dict').format(EPOCH), 'wb') as f:
        pickle.dump(global_index_to_ref_lig_src, f)

print('Done!!!')

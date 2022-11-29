"""
Main code for generation
"""
import pickle
import torch
from torch import device
from config import conf
from runner import Runner

runner = Runner(conf)

KNOWN_BINDING_SITE = True


NODE_TEMP = 0.5
DIST_TEMP = 0.3
ANGLE_TEMP = 0.4
TORSION_TEMP = 1.0

MIN_ATOMS = 10
MAX_ATOMS = 45
FOCUS_TH = 0.5
CONTACT_TH = 0.5
NUM_GEN = 10  # number generate for each reference rec-lig pair

TRAINED_MODEL_PATH = 'trained_model'
epochs = [33]

def molecularDictGen():
    for epoch in epochs:
        print('Epoch:', epoch)
        runner.model.load_state_dict(
            torch.load(
                f'{TRAINED_MODEL_PATH}/model_{epoch}.pth',
                map_location=device("cpu")))
        all_mol_dicts = runner.generate(
            NUM_GEN,
            temperature=[
                NODE_TEMP,
                DIST_TEMP,
                ANGLE_TEMP,
                TORSION_TEMP],
            max_atoms=MAX_ATOMS,
            min_atoms=MIN_ATOMS,
            focus_th=FOCUS_TH,
            contact_th=CONTACT_TH,
            add_final=True,
            known_binding_site=KNOWN_BINDING_SITE)

        with open('{TRAINED_MODEL_PATH}/{epoch}_mols.mol_dict', 'wb') as f:
            pickle.dump(all_mol_dicts, f)

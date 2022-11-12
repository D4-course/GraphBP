import pytest
import torch

from runner import Runner
from config import conf


def test_generate():	

	runner.model.load_state_dict(
        torch.load(
            f'{"trained_model"}/model_{33}.pth',
            map_location=torch.device("cpu")))

	runner = Runner(conf)
	all_mol_dicts = runner.generate(
        5,
        temperature=[
            0.5,
            0.3,
            0.4,
            1.0],
        max_atoms=20,
        min_atoms=2,
        focus_th=0.5,
        contact_th=0.5,
        add_final=True,
		data_file='./data/crossdock2020/types/it2_tt_v1.1_10p20n_test0.types',
        known_binding_site=True)

	assert 'all_mol_dicts' in locals(), "test failed"
	assert all_mol_dicts is not None, "test failed"
	assert all_mol_dicts != None, "test failed"


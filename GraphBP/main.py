"""
Main code for GraphBP
"""
import os
from config import conf
from runner import Runner


BINDING_SITE_RANGE = 15.0


OUT_PATH = 'trained_model'
if not os.path.isdir(OUT_PATH):
    os.mkdir(OUT_PATH)

runner = Runner(conf, out_path=OUT_PATH)
runner.train(BINDING_SITE_RANGE)

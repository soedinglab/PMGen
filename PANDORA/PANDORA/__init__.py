###  ###
import os
from os.path import exists
from pathlib import Path
import json

version='2.0.0'
PANDORA_path = os.path.dirname(os.path.abspath(__file__))
module_path = Path(__file__).parents[0]

'''
Path to PANDORA installation
:meta hide-value:
'''

if exists(module_path / 'config.json'):
    with open(module_path / 'config.json') as f:
        data = json.load(f)
        data_folder = data['data_folder_name']
else:
    data_folder = '~/PANDORA_databases/default'

PANDORA_data = os.path.expanduser(data_folder)

alpha_genes = ['HLA-A', 'HLA-B', 'HLA-C', 'HLA-E', 'HLA-F', 'HLA-G',
                'HLA-DQA', 'HLA-DRA', 'HLA-DPA',
                'H2-', 'MH2-', 'MH1-']
beta_genes = ['HLA-DQB', 'HLA-DRB', 'HLA-DPB',
               'H2-EB', 'MH2-AB', 'H2-AB']

MHCI_G_domain=[(1,187)] #182 + 5 for tolerance
MHCII_G_domain=[(1,86),(1,95)] #81 + 5 and 90 + 5 for tolerance

# populate package namespace
from PANDORA.Contacts import Contacts
from PANDORA.PMHC import Model
from PANDORA.Pandora import Modelling_functions
from PANDORA.PMHC import Anchors
from PANDORA.PMHC.PMHC import Template
from PANDORA.Database import Database_functions
from PANDORA.Database import Database
from PANDORA.Pandora import Align
from PANDORA.Pandora import Pandora
from PANDORA.PMHC.PMHC import Target
from PANDORA.PMHC.PMHC import PMHC
from PANDORA.Wrapper import Wrapper
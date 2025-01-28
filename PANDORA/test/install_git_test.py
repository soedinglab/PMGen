import os
from os.path import exists
from pathlib import Path
import json

user_folder_path = Path(__file__).parents[1]
print(user_folder_path)

data_folder = './PANDORA_databases/test'

with open(f'{user_folder_path}/PANDORA/config.json', 'w') as f:
   f.write('{"data_folder_name" : "%s"}' %data_folder)

# if exists(user_folder_path / 'config.json'):
#     with open(user_folder_path / 'config.json') as f:
#         data = json.load(f)
#         data_folder = data['data_folder_name']
# else:
# data_folder = 'test'

os.system('mkdir ./PANDORA_databases')
os.system(f'mkdir {data_folder}')
os.system(f'mkdir {data_folder}/database')
os.system(f'mkdir {data_folder}/PDBs')
os.system(f'mkdir {data_folder}/mhcseqs') 
os.system(f'mkdir {data_folder}/BLAST_databases') 
#os.system(f'mkdir ./Databases/{data_folder}/PDBs/pMHCI')
#os.system(f'mkdir ./Databases/{data_folder}/PDBs/pMHCII')
os.system(f'mkdir {data_folder}/PDBs/Bad')
os.system(f'mkdir {data_folder}/PDBs/Bad/pMHCI')
os.system(f'mkdir {data_folder}/PDBs/Bad/pMHCII')
os.system(f'mkdir {data_folder}/PDBs/IMGT_retrieved/')

#os.system('mkdir ./test/test_data/PDBs/Bad')
#os.system('mkdir ./test/test_data/PDBs/Bad/pMHCI')
#os.system('mkdir ./test/test_data/PDBs/Bad/pMHCII')
os.system(f'cp -r ./test/test_data/PDBs/* {data_folder}/PDBs/')
#os.system(f'cp ./test/test_data/PANDORA_database.pkl {data_folder}/PANDORA_database.pkl')

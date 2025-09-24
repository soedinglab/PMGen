import os
import pandas as pd
import shutil

folders = [i for i in os.listdir('./') if '.pdb' in i]
for folder in folders:
    #file = os.path.join(folder, [i for i in os.listdir(folder) if '.pdb' in i][0])
    #shutil.copy(file, f'../affine/{folder}_affine.pdb')
    file = folder.split('_')[0]
    shutil.copy(folder, f'../affine/{file}_affine.pdb')

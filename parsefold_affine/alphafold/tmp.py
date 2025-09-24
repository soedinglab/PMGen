import os
import pandas as pd
import shutil

folders = [i for i in os.listdir('./') if os.path.isdir(i)]
for folder in folders:
    file = os.path.join(folder, [i for i in os.listdir(folder) if '.pdb' in i][0])
    shutil.copy(file, f'../parsefold+affine/{folder}_parsefold+affine.pdb')
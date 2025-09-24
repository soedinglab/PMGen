import pandas as pd
import os
files = [i for i in os.listdir('./') if '.tsv' in i]
for file in files:
    df = pd.read_csv(file, sep='\t', header=0)
    df = df.iloc[0:1, :]
    '''
    pdb = file.split('_')[0]
    iden = int(df['target_len'][0])
    target_to_template_alignstring = [';'.join([f'{i}:{i}' for i in range(iden)]),
                                      ';'.join([f'{i}:{i}' for i in range(iden)]),
                                      df['target_to_template_alignstring'][2]]
    template_len = [iden, iden, int(target_to_template_alignstring[-1].split(':')[-1]) + 1]
    template_pdbfile = [f'predictions/alphafold_finetune/multimer2.2/{pdb}_multimer2.2.pdb',
                        f'predictions/alphafold_finetune/multimer2.3/{pdb}_multimer2.3.pdb',
                        f'predictions/parsefold_affine/pandora/{pdb}/mod1.pdb']
    df['template_pdbfile'] = template_pdbfile
    df['target_to_template_alignstring'] = target_to_template_alignstring
    df['template_len'] = template_len
    df = df.iloc[[0, -1], :]
    '''
    df.to_csv(file, sep='\t', index=False)
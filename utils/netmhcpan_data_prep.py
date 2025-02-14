
import pandas as pd
import os
import re
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from processing_functions import BA_raw_processing, HLA_annotation, merge_mhc_data, filter_fasta_by_length, get_aminoacid_sequences

#exclude inputs from previous csv
list_to_keep = ['name','sequence','id','BA','peptide','Label','type']
other1 = pd.read_csv("/scratch/users/a.hajialiasgarynaj01/Master_thesis/Pipline_test_PANDORA_AFoptimization/Training/inputs/Updated_netmhcpan.tsv", sep="\t", header=0)[list_to_keep]
other2 = pd.read_csv("/scratch/users/a.hajialiasgarynaj01/Master_thesis/Pipline_test_PANDORA_AFoptimization/Training_2/inputs/Updated_netmhcpan.tsv", sep="\t", header=0)[list_to_keep]
other3 = pd.read_csv("/scratch/users/a.hajialiasgarynaj01/Master_thesis/Pipline_test_PANDORA_AFoptimization/Training_3/inputs/Updated_netmhcpan.tsv", sep="\t", header=0)[list_to_keep]
other4 = pd.read_csv("/scratch/users/a.hajialiasgarynaj01/Master_thesis/Pipline_test_PANDORA_AFoptimization/data/pre_input/round_4/concatenated_filtered.tsv", sep="\t", header=0)[list_to_keep]
other = pd.concat([other1,other2,other3,other4])
other['Allele']=other['id']
print(other)
print("previous files to exclude, len",len(other))

##################
#filter_fasta_by_length()
if not os.path.exists("/scratch/users/a.hajialiasgarynaj01/Master_thesis/Pipline_test_PANDORA_AFoptimization/data/pre_input/round_5/"):
    os.mkdir("/scratch/users/a.hajialiasgarynaj01/Master_thesis/Pipline_test_PANDORA_AFoptimization/data/pre_input/round_5/")

DFS = []
for i in range(1,6):
    df = pd.read_csv(f"/scratch/users/a.hajialiasgarynaj01/Master_thesis/NetMHCpan_data/NetMHCIIpan_train/train_BA{i}.txt",
                     sep="\t", header=None)
    DFS.append(df)
DFS = pd.concat(DFS, ignore_index=True)
print("MHC2 len:",len(DFS))

# in pos_portion=None, takes the whole data for you and excludes the  non allel_types
netmhc_pan_dataframe = BA_raw_processing(input_df=DFS, Allele_type=["DR","DQ","DO", "DP","DM"], num_queries=360000,
                      Allele_col=2, BA_col=1, peptide_col=0, pos_portion=None,  random_state=None)
print(f'MHC2 before filter: {len(netmhc_pan_dataframe)}')
# Identify rows to remove from df2 based on conditions
condition = ~netmhc_pan_dataframe[['peptide', "Allele"]].apply(tuple, axis=1).isin(other[['peptide', "Allele"]].apply(tuple, axis=1))                  
netmhc_pan_dataframe = netmhc_pan_dataframe[condition].reset_index(drop=True)
print(f'after filter: {len(netmhc_pan_dataframe)}')

mhc_2_dict, NF= HLA_annotation(df=netmhc_pan_dataframe, MHC_type=2)
aaa = pd.DataFrame(mhc_2_dict)
aaa.to_csv("/scratch/users/a.hajialiasgarynaj01/Master_thesis/Pipline_test_PANDORA_AFoptimization/data/pre_input/round_5/netmhcpan_2_processed_df.csv",sep="\t",index=False)
NF.to_csv("/scratch/users/a.hajialiasgarynaj01/Master_thesis/Pipline_test_PANDORA_AFoptimization/data/pre_input/round_5/notfound_2.csv",sep="\t",index=False)

###################

DFS = []
for i in range(5):
    df = pd.read_csv(f"/scratch/users/a.hajialiasgarynaj01/Master_thesis/NetMHCpan_data/NetMHCpan4.0/f00{i}_ba", sep=" ",
                     header=None)
    DFS.append(df)
DFS = pd.concat(DFS, ignore_index=True)
print(DFS)
netmhc_pan_dataframe = BA_raw_processing(input_df=DFS, Allele_type=["A","B","C"], num_queries=320000,
                      Allele_col=2, BA_col=1, peptide_col=0, pos_portion=None)

print(f'MHC1 before filter: {len(netmhc_pan_dataframe)}')
condition = ~netmhc_pan_dataframe[['peptide', 'BA', "Allele"]].apply(tuple, axis=1).isin(other[['peptide', 'BA', "Allele"]].apply(tuple, axis=1))                     
netmhc_pan_dataframe = netmhc_pan_dataframe[condition].reset_index(drop=True)
print(f'after filter: {len(netmhc_pan_dataframe)}')

mhc_2_dict, NF= HLA_annotation(df=netmhc_pan_dataframe, MHC_type=1)
aaa = pd.DataFrame(mhc_2_dict)
aaa.to_csv("/scratch/users/a.hajialiasgarynaj01/Master_thesis/Pipline_test_PANDORA_AFoptimization/data/pre_input/round_5/netmhcpan_1_processed_df.csv",sep="\t",index=False)
NF.to_csv("/scratch/users/a.hajialiasgarynaj01/Master_thesis/Pipline_test_PANDORA_AFoptimization/data/pre_input/round_5/notfound_1.csv",sep="\t",index=False)


### Concatenate
df_1 = pd.read_csv("/scratch/users/a.hajialiasgarynaj01/Master_thesis/Pipline_test_PANDORA_AFoptimization/data/pre_input/round_5/netmhcpan_1_processed_df.csv", sep="\t", header=0)
df_2 = pd.read_csv("/scratch/users/a.hajialiasgarynaj01/Master_thesis/Pipline_test_PANDORA_AFoptimization/data/pre_input/round_5/netmhcpan_2_processed_df.csv", sep="\t", header=0)
df_1["type"] = [1] * len(df_1.index)
df_2["type"] = [2] * len(df_2.index)
DF = pd.concat([df_1,df_2]).reset_index(drop=True)
DF.to_csv("/scratch/users/a.hajialiasgarynaj01/Master_thesis/Pipline_test_PANDORA_AFoptimization/data/pre_input/round_5/concatenated_filtered.tsv", sep="\t", index=False)



















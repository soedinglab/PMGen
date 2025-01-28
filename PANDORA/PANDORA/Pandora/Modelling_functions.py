from Bio.Align import substitution_matrices
import os
import traceback
import subprocess
import PANDORA
import pickle
from PANDORA import Model
# from Bio import Align
from Bio import pairwise2
#from PANDORA import Align
#import statistics
from Bio.Align import PairwiseAligner
from datetime import datetime
from copy import deepcopy
import re


def check_target_template(target, template):
    """ Checks if the target and the template are the same. If the user gave sequence info in the target, use that, else
        use the allele type.

    Args:
        target: (:obj:`Target`): Target object
        template: (:obj:`Template`): Template object

    Returns: (bool): True if target/template are the same, False if they are not.

    """
    out = False
    # Check if target peptide and template peptide are the same
    if target.peptide == template.peptide:
        # If the target has no sequence information, use allele type
        if target.M_chain_seq == '':
            # Check if the allele of target and template are the same
            if any(x in template.allele_type for x in target.allele_type):
                out = True

        # If the target has sequence information..
        elif target.M_chain_seq != '':
            # For MHCI, check if the M chain sequence of target and template are the same
            if target.MHC_class == 'I':
                if target.M_chain_seq == template.M_chain_seq:
                    out = True
            # For MHCII, check if the M and N chain sequence of target and template are the same
            elif target.MHC_class == 'II' and target.N_chain_seq != '':
                if target.M_chain_seq == template.M_chain_seq and target.N_chain_seq == template.N_chain_seq:
                    out = True
    if out:
        print('\n\t---- THE TARGET HAS THE SAME PEPTIDE AND ALLELE/SEQUENCE INFORMATION AS THE TEMPLATE ----')
        print('\tYou can find it at: http://www.imgt.org/3Dstructure-DB/cgi/details.cgi?pdbcode=%s\n' %(template.id))

    return out


def check_presence(target, database, seq_based_templ_selection = False):
    ''' Checks if the target the user submitted, already exists in has a template in the database with the same allele
        and peptide.

    Args:
        target: Target object
        database: Database object
        seq_based_templ_selection: bool, select the template based on the chain sequences.

    Returns: bool/Template object. If the target is already in the db, return the Template, otherwise return False

    '''
    putative_templates = []
    target_in_db = False
    if not seq_based_templ_selection:
        # For MHC I
        if target.MHC_class == 'I':
            # Check if there are templates with the same alleles
            for id in database.MHCI_data:
                if any(x in database.MHCI_data[id].allele_type for x in target.allele_type):
                    putative_templates.append(id)
            # Check if there is a putative template that also has the same peptide as the target
            for i in putative_templates:
                if database.MHCI_data[i].peptide == target.peptide:
                    target_in_db = database.MHCI_data[i]
        # For MHC II
        elif target.MHC_class == 'II':
            # Check if there are templates with the same alleles
            for id in database.MHCII_data:
                if any(x in database.MHCII_data[id].allele_type for x in target.allele_type):
                    putative_templates.append(id)
            # Check if there is a putative template that also has the same peptide as the target
            for i in putative_templates:
                if database.MHCII_data[i].peptide == target.peptide:
                    target_in_db = database.MHCII_data[i]

    elif seq_based_templ_selection:
        # Check for MHC I
        if target.MHC_class == 'I':
            # Check if there are templates with the same M chain sequence
            for id in database.MHCI_data:
                if database.MHCI_data[id].M_chain_seq == target.M_chain_seq:
                    putative_templates.append(id)
            # Check if there is a putative template that also has the same peptide as the target
            for i in putative_templates:
                if database.MHCI_data[i].peptide == target.peptide:
                    target_in_db = database.MHCI_data[i]
        # Check for MHC I
        if target.MHC_class == 'II':
            # Check if there are templates with the same M chain sequence
            for id in database.MHCII_data:
                if database.MHCII_data[id].M_chain_seq == target.M_chain_seq:
                    if database.MHCII_data[id].N_chain_seq == target.N_chain_seq:
                        putative_templates.append(id)
            # Check if there is a putative template that also has the same peptide as the target
            for i in putative_templates:
                if database.MHCII_data[i].peptide == target.peptide:
                    target_in_db = database.MHCII_data[i]

    return target_in_db


def predict_anchors_netMHCIIpan(peptide, allele_type, output_dir, verbose=True, rm_netmhcpan_output=True):
    '''Uses netMHCIIpan to predict the binding core of a peptide and infer the anchor positions from that.

    Args:
        peptide: (str): AA sequence of the peptide
        allele_type: (lst): list of strings of allele types
        output_dir: (string) Path to output directory 
        verbose: (bool): Print information. Default = True
        rm_netmhcpan_output: (bool): If True, removes the netmhcpan infile and outfile after having used them for netmhcpan.

    Returns: (lst): list of predicted anchor predictions

    '''
    
    # Retrieves the enviroment variable netMHCIIpan
    netmhcpan_file_path = set([x for x in [os.getenv('netMHCIIpan', default=None), 
                          os.popen('which netMHCIIpan').read().strip()] 
                          if type(x) == str])
    try:
        netmhcpan_file_path = netmhcpan_file_path.pop()
    except:
        raise Exception("Need netMHCIIpan to predict anchor positions. Please download and install netMHCIIpan.\n\n"
        "You can request the software at https://services.healthtech.dtu.dk/service.php?NetMHCpan-4.1 in the 'Downloads' section.\n"
        "After installing netMHCpan, make sure it's added to your PATH or as an alias to your .bashrc / .bash_profile.\n")
        
    netmhcpan_path = os.path.dirname(netmhcpan_file_path)

    all_netMHCpan_alleles = []
    with open(os.path.join(netmhcpan_path, 'data/allelelist.txt')) as f:
        for line in f:
            all_netMHCpan_alleles.append(line.split()[0].replace('\n', ''))

    # Format the alles to netMHCIIpan readable format
    target_alleles = [i.split('-')[-1].replace('*', '_') for i in allele_type]

    # The DQ and DP alleles only function in pairs in netMHCIIpan, which we cannot match from our alleles
    # So take the first 3 partially matched allele combinations
    for i in target_alleles:
        if 'DRB' not in i:
            target_alleles = target_alleles + [al for al in all_netMHCpan_alleles if i.replace('_', '') in al][:3]

    # for the DQ and DP cases, alleles are matched (e.g. 'HLA-DQA10102-DQB10602')
    # If two alleles are present is such a combi case, select that combi case as the target allele
    target_alleles_matched = []
    for al in target_alleles:
        hits = 0
        for part in [i.split('*')[-1] for i in allele_type]:
            if part in al:
                hits +=1
        if hits == 2:
            target_alleles_matched.append(al)
    if len(target_alleles_matched) > 0:
        target_alleles = target_alleles_matched

    target_alleles = [i for i in target_alleles if i in all_netMHCpan_alleles]
    # If there are no target alleles that occur in netMHCIIpan, but there is a mouse allele, use all mouse alleles
    # that are supported by netMHCIIpan
    if target_alleles == [] and any(al.startswith('H2') for al in allele_type):
        target_alleles = [i for i in all_netMHCpan_alleles if i.startswith('H-')]

    # If there is no target allele that occurs in netMHCIIpan, just use the standard DRB1_0101
    if target_alleles == []:
        target_alleles = ['DRB1_0101']

    target_alleles_str = ','.join(target_alleles)

    # Setup files
    infile = os.path.join(output_dir,f'{peptide}_{target_alleles[0].replace("*","").replace(":","")}_{datetime.today().strftime("%Y%m%d_%H%M%S")}.txt')
    outfile = os.path.join(output_dir, f'{peptide}_{target_alleles[0].replace("*","").replace(":","")}_{datetime.today().strftime("%Y%m%d_%H%M%S")}_prediction.txt')

    # Write peptide sequence to input file for netMHCIIpan
    with open(infile, 'w') as f:
        f.write(peptide)

    try:
        # run netMHCIIpan
        subprocess.check_call('%s -f %s -inptype 1 -a %s > %s' % (netmhcpan_file_path, infile, target_alleles_str, outfile), shell=True)

        # Get the output from the netMHCIIpan prediction
        # {allele: (offset, core, core_reliability, score_EL, %rank_EL)}
        pred = {}
        with open(outfile) as f:
            for line in f:
                if peptide in line and not line.startswith('#'):
                    ln = [i for i in line[:-1].split(' ') if i != '']
                    pred[ln[1]] = (int(ln[3]), ln[4], float(ln[5]))

        # For each predicted core offset, show the best prediction
        max_scores = [max((i[::-1]) for i in list(pred.values()) if i[0] == s) for s in set([pred[i][0] for i in pred])]
        # order to offset, core, core_reliability
        max_scores = [i[::-1] for i in sorted(max_scores, reverse=True)]

    except ValueError:
        print('Could not predict binding core using netMHCIIpan. Will use the most common anchor positions instead')
        return [3, 6, 8, 11]

    offset, core, core_reliability = max_scores[0]
    # Use the canonical spacing for 9-mer binding cores to predict the anchor positions
    predicted_anchors = [offset + 1, offset + 4, offset + 6, offset + 9]
    # Make sure the prediction is not longer than the peptide just in case
    predicted_anchors = [i for i in predicted_anchors if i <= len(peptide)]

    if verbose:
        print('\tPredicted the binding core using netMHCIIpan (4.0):\n')
        print('\toffset:\t%s\n\tcore:\t%s\n\tprob:\t%s\n' % (offset, core, core_reliability))
        print('\tPredicted peptide anchor residues (assuming canonical spacing): %s' % predicted_anchors)

    if rm_netmhcpan_output:
        subprocess.check_call('rm %s' %infile, shell=True)
        subprocess.check_call('rm %s' %outfile, shell=True)

    return predicted_anchors
   

def predict_anchors_netMHCpan(peptide, allele_type, output_dir, verbose=True, rm_netmhcpan_output=True):
    '''Uses netMHCIpan to predict the binding core of a peptide and infer the
    anchor positions from that.

    Args:
        peptide: (str): AA sequence of the peptide
        allele_type: (lst): list of strings of allele types
        output_dir: (string) Path to output directory
        verbose: (bool): Print information. Default = True
        rm_netmhcpan_output: (bool): If True, removes the netmhcpan infile and outfile after having used them for netmhcpan.

    Returns: (lst): list of predicted anchor predictions

    '''
    
    # Retrieves the enviroment variable netMHCpan
    netmhcpan_file_path = set([x for x in [os.getenv('netMHCpan', default=None), 
                          os.popen('which netMHCpan').read().strip()] 
                          if type(x) == str])
    try:
        netmhcpan_file_path = netmhcpan_file_path.pop()
    except:
        raise Exception("Need netMHCpan to predict anchor positions. Please download and install netMHCpan.\n\n"
        "You can request the software at https://services.healthtech.dtu.dk/service.php?NetMHCpan-4.1 in the 'Downloads' section.\n"
        "After installing netMHCpan, make sure it's added to your PATH or as an alias to your .bashrc / .bash_profile.\n")
    
    netmhcpan_path = os.path.dirname(netmhcpan_file_path)

    all_netMHCpan_alleles = []
    with open(os.path.join(netmhcpan_path, 'data/allelenames')) as f:
        for line in f:
            all_netMHCpan_alleles.append(line.split()[0])#.replace(':',''))
        
    ## Format alleles
    if any(x.startswith('HLA') for x in allele_type):
        target_alleles = [i.replace('*','') for i in allele_type]
    elif any(x.startswith('BoLA') for x in allele_type):
        target_alleles = [i.replace(':','').replace('*',':') for i in allele_type]
    elif any(x.startswith('DLA') for x in allele_type):
        target_alleles = [i.replace(':','').replace('*','') for i in allele_type]
    elif any(x.startswith('Eqca') for x in allele_type):
        target_alleles = [i.replace(':','').replace('*','') for i in allele_type]
    elif any(x.startswith('Gogo') for x in allele_type):
        target_alleles = [i.replace(':','').replace('*','') for i in allele_type]
    elif any(x.startswith('Mamu') for x in allele_type):
        target_alleles = [i.replace(':','').replace('*',':') for i in allele_type]
    elif any(x.startswith('Patr') for x in allele_type):
        target_alleles = [i.replace(':','').replace('*','') for i in allele_type]
    elif any(x.startswith('SLA') for x in allele_type):
        target_alleles = [i.replace(':','').replace('*',':') for i in allele_type]
    
    ## Make sure only netMHCpan available alleles are used
    target_alleles = [i for i in target_alleles if i in all_netMHCpan_alleles]
    
    if len(target_alleles) == 0:
        print('ERROR: The provided Target allele is not available in NetMHCpan-4.1')
        return None
        
    target_alleles_str = ','.join(target_alleles)
        
    # Setup files
    infile = os.path.join(output_dir,f'{peptide}_{target_alleles[0].replace("*","").replace(":","")}_{datetime.today().strftime("%Y%m%d_%H%M%S")}.txt')
    outfile = os.path.join(output_dir, f'{peptide}_{target_alleles[0].replace("*","").replace(":","")}_{datetime.today().strftime("%Y%m%d_%H%M%S")}_prediction.txt')

    # Write peptide sequence to input file for netMHCIIpan
    with open(infile, 'w') as f:
        f.write(peptide)

    subprocess.check_call('%s -p %s -a %s > %s' %(netmhcpan_file_path, infile, target_alleles_str, outfile), shell=True)
        
    # Get the output from the netMHCIIpan prediction
    # {allele: (core, %rank_EL)}
    pred = {}
    with open(outfile) as f:
        for line in f:
            if peptide in line and not line.startswith('#'):
                ln = [i for i in line[:-1].split(' ') if i != '']
                #ln[3] is core, ln[9] is Icore
                try:
                    pred[ln[1]].append((ln[3], float(ln[12])))
                except KeyError:
                    pred[ln[1]] = [(ln[3], float(ln[12]))]

    # Sort each allele result per Rank_EL
    for allele in pred:
        pred[allele] = list(sorted(pred[allele], key=lambda x:x[1]))
        
    if len(pred) == 0:
        print('ERROR: NetMHCpan-4.1 was not able to find any binding core for')
        print('the provided peptide and MHC allele')
        return None
        
    # For every allele, the binding core is predicted. Take the allele with the highest reliability score
    best_allele = min((pred[i][0][1], i) for i in pred)[1]
        
    # Do a quick alignment of the predicted core and the peptide to find the anchors. (the predicted binding core can
    # contain dashes -. Aligning them makes sure you take the right residue as anchor.
    alignment = pairwise2.align.globalxx(peptide, pred[best_allele][0][0])
        
    #If there are multiple possible solutions, take the one with no gap at the anchor (second) position
    if len(alignment)>1:
        flag = False
        #Search for the options without gap in the second postions
        for prediction in alignment:
            if prediction[1][1] != '-' and prediction[0][1] != '-':
                pept1 = prediction[0]
                pept2 = prediction[1]
                flag = True
                break
        #If no options are available, take the first one
        if flag==False:
            pept1 = alignment[0][0]
            pept2 = alignment[0][1]
            
    else:
        pept1 = alignment[0][0]
        pept2 = alignment[0][1]
        
    # Remove gaps if in the same position
    to_remove = []
    for i, (aa1, aa2) in enumerate(zip(pept1, pept2)):
        if aa1 == aa2 == '-' and i != 0:
            to_remove.append(i)
    for x in reversed(to_remove):
        pept1 = pept1[0:x:]+pept1[x+1::]
        pept2 = pept2[0:x:]+pept2[x+1::]

    if verbose:
        print('Query peptide aligned to the core:')
        print(pept1)
        print(pept2)
        
    # Find the anchors by finding the first non dash from the left and from the right
    # Define chanonical ancors as starting list
    predicted_anchors = [2,len(peptide)]

    # Find the first anchor
    p1 = 0
    p2 = 0
    for i in range(len(pept2)):
        # if the second position has no gaps
        if i == 1 and pept2[i] != '-' and pept1[i] != '-':
            predicted_anchors[0] = p1 + 1
            break
        elif i > 1 and pept2[i] != '-':
            predicted_anchors[0] = p1 + 1
            break
        if pept1[i] != '-':
            p1 += 1
        if pept2[i] != '-':
            p2 += 1
        
    # Find the second anchor
    for i in range(len(pept2)):
        if pept2[::-1][i] != '-':
            predicted_anchors[1] = len([j for j in pept1[:len(pept1) -i] if j != '-'])
            #predicted_anchors[1] = len([j for j in pept2[::-1][i] if j != '-'])
            break
        
    if verbose:
        print('\tPredicted the binding core using netMHCpan (4.1):\n')
        print('\tIcore:\t%s\n\t%%Rank EL:\t%s\n' %(pred[best_allele][0][0], pred[best_allele][0][1] ))
        print('\tPredicted peptide anchor residues (assuming canonical spacing): %s' %predicted_anchors)
        
    if rm_netmhcpan_output:
        subprocess.check_call('rm %s' %infile, shell=True)
        subprocess.check_call('rm %s' %outfile, shell=True)
    
    return predicted_anchors


def score_peptide_alignment(target, template, substitution_matrix='PAM30'):
    ''' Calculate the alignment score of the target and template peptide

    Args:
        target: (Target): Target object
        template: (Template): Template object
        substitution_matrix: (str): name of subtitution matrix, default is PAM30 (BLOSUM80 etc)

    Returns: (flt): alignment score

    '''
    if target.MHC_class == 'I':
        substitution_matrix = substitution_matrices.load(substitution_matrix)
        score = 0
        try:
            pept_anchs = target.anchors
        except:
            pept_anchs = [1, len(target.peptide) - 1]
    
        temp_pept = template.peptide
        temp_anchs = template.anchors
        aligned_pept, aligned_temp_pept = align_peptides(target.peptide,
                                                         pept_anchs[0], pept_anchs[1],
                                                         temp_pept,
                                                         temp_anchs[0], temp_anchs[1])
    
        aligned_pept = aligned_pept.replace('-', '*')
        aligned_temp_pept = aligned_temp_pept.replace('-', '*')
        # min_len = min([len(target.peptide), len(temp_pept)])
        # score -= ((abs(len(target.peptide) - len(temp_pept)) ** 2.4))  # !!!  ## Gap Penalty #Is now handled by normal PAM30
        for i, (aa, bb) in enumerate(zip(aligned_pept, aligned_temp_pept)):
            try:
                # gain = MatrixInfo.pam30[aa, bb]
                gain = substitution_matrix[aa, bb]
                score += gain
            except KeyError:
                try:
                    # gain = MatrixInfo.pam30[bb, aa]
                    gain = substitution_matrix[bb, aa]
                    score += gain
                except KeyError:
                    score = -50
                    pass
    
        return score
    
    elif target.MHC_class == 'II':
        # define the peptide and p1 anchor position
        temp_pept = template.peptide
        temp_p1 = template.anchors[0]
        tar_pept = target.peptide
        tar_p1 = target.anchors[0]
    
        # align based on first anchor position, fill in the ends with '-' to make them equal length
        temp_pept = '*' * (tar_p1 - temp_p1) + temp_pept
        tar_pept = '*' * (temp_p1 - tar_p1) + tar_pept
        temp_pept = temp_pept + '*' * (len(tar_pept) - len(temp_pept))
        tar_pept = tar_pept + '*' * (len(temp_pept) - len(tar_pept))
    
        # Perform a pairwise alignment. Make sure no gaps are introduced.
        aligner = PairwiseAligner()
        aligner.substitution_matrix = substitution_matrices.load(substitution_matrix)
        aligner.gap_score = -1000
        aligner.end_open_gap_score = -1000000
        aligner.internal_open_gap_score = -10000
    
        # Align the sequences
        aligned = aligner.align(tar_pept, temp_pept)
    
        return aligned.score


def find_template(target, database, best_n_templates = 1, benchmark=False, 
                  blastdb=PANDORA.PANDORA_data + '/BLAST_databases/templates_blast_db/templates_blast_db'):
    ''' Selects the template structure that is best suited as template for homology modelling of the target

    Args:
        target (PMHC.Target): Target object
        database (Database.Database): Database object
        blastdb (str): Path to blast database to use for sequence-based template selection.

    Returns: Template object

    '''

    putative_templates = {}
    
    if target.MHC_class == 'I':
        class_variables = [PANDORA.MHCI_G_domain[0][1], 'MHCI_data', 'M_score']
    elif target.MHC_class == 'II':
        class_variables = [PANDORA.MHCII_G_domain[0][1], 'MHCII_data', 'Avg_score']
        
    no_seq_chains = []
    if target.M_chain_seq != '':
        #Sequence based template selection
        
        # Sequence based template selection
        #Keep only G-domain
        M_chain = target.M_chain_seq[:class_variables[0]]
        #Blast M chain sequence
        try:
            M_chain_result = blast_mhc_seq(M_chain, chain='M', blastdb=blastdb)
            #FIll in putative_templates M identity score
            for result in M_chain_result:
                ID = result[0][:4]
                score = result[1]
                putative_templates[ID] = {'M_score': score}
        except Exception as e:
            print(e)
            print('WARNING: something went wrong with blast-based template selection.')
            print('Is blastp properly installed?')
            #If blast didn't work properly, consider this sequence missing
            no_seq_chains.append('M_score')
    
    else:
        no_seq_chains.append('M_score')
        
    if target.MHC_class == 'II':
        if target.N_chain_seq != '':
            #Keep only G-domain
            N_chain = target.N_chain_seq[:PANDORA.MHCII_G_domain[1][1]]
            #Blast N chain sequence
            try:
                N_chain_result = blast_mhc_seq(N_chain, chain='N', blastdb=blastdb)
            
                #FIll in putative_templates N  and average identity score
                for result in N_chain_result:
                    ID = result[0][:4]
                    score = result[1]
                    try:
                        putative_templates[ID]['N_score']= score
                        #Get average score
                    except KeyError:
                        putative_templates[ID] = {'N_score': score}
                        
            except Exception as e:
                print(e)
                print('WARNING: something went wrong with blast-based template selection.')
                print('Is blastp properly installed?')
                #If blast didn't work properly, consider this sequence missing
                no_seq_chains.append('N_score')
            
                
        else: 
            no_seq_chains.append('N_score')
    
    #For every chain withous a seq, fill in the relative score to 100
    #For each template with at least one matching allele
    if no_seq_chains !=[]:
        for C in no_seq_chains:
            # Fill in available alleles list
            available_alleles = []
            for ID in getattr(database, class_variables[1]):
                if benchmark and ID == target.id:
                    pass
                else:
                    available_alleles.extend(getattr(database, class_variables[1])[ID].allele_type)
            available_alleles = list(set(available_alleles))
            
            # Find template structures with matching alleles
            target_alleles = allele_name_adapter(target.MHC_class, target.allele_type, available_alleles)
            target_alleles = list(set(target_alleles))
            for ID in getattr(database, class_variables[1]):
                if benchmark:
                    if ID != target.id:
                        if any(y in x for x in getattr(database, class_variables[1])[ID].allele_type for y in target_alleles):
                            try:
                                putative_templates[ID][C] = 100.0
                            except KeyError:
                                putative_templates[ID] = {C : 100.0}
                else:
                    if any(y in x for x in getattr(database, class_variables[1])[ID].allele_type for y in target_alleles):
                        try:
                            putative_templates[ID][C] = 100.0
                        except KeyError:
                            putative_templates[ID]= {C : 100.0}
    
    #Keep only templates present in the template db.
    #This prevents errors caused by different blast and pandora db.
    putative_templates = {k:v for k,v in putative_templates.items() if k in getattr(database, class_variables[1]).keys()}
    
    #Remove target from putative_templates if benchmark run
    if benchmark:
        if target.id in putative_templates.keys():
            del putative_templates[target.id]
                                
    if target.MHC_class == 'II':
        for ID in putative_templates:
            if len(putative_templates[ID].keys()) == 2:
                putative_templates[ID]['Avg_score'] = (putative_templates[ID]['M_score'] + putative_templates[ID]['N_score']) /2
        
        putative_templates = {x : putative_templates[x] for x in putative_templates 
                              if 'Avg_score' in list(putative_templates[x].keys())}
         # Make sure there is no template with only 3 anchors for benchmarking.
        if benchmark:
            putative_templates = {k:v for k,v in putative_templates.items() if len(database.MHCII_data[k].anchors) == 4}
        

    # For both chains
    #Sort for average score
    putative_templates = sorted(putative_templates.items(), 
                                key=lambda x: x[1][class_variables[2]], reverse=True)
    putative_templates = {x[0] : x[1] for x in putative_templates}
    #Keep only max score templates
    try:
        max_score = list(putative_templates.values())[0][class_variables[2]]
    except IndexError:
        raise Exception('Putative templates list empty.')
    putative_templates = {x : putative_templates[x] for x in putative_templates 
                          if putative_templates[x][class_variables[2]] == max_score}

    # Find the putative template with the best matching peptide
    pos_list = []
    for ID in putative_templates:
        score = score_peptide_alignment(target, getattr(database, class_variables[1])[ID], substitution_matrix='PAM30')
        pos_list.append((score, getattr(database, class_variables[1])[ID].peptide, ID))

    if len(pos_list) == 0:
        raise Exception('Pandora could not find any putative template! Please try to define your own template or contact us for help')
    
    # Sort templates per peptide score
    template_id = [i[-1] for i in sorted(pos_list, key=lambda elem: elem[0], reverse=True)][:best_n_templates]
    scores = sorted(pos_list, key=lambda elem: elem[0], reverse=True)[:best_n_templates]

    templates = [getattr(database, class_variables[1])[tmpl] for tmpl in template_id]
    keep_IL = any(check_target_template(target, tmpl) for tmpl in templates)

    return templates, scores, keep_IL

def write_ini_script(target, template, alignment_file, output_dir, clip_C_domain=False):
    ''' Writes the MyLoop.py and cmd_modeller_ini.py files. This function takes two template python scripts and fills
        in the required information: Anchor positions for the MyLoop file and structure name + alignment file for the
        cmd_modeller_ini file.

    Args:
        target: Target object
        template: Template object
        alignment_file: (string) path to alignment file
        output_dir: (string) path to output directory

    '''

    anch = target.anchors

    if target.MHC_class == 'I':
        with open(output_dir+ '/MyLoop.py', 'w') as myloopscript:
            MyL_temp = open(PANDORA.PANDORA_path + '/Pandora/MyLoop_template.py', 'r')
            for line in MyL_temp:
                # Include or not B2M depending on clip_C_domain
                if '#RENAME SEGMENTS PLACEHOLDER' in line:
                    if not clip_C_domain:
                        myloopscript.write("        self.rename_segments(segment_ids=['M', 'B', 'P'], renumber_residues=[1, 1, 1])")
                    else:
                        myloopscript.write("        self.rename_segments(segment_ids=['M', 'P'], renumber_residues=[1, 1])")
                elif 'self.residue_range' in line and 'M.selection' in line:
                    myloopscript.write(line % (anch[0]+1, anch[-1]-1))
                elif 'SPECIAL_RESTRAINTS_BREAK' in line:
                    break
                elif 'contact_file = open' in line:
                    myloopscript.write(line %target.id)
                else:
                    myloopscript.write(line)
            MyL_temp.close()

    if target.MHC_class == 'II':
        with open(output_dir + '/MyLoop.py', 'w') as myloopscript:
            MyL_temp = open(PANDORA.PANDORA_path + '/Pandora/MyLoop_template_II.py', 'r')
            for line in MyL_temp:
                if 'self.residue_range' in line and 'M.selection' in line:
                    myloopscript.write("        return M.selection(self.residue_range('%i:P', '%i:P'))\n" %(1, len(target.peptide)))

                elif 'SPECIAL_RESTRAINTS_BREAK' in line:
                    break
                elif 'contact_file = open' in line:
                    myloopscript.write(line %target.id)
                else:
                    myloopscript.write(line)
            MyL_temp.close()

    with open(output_dir.replace('\\ ', ' ') + '/cmd_modeller_ini.py', 'w') as modscript:
        cmd_m_temp = open(PANDORA.PANDORA_path + '/Pandora/cmd_modeller_ini.py', 'r')
        for line in cmd_m_temp:
            if 'alnfile' in line:
                modscript.write(line % os.path.basename(alignment_file))
            elif 'knowns' in line:
                if type(template)==list:
                    modscript.write(
                        'knowns = (%s), sequence = "%s",\n' % (','.join(['"' + i.id + '"' for i in template]), target.id))
                else:
                    modscript.write(
                        'knowns = (%s), sequence = "%s",\n' % ('"' + template.id + '"', target.id))
                # modscript.write(line % ('(' + ','.join([i.id for i in template]) + ')', target.id))
            else:
                modscript.write(line)
        cmd_m_temp.close()


def write_modeller_script(target, template, alignment_file, output_dir, n_homology_models=1, n_loop_models = 20,
                          loop_refinement='slow', n_jobs=None, helix = False, sheet = False, 
                          restraints_stdev=False, clip_C_domain=False):
                          
    ''' Write script that refines the loops of the peptide
    
    Args:
        target (PANDORA.PMHC.PMHC.Target): Target object
        template (PANDORA.PMHC.PMHC.Template): Template object
        alignment_file (str): path to alignment file
        output_dir (str): path to output directory
        n_homology_models (int): number of homology models that are generated per run.
        n_loop_models (int): number of loop models modeller generates per homology model
        n_jobs (int): number of parallel jobs. Is recommended to use at most as many jobs as the number of models:
            ore will not add any benefit but might occupy cores unnecessarily.
        loop_refinement (str): Level of loop refinement: very_fast,fast,slow,very_slow,slow_large.
            Defaults to slow
        helix (list): List of the alpha helix start and end-positions as integers. I.e. [3,8] for a helix between
            peptide residue 3 and 8.
        sheet (list): List containing: start position of B-sheet 1, start position of B-sheet 2 and the length of the
            B-sheet in h-bonds. For example: ["O:2:P","N:54:M",2] for a parallel B-sheet; The sheet starts
            at the Oxigen atom of the 2nd residue of chain P and at the Nitrogen of the 54th residue of
            chain M and has a length of 2 H-bonds. Or; ["N:6:P", "O:13:P", -3], with -3 denoting an
            anti-parallel B-sheet with a length of 3 H-bonds.
        restraints_stdev (bool or float): if True, keeps the whole peptide flexible. Increases computational time by 30-50% 
            but increases accuracy. If float, it used as standard deviation of modelling restraints. Higher = more flexible restraints. 
            Defaults to False. Setting it to True only will set the default standard dev iation to 0.1.

    '''

    anch = target.anchors
    if type(restraints_stdev) == float:
        stdev = restraints_stdev
    else:
        stdev = 0.1

    if target.MHC_class == 'I':
        with open(output_dir.replace('\\ ', ' ') + '/MyLoop.py', 'w') as myloopscript:
            MyL_temp = open(PANDORA.PANDORA_path + '/Pandora/MyLoop_template.py', 'r')
            for line in MyL_temp:
                # Include or not B2M depending on clip_C_domain
                if '#RENAME SEGMENTS PLACEHOLDER' in line:
                    if not clip_C_domain:
                        myloopscript.write("        self.rename_segments(segment_ids=['M', 'B', 'P'], renumber_residues=[1, 1, 1])")
                    else:
                        myloopscript.write("        self.rename_segments(segment_ids=['M', 'P'], renumber_residues=[1, 1])")
                # Add flexible region selection range
                elif 'self.residue_range' in line and 'M.selection' in line:
                    if restraints_stdev:
                        myloopscript.write(line %(1, len(target.peptide)))  # write the first anchor
                    else:
                        myloopscript.write(line %(anch[0]+1, anch[-1]-1))
                # Add restraints standard deviation (only effective on non-fixed residues)
                elif 'STDEV MARKER' in line:
                    myloopscript.write(line %(stdev))
                # Add contact file name
                elif 'contact_file = open' in line:
                    myloopscript.write(line %(target.id))
                # Add Alpha helix restraints
                elif helix and 'ALPHA-HELIX-MARKER' in line:
                    myloopscript.write(line.replace('# ALPHA-HELIX-MARKER', 'rsr.add(M.secondary_structure.alpha(self.residue_range("%s:P", "%s:P")))' %(helix[0], helix[1])))
                # Add Beta sheet restraints
                elif sheet and 'BETA-SHEET-MARKER' in line:
                    myloopscript.write(line.replace('# BETA-SHEET-MARKER', 'rsr.add(M.secondary_structure.sheet(atoms["%s"], atoms["%s"], sheet_h_bonds=%s))' %(sheet[0], sheet[1], sheet[2])))
                else:
                    myloopscript.write(line)
            MyL_temp.close()

    if target.MHC_class == 'II':
        with open(output_dir.replace('\\ ', ' ') + '/MyLoop.py', 'w') as myloopscript:
            MyL_temp = open(PANDORA.PANDORA_path + '/Pandora/MyLoop_template_II.py', 'r')
            for line in MyL_temp:
                if 'ANCHORS_PLACEHOLDER' in line:
                    if anch[0] == 0:
                        anch_1 = 1
                    else:
                        anch_1 = anch[0]
                    if anch[-1] == (len(target.peptide)-1):
                        anch_term = len(target.peptide)
                    else:
                        anch_term = anch[-1]
                    #Write first and last anchors, to keep only the flanking regions flexible
                    if restraints_stdev:
                        #myloopscript.write(line % (1, len(target.peptide)))
                        myloopscript.write("        return M.selection(self.residue_range('%i:P', '%i:P'))\n" %(1, len(target.peptide)))
                    else:
                        myloopscript.write("        return M.selection(self.residue_range('%i:P', '%i:P'), self.residue_range('%i:P', '%i:P'))\n" % 
                                            (1, anch_1, anch_term, len(target.peptide)))
                        #self.residue_range('%i:P', '%i:P'), self.residue_range('%i:P', '%i:P')
                    #for i in range(len(anch)-1): # Write all the inbetween acnhors if they are there
                    #    myloopscript.write(line % (anch[i] + 2, anch[i+1]))
                    #myloopscript.write(line % (anch[-1] + 2, len(target.peptide))) # Write the last anchor
                elif 'contact_file = open' in line:
                    myloopscript.write(line %(target.id))
                elif 'STDEV MARKER' in line:
                    myloopscript.write(line %(stdev))
                elif helix and 'ALPHA-HELIX-MARKER' in line:
                    myloopscript.write(line.replace('# ALPHA-HELIX-MARKER', 'rsr.add(M.secondary_structure.alpha(self.residue_range("%s:P", "%s:P")))' %(helix[0], helix[1])))
                elif sheet and 'BETA-SHEET-MARKER' in line:
                    myloopscript.write(line.replace('# BETA-SHEET-MARKER', 'rsr.add(M.secondary_structure.sheet(atoms["%s"], atoms["%s"], sheet_h_bonds=%s))' %(sheet[0], sheet[1], sheet[2])))
                else:
                    myloopscript.write(line)
            MyL_temp.close()

    with open(output_dir.replace('\\ ', ' ') + '/cmd_modeller.py', 'w') as modscript:
        cmd_m_temp = open(PANDORA.PANDORA_path + '/Pandora/cmd_modeller_template.py', 'r')
        for line in cmd_m_temp:
            if 'alnfile' in line:
                modscript.write(line %(os.path.basename(alignment_file)))
            elif 'knowns' in line:
                if type(template)==list:
                    modscript.write(
                        'knowns = (%s), sequence = "%s",\n' % (','.join(['"' + i.id + '"' for i in template]), target.id))
                else:
                    modscript.write(
                        'knowns = (%s), sequence = "%s",\n' % ('"' + template.id + '"', target.id))
                # modscript.write(line %(','.join([i.id for i in template]), target.id))
            elif 'a.ending_model' in line:
                modscript.write(line % (n_homology_models))
            elif 'a.loop.ending_model' in line:
                modscript.write(line % (n_loop_models))
            elif 'a.loop.md_level' in line:
                modscript.write('a.loop.md_level = MA.refine.%s   # Loop model refinement level' %(loop_refinement))
            else:
                if n_jobs != None: #If this is a parallel job
                    if 'PARALLEL_JOB_LINE_TO_COMPLETE' in line:
                        modscript.write(line %(str(n_jobs))) #specify the number of cores
                    else:
                        modscript.write(line)  #Write the line as it is
                else: #If this is not a parallel job
                    if 'PARALLEL_JOB_LINE' in line: #do not write the lines requested for parallelization
                        pass
                    else:
                        modscript.write(line)  #Write the line as it is
        cmd_m_temp.close()



def run_modeller(output_dir, target, python_script = 'cmd_modeller.py', benchmark = False, pickle_out = True,
                 keep_IL = False, RMSD_atoms = ['C', 'CA', 'N', 'O']):
    ''' Perform the homology modelling.

    Args:
        output_dir: (string) path to output directory
        target: Target object
        python_script:  (string) path to script that performs the modeller modelling. cmd_modeller.py
        benchmark: (bool) Perform L-RMSD calculations? only works if the target id is an existing pdb id
        pickle_out: (bool) Save a .pkl with the results

    Returns: (list) of Model objects

    '''
    # Identify current working directory
    cwd = os.getcwd()

    # Change working directory
    os.chdir(output_dir)
    # run Modeller to perform homology modelling
    os.popen('python3 %s > modeller.log' %python_script).read()
    os.chdir(cwd)

    # Parse .log file
    logf = []
    f = open(output_dir + '/modeller.log')
    for line in f:
        if line.startswith(target.id + '.'):   #target.id
            l = line.split()
            if len(l) == 3: #TODO: make sure the line is reporting the model with tis score. Format: model, molpdf, dope.
                logf.append(tuple(l))
    f.close()

    # If keep_IL is true (happens if the target and template are the same), also use the initial model as one of the
    # results. This will also happen while benchmarking.
    if keep_IL:
        # Also take the Initial Loop model. Take the molpdf from the pdb header.
        il_file = [i for i in os.listdir(output_dir) if i.startswith(target.id + '.IL')][0]
        # Create a fake molpdf/dope score for the IL model: the best molpdf/dope from the real models - 1
        try:
            fake_molpdf = str(min(float(i[1]) for i in logf) - 1)
            fake_dope = str(min(float(i[2]) for i in logf) - 1)
        except ValueError:
            fake_molpdf = -10000
            fake_dope = -10000
            print('WARNING: ValueError exception raised while assigning fake molpdf and dope to IL model')
        # Append the filename and molpdf to the rest of the data
        logf.append((il_file, fake_molpdf, fake_dope))

    # Sort output by molpdf
    logf.sort(key=lambda tup:float(tup[1]))

    # Write to output file
    f = open(output_dir + '/molpdf_DOPE.tsv', 'w')
    for i in logf:
        f.write(i[0] + '\t' + i[1] + '\t' + i[2] + '\n')
    f.close()


    # Create Model object of each theoretical model and add it to results
    results = []
    for i in range(len(logf)):
        try:
            m = Model.Model(target, model_path=output_dir + '/' + logf[i][0], output_dir = output_dir,
                                            molpdf=logf[i][1], dope=logf[i][2])

        except:
            print('WARNING: Error raised while calling Model.Model() for case %s' %target.id)
            print(traceback.format_exc())
            m = None
        results.append(m)


    # Save results as pickle
    if pickle_out:
        pickle.dump(results, open("%s/results_%s.pkl" %(output_dir, os.path.basename(os.path.normpath(output_dir))), "wb"))

    return results

def blast_mhc_seq(seq, chain='M', blastdb=PANDORA.PANDORA_data + '/BLAST_databases/refseq_blast_db/refseq_blast_db'):
    try:
        command = (' ').join(['blastp','-db',blastdb, 
                                                 '-query',
                                                 '<(echo %s)' %seq,
                                                 '-outfmt','6'])
        proc = subprocess.Popen(command,  executable='/bin/bash',
                                     shell=True, stdout=subprocess.PIPE)
        blast_result = proc.stdout.read()
        blast_result = blast_result.decode()
    except subprocess.CalledProcessError as e:
        raise Exception('An error occurred while blasting %s chain seq: %s' %(chain, e.output))
    
    if not blast_result:
        raise Exception('An error occurred while blasting %s chain seq: blast output empty' %(chain))

    blast_result = blast_result.split('\n')
    blast_result = [x.replace(';',' ').split('\t') for x in blast_result]
    blast_result = [x for x in blast_result if x != ['']]
    
    #FIll in putative_templates M identity score
    results = []
    for result in blast_result:
        ID = result[1]
        score = float(result[2])
        results.append((ID, score))
    results = sorted(results, key=lambda x: x[1], reverse=True)
    
    return results
        
    

def align_peptides(seq1, anch1_seq1, anch2_seq1, seq2, anch1_seq2, anch2_seq2):
    '''
    Align two MHC-I peptides making overlap the anchors.
    This function does NOT use an alignment matrix (e.g. BLOSUM, PAM, etc).
    It computes a simple anchor position alignment and inserts gap in the
    middle part to make the final sequences have the same lenghts.

    Args:
        seq1(str) : sequence of the first peptide.
        anch1_seq1(int) : position of the first anchor of seq1. Position must be given in Python numbering (0-N)
        anch2_seq1(int) : position of the second anchor of seq1. Position must be given in Python numbering (0-N)
        seq2(str) : sequence of the second peptide.
        anch1_seq1(int) : position of the first anchor of seq1. Position must be given in Python numbering (0-N)
        anch2_seq1(int) : position of the second anchor of seq1. Position must be given in Python numbering (0-N)

    Returns:
        ali_seq1(str)
    '''

    seq1_core = anch2_seq1 - anch1_seq1
    seq2_core = anch2_seq2 - anch1_seq2
    tail1 = [x for x in seq1[anch2_seq1:]]
    tail2 = [x for x in seq2[anch1_seq2:]]

    list1 = [x for x in seq1]
    list2 = [x for x in seq2]
    #Adding gaps in cores
    if seq1_core > seq2_core:
        for x in range(seq1_core - seq2_core):
            list2.insert(int(len(seq2)/2), '-')
    elif seq1_core < seq2_core:
        for x in range(seq2_core - seq1_core):
            list1.insert(int(len(seq1)/2), '-')
    ### Adding gaps in heads
    if anch1_seq1 > anch1_seq2:
        for x in range(anch1_seq1 - anch1_seq2):
            list2.insert(0, '-')
    elif anch1_seq1 < anch1_seq2:
        for x in range(anch1_seq2 - anch1_seq1):
            list1.insert(0, '-')
    ### Adding gaps in heads
    if len(tail1) > len(tail2):
        for x in range(len(tail1) - len(tail2)):
            list2.insert(-1, '-')
    elif len(tail1) < len(tail2):
        for x in range(len(tail1) - len(tail2)):
            list1.insert(-1, '-')

    ali_seq1 = ('').join(list1)
    ali_seq2 = ('').join(list2)
    return ali_seq1, ali_seq2

def allele_name_adapter(MHC_class, ori_alleles, available_alleles):
    '''
    Cuts the given allele name to make it consistent with the alleles in allele_ID.

    Args:
        allele(list) : Allele names
        allele_ID(dict) : Dictionary of structure IDs (values) in the dataset for each allele (keys)
        
    Returns:
        allele(list) : List of adapted (cut) allele names
    '''
    #homolog_allele = '--NONE--'
    alleles = deepcopy(ori_alleles)
    if MHC_class =='I':
        for a in range(len(alleles)):
            if alleles[a].startswith('HLA'):      # Human
                if any(alleles[a] in key for key in list(available_alleles)):
                    pass
                elif any(alleles[a][:8] in key for key in list(available_alleles)):
                    alleles[a] = alleles[a][:8]
                elif any(alleles[a][:6] in key for key in list(available_alleles)):
                    alleles[a] = alleles[a][:6]
                else:
                    alleles[a] = alleles[a][:4]
            elif alleles[a].startswith('H2'):    # Mouse
                #homolog_allele = 'RT1'
                if any(alleles[a] in key for key in list(available_alleles)):
                    pass
                elif any(alleles[a][:4] in key for key in list(available_alleles)):
                    alleles[a] = alleles[a][:4]
                else:
                    alleles[a] = alleles[a][:3]
            elif alleles[a].startswith('RT1'):          # Rat
                #homolog_allele = 'H2'
                if any(alleles[a] in key for key in list(available_alleles)):
                    pass
                elif any(alleles[a][:5] in key for key in list(available_alleles)):
                    alleles[a] = alleles[a][:5]
                else:
                    alleles[a] = alleles[a][:4]
            elif alleles[a].startswith('BoLA'):        # Bovine
                if any(alleles[a] in key for key in list(available_alleles)):
                    pass
                elif any(alleles[a][:10] in key for key in list(available_alleles)):
                    alleles[a] = alleles[a][:10]
                elif any(alleles[a][:7] in key for key in list(available_alleles)):
                    alleles[a] = alleles[a][:7]
                else:
                    alleles[a] = alleles[a][:5]
            elif alleles[a].startswith('SLA'):        # Suine
                if any(alleles[a] in key for key in list(available_alleles)):
                    pass
                elif any(alleles[a][:9] in key for key in list(available_alleles)):
                    alleles[a] = alleles[a][:9]
                elif any(alleles[a][:6] in key for key in list(available_alleles)):
                    alleles[a] = alleles[a][:6]
                else:
                    alleles[a] = alleles[a][:4]
            elif alleles[a].startswith('MH1-B'):        # Chicken
                if any(alleles[a] in key for key in list(available_alleles)):
                    pass
                elif any(alleles[a][:8] in key for key in list(available_alleles)):
                    alleles[a] = alleles[a][:8]
                else:
                    alleles[a] = alleles[a][:6]
            elif alleles[a].startswith('MH1-N'):        # Chicken
                if any(alleles[a] in key for key in list(available_alleles)):
                    pass
                elif any(alleles[a][:9] in key for key in list(available_alleles)):
                    alleles[a] = alleles[a][:9]
                else:
                    alleles[a] = alleles[a][:6]
            elif alleles[a].startswith('BF2'):        # Chicken
                if any(alleles[a] in key for key in list(available_alleles)):
                    pass
                elif any(alleles[a][:6] in key for key in list(available_alleles)):
                    alleles[a] = alleles[a][:6]
                else:
                    alleles[a] = alleles[a][:4]
            elif alleles[a].startswith('Mamu'):       # Monkey
                if any(alleles[a] in key for key in list(available_alleles)):
                    pass
                elif any(alleles[a][:13] in key for key in list(available_alleles)):
                    alleles[a] = alleles[a][:13]
                elif any(alleles[a][:9] in key for key in list(available_alleles)):
                    alleles[a] = alleles[a][:9]
                else:
                    alleles[a] = alleles[a][:5]
            elif alleles[a].startswith('Eqca'):        # Horse
                if any(alleles[a] in key for key in list(available_alleles)):
                    pass
                elif any(alleles[a][:10] in key for key in list(available_alleles)):
                    alleles[a] = alleles[a][:10]
                elif any(alleles[a][:7] in key for key in list(available_alleles)):
                    alleles[a] = alleles[a][:7]
                else:
                    alleles[a] = alleles[a][:5]
                    
    elif MHC_class =='II':
        for a in range(len(alleles)):
            if alleles[a].startswith('HLA'):      # Human
                prefix = alleles[a].split('-')[0]
                gene = re.split('-|\*', alleles[a])[1][:2]
                chain = re.split('-|\*', alleles[a])[1][2]
                subgene = re.split('-|\*', alleles[a])[1][3:]
                group = re.split(':|\*', alleles[a])[1]
                subgroup = re.split(':|\*', alleles[a])[2]
                
                if any(alleles[a] in key for key in list(available_alleles)):
                    pass
                elif any(prefix+'-'+gene+chain+subgene+'*'+group in key for key in list(available_alleles)):
                    print('WARNING: The provided allele subgroup has not been found. PANDORA will treat this case as %s' %(prefix+'-'+gene+chain+subgene+'*'+group))
                    alleles[a] = prefix+'-'+gene+chain+subgene+'*'+group
                elif any(prefix+'-'+gene+chain+subgene in key for key in list(available_alleles)):
                    alleles[a] = prefix+'-'+gene+chain+subgene
                    print('WARNING: The provided allele group has not been found. PANDORA will treat this case as %s' %(prefix+'-'+gene+chain+subgene))
                else:
                    alleles[a] = prefix+'-'+gene+chain
            
            else:                               #Other spieces might be implemented later
                pass
    return alleles #, homolog_allele)
    
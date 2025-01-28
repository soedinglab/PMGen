#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
from joblib import Parallel, delayed
import subprocess
import traceback
import os
from PANDORA import Target
from PANDORA import Pandora
import re
import random
import string

class Wrapper():
    def __init__(self, data_file, database, MHC_class,  num_cores=1, delimiter = '\t',
                    header=True, IDs_col=None, peptides_col=0, allele_name_col=1,
                    anchors_col=None, M_chain_col=None, N_chain_col=None,
                    outdir_col=None, template_col=None, benchmark=False, verbose=False,
                    start_row=None, end_row=None, use_netmhcpan=False,
                    use_templ_seq=False, n_loop_models=20, n_jobs=None,
                    collective_output_dir=False, pickle_out=False, 
                    clip_C_domain=False, restraints_stdev=False,
                    archive=False, wrapper_id=False, rm_netmhcpan_output=True,
                    ):
        """Pandora wrapper object.
        Create PANDORA targets from csv or tsv file and models them.
        Args:
            data_file (str): Path to the input tsv/csv file containing targets
            information.
            database (PANDORA.Database.Database): Database object.
            MHC_class (str): MHC class of the targets, as 'I' or 'II'.
            num_cores (int, optional): Number of parallel PANDORA jobs.
            Each one will be sent to a different core. Defaults to 1.
            delimiter (str, optional): data_file delimiter. Do not use
            semicolons (';') as separators. Defaults to '\t'.
            header (bool, optional): If True, assumes the data_file has a
            header line and skips it. If your file has no header line,
            set it as False. Defaults to True.
            IDs_col (int or None, optional): Column of data_file containing
            the targets IDs. If None, will automatically assign an ID
            according to the row number. Defaults to None.
            peptides_col (int, optional): Column of data_file containing
            the targets peptides. Defaults to 0.
            allele_name_col (int, optional): Column of data_file containing
            the targets alleles. Umbiguous allele cases (where the allele
            might have multiple names) should be separated by a
            semicolon (';'). Defaults to 1.
            anchors_col (int, optional): Column of data_file containing
            the targets anchors. Anchors should be two numbers separated
            by a semicolon (';'). Defaults to 2.
            M_chain_col (None or int, optional): Column of data_file containing
            the targets M chain sequences.
            N_chain_col (None or int, optional): Column of data_file containing
            the targets N chain sequences (only for MHCII).
            outdir_col (None or int, optional): Column of data_file containing
            the paths to the output folder for each case.
            template_col (None or int, optional): 0-index column containing the template
            ID to be used for each case. Defaults to None.
            collective_output_dir (str, optional): Output directory path for
            all the cases. Note: This argument will be ignored if  'outdir_col'
            has been used to generate targets with Wrapper.create_targets().
            Defaults to False.
            benchmark (bool, optional): Set True only for benchmarking purpose,
            if target structures are available. Defaults to False.
            start_row (None or int): Starting row of data_file, to use when
            splitting the data_file into multiple batches. This allows to
            specify from which row the samples for this job start.
            end_row (None or int): Ending row of data_file, to use when
            splitting the data_file into multiple batches. This allows to
            specify at which row the samples for this job end.
            use_netmhcpan (bool, optional): If True, uses local installation
            of netMHCPan to predict anchor positions for each target.
            use_templ_seq (bool, optional): If true, it uses the template MHC sequence 
            for each chain a sequence could not be found. This function is mainly
            for benchmarking purposes. Defaults to False.
            num_cores (int, optional): Number of parallel PANDORA jobs.
            Each one will be sent to a different core. Defaults to 1.
            n_loop_models (int, optional): Number of  loop models.
            Defaults to 20.
            n_jobs (int, optional): Number of parallel MODELLER loop jobs.
            Do not increase further than n_loop_models. Defaults to None.
            pickle_out (bool, optional): If True, outputs a pickle file
            containing every model object. Defaults to False.
            clip_C_domain (bool or list): if True, clips away the C-like domain, levaing only
            the G-domain according to IMGT. If a listcontaining the G domain(s) 
            span is provided, will use it to cut the sequence. The list should have 
            this format: [(1,182)] for MHCI and [(1,91),(1,86)] for MHCII.
            restraints_stdev (bool or float): if True, keeps the whole peptide flexible. Increases computational time by 50-90% 
            but increases accuracy and prevents from artifacts at the anchor positions.
            If float, it used as standard deviation of modelling restraints. Higher = more flexible restraints. 
            Defaults to False. Setting it to True only will set the default standard deviation to 0.1.
            wrapper_id (string): id of the wrapper. Should be alphanumeric only. 
            If not, non-alphanumeric characters will be replaced with dashes.
            If False, it will be randomly generated. Defaults to False.
            rm_netmhcpan_output: (bool) If True, removes the netmhcpan infile and 
            outfile after having used them for netmhcpan.
        Returns:
            None.
        """

        self.MHC_class = MHC_class
        self.data_file = ''
        self.db = None
        self.targets = {}
        self.jobs = {}

        self.data_file = data_file
        self.db = database

        # Determine the wrapper id
        if wrapper_id == False:
            random_id = ''.join(random.choice(string.ascii_uppercase + string.ascii_lowercase + string.digits) for _ in range(6))
            self.wrapper_id = f'PandoraWrapper_{random_id}'
        else:
            self.wrapper_id = wrapper_id
            self.wrapper_id = re.sub('[^a-zA-Z0-9]', '_', self.wrapper_id)


        if outdir_col == None:
            # Determine the wrapper output directory
            if collective_output_dir == False:
                self.collective_output_dir = os.getcwd()
            else:
                self.collective_output_dir = collective_output_dir

            self.collective_output_dir = os.path.join(self.collective_output_dir, self.wrapper_id)
            
            self.prep_collective_output_dir()

        else:
            self.collective_output_dir = collective_output_dir

        ## Extract targets from data_file
        self.__get_targets_from_file(data_file, delimiter=delimiter,
                                        header=header, IDs_col=IDs_col,
                                        peptides_col=peptides_col, allele_name_col=allele_name_col,
                                        anchors_col=anchors_col, M_chain_col=M_chain_col,
                                        N_chain_col=N_chain_col,outdir_col=outdir_col,
                                        start_row=start_row,end_row=end_row)

        ## Print targets info
        if verbose:
            for target_id in self.targets:
                print('\n')
                print('Target ID: ', target_id)
                print('Target MHC_class: ', MHC_class)
                print('Target allele: ', self.targets[target_id]['allele'])
                print('Target peptide: ', self.targets[target_id]['peptide_sequence'])
                print('Target M chain seq: ', self.targets[target_id]['M_chain_seq'])
                if N_chain_col:
                    print('Target N chain seq: ', self.targets[target_id]['N_chain_seq'])
                print('Target Anchors: ', self.targets[target_id]['anchors'])
                
        for target_id in self.targets:
            self.targets[target_id].update({'target_id':target_id, 'MHC_class':MHC_class,
                                    'n_loop_models':n_loop_models, 
                                    'n_jobs':n_jobs, 
                                    'benchmark':benchmark, 'pickle_out':pickle_out,
                                    'collective_output_dir':self.collective_output_dir,
                                    'clip_C_domain':clip_C_domain,
                                    'restraints_stdev':restraints_stdev,
                                    'archive_output': archive,
                                    'db':database, 'use_netmhcpan':use_netmhcpan,
                                    'use_templ_seq':use_templ_seq,
                                    'rm_netmhcpan_output':rm_netmhcpan_output})
        Parallel(n_jobs = num_cores, verbose = 1)(delayed(run_case)(target) for target in list(self.targets.values()))

    def __get_targets_from_file(self, data_file, delimiter='\t', header=True,
                               IDs_col=None, peptides_col=0,
                               allele_name_col=1, anchors_col=None,
                               M_chain_col=None, N_chain_col=None,
                               outdir_col=None, template_col=None,
                               start_row=None, end_row=None,
                               ):
        """Extracts peptide sequences, alleles and anchors (if specified)
            from the target file.
            Default input should be a .tsv file without any header with
            the following structure: peptides_sequence_col \t alleles_name_col
        Args:
            data_file (str): Path to the input tsv/csv file containing targets
                information.
            delimiter (str, optional): data_file delimiter. Do not use
                semicolons (';') as separators. Defaults to '\t'.
            header (bool, optional): If True, assumes the data_file has a
                header line and skips it. If your file has no header line,
                set it as False. Defaults to True.
            IDs_col (int or None, optional): 0-index column of data_file containing
                the targets IDs. If None, will automatically assign an ID
                according to the row number. Defaults to None.
            peptides_col (int, optional): 0-index column of data_file containing
                the targets peptides. Defaults to 0.
            allele_name_col (int, optional): 0-index column of data_file containing
                the targets alleles. Umbiguous allele cases (where the allele
                might have multiple names) should be separated by a
                semicolon (';'). Defaults to 1.
            anchors_col (None or int, optional): 0-index column of data_file containing
                the targets anchors. Anchors should be two numbers separated
                by a semicolon (';'). Defaults to 2.
            M_chain_col (None or int, optional): 0-index column of data_file containing
                the targets M chain sequences.
            N_chain_col (None or int, optional): 0-index column of data_file containing
                the targets N chain sequences (only for MHCII).
            outdir_col (None or int, optional): 0-index column of data_file containing
                the paths to the output folder for each case.
            template_col (None or int, optional): 0-index column containing the template
                ID to be used for each case. Defaults to None.
            start_row (None or int, optional): Starting row of data_file, to use when
                splitting the data_file into multiple batches. This allows to
                specify from which row the samples for this job start.
            end_row (None or int, optional): Ending row of data_file, to use when
                splitting the data_file into multiple batches. This allows to
                specify at which row the samples for this job end.
        Returns:
            None.
        """

        targets = {}
        with open(data_file, 'r') as infile:
            spamreader = csv.reader(infile, delimiter=delimiter)
            if header == True:
                next(spamreader)
            for i, row in enumerate(spamreader):
                if start_row != None and i < start_row:
                    pass
                elif end_row != None and i >= end_row:
                    break
                else:
                    ## Assign target ID
                    if IDs_col != None:
                        target_id = row[IDs_col]
                    else:
                        target_id = 'Target_%i' %(i+1)

                    ## Assign peptide sequence
                    peptide_seq = row[peptides_col]

                    ## Assign allele name
                    allele = row[allele_name_col].split(';')

                    ## Make target entry
                    targets[target_id] = {'peptide_sequence' : peptide_seq,
                                            'allele' : allele, 'ID':target_id}

                    ## Assign optional arguments. Be sure the empty values correspond
                    ## to the default values in PMHC.Target.__init__()
                    ## Assign anchors
                    if anchors_col:
                        anchors = list([int(x) for x in row[anchors_col].split(';')])
                        targets[target_id]['anchors'] = anchors
                    else:
                        targets[target_id]['anchors'] = []

                    ## Assign M chain sequence
                    if M_chain_col:
                        M_chain_seq = row[M_chain_col]
                        targets[target_id]['M_chain_seq'] = M_chain_seq
                    else:
                        targets[target_id]['M_chain_seq'] = ''

                    ## Assign N chain sequence
                    if N_chain_col:
                        N_chain_seq = row[N_chain_col]
                        targets[target_id]['N_chain_seq'] = N_chain_seq
                    else:
                        targets[target_id]['N_chain_seq'] = ''

                    ## Assign output directory per case
                    if outdir_col:
                        outdir = row[outdir_col]
                        targets[target_id]['outdir'] = outdir
                    else:
                        targets[target_id]['outdir'] = ''

                    ## Assign template per case
                    if template_col:
                        template = row[template_col]
                        targets[target_id]['template'] = [template]
                    else:
                        targets[target_id]['template'] = None

        self.targets = targets
        
    def prep_collective_output_dir(self):
        ''' Create an output directory and move the template pdb there
            Uses self.output_dir (str): Path to output directory. Defaults to os.getcwd().
            
        Args:
            None
        '''

        # create an output directory
        try:
            if not os.path.exists(self.collective_output_dir):
                os.makedirs(self.collective_output_dir)
                if not os.path.exists(self.collective_output_dir):
                    raise Exception('A problem occurred while creating wrapper output directory')
        except:
            raise Exception('A problem occurred while creating wrapper output directory')
            
            
def archive_and_remove(case):
    """Archives the case folder as a .tar file to save inode space

    Args:
        case (str): directory name of case to be archived
    """ 
    prefix_case_folder = os.path.split(case.rstrip('/'))[0]
    case_folder = os.path.split(case.rstrip('/'))[1]   
    try:
        subprocess.run(f"tar -cf {case}.tar -C {prefix_case_folder} {case_folder} \
                       --remove-files", shell=True, check=True)
    except subprocess.CalledProcessError as cpe:
        print(f"Something went wrong in archive case: {case}\n{cpe}")
    except Exception as e:
        print(e)


def run_case(args):
    """Runs one modelling job. Meant to be runned from Pandora.Wrapper
    
    Args:
        args (list): List of arguments. Should be containing the following, in
        order.
        target_id (str): Target id.
        n_loop_models (int, optional): Number of loop models. Defaults to 20.
        benchmark (bool, optional): Set True if running a benchmark to retrieve
        models RMSD with reference structures. Defaults to False.
        
    Returns:
        None.
    """

    target_id = args['target_id']

    # Create Pandora Object
    if args['outdir'] != '':
        output_dir = args['outdir']
    elif args['outdir'] == '' and args['collective_output_dir']:
        output_dir = args['collective_output_dir']
    else:
        output_dir = False
    
    try:
        tar = Target(target_id, allele_type=args['allele'],
                            peptide=args['peptide_sequence'] ,
                            MHC_class=args['MHC_class'], anchors=args['anchors'],
                            M_chain_seq=args['M_chain_seq'],
                            N_chain_seq=args['N_chain_seq'],
                            use_netmhcpan=args['use_netmhcpan'], use_templ_seq=args['use_templ_seq'], 
                            output_dir=output_dir, rm_netmhcpan_output=args['rm_netmhcpan_output'])
                                             
    except Exception as err:
        print('Skipping Target %s at Target object generation step for the following reason:' %target_id)
        print(("Exception: {0}".format(err)))
        return
    
    try:
        case = Pandora.Pandora(tar, database=args['db'], template=args['template'])
    except Exception as e:
        print(f"Modelling case {target_id} failed at Pandora object creation step")
        print(f"Captured error: {e}")
        print(traceback.format_exc())
        return

    # Run the modelling
    try:
        case.model(n_loop_models=args['n_loop_models'], n_jobs=args['n_jobs'],
                benchmark=args['benchmark'], pickle_out=args['pickle_out'],
                clip_C_domain=args['clip_C_domain'], restraints_stdev=args['restraints_stdev'])

    except Exception as e:
        print(f"Modelling case {target_id} failed at modelling step")
        print(f"Captured error: {e}")
        print(traceback.format_exc())
        return
    
    try:
        if args['archive_output']:
            archive_and_remove(tar.output_dir)
    except Exception as e:
        print(f"Modelling case {target_id} failed at archiving step")
        print(f"Captured error: {e}")
        print(traceback.format_exc())
        return
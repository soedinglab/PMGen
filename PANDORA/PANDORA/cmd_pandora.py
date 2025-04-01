import argparse

from PANDORA import Target
from PANDORA import Pandora
from PANDORA import Database
from PANDORA import Wrapper

def cmd_install_database():
    """Command line tool to download the database from zotero
    """    
    parser = argparse.ArgumentParser(
    prog="PANDORA",
    description="PANDORA fetch database from zotero",
    )

    parser.add_argument(
        '-d','--destination-path', default='~/PANDORA_databases/default',
        help='Full path to the database destination'
    )

    args = parser.parse_args()

    Database.install_database(db_path=args.destination_path)

def cmd_create_database():
    """Command line tool to generate a new database crom scratch
    """    
    parser = argparse.ArgumentParser(
    prog="PANDORA",
    description="Generate a new PANDORA database. As the procedure can be long,\
     we advice to use the --num-cores option to parallelize the process.",
    )

    parser.add_argument(
        '-d','--destination-path', default='~/PANDORA_databases/default',
        help='Full path to the database destination'
    )

    parser.add_argument(
        '-c','--num-cores', default=1, type=int,
        help='Number of CPU cores to use in parallel to generate the database\
            using multiple cores is highly recommended as it drastically reduces\
            the database generation time.',
    )

    args = parser.parse_args()

    db = Database.Database()
    db.construct_database(n_jobs=args.num_cores)


def cmd_run_pandora():
    """Command line tool to run one pandora case
    """
    parser = argparse.ArgumentParser(
    prog="PANDORA",
    description="Run one pMHC modelling case",
    epilog="Example: pandora-run -p LLFGYPVYV -a HLA-A*02:01 -m I",
    )
    parser.add_argument(
        '-a','--allele', required=True, type=str,
        help='Name of the target MHC allele. Multiple alleles should be separated by a comma.',
    )
    parser.add_argument(
        '-c','--clip-C-domain',
        help='If provided, ignores C-like domains and Beta-2 microglobulin and only \
                models binding groove and peptide',
        action='store_true'
    )
    parser.add_argument(
        '-i','--id', 
        help='ID of the target case, used to name the output folder.\
             If not provided, it will default to peptide_allele',
    )
    parser.add_argument(
        '-k','--anchors', 
        help='Peptide anchor positions.\
            To be provided as series of integers separated by comma only.\
            Example: 2,9 for pMHC-I or 4,7,9,12 for pMHC-II',
    )
    parser.add_argument(
        '-l','--loop-models', default=20, type=int,
        help='Number of loop models to produce',
    )
    parser.add_argument(
        '-m','--mhc-class', required=True, type=str,
        help='MHC class', choices=['I','II'],
    )
    parser.add_argument(
        '-o','--output-path', default='./',
        help='Output folder.',
    )
    parser.add_argument(
        '-p','--peptide', required=True, type=str,
        help='One-letter sequence of the target peptide.',
    )
    
    parser.add_argument(
        '-N','--use-netmhcpan', default=False,
        help='Wether or not to use Netmhcpan to predict the anchors.',
        action='store_true'
    )
      
    parser.add_argument(
        '-r', '--restraints-stdev', default=False,
        help='Modelling restraints standard deviation. Keep False for a faster \
            but more rigid modelling. Defaults to False.'
    )

    args, leftovers = parser.parse_known_args()

    # Parse arguments
    if not args.id:
        args.id = f'{args.peptide}_{args.allele}'
    
    if not args.anchors:
        args.anchors = []
    else:
        args.anchors = [int(x) for x in args.anchors.split(',')]

    args.allele=args.allele.split(',')
    
    if args.anchors is None and args.use_netmhcpan == False:
        print('WARNING: No anchors provided and netmhcpan is set to false. PANDORA will use canonical anchoring for MHC-I and raise an error for MHC-II')
    elif args.anchors is not None and args.use_netmhcpan == True:
        print('WARNING: both anchors and use_netmhcpan provided as arguments. PANDORA will use the provided anchors and skip netmhcpan predictions.')
    
    # Load local Database
    db = Database.load()

    # Create Target object
    target = Target(id = args.id,
        allele_type = args.allele,
        peptide = args.peptide,
        anchors = args.anchors,
        use_netmhcpan = args.use_netmhcpan)

    # Perform modelling
    case = Pandora.Pandora(target, db)
    case.model(n_loop_models=args.loop_models, clip_C_domain=args.clip_C_domain,
               restraints_stdev=args.restraints_stdev)

def cmd_run_wrapper():
    """Command line tool to run the Wrapper module
        over one tsv/csv file.
    """
    parser = argparse.ArgumentParser(
    prog="PANDORA",
    description="Run the PANDORA.Wrapper on a tsv or csv file.",
    )

    parser.add_argument(
        '-a','--allele-name-column', required=True, type=int,
        help='0-index of the column containing MHC allele names, separated by semicolon (;)', 
    )
    
    parser.add_argument(
        '-c','--clip-C-domain',
        help='If provided, ignores C-like domains and Beta-2 microglobulin and only \
                models binding groove and peptide',
        action='store_true'
    )
    
    parser.add_argument(
        '-f','--input-file', required=True, type=str,
        help='Input .tsv or .csv file',
    )

    parser.add_argument(
        '-m','--mhc-class', required=True, type=str,
        help='MHC class', choices=['I','II'],
    )

    parser.add_argument(
        '-d','--delimiter', type=str, default=',',
        help='Input file delimiter', choices=['tab',','],
    )    

    parser.add_argument(
        '-H','--header', type=bool, required=True,
        help='Whether there is or not a header. If True, the first line\
            of the input file will be ignored for the modellings.',
    )    

    parser.add_argument(
        '-i','--id', default=False,
        help='ID of the wrapper, used to name the output folder.Should be alphanumeric only.\
                If not, non-alphanumeric characters will be replaced with dashes.\
                If False, it will be randomly generated. Defaults to False.',
    )

    parser.add_argument(
        '-n','--num-cores', default=1, type=int,
        help='Number of CPU cores to use to model the targets in parallel. Default to 1.',
    )

    parser.add_argument(
        '-t','--targets-id-column', type=int,
        help="0-index of the column containing each target's ID. ",
    )

    parser.add_argument(
        '-p','--peptides-column', required=True, type=int,
        help='0-index of the column containing one-letter target peptides sequences.',
    )

    parser.add_argument(
        '-k','--anchors-column', type=int,
        help='0-index of the column containing peptide anchor positions.\
            To be provided as series of integers separated by semicolon only.\
            Example: 2;9 for pMHC-I or 4;7;9;12 for pMHC-II',
    )

    parser.add_argument(
        '-l','--loop-models', default=20, type=int,
        help='Number of loop models to produce',
    )

    parser.add_argument(
        '-o','--output-path', default='./',
        help='Output folder. Defaults to the current working directory.',
    )
    
    parser.add_argument(
        '-r', '--restraints-stdev', default=False,
        help='Modelling restraints standard deviation. Keep False for a faster \
            but more rigid modelling. Defaults to False.'
    )
    
    parser.add_argument(
        '-N','--use-netmhcpan', default=False,
        help='Wether or not to use Netmhcpan to predict the anchors.',
        action='store_true'
    )


    args, leftovers = parser.parse_known_args()

    # Load local database
    db = Database.load()

    if args.delimiter=='tab':
        args.delimiter='\t'

    if args.anchors_column is None and args.use_netmhcpan == False:
        print('WARNING: No anchors column provided and netmhcpan is set to false. PANDORA will use canonical anchoring for MHC-I and raise an error for MHC-II')
    elif args.anchors_column is not None and args.use_netmhcpan == True:
        print('WARNING: both anchors column and use_netmhcpan provided as arguments. PANDORA will use the provided anchors and skip netmhcpan predictions.')
             
    # Create the wrapper object. It will also run the modelling for each case.
    wrap =  Wrapper.Wrapper(data_file=args.input_file, database=db, 
        num_cores=args.num_cores, wrapper_id=args.id, header=args.header,
        MHC_class=args.mhc_class, delimiter=args.delimiter,
        IDs_col=args.targets_id_column, peptides_col=args.peptides_column,
        allele_name_col=args.allele_name_column, anchors_col=args.anchors_column,
        n_loop_models=args.loop_models, collective_output_dir=args.output_path,
        clip_C_domain=args.clip_C_domain, restraints_stdev=args.restraints_stdev,
        use_netmhcpan=args.use_netmhcpan)

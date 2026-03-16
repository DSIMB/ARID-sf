import sys,os
import pandas as pd
import numpy as np
import make_ensemble

def create_cfg(ensemble_path,path_directory,ncores):

    content = ['run_dir = "{}"'.format(os.path.join(path_directory,'topology')),
                        '',
                        'ncores = {}'.format(ncores),
                        'mode = "local"',
                        'clean = false',
                        'molecules = [',
                        f'"{ensemble_path}",',
                        ']',
                        '',
                        '[topoaa]',
                        'tolerance = 99',
                        'set_bfactor = false',
                        # you can add modules and parameters here
                        ]

    
    return content


def write_cfg(cfg_content,cfg_file_path):
    with open(cfg_file_path,'w') as out:
        for l in cfg_content:
            out.write(l + '\n')


def load_data(guideline_files):
    guideline = pd.read_csv(guideline_files,engine='c',sep='\t')
    
    dictionnary_elems = {}
    for row in guideline.itertuples():
        if not os.path.exists(row.PDBpath):
            print(f'Did not found {row.PDBpath}')
            continue
        dictionnary_elems[row.PDBpath] = {'chains_AG':row.chains_AG.split(','),'chains_AB':row.chains_AB.split(',')}
    return dictionnary_elems


def organizer(ensemble_path,guideline_files,ncores):
    
    dictionnary_elems = load_data(guideline_files)
    make_ensemble.organizer(dictionnary_elems,ensemble_path)
    path_directory = os.path.dirname(os.path.abspath(ensemble_path))
    cfg_content = create_cfg(ensemble_path,path_directory,ncores)
    cfg_file_path = os.path.join(path_directory,'topology.cfg')
    write_cfg(cfg_content,cfg_file_path)

if __name__ == '__main__':
    
    ensemble_path = sys.argv[1] # path to the directory containing the pdbs to format
    guideline_files = sys.argv[2] # a .csv file with for each line: PDBpath,chains_AG,chains_AB
    ncores = int(sys.argv[3])

    organizer(ensemble_path,guideline_files,ncores)
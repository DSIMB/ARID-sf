"""
Usage:
python parse_haddock_top.py path_haddock_topol path_models_output
"""

import sys,os
import pandas as pd
import shutil

def process_digit(path_digit_in,path_digit_out):
    # read traceback
    os.makedirs(path_digit_out,exist_ok=True)

    tbf = os.path.join(path_digit_in,'topology/traceback/traceback.tsv')
    if not os.path.exists(tbf):
        path_topoaa = os.path.join(path_digit_in,'topology/0_topoaa/')
        list_pdbs = [p for p in os.listdir(path_topoaa) if p.endswith('.pdb')]

        dict_association = {}
        for pdbf in list_pdbs:
            if pdbf.startswith('ensemble_'):
                continue
            elif pdbf.count('from_ensemble_') == 1:
                ensemble_correspond = os.path.join(path_topoaa,pdbf)
                if os.path.exists(ensemble_correspond):
                    model_name = pdbf.split('_from_')[0]
                    dict_association[model_name] = ensemble_correspond
        
        for model_name,file_in in dict_association.items():
            file_out = os.path.join(path_digit_out, model_name + '.pdb')
            print(file_in)

            shutil.copy2(file_in,file_out)


if __name__ == '__main__':
    path_digit_in = sys.argv[1]
    path_digit_out = sys.argv[2]
    process_digit(path_digit_in,path_digit_out)
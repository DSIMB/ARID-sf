import sys,os
import numpy as np
import format_structure

def organizer(dictionnary_elems,output_path):
    remarks = []
    ensemble = []
    modeli = 1
    for f,elems in dictionnary_elems.items():
        try:
            formated_content = format_structure.organizer(pdbfile=f,chains_AG=np.array(elems['chains_AG']),chains_AB=np.array(elems['chains_AB']),output_path=None)
            if len(formated_content) == 0:
                continue
            ensemble.append('MODEL{:>9}                                                                  \n'.format(modeli))
            ensemble.extend(formated_content)
            ensemble.append('ENDMDL                                                                          \n')
            remarks.append('REMARK     MODEL {} FROM {}                                              \n'.format(modeli,os.path.basename(f)))
            modeli += 1
        except:
            print(f'skipping {f}')
            continue

    ensemble.append('END                                                                             \n')
    ensemble = remarks + ensemble
    
    with open(output_path,'w') as fout:
        for l in ensemble:
            fout.write(l)

    return output_path

if __name__ == '__main__':
    
    path_files = sys.argv[1]
    chains_AG = sys.argv[2].split(',') # chains in the original file
    chains_AB = sys.argv[3].split(',')
    output_path = sys.argv[4] # do we return lines or write a file if output_path is not None

    
    # usage: pdbfilepath A H,L Yes
    list_pdbfiles = [os.path.join(path_files,f) for f in os.listdir(path_files) if f.endswith('.pdb')]
    dictionnary_elems = {f:{'chains_AG':chains_AG,'chains_AB':chains_AB} for f in list_pdbfiles}
    organizer(dictionnary_elems,output_path)
import sys, os
import pandas as pd
import scorer
import time
import numpy as np




def getscores(list_pdb_paths,params_kwargs):
    try:
        df_by_model = scorer.run_scorer(list_pdb_paths,
                    path_weights=os.path.join(os.path.abspath(os.path.dirname(__file__)),'ARID_20_Std.pt') ,
                    cap_length=75,
                    batch_size=1000,
                    n_workers=40,
                    feature_cap=1000,
                    params_kwargs=params_kwargs,
                    )
        return df_by_model
    except:
        print('Failed entire computation')
        return pd.DataFrame()


path_dir = sys.argv[1]
output_name = sys.argv[2]

list_dir_paths = [os.path.join(path_dir,d) for d in os.listdir(path_dir)]
list_dir_paths = [d for d in list_dir_paths if os.path.isdir(d)]


CAP_MEMORY = 1000
params_kwargs = {
    "n_neighbors": 20,               # number of neighbors to consider per residue
    "cutoff_interface": 0.5,         # nm, defines contact at the interface
    "cutoff_potential": 0.85,        # nm, cutoff for potential energy computation
    "r_on": 0.65,                    # nm, smoothing radius for shifting function
    "grid_size": 3,                  # voxel grid size (number of voxels along one edge)
    "voxel_size": 1.0,               # voxel size in nm (depending on feature generator)
    "target_shape_residue": 40,      # length of residue-level local feature vector
    "target_shape_residue_large": 96 # length of Ab,ag and interface-level feature vector
}


print(f'Processing {len(list_dir_paths)} directories')
ti = time.time()
ldfs = []
for i,d in enumerate(list_dir_paths):
    if not os.path.exists(d):
        print(f'skipping {d} path do not exist')
        continue

    list_pdb_paths = [os.path.join(d,f) for f in os.listdir(d) if f.endswith('.pdb')]
    if len(list_pdb_paths) == 0:
        print(f'skipping {d} not pdb found')
        continue

    print(f'\nProcessing  {os.path.basename(d)} {i}')
    if len(list_pdb_paths) > CAP_MEMORY:
        list_pdb_paths = np.array_split(list_pdb_paths,int(len(list_pdb_paths)/CAP_MEMORY))
    else:
        list_pdb_paths = [list_pdb_paths]
    
    for a_list_pdb_paths in list_pdb_paths:
        ldfs.append(getscores(a_list_pdb_paths,params_kwargs))

ldfs = pd.concat(ldfs)

print('Writing',output_name)
ldfs.to_csv(output_name,index=False)

tt = time.time() - ti
print(f'Finished processing {len(list_dir_paths)} directories in {tt/3600:.2f}h') 
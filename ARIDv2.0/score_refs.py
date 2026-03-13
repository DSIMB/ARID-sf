import sys, os
import pandas as pd
import scorer
import time
import numpy as np


def getscores(list_pdb_paths,params_kwargs):
    try:
        df_by_model = scorer.run_scorer(list_pdb_paths,
                    path_weights='./ARID_20_Std.pt',
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

list_ref_paths = [os.path.join(path_dir,f) for f in os.listdir(path_dir)]
list_ref_paths = [f for f in list_ref_paths if f.endswith('.pdb')]


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

print(f'Processing {len(list_ref_paths)} references')
ti = time.time()
ldfs = []
for i,f in enumerate(list_ref_paths):
    if not os.path.exists(f):
        print(f'skipping {f} path do not exist')
        continue

    print(f'\nProcessing  {os.path.basename(f)} {i}')
    ldfs.append(getscores([f],params_kwargs))

ldfs = pd.concat(ldfs)

print('Writing',output_name)
ldfs.to_csv(output_name,index=False)

tt = time.time() - ti
print(f'Finished processing {len(list_ref_paths)} directories in {tt/3600:.2f}h') 
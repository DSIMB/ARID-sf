# Nos imports
import sys,os
import numpy as np
import distances_v2
import voxel_features
import get_esm_embeddings
import lookup_dict
import time

# Notre selection de statiques
ATOMIC_ELEM = {'C':0,'N':1,'O':2,'S':3,'H':4}
n_atom_elem = len(ATOMIC_ELEM.keys())

RCLASS = {
    # Hydrophobic (nonpolar)
    "ALA": "hyd","VAL": "hyd","LEU": "hyd","ILE": "hyd","MET": "hyd","PRO": "hyd","GLY": "hyd", 
    "PHE": "aro","TRP": "aro","TYR": "aro","HIS": "aro",  # can be neutral or positively charged depending on pH
    # Polar, uncharged
    "SER": "pol","THR": "pol","CYS": "pol","CYX": "pol","ASN": "pol","GLN": "pol",
    "ASP": "neg","GLU": "neg",
    "LYS": "pos","ARG": "pos","HIP": "pos",  # can be neutral or positively charged depending on pH
    }

BBCLASS = {
    'CA':'bb','O':'bb','N':'bb','C':'bb','HN':'bb',}

AGENERAL = {
    'H':'pol','C':'hyd','O':'pol','N':'pol','S':'pol',}

oredered_atom_classes = ['bb', 'hyd', 'pol', 'aro',  'neg',  'pos']
ATOM_CLASSES_DICT = {c:i for i,c in enumerate(oredered_atom_classes)}
oredered_res_classes = ['hyd', 'pol', 'aro',  'neg',  'pos']
RES_CLASSES_DICT = {c:i for i,c in enumerate(oredered_res_classes)}
n_atom_classes=len(oredered_atom_classes)
n_res_classes=len(oredered_res_classes)

lookup_table = lookup_dict.full_lookup

# ============================ ============================ ========Le reader Mr======== ============================ ============================
# ============================ ============================ ========Le reader Mr======== ============================ ============================

def read_pdbfile(file_path):
    n_atoms = 0
    n_residues = 0
    seen_chain_B = False
    start_of_chain_B = 0
    atom_ixs = []
    atom_elems = []
    atom_types = []
    residue_types = []
    residue_ixs = []
    chain_ids = []
    cx = []
    cy = []
    cz = []

    with open(file_path, 'r') as fin:
        for line in fin:
            if line.startswith('ATOM'):
                atom_ixs.append(n_atoms)
                atom_name = line[12:16].strip()
                atom_elems.append(ATOMIC_ELEM[atom_name[0]])
                atom_types.append(atom_name)


                # Parse residue name
                resname = line[17:20].strip()
                residue_types.append(resname)
                
                # Parse residue number
                res_ix = int(line[22:26])-1
                residue_ixs.append(res_ix) # start at 0
                if res_ix != n_residues:
                    n_residues += 1

                # Parse Chain id
                chain_id = 0 if (line[21:22]  == 'A') else 1 
                chain_ids.append(chain_id)
                if (not seen_chain_B) and chain_id:
                    start_of_chain_B = n_atoms
                    seen_chain_B = True
                
                # Parse coordinates
                cx.append(float(line[30:38])/10) # in nm
                cy.append(float(line[38:46])/10) # in nm
                cz.append(float(line[46:54])/10) # in nm

                n_atoms += 1
    n_residues += 1 # its a length not an index


    return (
        n_atoms,n_residues,start_of_chain_B,np.array(atom_ixs,dtype=np.int32),
        np.array(atom_elems,dtype=np.int32),atom_types,residue_types,np.array(residue_ixs,dtype=np.int32),np.array(chain_ids,dtype=np.int32),
        np.array(cx,dtype=np.float64),np.array(cy,dtype=np.float64),np.array(cz,dtype=np.float64) 
    )


# ============================ ============================ ========Le parametre Mme======== ============================ ============================
# ============================ ============================ ========Le parametre Mme======== ============================ ============================

def make_atom_residue_classes_and_assign_parameters(atom_types,residue_types,n_atoms,residue_first_atom,residue_atom_count,lookup_table):

    atom_classes = np.zeros(n_atoms,dtype=np.int32)
    residue_class = np.zeros(n_atoms,dtype=np.int32)
    sigmas = np.zeros(n_atoms,dtype=np.float64)
    epsilons = np.zeros(n_atoms,dtype=np.float64)
    charges = np.zeros(n_atoms,dtype=np.float64)
    atom_bbelems = np.full(n_atoms,fill_value=-1,dtype=np.int32)

    for ri,istart in enumerate(residue_first_atom):
        curr_res = residue_types[istart]

        for iatom in range(istart,istart+residue_atom_count[ri]):

            # assign force field params

            if curr_res == 'HIS':
                if residue_atom_count[ri] == 13:
                    curr_res = 'HIP'
            elif curr_res == 'CYS':
                if residue_atom_count[ri] == 7:
                    curr_res = 'CYX'

            at = atom_types[iatom]
            sigmas[iatom] = lookup_table[curr_res][at]['sigma']
            epsilons[iatom] = lookup_table[curr_res][at]['epsilon']
            charges[iatom] = lookup_table[curr_res][at]['charge']

            # assign atom and residue classes for contacts

            rc = RCLASS[curr_res]
            if at in BBCLASS.keys():
                ac = BBCLASS[at]
                if at == 'CA':
                    atom_bbelems[iatom] = 0
                elif at == 'N':
                    atom_bbelems[iatom] = 1
            else:
                ac = AGENERAL[at[0]]
                if at[0] != 'H':
                    if ((rc == 'aro') or (curr_res == 'HIP')) and (ac == 'apo'): # consider HIP C ring contact as aromatic
                        ac = rc
                    elif ((rc == 'neg') or (rc == 'pos')) and (ac == 'pol'):
                        ac = rc
            atom_classes[iatom] = ATOM_CLASSES_DICT[ac]
            residue_class[iatom] = RES_CLASSES_DICT[rc]


    return atom_classes,residue_class,sigmas,epsilons,charges,atom_bbelems


# ============================ ============================ ========Le parametre Gll======== ============================ ============================
# ============================ ============================ ========Le parametre Gll======== ============================ ============================


class Parameters():
    """Parameters for featurisation"""
    def __init__(self, 
                 n_neighbors=20, # number of neighbors to consider 
                 cutoff_interface=0.5, # in nm 
                 cutoff_potential=0.85, # in nm
                 r_on=0.65, # smoothing for shifting function
                 grid_size=3, # number of voxels that make the cube
                 voxel_size=1.0, #  voxels size
                 target_shape_residue = 40,
                 target_shape_residue_large = 96,

                 ):
        self.n_neighbors = n_neighbors
        self.cutoff_interface = cutoff_interface
        self.cutoff_potential = cutoff_potential
        self.r_on = r_on
        self.r_on2 = r_on*r_on
        self.r_off2 = cutoff_potential*cutoff_potential
        self.denominator = (self.r_off2 - self.r_on2)**3
        if grid_size % 2 == 0:
            raise ValueError("Grid size must not be even to properly center the cube on the interface residue")
        
        self.grid_size = grid_size
        self.voxel_size = voxel_size

        self.system_dict = None
        # internal
        self.target_shape_residue = target_shape_residue
        self.target_shape_residue_large = target_shape_residue_large
        self.size_interface_features = target_shape_residue_large*3 + 5
        self.size_contact_array = n_atom_classes*n_atom_classes*2 + n_res_classes*n_res_classes*2
        self.size_residue_features =  6 + target_shape_residue + self.size_contact_array
        self.size_cube_features = 16 # 16 : 3 posix identifier, 5atom types *2 chains + 3 A,B,empty
        self.size_features = self.size_interface_features + self.size_residue_features*(n_neighbors+1) + self.size_cube_features*grid_size*grid_size*grid_size

    def set_system(self,dictionnary):
        self.system_dict = dictionnary

    def output_sizes(self):
        print('Sizes:')
        print(f'\tsize_embedding_residue:{self.target_shape_residue}')
        print(f'\tsize_embedding_interface:{self.target_shape_residue_large}')
        print(f'\tsize_interface_features:{self.size_interface_features}')
        print(f'\tsize_contact_array:{self.size_contact_array}')
        print(f'\tsize_residue_features:{self.size_residue_features}')
        print(f'\tsize_cube_features:{self.size_cube_features}')
        print(f'\tsize_features:{self.size_features}')


def concatenate_per_residue(
        n_interface_residues,n_neighbors,unique_residues,size_features,size_interface_features,size_residue_features,
        chain_ids, residue_first_atom,
        residue_lj_attract,residue_lj_repulsi,residue_electrosta, # potential size n_residues
        residue_embedding_identifiers_small,em_id, # embeddings and identifiers size n_residues
        contact_array2, # contact features size unique residues
        all_neighbors, all_distances, all_angles, # neigbors features size ninterface
        cube_per_interface_residue, # cube features size ninterface
    ):
    # Pre-allocate all temporary arrays
    single_values = np.zeros(6, dtype=np.float64)
    unique_to_idx = {res: idx for idx, res in enumerate(unique_residues)}
    
    feature_array = np.zeros((n_interface_residues, size_features), dtype=np.float64)
    
    # Vectorized interface features
    feature_array[:, :size_interface_features] = em_id


    for i in range(n_interface_residues):
        # Per interface features
        feature_array[i, :size_interface_features] = em_id
        
        # Per neighbor features
        for n in range(n_neighbors + 1):
            ni = all_neighbors[i][n]
            
            # Check validity
            if ni < 0:  # Invalid neighbor marker
                raise ValueError(f"Invalid neighbor marker {ni}")
                
            start_fi = size_interface_features + n * size_residue_features
            end_fi = start_fi + size_residue_features
            
            # Get unique residue index efficiently
            u = unique_to_idx.get(ni, None)
            if u is None:
                raise ValueError(f"Residue {ni} not in unique_residues")
            
            # Build features
            single_values = np.array([
                all_distances[i][n],
                all_angles[i][n],
                chain_ids[residue_first_atom[ni]],
                residue_lj_attract[ni],
                residue_lj_repulsi[ni],
                residue_electrosta[ni]
            ])
            
            # Concatenate
            feature_array[i, start_fi:end_fi] = np.concatenate([
                single_values,
                residue_embedding_identifiers_small[ni],
                contact_array2[u]
            ])
        
        cube_start = size_interface_features + (n_neighbors + 1) * size_residue_features
        feature_array[i, cube_start:] = cube_per_interface_residue[i]
    
    return feature_array

def validate_and_filter_features(feature_arrays, identifiers):
    """
    Check for NaN/Inf and return only valid arrays.
    
    """
    # Check for invalid values
    has_nan = np.any(np.isnan(feature_arrays), axis=(1, 2))
    has_inf = np.any(np.isinf(feature_arrays), axis=(1, 2))
    invalid_mask = has_nan | has_inf
    
    if np.any(invalid_mask):
        print(f"Found {invalid_mask.sum()} invalid cases:")
        for i, (case_id, is_invalid) in enumerate(zip(identifiers, invalid_mask)):
            if is_invalid:
                print(f"  Case {case_id}: NaN={has_nan[i]}, Inf={has_inf[i]}")
    
    # Return only valid
    valid_mask = ~invalid_mask
    return feature_arrays[valid_mask,:], [id for id, v in zip(identifiers, valid_mask) if v]


def sort_feature_array_by_inter_contacts(feature_array, 
                                         all_interface_residues, 
                                         unique_residues, 
                                         contact_array2,
                                         chain_ids,
                                         n_atom_classes,
                                         n_res_classes,
                                         residue_first_atom):
    """
    Sort the feature_array so residues with most intermolecular contacts appear first.
    Returns: (sorted_feature_array, sorted_residues, contact_counts)
    """
    n_res_pairs = n_res_classes * n_res_classes
    n_atom_pairs = n_atom_classes * n_atom_classes
    
    # Map residue -> index in contact_array2
    unique_to_idx = {res: idx for idx, res in enumerate(unique_residues)}
    
    contact_counts = []
    for res in all_interface_residues:
        u = unique_to_idx[res]
        contact_row = contact_array2[u]
        
        chain = chain_ids[residue_first_atom[res]]
        
        if chain == 0:  # chain A
            inter = np.sum(contact_row[n_res_pairs:2*n_res_pairs]) \
                    + np.sum(contact_row[2*n_res_pairs + n_atom_pairs:])
        else:  # chain B
            inter = np.sum(contact_row[0:n_res_pairs]) \
                    + np.sum(contact_row[2*n_res_pairs:2*n_res_pairs + n_atom_pairs])
        contact_counts.append(inter)
    
    contact_counts = np.array(contact_counts)
    sort_idx = np.argsort(-contact_counts)  # descending order
    
    sorted_feature_array = feature_array[sort_idx]
    sorted_residues = all_interface_residues[sort_idx]
    return sorted_feature_array, sorted_residues, contact_counts[sort_idx]

def create_features_interface(f,p):
    # read
    n_atoms,n_residues,start_of_chain_B,atom_ixs,atom_elems,atom_types,residue_types,residue_ixs,chain_ids,cx,cy,cz = read_pdbfile(f)
    
    if len(chain_ids) == 0:
        raise ValueError('No chain detected')

    if chain_ids[0] != 0:
        raise ValueError('First atom should be chain A')

    if chain_ids[-1] != 1:
        raise ValueError('Last atom should be chain B')

    # get parent child arrays
    residue_first_atom, residue_atom_count = distances_v2.get_residue_atom_mapping(residue_ixs,n_residues)
    atom_classes,residue_classes,sigmas,epsilons,charges,atom_bbelems = make_atom_residue_classes_and_assign_parameters(atom_types,residue_types,n_atoms,residue_first_atom,residue_atom_count,lookup_table)

    # distance processing
    # get interface residue and in_range interatomic distances
    interface_A, interface_B, pairs_array, distances_array, residue_centers_x, residue_centers_y, residue_centers_z = distances_v2.detect_interface_atoms(cx,cy,cz,n_atoms,n_residues,residue_ixs,start_of_chain_B,p.cutoff_potential,p.cutoff_interface)
    if (interface_A.shape[0] == 0) or (interface_B.shape[0] == 0):
        print(f"Skipping {os.path.basename(f)}: no interface residues")
        return None

    # get geometrical center of the interface
    all_interface_residues = np.concatenate([interface_A,interface_B])

    n_interface_residues = all_interface_residues.shape[0]
    geom_a = np.mean([residue_centers_x[interface_A],residue_centers_y[interface_A],residue_centers_z[interface_A]],axis=1)
    geom_b = np.mean([residue_centers_x[interface_B],residue_centers_y[interface_B],residue_centers_z[interface_B]],axis=1)
    geom_center_interface = np.mean([geom_a,geom_b],axis=0)

    # get neighborhood and all residues involved 
    unique_residues, all_neighbors, all_distances, all_angles = distances_v2.get_interface_neighborhood(all_interface_residues,residue_centers_x,residue_centers_y,residue_centers_z,n_residues,geom_center_interface,p.n_neighbors)

    # get contact residue and atomic definitions
    contact_array2 = distances_v2.get_contact_definition(unique_residues,
    residue_first_atom, # start of residue 
    residue_atom_count, # add this to get the residue span
    residue_centers_x,residue_centers_y,residue_centers_z, # residue center coordinates
    residue_ixs,
    cx,cy,cz, # atom coordinates
    chain_ids, # atom chains [int]
    atom_classes, # atom_classes (predefined) [int]
    residue_classes, # resid_classes (predefined) [int]
    n_atom_classes,n_res_classes,n_residues,p.cutoff_interface,)
    #The contact array looks like: [ res_to_chain0 | res_to_chain1 | atom_to_chain0 | atom_to_chain1 ]

    # calculate potential
    # constants for the calculation
    residue_lj_attract,residue_lj_repulsi,residue_electrosta,total_lj_attract,total_lj_repulsi,total_electrosta = distances_v2.potential_calculator(pairs_array,
                            distances_array, # already sqrt
                            charges, # valences
                            sigmas, # in UI nm
                            epsilons, # in UI kJ/mol
                            residue_ixs, n_residues, 
                            p.cutoff_potential,p.r_off2,p.r_on,p.r_on2,p.denominator,
                            )

    cube_per_interface_residue = voxel_features.compute_voxel_features_for_interface(    atom_elems,
        n_atoms,chain_ids,atom_bbelems,n_atom_elem,
        cx,cy,cz, # atom coordinates (nm)
        residue_first_atom, # start of residue 
        residue_atom_count, # add this to get the residue span
        all_interface_residues,  # Set of (chain, residue_id) tuples
        geom_a,geom_b, # geometrical centers of chain A and B interfaces 
        p.grid_size,p.voxel_size,)

    # create numpy arrays per interface residue
    em_i = np.mean(p.system_dict["residue_embedding_identifiers_large"][all_interface_residues],axis=0)
    em_id = np.concatenate([em_i,p.system_dict['em_a'],p.system_dict['em_b'],[p.system_dict['is_small_ab'],p.system_dict['size_ag'],total_lj_attract,total_lj_repulsi,total_electrosta]])
    
    feature_array = concatenate_per_residue(
        n_interface_residues,p.n_neighbors,unique_residues,p.size_features,p.size_interface_features,p.size_residue_features,
        chain_ids, residue_first_atom,
        residue_lj_attract,residue_lj_repulsi,residue_electrosta, # potential size n_residues
        p.system_dict['residue_embedding_identifiers_small'],em_id, # embeddings and identifiers size n_residues
        contact_array2, # contact features size unique residues
        all_neighbors, all_distances, all_angles, # neigbors features size ninterface
        cube_per_interface_residue, # cube features size ninterface
    )

    dm_identifier = os.path.basename(f).split('.')[0]
    feature_array, all_interface_residues, _ =  sort_feature_array_by_inter_contacts(feature_array, 
                                         all_interface_residues, 
                                         unique_residues, 
                                         contact_array2,
                                         chain_ids,
                                         n_atom_classes,
                                         n_res_classes,
                                         residue_first_atom)
    return feature_array ,all_interface_residues, dm_identifier


def intialize_system(target_system_path,p):
    
    print('Init... System...')
    residue_embedding_identifiers_small,residue_embedding_identifiers_large,em_a,em_b,size_ag,size_ab = get_esm_embeddings.get_folded_embeddings(target_system_path,p.target_shape_residue,p.target_shape_residue_large)
    dictionnary_system = {'residue_embedding_identifiers_small':residue_embedding_identifiers_small,
                        'residue_embedding_identifiers_large':residue_embedding_identifiers_large,
                        'em_a':em_a,'em_b':em_b,
                        'is_small_ab': 1 if size_ab < 150 else 0,
                        'size_ag': size_ag,
                        # detect symmetry ...
                        # fpts properties ...
                        }
    p.set_system(dictionnary_system)
    print('System initialized -8-')
    return p

def organizer(list_of_files,p,system_name):
    
    # load esm 
    print('Preparation... System...')
    target_system_path = list_of_files[0]

    p = intialize_system(target_system_path)
    

    # exemple of usage
    docking_model_n = len(list_of_files)
    print(f'\nProcessing {docking_model_n} docking models')

    tstart = time.time()
    lfeatures = []

    for f in list_of_files:
        lfeatures.append(create_features_interface(f,p)[0])

    lfeatures = validate_and_filter_features(np.array(lfeatures), ['identifiers']*lfeatures.shape[0]) 
    tend = time.time() - tstart

    print(f'\nProcessed {docking_model_n} files in {tend:.2f}')
    print('Size of array',np.array(lfeatures).shape)
    print(f'Finished  !! for system {system_name}')
    return


if __name__ == '__main__':
    # FOR TESTING FEATURES CALCULATION SPEED FOR 1 CPU:
    file_path = 'pathto/structures/mdref_123.pdb'
    N = 50 # number of repeats
    list_of_files = [file_path] * N
    organizer(list_of_files,'system_name')
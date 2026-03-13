# distutils: language = c
# cython: language_level=3

import numpy as np
cimport numpy as np
cimport cython
from libc.stdlib cimport malloc, realloc, free, calloc
from libc.math cimport sqrt
from libc.string cimport memset
from libc.stdlib cimport qsort
from libc.math cimport sqrt

# Initialize numpy
np.import_array()

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def detect_interface_atoms(
    double[:] cx, double[:] cy, double[:] cz,
    int n_atoms, int n_residues,
    int[:] residue_ixs,
    int start_of_chain_B,
    double cutoff_potential,
    double cutoff_interface
):
    """
    Detect interface atoms with dynamic memory allocation.
    REM:[Future version must be static allocation for all functions]

    Returns:
        interface_A: array of residue indices in chain A at interface
        interface_B: array of residue indices in chain B at interface
        pairs_array: array of atom pairs within cutoff_potential
        distances_array: distances for each pair
    """
    cdef:
        double cutoff_interface_squared = cutoff_interface * cutoff_interface
        double cutoff_potential_squared = cutoff_potential * cutoff_potential
        int ai, aj, rxi, rxj, i
        double dx, dy, dz, dist_squared
        
        #  allocation variables
        int capacity = 10000
        int n_pairs = 0
        int* pairs_atoms = <int*>malloc(capacity * 2 * sizeof(int))
        double* pairs_distances = <double*>malloc(capacity * sizeof(double))
        
        # pointers for reallocation
        int* temp_pairs
        double* temp_distances
        
        # Interface residue 
        int* interface_res_A = <int*>malloc(n_residues * sizeof(int))
        int* interface_res_B = <int*>malloc(n_residues * sizeof(int))
        int n_interface_A = 0
        int n_interface_B = 0

        # geom center calculations 
        double* residues_cxs = <double*>malloc(n_residues * sizeof(double))
        double* residues_cys = <double*>malloc(n_residues * sizeof(double))
        double* residues_czs = <double*>malloc(n_residues * sizeof(double))
        int* residues_n_atoms = <int*>malloc(n_residues * sizeof(int))
        
        # tracking unique residues
        char* seen_res_A = <char*>malloc(n_residues * sizeof(char))
        char* seen_res_B = <char*>malloc(n_residues * sizeof(char))
        
        # numpy arrays 
        np.ndarray[np.int32_t, ndim=1] interface_A
        np.ndarray[np.int32_t, ndim=1] interface_B
        np.ndarray[np.int32_t, ndim=2] pairs_array
        np.ndarray[np.float64_t, ndim=1] distances_array
        np.ndarray[np.float64_t, ndim=1] residue_centers_x
        np.ndarray[np.float64_t, ndim=1] residue_centers_y
        np.ndarray[np.float64_t, ndim=1] residue_centers_z
    
    # Check initial allocations
    if (not pairs_atoms or not pairs_distances or not interface_res_A or not interface_res_B or 
        not seen_res_A or not seen_res_B or not residues_cxs or not residues_cys or 
        not residues_czs or not residues_n_atoms):
        # free before raising
        if pairs_atoms: free(pairs_atoms)
        if pairs_distances: free(pairs_distances)
        if interface_res_A: free(interface_res_A)
        if interface_res_B: free(interface_res_B)
        if seen_res_A: free(seen_res_A)
        if seen_res_B: free(seen_res_B)
        if residues_cxs: free(residues_cxs)
        if residues_cys: free(residues_cys)
        if residues_czs: free(residues_czs)
        if residues_n_atoms: free(residues_n_atoms)
        raise MemoryError("Initial allocation failed")
    
    # Initialize arrays to 0
    memset(seen_res_A, 0, n_residues * sizeof(char))
    memset(seen_res_B, 0, n_residues * sizeof(char))
    memset(residues_cxs, 0, n_residues * sizeof(double))
    memset(residues_cys, 0, n_residues * sizeof(double))
    memset(residues_czs, 0, n_residues * sizeof(double))
    memset(residues_n_atoms, 0, n_residues * sizeof(int))
    
    try:
        # First pass: accumulate residue centers for chain B atoms
        for aj in range(start_of_chain_B, n_atoms):
            rxj = residue_ixs[aj]
            residues_cxs[rxj] += cx[aj]
            residues_cys[rxj] += cy[aj]
            residues_czs[rxj] += cz[aj]
            residues_n_atoms[rxj] += 1
        
        # main loop
        for ai in range(start_of_chain_B):
            rxi = residue_ixs[ai]
            residues_cxs[rxi] += cx[ai]
            residues_cys[rxi] += cy[ai]
            residues_czs[rxi] += cz[ai]
            residues_n_atoms[rxi] += 1

            for aj in range(start_of_chain_B, n_atoms):
                rxj = residue_ixs[aj]
                
                dx = cx[ai] - cx[aj]
                dy = cy[ai] - cy[aj]
                dz = cz[ai] - cz[aj]
                dist_squared = dx*dx + dy*dy + dz*dz
                
                if dist_squared <= cutoff_potential_squared:
                    # Check if we need to grow arrays
                    if n_pairs >= capacity:
                        capacity *= 2
                        
                        temp_pairs = <int*>realloc(pairs_atoms, capacity * 2 * sizeof(int))
                        if not temp_pairs:
                            raise MemoryError("Failed to reallocate pairs array")
                        pairs_atoms = temp_pairs
                        
                        temp_distances = <double*>realloc(pairs_distances, capacity * sizeof(double))
                        if not temp_distances:
                            raise MemoryError("Failed to reallocate distances array")
                        pairs_distances = temp_distances
                    
                    # register the pair
                    pairs_atoms[n_pairs * 2] = ai
                    pairs_atoms[n_pairs * 2 + 1] = aj
                    pairs_distances[n_pairs] = sqrt(dist_squared)
                    n_pairs += 1
                    
                    # if interface
                    if dist_squared <= cutoff_interface_squared:

                        
                        # add unique residues to interface
                        if not seen_res_A[rxi]:
                            interface_res_A[n_interface_A] = rxi
                            n_interface_A += 1
                            seen_res_A[rxi] = 1
                        
                        if not seen_res_B[rxj]:
                            interface_res_B[n_interface_B] = rxj
                            n_interface_B += 1
                            seen_res_B[rxj] = 1
        
        # create numpy arrays with exact sizes
        interface_A = np.empty(n_interface_A, dtype=np.int32)
        interface_B = np.empty(n_interface_B, dtype=np.int32)
        pairs_array = np.empty((n_pairs, 2), dtype=np.int32)
        distances_array = np.empty(n_pairs, dtype=np.float64)
        
        for i in range(n_interface_A):
            interface_A[i] = interface_res_A[i]
        
        for i in range(n_interface_B):
            interface_B[i] = interface_res_B[i]
        
        for i in range(n_pairs):
            pairs_array[i, 0] = pairs_atoms[i * 2]
            pairs_array[i, 1] = pairs_atoms[i * 2 + 1]
            distances_array[i] = pairs_distances[i]
        
        residue_centers_x = np.empty(n_residues, dtype=np.float64)
        residue_centers_y = np.empty(n_residues, dtype=np.float64)
        residue_centers_z = np.empty(n_residues, dtype=np.float64)

        # calculate geometric centers and copy to numpy arrays
        for i in range(n_residues):
            if residues_n_atoms[i] > 0:  
                residue_centers_x[i] = residues_cxs[i] / residues_n_atoms[i]
                residue_centers_y[i] = residues_cys[i] / residues_n_atoms[i]
                residue_centers_z[i] = residues_czs[i] / residues_n_atoms[i]
            else:
                residue_centers_x[i] = 0.0
                residue_centers_y[i] = 0.0
                residue_centers_z[i] = 0.0

        return interface_A, interface_B, pairs_array, distances_array, residue_centers_x, residue_centers_y, residue_centers_z

    finally:
        # memory
        if pairs_atoms: free(pairs_atoms)
        if pairs_distances: free(pairs_distances)
        if interface_res_A: free(interface_res_A)
        if interface_res_B: free(interface_res_B)
        if seen_res_A: free(seen_res_A)
        if seen_res_B: free(seen_res_B)
        if residues_cxs: free(residues_cxs)
        if residues_cys: free(residues_cys)
        if residues_czs: free(residues_czs)
        if residues_n_atoms: free(residues_n_atoms)


# ================================== ================================== ===============Neighbors========== ================================== ==================================
# ================================== ================================== ===============Neighbors========== ================================== ==================================

# Structure for sorting
cdef struct DistIndex:
    double dist
    int idx

# comparison function 
cdef int dist_compare(const void* a, const void* b) noexcept nogil:
    cdef double diff = (<DistIndex*>a).dist - (<DistIndex*>b).dist
    if diff < 0:
        return -1 # a < b (a comes first)
    elif diff > 0:
        return 1 # a > b (b comes first)
    else:
        return 0 # equals

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void neighborhood_identification(
    int central_residue,
    double[:] residue_centers_x,
    double[:] residue_centers_y,
    double[:] residue_centers_z,
    int n_residues,
    double geom_center_x,
    double geom_center_y,
    double geom_center_z,
    int n_neighbors,
    int[:] neighbors_out,
    double[:] distances_out,
    double[:] angles_out
) nogil:
    """
    Find N nearest neighbors for a central residue.
    """
    cdef:
        double central_x = residue_centers_x[central_residue]
        double central_y = residue_centers_y[central_residue]
        double central_z = residue_centers_z[central_residue]
        double dx, dy, dz, dist_squared, dist
        double vec1_x, vec1_y, vec1_z, vec2_x, vec2_y, vec2_z
        double dot_product, norm1, norm2, cos_angle
        int ri, i, actual_neighbors
        int n_to_copy
        
        # allocazion
        DistIndex* dist_indices = <DistIndex*>malloc((n_residues-1) * sizeof(DistIndex))
    
    if not dist_indices:
        return
    
    # compute distances for all residues
    actual_neighbors = 0
    for ri in range(n_residues):
        if ri != central_residue:
            dx = residue_centers_x[ri] - central_x
            dy = residue_centers_y[ri] - central_y
            dz = residue_centers_z[ri] - central_z
            dist_squared = dx*dx + dy*dy + dz*dz
            
            dist_indices[actual_neighbors].dist = dist_squared
            dist_indices[actual_neighbors].idx = ri
            actual_neighbors += 1
    
    # Sort by distance
    qsort(dist_indices, actual_neighbors, sizeof(DistIndex), dist_compare)
    
    # set central residue as first element
    neighbors_out[0] = central_residue
    distances_out[0] = 0.0
    angles_out[0] = -2.0
    
    n_to_copy = n_neighbors if n_neighbors < actual_neighbors else actual_neighbors
    
    # fill results with N nearest neighbors
    for i in range(n_to_copy):
        ri = dist_indices[i].idx
        neighbors_out[i+1] = ri
        distances_out[i+1] = sqrt(dist_indices[i].dist)
        
        # angle
        vec1_x = residue_centers_x[ri] - central_x
        vec1_y = residue_centers_y[ri] - central_y
        vec1_z = residue_centers_z[ri] - central_z
        
        vec2_x = geom_center_x - central_x
        vec2_y = geom_center_y - central_y
        vec2_z = geom_center_z - central_z
        
        dot_product = vec1_x*vec2_x + vec1_y*vec2_y + vec1_z*vec2_z
        norm1 = sqrt(vec1_x*vec1_x + vec1_y*vec1_y + vec1_z*vec1_z)
        norm2 = sqrt(vec2_x*vec2_x + vec2_y*vec2_y + vec2_z*vec2_z)
        
        if norm1 > 1e-8 and norm2 > 1e-8:
            cos_angle = dot_product / (norm1 * norm2)
            if cos_angle > 1.0: # float point approx
                cos_angle = 1.0
            elif cos_angle < -1.0: 
                cos_angle = -1.0
            angles_out[i+1] = cos_angle
        else:  # in case of one residue being at the exact center of interface
            angles_out[i+1] = -2.0
    
    #  remainings
    for i in range(n_to_copy + 1, n_neighbors + 1):
        neighbors_out[i] = -1
        distances_out[i] = -1.0
        angles_out[i] = -2.0
    
    free(dist_indices)

@cython.boundscheck(False)
@cython.wraparound(False)
def get_interface_neighborhood(
    int[:] central_residues,
    double[:] residue_centers_x,
    double[:] residue_centers_y,
    double[:] residue_centers_z,
    int n_residues,
    double[:] geom_center_interface,
    int n_neighbors
):
    """
    Get neighborhoods for all interface residues.
    """
    cdef:
        int n_interfacial = central_residues.shape[0]
        int i, j, k, res_idx
        double geom_x = geom_center_interface[0]
        double geom_y = geom_center_interface[1]
        double geom_z = geom_center_interface[2]
        int n_unique = 0
        
        # output arrays
        np.ndarray[np.int32_t, ndim=2] all_neighbors = np.zeros((n_interfacial, n_neighbors+1), dtype=np.int32)
        np.ndarray[np.float64_t, ndim=2] all_distances = np.zeros((n_interfacial, n_neighbors+1), dtype=np.float64)
        np.ndarray[np.float64_t, ndim=2] all_angles = np.zeros((n_interfacial, n_neighbors+1), dtype=np.float64)
        
        np.ndarray[np.int32_t, ndim=1] unique_residues
        
        int[:, :] neighbors_view = all_neighbors
        double[:, :] distances_view = all_distances
        double[:, :] angles_view = all_angles
        
        char* seen_residues = <char*>calloc(n_residues, sizeof(char))
    
    if not seen_residues:
        raise MemoryError("Failed to allocate memory")
    
    try:
        # process each central residue
        for i in range(n_interfacial):
            neighborhood_identification(
                central_residues[i],
                residue_centers_x,
                residue_centers_y,
                residue_centers_z,
                n_residues,
                geom_x, geom_y, geom_z,
                n_neighbors,
                neighbors_view[i],
                distances_view[i],
                angles_view[i]
            )
            
            # unique residues
            for j in range(n_neighbors + 1):
                res_idx = neighbors_view[i, j]
                if res_idx >= 0 and not seen_residues[res_idx]:
                    seen_residues[res_idx] = 1
                    n_unique += 1
        
        # allocate with known size
        unique_residues = np.zeros(n_unique, dtype=np.int32)
        k = 0
        for i in range(n_residues):
            if seen_residues[i]:
                unique_residues[k] = i
                k += 1
        
        return unique_residues, all_neighbors, all_distances, all_angles
    
    finally:
        free(seen_residues)

# ================================== ================================== ===============Residue Atom========== ================================== ==================================
# ================================== ================================== ===============Residue Atom========== ================================== ==================================


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void build_residue_atom_mapping(
    int[:] residue_ixs,  # residue index for each atom
    int n_atoms,
    int n_residues,
    int[:] residue_first_atom,  # output: first atom index for each residue
    int[:] residue_atom_count   # output: number of atoms in each residue
) nogil:
    """
    Build mapping from residues to their atoms.
    """
    cdef int i, current_residue, prev_residue
    
    # Init arrays to -1/0
    for i in range(n_residues):
        residue_first_atom[i] = -1
        residue_atom_count[i] = 0
    
    if n_atoms == 0:
        return
    
    # First atom
    prev_residue = residue_ixs[0]
    residue_first_atom[prev_residue] = 0
    residue_atom_count[prev_residue] = 1
    
    # remainings
    for i in range(1, n_atoms):
        current_residue = residue_ixs[i]
        
        if current_residue == prev_residue:
            # same residue
            residue_atom_count[current_residue] += 1
        else:
            # new residue
            residue_first_atom[current_residue] = i
            residue_atom_count[current_residue] = 1
            prev_residue = current_residue


@cython.boundscheck(False)
@cython.wraparound(False)
def get_residue_atom_mapping(int[:] residue_ixs, int n_residues):
    """
    Build mapping from residues to their atoms.
    
    """
    cdef:
        int n_atoms = residue_ixs.shape[0]
        np.ndarray[np.int32_t, ndim=1] residue_first_atom = np.full(n_residues, -1, dtype=np.int32)
        np.ndarray[np.int32_t, ndim=1] residue_atom_count = np.zeros(n_residues, dtype=np.int32)
        int[:] first_atom_view = residue_first_atom
        int[:] atom_count_view = residue_atom_count
    
    build_residue_atom_mapping(residue_ixs, n_atoms, n_residues, 
                              first_atom_view, atom_count_view)
    
    return residue_first_atom, residue_atom_count

# ================================== ================================== ===============Contacts========== ================================== ==================================
# ================================== ================================== ===============Contacts========== ================================== ==================================


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef int evaluate_list_of_potential_neighbors(
    int considered_residue,
    double[:] residue_centers_x,
    double[:] residue_centers_y,
    double[:] residue_centers_z,
    int n_residues,
    double cutoff_residue_consideration,
    int[:] neighbor_buffer  # Pre-allocated output buffer
) nogil:
    """
    Find residues within cutoff distance from considered residue.
    
    Returns:
        Number of neighbors found (excluding the considered residue itself)
    
    Note: Results are written to neighbor_buffer[0:n_neighbors]
    """
    cdef:
        int ri, k = 0
        double dx, dy, dz, d_squared
        double center_x = residue_centers_x[considered_residue]
        double center_y = residue_centers_y[considered_residue]
        double center_z = residue_centers_z[considered_residue]
    
    for ri in range(n_residues):
        if ri == considered_residue:
            continue
            
        dx = residue_centers_x[ri] - center_x
        dy = residue_centers_y[ri] - center_y
        dz = residue_centers_z[ri] - center_z
        d_squared = dx*dx + dy*dy + dz*dz
        
        if d_squared <= cutoff_residue_consideration:
            neighbor_buffer[k] = ri
            k += 1
    
    return k

# Python-callable wrapper
@cython.boundscheck(False)
@cython.wraparound(False)
def get_potential_neighbors(
    int considered_residue,
    double[:] residue_centers_x,
    double[:] residue_centers_y,
    double[:] residue_centers_z,
    int n_residues,
    double cutoff_residue_consideration
):
    """
    Python wrapper that returns neighbors as numpy array.
    """
    cdef:
        int[:] neighbor_buffer = np.empty(n_residues, dtype=np.int32)
        int n_neighbors
    
    n_neighbors = evaluate_list_of_potential_neighbors(
        considered_residue,
        residue_centers_x,
        residue_centers_y,
        residue_centers_z,
        n_residues,
        cutoff_residue_consideration,
        neighbor_buffer
    )
    
    # Return only the filled portion
    return np.asarray(neighbor_buffer[:n_neighbors])



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void evaluate_atomic_and_residue_contacts(
    int considered_residue,
    int[:] neighbor_residues,
    int n_neighbors,
    int[:] residue_first_atom,
    int[:] residue_atom_count,
    double[:] cx, double[:] cy, double[:] cz,
    char[:] seen_residues_buffer, 
    int n_residues, 
    int[:] chains_atom,
    int[:] atom_classes,
    int[:] residue_classes,
    int n_atom_classes,
    int n_res_classes,
    double cutoff_contact_squared,
    int[:] output_array  # Pre-allocated output
) nogil:
    """
    Calculate contacts for one residue against its neighbors.
    Output format: [res_contacts_A, res_contacts_B, atom_contacts_A, atom_contacts_B]
    PythonVers: Instead of objects, dicts or multiple arrays, everything is coded on the same array, ready to use
    """
    cdef:
        int i, j, atom_i, atom_j, res_j, chain_j
        int first_i, count_i, first_j, count_j
        double dx, dy, dz, d_squared
        int pos_class_contact
        int n_res_pairs = n_res_classes * n_res_classes
        int n_atom_pairs = n_atom_classes * n_atom_classes
        
    
    # output to zero
    for i in range(2 * n_res_pairs + 2 * n_atom_pairs):
        output_array[i] = 0
    
    # seen residues init
    for i in range(n_residues):
        seen_residues_buffer[i] = 0
    
    # get atoms for considered residue
    first_i = residue_first_atom[considered_residue]
    count_i = residue_atom_count[considered_residue]
    
    # loop over atoms of considered residue
    for i in range(count_i):
        atom_i = first_i + i
        
        # over neighbor residues
        for j in range(n_neighbors):
            res_j = neighbor_residues[j]
            first_j = residue_first_atom[res_j]
            count_j = residue_atom_count[res_j]
            
            # over neighbor residue atoms 
            for atom_j in range(first_j, first_j + count_j):
                dx = cx[atom_i] - cx[atom_j]
                dy = cy[atom_i] - cy[atom_j]
                dz = cz[atom_i] - cz[atom_j]
                d_squared = dx*dx + dy*dy + dz*dz
                
                if d_squared <= cutoff_contact_squared:
                    chain_j = chains_atom[atom_j]
                    
                    # atom-atom contact
                    pos_class_contact = atom_classes[atom_i] * n_atom_classes + atom_classes[atom_j]
                    # position in output: atom_contacts start after residue_contacts
                    output_array[2 * n_res_pairs + chain_j * n_atom_pairs + pos_class_contact] += 1
                    
                    # residue-residue contact (only count once per residue pair)
                    if not seen_residues_buffer[res_j]:
                        seen_residues_buffer[res_j] = 1
                        pos_class_contact = residue_classes[atom_i] * n_res_classes + residue_classes[atom_j]
                        # position in output: residue_contacts come first
                        output_array[chain_j * n_res_pairs + pos_class_contact] += 1

@cython.boundscheck(False)
@cython.wraparound(False)
def get_contact_definition(
    int[:] unique_residues,
    int[:] residue_first_atom,
    int[:] residue_atom_count,
    double[:] residue_centers_x,
    double[:] residue_centers_y,
    double[:] residue_centers_z,
    int[:] residue_indexes,
    double[:] cx, double[:] cy, double[:] cz,
    int[:] chains_atom,
    int[:] atom_classes,
    int[:] residue_classes,
    int n_atom_classes,
    int n_res_classes,
    int n_residues,
    double cutoff_contact
):
    """
    Calculate contacts for all unique residues.
    Returns 2D array: [n_unique_residues, n_contact_types]
    """
    cdef:
        int n_unique = unique_residues.shape[0]
        int n_res_pairs = n_res_classes * n_res_classes
        int n_atom_pairs = n_atom_classes * n_atom_classes
        int n_contact_types = 2 * n_res_pairs + 2 * n_atom_pairs
        int ur, considered_residue, n_neighbors
        double cutoff_residue_squared = (cutoff_contact * 3) ** 2
        double cutoff_contact_squared = cutoff_contact * cutoff_contact
        
        cdef char[:] seen_buffer = np.zeros(n_residues, dtype=np.uint8)

        # out array
        np.ndarray[np.int32_t, ndim=2] contact_matrix = np.zeros((n_unique, n_contact_types), dtype=np.int32)
        int[:, :] contact_view = contact_matrix
        
        int[:] neighbor_buffer = np.empty(n_residues, dtype=np.int32)
        int[:] contact_buffer = np.zeros(n_contact_types, dtype=np.int32)
    
    # loop unique residue
    for ur in range(n_unique):
        considered_residue = unique_residues[ur]
        
        # fetch neighbors
        n_neighbors = evaluate_list_of_potential_neighbors(
            considered_residue,
            residue_centers_x, residue_centers_y, residue_centers_z,
            n_residues, cutoff_residue_squared,
            neighbor_buffer
        )
        
        # calculate contacts
        evaluate_atomic_and_residue_contacts(
            considered_residue,
            neighbor_buffer, n_neighbors,
            residue_first_atom, residue_atom_count,
            cx, cy, cz,
            seen_buffer,n_residues, chains_atom,
            atom_classes, residue_classes,
            n_atom_classes, n_res_classes,
            cutoff_contact_squared,
            contact_view[ur]
        )
    
    return contact_matrix



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline double calculate_electrostatic_potential_shifted(
    double qi, double qj, double dij, double dij2, 
    double r_off, double r_off2, double dielectric
) nogil:
    """Calculate electrostatic potential with shifting function"""
    if dij >= r_off:
        return 0.0
    
    cdef double shift_factor = 1.0 - (dij2 / r_off2)
    cdef double epot = (138.935458 / dielectric) * (qi * qj) / dij * shift_factor
    return epot

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline void calculate_lennard_jones_potential(
    double sigmai, double sigmaj, double epsiloni, double epsilonj,
    double dij, double dij2, double r_on, double r_on2, 
    double r_off, double r_off2, double denominator,
    double* lj_r, double* lj_a
) nogil:
    """Calculate Lennard-Jones potential with switching function"""
    cdef double sr, numerator
    cdef double sigmaij, epsilonij, rapport_sd, attractive_sd, repulsive_sd
    
    # Switching function
    if dij <= r_on:
        sr = 1.0
    elif dij >= r_off:
        sr = 0.0
    else:
        numerator = (r_off2 - dij2) * (r_off2 - dij2) * (r_off2 + 2*dij2 - 3*r_on2)
        sr = numerator / denominator
    
    # LJ parameters
    sigmaij = (sigmai + sigmaj) * 0.5
    epsilonij = sqrt(epsiloni * epsilonj)
    rapport_sd = sigmaij / dij
    attractive_sd = rapport_sd * rapport_sd * rapport_sd * rapport_sd * rapport_sd * rapport_sd
    repulsive_sd = attractive_sd * attractive_sd
    
    lj_r[0] = 4.0 * epsilonij * repulsive_sd * sr
    lj_a[0] = -4.0 * epsilonij * attractive_sd * sr

@cython.boundscheck(False)
@cython.wraparound(False)
def potential_calculator(
    int[:, :] pairs_atom_index,
    double[:] distance_between_atoms,
    double[:] charges,
    double[:] sigmas,
    double[:] epsilons,
    int[:] residue_ixs,
    int n_residues,
    double r_off,
    double r_off2,
    double r_on,
    double r_on2,
    double denominator,
    double dielectric=10.0
):
    """
    Calculate potentials for all atom pairs.
    
    Returns:
        Per-residue arrays and totals for LJ attractive, LJ repulsive, and electrostatic
    """
    cdef:
        int n_pairs = pairs_atom_index.shape[0]
        int pi, atmi, atmj, rxi, rxj
        double dij, dij2, lj_r, lj_a, elec
        double total_lj_attract = 0.0
        double total_lj_repulsi = 0.0
        double total_electrosta = 0.0
        
        # out arrays
        np.ndarray[np.float64_t, ndim=1] residue_lj_attract = np.zeros(n_residues, dtype=np.float64)
        np.ndarray[np.float64_t, ndim=1] residue_lj_repulsi = np.zeros(n_residues, dtype=np.float64)
        np.ndarray[np.float64_t, ndim=1] residue_electrosta = np.zeros(n_residues, dtype=np.float64)
        
        # memviews
        double[:] lj_attract_view = residue_lj_attract
        double[:] lj_repulsi_view = residue_lj_repulsi
        double[:] electrosta_view = residue_electrosta
    
    # loop pairs
    for pi in range(n_pairs):
        atmi = pairs_atom_index[pi, 0]
        atmj = pairs_atom_index[pi, 1]
        
        dij = distance_between_atoms[pi]
        dij2 = dij * dij
        
        calculate_lennard_jones_potential(
            sigmas[atmi], sigmas[atmj], epsilons[atmi], epsilons[atmj],
            dij, dij2, r_on, r_on2, r_off, r_off2, denominator,
            &lj_r, &lj_a
        )
        
        elec = calculate_electrostatic_potential_shifted(
            charges[atmi], charges[atmj], dij, dij2, r_off, r_off2, dielectric
        )
        
        # get res indexs
        rxi = residue_ixs[atmi]
        rxj = residue_ixs[atmj]
        
        # accumulate per-residue
        lj_attract_view[rxi] += lj_a
        lj_attract_view[rxj] += lj_a
        lj_repulsi_view[rxi] += lj_r
        lj_repulsi_view[rxj] += lj_r
        electrosta_view[rxi] += elec
        electrosta_view[rxj] += elec
        
        # accumulate totals
        total_lj_attract += 0.5 * lj_a
        total_lj_repulsi += 0.5 * lj_r
        total_electrosta += 0.5 * elec
    
    return (residue_lj_attract, residue_lj_repulsi, residue_electrosta,
            total_lj_attract, total_lj_repulsi, total_electrosta)
# voxel_features.pyx
# distutils: language = c
# cython: language_level=3

import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sqrt, fabs, pow
from libc.stdlib cimport malloc, free

np.import_array()

# Constants
cdef double PI = 3.141592653589793
cdef double[6] VDW_RADII = [0.170, 0.155, 0.152, 0.180, 0.180, 0.120]  # in nm

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline double get_vdw_radius(int atom_type) nogil:
    if 0 <= atom_type < 6:
        return VDW_RADII[atom_type]
    return VDW_RADII[5]  # default to H

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline double calculate_vdw_volume(int atom_type) nogil:
    cdef double radius = get_vdw_radius(atom_type)
    return (4.0/3.0) * PI * radius * radius * radius

# Vector operations
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void normalize_vector(double* v) nogil:
    cdef double norm = sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])
    if norm > 1e-8:
        v[0] /= norm
        v[1] /= norm
        v[2] /= norm

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void cross_product(double* a, double* b, double* result) nogil:
    result[0] = a[1]*b[2] - a[2]*b[1]
    result[1] = a[2]*b[0] - a[0]*b[2]
    result[2] = a[0]*b[1] - a[1]*b[0]

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void build_residue_frame(double* ca, double* n, double* com_opposite, 
                              double rotation_matrix[3][3]) nogil:
    cdef double z_axis[3]
    cdef double ref_vec[3]
    cdef double x_axis[3]
    cdef double y_axis[3]
    cdef int i
    
    # Z axis: from CA to center of mass
    for i in range(3):
        z_axis[i] = com_opposite[i] - ca[i]
    normalize_vector(z_axis)
    
    # ref vector for X (N - CA)
    for i in range(3):
        ref_vec[i] = n[i] - ca[i]
    normalize_vector(ref_vec)
    
    # X axis
    cross_product(z_axis, ref_vec, x_axis)
    normalize_vector(x_axis)
    
    # Y axis
    cross_product(z_axis, x_axis, y_axis)
    
    # matrix
    for i in range(3):
        rotation_matrix[i][0] = x_axis[i]
        rotation_matrix[i][1] = y_axis[i]
        rotation_matrix[i][2] = z_axis[i]

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void matrix_vector_multiply(double rotation_matrix[3][3], double* vector, 
                                 double* result) nogil:
    cdef int i
    cdef int j
    for i in range(3):
        result[i] = 0.0
        for j in range(3):
            result[i] += rotation_matrix[i][j] * vector[j]

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void compute_voxel_center(double* ca_coords, double rotation_matrix[3][3],
                               int rel_i, int rel_j, int rel_k,
                               int grid_size, double voxel_size,
                               double* voxel_center) nogil:
    cdef double local_offset[3] 
    cdef double global_offset[3]
    cdef double half_grid = (grid_size - 1) / 2.0
    cdef int i
    
    local_offset[0] = (rel_i - half_grid) * voxel_size
    local_offset[1] = (rel_j - half_grid) * voxel_size
    local_offset[2] = (rel_k - half_grid) * voxel_size
    
    matrix_vector_multiply(rotation_matrix, local_offset, global_offset)
    
    for i in range(3):
        voxel_center[i] = ca_coords[i] + global_offset[i]

@cython.boundscheck(False)
@cython.wraparound(False)
cdef bint is_atom_in_voxel(double ax, double ay, double az,
                          double* voxel_center, double rotation_matrix[3][3],
                          double voxel_size) nogil:
    cdef double atom_pos[3]
    cdef double local_pos[3]
    cdef double inv_rotation[3][3]
    cdef double half_size = voxel_size / 2.0
    cdef int i, j
    
    # relative position
    atom_pos[0] = ax - voxel_center[0]
    atom_pos[1] = ay - voxel_center[1]
    atom_pos[2] = az - voxel_center[2]
    
    # transposition for inverse
    for i in range(3):
        for j in range(3):
            inv_rotation[i][j] = rotation_matrix[j][i]
    
    matrix_vector_multiply(inv_rotation, atom_pos, local_pos)
    
    return (fabs(local_pos[0]) <= half_size and 
            fabs(local_pos[1]) <= half_size and 
            fabs(local_pos[2]) <= half_size)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void compute_voxel_volumes(int* atoms, double* cx, double* cy, double* cz,
                               int n_atoms, double* voxel_center,
                               int* atom_elems, int* chain_ids, int n_atom_elem,
                               double rotation_matrix[3][3], double voxel_size,
                               double* volume_array) nogil:
    cdef int i, atmi, atom_elem, chain, pos_partial
    cdef double atom_volume
    cdef double voxel_volume
    cdef double total_vol_AB[2]
    cdef int n_features = n_atom_elem * 2
    
    # init to empty
    for i in range(n_features + 3):
        volume_array[i] = 0.0
    total_vol_AB[0] = 0.0
    total_vol_AB[1] = 0.0
    
    voxel_volume = voxel_size * voxel_size * voxel_size
    
    # process atoms and calc
    for i in range(n_atoms):
        atmi = atoms[i]
        if is_atom_in_voxel(cx[atmi], cy[atmi], cz[atmi], voxel_center, 
                           rotation_matrix, voxel_size):
            atom_elem = atom_elems[atmi]
            chain = chain_ids[atmi]
            atom_volume = calculate_vdw_volume(atom_elem)
            
            pos_partial = atom_elem + n_atom_elem * chain
            volume_array[3 + pos_partial] += atom_volume
            total_vol_AB[chain] += atom_volume
    
    # normalisation
    if voxel_volume > 0:
        for i in range(3, n_features + 3):
            volume_array[i] = min(1.0, volume_array[i] / voxel_volume)
        volume_array[0] = min(1.0, total_vol_AB[0] / voxel_volume)
        volume_array[1] = min(1.0, total_vol_AB[1] / voxel_volume)
    
    # calculation of the empty volume
    volume_array[2] = max(0.0, 1.0 - volume_array[0] - volume_array[1])

@cython.boundscheck(False)
@cython.wraparound(False)
def compute_voxel_features_for_interface(
    int[:] atom_elems,
    int n_atoms,
    int[:] chain_ids,
    int[:] atom_bbelems,
    int n_atom_elem,
    double[:] cx, double[:] cy, double[:] cz,
    int[:] residue_first_atom,
    int[:] residue_atom_count,
    int[:] interface_residues,
    double[:] geom_a, double[:] geom_b,
    int grid_size,
    double voxel_size
):
    """Compute voxel features for interface residues."""
    
    if grid_size % 2 == 0:
        raise ValueError("Grid size must be odd")
    
    cdef:
        int n_interface_residues = interface_residues.shape[0]
        int n_volumic_features = 3 + n_atom_elem * 2
        int n_features_per_voxel = 3 + n_volumic_features
        int total_voxels = grid_size * grid_size * grid_size
        int central_position = (grid_size - 1) // 2
        
        # Output array
        np.ndarray[np.float64_t, ndim=2] feature_array = np.zeros(
            (n_interface_residues, n_features_per_voxel * total_voxels), 
            dtype=np.float64
        )
        double[:, :] feature_view = feature_array
        
        # Workspace
        double global_min[3]
        double global_max[3]
        double ca_pos[3]
        double n_pos[3]
        double com_opposite[3]
        double rotation_matrix[3][3]
        double voxel_center[3]
        double* volume_buffer
        int* filtered_atoms
        int n_filtered, i, j, k, ri, residue, atm_start
        int voxel_idx, start_idx, vi, chain_curr
        double max_radius = 0.0
        bint ca_found, n_found
    
    # Find max VDW radius
    for i in range(6):
        if VDW_RADII[i] > max_radius:
            max_radius = VDW_RADII[i]
    
    # Allocate workspace
    volume_buffer = <double*>malloc((n_volumic_features + 3) * sizeof(double))
    filtered_atoms = <int*>malloc(n_atoms * sizeof(int))
    
    if not volume_buffer or not filtered_atoms:
        free(volume_buffer)
        free(filtered_atoms)
        raise MemoryError("Failed to allocate workspace")
    
    try:
        # Initialize bounds
        for i in range(3):
            global_min[i] = 1e5
            global_max[i] = -1e5
        
        # First pass: compute bounds
        for ri in range(n_interface_residues):
            residue = interface_residues[ri]
            atm_start = residue_first_atom[residue]
            
            # Get backbone atoms
            ca_pos[0] = ca_pos[1] = ca_pos[2] = -1.0
            n_pos[0] = n_pos[1] = n_pos[2] = -1.0
            ca_found = False
            n_found = False

            for i in range(residue_atom_count[residue]):
                j = atm_start + i
                if atom_bbelems[j] == 0:  # CA
                    ca_pos[0] = cx[j]
                    ca_pos[1] = cy[j]
                    ca_pos[2] = cz[j]
                    ca_found = True
                elif atom_bbelems[j] == 1:  # N
                    n_pos[0] = cx[j]
                    n_pos[1] = cy[j]
                    n_pos[2] = cz[j]
                    n_found = True

            if not ca_found or not n_found:
                raise ValueError(f"Missing backbone atoms for residue {residue}")
            
            # get opposite GEOM
            chain_curr = chain_ids[atm_start]
            if chain_curr == 0:
                com_opposite[0] = geom_b[0]
                com_opposite[1] = geom_b[1]
                com_opposite[2] = geom_b[2]
            else:
                com_opposite[0] = geom_a[0]
                com_opposite[1] = geom_a[1]
                com_opposite[2] = geom_a[2]
            
            # construct the frame
            build_residue_frame(ca_pos, n_pos, com_opposite, rotation_matrix)
            
            # boundaries
            for i in range(3):
                global_min[i] = min(global_min[i], 
                                   ca_pos[i] - grid_size * voxel_size - max_radius)
                global_max[i] = max(global_max[i], 
                                   ca_pos[i] + grid_size * voxel_size + max_radius)
        
        # filter atoms within bounds
        n_filtered = 0
        for i in range(n_atoms):
            if (cx[i] >= global_min[0] and cx[i] <= global_max[0] and
                cy[i] >= global_min[1] and cy[i] <= global_max[1] and
                cz[i] >= global_min[2] and cz[i] <= global_max[2]):
                filtered_atoms[n_filtered] = i
                n_filtered += 1
        
        # loop residue
        for ri in range(n_interface_residues):
            residue = interface_residues[ri]
            atm_start = residue_first_atom[residue]
            
            # get CA and N positions
            for i in range(residue_atom_count[residue]):
                j = atm_start + i
                if atom_bbelems[j] == 0:  # CA
                    ca_pos[0] = cx[j]
                    ca_pos[1] = cy[j]
                    ca_pos[2] = cz[j]
                elif atom_bbelems[j] == 1:  # N
                    n_pos[0] = cx[j]
                    n_pos[1] = cy[j]
                    n_pos[2] = cz[j]
            
            # get opposite geom
            chain_curr = chain_ids[atm_start]
            if chain_curr == 0:
                for i in range(3):
                    com_opposite[i] = geom_b[i]
            else:
                for i in range(3):
                    com_opposite[i] = geom_a[i]
            
            # frame
            build_residue_frame(ca_pos, n_pos, com_opposite, rotation_matrix)
            
            # process voxels
            voxel_idx = 0
            for i in range(grid_size):
                for j in range(grid_size):
                    for k in range(grid_size):
                        compute_voxel_center(ca_pos, rotation_matrix, i, j, k,
                                           grid_size, voxel_size, voxel_center)
                        
                        compute_voxel_volumes(filtered_atoms, &cx[0], &cy[0], &cz[0], n_filtered,
                                voxel_center, &atom_elems[0], &chain_ids[0],
                                n_atom_elem, rotation_matrix, voxel_size,
                                volume_buffer)
                        
                        start_idx = voxel_idx * n_features_per_voxel
                        feature_view[ri, start_idx] = i - central_position
                        feature_view[ri, start_idx + 1] = j - central_position
                        feature_view[ri, start_idx + 2] = k - central_position
                        
                        for vi in range(n_volumic_features):
                            feature_view[ri, start_idx + 3 + vi] = volume_buffer[vi]
                        
                        voxel_idx += 1
        
        return feature_array
    
    finally:
        free(volume_buffer)
        free(filtered_atoms)
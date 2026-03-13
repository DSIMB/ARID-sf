from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

# Define the extension
extensions = [
    Extension(
        "distances_v2",
        ["distances_v2.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=[
            "-O3", 
            "-march=native",  
            "-fopenmp",  
            "-ffast-math",  
        ],
        extra_link_args=["-fopenmp"],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    )
    ,
    Extension(
        "voxel_features",
        ["voxel_features.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3", "-march=native", "-ffast-math"],
    )
]

setup(
    name="distances_voxels",
    
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            'language_level': "3",  
            'boundscheck': False,
            'wraparound': False,
            'initializedcheck': False,
            'cdivision': True,
        }
    ),
    zip_safe=False,
)
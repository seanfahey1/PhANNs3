from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import glob
import numpy


# Find all .pyx files in the app directory
pyx_files = glob.glob('utils/*.pyx', recursive=True)

# Generate a list of Extension objects from the pyx files
extensions = [Extension(pyx_file.split('.')[0].replace('/', '.'), [pyx_file]) for pyx_file in pyx_files]

setup(
    packages=find_packages(),
    ext_modules=cythonize(extensions),
    include_dirs=[numpy.get_include()]
)

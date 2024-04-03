import glob

import numpy
from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup

# Find all .pyx files in the app directory
pyx_files = glob.glob("src/utils/*.pyx", recursive=True)

# Generate a list of Extension objects from the pyx files
extensions = [
    Extension(pyx_file.split(".")[0].replace("/", "."), [pyx_file])
    for pyx_file in pyx_files
]

setup(
    packages=find_packages(where="src"),
    package_dir={"": ""},
    ext_modules=cythonize(extensions),
    include_dirs=[numpy.get_include()],
)

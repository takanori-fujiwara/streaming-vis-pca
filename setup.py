import glob
from distutils.core import setup

inc_pca_cpp_so = glob.glob('inc_pca_cpp*.so')[0]

setup(
    name='inc-pca',
    version=0.1,
    packages=[''],
    package_dir={'': '.'},
    package_data={'': [inc_pca_cpp_so]},
    py_modules = ['inc_pca_cpp', 'inc_pca']
)

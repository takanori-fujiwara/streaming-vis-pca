import os
import sys
import glob
import sysconfig
from distutils.core import setup

extension_suffix = sysconfig.get_config_var('EXT_SUFFIX')

# remove existing shared objects, etc.
for removing_extension in ['so', 'exp', 'lib', 'obj', 'pyd', 'dll']:
    for removing_file in glob.glob(f'*.{removing_extension}'):
        try:
            os.remove(removing_file)
        except OSError:
            print("Error while deleting existing compiled files")

if sys.platform.startswith('darwin'):
    if os.system('which brew') > 0:
        print('installing homebrew')
        os.system(
            '/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install.sh)"'
        )
    print('installing python3, eigen, pybind11')
    os.system('brew install pkg-config python3 eigen pybind11')

    ## This part can be used to build with CMake (but for anaconda env, this doesn't work well)
    # print('processing cmake')
    # os.system('rm -f CMakeCache.txt')
    # os.system('cmake .')
    # print('processing make')
    # os.system('make')

    print('building inc_pca')
    os.system(
        f'c++ -O3 -Wall -mtune=native -march=native -shared -std=c++11 -undefined dynamic_lookup -I/usr/local/include/eigen3/ $(python3 -m pybind11 --includes) inc_pca.cpp inc_pca_wrap.cpp -o inc_pca_cpp{extension_suffix}'
    )
elif sys.platform.startswith('linux'):
    print('installing pybind11')
    os.system('pip3 install pybind11')
    print('building inc_pca')
    os.system(
        f'c++ -O3 -Wall -mtune=native -march=native -shared -std=c++11 -I/usr/include/eigen3/ -fPIC `python3 -m pybind11 --includes` inc_pca.cpp inc_pca_wrap.cpp -o inc_pca_cpp{extension_suffix}'
    )
elif sys.platform.startswith('win'):
    print('installing pybind11 requests')
    os.system('pip3 install pybind11 requests')

    print('downloading eigen')
    import requests
    import zipfile

    eigen_ver = '3.4.0'
    eigen_name = f'eigen-{eigen_ver}'
    eigen_zip = f'{eigen_name}.zip'
    url = f'https://gitlab.com/libeigen/eigen/-/archive/{eigen_ver}/{eigen_zip}'
    req = requests.get(url)
    with open(eigen_zip, 'wb') as of:
        of.write(req.content)

    with zipfile.ZipFile(eigen_zip, 'r') as zip_ref:
        zip_ref.extractall()

    print('preparing env info')
    import subprocess
    pybind_includes = subprocess.check_output('python -m pybind11 --includes')
    pybind_includes = pybind_includes.decode()
    pybind_includes = pybind_includes[:-2]  # exclude /r/n
    # add double quotes to handle spaces in file paths
    pybind_includes = ' '.join([
        f'/I"{pybind_include}"'
        for pybind_include in pybind_includes.split('-I')[1:]
    ]).replace(' "', '"')

    pyver = os.path.dirname(
        sys.executable).split('\\')[-1].lower()  # => e.g., python310
    pythonlib_path = os.path.dirname(sys.executable) + f'\\libs\\{pyver}.lib'

    # requires VS C++ compiler (https://aka.ms/vs/17/release/vs_BuildTools.exe)
    # also, use an appropriate command prompt for VS (e.g., x64 instead of x86 if using 62-bit Python3)
    print('building inc_pca')
    os.system(f'cl /c /O2 /std:c11 /EHsc /I./{eigen_name}/ inc_pca.cpp')
    os.system(
        f'cl /c /O2 /std:c11 /EHsc /I./{eigen_name}/ {pybind_includes} inc_pca_wrap.cpp'
    )
    os.system(
        f'link inc_pca.obj inc_pca_wrap.obj "{pythonlib_path}" /DLL /OUT:inc_pca_cpp{extension_suffix}'
    )
    ####
    # cl /c /O2 /std:c11 /EHsc /I./eigen-3.4.0/ inc_pca.cpp
    # cl /c /O2 /std:c11 /EHsc /I./eigen-3.4.0/ /I"C:\\Users\\Takanori Fujiwara\\AppData\\Local\\Programs\\Python\\Python310\\Include" /I"C:\\Users\\Takanori Fujiwara\\AppData\\Local\\Programs\\Python\\Python310\\lib\\site-packages\\pybind11\\include" inc_pca_wrap.cpp
    # link inc_pca.obj inc_pca_wrap.obj "C:\\Users\\Takanori Fujiwara\\AppData\\Local\\Programs\\Python\\Python310\\libs\\python310.lib" /DLL /OUT:inc_pca_cpp.cp310-win_amd64.pyd
else:
    print(
        f'ccPCA only supports macos, linux, windows. Your platform: {sys.platform}'
    )

inc_pca_cpp_so = f'inc_pca_cpp{extension_suffix}'

setup(name='inc-pca',
      version=0.1,
      packages=[''],
      package_dir={'': '.'},
      package_data={'': [inc_pca_cpp_so]},
      install_requires=['numpy'],
      py_modules=['inc_pca_cpp', 'inc_pca'])

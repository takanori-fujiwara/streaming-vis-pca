import sysconfig
from setuptools import setup

extension_suffix = sysconfig.get_config_var("EXT_SUFFIX")
inc_pca_cpp_so = f"inc_pca_cpp{extension_suffix}"

setup(
    name="inc-pca",
    version=0.1,
    packages=[""],
    package_dir={"": "."},
    package_data={"": [inc_pca_cpp_so]},
    install_requires=["numpy"],
    py_modules=["inc_pca_cpp", "inc_pca"],
)

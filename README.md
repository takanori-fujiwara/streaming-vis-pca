## C++ Library and Python Module for Incremental PCA for Streaming Multidimensional Data

Incremental PCA for visualizing streaming multidimensional data from [Fujiwara et al., 2019].

About
-----
* Incremental PCA for visualizing streaming multidimensional data from:    
***An Incremental Dimensionality Reduction Method for Visualizing Streaming Multidimensional Data***    
Takanori Fujiwara, Jia-Kai Chou, Shilpika, Panpan Xu, Liu Ren, and Kwan-Liu Ma   
IEEE Transactions on Visualization and Computer Graphics and IEEE VIS 2019 (InfoVis).
DOI: [10.1109/TVCG.2019.2934433](https://doi.org/10.1109/TVCG.2019.2934433)

* Demonstration videos from the application implemented in the paper above are available [here](https://takanori-fujiwara.github.io/s/inc-dr/index.html). Also, from [the same site](https://takanori-fujiwara.github.io/s/inc-dr/index.html), you can download a source code to reproduce the performance evaluation in the paper.

* Features
  * Fast C++ implementation with Eigen3 of Incremental PCA from [Ross et al., 2008].
    * D. Ross, J. Lim, R. Lin, M. Yang. Incremental Learning for Robust Visual Tracking, International Journal of Computer Vision, Volume 77, Issue 1-3, pp. 125-141, 2008.

  * Mental map preservation with Procrustes transformation.

  * Position estimation for handling new data points' incomplete features.

  * Uncertainty measures for the position estimation.

******

Requirements
-----
* C++11 compiler, Python3, Eigen3, Pybind11, Numpy

* Note: Tested on macOS Mojave and Ubuntu 19.0.4 LTS.

******

Setup
-----
#### Mac OS with Homebrew
* Install libraries

    `brew install python3`

    `brew install eigen`

    `brew install pybind11`

    `pip3 install numpy`

* Build with cmake

    `cmake .`

    `make`

* This generates a shared library, "inc_pca_cpp.xxxx.so" (e.g., inc_pca_cpp.cpython-37m-darwin.so).

* Install the modules with pip3.

    `pip3 install .`

#### Linux (tested on Ubuntu 19.0.4 LTS)
* Install libraries

    `sudo apt update`

    `sudo apt install libeigen3-dev`

    `sudo apt install python3-pip python3-dev`

    `pip3 install pybind11`

    `pip3 install numpy`

*  Compile with:

    ``c++ -O3 -Wall -mtune=native -march=native -shared -std=c++11 -I/usr/include/eigen3/ -fPIC `python3 -m pybind11 --includes` inc_pca.cpp inc_pca_wrap.cpp -o inc_pca_cpp`python3-config --extension-suffix` ``

* This generates a shared library, "inc_pca_cpp.xxxx.so" (e.g., inc_pca_cpp.cpython-37m-x86_64-linux-gnu.so).

* Install the modules with pip3.

    `pip3 install .`

******

Usage
-----
* With Python3
    * Import "inc_pca" from python (`from inc_pca import IncPCA`). See sample.ipynb (jupyter notebook), inc_pca.py, or docs/index.html for detailed usage examples.

* With C++
    * Include inc_pca.hpp in C++ code with inc_pca.cpp.

******

## How to Cite
Please, cite:    
Takanori Fujiwara, Jia-Kai Chou, Shilpika, Panpan Xu, Liu Ren, and Kwan-Liu Ma, "An Incremental Dimensionality Reduction Method for Visualizing Streaming Multidimensional Data".
IEEE Transactions on Visualization and Computer Graphics, 2019.
DOI: [10.1109/TVCG.2019.2934433](https://doi.org/10.1109/TVCG.2019.2934433)

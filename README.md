## C++ Library and Python Module for Incremental PCA for Streaming Multidimensional Data

Incremental PCA for visualizing streaming multidimensional data from [Fujiwara et al., 2019].

New
-----
* All major OSs, Mac OS, Linux, Windows, are supported now (2022-04-28).

About
-----
* Incremental PCA for visualizing streaming multidimensional data from:    
***An Incremental Dimensionality Reduction Method for Visualizing Streaming Multidimensional Data***    
Takanori Fujiwara, Jia-Kai Chou, Shilpika, Panpan Xu, Liu Ren, and Kwan-Liu Ma   
IEEE Transactions on Visualization and Computer Graphics and IEEE VIS 2020 (InfoVis 2019).
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

* Note: Tested on macOS Ventura, Ubuntu 22.0.4 LTS, Windows 10.

******

Setup
-----
#### Mac OS with Homebrew

* Make sure if you have C++ compiler. For example,

  `which c++`

  should return the c++ compiler path (e.g., /usr/bin/c++) if it exists. If it does not exist, run:

  `xcode-select --install`

* Install the modules with pip3 (this also installs homebrew, pkg-config, python3, eigen, pybind11, numpy, if they do not exist).

    `pip3 install .`

#### Linux (tested on Ubuntu 22.0.4 LTS)
* Install libraries

    `sudo apt update`

    `sudo apt install libeigen3-dev python3-pip python3-dev`

    * Note: Replace apt commands based on your Linux OS.

* Install the modules with pip3.

    `pip3 install .`

    * Note: If installation does not work, check setup.py and replace c++ commands based on your environment.

#### Windows (tested on Windows 10, <span style="color:#ff8888">requiring MSVC as a C++ compiler</span>)
* Install required compiler and library

    - Install MSVC (Microsoft C++): For example, you can download from https://visualstudio.microsoft.com/downloads/?q=build+tools
      (note: other compilers are not supported, e.g., MinGW)

    - Install Python3 (https://www.python.org/downloads/windows/)

* Install the modules with pip3 in "*Command Prompt for VS*". <span style="color:#ff8888">Note: if you installed 64-bit Python3, use *x64 Native Command Prompt for VS*</span>.

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
IEEE Transactions on Visualization and Computer Graphics, Vol. 26, No. 1, pp. 418-428, 2020.
DOI: [10.1109/TVCG.2019.2934433](https://doi.org/10.1109/TVCG.2019.2934433)

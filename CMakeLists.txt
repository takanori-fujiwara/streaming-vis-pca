cmake_minimum_required(VERSION 3.9)
project(inc_pca)
set(PYBIND11_CPP_STANDARD -std=c++11)
set(CMAKE_CXX_FLAGS "-Wall -O3 -mtune=native -march=native")

find_package(PkgConfig)

set(Python_ADDITIONAL_VERSIONS 3.7)

find_package(PythonInterp)
if(NOT PYTHONINTERP_FOUND)
    message(FATAL_ERROR "Unable to find PythonInterp")
endif()

pkg_check_modules(PYTHON python-3.7)
if(PYTHON_FOUND)
    include_directories(${PYTHON_INCLUDE_DIRS})
    link_directories(${PYTHON_LIBRARY_DIRS})
    message("PYTHON_INCLUDE_DIRS: ${PYTHON_INCLUDE_DIRS}")
    message("PYTHON_LIBRARY_DIRS: ${PYTHON_LIBRARY_DIRS}")
else()
    message(FATAL_ERROR "Unable to find Python")
endif()

find_package(pybind11 REQUIRED)

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

pybind11_add_module(inc_pca_cpp SHARED inc_pca.cpp inc_pca_wrap.cpp)



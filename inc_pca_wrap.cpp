#include "inc_pca.hpp"

#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(inc_pca_cpp, m) {
  m.doc() = "Incremental PCA wrapped with pybind11";
  py::class_<IncPCA>(m, "IncPCA")
      .def(py::init<Eigen::Index const, double const>())
      .def("initialize", &IncPCA::initialize)
      .def("transform", &IncPCA::transform)
      .def("partial_fit", &IncPCA::partialFit)
      .def("get_loadings", &IncPCA::getLoadings)
      .def("geom_trans", &IncPCA::geomTrans)
      .def("pos_est", &IncPCA::posEst)
      .def("get_uncert_v", &IncPCA::getUncertV)
      .def("update_uncert_weight", &IncPCA::updateUncertWeight);
}

#include <pybind11/pybind11.h>

#include "fftwpp/fftwpp.hpp"

PYBIND11_MODULE(pyfftwpp, m) {
  m.doc() = "Python bindings to the fftwpp library";
  m.attr("__author__") = pybind11::cast(fftwpp::author());
  m.attr("__version__") = pybind11::cast(fftwpp::version());
  m.def("return_one", &fftwpp::return_one, "Return one in all circumstances.");
}

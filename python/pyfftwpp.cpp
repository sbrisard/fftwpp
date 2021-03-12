#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "fftwpp/fftwpp.hpp"

PYBIND11_MODULE(pyfftwpp, m) {
  m.doc() = "Python bindings to the fftwpp library";
  m.attr("__author__") = pybind11::cast(__FFTWPP_AUTHOR__);
  m.attr("__version__") = pybind11::cast(__FFTWPP_VERSION__);

  using Plan = fftw::Plan;
  using array = pybind11::array_t<std::complex<double>>;

  pybind11::class_<Plan>(m, "Plan")
      .def(pybind11::init([](array in, array out) {
             pybind11::buffer_info info_in = in.request();
             pybind11::buffer_info info_out = out.request();
             if ((info_in.ndim != 1) || (info_out.ndim != 1)) {
               throw std::runtime_error("expected one dimensional arrays");
             }
             if (info_out.size < info_in.size) {
               throw std::runtime_error(
                   "output array must be larger than input array");
             }
             int size = info_in.size;
             return new Plan{size, in.mutable_data(), out.mutable_data()};
           }))
      .def("execute", &Plan::execute)
      .def("cost", &Plan::cost)
      .def("flops", &Plan::flops)
      .def("__repr__", &Plan::repr);
}

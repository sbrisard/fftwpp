#include <ostream>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "fftwpp/fftwpp.hpp"

using DoubleArray = pybind11::array_t<double>;
using ComplexArray = pybind11::array_t<std::complex<double>>;

template <typename T1, typename T2>
void assert_same_shape(pybind11::array_t<T1> arr1, pybind11::array_t<T2> arr2) {
  pybind11::buffer_info info1 = arr1.request();
  pybind11::buffer_info info2 = arr2.request();
  if (info1.ndim != info2.ndim) {
    std::ostringstream stream;
    stream << "arrays must have same rank: " << info1.ndim
           << " != " << info2.ndim;
    throw std::invalid_argument(stream.str());
  }
  for (int i = 0; i < info1.ndim; i++) {
    if (info2.shape[i] != info1.shape[i]) {
      std::ostringstream stream;
      stream << "arrays must have same shape: (";
      for (int j = 0; j < info1.ndim; j++) stream << info1.shape[j] << ",";
      stream << ") != (";
      for (int j = 0; j < info2.ndim; j++) stream << info2.shape[j] << ",";
      stream << ")";
      throw std::invalid_argument(stream.str());
    }
  }
}

template <typename T>
void assert_c_contiguous(pybind11::array_t<T> array) {
  pybind11::buffer_info info = array.request();
  size_t stride = info.itemsize;
  for (int i = info.ndim - 1; i >= 0; i--) {
    if (info.strides[i] != stride) {
      throw std::invalid_argument("expected C contiguous array");
    }
    stride *= info.shape[i];
  }
}

PYBIND11_MODULE(pyfftwpp, m) {
  m.doc() = "Python bindings to the fftwpp library";
  m.attr("__author__") = pybind11::cast(__FFTWPP_AUTHOR__);
  m.attr("__version__") = pybind11::cast(__FFTWPP_VERSION__);

  pybind11::class_<fftw::PlanFactory>(m, "PlanFactory")
      .def(pybind11::init<>())
      .def("set_estimate", &fftw::PlanFactory::set_estimate)
      .def("unset_estimate", &fftw::PlanFactory::unset_estimate)
      .def("set_measure", &fftw::PlanFactory::set_measure)
      .def("unset_measure", &fftw::PlanFactory::unset_measure)
      .def("set_patient", &fftw::PlanFactory::set_patient)
      .def("unset_patient", &fftw::PlanFactory::unset_patient)
      .def("set_exhaustive", &fftw::PlanFactory::set_exhaustive)
      .def("unset_exhaustive", &fftw::PlanFactory::unset_exhaustive)
      .def("set_wisdom_only", &fftw::PlanFactory::set_wisdom_only)
      .def("unset_wisdom_only", &fftw::PlanFactory::unset_wisdom_only)
      .def("set_destroy_input", &fftw::PlanFactory::set_destroy_input)
      .def("unset_destroy_input", &fftw::PlanFactory::unset_destroy_input)
      .def("set_preserve_input", &fftw::PlanFactory::set_preserve_input)
      .def("unset_preserve_input", &fftw::PlanFactory::unset_preserve_intput)
      .def("set_unaligned", &fftw::PlanFactory::set_unaligned)
      .def("unset_unaligned", &fftw::PlanFactory::unset_unaligned)
      .def(
          "create_plan",
          [](fftw::PlanFactory& self, int rank, ComplexArray in,
             ComplexArray out, int sign) {
            assert_c_contiguous(in);
            assert_c_contiguous(out);
            assert_same_shape(in, out);
            pybind11::buffer_info info = in.request();
            if (rank > info.ndim) {
              std::ostringstream stream;
              stream << "rank must be lower than ndim: " << rank << " > "
                     << info.ndim;
              throw std::invalid_argument(stream.str());
            }
            std::vector<int> shape(info.shape.size());
            for (auto i = 0; i < info.ndim; i++) {
              shape[i] = info.shape[i];
            }
            return self.create_plan(rank, shape, in.mutable_data(),
                                    out.mutable_data(), sign);
          },
          "", pybind11::arg("rank"), pybind11::arg("in"), pybind11::arg("out"),
          pybind11::arg("sign") = -1)
      .def_property_readonly("flags", &fftw::PlanFactory::get_flags);

  pybind11::class_<fftw::Plan>(m, "Plan")
      .def("execute", &fftw::Plan::execute)
      .def("cost", &fftw::Plan::cost)
      .def("flops", &fftw::Plan::flops)
      .def("__repr__", &fftw::Plan::repr);
}
#include <ostream>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "fftwpp/fftwpp.hpp"

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

  pybind11::enum_<fftw::PlannerFlag::PlannerFlag_>(m, "PlannerFlag")
      .value("estimate", fftw::PlannerFlag::estimate)
      .value("measure", fftw::PlannerFlag::measure)
      .value("patient", fftw::PlannerFlag::patient)
      .value("exhaustive", fftw::PlannerFlag::exhaustive)
      .value("wisdom_only", fftw::PlannerFlag::wisdom_only)
      .value("destroy_input", fftw::PlannerFlag::destroy_input)
      .value("preserve_input", fftw::PlannerFlag::preserve_input)
      .value("unaligned", fftw::PlannerFlag::unaligned);

  using Plan = fftw::Plan;
  using array = pybind11::array_t<std::complex<double>>;

  pybind11::class_<Plan>(m, "Plan")
      .def(pybind11::init(
          [](int rank, array in, array out, int sign, unsigned flags) {
            assert_c_contiguous(in);
            assert_c_contiguous(out);
            assert_same_shape(in, out);
            if ((sign != -1) && (sign != 1)) {
              throw std::invalid_argument("sign must be -1 or +1");
            }
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
            return new Plan{rank, shape, in.mutable_data(), out.mutable_data(),
                            sign, flags};
          }))
      .def("execute", &Plan::execute)
      .def("cost", &Plan::cost)
      .def("flops", &Plan::flops)
      .def("__repr__", &Plan::repr);
}

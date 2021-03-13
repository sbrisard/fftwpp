#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "fftwpp/fftwpp.hpp"

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
      .def(pybind11::init([](array in, array out, int sign, unsigned flags) {
        pybind11::buffer_info info_in = in.request();
        pybind11::buffer_info info_out = out.request();
        if (info_in.ndim != info_out.ndim) {
          throw std::invalid_argument(
              "input and output arrays must have same rank");
        }
        for (pybind11::ssize_t i = 0; i < info_in.ndim; i++) {
          if (info_out.shape[i] != info_in.shape[i]) {
            throw std::invalid_argument(
                "input and output arrays must have same shape");
          }
        }
        assert_c_contiguous(in);
        assert_c_contiguous(out);
        if ((sign != -1) && (sign != 1)) {
          throw std::invalid_argument("sign must be -1 or +1");
        }
        std::vector<int> shape(info_in.shape.size());
        for (auto i = 0; i < info_in.ndim; i++) {
          shape[i] = info_in.shape[i];
        }
        return new Plan{shape, in.mutable_data(), out.mutable_data(), sign,
                        flags};
      }))
      .def("execute", &Plan::execute)
      .def("cost", &Plan::cost)
      .def("flops", &Plan::flops)
      .def("__repr__", &Plan::repr);
}

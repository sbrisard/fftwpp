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

template <typename InputType, typename OutputType>
void create_bindings_for_plan_factory(pybind11::module m,
                                      char* const class_name) {
  using Factory = fftw::PlanFactory<InputType, OutputType>;
  using InputArray = pybind11::array_t<InputType>;
  using OutputArray = pybind11::array_t<OutputType>;
  pybind11::class_<Factory>(m, class_name)
      .def(pybind11::init<>())
      .def("set_forward", &Factory::set_forward)
      .def("set_backward", &Factory::set_backward)
      .def("set_estimate", &Factory::set_estimate)
      .def("unset_estimate", &Factory::unset_estimate)
      .def("set_measure", &Factory::set_measure)
      .def("unset_measure", &Factory::unset_measure)
      .def("set_patient", &Factory::set_patient)
      .def("unset_patient", &Factory::unset_patient)
      .def("set_exhaustive", &Factory::set_exhaustive)
      .def("unset_exhaustive", &Factory::unset_exhaustive)
      .def("set_wisdom_only", &Factory::set_wisdom_only)
      .def("unset_wisdom_only", &Factory::unset_wisdom_only)
      .def("set_destroy_input", &Factory::set_destroy_input)
      .def("unset_destroy_input", &Factory::unset_destroy_input)
      .def("set_preserve_input", &Factory::set_preserve_input)
      .def("unset_preserve_input", &Factory::unset_preserve_intput)
      .def("set_unaligned", &Factory::set_unaligned)
      .def("unset_unaligned", &Factory::unset_unaligned)
      .def("create_plan",
           [](Factory& self, int rank, InputArray in, OutputArray out) {
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
             auto p = self.create_plan(rank, shape, in.mutable_data(),
                                       out.mutable_data());
             return new fftw::Plan{p};
           })
      .def_property_readonly("flags", &Factory::get_flags);
}

template <typename InputType, typename OutputType>
void create_bindings_for_plan(pybind11::module m, char* const class_name) {
  using Plan = fftw::Plan;
  using InputArray = pybind11::array_t<InputType>;
  using OutputArray = pybind11::array_t<OutputType>;

  pybind11::class_<Plan>(m, class_name)
      //.def(pybind11::init())
      .def("execute", &Plan::execute)
      .def("cost", &Plan::cost)
      .def("flops", &Plan::flops)
      .def("__repr__", &Plan::repr);
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

  create_bindings_for_plan_factory<std::complex<double>, std::complex<double>>(
      m, "PlanFactory_c128_c128");
  create_bindings_for_plan_factory<double, std::complex<double>>(
      m, "PlanFactory_f64_c128");
//  create_bindings_for_plan_factory<std::complex<double>, double>(
//      m, "PlanFactory_c128_f64");

  create_bindings_for_plan<std::complex<double>, std::complex<double>>(
      m, "Plan_c128_c128");
  create_bindings_for_plan<double, std::complex<double>>(m, "Plan_f64_c128");
}

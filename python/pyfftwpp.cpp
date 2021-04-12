#include <ostream>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "fftwpp/fftwpp.hpp"

using DoubleArray = pybind11::array_t<double>;
using ComplexArray = pybind11::array_t<std::complex<double>>;

void assert_rank_lower_than_ndim(size_t rank, size_t ndim) {
  if (rank > ndim) {
    std::ostringstream stream;
    stream << "rank must be lower than ndim: " << rank << " > " << ndim;
    throw std::invalid_argument(stream.str());
  }
}

template <typename T1, typename T2>
void assert_same_rank(pybind11::array_t<T1> arr1, pybind11::array_t<T2> arr2) {
  if (arr1.ndim() != arr2.ndim()) {
    std::ostringstream stream;
    stream << "arrays must have same rank: " << arr1.ndim()
           << " != " << arr2.ndim();
    throw std::invalid_argument(stream.str());
  }
}

template <typename T1, typename T2>
void assert_same_shape(pybind11::array_t<T1> arr1, pybind11::array_t<T2> arr2) {
  assert_same_rank(arr1, arr2);
  for (int i = 0; i < arr1.ndim(); i++) {
    if (arr2.shape(i) != arr1.shape(i)) {
      std::ostringstream stream;
      stream << "arrays do not have same dimension along axis " << i << ": "
             << arr1.shape(i) << " != " << arr2.shape(i);
      throw std::invalid_argument(stream.str());
    }
  }
}

void assert_compatible_shapes(int rank, DoubleArray real,
                              ComplexArray complex) {
  assert_same_rank(real, complex);
  for (int i = 0; i < real.ndim(); i++) {
    auto actual = complex.shape(i);
    auto expected = real.shape(i);
    if (i == rank - 1) expected = expected / 2 + 1;
    if (actual != expected) {
      std::ostringstream stream;
      stream << "real array has invalid dimension along axis " << i
             << ": expected(" << expected << ") !=  actual(" << actual << ")";
      throw std::invalid_argument(stream.str());
    }
  }
}

template <typename T>
void assert_c_contiguous(pybind11::array_t<T> array) {
  pybind11::buffer_info info = array.request();
  size_t stride = info.itemsize;
  for (auto i = info.ndim - 1; i >= 0; i--) {
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

  using PlanFactory = fftwpp::PlanFactory;

  pybind11::class_<PlanFactory>(m, "PlanFactory")
      .def(pybind11::init<>())
      .def("set_estimate", &PlanFactory::set_estimate)
      .def("unset_estimate", &PlanFactory::unset_estimate)
      .def("set_measure", &PlanFactory::set_measure)
      .def("unset_measure", &PlanFactory::unset_measure)
      .def("set_patient", &PlanFactory::set_patient)
      .def("unset_patient", &PlanFactory::unset_patient)
      .def("set_exhaustive", &PlanFactory::set_exhaustive)
      .def("unset_exhaustive", &PlanFactory::unset_exhaustive)
      .def("set_wisdom_only", &PlanFactory::set_wisdom_only)
      .def("unset_wisdom_only", &PlanFactory::unset_wisdom_only)
      .def("set_destroy_input", &PlanFactory::set_destroy_input)
      .def("unset_destroy_input", &PlanFactory::unset_destroy_input)
      .def("set_preserve_input", &PlanFactory::set_preserve_input)
      .def("unset_preserve_input", &PlanFactory::unset_preserve_input)
      .def("set_unaligned", &PlanFactory::set_unaligned)
      .def("unset_unaligned", &PlanFactory::unset_unaligned)
      .def(
          "create_plan",
          [](PlanFactory& self, size_t rank, ComplexArray in, ComplexArray out,
             int sign) {
            assert_c_contiguous(in);
            assert_c_contiguous(out);
            assert_same_shape(in, out);
            auto info = in.request();
            assert_rank_lower_than_ndim(rank, info.ndim);
            auto shape =
                std::vector<size_t>{info.shape.cbegin(), info.shape.cend()};
            return self.create_plan(rank, shape, in.mutable_data(),
                                    out.mutable_data(), sign);
          },
          "", pybind11::arg("rank"), pybind11::arg("in"), pybind11::arg("out"),
          pybind11::arg("sign") = -1)
      .def(
          "create_plan",
          [](PlanFactory& self, size_t rank, DoubleArray in, ComplexArray out,
             int sign) {
            assert_c_contiguous(in);
            assert_c_contiguous(out);
            assert_compatible_shapes(rank, in, out);
            auto info = in.request();
            assert_rank_lower_than_ndim(rank, info.ndim);
            auto shape =
                std::vector<size_t>{info.shape.cbegin(), info.shape.cend()};
            return self.create_plan(rank, shape, in.mutable_data(),
                                    out.mutable_data(), sign);
          },
          "", pybind11::arg("rank"), pybind11::arg("in"), pybind11::arg("out"),
          pybind11::arg("sign") = -1)
      .def(
          "create_plan",
          [](PlanFactory& self, size_t rank, ComplexArray in, DoubleArray out,
             int sign) {
            assert_c_contiguous(in);
            assert_c_contiguous(out);
            assert_compatible_shapes(rank, out, in);
            auto info = out.request();
            assert_rank_lower_than_ndim(rank, info.ndim);
            auto shape =
                std::vector<size_t>{info.shape.cbegin(), info.shape.cend()};
            return self.create_plan(rank, shape, in.mutable_data(),
                                    out.mutable_data(), sign);
          },
          "", pybind11::arg("rank"), pybind11::arg("in"), pybind11::arg("out"),
          pybind11::arg("sign") = -1)
      .def_property_readonly("flags", &PlanFactory::get_flags);

  using Plan = fftwpp::Plan;
  pybind11::class_<Plan>(m, "Plan")
      .def("execute", &Plan::execute)
      .def("cost", &Plan::cost)
      .def("flops", &Plan::flops)
      .def("__repr__", &Plan::repr);

  m.def("init_threads", &fftwpp::init_threads);
  m.def("plan_with_nthreads", &fftwpp::plan_with_nthreads);
}
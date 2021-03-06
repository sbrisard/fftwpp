#include <ostream>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <fftwpp/fftwpp.hpp>

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
  m.doc() =
#include "docstrings/pyfftwpp.txt"
      ;
  pybind11::dict metadata;
  metadata["author"] = pybind11::cast(fftwpp::metadata::author);
  metadata["description"] = pybind11::cast(fftwpp::metadata::description);
  metadata["author_email"] = pybind11::cast(fftwpp::metadata::author_email);
  metadata["license"] = pybind11::cast(fftwpp::metadata::license);
  metadata["name"] = pybind11::cast(fftwpp::metadata::name);
  metadata["url"] = pybind11::cast(fftwpp::metadata::url);
  metadata["version"] = pybind11::cast(fftwpp::metadata::version);
  metadata["year"] = pybind11::cast(fftwpp::metadata::year);
  m.attr("metadata") = metadata;

  using PlanFactory = fftwpp::PlanFactory;

  pybind11::class_<PlanFactory>(m, "PlanFactory",
#include "docstrings/PlanFactory/PlanFactory.txt"
                                )
      .def(pybind11::init<>())
      .def("set_estimate", &PlanFactory::set_estimate,
           "Set the ``FFTW_ESTIMATE`` flag.")
      .def("unset_estimate", &PlanFactory::unset_estimate,
           "Unset the ``FFTW_ESTIMATE`` flag.")
      .def("set_measure", &PlanFactory::set_measure,
           "Set the ``FFTW_MEASURE`` flag.")
      .def("unset_measure", &PlanFactory::unset_measure,
           "Unset the ``FFTW_MEASURE`` flag.")
      .def("set_patient", &PlanFactory::set_patient,
           "Set the ``FFTW_PATIENT`` flag.")
      .def("unset_patient", &PlanFactory::unset_patient,
           "Unset the ``FFTW_PATIENT`` flag.")
      .def("set_exhaustive", &PlanFactory::set_exhaustive,
           "Set the ``FFTW_EXHAUSTIVE`` flag.")
      .def("unset_exhaustive", &PlanFactory::unset_exhaustive,
           "Unset the ``FFTW_EXHAUSTIVE`` flag.")
      .def("set_wisdom_only", &PlanFactory::set_wisdom_only,
           "Set the ``FFTW_WISDOM_ONLY`` flag.")
      .def("unset_wisdom_only", &PlanFactory::unset_wisdom_only,
           "Unset the ``FFTW_WISDOM_ONLY`` flag.")
      .def("set_destroy_input", &PlanFactory::set_destroy_input,
           "Set the ``FFTW_DESTROY_INPUT`` flag.")
      .def("unset_destroy_input", &PlanFactory::unset_destroy_input,
           "Unset the ``FFTW_DESTROY_INPUT`` flag.")
      .def("set_preserve_input", &PlanFactory::set_preserve_input,
           "Set the ``FFTW_PRESERVE_INPUT`` flag.")
      .def("unset_preserve_input", &PlanFactory::unset_preserve_input,
           "Unset the ``FFTW_PRESERVE_INPUT`` flag.")
      .def("set_unaligned", &PlanFactory::set_unaligned,
           "Set the ``FFTW_UNALIGNED`` flag.")
      .def("unset_unaligned", &PlanFactory::unset_unaligned,
           "Unset the ``FFTW_UNALIGNED`` flag.")
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
#include "docstrings/PlanFactory/create_plan-c2c.txt"
          , pybind11::arg("rank"), pybind11::arg("in"), pybind11::arg("out"),
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
#include "docstrings/PlanFactory/create_plan-r2c.txt"
          , pybind11::arg("rank"), pybind11::arg("in"), pybind11::arg("out"),
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
#include "docstrings/PlanFactory/create_plan-c2r.txt"
          , pybind11::arg("rank"), pybind11::arg("in"), pybind11::arg("out"),
          pybind11::arg("sign") = -1)
      .def_property_readonly("flags", &PlanFactory::get_flags,
#include "docstrings/PlanFactory/flags.txt"
      );

  using Plan = fftwpp::Plan;
  pybind11::class_<Plan>(m, "Plan",
#include "docstrings/Plan/Plan.txt"
                         )
      .def("execute", &Plan::execute,
           "Call ``fftw_execute()`` for the wrapped ``fftw_plan``.")
      .def("cost", &Plan::cost,
           "Call ``fftw_cost()`` for the wrapped ``fftw_plan``.")
      .def("flops", &Plan::flops,
#include "docstrings/Plan/flops.txt"
           )
      .def("__repr__", &Plan::repr);

#ifdef _OPENMP
  m.def("init_threads", &fftwpp::init_threads,
#include "docstrings/init_threads.txt"
  );
  m.def("plan_with_nthreads", &fftwpp::plan_with_nthreads,
        "Set the number of threads to be used by all subsequently created "
        "plans.",
        pybind11::arg("nthreads"));
#endif
}
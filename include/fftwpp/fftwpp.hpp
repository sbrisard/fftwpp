#pragma once

#include <fftw/fftw3.h>
#include <complex>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace fftw {
struct PlannerFlag {
  enum PlannerFlag_ : unsigned {
    estimate = FFTW_ESTIMATE,
    measure = FFTW_MEASURE,
    patient = FFTW_PATIENT,
    exhaustive = FFTW_EXHAUSTIVE,
    wisdom_only = FFTW_WISDOM_ONLY,
    destroy_input = FFTW_DESTROY_INPUT,
    preserve_input = FFTW_PRESERVE_INPUT,
    unaligned = FFTW_UNALIGNED
  };
};

template <typename T>
class Plan {
 public:
  Plan(fftw_plan const p) : p{p} {}

  Plan(int rank, std::vector<int> const &shape, T *in, T *out, int sign,
       unsigned flags)
      : p(create_plan(rank, shape, in, out, sign, flags)) {}

  Plan(const Plan &) = delete;
  Plan &operator=(const Plan &) = delete;
  Plan(Plan &&) = delete;
  Plan &operator=(Plan &&) = delete;

  ~Plan() { fftw_destroy_plan(p); }

  void execute() const { fftw_execute(p); }

  double cost() const { return fftw_cost(p); }

  std::tuple<double, double, double> flops() const {
    double add, mul, fma;
    fftw_flops(p, &add, &mul, &fma);
    return std::make_tuple(add, mul, fma);
  }

  std::string repr() const {
    char *c_str = fftw_sprint_plan(p);
    std::string cpp_str{c_str};
    // TODO: this leads to the computer crashing.
    // std::free(c_str);
    return cpp_str;
  }

 private:
  fftw_plan p;
};

template <typename T>
fftw_plan create_plan(int rank, std::vector<int> const &shape, T *in, T *out,
                      int sign, unsigned flags);

template <>
fftw_plan create_plan<std::complex<double>>(int rank,
                                            std::vector<int> const &shape,
                                            std::complex<double> *in,
                                            std::complex<double> *out, int sign,
                                            unsigned flags) {
  auto ndim = shape.size();
  int stride = 1;
  for (int i = rank; i < ndim; i++) stride *= shape[i];
  return fftw_plan_many_dft(rank, shape.data(), stride,
                            reinterpret_cast<fftw_complex *>(in), nullptr,
                            stride, 1, reinterpret_cast<fftw_complex *>(out),
                            nullptr, stride, 1, sign, flags);
}
}  // namespace fftw

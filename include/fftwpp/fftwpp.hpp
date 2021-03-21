#pragma once

#include <fftw/fftw3.h>
#include <complex>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace fftw {

class Plan {
 public:
  Plan(fftw_plan const p) : p{p} {}

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

class PlanFactory {
 private:
  unsigned flags = 0;

  PlanFactory &set_flag(unsigned flag) {
    flags |= flag;
    return *this;
  }

  PlanFactory &unset_flag(unsigned flag) {
    flags ^= flag;
    return *this;
  }

 public:
  // FIXME: return a fftw_plan rather than a fftw::Plan because I don't know
  // how to call new in the python bindings.
  fftw_plan create_plan(int rank, std::vector<int> const &shape,
                        std::complex<double> *in, std::complex<double> *out,
                        int sign) {
    if ((sign != -1) && (sign != 1)) {
      throw std::invalid_argument("sign must be -1 or +1");
    }
    auto ndim = shape.size();
    int stride = 1;
    for (int i = rank; i < ndim; i++) stride *= shape[i];
    return fftw_plan_many_dft(rank, shape.data(), stride,
                              reinterpret_cast<fftw_complex *>(in), nullptr,
                              stride, 1, reinterpret_cast<fftw_complex *>(out),
                              nullptr, stride, 1, sign, flags);
  }

  fftw_plan create_plan(int rank, std::vector<int> const &shape, double *in,
                        std::complex<double> *out, int sign) {
    if (sign != -1) {
      throw std::invalid_argument("sign must be -1");
    }
    auto ndim = shape.size();
    int stride = 1;
    for (int i = rank; i < ndim; i++) stride *= shape[i];
    return fftw_plan_many_dft_r2c(
        rank, shape.data(), stride, in, nullptr, stride, 1,
        reinterpret_cast<fftw_complex *>(out), nullptr, stride, 1, flags);
  }

  unsigned get_flags() { return flags; }

  PlanFactory &set_estimate() { return set_flag(FFTW_ESTIMATE); }

  PlanFactory &unset_estimate() { return unset_flag(FFTW_ESTIMATE); }

  PlanFactory &set_measure() { return set_flag(FFTW_MEASURE); }

  PlanFactory &unset_measure() { return unset_flag(FFTW_MEASURE); }

  PlanFactory &set_patient() { return set_flag(FFTW_PATIENT); }

  PlanFactory &unset_patient() { return unset_flag(FFTW_PATIENT); }

  PlanFactory &set_exhaustive() { return set_flag(FFTW_EXHAUSTIVE); }

  PlanFactory &unset_exhaustive() { return unset_flag(FFTW_EXHAUSTIVE); }

  PlanFactory &set_wisdom_only() { return set_flag(FFTW_WISDOM_ONLY); }

  PlanFactory &unset_wisdom_only() { return unset_flag(FFTW_WISDOM_ONLY); }

  PlanFactory &set_destroy_input() { return set_flag(FFTW_DESTROY_INPUT); }

  PlanFactory &unset_destroy_input() { return unset_flag(FFTW_DESTROY_INPUT); }

  PlanFactory &set_preserve_input() { return set_flag(FFTW_PRESERVE_INPUT); }

  PlanFactory &unset_preserve_intput() {
    return unset_flag(FFTW_PRESERVE_INPUT);
  }

  PlanFactory &set_unaligned() { return set_flag(FFTW_UNALIGNED); }

  PlanFactory &unset_unaligned() { return unset_flag(FFTW_UNALIGNED); }
};

}  // namespace fftw

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

template <typename InputType, typename OutputType>
class PlanFactory {
 public:
  unsigned get_flags() {
    return flags;
  }

  PlanFactory<InputType, OutputType> &set_forward() {
    sign = 1;
    return *this;
  }

  PlanFactory<InputType, OutputType> &set_backward() {
    sign = -1;
    return *this;
  }

  PlanFactory<InputType, OutputType> &set_estimate() {
    return set_flag(FFTW_ESTIMATE);
  }

  PlanFactory<InputType, OutputType> &unset_estimate() {
    return unset_flag(FFTW_ESTIMATE);
  }

  PlanFactory<InputType, OutputType> &set_measure() {
    return set_flag(FFTW_MEASURE);
  }

  PlanFactory<InputType, OutputType> &unset_measure() {
    return unset_flag(FFTW_MEASURE);
  }

  PlanFactory<InputType, OutputType> &set_patient() {
    return set_flag(FFTW_PATIENT);
  }

  PlanFactory<InputType, OutputType> &unset_patient() {
    return unset_flag(FFTW_PATIENT);
  }

  PlanFactory<InputType, OutputType> &set_exhaustive() {
    return set_flag(FFTW_EXHAUSTIVE);
  }

  PlanFactory<InputType, OutputType> &unset_exhaustive() {
    return unset_flag(FFTW_EXHAUSTIVE);
  }

  PlanFactory<InputType, OutputType> &set_wisdom_only() {
    return set_flag(FFTW_WISDOM_ONLY);
  }

  PlanFactory<InputType, OutputType> &unset_wisdom_only() {
    return unset_flag(FFTW_WISDOM_ONLY);
  }

  PlanFactory<InputType, OutputType> &set_destroy_input() {
    return set_flag(FFTW_DESTROY_INPUT);
  }

  PlanFactory<InputType, OutputType> &unset_destroy_input() {
    return unset_flag(FFTW_DESTROY_INPUT);
  }

  PlanFactory<InputType, OutputType> &set_preserve_input() {
    return set_flag(FFTW_PRESERVE_INPUT);
  }

  PlanFactory<InputType, OutputType> &unset_preserve_intput() {
    return unset_flag(FFTW_PRESERVE_INPUT);
  }

  PlanFactory<InputType, OutputType> &set_unaligned() {
    return set_flag(FFTW_UNALIGNED);
  }

  PlanFactory<InputType, OutputType> &unset_unaligned() {
    return unset_flag(FFTW_UNALIGNED);
  }

 private:
  int sign = -1;
  unsigned flags = 0;

  PlanFactory<InputType, OutputType> &set_flag(unsigned flag) {
    flags |= flag;
    return *this;
  }

  PlanFactory<InputType, OutputType> &unset_flag(unsigned flag) {
    flags ^= flag;
    return *this;
  }
};

template <typename InputType, typename OutputType>
class Plan {
 public:
  Plan(fftw_plan const p) : p{p} {}

  Plan(int rank, std::vector<int> const &shape, InputType *in, OutputType *out,
       int sign, unsigned flags)
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

template <typename InputType, typename OutputType>
fftw_plan create_plan(int rank, std::vector<int> const &shape, InputType *in,
                      OutputType *out, int sign, unsigned flags);

template <>
fftw_plan create_plan<std::complex<double>, std::complex<double>>(
    int rank, std::vector<int> const &shape, std::complex<double> *in,
    std::complex<double> *out, int sign, unsigned flags) {
  auto ndim = shape.size();
  int stride = 1;
  for (int i = rank; i < ndim; i++) stride *= shape[i];
  return fftw_plan_many_dft(rank, shape.data(), stride,
                            reinterpret_cast<fftw_complex *>(in), nullptr,
                            stride, 1, reinterpret_cast<fftw_complex *>(out),
                            nullptr, stride, 1, sign, flags);
}

template <>
fftw_plan create_plan<double, std::complex<double>>(
    int rank, std::vector<int> const &shape, double *in,
    std::complex<double> *out, int sign, unsigned flags) {
  if (sign == -1) {
    throw std::invalid_argument("sign must be -1");
  }
  auto ndim = shape.size();
  int stride = 1;
  for (int i = rank; i < ndim; i++) stride *= shape[i];
  return fftw_plan_many_dft_r2c(rank, shape.data(), stride, in, nullptr, stride,
                                1, reinterpret_cast<fftw_complex *>(out),
                                nullptr, stride, 1, flags);
}

}  // namespace fftw

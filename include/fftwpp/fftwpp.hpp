/**
 * C++ bindings to the [FFTW](http://fftw.org/) library.
 *
 * `fftwpp` strives to remain as close as possible to the orginal C library,
 * while being more idiomatic. The user should refer to the documentations of
 * the FFTW library for a more in-depth description.
 */
#pragma once

#include <fftw/fftw3.h>
#include <complex>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace fftw {

/**
 * A wrapper class around `fftw_plan`.
 *
 * The user is referred to section 4.2, *Using Plans*, in the FFTW
 * documentation (http://fftw.org/fftw3_doc/Using-Plans.html#Using-Plans).
 *
 * This class implements the RAII paradigm: the underlying `fftw_plan` is
 * released when the wrapper object falls out of scope.
 *
 * The constructor of this class should in general not be called directly.
 * Rather, use a PlanFactory.
 */
class Plan {
 public:
  /** Create a new instance around the specified `fftw_plan`. */
  Plan(fftw_plan const p) : p{p, [](fftw_plan p) { fftw_destroy_plan(p); }} {}

  /** Call `fftw_execute()` for the wrapped `fftw_plan`. */
  void execute() const { fftw_execute(p.get()); }

  /** Call `fftw_cost()` for the wrapped `fftw_plan`. */
  double cost() const { return fftw_cost(p.get()); }

  /**
   * Call `fftw_flops()` for the wrapped `fftw_plan`.
   *
   * The returned tuple is the number of floating-point additions,
   * multiplications and fused multiply-add operations, in this order.
   */
  std::tuple<double, double, double> flops() const {
    double add, mul, fma;
    fftw_flops(p.get(), &add, &mul, &fma);
    return std::make_tuple(add, mul, fma);
  }

  /** Call `fftw_sprint_plan()` for the wrapped `fftw_plan`. */
  std::string repr() const {
    char *c_str = fftw_sprint_plan(p.get());
    std::string cpp_str{c_str};
    std::free(c_str);
    return cpp_str;
  }

 private:
  std::shared_ptr<struct fftw_plan_s> p;
};

/**
 * This factory class is used to create new instances of Plan.
 *
 * Planner flags (see section 4.3.2, *Planner Flags*, in the FFTW documentation,
 * http://fftw.org/fftw3_doc/Planner-Flags.html#Planner-Flags) are set through
 * `set_XXX()/unset_XXX()` methods.
 *
 * This class exposes a fluent interface, which allows to chain
 * `set_XXX()/unset_XXX()` methods like so
 *
 * @code{.cpp}
 * auto plan = factory.set_estimate().set_preserve_input().create_plan();
 * @endcode
 */
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
  Plan create_plan(int rank, std::vector<int> const &shape,
                   std::complex<double> *in, std::complex<double> *out,
                   int sign) {
    if ((sign != -1) && (sign != 1)) {
      throw std::invalid_argument("sign must be -1 or +1");
    }
    auto ndim = shape.size();
    int stride = 1;
    for (int i = rank; i < ndim; i++) stride *= shape[i];
    fftw_plan p = fftw_plan_many_dft(
        rank, shape.data(), stride, reinterpret_cast<fftw_complex *>(in),
        nullptr, stride, 1, reinterpret_cast<fftw_complex *>(out), nullptr,
        stride, 1, sign, flags);
    return Plan{p};
  }

  Plan create_plan(int rank, std::vector<int> const &shape, double *in,
                   std::complex<double> *out, int sign) {
    if (sign != -1) {
      throw std::invalid_argument("sign must be -1");
    }
    auto ndim = shape.size();
    int stride = 1;
    for (int i = rank; i < ndim; i++) stride *= shape[i];
    fftw_plan p = fftw_plan_many_dft_r2c(
        rank, shape.data(), stride, in, nullptr, stride, 1,
        reinterpret_cast<fftw_complex *>(out), nullptr, stride, 1, flags);
    return Plan{p};
  }

  Plan create_plan(int rank, std::vector<int> const &shape,
                   std::complex<double> *in, double *out, int sign) {
    if (sign != -1) {
      throw std::invalid_argument("sign must be -1");
    }
    auto ndim = shape.size();
    int stride = 1;
    for (int i = rank; i < ndim; i++) stride *= shape[i];
    fftw_plan p = fftw_plan_many_dft_c2r(
        rank, shape.data(), stride, reinterpret_cast<fftw_complex *>(in),
        nullptr, stride, 1, out, nullptr, stride, 1, flags);
    return Plan{p};
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

int init_threads() { return fftw_init_threads(); }

void plan_with_nthreads(int nthreads) { fftw_plan_with_nthreads(nthreads); }

}  // namespace fftw

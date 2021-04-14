/**
 * C++ bindings to the [FFTW](http://fftw.org/) library.
 *
 * `fftwpp` strives to remain as close as possible to the original C library,
 * while being more idiomatic. The user should refer to the documentations of
 * the FFTW library for a more in-depth description.
 */
#pragma once

#include <algorithm>
#include <complex>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include <fftw3.h>

namespace fftwpp {

/**
 * Wrapper class around `fftw_plan`.
 *
 * Refer to section 4.2, *Using Plans*, in the FFTW documentation
 * (http://fftw.org/fftw3_doc/Using-Plans.html#Using-Plans).
 *
 * This class implements the RAII paradigm: the underlying `fftw_plan` is
 * released when the wrapper object falls out of scope.
 *
 * The constructor of this class should in general not be called directly.
 * Rather, use a PlanFactory.
 */
class Plan {
 public:
  /** Create a new wrapper around the specified `fftw_plan`. */
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
    // TODO Check that fftw_free is really the function that needs to be called
    // (see https://github.com/FFTW/fftw3/issues/238)
    fftw_free(c_str);
    return cpp_str;
  }

 private:
  std::shared_ptr<struct fftw_plan_s> p;
};

std::ostream &operator<<(std::ostream &os, const Plan &plan) {
  return os << plan.repr();
}

/**
 * Factory class that is used to create new instances of Plan.
 *
 * Planner flags are set through `set_XXX()/unset_XXX()` methods. The user is
 * referred to section 4.3.2, *Planner Flags*, in the FFTW documentation
 * (http://fftw.org/fftw3_doc/Planner-Flags.html#Planner-Flags), for a full
 * description of the various flags.
 *
 * This class exposes a fluent interface: all `set_XXX()/unset_XXX()` methods
 * return the current object. This allows for chaining, like so
 *
 * @code{.cpp}
 * auto factory = fftwpp::Factory();
 * auto plan = factory.set_estimate().set_preserve_input().create_plan();
 * @endcode
 *
 * Note that if no planner flags are set/unset, Plan instances will be created
 * with `flags` set to `0`.
 */
class PlanFactory {
 private:
  unsigned flags = 0;

  /** Set the specified flag through a bitwise or (`|`). */
  PlanFactory &set_flag(unsigned flag) {
    flags |= flag;
    return *this;
  }

  /** Unset the specified flag though a bitwise xor (`^`). */
  PlanFactory &unset_flag(unsigned flag) {
    flags ^= flag;
    return *this;
  }

 public:
  /**
   * Create a Plan for a complex-to-complex transform.
   *
   * Uses the so-called *advanced interface* that allows to compute multiple
   * transforms at a time. Refer to section 4.4, *Advanced Interface*, in the
   * FFTW documentation
   * (http://fftw.org/fftw3_doc/Advanced-Interface.html#Advanced-Interface), for
   * more details.
   *
   * The input (`in`) and output (`out`) data arrays are stored in row-major
   * order. Both arrays are assumed to have the same `shape`. Transforms are
   * performed along the `rank` *first* axes of the arrays (the *last* axes are
   * merely iterated over).
   *
   * The `sign` parameter is used to ask for a forward (`sign == -1`, default
   * value) or backward (`sign == +1`) transform. Note that backward transforms
   * are *not normalized*.
   */
  Plan create_plan(size_t rank, std::vector<size_t> const &shape,
                   std::complex<double> *in, std::complex<double> *out,
                   int sign) {
    if ((sign != -1) && (sign != 1)) {
      throw std::invalid_argument("sign must be -1 or +1");
    }
    int stride =
        static_cast<int>(std::reduce(shape.cbegin() + rank, shape.cend(),
                                     size_t{1}, std::multiplies<size_t>()));
    std::vector<int> shape_(shape.size());
    std::transform(shape.cbegin(), shape.cend(), shape_.begin(),
                   [](size_t n) { return static_cast<int>(n); });
    fftw_plan p = fftw_plan_many_dft(
        static_cast<int>(rank), shape_.data(), stride,
        reinterpret_cast<fftw_complex *>(in), nullptr, stride, 1,
        reinterpret_cast<fftw_complex *>(out), nullptr, stride, 1, sign, flags);
    return Plan{p};
  }

  // clang-format off
  /**
   * Create a Plan for a real-to-complex transform.
   *
   * Note that the `sign` parameter is meaningless in the present case, and
   * should not be specified.
   *
   * See @link #create_plan(int, std::vector<int> const &, std::complex<double> *, std::complex<double> *, int) complex-to-complex plan creation@endlink.
   */
  // clang-format on
  Plan create_plan(size_t rank, std::vector<size_t> const &shape, double *in,
                   std::complex<double> *out, int sign) {
    if (sign != -1) {
      throw std::invalid_argument("sign must be -1");
    }
    int stride =
        static_cast<int>(std::reduce(shape.cbegin() + rank, shape.cend(),
                                     size_t{1}, std::multiplies<size_t>()));
    std::vector<int> shape_(shape.size());
    std::transform(shape.cbegin(), shape.cend(), shape_.begin(),
                   [](size_t n) { return static_cast<int>(n); });
    fftw_plan p = fftw_plan_many_dft_r2c(
        static_cast<int>(rank), shape_.data(), stride, in, nullptr, stride, 1,
        reinterpret_cast<fftw_complex *>(out), nullptr, stride, 1, flags);
    return Plan{p};
  }

  // clang-format off
  /**
   * Create a Plan for a complex-to-real transform.
   *
   * Note that the `sign` parameter is meaningless in the present case, and
   * should not be specified.
   *
   * See @link #create_plan(int, std::vector<int> const &, std::complex<double> *, std::complex<double> *, int) complex-to-complex plan creation@endlink.
   */
  // clang-format on
  Plan create_plan(size_t rank, std::vector<size_t> const &shape,
                   std::complex<double> *in, double *out, int sign) {
    if (sign != -1) {
      throw std::invalid_argument("sign must be -1");
    }
    int stride =
        static_cast<int>(std::reduce(shape.cbegin() + rank, shape.cend(),
                                     size_t{1}, std::multiplies<size_t>()));
    std::vector<int> shape_(shape.size());
    std::transform(shape.cbegin(), shape.cend(), shape_.begin(),
                   [](size_t n) { return static_cast<int>(n); });
    fftw_plan p =
        fftw_plan_many_dft_c2r(static_cast<int>(rank), shape_.data(), stride,
                               reinterpret_cast<fftw_complex *>(in), nullptr,
                               stride, 1, out, nullptr, stride, 1, flags);
    return Plan{p};
  }

  /**
   * Return the current bitwise or combination of planner flags.
   *
   * These planner flags apply to all subsequently created instances of Plan.
   */
  unsigned get_flags() { return flags; }

  /** Set the `FFTW_ESTIMATE` flag. */
  PlanFactory &set_estimate() { return set_flag(FFTW_ESTIMATE); }

  /** Unset the `FFTW_ESTIMATE` flag. */
  PlanFactory &unset_estimate() { return unset_flag(FFTW_ESTIMATE); }

  /** Set the `FFTW_MEASURE` flag. */
  PlanFactory &set_measure() { return set_flag(FFTW_MEASURE); }

  /** Unset the `FFTW_MEASURE` flag. */
  PlanFactory &unset_measure() { return unset_flag(FFTW_MEASURE); }

  /** Set the `FFTW_PATIENT` flag. */
  PlanFactory &set_patient() { return set_flag(FFTW_PATIENT); }

  /** Unset the `FFTW_PATIENT` flag. */
  PlanFactory &unset_patient() { return unset_flag(FFTW_PATIENT); }

  /** Set the `FFTW_EXHAUSTIVE` flag. */
  PlanFactory &set_exhaustive() { return set_flag(FFTW_EXHAUSTIVE); }

  /** Unset the `FFTW_EXHAUSTIVE` flag. */
  PlanFactory &unset_exhaustive() { return unset_flag(FFTW_EXHAUSTIVE); }

  /** Set the `FFTW_WISDOM_ONLY` flag. */
  PlanFactory &set_wisdom_only() { return set_flag(FFTW_WISDOM_ONLY); }

  /** Unset the `FFTW_WISDOM_ONLY` flag. */
  PlanFactory &unset_wisdom_only() { return unset_flag(FFTW_WISDOM_ONLY); }

  /** Set the `FFTW_DESTROY_INPUT` flag. */
  PlanFactory &set_destroy_input() { return set_flag(FFTW_DESTROY_INPUT); }

  /** Unset the `FFTW_DESTROY_INPUT` flag. */
  PlanFactory &unset_destroy_input() { return unset_flag(FFTW_DESTROY_INPUT); }

  /** Set the `FFTW_PRESERVE_INPUT` flag. */
  PlanFactory &set_preserve_input() { return set_flag(FFTW_PRESERVE_INPUT); }

  /** Unset the `FFTW_PRESERVE_INPUT` flag. */
  PlanFactory &unset_preserve_input() {
    return unset_flag(FFTW_PRESERVE_INPUT);
  }

  /** Set the `FFTW_UNALIGNED` flag. */
  PlanFactory &set_unaligned() { return set_flag(FFTW_UNALIGNED); }

  /** Unset the `FFTW_UNALIGNED` flag. */
  PlanFactory &unset_unaligned() { return unset_flag(FFTW_UNALIGNED); }
};

/**
 * Call `fftw_init_threads()`.
 *
 * This function should be called before calling *any* `FFTW` or `fftwpp`
 * functions.
 */
int init_threads() { return fftw_init_threads(); }

/** Set the number of threads to be used by all subsequently created plans. */
void plan_with_nthreads(int nthreads) { fftw_plan_with_nthreads(nthreads); }

}  // namespace fftwpp

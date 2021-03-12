#pragma once

#include <fftw/fftw3.h>
#include <complex>
#include <cstdlib>
#include <string>

namespace fftw {
class Plan {
 public:
  Plan(int size, std::complex<double> *in, std::complex<double> *out)
      : p{fftw_plan_dft_1d(size, reinterpret_cast<fftw_complex *>(in),
                           reinterpret_cast<fftw_complex *>(out), FFTW_FORWARD,
                           FFTW_ESTIMATE)} {}

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
  fftw_plan const p;
};
}  // namespace fftw

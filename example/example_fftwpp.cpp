#include <complex>
#include <iostream>
#include <numbers>
#include <vector>

#include <fftwpp/fftwpp.hpp>

int main() {
  size_t N = 16;
  double z = 1.5;
  std::vector<std::complex<double>> in(N);
  std::vector<std::complex<double>> out(N);

  auto factory = fftwpp::PlanFactory().set_estimate();
  auto plan = factory.create_plan(1, {N}, in.data(), out.data(), -1);
  std::cout << "Created plan: " << std::endl;
  std::cout << plan << std::endl;
  in[0] = 1.0;
  for (size_t n = 1; n < N; n++) in[n] = z * in[n - 1];
  auto z_pow_N = z * in[N - 1];
  plan.execute();
  for (size_t n = 0; n < N; n++) {
    double theta = 2 * std::numbers::pi * n / (double)N;
    std::complex<double> q{cos(theta), -sin(theta)};
    std::complex<double> exp = (1. - z_pow_N) / (1. - q * z);
    std::cout << "expected = " << exp << ", actual = " << out[n] << std::endl;
  }
}

#include <array>
#include <complex>
#include <functional>
#include <iostream>
#include <numbers>
#include <numeric>
#include <vector>

#include <fftwpp/fftwpp.hpp>

using Real = double;
using Complex = std::complex<Real>;
// end20210418063811

// begin20210418063940
template <size_t DIM>
std::vector<Real> powers_of(std::array<Real, DIM> x, size_t n) {
  std::vector<Real> x_pow(n * DIM);
  std::fill(x_pow.begin(), x_pow.begin() + DIM, Real{1});
  for (size_t k = 1; k < n; k++) {
    std::transform(x.cbegin(), x.cend(), x_pow.cbegin() + (k - 1) * DIM,
                   x_pow.begin() + k * DIM, std::multiplies());
  }
  return x_pow;
}
// end20210418063940

// begin20210418064253
std::vector<Complex> powers_of_unit_complex(Real arg, size_t n) {
  std::vector<Complex> phi_pow(n);
  for (size_t k = 0; k < n; k++) {
    phi_pow[k] = Complex{cos(k * arg), sin(k * arg)};
  }
  return phi_pow;
}
// end20210418064253

// begin20210418064539
template <size_t DIM>
std::vector<Complex> dft_of_powers_of(std::array<Real, DIM> x, size_t M) {
  Complex one{1};
  std::array<Real, DIM> x_pow_M;
  std::transform(x.cbegin(), x.cend(), x_pow_M.begin(),
                 [M](auto x_) { return pow(x_, M); });

  Real arg_phi = -2 * std::numbers::pi_v<Real> / (Real)M;
  auto phi_pow = powers_of_unit_complex(arg_phi, M);

  std::vector<Complex> x_hat(M * DIM);
  for (size_t m = 0; m < M; m++) {
    auto phi_pow_m = phi_pow[m];
    std::transform(x.cbegin(), x.cend(), x_pow_M.cbegin(),
                   x_hat.begin() + m * DIM,
                   [one, phi_pow_m](auto x_, auto x_pow_M_) {
                     return (one - x_pow_M_) / (one - x_ * phi_pow_m);
                   });
  }
  return x_hat;
}
// end20210418064539

// begin20210418065331
template <size_t DIM>
void tutorial(std::array<Real, DIM> x, size_t M, std::array<Real, DIM> y,
              size_t N) {
  constexpr auto one = Real{1};
  using Complex = std::complex<Real>;

  auto x_pow = powers_of(x, M);
  auto y_pow = powers_of(y, N + 1);

  auto x_hat = dft_of_powers_of(x, M);
  auto y_hat = dft_of_powers_of(y, N);
  // end20210418065331

  // begin20210418065501
  std::vector<size_t> shape{M, N, DIM};
  size_t size =
      std::reduce(shape.cbegin(), shape.cend(), size_t{1}, std::multiplies());
  std::vector<Complex> in(size), exp(size), act(size);
  // end20210418065501

  // begin20210418065917
  auto factory = fftwpp::PlanFactory().set_estimate();
  // end20210418065917
  // begin20210418075847
  auto plan = factory.create_plan(2, shape, in.data(), act.data(), -1);
  // end20210418075847
  std::cout << "The following plan was created plan: " << plan << std::endl
            << std::endl;

  // begin20210418080158
  for (size_t m = 0; m < M; m++) {
    auto x1 = x_pow.cbegin() + m * DIM;
    auto x2 = x_pow.cbegin() + (m + 1) * DIM;
    auto x_hat1 = x_hat.cbegin() + m * DIM;
    auto x_hat2 = x_hat.cbegin() + (m + 1) * DIM;
    for (size_t n = 0; n < N; n++) {
      auto y1 = y_pow.cbegin() + n * DIM;
      auto y_hat1 = y_hat.cbegin() + n * DIM;
      size_t index = DIM * (m * N + n);
      std::transform(x1, x2, y1, in.begin() + index, std::multiplies());
      std::transform(x_hat1, x_hat2, y_hat1, exp.begin() + index,
                     std::multiplies());
    }
  }
  // end20210418080158
  // begin20210418080316
  plan.execute();
  // end20210418080316
  //begin20210418202159
  for (size_t i = 0; i < size; i++) {
    std::cout << "expected = " << exp[i] << ", actual = " << act[i]
              << std::endl;
  }
  // end20210418202159
}

// begin20210418080545
int main() {
  constexpr size_t dim = 2;
  std::array<Real, dim> x{0.8, -0.9};
  std::array<Real, dim> y{-1.1, 1.2};

  size_t M = 7;
  size_t N = 8;

  tutorial(x, M, y, N);
  return 0;
}
// end20210418080545

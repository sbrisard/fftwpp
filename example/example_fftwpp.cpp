#include <iostream>
#include "fftwpp/fftwpp.hpp"

int main() {
  std::cout << "version = " << fftwpp::version() << std::endl;
  std::cout << "author = " << fftwpp::author() << std::endl;
  std::cout << "return_one() = " << fftwpp::return_one() << std::endl;
  return 0;
}

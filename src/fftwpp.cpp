#include "fftwpp/fftwpp.hpp"

namespace fftwpp
{
  std::string author() {
    return PROJECT_AUTHOR_QUOTED;
  }

  std::string version() {
    // This preprocessor macro is defined in the CMakeLists.txt file
    return PROJECT_VERSION_QUOTED;
  }

  int return_one() { return 1; }
}  // namespace fftwpp

#pragma once

#if _WIN32
#define DllExport __declspec(dllexport)
#else
#define DllExport
#endif

#include <string>

namespace fftwpp
{
  DllExport std::string author();

  DllExport std::string version();

  DllExport int return_one();
}  // namespace fftwpp

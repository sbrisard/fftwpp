cmake_minimum_required(VERSION 3.13)

set(METADATA_REGEX ".*constexpr[ \t\r\n]*std::string_view[ \t\r\n]*XXX[ \t\r\n]*\{[ \t\r\n]*\"([^\"]*)\"[ \t\r\n]*\};.*")

file(STRINGS include/fftwpp/fftwpp.hpp METADATA_ALL
  REGEX "constexpr[ \t\r\n]*std::string_view[ \t\r\n]*")

string(REPLACE "XXX" "version" VERSION_REGEX ${METADATA_REGEX})
string(REGEX REPLACE ${VERSION_REGEX} "\\1" METADATA_VERSION ${METADATA_ALL})

string(REPLACE "XXX" "description" DESCRIPTION_REGEX ${METADATA_REGEX})
string(REGEX REPLACE ${DESCRIPTION_REGEX} "\\1" METADATA_DESCRIPTION ${METADATA_ALL})

string(REPLACE "XXX" "author" AUTHOR_REGEX ${METADATA_REGEX})
string(REGEX REPLACE ${AUTHOR_REGEX} "\\1" METADATA_AUTHOR ${METADATA_ALL})

string(CONCAT METADATA_DESCRIPTION ${METADATA_DESCRIPTION} " -- " ${METADATA_AUTHOR})

project(
  fftwpp
  VERSION ${METADATA_VERSION}
  DESCRIPTION ${METADATA_AUTHOR}
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
include(CTest)

add_subdirectory(include)
add_subdirectory(python)
#add_subdirectory(tests)

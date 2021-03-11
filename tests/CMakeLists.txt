add_executable(test_fftwpp
  test_fftwpp_main.cpp
  test_fftwpp.cpp
)

target_include_directories(test_fftwpp PUBLIC ../include)

if (MATH_LIBRARY)
    target_link_libraries(test_fftwpp ${MATH_LIBRARY})
endif ()

find_package(Catch2 REQUIRED)
target_link_libraries(test_fftwpp PRIVATE fftwpp Catch2::Catch2)

add_test(NAME test_fftwpp COMMAND test_fftwpp)

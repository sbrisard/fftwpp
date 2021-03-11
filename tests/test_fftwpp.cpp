#define _USE_MATH_DEFINES

#include "catch2/catch.hpp"
#include "fftwpp/fftwpp.hpp"

TEST_CASE("Test case #1") {
  SECTION("Section #1") {
    REQUIRE(fftwpp::author() == "S. Brisard");
    REQUIRE(fftwpp::version() == "0.1");
    REQUIRE(fftwpp::return_one() == 1);
  }
}

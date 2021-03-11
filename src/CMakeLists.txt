find_library(MATH_LIBRARY m)

add_compile_definitions(PROJECT_VERSION_QUOTED=\"${METADATA_VERSION}\")
add_compile_definitions(PROJECT_AUTHOR_QUOTED=\"${METADATA_AUTHOR}\")

add_library(
  ${PROJECT_NAME} SHARED
  fftwpp.cpp
  ../include/${PROJECT_NAME}/fftwpp.hpp)

target_include_directories(
  ${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
  $<INSTALL_INTERFACE:include>)

if (MATH_LIBRARY)
  target_link_libraries(${PROJECT_NAME} ${MATH_LIBRARY})
endif()

install(
  TARGETS ${PROJECT_NAME}
  DESTINATION lib
  EXPORT ${PROJECT_NAME}_targets)

install(
  EXPORT ${PROJECT_NAME}_targets
  FILE ${PROJECT_NAME}-targets.cmake
  DESTINATION lib/cmake/${PROJECT_NAME})

include(CMakePackageConfigHelpers)

configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/config.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config.cmake"
  INSTALL_DESTINATION "lib/cmake/${PROJECT_NAME}"
  NO_SET_AND_CHECK_MACRO
  NO_CHECK_REQUIRED_COMPONENTS_MACRO)

# write_basic_package_version_file(
#   "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config-version.cmake"
#   VERSION "${Tutorial_VERSION_MAJOR}.${Tutorial_VERSION_MINOR}"
#   COMPATIBILITY AnyNewerVersion
# )

install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config.cmake
  DESTINATION lib/cmake/${PROJECT_NAME})

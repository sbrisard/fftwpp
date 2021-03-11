find_package(pybind11 REQUIRED)

add_compile_definitions(
        __FFTWPP_AUTHOR__="\"\"${METADATA_AUTHOR}\"\""
        __FFTWPP_VERSION__="\"\"${METADATA_VERSION}\"\""
)

pybind11_add_module(pyfftwpp pyfftwpp.cpp)
target_link_libraries(pyfftwpp PRIVATE fftwpp)

#configure_file(demo.py demo.py COPYONLY)
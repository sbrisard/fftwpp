mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=C:\opt\fftwpp ..
cmake --build . --config Release
cmake --install . --config Release

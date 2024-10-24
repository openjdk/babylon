git clone https://github.com/oneapi-src/level-zero.git
cd level-zero
mkdir build
cd build
cmake .. -D CMAKE_BUILD_TYPE=Release
cmake --build . --target package
cmake --build . --target install

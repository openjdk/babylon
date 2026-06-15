rm -rf \
   build\
   conf\
   extractions/cmake-build-debug\
   extractions/cuda/src\
   extractions/opencl/src\
   extractions/opengl/src

rm -rf $(find . -type d -name target)

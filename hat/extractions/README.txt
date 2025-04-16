This does not compile the java files extracted, we leave that to hat/bld or maven

Ensure cmake and jextract are in your path then :)

$ cmake -B build
$ cmake --build build --target extract

To remove previously extracted sources (careful with this axe eugene !)

$ rm -rf */src
$ rm -rf build

Then

$ tree

Should yield

├── CMakeLists.txt
├── cuda
│   └── CMakeLists.txt
├── opencl
│   └── CMakeLists.txt
├── opengl
│   └── CMakeLists.txt
└── README.txt

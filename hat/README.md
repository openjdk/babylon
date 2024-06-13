# HAT Project

This is a fairly large project with Java and native (C/C++) artifacts.

To use HAT you will need to clone and build the  `babylon` (JDK23+Babylon) project or of course your fork of the babylon project (say `babylon-myfork`)

[github.com/openjdk/babylon](https://github.com/openjdk/babylon)

The HAT project is in the 'hat' subdir of the babylon project.

We will walk through the initial steps to clone and build babylon and then focus on the steps for building HAT.

If you follow these steps, you should not have to change any of the maven or cmake build files.

We suggest starting with a 'github' dir and a dir where we will need to install jdk22

```
export GITHUB=${HOME}/github
mkdir -p ${GITHUB}
mkdir -p ${HOME}/java
```

We need an existing jdk22 build to build babylon. These steps are slighly different for Mac OSX and Ubuntu

### Ubuntu

```
cd ${HOME}/java
wget https://download.java.net/java/GA/jdk22.0.1/c7ec1332f7bb44aeba2eb341ae18aca4/8/GPL/openjdk-22.0.1_linux-x64_bin.tar.gz
export BOOT_JDK=${HOME}/java/jdk-22.0.1.jdk
```

### Mac OSX Aaarch64
```
cd ${HOME}/java
wget https://download.java.net/java/GA/jdk22.0.1/c7ec1332f7bb44aeba2eb341ae18aca4/8/GPL/openjdk-22.0.1_linux-x64_bin.tar.gz
export BOOT_JDK=${HOME}/java/jdk-22.0.1.jdk/Contents/Home
```

From now on the Mac and Ubuntu steps are the same.

If you have your own fork of the babylon repo use it rather that babylon.git below.

```
cd ${GITHUB}
git clone https://github.com/openjdk/babylon.git
cd ${GITHUB}/babylon
bash configure  --with-boot-jdk=${BOOT_JDK}
make clean
make images
```
If you have never built JDK before you may find that the 'configure' step will suggest packages to install.

I usually just take the suggestions and try running `bash configure` again.

Eventually you will have a successful babylon build.

You now should have

```
github
├── babylon (or babylon-my-fork)
│   ├── build
│   │   └── XXXX-server-release
│   │       ├── jdk
│   │       └── ...
│   ├── hat
│   │   ├── README.md

```
Where XXXX is either linux-x64 or macosx-aarch64 and contains your build of babylon JDK.

## Building HAT

HAT uses maven and cmake.

Maven controls the build but delegates to cmake to build the native code for the various backends.

To build HAT we need to ensure that `JAVA_HOME` is setup to point to the babylon build (that we just created) and that `${JAVA_HOME}/bin` is in your PATH.

The `env.bash` shell script can be sourced (dot included) in your shell to set up these vars.

It should detect the arch type and select the correct relative parent dir and inject that dir in your PATH.

```bash
cd hat
. ./env.bash
echo ${JAVA_HOME}
/Users/ME/github/babylon/hat/../build/macosx-aarch64-server-release/jdk
echo ${PATH}
/Users/ME/github/babylon/hat/../build/macosx-aarch64-server-release/jdk/bin:/usr/local/bin:......
```

Now we should be able to use maven to build, it will do its magic and place all jars and libs in `maven-build` dir

```
cd hat
. ./env.bash
mvn clean  compile jar:jar install
ls maven-build
hat-1.0.jar                     hat-example-heal-1.0.jar        libptx_backend.dylib
hat-backend-cuda-1.0.jar        hat-example-mandel-1.0.jar      libspirv_backend.dylib
hat-backend-mock-1.0.jar        hat-example-squares-1.0.jar     mock_info
hat-backend-opencl-1.0.jar      hat-example-view-1.0.jar        opencl_info
hat-backend-ptx-1.0.jar         hat-example-violajones-1.0.jar  ptx_info
hat-backend-spirv-1.0.jar       libmock_backend.dylib           spirv_info
hat-example-experiments-1.0.jar libopencl_backend.dylib
```

To run an example
```
${JAVA_HOME}/bin/java \
   --enable-preview --enable-native-access=ALL-UNNAMED \
   --class-path maven-build/hat-1.0.jar:maven-build/hat-example-mandel-1.0.jar:maven-build/hat-backend-opencl-1.0.jar \
   --add-exports=java.base/jdk.internal=ALL-UNNAMED \
   -Djava.library.path=maven-build\
   mandel.MandelCompute
```

The `hatrun.bash` script simlifies this somewhat

```
bash hatrun.bash opencl mandel MandelCompute
```

### Intellij and Clion
We can use JetBrains' `intelliJ` and `clion` but care must be taken as these tools
do not play well together, specifically we cannot have `Clion` and `Intellij`
project artifacts rooted under each other or in the same dir.

The `intellij` subdir houses the various `*.iml` module files and the project `.idea` so
just open that dir as an intellij project

Thankfully `clion` uses cmake. So we can reuse the same `hat/backends/CMakeLists.txt` that
maven uses to build the backends.

### Initial Project Layout

```
${BABYLON_JDK}
   └── hat
        ├── maven-build (created by the build)
        │
        ├── intellij
        │    ├── .idea
        │    │    ├── compiler.xml
        │    │    ├── misc.xml
        │    │    ├── modules.xml
        │    │    ├── uiDesigner.xml
        │    │    ├── vcs.xml
        │    │    └── workspace.xml
        │    │
        │    ├── hat.iml
        │    ├── backend_(spirv|mock|cuda|ptx|opencl).iml
        │    └── (mandel|violajones|experiments|heal|view).iml
        │
        ├── hat
        │    ├── pom.xml
        │    └── src
        │         ├── src/main/java
        │         └── src/main/resources
        │
        ├── backends
        │    ├── pom.xml
        │    ├── CMakeLists.txt
        │    └── (opencl|cuda|ptx|mock|shared)
        │          ├── pom.xml
        │          ├── CMakeLists.txt
        │          ├── cpp
        │          ├── include
        │          ├── src/main/java
        │          └── src/main/resources
        └── examples
             ├── pom.xml
             └── (mandel|violajones|squares|heal|view|experiments)
                    ├── pom.xml
                    ├── src/main/java
                    └── src/main/resources
```
As you will note the `intellij` dir is somewhat self contained.  the various `*.iml`
files refer to the source dirs using relative paths.

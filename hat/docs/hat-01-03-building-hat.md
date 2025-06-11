# Building HAT

----

* [Contents](hat-00.md)
* House Keeping
    * [Project Layout](hat-01-01-project-layout.md)
    * [Building Babylon](hat-01-02-building-babylon.md)
    * [Building HAT](hat-01-03-building-hat.md)
* Programming Model
    * [Programming Model](hat-03-programming-model.md)
* Interface Mapping
    * [Interface Mapping Overview](hat-04-01-interface-mapping.md)
    * [Cascade Interface Mapping](hat-04-02-cascade-interface-mapping.md)
* Implementation Detail
    * [Walkthrough Of Accelerator.compute()](hat-accelerator-compute.md)
    * [How we minimize buffer transfers](hat-minimizing-buffer-transfers.md)

---

# Building HAT with Script

We initially used maven and cmake to build hat.  If you feel more comfortable
with maven consider [building with maven and cmake](hat-01-03-building-hat-with-maven.md)
but it is possible that maven support will be removed if the `Script` approach takes off.

## Dependencies

Before we start to build HAT we will need `cmake` and `jextract` installed.

You can download jextract from [here](https://jdk.java.net/jextract/)

Use `sudo apt` on Linux or `brew install`.

```bash
sudo apt install cmake

```

```bash
brew install cmake
```


You will also need a Babylon JDK built (the one we built [here](hat-01-02-building-babylon.md))


## Setting your PATH variable

To build HAT we will need `JAVA_HOME` to point to our prebuilt babylon jdk

I suggest you also create a `JEXTRACT_HOME` var to point to the location where you placed JEXTRACT)

In my case
```
export JEXTRACT_HOME=/Users/me/jextract-22
```

Make sure also that `cmake` in in your PATH

## ./env.bash

Thankfully just sourcing the top level `env.bash` script should then be able to set up your PATH for you.

It should detect the arch type (AARCH64 or X86_46) and
select the correct relative parent dir for your BABYLON_JDK and inject that dir in your PATH.

It should also add jextract to your PATH (based on the value you set above for JEXTRACT_HOME)



```bash
cd hat
export JEXTRACT_HOME=/Users/me/jextract-22
. ./env.bash
echo ${JAVA_HOME}
/Users/me/github/babylon/hat/../build/macosx-aarch64-server-release/jdk
echo ${PATH}
/Users/me/github/babylon/hat/../build/macosx-aarch64-server-release/jdk/bin:/Users/me/jextract-22/bin:/usr/local/bin:......
```

## Building using bld

To build hat artifacts (hat jar + backends and examples)
```bash
java @hat/bld
```

This places build artifacts in the `build` and `stages` dirs

```bash
cd hat
. ./env.bash
java @hat/bld
ls build
hat-1.0.jar                         hat-example-heal-1.0.jar        libptx_backend.dylib
hat-backend-ffi-cuda-1.0.jar        hat-example-mandel-1.0.jar      libspirv_backend.dylib
hat-backend-ffi-mock-1.0.jar        hat-example-squares-1.0.jar     mock_info
hat-backend-ffi-opencl-1.0.jar      hat-example-view-1.0.jar        opencl_info
hat-backend-ffi-ptx-1.0.jar         hat-example-violajones-1.0.jar  ptx_info
hat-backend-ffi-spirv-1.0.jar       libmock_backend.dylib           spirv_info
hat-example-experiments-1.0.jar     libopencl_backend.dylib
ls stage
opencl_jextracted    opengl_jextracted
```

`bld` relies on cmake to build native code for backends, so if cmake finds OpenCL libs/headers, you will see libopencl_backend (.so or .dylib) in the build dir, if cmake finds CUDA you will see libcuda_backend(.so or .dylib)

We have another script called `sanity` which will check all  .md/.java/.cpp/.h for tabs, lines that end with whitespace
or files without appropriate licence headers

This is run using

```
java @hat/sanity
```


## Running an example

To run a HAT example we can run from the artifacts in `build` dir

```bash
${JAVA_HOME}/bin/java \
   --add-modules jdk.incubator.code --enable-preview --enable-native-access=ALL-UNNAMED \
   --class-path build/hat-core-1.0.jar:build/hat-backend-ffi-shared-1.0.jar:build/hat-backend-ffi-opencl-1.0.jar:build/hat-example-mandel-1.0.jar \
   --add-exports=java.base/jdk.internal=ALL-UNNAMED \
   -Djava.library.path=build\
   mandel.Main
```

The `hat/run.java` script can also be used which simply needs the backend
name `ffi-opencl|ffi-java|ffi-cuda|ffi-ptx|ffi-mock` and the package name `mandel`

```bash
java @hat/run ffi-opencl mandel
```

If you pass `headless` as the first arg

```bash
java @hat/run headless ffi-opencl mandel
```

This sets `-Dheadless=true` and passes '--headless' to the example.  Some examples can use this to avoid launching UI.


# More Bld info
`hat/Script.java` is an evolving set of static methods and types required (so far.. ;) )
to be able to build HAT, hat backends and examples via the `bld` script

We rely on java's ability to launch java source directly (without needing to javac first)

* [JEP 458: Launch Multi-File Source-Code Program](https://openjdk.org/jeps/458)
* [JEP 330: Launch Single-File Source-Code Programs](https://openjdk.org/jeps/330)

The `hat/bld.java` script (really java source) can be run like this

```bash
java --add-modules jdk.incubator.code --enable-preview --source 26 hat/bld.java
```

In our case the  magic is under the `hat`subdir

We also have a handy `hat/XXXX` which allows us to avoid specifying common args `--enable-preview --source 26` eash time we launch a script

```
hat
├── hat
|   ├── Script.java
|   ├── sanity      (the args for sanity.java)  "--enable-preview --source 26 sanity"
|   |-- sanity.java (the script)
|   ├── run         (the args for sanity.java)  "--enable-preview --source 26 hatrun"
|   |-- run.java    (the script)
|   ├── bld         (the args for bld.java)      "--enable-preview --source 26 bld"
|   ├── bld.java    (the script)

```

For example
```bash
java @hat/bld
```

Is just a shortcut for
```bash
java --add-modules jdk.incubator.code --enable-preview --source 26 hat/bld.java
```

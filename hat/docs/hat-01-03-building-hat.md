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
   --class-path build/core-1.0.jar:build/hat-backend-ffi-shared-1.0.jar:build/hat-backend-ffi-opencl-1.0.jar:build/hat-example-mandel-1.0.jar \
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


----
### HAT runtime environment variable.

During ffi-backend development we added some useful flags to pass to native code to allow us to trace calls, inject logging, control options.

These should be considered just development flags.

At runtime the ffi-backends all communicate from the java side to the native side via a 32 bit 'config' int.

The Java side class [hat.backend.ffi.Config](https://github.com/openjdk/babylon/blob/code-reflection/hat/backends/ffi/shared/src/main/java/hat/backend/ffi/Config.java)

Is initialized from an environment variable `HAT` at runtime

So for example when we launch
```bash
java @hat/run ffi-opencl heal
```

We can pass config info to `HAT` via either `-DHAT=xxx` or via the `HAT` ENV variable.

So for example to get HAT to dump the text form of the kernel code models.

```
HAT=INFO,SHOW_KERNEL_MODEL java @hat/run ffi-opencl heal
```

Or to show generated opencl or cuda code
```
HAT=INFO,SHOW_CODE java @hat/run ffi-opencl heal
```

This is particularly useful for selecting PLATFORM + DEVICE if you have multiple GPU devices.

Here we select DEVICE 0 on PLATFORM 0 (actually the default)
```
HAT=INFO,PLATFORM:0,DEVICE:0,SHOW_CODE ...
```
or for DEVICE 1 on PLATFORM 1
```
HAT=INFO,PLATFORM:1,DEVICE:1,SHOW_CODE ...
```

No the platform and device id's are 4 bits. This probably works for development, but we will need a more expressive way of capturing this via the accelerator selection.

This just allows us to test without code changes.

To keep he java code and the native code in sync.  The `main` method at the end of
 [hat.backend.ffi.Config](https://github.com/openjdk/babylon/blob/code-reflection/hat/backends/ffi/shared/src/main/java/hat/backend/ffi/Config.java)
is actually used to create the C99 header for Cuda and OpenCL ffi-backends.

So whenever we change the Java config class we should run the main method to generate the header.   This is not really robust, (but proved way better than trying to remember for all backends) but you need to know, if you add move config bits to the Java side.

The Main method uses a variant of the code builders used for C99 style code (CUDA/OpenCL) to generate the config.h header.

This is how we keep the ffi based backends in sync

Some more useful `config bits`

```
HAT=PTX java @hat/run ffi-cuda ....
```
Sends PTX generated by our prototype PTX generator to the backend (for CUDA) rather than C99 code.

```
HAT=INFO java @hat/run ....
```

Will dump all of the `config bits` from the native side.

At present this yields
```
native minimizeCopies 0
native trace 0
native profile 0
native showCode 0
native showKernelModel 0
native showComputeModel 0
native info 1
native traceCopies 0
native traceSkippedCopies 0
native traceEnqueues 0
native traceCalls 0
native showWhy 0
native showState 0
native ptx 0
native interpret 0
```
Generally the flags all represent single bits (except PLATFORM:n and DEVICE:n) and are set using comma separated uppercase+underscore string forms.


So to experiment with `minimizeCopies` and to  `showCode` we set `HAT=MINIMIZE_COPIES,SHOW_CODE`

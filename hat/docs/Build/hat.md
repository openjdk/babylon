# Building HAT

[Back to Index ../](../index.md)

We use `maven` and `cmake` to build HAT.

Maven controls the build but delegates to cmake for native artifacts (such as
`ffi-backends`), we also use `cmake` to provide `jextract`
with location information needed to reference various include and lib paths.
(See the [extractions/pom.xml](../../extractions/pom.xml) file).


## Prerequisites

On Debian-based:

```bash
sudo apt install cmake maven
```

On RHEL-based:

```bash
sudo dnf install cmake maven
```

Or using `brew`:

```bash
brew install cmake
brew install maven
```

## Setting environment variables JAVA_HOME and PATH

To build HAT we need to ensure that `JAVA_HOME` points to our Babylon JDK (the
one we built [here](babylon.md))

We also need to ensure that `${JAVA_HOME}/bin` is in our PATH (before any other
JAVA location).

We also need `jextract` for some maven targets.

We can download and install `jextract` from [here](https://jdk.java.net/jextract/)

The `env.bash` shell script can be sourced (dot included) in your shell to set
`JAVA_HOME` and `PATH`.

```bash
cd hat
. ./env.bash
```

This should detect the arch type (`AARCH64` or `X86_64`) and select the correct
relative parent directory and inject that directory in your `PATH` if you are
working with HAT in the original subdir of the babylon project.

```bash
cd hat
source env.bash
```

## Building

Now we should be able to use maven to build, if successful maven will place all
jars and libs in a newly created `build` dir in your top level hat dir.

```bash
cd hat
source env.bash
mvn clean package
```

## Running examples

To run we use some common java `opt` files.

When executing with `java` we can place common java parameters in files (say opts),
and then use `java @opts` to avoid typing all the opts each time.

To declutter the hat directory, we prefix some useful `opt` files with `.`

For example in the `.ffi-opencl-example` opt file we have:

```bash
cat .ffi-opencl-example
--enable-preview --add-modules=jdk.incubator.code --enable-native-access=ALL-UNNAMED -Djava.library.path=build --class-path build/hat-ffi-opencl-examples-1.0.jar
```

### NBody

So you can run an example (say `nbody`)  using the `opencl` backend using

```bash
java @.ffi-opencl-example nbody.Main
```

### Matmul

```bash
java @.ffi-opencl-example matmul.Main
```

### Flash Attention
```bash
java @.ffi-opencl-example flashattention.Main
```

### ViolaJones

```bash
HAT=MC java @.ffi-opencl-example violajones.Main
```

### Life

```bash
HAT=MC java @.ffi-opencl-example life.Main
```

### Mandelbrot

```bash
HAT=MC java @.ffi-opencl-example mandel.Main
```

## Running HAT Test Suite

#### OpenCL Test Suite

```bash
java @.test-suite ffi-opencl
```

#### CUDA Test Suite

```bash
java @.test-suite ffi-cuda
```

----

## HAT runtime environment variables

During ffi-backend development we added some useful flags to pass to native code
to allow us to trace calls, inject logging, control options.

These should be considered just development flags.

At runtime the ffi-backends all communicate from the java side to the native
side via a 32 bit 'config' int.

The Java side
class [hat.backend.ffi.Config](https://github.com/openjdk/babylon/blob/code-reflection/hat/backends/ffi/shared/src/main/java/hat/backend/ffi/Config.java)

Is initialized from an environment variable `HAT` at runtime

So for example when we launch

```bash
java @hat/run ffi-opencl heal
```

We can pass config info to `HAT` via either `-DHAT=xxx` or via the `HAT` ENV
variable.

So for example to get HAT to dump the text form of the kernel code models.

```bash
HAT=INFO,SHOW_KERNEL_MODEL java @hat/run ffi-opencl heal
```

Or to show generated opencl or cuda code

```
HAT=INFO,SHOW_CODE java @hat/run ffi-opencl heal
```

This is particularly useful for selecting `PLATFORM` + `DEVICE` if you have multiple
GPU devices.

Here we select DEVICE 0 on PLATFORM 0 (actually the default)

```bash
HAT=INFO,PLATFORM:0,DEVICE:0,SHOW_CODE ...
```

or for DEVICE 1 on PLATFORM 1

```bash
HAT=INFO,PLATFORM:1,DEVICE:1,SHOW_CODE ...
```

Note the platform and device id's are 3 bits. This probably works for
development, but we will need a more expressive way of capturing this via the
accelerator selection.

This just allows us to test without code changes.

To keep he java code and the native code in sync. The `main` method at the end
of
[hat.backend.ffi.Config](https://github.com/openjdk/babylon/blob/code-reflection/hat/backends/ffi/shared/src/main/java/hat/backend/ffi/Config.java)
is actually used to create the C99 header for CUDA and OpenCL ffi-backends.

So whenever we change the Java config class we should run the main method to
generate the header. This is not really robust, (but proved way better than
trying to remember for all backends) but you need to know, if you add move
config bits to the Java side.

The Main method uses a variant of the code builders used for C99 style code
(CUDA/OpenCL) to generate the config.h header.

This is how we keep the ffi based backends in sync

Some more useful `config bits`:

```bash
HAT=PTX java @hat/run ffi-cuda ....
```

Sends PTX generated by our prototype PTX generator to the backend (for CUDA)
rather than C99 code.

```bash
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

Generally the flags all represent single bits (except PLATFORM:n and DEVICE:n)
and are set using comma separated uppercase+underscore string forms.

So to experiment with `minimizeCopies` and to  `showCode` we set
`HAT=MINIMIZE_COPIES,SHOW_CODE`

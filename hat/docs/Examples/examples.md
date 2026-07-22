# Running HAT Examples
[Back to Index ../](../index.md)

The [examples-package]() in HAT contains a list of examples varying from matrix operations, DFT, Flash-Attention, nbody, shaders, and image detection.

To run an example:

```bash
java @.ffi-<backend>-example <example>
```

For instance, to run `flash-attention` with the OpenCL backend:

```bash
java @.ffi-opencl-example flashattention.Main
```
For the CUDA backend:
```bash
java @.ffi-cuda-example  flashattention.Main
```

## Options for Examples

Some of the examples accept command line options to specify input size, kernel version, etc.

For example, `flashattention`:

```bash
java @.ffi-opencl-example flashattention.Main --size=2048 --iterations=10 --verbose
```

For flash attention You can see the full list of options by using `--help`:

```bash
    --size=<size>                   Specify an input size
    --iterations=<numIterations>    Specify the number of iterations to perform
    --skip-sequential               Flag to bypass the sequential execution in Java
    --check                         Flag to check the results. This implies the Java sequential version runs.
    --verbose                       Flag to print information between runs (e.g., total time).
    --help                          Print this help.
```

### Obtaining information about the accelerator used to launch the application

You can use the variable `INFO` to indicate the HAT runtime to dump the device name and device version used to launch the generated GPU kernel:


```bash
$ HAT=INFO java @.ffi-cuda-example matmul.Main

[INFO] Input Size     : 1024x1024
[INFO] Check Result:  : false
[INFO] Num Iterations : 100
[INFO] NDRangeConfiguration: 2DTILING

[INFO] Using NVIDIA GPU: NVIDIA GeForce RTX 5060   << an NVIDIA 5060 was used
[INFO] Dispatching the CUDA kernel
        \_ BlocksPerGrid   = [64,64,1]    << Num blocks dispatched
        \_ ThreadsPerBlock = [16,16,1]    << threads-per-block dispatched
```

## Running Headless Versions

Most GUI applications contains a `headless` version, which can be able by setting the following

```bash
java @.ffi-opencl-example -Dheadless=true  mandel.Main
```

Alternatively you can usually pass an arg to the app itself

```bash
java @.ffi-opencl-example  mandel.Main --headless
```

This sets `-Dheadless=true` and passes '--headless' to the example.


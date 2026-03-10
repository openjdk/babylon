
# Running Examples on the GPU

----
* [Contents](hat-00.md)
* Build Babylon and HAT
    * [Quick Install](hat-01-quick-install.md)
    * [Building Babylon with jtreg](hat-01-02-building-babylon.md)
    * [Building HAT with jtreg](hat-01-03-building-hat.md)
        * [Enabling the NVIDIA CUDA Backend](hat-01-05-building-hat-for-cuda.md)
* [Testing Framework](hat-02-testing-framework.md)
* [Running Examples](hat-03-examples.md)
* [HAT Programming Model](hat-03-programming-model.md)
* Interface Mapping
    * [Interface Mapping Overview](hat-04-01-interface-mapping.md)
    * [Cascade Interface Mapping](hat-04-02-cascade-interface-mapping.md)
* Development
    * [Project Layout](hat-01-01-project-layout.md)
* Implementation Details
    * [Walkthrough Of Accelerator.compute()](hat-accelerator-compute.md)
    * [How we minimize buffer transfers](hat-minimizing-buffer-transfers.md)
---

The [examples-package]() in HAT contains a list of of examples varying from matrix operations, DFT, Flash-Attention, nbody, shaders, and image detection.

To run an example:

```bash
java @hat/run ffi-<backend> <example>
```

For instance, to run `flash-attention` with the OpenCL backend:

```bash
java @hat/run ffi-opencl flashattention
```

For the CUDA backend:

```backend
java @hat/run ffi-cuda flashattention
```

## Options for Examples

Some of the examples accept command line options to specify input size, kernel version, etc.

For example, `flashattention`:

```bash
java -cp hat/job.jar hat.java run ffi-opencl flashattention --size=2048 --iterations=10 --verbose
```

You can see the full list of options by using `--help`:

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
$ HAT=INFO java @hat/run ffi-cuda matmul

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

GUI applications contains a `headless` version, which can be able by passing the following argument:

```bash
java @hat/run headless ffi-opencl mandel
```

This sets `-Dheadless=true` and passes '--headless' to the example.


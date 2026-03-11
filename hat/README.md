# Heterogeneous Accelerator Toolkit (HAT)

[![repo](https://img.shields.io/badge/github-repo-blue?logo=github)](https://github.com/openjdk/babylon/tree/code-reflection/hat)

HAT is a toolkit that allows developers to express data-parallel applications in Java, optimize, offload and execute them on hardware accelerators.
- **Heterogeneous**: a variety of devices and their corresponding programming languages.
- **Accelerator**: GPUs, FPGA, CPUs, etc.
- **Toolkit**: a set of libraries for Java developers.

HAT uses the code reflection API from the [Project Babylon](https://github.com/openjdk/babylon).

The toolkit offers:
- An API for Kernel Programming on Accelerators from Java.
- An API for Combining multiple kernels into a compute-graph.
- An API for Java object mapping to hardware accelerators using Panama FFM.
- An extensible backend system for multiple accelerators:
  - OpenCL
  - CUDA
  - Java

## Prerequisites

- HAT currently requires Babylon JDK, which contains the code reflection APIs.
- A base JDK >= 25. We currently use OpenJDK 26 for development.
- A GPU SDK (one or more of the SDKs below) to be able to run on GPUs:
  - An OpenCL implementation (e.g., Intel, Apple Silicon, CUDA SDK)
    - OpenCL >= 1.2
  - CUDA SDK >= 12.9
- `cmake` >= `3.22.1`
- `gcc` >= 12.0, or `clang` >= 17.0

## Compatible systems

We actively develop and run tests on the following systems:

- Apple Silicon M1-M4
- Linux Fedora >= 42
- Oracle Linux 10
- Ubuntu >= 22.04

## Quick Start

### 1. Build Babylon JDK

```bash
git clone https://github.com/openjdk/babylon
cd babylon
bash configure --with-boot-jdk=${JAVA_HOME}
make clean
make images
```

### 2. Update JAVA_HOME and PATH

```bash
export JAVA_HOME=<BABYLON-DIR>/build/macosx-aarch64-server-release/jdk/
export PATH=$JAVA_HOME/bin:$PATH
```

### 3. Build HAT

```bash
sdk install jextract #if needed
source env.bash
cd hat
java @.bld
```

Done!

## Run Examples

For instance, matrix-multiply:

```bash
java @.run ffi-opencl matmul --size=1024
```

Some examples have a GUI implementation:

```java
java @.run ffi-opencl mandel
```

Full list of examples:
- [link](https://github.com/openjdk/babylon/tree/code-reflection/hat/examples)


## Run Unit-Tests

OpenCL backend:

```bash
java @.test-suite ffi-opencl
```

CUDA backed:

```bash
java @.test-suite ffi-cuda
```

## Full Example Explained

The following example compute the square value of an input vector.
The example is self-contained and it can be directly run with the `java` command.

Place the following code in the `hat` directory.

```java
import hat.*;
import hat.Accelerator.Compute;
import hat.backend.*;
import hat.buffer.*;
import optkl.ifacemapper.MappableIface.*;
import jdk.incubator.code.Reflect;
import java.lang.invoke.MethodHandles;

public class ExampleHAT {

    // Kernel Code: This is the function to be offloaded to the accelerator (e.g.,
    // a GPU). The kernel will be executed by many GPU threads, in this case,
    // as many threads as elements in `array`.
    // The `kc` object can be used to obtain the thread identifier and map
    // the data element to process.
    // HAT kernels follow the SIMT programming model (Single Instruction Multiple Thread)
    // mode.
    // Kernel code is reflectable. Thus, the HAT runtime and HAT compiler can build
    // and optimize the code model. Once the code model is optimized, HAT generates
    // OpenCL/CUDA C99 code.
    @Reflect
    public static void squareKernel(@RO KernelContext kc, @RW S32Array array) {
        // HAT kernels support a reduced set of Java.
        // Kernels express the work to be done per thread (GPU/accelerator thread).
        if (kc.gix < array.length()) {
            int value = array.array(kc.gix);
            array.array(kc.gix, (value * value));
        }
    }

    // The following method represents the compute layer, in which we specify
    // the number of threads to be deployed on the accelerator. The number of threads
    // is specified in an ND-Range. An ND-Range could be 1D, 2D and 3D.
    // In this example, we launch 1D-range with the number of threads equal to
    // the input array size.
    @Reflect
    public static void square(@RO ComputeContext cc, @RW S32Array array) {
        var ndRange = NDRange.of1D(array.length());

        // Dispatch the kernel. The HAT runtime will offload the kernels
        // reached from this point and run the generated GPU kernels on the
        // target accelerator.
        // Furthermore, HAT automatically transfers data to the accelerator.
        // This is a blocking call, and when it returns control to the main
        // Java thread, results (outputs) are available to be consumed.
        cc.dispatchKernel(ndRange, kc -> squareKernel(kc, array));
    }

    static void main(String[] args) {
    	final int size = 4096;
        // Create a new accelerator object
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);

        // Instantiate an array on the target accelerator.
        // Data is stored off-heap using the Panama FFM API.
        var array = S32Array.create(accelerator, size);

        // Data initialization
        for (int i = 0; i < array.length(); i++) {
            array.array(i, i);
        }

        // Offload and dispatch of the compute-graph on the target accelerator.
        // This is a blocking call. Once this call finalizes, the results (outputs)
        // will be available to consume by the current Java thread.
        accelerator.compute((@Reflect Compute) cc -> ExampleHAT.square(cc, array));

        // Test result
        boolean isCorrect = true;
        for (int i = 0; i < size; i++) {
            if (array.array(i) != i * i) {
                isCorrect = false;
            }
        }
        if (isCorrect) {
            IO.println("Result is correct");
        } else {
            IO.println("Result is NOT correct");
        }
    }
}
```

Run this example in the `babylon/hat` directory.
If you run from another directory, update the `classpath` file accordingly.
Use the `java` version built with the Babylon JDK.

```bash
java --enable-preview \
   --add-modules=jdk.incubator.code \
   --enable-native-access=ALL-UNNAMED \
   --class-path $PWD/build/hat-optkl-1.0.jar:$PWD/build/hat-core-1.0.jar:$PWD/build/hat-backend-ffi-shared-1.0.jar:$PWD/build/hat-backend-ffi-opencl-1.0.jar \
   -Djava.library.path=/Users/juanfumero/repos/babylon/hat/build \
   ExampleHAT
```

## Documentation

Visit the [docs](docs/) folder.

## Contributing

Contributions are welcome. Please see the [OpenJDK Developers' Guide](https://openjdk.org/guide/).

## Development Workflow

1. Fork the repository
2. Create a feature branch: `git checkout -b <branch>`
3. Commit with clear messages
4. Run formatting and tests:
   1. For OpenCL: `java -cp hat/job.jar hat.java test-suite ffi-opencl`
   1. For CUDA: `java -cp hat/job.jar hat.java test-suite ffi-cuda`
5. Submit a pull request


## Contacts/Questions

You can interact, provide feedback and ask questions using the [babylon-dev](https://mail.openjdk.org/pipermail/babylon-dev/) mailing list.


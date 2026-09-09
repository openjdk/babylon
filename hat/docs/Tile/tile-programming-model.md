# HAT's Tile Programming Model implementation
[Back to Index ../](../index.md)

## Introduction

A tile model is a high-level programming abstraction that facilitates expressing
array-oriented algorithms across hardware accelerators, including GPUs.

In contrast to more conventional GPU programming models, such as OpenCL and CUDA,
in which programmers often organize work explicitly around threads and thread
blocks under a SIMT (Single Instruction Multiple Thread) model, a tile-based approach
describes operations on block of data (tiles). Then, an underlying compiler is free
to map logical blocks of data to hardware resources more efficiently.

HAT provides an initial implementation of a Tile Programming Model based on the [Triton
programming language](https://triton-lang.org/main/index.html) and inspired by the NVIDIA [Tile
programming model](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-tile-kernels.html).

While this prototype has been inspired by the aforementioned programmin models, the HAT Tile
programming model implementation does not necessarily follow the same parallel constructs.
Instead, HAT is free to evolve to incorporate those ideas and make them available for Java
programs, not only for GPUs, but also for other hardware.

## Disclaimer

HAT's tile programming model implementation is a work in progress that demonstrates how
tile programming can be integrated into Java through code reflection.

Currently, HAT only provides an implementation for CUDA, and maps the tile programs expressed
with Java to CUDA Tile C++. To be able to run Tile programs on NVIDIA hardware, developers
must have a GPU >= Ampere (Blackwell recommended), and use the NVIDIA driver >= 610.
See full [list of requirements below](#requirements).

The tile implementation in HAT does not include an OpenCL, or CPU implementations. However,
it is in our plans to extend support with both models (by mapping to OpenCL devices,
and providing a Java implementation).

## Requirements

- NVIDIA GPU Graphics Card, Ampere or later (Blackwell recommended).
- NVIDIA Driver `610.57.04` or later.
- CUDA SDK: `13.3` or later.
- [Babylon build for Java](../Build/babylon.md).

## Installation

If dependencies are satisfied, the build is identical to upstream `HAT`.

```bash
mvn clean package
```

## Example and Execution

```java
// Tile kernel to be offloaded and accelerator on the GPU
@Reflect
public static void vectorAddTile(TensorF32 inputA,
                                 TensorF32 inputB,
                                 TensorF32 output,
                                 final int tileSize) {

    // Access the thread-block id
    final var pid = TileContext.BIDX();

    // Load the tiles from the input tensors
    var tileA = TileContext.load(inputA, pid, tileSize);
    var tileB = TileContext.load(inputB, pid, tileSize);

    // Perform tile addition
    var result = TileOp.add(tileA, tileB);

    // Store the result into the output tensor
    TileContext.store(output, pid, result);
}

// Method dispatch to invoke a tile kernel
@Reflect
public static void vectorAddTile(ComputeContext computeContext,
                                 TensorF32 inputA,
                                 TensorF32 inputB,
                                 TensorF32 output,
                                 final int tileSize) {
    // invoke to dispatch tile method
    computeContext.dispatchTile(
            NDRange.of1D(inputA.m(), tileSize),    // 1D-Range Tile
            () -> vectorAddTile(inputA, inputB, output, tileSize)); // Invoke the Tile Kernel
}

public void run() {
    var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);

    final int size = Math.powExact(2, 16);
    final int tileSize = 64;

    TensorF32 inputA = TensorF32.create(accelerator, size);
    TensorF32 inputB = TensorF32.create(accelerator, size);
    TensorF32 result = TensorF32.create(accelerator, size);

    // Fill data
    Random r = new Random(19);
    for (int i = 0; i < size; i++) {
        inputA.array(i, r.nextFloat());
        inputB.array(i, r.nextFloat());
    }

    accelerator.compute((@Reflect Compute) computeContext ->
            vectorAddTile(computeContext, inputA, inputB, result, tileSize));
}
```

Run vector addition:

```bash
java @.ffi-cuda-test hat.test.TestTileAPI#test_hat_tile_01
```


## Limitations

- Current implementation in HAT implements a few Tile operations (`mma`, `add`, `sub`, `min`, etc.). More operations are planned.
- Current implementations only maps to CUDA Tile C++. Future versions will extend with Java and OpenCL implementations.


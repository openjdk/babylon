
# Babylon and HAT Quick Install

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
* [Running HAT with Docker on NVIDIA GPUs](hat-07-docker-build-nvidia.md)
---

## General Overview

This page shows a minimal installation and configuration for Babylon and HAT.
You can follow this guidelines to get started.

Alternatively, if you want to control more options during the installation,
such as `jtreg`, follow these links:
- [Building Babylon](hat-01-02-building-babylon.md)
- [Building HAT](hat-01-03-building-hat.md)

If you want to enable the CUDA backend for running on NVIDIA GPUs, use the following link
to obtain the list of dependencies for the CUDA SDK.

- [Enabling the NVIDIA CUDA Backend](hat-01-05-building-hat-for-cuda.md)

## 1. Build Babylon JDK

```bash
git clone https://github.com/openjdk/babylon
cd babylon
bash configure --with-boot-jdk=${JAVA_HOME}
make clean
make images
```

## 2. Update JAVA_HOME and PATH

```bash
export JAVA_HOME=<BABYLON-DIR>/build/macosx-aarch64-server-release/jdk/
export PATH=$JAVA_HOME/bin:$PATH
```

## 3. Build HAT

```bash
sdk install jextract #if needed
cd hat
java @.bld
```

Done!

## 4. Run Examples

For instance, matrix-multiply:

```bash
java @.run ffi-opencl matmul --size=1024
```

## 5. Unit-Tests

OpenCL backend:

```bash
java @.test-suite ffi-opencl
```

CUDA backed:

```bash
java @.test-suite ffi-cuda
```



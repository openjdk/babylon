
# Babylon and HAT Quick Install
[Back to Index ../](../index.md)

This page shows a minimal installation and configuration for Babylon and HAT.
You can follow this guidelines to get started.

Alternatively, if you want to control more options during the installation,
such as `jtreg` and/or jextract, follow these links:
- [Building Babylon](babylon.md)
- [Building HAT](hat.md)

If you want to enable the CUDA backend for running on NVIDIA GPUs, use the following link
to obtain the list of dependencies for the CUDA SDK.

- [Enabling the NVIDIA CUDA Backend](nvidia-notes.md)

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

## 3. Install HAT specific (cmake and maven) dependencies
Either
```bash
sudo apg-get install maven cmake
```
Or
```bash
brew install cmake
brew install maven
````
## 4. Build HAT
```bash
cd hat
java mvn clean package
```
Done!
## 5. Run Examples
For instance, matrix-multiply:
```bash
java @.ffi-opencl-example matmul.Main --size=1024
```
## 6. Unit-Tests
OpenCL backend:
```bash
java @.ffi-opencl-test-suite
```
CUDA backed:
```bash
java @.ffi-cuda-test-suite
```



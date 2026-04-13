
# Building Babylon

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
    * [IntelliJ Code Formatter](hat-development.md)
* Implementation Details
    * [Walkthrough Of Accelerator.compute()](hat-accelerator-compute.md)
    * [How we minimize buffer transfers](hat-minimizing-buffer-transfers.md)
* [Running HAT with Docker on NVIDIA GPUs](hat-07-docker-build-nvidia.md)
---

# Building Babylon

Openjdk Babylon can be found here [https://github.com/openjdk/babylon](https://github.com/openjdk/babylon)

If you follow the steps below to build Babylon, you should not have to  change any of
the `maven` or `cmake` build files for hat.

## Some useful vars

You will need an existing version of JDK to build Babylon and [jtreg](https://github.com/openjdk/jtreg).

The following build process assumes you have `BOOT_JDK` set to an existing JDK ([JDK 25+](https://jdk.java.net/25/)).
Note that for the Babylon development we use [JDK 26](https://jdk.java.net/26/).

```bash
export BOOT_JDK=${HOME}/java/jdk-25.0.1.jdk/Contents/Home/
```

### Clone Babylon from GitHub

[https://github.com/opendjk/babylon.git](https://github.com/opendjk/babylon.git)

```bash
git clone https://github.com/opendjk/babylon.git
```
### Get and build jtreg

In order to run openjdk tests we will need to get and build `jtreg`

[https://github.com/openjdk/jtreg](https://github.com/openjdk/jtreg)

```bash
cd ${GITHUB}
git clone https://github.com/openjdk/jtreg
```

We will build it now using our `BOOT_JDK`

```bash
export JTREG_ROOT=${GITHUB}/jtreg
cd ${JTREG_ROOT}
bash make/build.sh --jdk ${BOOT_JDK}
export JTREG_HOME=${JTREG_ROOT}/build/images/jtreg
```

### Configure

```bash
cd ${GITHUB}
git clone https://github.com/openjdk/babylon.git
cd ${GITHUB}/babylon
bash configure  --with-boot-jdk=${BOOT_JDK} --with-jtreg=${JTREG_HOME}
```
If you have never built JDK before you may find that the `configure` step will suggest packages to install.
I usually just keep running `bash configure` and take the suggestions (missing packages) I you get a successful Babylon build.

You now should have:

```bash
github
├── babylon
│   ├── build
│   │   └── XXXX-server-release
│   │       ├── Makefile
│   │       └── ...
│   ├── hat
│   │   ├── ...
```
Where XXXX is either linux-x64 or macosx-aarch64 and contains your build of Babylon JDK.

### Build Babylon

```bash
make clean
make images
#Coffee time (about 10 mins?)
```
You now should have:

```bash
github
├── babylon
│   ├── build
│   │   └── XXXX-server-release
│   │       ├── Makefile
│   │       └── jdk
│   │       └── images
│   │       └── hotspot
│   │       └── ...
│   ├── hat
│   │   ├── ...

```
### Run JTREG Tests

If we included `jtreg` above we can run the `babylon` code reflection tests using

```bash
cd ${GITHUB}/babylon
make test TEST=jdk_lang_reflect_code
```

This works because we added:

```bash
8<-
jdk_lang_reflect_code = \
   java/lang/reflect/code
->8
```
To the file `${GITHUB}/babylon/test/jdk/TEST.groups`

The tests themselves can be found in this directory

```bash
tree {GITHUB}/babylon}/test/jdk/java/lang/reflect/code
```

[Next Building HAT](hat-01-03-building-hat.md)


# Building Babylon

----

* [Contents](hat-00.md)
* House Keeping
  * [Project Layout](hat-01-01-project-layout.md)
  * [Building Babylon](hat-01-02-building-babylon.md)
  * [Building HAT](hat-01-03-building-hat.md)
    * [Enabling the CUDA Backend](hat-01-05-building-hat-for-cuda.md)
* Programming Model
  * [Programming Model](hat-03-programming-model.md)
* Interface Mapping
  * [Interface Mapping Overview](hat-04-01-interface-mapping.md)
  * [Cascade Interface Mapping](hat-04-02-cascade-interface-mapping.md)
* Implementation Detail
  * [Walkthrough Of Accelerator.compute()](hat-accelerator-compute.md)
  * [How we minimize buffer transfers](hat-minimizing-buffer-transfers.md)

---

# Building Babylon

Openjdk Babylon can be found here [https://github.com/openjdk/babylon](https://github.com/openjdk/babylon)

If you follow the steps below to build babylon, you should not have to
change any of the maven or cmake build files for hat.

## Some useful vars

You will need an existing version of JDK to build babylon and [jtreg](https://github.com/openjdk/jtreg).

The following build process assumes you have `BOOT_JDK` set to an existing JDK ([JDK 25+](https://jdk.java.net/25/)).
Note that for the Babylon development we use [JDK 26](https://jdk.java.net/26/).

```bash
export BOOT_JDK=${HOME}/java/jdk-25.0.1.jdk/Contents/Home/
```

### Create a suitable github dir

We suggest starting with a 'github' dir where we will install babylon, hat, jtreg and
other hat dependencies

The HAT maven build will assume that `${GITHUB}` -> `${HOME}/github`

```bash
export GITHUB=${HOME}/github
mkdir -p ${GITHUB}
mkdir -p ${HOME}/java
```

### Clone Babylon from github

[https://github.com/opendjk/babylon.git](https://github.com/opendjk/babylon.git)

```bash
cd ${GITHUB}
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
If you have never built JDK before you may find that the 'configure'
step will suggest packages to install.

I usually just keep running `bash configure` and take suggestions until I get a successful babylon build.

You now should have

```
github
├── babylon
│   ├── build
│   │   └── XXXX-server-release
│   │       ├── Makefile
│   │       └── ...
│   ├── hat
│   │   ├── ...

```
Where XXXX is either linux-x64 or macosx-aarch64 and contains your build of babylon JDK.

### Build Babylon

```bash
make clean
make images
#Coffee time (about 10 mins?)
```
You now should have

```
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
If we included jtreg above we can run the `babylon` code reflection tests using

```bash
cd ${GITHUB}/babylon
make test TEST=jdk_lang_reflect_code
```

This works because we added
```
8<-
jdk_lang_reflect_code = \
   java/lang/reflect/code
->8
```
To the file `${GITHUB}/babylon/test/jdk/TEST.groups`

The tests themselves can be found in this directory

```
tree {GITHUB}/babylon}/test/jdk/java/lang/reflect/code
```

[Next Building HAT](hat-01-03-building-hat.md)


# Building Babylon

----

* [Contents](hat-00.md)
* House Keeping
  * [Project Layout](hat-01-01-project-layout.md)
  * [Building Babylon](hat-01-02-building-babylon.md)
  * [Maven and CMake](hat-01-03-maven-cmake.md)
* Programming Model
  * [Programming Model](hat-03-programming-model.md)
* Interface Mapping
  * [Interface Mapping Overview](hat-04-01-interface-mapping.md)
  * [Cascade Interface Mapping](hat-04-02-cascade-interface-mapping.md)
* Implementation Detail
  * [Walkthrough Of Accelerator.compute()](hat-accelerator-compute.md)

---

# Building Babylon

Openjdk Babylon can be found here [https://github.com/openjdk/babylon](https://github.com/openjdk/babylon)

## Some useful vars

You will need an existing version of JDK to build babylon and jtreg.

The following build process assumes you have `BOOT_JDK` set to an existing JDK

```bash
export BOOT_JDK=${HOME}/java/jdk-22.0.1.jdk/Contents/Home/
```
### Clone Babylon from github

[https://github.com/opendjk/babylon.git](https://github.com/opendjk/babylon.git)

```bash
export GITHUB=${HOME}/github
mkdir -p ${GITHUB}
cd ${GITHUB}
git clone https://github.com/opendjk/babylon.git
```
### Get and build jtreg

In order to run openjdk tests we will need to build `jtreg`

[https://github.com/openjdk/jtreg](https://github.com/openjdk/jtreg)

```bash
cd ${GITHUB}
git clone https://github.com/openjdk/jtreg
export JTREG=${GITHUB}/jtreg
cd ${JTREG}
bash make/build.sh --jdk ${BOOT_JDK}
```
### Configure

```bash
cd ${GITHUB}/babylon
bash configure  --with-boot-jdk=${BOOT_JDK} --with-jtreg=${JTREG}/build/images/jtreg 
```
On your first build configure might exit and suggest installing other 
dependencies.  Generally I suggest just taking its recommendations and
restarting configure

Eventually we should complete and are ready to build

### Build

```bash
make clean
make images
#Coffee time (about 10 mins?)
```

### Run JTREG Tests
If we included jtreg above we can run the `babylon` tests using

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

# Building Babylon
[Back to Index ../](../index.md)

OpenJDK Babylon can be found here [https://github.com/openjdk/babylon](https://github.com/openjdk/babylon)

If you follow the steps below to build Babylon, you should not have to
change any of the `maven` or `cmake` build files for hat.

## Some useful vars
You will need an existing version of JDK to build Babylon and [jtreg](https://github.com/openjdk/jtreg).

The following build process assumes you have an env var called
`BOOT_JDK` set to an existing JDK ([JDK 26+](https://jdk.java.net/26/)).

For example assuming you have just installed java 26 on your machine into `${HOME}/java/jdk-26.jdk`
```bash
export BOOT_JDK=${HOME}/java/jdk-26.jdk/Contents/Home/
```

We will also assume that you have a `GITHUB` var pointing to location where you pull
repo's from GITHUB.

For example I always pull all my Babylon/HAT related repos in `${HOME}/github` so..
```bash
export GITHUB=${HOME}/github
```

### Clone jtreg from GitHub ([https://github.com/openjdk/jtreg](https://github.com/openjdk/jtreg))
```bash
cd ${GITHUB}
git clone https://github.com/openjdk/jtreg
```
### Build jtreg
```bash
export JTREG_ROOT=${GITHUB}/jtreg
cd ${JTREG_ROOT}
bash make/build.sh --jdk ${BOOT_JDK}
```
We will need to point a var called JTREG at the resulting image

```bash
export JTREG_HOME=${JTREG_ROOT}/build/images/jtreg
```

### Clone Babylon from GitHub ([https://github.com/opendjk/babylon.git](https://github.com/opendjk/babylon.git))
```bash
cd ${GITHUB}
git clone https://github.com/opendjk/babylon.git
```
### Configure Babylon
```bash
cd ${GITHUB}/babylon
bash configure  --with-boot-jdk=${BOOT_JDK} --with-jtreg=${JTREG_HOME}
```
If you have never built JDK before you may find that the `configure` step will suggest packages to install.
I usually just keep running `bash configure` and take the suggestions (missing packages) I you get a successful Babylon build.

### Build Babylon

```bash
make clean
make images
#Coffee time (between 2 mins and 10 mins usually depends on your machine)
```
You now should have:

```bash
github
├── babylon
│   ├── build
│   │   └── XXXX-server-release
│   │       ├── Makefile
│   │       ├── jdk
│   │       ├── images
│   │       ├── hotspot
│   │       └── ...
│   ├── hat
│   │   ├── ...
```

Where XXXX will depend on your architecture AARCH64 or X64_86

### Run can now JTREG Tests
If we included `jtreg` above we can run the `babylon` code reflection tests using
```bash
cd ${GITHUB}/babylon
make test TEST=jdk_lang_reflect_code
```
This works because babylon added added:
```txt
8<-
jdk_lang_reflect_code = \
   java/lang/reflect/code
->8
```
To the file `${GITHUB}/babylon/test/jdk/TEST.groups`

For the curious...the tests themselves can be found in this directory
```bash
tree {GITHUB}/babylon}/test/jdk/java/lang/reflect/code
```
[Next Building HAT](hat.md)

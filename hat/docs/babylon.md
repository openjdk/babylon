Openjdk Babylon is here https://github.com/openjdk/babylon

### Building Primer
Some useful vars

```bash
export GITHUB=${HOME}/github/grfrost/
export BOOT_JDK=${HOME}/java/jdk-22.0.1.jdk/Contents/Home/
```

```bash
cd ${GITHUB}
git clone https://github.com/opendjk/babylon.git
cd ${GITHUB}/babylon
bash configure  --with-boot-jdk=${BOOT_JDK}
make clean
make images
#Coffee time (about 10 mins?)
```

In order to run tests we need `jtreg` built using the JDK we just built (I think this is true). 

```bash
export BABYLON_JDK=${GITHUB}/babylon/build/macosx-aarch64-server-release/jdk
export BABYLON_JDK=${GITHUB}/babylon/build/linux-aarch64-server-release/jdk
cd ${GITHUB}
git clone https://github.com/openjdk/jtreg
cd ${GITHUB}/jtreg
#from doc/building.md
bash make/build.sh --jdk ${BABYLON_JDK}
```

Now we can go back and rebuild babylon using `jtreg` in our config  build 

```bash
export JTREG_HOME=${GITHUB}/jtreg/build/images/jtreg
cd ${GITHUB}/babylon
bash configure --with-boot-jdk=${BOOT_JDK}  --with-jtreg=${JTREG_HOME}
make clean
make images
```
Once we have built again we can run the `babylon` tests using

```bash
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

You can see (and add) the tests in this dir. 
```  
test/jdk/java/lang/reflect/code/TestGRF.java
```
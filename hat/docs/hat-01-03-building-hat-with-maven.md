# Building HAT

----

* [Contents](hat-00.md)
* House Keeping
    * [Project Layout](hat-01-01-project-layout.md)
    * [Building Babylon](hat-01-02-building-babylon.md)
    * [Building HAT](hat-01-03-building-hat.md)
* Programming Model
    * [Programming Model](hat-03-programming-model.md)
* Interface Mapping
    * [Interface Mapping Overview](hat-04-01-interface-mapping.md)
    * [Cascade Interface Mapping](hat-04-02-cascade-interface-mapping.md)
* Implementation Detail
    * [Walkthrough Of Accelerator.compute()](hat-accelerator-compute.md)

---

# Building HAT

HAT uses both maven and cmake.

Maven controls the build but delegates to cmake for native artifacts (such as various backends).


## Setting environment variables JAVA_HOME and PATH

To build HAT we need to ensure that `JAVA_HOME` is set
to point to our babylon jdk (the one we built [here](hat-01-02-building-babylon.md))

It will simplify our tasks going forward if we add `${JAVA_HOME}/bin` to our PATH (before any other JAVA installs).

The `env.bash` shell script can be sourced (dot included) in your shell to set JAVA_HOME and PATH

It should detect the arch type (AARCH64 or X86_46) and select the correct relative parent dir and inject that dir in your PATH.

```bash
cd hat
. ./env.bash
echo ${JAVA_HOME}
/Users/ME/github/babylon/hat/../build/macosx-aarch64-server-release/jdk
echo ${PATH}
/Users/ME/github/babylon/hat/../build/macosx-aarch64-server-release/jdk/bin:/usr/local/bin:......
```

## Root level maven pom.xml properties

If you followed the instructions for building babylon your `pom.xml`
properties should look like this, and should not need changing

```xml
<project>
    <!-- yada -->
    <properties>
        <project.build.sourceEncoding>UTF-8</project.build.sourceEncoding>
        <hat.build>${env.PWD}/build</hat.build>
    </properties>
    <!-- yada -->
</project>
```
## Sanity checking your env and root pom.xml

After sourcing `env.bash` or making changes to `pom.xml` we can
sanity check our setup by running

```bash
cd hat
java sanity.java
```

This will check that your `PATH` includes your babylon JDK's bin dir, and will parse the top level `pom.xml` to ensure that that
the properties are pointing to `sane` values.

## Building with maven

Now we should be able to use maven to build, if successful maven will place all jars and libs in a newly created `build` dir in your top level hat dir.

```bash
cd hat
. ./env.bash
mvn clean  compile jar:jar install
ls build
hat-1.0.jar                     hat-example-heal-1.0.jar        libptx_backend.dylib
hat-backend-cuda-1.0.jar        hat-example-mandel-1.0.jar      libspirv_backend.dylib
hat-backend-mock-1.0.jar        hat-example-squares-1.0.jar     mock_info
hat-backend-opencl-1.0.jar      hat-example-view-1.0.jar        opencl_info
hat-backend-ptx-1.0.jar         hat-example-violajones-1.0.jar  ptx_info
hat-backend-spirv-1.0.jar       libmock_backend.dylib           spirv_info
hat-example-experiments-1.0.jar libopencl_backend.dylib
```

The provided `build.sh` script contains the minimal maven commandline

```bash
bash build.sh
```

## Running an example

To run an example we should be able to use the maven artifacts in `build`

```bash
${JAVA_HOME}/bin/java \
   --enable-preview --enable-native-access=ALL-UNNAMED \
   --class-path build/hat-1.0.jar:build/hat-example-mandel-1.0.jar:build/hat-backend-opencl-1.0.jar \
   --add-exports=java.base/jdk.internal=ALL-UNNAMED \
   -Djava.library.path=build\
   mandel.Main
```

The provided `hatrun.bash` script simplifies this somewhat, we just need to pass the backend name `opencl` and the package name `mandel`
(all examples are assumed to be in `packagename/Main.java`

```bash
bash hatrun.bash opencl mandel
```

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

# Building HAT with Bldr

We initially used maven and cmake to build hat.  If you feel more comfortable
with maven consider [building with maven and cmake](hat-01-03-building-hat-with-maven.md)
but it is possible that maven support will be removed if the `Bldr` approach takes off.

We might even have `Bldr` create the maven artifacts....

## Setting environment variables JAVA_HOME and PATH

To build HAT we need to ensure that `JAVA_HOME` is set
to point to our babylon jdk (the one we built [here](hat-01-02-building-babylon.md))

It simplifes our tasks going forward if we
add `${JAVA_HOME}/bin` to our PATH (before any other JAVA installs).

Thankfully just sourcing the top level `env.bash` script will perform these tasks

It should detect the arch type (AARCH64 or X86_46) and
select the correct relative parent dir and inject that dir in your PATH.

```bash
cd hat
. ./env.bash
echo ${JAVA_HOME}
/Users/ME/github/babylon/hat/../build/macosx-aarch64-server-release/jdk
echo ${PATH}
/Users/ME/github/babylon/hat/../build/macosx-aarch64-server-release/jdk/bin:/usr/local/bin:......
```

# Introducing Bldr
`Bldr` is an evolving set of static methods and types required (so far.. ;) )
to be able to build HAT, hat backends and examples.

We rely on the ability to launch java source directly (without needing to javac first)

* [JEP 458: Launch Multi-File Source-Code Program](https://openjdk.org/jeps/458)
* [JEP 330: Launch Single-File Source-Code Programs](https://openjdk.org/jeps/330)

The `bld` script (really java source) can be run like this

```bash
java --add-modules jdk.incubator.code --enable-preview --source 24 bld
```

In our case the  magic is under the `hat/bldr`subdir

```
bldr
├── Bldr.java (symlink) -> src/main/java/bldr/Bldr.java
├── args      (text)       "--enable-preview --source 24"
└── src
    └── main
        └── java
            └── bldr
                └── Bldr.java
```

We also have a handy `bldr/args` which allows us to avoid specifying commmon args `--enable-preview --source 24` which are always needed

```bash
java @bldr/args bld
```

This `bld` script builds HAT, all the backends and examples and places buildable artifacts in `build` dir

```bash
cd hat
. ./env.bash
java @bld/args bld
ls build
hat-1.0.jar                     hat-example-heal-1.0.jar        libptx_backend.dylib
hat-backend-cuda-1.0.jar        hat-example-mandel-1.0.jar      libspirv_backend.dylib
hat-backend-mock-1.0.jar        hat-example-squares-1.0.jar     mock_info
hat-backend-opencl-1.0.jar      hat-example-view-1.0.jar        opencl_info
hat-backend-ptx-1.0.jar         hat-example-violajones-1.0.jar  ptx_info
hat-backend-spirv-1.0.jar       libmock_backend.dylib           spirv_info
hat-example-experiments-1.0.jar libopencl_backend.dylib
```

`bld` relies on cmake to build native code for backends, so if cmake finds OpenCL libs/headers, you will see libopencl_backend (.so or .dylib) in the build dir, if cmake finds CUDA you will see libcuda_backend(.so or .dylib)

We have another script called `sanity` which will check all  .md/.java/.cpp/.h for tabs, lines that end with whitespace
or files without appropriate licence headers

This is run using

```
java @bldr/args sanity
```


## Running an example

To run a HAT example we can run from the artifacts in `build` dir

```bash
${JAVA_HOME}/bin/java \
   --add-modules jdk.incubator.code --enable-preview --enable-native-access=ALL-UNNAMED \
   --class-path build/hat-1.0.jar:build/hat-example-mandel-1.0.jar:build/hat-backend-opencl-1.0.jar \
   --add-exports=java.base/jdk.internal=ALL-UNNAMED \
   -Djava.library.path=build\
   mandel.Main
```

The `hatrun` script can also be used which simply needs the backend
name `opencl|java|cuda|ptx|hip|mock` and the package name `mandel`

```bash
java @bldr/args hatrun opencl mandel
```

If you pass `headless` as the first arg

```bash
java @bldr/args hatrun headless opencl mandel
```

This sets `-Dheadless=true` and passes '--headless' to the example.  Some examples can use this to avoid launching UI.


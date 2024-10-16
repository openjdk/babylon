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

It will simplify our tasks going forward if we
add `${JAVA_HOME}/bin` to our PATH (before any other JAVA installs).

The top level `env.bash` shell script can be sourced (dot included)
into your shell to both set `JAVA_HOME` and update your `PATH`

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
`Bldr` is a minimal java build system which has all the capabilities needed so far
to build existing HAT as well as it's examples and backends. It is just a set of
static methods and helper classes wrapping javac/jar tooling.

`Bldr` itself is a java class in
```
bldr
  └── src
      └── main
          └── java
              └── bldr
                  └── Bldr.java
```

We do need to compile this one class using javac one time, then we can use this in `bld` scripts to actually bld.

Assuming we have our babylon JDK build in our path (via `. env.bash`) we should do this every time we 'pull' HAT.

```shell
mkdir bldr/classes
javac --enable-preview -source 24 -d bldr/classes bldr/src/main/java/bldr/Bldr.java
```

Now the `bldr/classes` dir contains all we need to create build scripts

In HAT's root dir is a `#!` (Hash Bang) java launcher style script called `bld`
which uses tools exposed by the precompiled `Bldr` to compile, create jars, run jextract, download dependencies, tar/untar etc.

As git does not allow us to check in scripts with execute permission, we need to `chmod +x` this `bld` file.

```bash
chmod +x bld
```

A simple example of `bld` script which just compiles core HAT source to `build/hat-1.0.jar` is shown.

```java
#!/usr/bin/env java --enable-preview --source 24 --class-path bldr/classes
import module java.compiler;
import static bldr.Bldr.*;
void main(String[] args) throws IOException, InterruptedException {
    var hatDir = Path.of(System.getProperty("user.dir"));
    var target = path(hatDir, "build");// mkdir(rmdir(path(hatDir, "build")));

    var hatJarResult = javacjar($ -> $
            .opts(  "--source", "24",
                    "--enable-preview",
                    "--add-exports=java.base/jdk.internal=ALL-UNNAMED",
                    "--add-exports=java.base/jdk.internal.vm.annotation=ALL-UNNAMED")
            .jar(path(target, "hat-1.0.jar"))
            .source_path(path(hatDir, "hat/src/main/java"))
    );
}
```
Note that the first line has the `#!` magic to allow this java code to be executed as if it
were a script.  Whilst `bld` is indeed real java code,  we do not need to compile it. Instead we just execute using

```bash
./bld
```

The real `bld` is more complicated, but not much more. It will will build hat-1.0.jar, along with all the backend jars hat-backend-?-1.0.jar,
all the example jars hat-example-?-1.0.jar and will try to build all native artifacts (.so/.dylib) it can.
So if cmake finds OpenCL libs/headers, you will see libopencl_backend (.so or .dylib)

On a CUDA machine you will see libcuda_backend(.so or .dylib)

`bld` will also sanity check .java/.cpp/.h files to make sure we don't have any tabs, lines that with whitespace
or files without appropriate licence headers

```bash
cd hat
. ./env.bash
./bld
ls build
hat-1.0.jar                     hat-example-heal-1.0.jar        libptx_backend.dylib
hat-backend-cuda-1.0.jar        hat-example-mandel-1.0.jar      libspirv_backend.dylib
hat-backend-mock-1.0.jar        hat-example-squares-1.0.jar     mock_info
hat-backend-opencl-1.0.jar      hat-example-view-1.0.jar        opencl_info
hat-backend-ptx-1.0.jar         hat-example-violajones-1.0.jar  ptx_info
hat-backend-spirv-1.0.jar       libmock_backend.dylib           spirv_info
hat-example-experiments-1.0.jar libopencl_backend.dylib
```

## Running an example

To run a HAT example we can run from the artifacts in `build` dir

```bash
${JAVA_HOME}/bin/java \
   --enable-preview --enable-native-access=ALL-UNNAMED \
   --class-path build/hat-1.0.jar:build/hat-example-mandel-1.0.jar:build/hat-backend-opencl-1.0.jar \
   --add-exports=java.base/jdk.internal=ALL-UNNAMED \
   -Djava.library.path=build\
   mandel.Main
```

The provided `hatrun.bash` script simplifies this somewhat, we just need to pass the backend
name `opencl` and the package name `mandel`
(all examples are assumed to be in `packagename/Main.java`

```bash
bash hatrun.bash opencl mandel
```

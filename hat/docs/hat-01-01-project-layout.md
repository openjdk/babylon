
# Project Layout

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
    * [How we minimize buffer transfers](hat-minimizing-buffer-transfers.md)

---

# Primer

This is a fairly large project with Java and Native artifacts which is completely dependant
on the `babylon` project, and as such is initially available in a sub directory
called `hat` under [github.com/openjdk/babylon](https://github.com/openjdk/babylon)

## Project Layout

```
${BABYLON_JDK}
     ./
     +--build/                     All jars, native libs and executables
     |    +--cmake-build-debug/    All intermediate cmake artifacts
     |
     +--stage/
     |    +--repo/                 All downloaded maven assets
     |    |
     |    +--jextract/             All jextracted files
     |    |    +--opencl
     |    |    +--opengl
     |    |    +--cuda
     |
     +--hat
     |    + Script.java
     |    + run.java + @run
     |    + bld.java + @bld
     |    + clean.java + @bld
     |
     +--hat-core                      * Note maven style layout
     |    +--src/main/java
     |    |    +--hat/
     |    |
     |    +--src/main/test
     |         +--hat/
     |
     +--backends
     |    +--java
     |    |    +--mt                    (*)
     |    |    +--seq                   (*)
     |    +--jextracted
     |    |    +--opencl                (*)
     |    +--ffi
     |    |    +--opencl                (*)
     |    |    +--ptx                   (*)
     |    |    +--mock                  (*)
     |    |    +--spirv                 (*)
     |    |    +--cuda                  (*)
     |    |    +--hip                   (*)
     |
     +--examples
     |    +--mandel                (*)
     |    +--squares               (*)
     |    +--heal                  (*)
     |    +--life                  (*)
     |    +--violajones            (*)

```

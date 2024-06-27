
# Project Layout

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

# Primer

This is a fairly large project with Java and Native artifacts which is completely dependant
on the `babylon` project, and as such is initially available in a sub directory
called `hat` under [github.com/openjdk/babylon](https://github.com/openjdk/babylon)

## Project Layout

```
${BABYLON_JDK}
   └── hat
        │
        ├── CMakeFile
        ├── build
        │
        ├── intellij
        │    ├── .idea
        │    │    ├── compiler.xml
        │    │    ├── misc.xml
        │    │    ├── modules.xml
        │    │    ├── uiDesigner.xml
        │    │    ├── vcs.xml
        │    │    └── workspace.xml
        │    │
        │    ├── hat.iml
        │    ├── backend_(spirv|mock|cuda|ptx|opencl).iml
        │    └── (mandel|violajones|experiments).iml
        │
        ├── hat
        │    └── src
        │         └── java
        │
        ├── backends
        │    └── (opencl|cuda|ptx|mock|shared)
        │          └── src
        │              ├── cpp
        │              ├── include
        │              ├── java
        │              └── services
        └── examples
             ├── mandel
             │    └── src
             │         └── java
             └── violajones
                  └── src
                       ├── java
                       └── resources
```

# HAT Project

This is a fairly large project with Java and native (C/C++) artifacts.

To use HAT you will need to clone and build the  `babylon` (JDK24+Babylon) project or of course your fork of the babylon project (say `babylon-myfork`)

[github.com/openjdk/babylon](https://github.com/openjdk/babylon)

The HAT project is in the 'hat' subdir of the babylon project.

## Building Babylon

[See](docs/hat-01-02-building-babylon.md)

## Building HAT

HAT uses both maven and cmake.

Maven controls the build but delegates to cmake for native artifacts (such as various backends).

[See](docs/hat-01-03-building-hat.md)


## Intellij and Clion
We can use JetBrains' `intelliJ` and `clion` but care must be taken as these tools
do not play well together, specifically we cannot have `Clion` and `Intellij`
project artifacts rooted under each other or in the same dir.

The `intellij` subdir houses the various `*.iml` module files and the project `.idea` so
just open that dir as an intellij project

Thankfully `clion` uses cmake. So we can reuse the same `hat/backends/CMakeLists.txt` that
maven uses to build the backends.

### Initial Project Layout

```
${BABYLON_JDK}
   └── hat
        ├── maven-build (created by the build)
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
        │    └── (mandel|violajones|experiments|heal|view).iml
        │
        ├── hat
        │    ├── pom.xml
        │    └── src
        │         ├── src/main/java
        │         └── src/main/resources
        │
        ├── backends
        │    ├── pom.xml
        │    ├── CMakeLists.txt
        │    └── (opencl|cuda|ptx|mock|shared)
        │          ├── pom.xml
        │          ├── CMakeLists.txt
        │          ├── cpp
        │          ├── include
        │          ├── src/main/java
        │          └── src/main/resources
        └── examples
             ├── pom.xml
             └── (mandel|violajones|squares|heal|view|experiments)
                    ├── pom.xml
                    ├── src/main/java
                    └── src/main/resources
```
As you will note the `intellij` dir is somewhat self contained.  the various `*.iml`
files refer to the source dirs using relative paths.

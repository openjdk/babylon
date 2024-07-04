
# Building HAT

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

# Building HAT

We use maven as the primary build tool, but we also required cmake to be available
as maven delegates to cmake to building native OpenCL/CUDA libs in the various backends.

Whilst the root level cmake `CMakeLists.txt` can create some java artifacts, it should not be
relied on, and will probably be 'deprecated soon'

To build with maven

```bash
cd ${GITHUB}/babylon
mvn clean compile jar:jar install
```


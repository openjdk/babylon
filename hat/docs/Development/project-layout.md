# Project Layout
[Back to Index ../](../index.md)

```tree
${BABYLON_JDK}
  ./
     +--core
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

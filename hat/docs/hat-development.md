# HAT Code Formatting in IntelliJ

----
* [Contents](hat-00.md)
* Build Babylon and HAT
    * [Quick Install](hat-01-quick-install.md)
    * [Building Babylon with jtreg](hat-01-02-building-babylon.md)
    * [Building HAT with jtreg](hat-01-03-building-hat.md)
        * [Enabling the NVIDIA CUDA Backend](hat-01-05-building-hat-for-cuda.md)
* [Testing Framework](hat-02-testing-framework.md)
* [Running Examples](hat-03-examples.md)
* [HAT Programming Model](hat-03-programming-model.md)
* Interface Mapping
    * [Interface Mapping Overview](hat-04-01-interface-mapping.md)
    * [Cascade Interface Mapping](hat-04-02-cascade-interface-mapping.md)
* Development
    * [Project Layout](hat-01-01-project-layout.md)
    * [IntelliJ Code Formatter](hat-development.md)
* Implementation Details
    * [Walkthrough Of Accelerator.compute()](hat-accelerator-compute.md)
    * [How we minimize buffer transfers](hat-minimizing-buffer-transfers.md)
* [Running HAT with Docker on NVIDIA GPUs](hat-07-docker-build-nvidia.md)
---

## How to configure it?

In IntelliJ:

1. Enable Actions on Save: Settings -> Tools -> Actions on Save -> Enable `Reformat Code`
2. Import XML for auto-formatter rules: Settings -> Editor -> Code Style -> Java -> Scheme (gear) -> Import Scheme -> IntelliJ IDEA code style XML
3. Select file from `hat/scripts/IDEA-xml-formatter/HAT-Formatter.xml`



# Babylon/HAT Intern Project 
#### Gary Frost - May 11th 2024


We recently made available an Openjdk project called [Babylon](https://openjdk.org/projects/babylon)

[Babylon Code](https://github.com/openjdk/babylon)


This project allows Java developers access to the 'model' of Java methods which have been annotated with `@CodeRelection`, access to the code model allows the developer to analyze the code (possibly to reinterpret), as well as to modify the code (possibly to inject new behaviour, or make it suitable for a execution by a completely different runtime). 

We (well mostly Paul Sandoz) have collected a number of technical use cases which showcase Babylon, but one that we think is particularly compelling is the notion of using Babylon to help accelerate suitable Java applications using GPU's, or more generallt by allowing execution to be moved to a heterogeneous 'accelerator' device such as a GPU.

To this end we also plan to make a `Heterogeneous Accelerator Toolkit` or `HAT` available as a Babylon demonstration project. 

We demonstrated the concept at JVMLS 2023   

[Java on GPU's... are we there yet](https://www.youtube.com/watch?v=lbKBu3lTftc)

[Java On The GPU - Inside Java Newscast #58](https://www.youtube.com/watch?v=q8pxRkdKeR0)

Our plan is to allow a GPU kernel code to be expressed in Java, and then (using Babylon) convert the code to a form suitable for GPU's. 

We will also use Babylon to analyse the use of the kernel to minimize data transfers beetween host and accelerator. 

So given a simple kernel described like this
 
```
@CodeReflection 
public static 
   void squareKernel(NDRange id, S32Array arr) {
       int value = arr.array(id.x);
       s32Array.array(id.x, value * value);
   }
```

Javac will generate a Babylon model of this method and the Java runtime will allow us to reflectively access the model. 

To move compute to a GPU, we need to convert this model into a form that is suitable for the particular hardware we have available. 

Both CUDA and OpenCL can accept `kernels` written in a C99 variant.  In fact using just a few #defines  and typedefs we can generate C99 code which is valid for both CUDA and OpenCL backends.   

Ideally though we would prefer not to generate C99 intermediate source.  We would prefer to target the intermediate format used by the various backends. 

Intel have generously contributed a [SPIRV](https://www.khronos.org/spir/) prototype backend for Babylon/HAT, which takes a Babylon code model and generats SPIRV code. This code is the intermediate form that  OpenCL and [Sycl](https://www.intel.com/content/www/us/en/developer/tools/oneapi/data-parallel-c-plus-plus.html#gs.97r3zh) use to offload to their hardware and is part of Intels [One API](https://www.intel.com/content/www/us/en/developer/tools/oneapi/overview.html#gs.97r51v) programming domain.
 
For NVida devices we would like to generate a [PTX](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html) backend. PTX is (well one of..) the forms that NVidia use to turn CUDA code into a form suitable for their hardware.

So for this project, we would like to create a library toolkit (using Babylon and existing HAT infrastructure) to generate PTX from a captured Babylon kernel code model.

I think the SPIRV contribution is a nice place to start.  

Here is a link to the code in the Babylon repo. 

[SPIRV Babylon Code](https://github.com/openjdk/babylon/tree/code-reflection/cr-examples/spirv)

I would suggest the following steps to start with 

1) Setup dev environment and build Babylon (from the OpenJDK github link above)
2) Pull the HAT repo from `orahub` and setup a dev enviroment (Intellij is probably easiest).

The above steps can be performed on the Mac Oracle has provided. 

3) Get access to a machine with NVdia toolchain (nvcc compiler and runtime)

We are working on this step ;)

4) We will use a few of the HAT demos to generate kernel code models, and create a PTX backend.

5) You should spend a little time reading the PTX spec, Pauls various Babylon talks and  

6) Build some of the simpler [Cuda Samples](https://github.com/NVIDIA/cuda-samples/) and become and generate PTX to become familiar with the relationship between the CUDA C99 style source and generated code. 

7) Code test code test repeat ;)    


Some interesting links. 

[Cuda Compiler Driver](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#virtual-architectures)

[NVPTX Usage](https://llvm.org/docs/NVPTXUsage.html)




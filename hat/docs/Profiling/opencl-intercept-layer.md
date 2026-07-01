# Using OpenCL Intercept Layer for HAT
[Back to Index ../](../index.md)

The [OpenCL Intercept Layer](https://github.com/intel/opencl-intercept-layer) is a tool that intercepts OpenCL calls
for debugging and performance analysis. We can use this tool for multiple OpenCL platforms, including Intel, NVIDIA and macOS.

## How to install OpenCL Intercept Layer?

```bash
git clone https://github.com/intel/opencl-intercept-layer.git
cd opencl-intercept-layer
mkdir build
cd build
## We can optionally enable cliprof, but we mainly use cliloader
cmake .. -DENABLE_CLIPROF=1
```

Then, add in your `PATH` the `opencl-intercept-layer/build/cliloader` directory.

```bash
export PATH=/path/to/opencl-intercept-layer/build/cliloader:$PATH
```

## How to use with HAT

```bash
cliloader \
  -d -h \
  java @.ffi-opencl-example tensors.Main --iterations=10 --verbose
```

Example of output:

```bash
Host Performance Timing Results:

Total Time (ns): 374760223

                          Function Name,  Calls,     Time (ns), Time (%),  Average (ns),      Min (ns),      Max (ns)
               (device timing overhead),     60,         57423,    0.02%,           957,             0,          4459
                        iclBuildProgram,      3,      70517666,   18.82%,      23505888,        677208,      61181833
                        iclCreateBuffer,     10,         20667,    0.01%,          2066,           375,          5666
                  iclCreateCommandQueue,      1,         45291,    0.01%,         45291,         45291,         45291
                       iclCreateContext,      1,        448625,    0.12%,        448625,        448625,        448625
                        iclCreateKernel,      3,     133957582,   35.74%,      44652527,        292833,     133316541
             iclCreateProgramWithSource,      3,         43709,    0.01%,         14569,         12667,         16208
           iclEnqueueMarkerWithWaitList,    120,        300161,    0.08%,          2501,           125,         11166
 iclEnqueueNDRangeKernel( mxmNaiveF16 ),     10,          9917,    0.00%,           991,           542,          1750
 iclEnqueueNDRangeKernel( mxmNaiveF32 ),     10,         13252,    0.00%,          1325,           500,          2583
iclEnqueueNDRangeKernel( mxmTensorsCM ),     10,          9624,    0.00%,           962,           583,          1625
                   iclEnqueueReadBuffer,     30,        163125,    0.04%,          5437,          3834,          9667
                  iclEnqueueWriteBuffer,     90,        683671,    0.18%,          7596,           542,         70291
                        iclGetDeviceIDs,      2,      24621167,    6.57%,      12310583,           250,      24620917
                       iclGetDeviceInfo,    660,         38704,    0.01%,            58,             0,           875
                      iclGetPlatformIDs,      2,            83,    0.00%,            41,            41,            42
                     iclGetPlatformInfo,    180,        338296,    0.09%,          1879,             0,        330709
                 iclGetProgramBuildInfo,      9,          5959,    0.00%,           662,            42,          2333
                        iclReleaseEvent,    270,         45167,    0.01%,           167,            41,          1125
                        iclSetKernelArg,    150,         25224,    0.01%,           168,            41,           750
                       iclWaitForEvents,     60,     143414910,   38.27%,       2390248,         15166,      20305042

Device Performance Timing Results for Apple M4 Max (40CUs, 1000MHz):

Total Time (ns): 3174486

                   Function Name,  Calls,     Time (ns), Time (%),  Average (ns),      Min (ns),      Max (ns)
            iclEnqueueReadBuffer,     30,         48206,    1.52%,          1606,           758,          6029
           iclEnqueueWriteBuffer,     90,         90860,    2.86%,          1009,            53,          9861
                     mxmNaiveF16,     10,        906729,   28.56%,         90672,         89916,         96693
                     mxmNaiveF32,     10,       1675136,   52.77%,        167513,         98113,        382520
                    mxmTensorsCM,     10,        453555,   14.29%,         45355,         38614,         46225
```

## How to use with Chrome Tracing

```bash
cliloader -d -h \
  --chrome-call-logging \
  --chrome-device-timeline \
  --chrome-kernel-timeline \
  --chrome-device-stages \
  java @.ffi-opencl-example tensors.Main --iterations=10 --verbose
```

The same functionality could be achived by invoking the `scripts/cliloader-chrome-opencl.bash` script.

```bash
sh scripts/cliloader-opencl.bash tensors.Main --iterations=10 --verbose
```

Then open Chrome and enter the following url: `chrome://tracing`.

Then load the traces (usually a file called `CLIntercept_Trace.json`) that is stored in the default location of the `cliloader` tool.

To obtain the default location, run `cliloader | grep dump-dir -A 3`.


## Documentation
- https://github.com/intel/opencl-intercept-layer/tree/main/docs
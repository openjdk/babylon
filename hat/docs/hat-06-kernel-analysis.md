
# Compute Analysis or Runtime tracing

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

----

# Compute Analysis or Runtime tracing

HAT does not dictate how a backend chooses to optimize execution, but does 
provide the tools (Babylon's Code Models) and some helpers which the Backend is encouraged 
use.  

The ComputeContext contains all the information that the backend needs, but does not 
include any 'policy' for minimizing data movements.

Our assumption is that backend can use various tools to deduce the most efficient execution strategy.

## Some possible strategies..

### Copy data every time 'just in case' (JIC execution ;) ) 
Just naiively execute the code as described in Compute graph. So the backend will copy each buffer to the device, execute the kernel and copy the data back again. 

### Use kernel knowledge to minimise data movement
Execute the code described in the Compute Graph, but use knowledge extracted from kernel models
to only copy to device buffers that the kernel is going to read, and only copy back from the device
buffers that the kernel has written to. 

### Use Compute knowledge and kernel knowledge to further minimise data movement
Use knowledge extracted from the compute reachable graph and the kernel 
graphs to determine whether Java has mutated buffers between kernel dispatches
and only copy data to the device that we know the Java code has mutated. 

This last strategy is ideal

We can achieve this using static analysis of the compute and kernel models or by being 
involved in the execution process at runtime.

#### Static analysis 

#### Runtime Tracking  

* Dynamical
1. We 'close over' the call/dispatch graph from the entrypoint to all kernels and collect the kernels reachable from the entrypoint and all methods reachable from methods reachable by kernels.
2. We essentially end up with a graph of codemodels 'rooted' at the entrypoint
3. For each kernel we also determine how the kernel accesses it's 'MemorySegment` parameters, for each MemorySegment parameters we keep a side table of whther the kernel reads or writes to the segment. We keep this infomation in a side map.

This resulting 'ComputeClosure' (tree of codemodels and relevant side tables) is made available to the accelerator to coordinate execution.

Note that our very simple Compute::compute method neither expresses the movement of the MemorySegment to a device, or the retrieval of the data from a device when the kernel has executed.

Our assumption is that given the ComputeClosure we can deduce such movements.

There are many ways to achieve this.  One way would be by static analysis.

Given the Compute::compute entrypoint it is easy to determine that we are always (no conditional or loops) passing (making available
might be a better term) a memory segment to a kernel (Compute::kernel) and this kernel only mutates the  `MemorySegment`.

So from simple static analysis we could choose to inject one or more calls into the model representing the need for the accelerator to move data to the devices and/ord back from the device, after the kernel dispatch.

This modified model, would look like we had presented it with this code.

```java
 void compute(Accelerator accelerator, MemorySegment memorySegment, int len) {
        Accelerator.Range range = accelerator.range(len);
        accelerator.run(Compute::kernel, range, memorySegment);
        accelerator.injectedCopyFromDevice(memorySegment);
    }
```

Note the ```injectedCopyFromDevice()``` call.

Because the kernel does not read the `MemorySegment` we only need inject the code to request a move back from the device.

To do this requires HAT to analyse the kernel(s) and inject appropriate code into
the Compute::compute method to inform the vendor backend when it should perform such moves.

Another strategy would be to not rely on static analysis but to inject code to trace 'actual' mutations of the MemorySegments and use these flags to guard against unnecessary copies

```java
 void compute(Accelerator accelerator, MemorySegment memorySegment, int len) {
        boolean injectedMemorySegmentIsDirty = false;
        Accelerator.Range range = accelerator.range(len);
        if (injectedMemorySegmentIsDirty){
            accelerator.injectedCopyToDevice(memorySegment);
        }
        accelerator.run(Compute::kernel, range, memorySegment);
        injectedMemorySegmentIsDirty = true; // based on Compute::kernel sidetable
        if (injectedMemorySegmentIsDirty) {
            accelerator.injectedCopyFromDevice(memorySegment);
        }
    }
```


Whether this code mutation generates Java bytecode and executes (or interprets) on the JVM or whether the
CodeModels for the closure are handed over to a backend which reifies the kernel code and the
logic for dispatch is not defined.

The code model for the compute will be mutated to inject the appropriate nodes to achieve the goal

It is possible that some vendors may just take the original code model and analyse themselves.

Clearly this is a trivial compute closure.   Lets discuss the required kernel analysis
and proposed pseudo code.

## Copying data based on kernel MemorySegment analysis

Above we showed that we should be able to determine whether a kernel mutates or accesses any of
it's Kernel MemorySegment parameters.

We determined above that the kernel only called set() so we need
not copy the data to the device.

The following example shows a kernel which reads and mutates a memorysegment
```java
    static class Compute {
    @CodeReflection  public static
    void doubleup(Accelerator.NDRange ndrange, MemorySegment memorySegment) {
        int temp = memorySegment.get(JAVA_INT, ndrange.id.x);
        memorySegment.set(JAVA_INT, temp*2);
    }

    @CodeReflection public static
    void compute(Accelerator accelerator, MemorySegment memorySegment, int len) {
        Accelerator.Range range = accelerator.range(len);
        accelerator.run(Compute::doubleup, range, memorySegment);
    }
}
```
Here our analysis needs to determine that the kernel reads and writes to the segment (it does)
so the generated compute model would equate to

```java
 void compute(Accelerator accelerator, MemorySegment memorySegment, int len) {
        Accelerator.Range range = accelerator.range(len);
        accelerator.copyToDevice(memorySegment); // injected via Babylon
        accelerator.run(Compute::doubleup, range, memorySegment);
        accelerator.copyFromDevice(memorySegment); // injected via Babylon
    }
```
So far the deductions are fairly trivial

Consider
```java
 @CodeReflection public static
    void compute(Accelerator accelerator, MemorySegment memorySegment, int len, int count) {
        Accelerator.Range range = accelerator.range(len);
        for (int i=0; i<count; i++) {
            accelerator.run(Compute::doubleup, range, memorySegment);
        }
    }
```

Here HAT should deduce that the java side is merely looping over the kernel dispatch
and has no interest in the memorysegment between dispatches.

So the new model need only copy in once (before the fist kernel) and out once (prior to return)

```java
 @CodeReflection public static
    void compute(Accelerator accelerator, MemorySegment memorySegment, int len, int count) {
        Accelerator.Range range = accelerator.range(len);
        accelerator.copyToDevice(memorySegment); // injected via Babylon
        for (int i=0; i<count; i++) {
            accelerator.run(Compute::doubleup, range, memorySegment);
        }
        accelerator.copyFromDevice(memorySegment); // injected via Babylon
    }
```

Things get slightly more interesting when we do indeed access the memory segment
from the Java code inside the loop.

```java
 @CodeReflection public static
    void compute(Accelerator accelerator, MemorySegment memorySegment, int len, int count) {
        Accelerator.Range range = accelerator.range(len);
        for (int i=0; i<count; i++) {
            accelerator.run(Compute::doubleup, range, memorySegment);
            int slot0 = memorySegment.get(INTVALUE, 0);
            System.out.println("slot0 ", slot0);
        }
    }
```
Now we expect babylon to inject a read inside the loop to make the data available java side

```java
 @CodeReflection public static
    void compute(Accelerator accelerator, MemorySegment memorySegment, int len, int count) {
        Accelerator.Range range = accelerator.range(len);
        accelerator.copyToDevice(memorySegment); // injected via Babylon
        for (int i=0; i<count; i++) {
            accelerator.run(Compute::doubleup, range, memorySegment);
            accelerator.copyFromDevice(memorySegment); // injected via Babylon
            int slot0 = memorySegment.get(INTVALUE, 0);
            System.out.println("slot0 ", slot0);
        }

    }
```

Note that in this case we are only accessing 0th int from the segment so a possible
optimization might be to allow the vendor to only copy back this one element....
```java
 @CodeReflection public static
    void compute(Accelerator accelerator, MemorySegment memorySegment, int len, int count) {
        Accelerator.Range range = accelerator.range(len);
        accelerator.copyToDevice(memorySegment); // injected via Babylon
        for (int i=0; i<count; i++) {
            accelerator.run(Compute::doubleup, range, memorySegment);
            if (i+1==count){// injected
                accelerator.copyFromDevice(memorySegment); // injected
            }else {
                accelerator.copyFromDevice(memorySegment, 1); // injected via Babylon
            }
            int slot0 = memorySegment.get(INTVALUE, 0);
            System.out.println("slot0 ", slot0);
        }

    }
```

Again HAT will merely mutate the code model of the compute method,
the vendor may choose to interpret bytecode, generate bytecode and execute
or take complete control and execute the model in native code.

So within HAT we must find all set/get calls on MemorySegments and trace them back to kernel parameters.

We should allow aliasing of memory segments... but in the short term we may well throw an exception when we see such aliasing


```java
 @CodeReflection  public static
    void doubleup(Accelerator.NDRange ndrange, MemorySegment memorySegment) {
        MemorySegment alias = memorySegment;
        alias.set(JAVA_INT, ndrange.id.x, alias.get(JAVA_INT, ndrange.id.x)*2);
    }
```

## Weed warning #1

We could find common kernel errors when analyzing

This code is probably wrong, as it is racey writing to 0th element

```java
 void doubleup(Accelerator.NDRange ndrange, MemorySegment memorySegment) {
    MemorySegment alias = memorySegment;
    alias.set(JAVA_INT, 0, alias.get(JAVA_INT, ndrange.id.x)*2);
}
```

By allowing a 'lint' like plugin mechanism for code model it would be easy to find.
If we ever find a constant index in set(...., <constant> ) we are probably in a world of hurt.
Unless the set is included in some conditional which itself is dependant on a value extracted from a memory segment.

```java
 void doubleup(Accelerator.NDRange ndrange, MemorySegment memorySegment) {
    MemorySegment alias = memorySegment;
    if (????){
        alias.set(JAVA_INT, 0, alias.get(JAVA_INT, ndrange.id.x) * 2);
    }
}
```

There are a lot opportunities for catching such bugs.


## Flipping Generations

Many algorithms require us to process data from generations. Consider
Convolutions or Game Of Life style problems where we have an image or game state and
we need to calculate the result of applying rules to cells in the image or game.

It is important that when we process the next generation (either in parallel or sequentially) we
must ensure that we only use prev generation data to generate next generation data.

```
[ ][ ][*][ ][ ]       [ ][ ][ ][ ][ ]
[ ][ ][*][ ][ ]       [ ][*][*][*][ ]
[ ][ ][*][ ][ ]   ->  [ ][ ][ ][ ][ ]
[ ][ ][ ][ ][ ]       [ ][ ][ ][ ][ ]

```

This usually requires us to hold two copies,  and applying the kernel to one input set
which writes to the output.

In the case of the Game Of Life we may well use the output as the next input...

```java
@CodeReflection void conway(Accelerator.NDRange ndrange,
                            MemorySegment in, MemorySegment out, int width, int height) {
    int cx = ndrange.id.x % ndrange.id.maxx;
    int cy = ndrange.id.x / ndrange.id.maxx;

    int sum = 0;
    for (int dx = -1; dx < 2; dy++) {
        for (int dy = -1; dy < 2; dy++) {
            if (dx != 0 || dy != 0) {
                int x = cx + dx;
                int y = cy + dy;
                if (x >= 0 && x < widh && y >= 0 && y < height) {
                    sum += in.get(INT, x * width + h);
                }
            }
        }
    }
    result = GOLRules(sum, in.get(INT, ndrange.id.x));
    out.set(INT, ndrange.id.x);

}
```

In this case the assumption is that the compute layer will swap the buffers for alternate passes

```java
import java.lang.foreign.MemorySegment;

@CodeReflection
void compute(Accelerator accelerator, MemorySegment gameState,
             int width, int height, int maxGenerations) {
    MemorySegment s1 = gameState;
    MemorySegment s2 = allocateGameState(width, height);
    for (int generation = 0; generation < maxGenerations; generation++){
        MemorySegment from = generation%2==0?s1?s2;
        MemorySegment to = generation%2==1?s1?s2;
        accelerator.run(Compute::conway, from, to, range, width, height);
    }
    if (maxGenerations%2==1){ // ?
        gameState.copyFrom(s2);
    }
}
```

This common pattern includes some aliasing of MemorySegments that we need to untangle.

HAT needs to be able to track the aliases to determine the minimal number of copies.
```java
import java.lang.foreign.MemorySegment;

@CodeReflection
void compute(Accelerator accelerator, MemorySegment gameState, int width, int height, int maxGenerations,
             DisplaySAM displaySAM) {
    MemorySegment s1 = gameState;
    MemorySegment s2 = allocateGameState(width, height);

    for (int generation = 0; generation < maxGenerations; generation++){
        MemorySegment from = generation%2==0?s1?s2;
        MemorySegment to = generation%2==1?s1?s2;
        if (generation == 0) {             /// injected
            accerator.copyToDevice(from);    // injected
        }                                  // injected
        accelerator.run(Compute::conway, from, to, range, width, height, 1000);
        if (generation == maxGenerations-1){ // injected
            accerator.copyFromDevice(to);    //injected
        }                                    //injected
    }
    if (maxGenerations%2==1){ // ?
        gameState.copyFrom(s2);
    }

}
```

```java
import java.lang.foreign.MemorySegment;

@CodeReflection
void compute(Accelerator accelerator, MemorySegment gameState, int width, int height,
             int maxGenerations,
             DisplaySAM displaySAM) {
    MemorySegment s1 = gameState;
    MemorySegment s2 = allocateGameState(width, height);

    for (int generation = 0; generation < maxGenerations; generation++){
        MemorySegment from = generation%2==0?s1?s2;
        MemorySegment to = generation%2==1?s1?s2;
        accelerator.run(Compute::conway, from, to, range, width, height,1000);
        displaySAM.display(s2,width, height);
    }
    if (maxGenerations%2==1){ // ?
        gameState.copyFrom(to);
    }
}
```



### Example babylon transform to track buffer mutations.

One goal of hat was to automate the movement of buffers from Java to device.

One strategy employed by `NativeBackends` might be to track 'ifaceMappedSegment' accesses and inject tracking data into the compute method.

Here is a transformation for that

```java
 static FuncOpWrapper injectBufferTracking(ComputeClosure.ResolvedMethodCall resolvedMethodCall) {
        FuncOpWrapper original = resolvedMethodCall.funcOpWrapper();
        var transformed = original.transformInvokes((builder, invoke) -> {
                    if (invoke.isIfaceBufferMethod()) { // void array(long idx, T value) or T array(long idx)
                        // Get the first parameter (computeClosure)
                        CopyContext cc = builder.context();
                        Value computeClosure = cc.getValue(original.parameter(0));
                        // Get the buffer receiver value in the output model
                        Value receiver = cc.getValue(invoke.operand(0)); // The buffer we are mutatibg or accessing
                        if (invoke.isIfaceMutator()) {
                            // inject computeContext.preMutate(buffer);
                            builder.op(CoreOps.invoke(ComputeClosure.M_CC_PRE_MUTATE, computeClosure, receiver));
                            builder.op(invoke.op());
                           // inject computeContext.postMutate(buffer);
                            builder.op(CoreOps.invoke(ComputeClosure.M_CC_POST_MUTATE, computeClosure, receiver));
                        } else if ( invoke.isIfaceAccessor()) {
                           // inject computeContext.preAccess(buffer);
                            builder.op(CoreOps.invoke(ComputeClosure.M_CC_PRE_ACCESS, computeClosure, receiver));
                            builder.op(invoke.op());
                            // inject computeContext.postAccess(buffer);
                            builder.op(CoreOps.invoke(ComputeClosure.M_CC_POST_ACCESS, computeClosure, receiver));
                        } else {
                            builder.op(invoke.op());
                        }
                    }else{
                        builder.op(invoke.op());
                    }
                    return builder;
                }
        );
        transformed.op().writeTo(System.out);
        resolvedMethodCall.funcOpWrapper(transformed);
        return transformed;
    }
```

So in our `OpenCLBackend` for example
```java
    public void mutateIfNeeded(ComputeClosure.MethodCall methodCall) {
       injectBufferTracking(entrypoint);
    }

    @Override
    public void computeContextClosed(ComputeContext computeContext){
        var codeBuilder = new OpenCLKernelBuilder();
        C99Code kernelCode = createKernelCode(computeContext, codeBuilder);
        System.out.println(codeBuilder);
    }
```
I hacked the Mandle example. So the compute accessed and mutated it's arrays.

```java
  @CodeReflection
    static float doubleit(float f) {
        return f * 2;
    }

    @CodeReflection
    static float scaleUp(float f) {
        return doubleit(f);
    }

    @CodeReflection
    static public void compute(final ComputeContext computeContext, S32Array2D s32Array2D, float x, float y, float scale) {
        scale = scaleUp(scale);
        var range = computeContext.accelerator.range(s32Array2D.size());
        int i = s32Array2D.get(10,10);
        s32Array2D.set(10,10,i);
        computeContext.dispatchKernel(MandelCompute::kernel, range, s32Array2D, pallette, x, y, scale);
    }
```
So here is the transformation being applied to the above compute

BEFORE (note the !'s indicating accesses through ifacebuffers)
```
func @"compute" (%0 : hat.ComputeContext, %1 : hat.buffer.S32Array2D, %2 : float, %3 : float, %4 : float)void -> {
    %5 : Var<hat.ComputeContext> = var %0 @"computeContext";
    %6 : Var<hat.buffer.S32Array2D> = var %1 @"s32Array2D";
    %7 : Var<float> = var %2 @"x";
    %8 : Var<float> = var %3 @"y";
    %9 : Var<float> = var %4 @"scale";
    %10 : float = var.load %9;
    %11 : float = invoke %10 @"mandel.MandelCompute::scaleUp(float)float";
    var.store %9 %11;
    %12 : hat.ComputeContext = var.load %5;
    %13 : hat.Accelerator = field.load %12 @"hat.ComputeContext::accelerator()hat.Accelerator";
    %14 : hat.buffer.S32Array2D = var.load %6;
!   %15 : int = invoke %14 @"hat.buffer.S32Array2D::size()int";
    %16 : hat.NDRange = invoke %13 %15 @"hat.Accelerator::range(int)hat.NDRange";
    %17 : Var<hat.NDRange> = var %16 @"range";
    %18 : hat.buffer.S32Array2D = var.load %6;
    %19 : int = constant @"10";
    %20 : int = constant @"10";
!   %21 : int = invoke %18 %19 %20 @"hat.buffer.S32Array2D::get(int, int)int";
    %22 : Var<int> = var %21 @"i";
    %23 : hat.buffer.S32Array2D = var.load %6;
    %24 : int = constant @"10";
    %25 : int = constant @"10";
    %26 : int = var.load %22;
 !  invoke %23 %24 %25 %26 @"hat.buffer.S32Array2D::set(int, int, int)void";
    %27 : hat.ComputeContext = var.load %5;
    ...
```
AFTER
```
func @"compute" (%0 : hat.ComputeContext, %1 : hat.buffer.S32Array2D, %2 : float, %3 : float, %4 : float)void -> {
    %5 : Var<hat.ComputeContext> = var %0 @"computeContext";
    %6 : Var<hat.buffer.S32Array2D> = var %1 @"s32Array2D";
    %7 : Var<float> = var %2 @"x";
    %8 : Var<float> = var %3 @"y";
    %9 : Var<float> = var %4 @"scale";
    %10 : float = var.load %9;
    %11 : float = invoke %10 @"mandel.MandelCompute::scaleUp(float)float";
    var.store %9 %11;
    %12 : hat.ComputeContext = var.load %5;
    %13 : hat.Accelerator = field.load %12 @"hat.ComputeContext::accelerator()hat.Accelerator";
    %14 : hat.buffer.S32Array2D = var.load %6;
    invoke %0 %14 @"hat.ComputeClosure::preAccess(hat.buffer.Buffer)void";
!    %15 : int = invoke %14 @"hat.buffer.S32Array2D::size()int";
    invoke %0 %14 @"hat.ComputeClosure::postAccess(hat.buffer.Buffer)void";
    %16 : hat.NDRange = invoke %13 %15 @"hat.Accelerator::range(int)hat.NDRange";
    %17 : Var<hat.NDRange> = var %16 @"range";
    %18 : hat.buffer.S32Array2D = var.load %6;
    %19 : int = constant @"10";
    %20 : int = constant @"10";
    invoke %0 %18 @"hat.ComputeClosure::preAccess(hat.buffer.Buffer)void";
 !   %21 : int = invoke %18 %19 %20 @"hat.buffer.S32Array2D::get(int, int)int";
    invoke %0 %18 @"hat.ComputeClosure::postAccess(hat.buffer.Buffer)void";
    %22 : Var<int> = var %21 @"i";
    %23 : hat.buffer.S32Array2D = var.load %6;
    %24 : int = constant @"10";
    %25 : int = constant @"10";
    %26 : int = var.load %22;
    invoke %0 %23 @"hat.ComputeClosure::preMutate(hat.buffer.Buffer)void";
 !   invoke %23 %24 %25 %26 @"hat.buffer.S32Array2D::set(int, int, int)void";
    invoke %0 %23 @"hat.ComputeClosure::postMutate(hat.buffer.Buffer)void";
    %27 : hat.ComputeContext = var.load %5;
```
And here at runtime the ComputeClosure is reporting accesses when executing via the interpreter after the injected calls.

```
ComputeClosure.preAccess S32Array2D[width()=1024, height()=1024, array()=int[1048576]]
ComputeClosure.postAccess S32Array2D[width()=1024, height()=1024, array()=int[1048576]]
ComputeClosure.preAccess S32Array2D[width()=1024, height()=1024, array()=int[1048576]]
ComputeClosure.postAccess S32Array2D[width()=1024, height()=1024, array()=int[1048576]]
ComputeClosure.preMutate S32Array2D[width()=1024, height()=1024, array()=int[1048576]]
ComputeClosure.postMutate S32Array2D[width()=1024, height()=1024, array()=int[1048576]]
```
## Why inject this info?
So the idea is that the ComputeContext would maintain sets of dirty buffers, one set for `gpuDirty` and one set for `javaDirty`.

We have the code for kernel models. So we know which kernel accesses, mutates or accesses AND mutates particular parameters.

So when the ComputeContext receives  `preAccess(x)` or `preMutate(x)` the ComputeContext would determine if `x` is in the `gpuDirty` set.
If so it would delegate to the backend to  copy the GPU data back from device into the memory segment (assuming the memory is not coherent!)
before removing the buffer from `gpuDirty` set and returning.

Now the Java access to the segment sees the latest buffer.

After `postMutate(x)` it will place the buffer in `javaDirty` set.

When a kernel dispatch comes along, the parameters to the kernel are all checked against the `javaDirty` set.
If the parameter is 'accessed' by the kernel. The backend will copy the segment to device. Remove the parameter
from the `javaDirty` set and then invoke the kernel.
When the kernel completes (lets assume synchronous for a moment) all parameters are checked again, and if the parameter
is known to be mutated by the kernel the parameter is added to the 'gpuDirty' set.

This way we don't have to force the developer to request data movements.

BTW if kernel requests are async ;) then the ComputeContext maintains a map of buffer to kernel.  So `preAccess(x)` or `preMutate(x)` calls
can wait on the kernel that is due to 'dirty' the buffer to complete.

### Marking hat buffers directly. 










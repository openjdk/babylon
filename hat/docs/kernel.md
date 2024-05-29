# Compute and Kernel Analysis with Babylon
#### Gary Frost - January 2024

This is a primer for how we might use Babylon/Code Reflection
to help dispatch code to GPU devices.

----

First lets remind ourself of the structure of a HAT compute class.

```java
static class Squarer {
   @CodeReflection
   public static void
      squareKernel(KernelContext kc, S32Array arr) {
         int value = arr.array(kc.x);          // arr[cc.x]
         s32Array.array(kc.x, value * value);  // arr[cc.x]=value*value
   }

   @CodeReflection
   public static void
      square(ComputeContext cc, S32Array arr) {
         cc.dispatchKernel(
              arr.size,                   // Range for 2d maybe n,m
              kc->squareKernel(kc, arr)   // QuotableKernelContextConsumer
         );                               //    extends Quotable, Consumer<KernelContext>

   }

   public static void main(String args) {
      var lookup = java.lang.invoke.MethodHandles.lookup();
      var backend = Backend::JAVA_MULTITHREADED; // Predicate<Backend>
      var accelerator = new Accelerator(lookup, backend);

      var arr = S32Array.create(accelerator, 100);

      for (int i=0; i<arr.length(); i++){
          arr.array(i,i); // arr[i]=i;
      }
      accelerator.compute(
              cc->Squarer.square(cc,arr)  //QuotableComputeContextConsumer
      );                                  //   extends Quotable, Consumer<ComputeContext>

   }
}
```
Above we show a Compute class `Squarer` with compute entrypoint called `square`
which is responsible for dispatching a simple kernel `squareKernel` over a range `0..arr.len`.

```java
Accelerator accelerator = Accelerator.getGPUAccelerator();
accelerator.execute(computeClosure);
```
----
0. All Compute entrypoints and Kernels (and methods reachable by either), must have the ```@CodeReflection``` annotation
1. The developer should not call either the `Compute.compute()` method or the `Compute.kernel()` methods directly.
2. Instead, we pass the entrypoint  `Compute.compute()` to an `Accelerator` instance which 'conceptually' executes the `compute()`

      Note that the accelerator itself is also used within the compute entrypoint to dispatch the `kernel()` over a range.
3. The Accelerator has full control over the execution

      it is free to decide to execute the bytecode of the `compute()`

      ... or convert that code into some completely different form and interpret it

      ... convert to C99, compile and link to it

      It is not our concern.

4. Kernel methods will only allow a restricted subset of Java.
5. The first kernel argument is always and NDRange reference, which provides the kernels unique dispatch 'identity' (it's id 0..len, it's group, workload dimensions etc)
6. Kernel parameters are restricted to uniform primitive values and Panama FFM `MemorySegment`'s
7. Kernels may not access heap data (no `new`, no `Strings` no virtual dispatch....)
5. Kernels can call other static methods within the `Compute` class.  But the above kernel restrictions will apply to any code reachable from the kernel.
6. Compute antrypoints (such as `compute`) should have very few restrictions, although some code patterns may hamper performance.
7. Compute entrypoints alway receives an Accelerator as it's first arg.
----
## How do we use *babylon*?

1. We 'close over' the call/dispatch graph from the entrypoint to all kernels and collect the kernels reachable from the entrypoint and all methods reachable from methods reachable by kernels.
2. We essentially end up with a graph of codemodels 'rooted' at the entrypoint
2. For each kernel we also determine how the kernel accesses it's 'MemorySegment` parameters, for each MemorySegment parameters we keep a side table of whther the kernel reads or writes to the segment. We keep this infomation in a side map.

This resulting 'ComputeClosure' (tree of codemodels and relevant side tables) is made available to the accelerator to coordinate execution.

Note that our very simple Compute::compute method neither expresses the movement of the MemorySegment to a device, or the retrieval of the data from a device when the kernel has executed.

Our assumption is that given the ComputeClosure we can deduce such movements.
----
There are many ways to achieve this.  One way would be by static analysis.

Given the Compute::compute entrypoint it is easy to determine that we are always (no conditional or loops) passing (making available
might be a better term) a memory segment to a kernel (Compute::kernel) and this kernel only mutates the  `MemorySegment`.


So from simple static analysis we could choose to inject one or more calls into the model representing the need for the accelerator to move data to the devices and/ord back from the device, after the kernel dispatch.

----

This modified model, would look like we had presented it with this code.

```java
 void compute(Accelerator accelerator, MemorySegment memorySegment, int len) {
        Accelerator.Range range = accelerator.range(len);
        accelerator.run(Compute::kernel, range, memorySegment);
        accelerator.injectedCopyFromDevice(memorySegment);
    }
```

Note the ```injectedCopyFromDevice()``` call.

----

Because the kernel does not read the `MemorySegment` we only need inject the code to request a move back from the device.

To do this requires HAT to analyse the kernel(s) and inject appropriate code into
the Compute::compute method to inform the vendor backend when it should perform such moves.
----
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
----

Whether this code mutation generates Java bytecode and executes (or interprets) on the JVM or whether the
CodeModels for the closure are handed over to a backend which reifies the kernel code and the
logic for dispatch is not defined.

The code model for the compute will be mutated to inject the appropriate nodes to achieve the goal

It is possible that some vendors may just take the original code model and analyse themselves.

----

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
----
So far the deductions are fairly trivial

Consider
```java
 @CodeReflection public static
    void compute(Accelerator accelerator, MemorySegment memorySegment, int len, int count) {
        Accelerator.Range range = accelerator.range(len);
        for (int i=0; i&lt;count; i++) {
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
        for (int i=0; i&lt; count; i++) {
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
        for (int i=0; i&lt;count; i++) {
            accelerator.run(Compute::doubleup, range, memorySegment);
            int slot0 = memorySegment.get(INTVALUE, 0);
            System.out.println("slot0 ", slot0);
        }
    }
```
Now we expect babylon to inject a read inside the loop to make the data available java side
----
```java
 @CodeReflection public static
    void compute(Accelerator accelerator, MemorySegment memorySegment, int len, int count) {
        Accelerator.Range range = accelerator.range(len);
        accelerator.copyToDevice(memorySegment); // injected via Babylon
        for (int i=0; i&lt;count; i++) {
            accelerator.run(Compute::doubleup, range, memorySegment);
            accelerator.copyFromDevice(memorySegment); // injected via Babylon
            int slot0 = memorySegment.get(INTVALUE, 0);
            System.out.println("slot0 ", slot0);
        }

    }
```
----
Note that in this case we are only accessing 0th int from the segment so a possible
optimization might be to allow the vendor to only copy back this one element....
```java
 @CodeReflection public static
    void compute(Accelerator accelerator, MemorySegment memorySegment, int len, int count) {
        Accelerator.Range range = accelerator.range(len);
        accelerator.copyToDevice(memorySegment); // injected via Babylon
        for (int i=0; i&lt;count; i++) {
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
----
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
----
#### Weed warning #1

We could find common kernel errors when analyzing

This code is probably wrong, as it is racey writing to 0th element

```java
 void doubleup(Accelerator.NDRange ndrange, MemorySegment memorySegment) {
    MemorySegment alias = memorySegment;
    alias.set(JAVA_INT, 0, alias.get(JAVA_INT, ndrange.id.x)*2);
}
```
----

By allowing a 'lint' like plugin mechanism for code model it would be easy to find.
If we ever find a constant index in set(...., &lt;constant&gt; ) we are probably in a world of hurt.
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
----

### Flipping Generations

Many algorithms require us to process data from generations. Consider
Convolutions or Game Of Life style problems where we have an image or game state and
we need to calculate the result of applying rules to cells in the image or game.

It is important that when we process the next generation (either in parallel or sequentially) we
must ensure that we only use prev generation data to generate next generation data.

```
[ ][ ][*][ ][ ]       [ ][ ][ ][ ][ ]
[ ][ ][*][ ][ ]       [ ][*][*][*][ ]
[ ][ ][*][ ][ ]   -&gt;  [ ][ ][ ][ ][ ]
[ ][ ][ ][ ][ ]       [ ][ ][ ][ ][ ]

```
----
This usually requires us to hold two copies,  and applying the kernel to one input set
which writes to the output.

In the case of the Game Of Life we may well use the output as the next input...

```java
@CodeReflection void conway(Accelerator.NDRange ndrange,
                            MemorySegment in, MemorySegment out, int width, int height) {
    int cx = ndrange.id.x % ndrange.id.maxx;
    int cy = ndrange.id.x / ndrange.id.maxx;

    int sum = 0;
    for (int dx = -1; dx &lt; 2; dy++) {
        for (int dy = -1; dy &lt; 2; dy++) {
            if (dx != 0 || dy != 0) {
                int x = cx + dx;
                int y = cy + dy;
                if (x&gt;= 0 && x &lt; width && y &gt;= 0 && y &lt; height) {
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
----
```java
import java.lang.foreign.MemorySegment;

@CodeReflection
void compute(Accelerator accelerator, MemorySegment gameState,
             int width, int height, int maxGenerations) {
    MemorySegment s1 = gameState;
    MemorySegment s2 = allocateGameState(width, height);
    for (int generation = 0; generation &lt; maxGenerations; generation++){
        MemorySegment from = generation%2==0?s1?s2;
        MemorySegment to = generation%2==1?s1?s2;
        accelerator.run(Compute::conway, from, to, range, width, height);
    }
    if (maxGenerations%2==1){ // ?
        gameState.copyFrom(s2);
    }
}
```
----
This common pattern includes some aliasing of MemorySegments that we need to untangle.

HAT needs to be able to track the aliases to determine the minimal number of copies.
```java
import java.lang.foreign.MemorySegment;

@CodeReflection
void compute(Accelerator accelerator, MemorySegment gameState, int width, int height, int maxGenerations,
             DisplaySAM displaySAM) {
    MemorySegment s1 = gameState;
    MemorySegment s2 = allocateGameState(width, height);

    for (int generation = 0; generation &lt; maxGenerations; generation++){
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
----
```java
import java.lang.foreign.MemorySegment;

@CodeReflection
void compute(Accelerator accelerator, MemorySegment gameState, int width, int height,
             int maxGenerations,
             DisplaySAM displaySAM) {
    MemorySegment s1 = gameState;
    MemorySegment s2 = allocateGameState(width, height);

    for (int generation = 0; generation &lt; maxGenerations; generation++){
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
----




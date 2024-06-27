
# HAT's Programming Model 
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

#  HAT's Programming model

Let's consider a trivial opencl kernel which squares each element in an int buffer

```java
int square(int value){
    return value*value;
}

__kernel void squareKernel( __global int* s32Array){
    int value = s32Array[get_global_id(0)];
    s32Array[get_global_id(0)]=square(value);
    return;
}

```

We implement this in HAT by collecting the kernel(s) and compute method(s) in a `Compute` class.

```java
public class SquareCompute {
    @CodeReflection
    public static int square(int v) {
        return v * v;
    }

    @CodeReflection
    public static void squareKernel(KernelContext kc, S32Array s32Array) {
        int value = s32Array.array(kc.x);     // arr[cc.x]
        s32Array.array(kc.x, square(value));  // arr[cc.x]=value*value
    }

    @CodeReflection
    public static void square(ComputeContext cc, S32Array s32Array) {
        cc.dispatchKernel(s32Array.length(),
                kc -> squareKernel(kc, s32Array)
        );
    }
}
```
And we dispatch by creating the appropriate data buffer and then asking an `Accelerator` (bound to a typical vendor backend) to execute the compute method.. which in turn coordinates the dispatch of the various kernels.

```java
  // Create an accelerator bound to a particular backend
 
  var accelerator = new Accelerator(
      java.lang.invoke.MethodHandles.lookup(),
      Backend.FIRST  // Predicate<Backend>
  );
  
  // Ask the accelerator/backend to allocate an S32Array
  var s32Array = S32Array.create(accelerator, 32);
  
  // Fill it with data 
  for (int i = 0; i < s32Array.length(); i++) {
      s32Array.array(i, i);
  }
  
  // Tell the accelerator to execute the square() compute entrypoint
   
  accelerator.compute(
     cc -> SquareCompute.square(cc, s32Array) 
  );
  
  // Check the data                                    
  for (int i = 0; i < arr.length(); i++) {
      System.out.println(i + " " + arr.array(i));
  }
```

## Programming model notes 

The most important concept here is that we separate `normal java` code, 
from `compute` code from `kernel` code

We must not assume that Compute or Kernel code are ever executed by the JVM

### Kernel Code (kernel entrypoints and kernel reachable methods)
Kernel's and any kernel reachable methods will naturally be restricted to subset of Java.

* No exceptions (no exceptions! :) )
* No heap access (no `new`)
* No access to static or instance fields from this or any other classes )
    * Except `final static primitives` (which generally get constant pooled) 
    * Except fields of `KernelContext` (thread identity `.x`, `.maxX`, `.groups`... )
        - We may even decide to access these via methods (`.x()`);
* The only methods that can be called are either :-
   * Kernel reachable methods
      - Technically you can call a kernel entrypoint, but must pass your KernelContext
   * `ifaceMappedSegment` accessor/mutators (see later)
   * Calls on `KernelContext` (backend kernel features)
     - `KernelContext.barrier()`
     - `kernelContext.I32.hypot(x,y)`
#### Kernel Entrypoints 
* Declared `@CodeReflection static public void`
    * Later we may allow reductions to return data...
* Parameters
    * 0 is always a `KernelContext` (KernelContext2D, KernelContext3D logically follow)
    * 1..n are restricted to uniform primitive values and Panama FFM `ifaceMappedSegments`

#### Kernel Reachable Methods
* Declared `@CodeReflection static public`
* All Parameters are restricted to uniform primitive values and Panama FFM `ifaceMappedSegments`

### Compute Code (Compute entry points and compute reachable methods)
Code within the `compute entrypoint` and `compute reachable 
methods` have much fewer Java restrictions than kernels but generally...

* Exceptions are discouraged
* Java Synchronization is discouraged
* Don't assume any allocation of local `ifaceMappedSegmants` are allocated
* Java accesses/mutations to `ifaceMappedSegment` will likely impact performance
* Code should ideally just contain simple control flow and kernel dispatches.
* Data movements (to and from backend) will automatically be derived from control flow and `ifaceMappedSegment` accesses
   - We hope to never have to add `cc.moveToDevice(hatBuffer)`
* All methods reachable from a `compute entrypoint` are either :-
  * Compute Reachable Methods
      - Technically methods can be compute reachable and kernel reachable.
  * `ifaceMappedSegment` accessor/mutators (see later)
  * Calls on the `ComputeContext` to generate ranges, or dispatch kernels.
      
#### Compute Entry Points
* Declared `@CodeReflection static public void`
* Parameter 0 is `ComputeContext`


#### Compute Reachable Methods
* Declared `@CodeReflection static public `
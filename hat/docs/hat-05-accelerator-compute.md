# What happens when we call accelerator.compute(lambda)

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

# What happens when we call accelerator.compute(lambda)

# Back to our Squares example.

So what is going on here?

```java
  accelerator.compute(
     cc -> SquareCompute.square(cc, s32Array)
  );
```

Recall we have two types of code in our compute class. We have kernels (and kernel reachable methods) and we have
compute entrypoints (and compute reachable methods).

```java
public class SquareCompute{
    @CodeReflection public static int square(int v) {
        return  v * v;
    }

    @CodeReflection public static void squareKernel(KernelContext kc, S32Array s32Array) {
        int value = s32Array.array(kc.x);     // arr[cc.x]
        s32Array.array(kc.x, square(value));  // arr[cc.x]=value*value
    }

    @CodeReflection public static void square(ComputeContext cc, S32Array s32Array) {
        cc.dispatchKernel(s32Array.length(),
                kc -> squareKernel(kc, s32Array)
        );
    }
}
```

AGAIN.... NOTE that we cannot just call the compute entrypoint or the kernel directly.

```java
  SquareCompute.square(????, s32Array);  // We can't do this!!!!
```

We purposely make it inconvenient (ComputeContext and KernelContext construction is embedded in the framwork) to
mistakenly call the compute entrypoint directly.  Doing so is akin to calling `Thread.run()` directly, rather than
calling `Thread.start()` on a class extending `Thread` and providing an implementation of `Thread.run()`

Instead we use this pattern

```java
  accelerator.compute(
     cc -> SquareCompute.square(cc, s32Array)
  );
```

We pass a lambda to `accelerator.compute()` which is used to determine which compute method to invoke.

```
 User  |  Accelerator  |  Compute  |  Babylon  |        Backend            |
                          Context                 Java     C++     Vendor
+----+   +-----------+   +-------+   +-------+   +----+   +---+   +------+
|    |   |           |   |       |   |       |   |    |   |   |   |      |
+----+   +-----------+   +-------+   +-------+   +----+   +---+   +------+
    +--------> accelerator.compute(lambda)

```

Incidently, this lambda is never executed by Java JVM ;) instead, the accelerator uses Babylon's Code Reflection
capabilities to extract the model of this lambda to determine the compute entrypoint and it's captured args.

```
 User  |  Accelerator  |  Compute  |  Babylon  |        Backend            |
                          Context                 Java     C++     Vendor
+----+   +-----------+   +-------+   +-------+   +----+   +---+   +------+
|    |   |           |   |       |   |       |   |    |   |   |   |      |
+----+   +-----------+   +-------+   +-------+   +----+   +---+   +------+
    +--------> accelerator.compute( cc -> SquareCompute.square(cc, s32Array) )
                ------------------------->
                    getModelOf(lambda)
                <------------------------
```

This model describes the call that we want the accelerator to
execute or interpret (`SquareCompute.square()`) and the args that were captured from the call site (the `s32Array` buffer).

The accelerator uses Babylon again to get the
code model of `SquareCompute.square()` builds a ComputeReachableGraph with this method at the root.
So the accelerator walks the code model and collects the methods (and code models) of all methods
reachable from the entrypoint.

In our trivial case, the ComputeReachableGraph has a single root node representing the `SquareCompute.square()`.

```
 User  |  Accelerator  |  Compute  |  Babylon  |        Backend            |
                          Context                 Java     C++     Vendor
+----+   +-----------+   +-------+   +-------+   +----+   +---+   +------+
|    |   |           |   |       |   |       |   |    |   |   |   |      |
+----+   +-----------+   +-------+   +-------+   +----+   +---+   +------+
    +--------> accelerator.compute( cc -> SquareCompute.square(cc, s32Array) )
                ------------------------->
                     getModelOf(lambda)
                <------------------------
                ------------------------->
                     getModelOf(SquareCompute.square())
                <-------------------------
          forEachReachable method in SquareCompute.square() {
                ------------------------->
                     getModelOf(method)
                <------------------------
                add to ComputeReachableGraph
          }
```

The Accelertor then walks through the ComputeReachableGraph to determine which kernels are referenced..

For each kernel we extract the kernels entrypoint (again as a Babylon
Code Model) and create a KernelReachableGraph for each kernel.  Again by starting
at the kernel entrypoint and closing over all reachable methods (and Code Models).

We combine the compute and kernel reachable graphs and create an place them in a  `ComputeContext`.

This is the first arg that is 'seemingly' passed to the Compute class. Remember the compute
entrypoint is just a model of the code we expect to
execute. It may never be executed by the JVM.

```
 User  |  Accelerator  |  Compute  |  Babylon  |        Backend            |
                          Context                 Java     C++     Vendor
+----+   +-----------+   +-------+   +-------+   +----+   +---+   +------+
|    |   |           |   |       |   |       |   |    |   |   |   |      |
+----+   +-----------+   +-------+   +-------+   +----+   +---+   +------+

          forEachReachable kernel in ComputeReachableGraph {
                ------------------------->
                      getModelOf(kernel)
                <------------------------
                add to KernelReachableGraph
          }
          ComputeContext = {ComputeReachableGraph + KernelReachableGraph}

```

The accelerator passes the ComputeContext to backend (`computeContextHandoff()`), which will typically take
the opportunity to inspect/mutate the compute and kernel models and possibly build backend specific representations of
kernels and compile them.

The ComputeContext and the captured args are then passed to the backend for execution.

```
 User  |  Accelerator  |  Compute  |  Babylon  |        Backend            |
                          Context                 Java     C++     Vendor
+----+   +-----------+   +-------+   +-------+   +----+   +---+   +------+
|    |   |           |   |       |   |       |   |    |   |   |   |      |
+----+   +-----------+   +-------+   +-------+   +----+   +---+   +------+


                ----------------------------------->
                    computeContextHandoff(computeContext)
                                                    ------->
                                                             ------->
                                                         compileKernels()
                                                             <------
                                                      mutateComputeModels
                                                    <-------
                    dispatchCompute(computeContext, args)
                                                    ------->
                                                        dispatchCompute(...)
                                                            --------->
                                                               {
                                                               dispatchKernel()
                                                               ...
                                                               }
                                                            <--------
                                                    <------
                <----------------------------------

```

----
### Notes

In reality. The Accelerator receives a `QuotableComputeContextConsumer`

```java
   public interface QuotableComputeContextConsumer
        extends Quotable,
        Consumer<ComputeContext> {
    }
```
Here is how we extract the 'target' from such a lambda

```java
 public void  compute(QuotableComputeContextConsumer qccc) {
   Quoted quoted = qccc.quoted();
   LambdaOpWrapper lambda = OpTools.wrap((CoreOps.LambdaOp)quoted.op());

   Method method = lambda.getQuotableComputeContextTargetMethod();

   // Get from the cache or create a compute context which closes over compute entryppint
   // and reachable kernels.
   // The models of all compute and kernel methods are passed to the backend during creation
   // The backend may well mutate the models.
   // It will also use this opportunity to generate ISA specific code for the kernels.

   ComputeContext = this.cache.computeIfAbsent(method, (_) ->
           new ComputeContext(this/*Accelerator*/, method)
   );

   // Here we get the captured args from the Quotable and 'jam' in the computeContext in slot[0]
   Object[] args = lambda.getQuotableComputeContextArgs(quoted, method, computeContext);
   this.compute(computeContext, args);
}
```


# Hat Update 
#### Gary Frost - Apr 2024
Openjdk Babylon is here https://github.com/openjdk/babylon

First lets remind ourself of the structure of a HAT compute class circa JMLS 2023

```java
static class Squarer {
   @CodeReflection public static void 
      squareKernel(NDRange id, S32Array arr) {
         int value = arr.array(id.x);          // arr[cc.x]
         s32Array.array(id.x, value * value);  // arr[cc.x]=value*value
   }

   @CodeReflection public static void 
      square(Accelerator  acc, S32Array arr) {
         var range = acc.range(arr.size);
         acc.dispatcKernel(Compute::square, range, arr);
   }

   public static void main(String args) {
      var lookup = java.lang.invoke.MethodHandles.lookup();
      var backend = Backend::JAVA_MULTITHREADED; // Predicate<Backend>
      var accelerator = new Accelerator(lookup, backend);
      
      var arr = S32Array.create(accelerator, 100);  
      //arr.copyFrom(int[] );?
      for (int i=0; i<arr.length(); i++){
          arr.array(i,i); // arr[i]=i;
      }
      accelerator.compute(Squarer::square, arr);
      //arr.copyTo(int[] 
   }
}
```
Present API

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

### Kernel and Compute method constraints

#### Compute entrypoints and compute reachable functions

1. Are `@CodeReflection static public void` 
2. `compute entrypoinst` have a first arg of `ComputeContext`
3. All methods reachable from a `compute entrypoint` are either :-
   * `@CodeReflection static public` compute reachable functions
   * `ifaceMappedSegment` accessor/mutators (see later) 
   * Calls on the `ComputeContext` to generate ranges, or dispatch kernels.
4. The developer should not call  `Square.square()` directly, and we 
   make it hard for them to get a `ComputeContext` to stop them accidently doing so.
5. The bound trio of `Accelerator/Backend/ComputeContext` :-
   - Has full control over the execution from this point on
   - Is free to decide how to execute the bytecode of the `compute entrypoints`, `reachable compute methods` and `kernel dispatches`
   - It may execute the bytecodeconvert that code into some completely different form and interpret it
   - It may interpret the babylon model
   - It may mutate the babylon model and interpret
   - It may convert to C99, compile and link to it
   - ......

6. Code within the `compute entrypoint` and compute reachable 
functions have a few real Java restrictions.
   * Exceptions are discouraged. 
   * Allocation of local `ifaceMappedSegmants` may not actually need to be allocated 
   * Java accesses/mutations to `ifaceMappedSegment` will likely impact performance
   * But code should ideally be simple control of kernel dispatches. 
   * Data movements (to and from backend) are derived from control flow and `ifaceMappedSegment` accesses

#### Kernels and kernel reachable functions

1. Kernels are declared `@CodeReflection static public void`
2. For Kernels, parameter 0 is always a `KernelContext`  (maybe `KernelContext2D`?...`3D`.)
3. All methods reachable from a `kernel` are either :-
   * `@CodeReflection static public` kernel reachable functions
   * `ifaceMappedSegment` accessor/mutators (see later)
   * Calls on `KernelContext` (backend kernel features)...

5. Kernel parameters 1..n are restricted to uniform primitive values and Panama FFM `ifaceMappedSegments`
7. Kernel's and any kernel reachable functions
      (with any expectation of backend execution) will naturally be restricted to subset of Java.
   * must not access heap data (no `new`) 
   * must not access fields (except `final static primitive` from this or any other classes `*` )
     * Except fields of `KernelContext` (thread identity `.x`, `.maxX`, `.groups`... )
       - maybe even this access should be via methods?
5. Kernels can 
   * call other methods within the same class provided those methods also follow the rules described here

---
### IfaceMappedSegments

Earlier we stated that Kernel parameters 1..n are restricted to `uniform primitive args` and `ifaceMappedSegments`

At JVMLS 2023 I demonstrated a mechanism which used `Proxy` to bind interfaces (and nestings thereof) to memory 
segments to allow us to more easily access MemorySegments from the Java side AND to ease the mapping on the kernel side C99 structs. 

Recently Maurizio and Per implemented a much more performant implementation of this concept using the recent `Classfile` API 

In the above code we had
```java
 var arr = S32Array.create(accelerator, 100);
```

Here `S32Array` is *just* (?) an interface 
```java
public interface S32Array extends Array1D {
    static S32Array create(Accelerator accelerator, int length) {
        return Array1D.create(accelerator, S32Array.class, length, JAVA_INT);
    }
    int array(long idx);
    void array(long idx, int f);
}
```
Where `Array1D` extends the base marker class `Buffer` :-
```java
public interface Array1D extends Buffer {
    static <T extends Array1D> T create(Accelerator accelerator, Class<T> clazz, int length, MemoryLayout memoryLayout) {
       T buffer = SegmentMapper.of(accelerator.lookup, clazz,
                       JAVA_INT.withName("length"),
                       MemoryLayout.sequenceLayout(length, memoryLayout).withName("array")
       ).allocate(accelerator.arena);
        buffer.length(length);
        return buffer;
    }
    int length();
    void length(int length);
}
```

The `magic code` is in `SegmentMapper.of()` 

```java
//Where in this case 
    Class clazz = S32Array.getClass();
    java.lang.foreign.MemoryLayout memoryLayout= JAVA_INT;
    int length = 100; 
 
    SegmentMapper.of(accelerator.lookup, clazz,
                 JAVA_INT.withName("length"),
                 MemoryLayout.sequenceLayout(length, memoryLayout).withName("array")
       ).allocate(accelerator.arena);
    buffer.length(length);
```
`SegmentMapper.of(...)` takes an interface (`S32Array`) and a layout description.  In this case the layout is equiv to 

```C
typedef struct S32Array_s{
    int length;
    int array[1000]
}S32Array;
```

Note we have two accessor/mutate methods provided by the base `Array1D` interface
```java
 int length();
 void length(int length);
```
And two from the derived `S32Array` interface
```java
int array(long idx);
void array(long idx, int f);
```

So in total we have four unimplemented methods

```java
int length();
void length(int length);
int array(long idx);
void array(long idx, int f);
```

The `SegmentMapper` uses the `ClassFile` API to spin up an implementation of this interface.

This class's bytecode provides an implementation of these four methods.

Basically for any `T XXX()` method it creates an implementation `T XXX()` which uses a `VarHandle` to 
access the given offset of the layout member of Type `T` named `XXX` in the underlying segment, 
which when called will indeed return the in-segment data as a `T`

Similarly for `void XXX(T xxx)` it implements a method (again using `VarHandle`) 
to set the underlying segment data at the offest determined by the matching `XXX` layout

The upshot of which is we can now use 
```java
s32Array.length();  
```
This is much more appealing than manually binding a class or record via `VarHandles` to layout offsets from the
underlying layout and delegating to appropriate peeks/pokes via the MemorySegment API's

If the layout matching `XXX` such as `void XXX(long offset, T value)` is found, it binds a setter which sets the value at 
the appropriately `strided` element offset.  

Similarly it implements a `getter` as `T XXX(long offset)`.

So we can set/get elements on the array 
```Java 
s32Array.array(23, 42);        // int[23]=42; 
assert s32Array.array(23)==42  
```

I modified the SegmentMappers implementation to we can access the underlying segment ptr (which we can pass to native backends) 
an provided a method for accessing a text description `schema` of the now bound layout. 

We can rely on this to extract native ptrs inthe backend, and to offer a description of the layout which can be turned into C99 receiving structs/layout in the kernel.

```java
 var arr = S32Array.create(accelerator, 100);
 System.out.print(arr.schema());
```
Would yield the schema string form => `S32Array:{length:i4,array:[100:?:s4]}`

This a very simple mapping. Probably the minimal useful mapping.  The HaarCascade example has multiple nested ifaces representing nested embedded structs and unions.

```java
   XMLHaarCascadeModel haarCascade = XMLHaarCascadeModel.load(
        ViolaJonesRaw.class.getResourceAsStream("/cascades/haarcascade_frontalface_default.xml"));
   assert haarCascade instanceof Cascade; // See it is just an interface, but not segment bound...
   Cascade cascade = Cascade.create(accelerator, haarCascade);
   System.out.print(cascade.schema());
```
Here the schema includes nested structs, unions, padding and arrays of structs.  I added indenting to make the schema  more readable.
```
Cascade:{                      <-- curly brace=struct
   width:i4,
   height:i4,
   featureCount:i4,
   feature:[2913:Feature:{    <-- an array called feature of 2913 Feature structs   
      id:i4,
      threshold:f4,
      left:{
          hasValue:z1,
          ?:x7,                <-- padding of 7 bytes to align 
          anon:<               <-- chevrons desxcribe union 
             featureId:i4|
             value:f4
          >
      },
      right:{
          hasValue:z1,
          ?:x7,
          anon:<
              featureId:i4|
              value:f4
           >
      },
      rect:[3:Rect:{
         x:b1,
         y:b1,
         width:b1,
         height:b1,
         weight:f4
      }
    ]
  }],
  stageCount:i4,
  stage:[25:Stage:{
     id:i4,
     threshold:f4,
     firstTreeId:s2,
     treeCount:s2
   }],
   treeCount:i4,
   tree:[2913:Tree:{
      id:i4,
      firstFeatureId:s2,
      featureCount:s2
   }]
}
```

### Workflow.
## So what happens when we call compute using the new API.

```java
  accelerator.compute(
        cc->Squarer.square(cc,arr)  //Quotable Consumer<ComputeContext>
      );
```

Note that `Accelerator.compute()` takes a lambda which seemingly executes the `Square.square()` method. 


In reality. The Accelerator receives a `QuotableComputeContextConsumer`

```java 
   public interface QuotableComputeContextConsumer
        extends Quotable,
        Consumer<ComputeContext> {
    }
```
From this we can access the babylon model of the quotable lambda (with the call to `Squarer.square(cc,arr)`)
along with it's  captured args. 

From this `Compute Entrypoint` can construct a ComputeContext containing a closure over all 
compute reachable, kernels and kernel reachable methods.

The Backend gets passed this closure first and may mutate models. 
It also can construct ISA specific kernels and map them to the methods in the closure.  


Finally we pass the ComputeContext and the captured args to the backend to execute/interpret_ 
```java
 public void  compute(QuotableComputeContextConsumer qccc) {
   Quoted quoted = qccc.quoted();
   LambdaOpWrapper lambda = OpTools.wrap((CoreOps.LambdaOp)quoted.op());
   
   // This instead of cracking the Lambda....
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

The dispatch of kernels is similar.  But this time completely under the control of the backend.   

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
        if (methodCall instanceof ComputeClosure.Entrypoint entrypoint){
            injectBufferTracking(entrypoint);
        }else{
            System.out.println("OpenCL backend declined to mutate "+ methodCall + methodCall.method);
        }
    }

    @Override
    public void computeContextClosed(ComputeContext computeContext){
        System.out.println("OpenCL backend received closed closure");
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

### An alternative to pre/pos access/mutate
# TODO -   
Explain how we can spin up classes to hold onto iface buffers then pass these holders around, which maintain dirty state. 




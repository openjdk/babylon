
# Interface Mapping

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
# Interface Mapping

## or ... HAT from a Data POV

### Or ... what is this `S32Array` thing and why can't I just pass `int[]` to my kernel

Again here is the canonical HAT 'hello world' kernel, weill use this to describe itgerface mapping

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
Which we dispatch by creating the appropriate data buffer and then asking an `Accelerator` (bound to a typical vendor backend) to execute the compute method.. which in turn coordinates the dispatch of the various kernels.

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

HAT kernels only accept Java primitives and HAT buffers as parameters.

We don't directly support heap allocated data (such as int[])

From Java's point of view `S32Array` is a `hat.Buffer` and is defined as an interface.

```java
public interface S32Array extends Buffer {
    int length();
    void length(int i);
    int array(long idx);
    void array(long idx, int i);
}
```

From C99 style OpenCL/CUDA POV this will eventually be mapped to a typedef.

```C++
typedef struct S32Array_s{
    int length;
    int array[];  //<-- ?
}S32Array_t;
```

Our Java implementations should treat the interface as `data`, generally the only
methods that we include in a `hat.Buffer` should be

```java
T name();                    //getter for a field called name with type T, where T may be primitive or inner interface)
void name(T name);           //setter for a field called name with type T, T must be  primitive
T name(long idx);            //get an array element [idx] where array is called name and T is either primitive or inner interface
void name(long idx, T name); //set an array element [idx] where array is called name and T is primitive
```

Algorithms can assume that an interface is 'bound' to 'some' concrete data layout.

We could for example implement `S32Array` like this.

```java
class JavaS32Array implements S32Array{
     int[] arr;
     int length(){ return arr.length;}
     int array(long idx) {return arr[idx];}
     void array(long idx, int value) {arr[idx] = value;}
     void length(int len) ; // we'll come back to this ;)
}
```

But for HAT to access native memory, allocated by the appropriate backend we need interfaces bound to MemorySegents/

HAT includes an API which allows us to take an interface which extends `hat.Buffer`, and 'bind' it to a Panama FFM MemorySegment.

This binding process automatically maps the accessors (for example `length()`, `array(long idx, int v)`) to low level Method and Var handel trickery underlying MemorySegments.

Conceptually we might imagine that HAT creates something like this

```java
class PanamaS32Array implements S32Array{
     MemorySegment segment;
     final int SIZEOFINT = 4;
     final long lenOffset = 0;
     final long arrayOffset = lenOffset+SIZEOFINT;
     int length(){ return segment.getInt(lenOffset);}
     int array(long idx) {return segment.getInt(arrayOffset+idx*SIZEOFINT);}
     void array(long idx, int value) {segment.setInt(arrayOffset+idx*SIZEOFINT,value);}
     void length(int len) ; // we'll come back to this ;)
}
```

Much like Java's `Proxy` class, the iface mapper creates an implementation of the interface  'on the fly', the new Classfile API is used to 'spin up' the new class and the accessors are are composed using Var/Method Handles and offsets derived from the size and order of fields.

Sadly an interface is not quite enough to establish exactly what is needed to complete the mapping.  We need to tell the `iface mapper` the order and size of fields and possibly some padding information.

We do this by providing a 'layout description' using Panama's Layout api.

```java
MemoryLayout s32ArrayLayout = MemoryLayout.structLayout(
        JAVA_INT.withName("length"),
        MemoryLayout.sequenceLayout(N, JAVA_INT.withName("length")).withName("array")
).withName(S32Array.getSimpleName());
```

Eventually we came to a common pattern for describing HAT buffers by adding a `create` method to our interface which hides the mapping detail

So the complete `S32Array` looks a like this. (....ish)

```java
public interface S32Array extends Buffer {
    int length();

    void length(int i);

    int array(long idx);

    void array(long idx, int i);

    S32Array create(Accelerator a, int len) {
        MemoryLayout s32ArrayLayout = MemoryLayout.structLayout(
                JAVA_INT.withName("length"),
                MemoryLayout.sequenceLayout(len, JAVA_INT.withName("length")).withName("array")
        ).withName(S32Array.getSimpleName());

        S32Array s32Array = a.allocate(
                SegmentMapper.of(MethodHandles.lookup(), S32Array.class, s32ArrayLayout, len)
        );

        return s32Array;
    }
}
```

So now hopefully this code makes more sense.

```
var s32Array = S32Array.create(accelerator, 32);
```

Whilst this code is much nicer than hand mapping each method to offsets.  It is still quite verbose.

In the last few weeks we have been migrating to Schema builder which makes this code easier to express..

```java
public interface S32Array extends Buffer {
    int length();
    void length(int i);
    int array(long idx);
    void array(long idx, int i);
    Schema<S32Array> schema = Schema.of(S32Array.class, s->s
        .arrayLen("length")
        .array("array")
    );
}
```
The schema is embedded inside the interface and defines the order of fields. It also allows us to bind fields to each other (above we are telling the schema we have a `int length` field followed by an `int array[]` field and that the first defines the size of the second), we also can describe useful 'HAT' information for fields. Such as whether a field is 'atomic' ;)

Here is an example of a table of Results for the face detector.

```java
public interface ResultTable extends Buffer{
    interface Result extends Buffer.StructChild {
        float x();
        void x(float x);
        float y();
        void y(float y);
    }
    void count(int count);
    int count();
    int length();
    Result result(long idx);

    Schema<ResultTable> schema = Schema.of(ResultTable.class, s->s
            .atomic("count")
            .arrayLen("length")
            .array("result", r->r
               .field("x")
               .field("y")
            )
    );
}
```

Which in C99 OpenCL code will manifest as

```C++
typedef Result_s{
   int x,y
} Result_t;

typedef ResultTable_s{
   int count;
   int length;
   Result_t result[0];
} Result_t;
```

In our Java code this interface makes access to MemorySegments much cleaner

```java
    ResultTable resultTable = ResultTable.create(acc, 100);
    for (int i=0; i<resultTable.length(); i++){
        Result result = resultTable.result(i);
        result.x(0);
        result.y(0);
    }
```

The generated OpenCL/C99 code from Java kernel code is also quite clean

We might use a kernel to initialize the location of a bunch of Results

```java
    @CodeReflection public static void init(KernelContext kc, ResultTable resultTable) {
        if (kc.x < kc.maxX){
           Result result = resulTable.result(kc.x);
           result.x(kc.x);
           result.y(100);
        }
    }
```

Whose Kernel code will look like this.

```
typedef struct KernelContext_s{
    int x;
    int maxX;
}KernelContext_t;

typedef Result_s{
   int x,y
} Result_t;

typedef ResultTable_s{
   int count;
   int length;
   Result_t result[0];
} Result_t;

__kernel void init(
    __global KernelContext_t *empty,
    __global ResultTable_t* resultTable
){
    KernelContext_t kernelContext;
    KernelContext_t *kc = &kernelContext;
    kc->x=get_global_id(0);
    kc->maxX = get_global_id(0);

    if(kc->x<kc->maxX){
        __global Result_t *result = &resultTable[kc->x];
        result->x = kc->x;
    }
    return;
}
```

A few notes from this generated code...

* `KernelContext` is itself just an iface mapped segment.
    -  But we don't pass `kc.x` o `kc.maxX` in the segment.
        -  Instead initialize using appropriate  vendor calls

So for OpenCL all kernels start like this

```
__kernel void init(__global KernelContext_t *empty , ....){
    KernelContext_t kernelContext;
    KernelContext_t *kc = &kernelContext;
    kc->x=get_global_id(0);
    kc->maxX = get_global_id(0);
     ....
}
```

Whereas CUDA ;)

```
__kernel void init(__global KernelContext_t *empty , ....){
    KernelContext_t kernelContext;
    KernelContext_t *kc = &kernelContext;
    kc->x=blockIdx.x*blockDim.x+threadIdx.x;
    kc->maxX =gridDim.x*blockDim.x
    ....
}
```

This simplifies code gen. Generally the CUDA code and OpenCL code looks identical.

----

The iface mapping code in hat is a modified form of the code hereWe have a copy of Per's segment mapping code from

https://github.com/minborg/panama-foreign/blob/segment-mapper/src/java.base/share/classes

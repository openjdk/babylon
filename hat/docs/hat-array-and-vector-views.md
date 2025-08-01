## Leverage CodeReflection to expose array view of buffers in kernels

----

* [Contents](hat-00.md)
* House Keeping
  * [Project Layout](hat-01-01-project-layout.md)
  * [Building Babylon](hat-01-02-building-babylon.md)
  * [Building HAT](hat-01-03-building-hat.md)
    * [Enabling the CUDA Backend](hat-01-05-building-hat-for-cuda.md)
* Programming Model
  * [Programming Model](hat-03-programming-model.md)
* Interface Mapping
  * [Interface Mapping Overview](hat-04-01-interface-mapping.md)
  * [Cascade Interface Mapping](hat-04-02-cascade-interface-mapping.md)
* Implementation Detail
  * [Walkthrough Of Accelerator.compute()](hat-accelerator-compute.md)
  * [How we minimize buffer transfers](hat-minimizing-buffer-transfers.md)

----

Here is the canonical HAT example

```java
import jdk.incubator.code.CodeReflection;

@CodeReflection
public class Square {
  @CodeReflection
  public static void kernel(KernelContext kc, S32Arr s32Arr) {
    s32Arr.array(kc.x, s32Arr.array(kc.x) * s32Arr.array(kc.x));
  }

  @CodeReflection
  public static void compute(ComputeContext cc, S32Arr s32Arr) {
    cc.dispatchKernel(s32Arr.length(), kc -> kernel(kc, s32Arr));
  }
}
```

This code in the kernel has always bothered me.  One downside of using MemorySegment backed buffers,
in HAT is that we have made what in array form, would be simple code, look verbose.

```java
 @CodeReflection
  public static void kernel(KernelContext kc, S32Arr s32Arr) {
    s32Arr.array(kc.x, s32Arr.array(kc.x) * s32Arr.array(kc.x));
  }
```

But what if we added a method (`int[] arrayView()`) to `S32Arr` to extract a java int array `view` of simple arrays

Becomes way more readable.
```java
 @CodeReflection
  public static void kernel(KernelContext kc, S32Arr s32Arr) {
    int[] arr = s32Arr.arrayView();
    arr[kc.x] *= arr[kc.x];
  }
```
IMHO This makes code more readable.

For the GPU this is fine.  We can (thanks to CodeReflection) prove that the array is indeed just a view
and we just remove all references to `int arr[]` and replace array accessors with get/set accessors
on the original S32Arr.

But what about Java performance?. Won't it suck because we are copying the array in each kernel ;)

Well, we can use the same trick, we used for the GPU, we take the transformed model (with array
references removed) and create bytecode from that code we and run it.

Historically, we just run the original bytecode in the Java MT/Seq backends, but we don't have to.

This helps also with game of life
```java
  public static void lifePerIdx(int idx, @RO Control control, @RW CellGrid cellGrid) {
            int w = cellGrid.width();
            int h = cellGrid.height();
            int from = control.from();
            int to = control.to();
            int x = idx % w;
            int y = idx / w;
            byte cell = cellGrid.cell(idx + from);
            if (x > 0 && x < (w - 1) && y > 0 && y < (h - 1)) { // passports please
                int count =
                        val(cellGrid, from, w, x - 1, y - 1)
                                + val(cellGrid, from, w, x - 1, y + 0)
                                + val(cellGrid, from, w, x - 1, y + 1)
                                + val(cellGrid, from, w, x + 0, y - 1)
                                + val(cellGrid, from, w, x + 0, y + 1)
                                + val(cellGrid, from, w, x + 1, y + 0)
                                + val(cellGrid, from, w, x + 1, y - 1)
                                + val(cellGrid, from, w, x + 1, y + 1);
                cell = ((count == 3) || ((count == 2) && (cell == ALIVE))) ? ALIVE : DEAD;// B3/S23.
            }
            cellGrid.cell(idx + to, cell);
        }
```

This code uses a helper function `val(grid, offset, w, dx, dy)` to extract the neighbours
```java
 int count =
        val(cellGrid, from, w, x - 1, y - 1)
                + val(cellGrid, from, w, x - 1, y + 0)
                + val(cellGrid, from, w, x - 1, y + 1)
                + val(cellGrid, from, w, x + 0, y - 1)
                + val(cellGrid, from, w, x + 0, y + 1)
                + val(cellGrid, from, w, x + 1, y + 0)
                + val(cellGrid, from, w, x + 1, y - 1)
                + val(cellGrid, from, w, x + 1, y + 1);
```

Val is a bit verbose

```java
  @CodeReflection
        public static int val(@RO CellGrid grid, int from, int w, int x, int y) {
            return grid.cell(((long) y * w) + x + from) & 1;
        }
```

```java
  @CodeReflection
        public static int val(@RO CellGrid grid, int from, int w, int x, int y) {
            byte[] bytes = grid.byteView(); // bit view would be nice ;)
            return bytes[ y * w + x + from] & 1;
        }
```

We could now dispense with `val()` and just write

```java
 byte[] bytes = grid.byteView();
 int count =
        bytes[(y - 1) * w + x - 1 + from]&1
       +bytes[(y + 0) * w + x - 1 + from]&1
       +bytes[(y + 1) * w + x - 1 + from]&1
       +bytes[(y - 1) * w + x + 0 + from]&1
       +bytes[(y + 1) * w + x + 0 + from]&1
       +bytes[(y + 0) * w + x + 1 + from]&1
       +bytes[(y - 1) * w + x + 1 + from]&1
       +bytes[(y + 1) * w + x + 1 + from]&1 ;
```

BTW My inner verilog programmer has always wondered whether shift and oring each bit into a
9 bit value, which we use to index live/dead state from a prepopulated 512 array (GPU constant memory)
would allow us to sidestep the wave divergent conditional :)

```java
byte[] bytes = grid.byteView();
int idx = 0;
int to =0;
byte[] lookup = new byte[]{};
int lookupIdx =
        bytes[(y - 1) * w + x - 1 + from]&1 <<0
       |bytes[(y + 0) * w + x - 1 + from]&1 <<1
       |bytes[(y + 1) * w + x - 1 + from]&1 <<2
       |bytes[(y - 1) * w + x + 0 + from]&1 <<3
       |bytes[(y - 0) * w + x + 0 + from]&1 <<4 // current cell added
       |bytes[(y + 1) * w + x + 0 + from]&1 <<5
       |bytes[(y + 0) * w + x + 1 + from]&1 <<6
       |bytes[(y - 1) * w + x + 1 + from]&1 <<7
       |bytes[(y + 1) * w + x + 1 + from]&1 <<8 ;
// conditional removed!

bytes[idx + to] = lookup[lookupIdx];
```

So the task here is to process kernel code models and perform the appropriate analysis
(for tracking the primitive arrays origin) and transformations to map the array code back to buffer get/sets

----------
The arrayView trick actually leads us to other possibilities.

Let's look at current NBody code.

```java
  @CodeReflection
    static public void nbodyKernel(@RO KernelContext kc, @RW Universe universe, float mass, float delT, float espSqr) {
        float accx = 0.0f;
        float accy = 0.0f;
        float accz = 0.0f;
        Universe.Body me = universe.body(kc.x);

        for (int i = 0; i < kc.maxX; i++) {
            Universe.Body otherBody = universe.body(i);
            float dx = otherBody.x() - me.x();
            float dy = otherBody.y() - me.y();
            float dz = otherBody.z() - me.z();
            float invDist = (float) (1.0f / Math.sqrt(((dx * dx) + (dy * dy) + (dz * dz) + espSqr)));
            float s = mass * invDist * invDist * invDist;
            accx = accx + (s * dx);
            accy = accy + (s * dy);
            accz = accz + (s * dz);
        }
        accx = accx * delT;
        accy = accy * delT;
        accz = accz * delT;
        me.x(me.x() + (me.vx() * delT) + accx * .5f * delT);
        me.y(me.y() + (me.vy() * delT) + accy * .5f * delT);
        me.z(me.z() + (me.vz() * delT) + accz * .5f * delT);
        me.vx(me.vx() + accx);
        me.vy(me.vy() + accy);
        me.vz(me.vz() + accz);
    }
```

Contrast, the above code with the OpenCL code using `float4`

```java
 __kernel void nbody( __global float4 *xyzPos ,__global float4* xyzVel, float mass, float delT, float espSqr ){
      float4 acc = (0.0, 0.0,0.0,0.0);
      float4 myPos = xyzPos[get_global_id(0)];
      float4 myVel = xyzVel[get_global_id(0)];
      for (int i = 0; i < get_global_size(0); i++) {
             float4 delta =  xyzPos[i] - myPos;
             float invDist =  (float) 1.0/sqrt((float)((delta.x * delta.x) + (delta.y * delta.y) + (delta.z * delta.z) + espSqr));
             float s = mass * invDist * invDist * invDist;
             acc= acc + (s * delta);
      }
      acc = acc*delT;
      myPos = myPos + (myVel * delT) + (acc * delT)/2;
      myVel = myVel + acc;
      xyzPos[get_global_id(0)] = myPos;
      xyzVel[get_global_id(0)] = myVel;
}
```

Thanks to interface mapped segments we can approximate `float4` vector type, In fact the
existing `Universe` interface mapped segment actually embeds a `float6` kind of
object holding the `x,y,z,vx,vy,vz` values for each `body` (so position and velocity)

What if we created generalized `float4` interface mapped views `(x,y,z,w)` as placeholders for true vector types.

And modified Universe to hold `pos` and `vel` `float4` arrays.

So in Java code we would pass a MemorySegment version of a F32x4Arr.

Our java code could then start to approximate the OpenCL code.

Except for our lack of operator overloading...

```java
 void nbody(KernelContext kc,  F32x4Arr xyzPos ,F32x4Arr xyzVel, float mass, float delT, float espSqr ){
      float4 acc = float4.of(0.0,0.0,0.0,0.0);
      float4[] xyPosArr = xyzPos.float4View();

      float4 myPos = xyzPosArr[kc.x];
      float4 myVel = xyzVelArr[kc.x];
      for (int i = 0; i < kc.max; i++) {
             float4 delta =  float4.sub(xyzPosArr[i],myPos); // yucky but ok
             float invDist =  (float) 1.0/sqrt((float)((delta.x * delta.x) + (delta.y * delta.y) + (delta.z * delta.z) + espSqr));
             float s = mass * invDist * invDist * invDist;
             acc= float4.add(acc,(s * delta)); // adding scaler to float4 via overloaded add method
      }
      acc = float4.mul(acc*delT);  // scaler * vector
      myPos = float4.add(float4.add(myPos,float4.mul(myVel * delT)) , float4.mul(acc,delT/2));
      myVel = float4.add(myVel,acc);
      xyzPos[kc.x] = myPos;
      xyzVel[kc.x] = myVel;
}
```
The code is more compact, it's still weird though. Because we can't overload operators.

Well we can sort of.

What if we allowed a `floatView()` call on `float4` ;) which yields float value to be used as a proxy
for the `float4` we fetched it from...

So we would leverage the anonymity of `var`

`var myVelF4 = myVel.floatView()` // pst var is really a float

From the code model we can track the relationship from float views to the original vector...

Any action we perform on the 'float' view will be mapped back to calls on the origin, and performed on the actual origin float4.

So
`myVelF4 + myVelF4`  -> `float4.add(myVel,myVel)` behind the scenes.

Yeah, it's a bit crazy. The code would look something like this.  Perhaps this is a 'bridge too far'.

```java
void nbody(KernelContext kc,  F32x4Arr xyzPos ,F32x4Arr xyzVel, float mass, float delT, float espSqr ){
      float4 acc = float4.of(0.0,0.0,0.0,0.0);
      var accF4 = acc.floatView(); // var is really float
      float4[] xyPosArr = xyzPos.float4View();

      float4 myPos = xyzPosArr[kc.x];
      float4 myVel = xyzVelArr[kc.x];
      var myPosF4 = myPos.floatView(); // ;)  var is actually a float tied to myPos
      var myVelF4 = myVel.floatView(); //... myVel
      for (int i = 0; i < kc.max; i++) {
             var bodyF4 = xyzPosArr[i].floatView(); // bodyF4 is a float
             var  deltaF4 = bodyF4 - myPosF4;  // look ma operator overloading ;)
             float invDist =  (float) 1.0/sqrt((float)((delta.x * delta.x) + (delta.y * delta.y) + (delta.z * delta.z) + espSqr));
             float s = mass * invDist * invDist * invDist;
             accF4+=s * deltaF4; // adding scaler to float4 via overloaded add method
      }
      accF4 = accF4*delT;  // scalar * vector
      myPosF4 = myPosF4 + (myVelF4 * delT) + (accF4 * delT)/2;
      myVelF4 = myVelF4 + accF4;
      xyzPos[kc.x] = myPos;
      xyzVel[kc.x] = myVel;
}
```



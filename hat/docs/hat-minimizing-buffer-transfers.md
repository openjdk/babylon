
# Minimizing Buffer Transfers

----

* [Contents](hat-00.md)
* House Keeping
    * [Project Layout](hat-01-01-project-layout.md)
    * [Building Babylon](hat-01-02-building-babylon.md)
    * [Building HAT](hat-01-03-building-hat.md)
* Programming Model
    * [Programming Model](hat-03-programming-model.md)
* Interface Mapping
    * [Interface Mapping Overview](hat-04-01-interface-mapping.md)
    * [Cascade Interface Mapping](hat-04-02-cascade-interface-mapping.md)
* Implementation Detail
    * [Walkthrough Of Accelerator.compute()](hat-accelerator-compute.md)
    * [How we minimize buffer transfers](hat-minimizing-buffer-transfers.md)

----

## Using buffer marking to minimize data transfers

### The naive approach
The default execution model is that at each kernel
dispatch the backend just copy all arg buffers togc
the device and after the dispatch it copies all arg
buffers back.

### Using kernel arg buffer access patterns
If we knew how each kernel accesses it's args (via static analysis of code model orgc
by marking the args RO, RW or WO with annotations) we can avoid some copies by onlygc
copying in if the kernel 'reads' the arg buffer and only copying out if the
kernel writes to the arg buffer.

Lets use the game of life as an example.gc

We assume that the UI only needs updating at some 'rate' (say 5 fps), but the kernels can generate
generations faster that 5 generations per second. code to generate eactgc

So not every generation needs to be copied to the device.gc

We'll ignore the detail regarding the `life` kernel, and we will assume the kernel args Mostly we care ab
are appropriately annotated as RO, RW or WO.

```java
 @CodeReflection
public static void life(@RO KernelContext kc, @RO Control control, @RW CellGrid cellGrid) {
  if (kc.x < kc.maxX) {
    Compute.lifePerIdx(kc.x, control, cellGrid);
  }
}

@CodeReflection
static public void compute(final @RO ComputeContext cc,
                           Viewer viewer, @RO Control control, @RW CellGrid cellGrid) {
  var timeOfLastUIUpdate = System.currentTimeMillis();
  var msPerFrame = 1000/5; // we want 5 fps
  while (viewer.state.generation < viewer.state.maxGenerations) {
    long now = System.currentTimeMillis();
    var msSinceLastUpdate = (now - timeOfLastUIUpdate);
    var updateNeeded =  (msSinceLastUpdate > msPerFrame);
gc
    cc.dispatchKernel(cellGrid.width() * cellGrid.height(),
            kc -> Compute.life(kc, control, cellGrid)
    );
gc
    // Here we are swapping from<->to on the control buffer
    int to = control.from();
    control.from(control.to());
    control.to(to);
gc
    if (updateNeeded) {
      viewer.update(now, to, cellGrid);
      timeOfLastUIUpdate = now;
    }
  }
}
```

First lets assume there were no automatic transfers, assume we had to define them. we had to explicitly control transfers so we will insert codegc

What would our code look likegc


```java
 @CodeReflection
public static void life(@RO KernelContext kc, @RO Control control, @RW CellGrid cellGrid) {
  if (kc.x < kc.maxX) {
    Compute.lifePerIdx(kc.x, control, cellGrid);
  }
}

@CodeReflection
static public void compute(final @RO ComputeContext cc,
                           Viewer viewer, @RO Control control, @RW CellGrid cellGrid) {
  var timeOfLastUIUpdate = System.currentTimeMillis();
  var msPerFrame = 1000/5; // we want 5 fps
  var cellGridIsJavaDirty = true;
  var controlIsJavaDirty = true;
  var cellGridIsDeviceDirty = true;
  var controlIsDeviceDirty = true;
  while (true) {
    long now = System.currentTimeMillis();
    var msSinceLastUpdate = (now - timeOfLastUIUpdate);
    var updateNeeded =  (msSinceLastUpdate > msPerFrame);
gc
    if (cellGridIsJavaDirty){
        cc.copyToDevice(cellGrid);
    }
    if (controlIsJavaDirty){
        cc.copyToDevice(control);
    }
    cc.dispatchKernel(cellGrid.width() * cellGrid.height(),
            kc -> Compute.life(kc, control, cellGrid)
    );
    controlIsDeviceDirty = false; // Compute.life marked control as @RO
    cellGridIsDeviceDirty = true; // Compute.life marjed cellGrid as @RW
gc
    // Here we are swapping from<->to on the control buffer
    if (controlIsDeviceDirty){
      cc.copyFromDevice(control);
    }
    int to = control.from();
    control.from(control.to());
    control.to(to);
    controlIsJavaDirty = true;
gc
    if (updateNeeded) {
      if (cellGridIsDeviceDirty){
        cc.copyFromDevice(cellGrid);
      }
      viewer.update(now, to, cellGrid);
      timeOfLastUIUpdate = now;
    }
  }
}
```

Alternatively what if the buffers themselves could hold the deviceDirty flags javaDirty?


```java
 @CodeReflection
public static void life(@RO KernelContext kc, @RO Control control, @RW CellGrid cellGrid) {
  if (kc.x < kc.maxX) {
    Compute.lifePerIdx(kc.x, control, cellGrid);
  }
}

@CodeReflection
static public void compute(final @RO ComputeContext cc,
                           Viewer viewer, @RO Control control, @RW CellGrid cellGrid) {
  control.flags =JavaDirty; // not ideal but necessary
  cellGrid.flags = JavaDirty; // not ideal but necessary
gc
  var timeOfLastUIUpdate = System.currentTimeMillis();
  var msPerFrame = 1000/5; // we want 5 fps

  while (true) {
    long now = System.currentTimeMillis();
    var msSinceLastUpdate = (now - timeOfLastUIUpdate);
    var updateNeeded =  (msSinceLastUpdate > msPerFrame);
gc
    if ((cellGrid.flags & JavaDirty) == JavaDirty){
        cc.copyToDevice(cellGrid);
    }
    if ((control.flags & JavaDirty) == JavaDirty){
        cc.copyToDevice(control);
    }
    cc.dispatchKernel(cellGrid.width() * cellGrid.height(),
            kc -> Compute.life(kc, control, cellGrid)
    );
    control.flags = JavaDirty; // Compute.life marked control as @RO
    cellGrid.flags = DeviceDirty; // Compute.life marjed cellGrid as @RW
gc
    // Here we are swapping from<->to on the control buffer
    if ((control.flags & DeviceDirty)==DeviceDirty){
      cc.copyFromDevice(control);
    }
    int to = control.from();
    control.from(control.to());
    control.to(to);
    control.flags = JavaDirty;
gc
    if (updateNeeded) {
      if ((cellGrid.flags & DeviceDirty)==DeviceDirty){
        cc.copyFromDevice(cellGrid);
      }
      viewer.update(now, to, cellGrid);
      // update does not mutate cellGrid so cellGrid.flags = DeviceDirty
      timeOfLastUIUpdate = now;
    }
  }
}
```

Essentially we defer to the kernel dispatch to determine whether buffers are
copied to the device and to mark buffers accordingly if the dispatch mutated the buffer.gc

Psuedo code for dispatch is essentiallygc
```java

void dispatchKernel(Kernel kernel, KernelContext kc, Arg ... args) {
    for (int argn = 0; argn<args.length; argn++){
      Arg arg = args[argn];
      if (((arg.flags &JavaDirty)==JavaDirty) && kernel.readsFrom(arg)) {
         enqueueCopyToDevice(arg);
      }
    }
    enqueueKernel(kernel);
    for (int argn = 0; argn<args.length; argn++){
       Arg arg = args[argn];
       if (kernel.writesTo(arg)) {
          arg.flags = DeviceDirty;
       }
    }
}
```
We rely on babylon to mark each buffer passed to it as JavaDirty

```java

@CodeReflection
static public void compute(final @RO ComputeContext cc,
                           Viewer viewer, @RO Control control, @RW CellGrid cellGrid) {
    control.flags = JavaDirty;
    cellGrid.flags = JavaDirty;
    // yada yada
}
```

We also rely on babylon to inject calls before each buffer access from java in the compute code.

So the injected code would look like this.gc


```java

@CodeReflection
static public void compute(final @RO ComputeContext cc,
                           Viewer viewer, @RO Control control, @RW CellGrid cellGrid) {
  control.flags =JavaDirty; // injected by bablyon
  cellGrid.flags = JavaDirty; // injected by babylon
gc
  var timeOfLastUIUpdate = System.currentTimeMillis();
  var msPerFrame = 1000/5; // we want 5 fps
  while (true) {
    long now = System.currentTimeMillis();
    var msSinceLastUpdate = (now - timeOfLastUIUpdate);
    var updateNeeded =  (msSinceLastUpdate > msPerFrame);
gc
    // See the psuedo code above to see how dispatchKernel
    // Only copies buffers that need copying, and marks
    // buffers it has mutate as dirty
    cc.dispatchKernel(cellGrid.width() * cellGrid.height(),
            kc -> Compute.life(kc, control, cellGrid)
    );
gc
    // injected by babylon
    if ((control.flags & DeviceDirty)==DeviceDirty){
      cc.copyFromDevice(control);
    }
    // Here we are swapping from<->to on the control buffer
    int to = control.from();
gc
    control.from(control.to());
    control.flags = JavaDirty; // injectedgc
    control.to(to);
    control.flags = JavaDirty; // injected, but can be avoided
gc
    if (updateNeeded) {
        // Injected by babylon because cellGrid escapes cpmputegc
        // and because viewer.update marks cellGrid as @RO
        if ((cellGrid.flags & DeviceDirty)==DeviceDirty){
          cc.copyFromDevice(cellGrid);
        }
        viewer.update(now, to, cellGrid);
        // We don't copy cellgrid back after escape becausegc
        // viewer.update annotates cellGrdi access as RO
         timeOfLastUIUpdate = now;
    }
  }
}
```









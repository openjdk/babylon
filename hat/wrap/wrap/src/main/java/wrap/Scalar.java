package wrap;

import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.invoke.VarHandle;

public record Scalar(String name, MemorySegment memorySegment,
                     VarHandle varHandle) {

    public static Scalar of(MemorySegment memorySegment, MemoryLayout memoryLayout, String name) {
        return of(memorySegment, memoryLayout, MemoryLayout.PathElement.groupElement(name));
    }

    public static Scalar of(MemorySegment memorySegment, MemoryLayout memoryLayout, String name, Object initial) {
        return of(memorySegment, memoryLayout, name).set(initial);
    }

    public static Scalar of(MemorySegment memorySegment, MemoryLayout memoryLayout,
                            MemoryLayout.PathElement... pathElements) {
        VarHandle vh = memoryLayout.varHandle(pathElements);
        String name = null;
        for (int i = 0; i < pathElements.length; i++) {
            MemoryLayout.PathElement pathElement = pathElements[i];
            // Why can't I access LayoutPath?
            if (pathElement.toString().isEmpty()) {
                name = pathElement.toString();
            }
        }
        return new Scalar(name, memorySegment, vh);

    }

    public Scalar set(Object o) {
        varHandle.set(memorySegment, 0, o);
        return this;
    }

    public Object get() {
        return varHandle.get(memorySegment, 0);
    }

    public int i32() {
        return (int) get();
    }

    public long i64() {
        return (long) get();
    }

    public float f32() {
        return (float) get();
    }

    public double f64() {
        return (double) get();
    }
}

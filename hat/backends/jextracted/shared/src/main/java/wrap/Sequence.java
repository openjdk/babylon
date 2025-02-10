package wrap;

import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.invoke.VarHandle;

public record Sequence(String name, MemorySegment memorySegment,
                VarHandle varHandle) {

    public static Sequence of(MemorySegment memorySegment, MemoryLayout memoryLayout, String name) {
        return of(memorySegment, memoryLayout,
                MemoryLayout.PathElement.groupElement(name), MemoryLayout.PathElement.sequenceElement());
    }

    public static Sequence of(MemorySegment memorySegment, MemoryLayout memoryLayout,
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
        return new Sequence(name, memorySegment, vh);

    }

    public Object get(int idx) {
        return varHandle.get(memorySegment, 0, (long) idx);
    }

    public byte i8(int idx) {
        return (byte) get(idx);
    }

    public short i16(int idx) {
        return (short) get(idx);
    }

    public int i32(int idx) {
        return (int) get(idx);
    }

    public long i64(int idx) {
        return (long) get(idx);
    }

    public float f32(int idx) {
        return (float) get(idx);
    }

    public double f64(int idx) {
        return (double) get(idx);
    }

    public Sequence set(int idx, byte v) {
        varHandle.set(memorySegment, 0, (long) idx, v);
        return this;
    }
}

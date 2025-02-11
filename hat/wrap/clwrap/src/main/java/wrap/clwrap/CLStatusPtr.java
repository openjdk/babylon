package wrap.clwrap;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

import static java.lang.foreign.ValueLayout.JAVA_INT;
import static opencl.opencl_h.*;

public record CLStatusPtr(MemorySegment ptr) {
    public static CLStatusPtr of(Arena arena) {
        return new CLStatusPtr(arena.allocateFrom(JAVA_INT, CL_SUCCESS()));
    }

    public int set(int value) {
        ptr.set(JAVA_INT, 0, value);
        return value;
    }

    public int get() {
        return ptr.get(JAVA_INT, 0);
    }

    public boolean isOK() {
        return get() == CL_SUCCESS();
    }

    public long sizeof() {
        return JAVA_INT.byteSize();
    }
}

package wrap.glwrap;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

import static java.lang.foreign.ValueLayout.JAVA_INT;
import static opengl.opengl_h.*;


public record GLStatusPtr(MemorySegment ptr) {
    public static GLStatusPtr of(Arena arena) {
        return new GLStatusPtr(arena.allocateFrom(JAVA_INT, GL_TRUE()));
    }

    public int set(int value) {
        ptr.set(JAVA_INT, 0, value);
        return value;
    }

    public int get() {
        return ptr.get(JAVA_INT, 0);
    }

    public boolean isOK() {
        return get() == GL_TRUE();
    }

    public long sizeof() {
        return JAVA_INT.byteSize();
    }
}

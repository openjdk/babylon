package hat.buffer;

import hat.ifacemapper.SegmentMapper;

import java.lang.foreign.Arena;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.StructLayout;
import java.lang.invoke.MethodHandles;

import static java.lang.foreign.ValueLayout.JAVA_INT;

public interface KernelContext extends CompleteBuffer {
    StructLayout layout = MemoryLayout.structLayout(
            JAVA_INT.withName("x"),
            JAVA_INT.withName("maxX")
    ).withName(KernelContext.class.getSimpleName());

    static KernelContext create(Arena arena, MethodHandles.Lookup lookup, int x, int maxX) {
        KernelContext kernelContext = SegmentMapper.of(lookup, KernelContext.class,layout).allocate(arena);
        kernelContext.x(x);
        kernelContext.maxX(maxX);
        return kernelContext;
    }
    int x();
    void x(int x);
    int maxX();
    void maxX(int maxX);
}
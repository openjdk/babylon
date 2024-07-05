package hat.buffer;

import hat.ifacemapper.HatData;
import hat.ifacemapper.Schema;
import hat.ifacemapper.SegmentMapper;

import java.lang.foreign.Arena;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.StructLayout;
import java.lang.invoke.MethodHandles;

import static java.lang.foreign.ValueLayout.JAVA_INT;

public interface KernelContext extends CompleteBuffer {
    int x();
    void x(int x);
    int maxX();
    void maxX(int maxX);

    Schema<KernelContext> schema = Schema.of(KernelContext.class, s->s.fields("x","maxX"));

    static KernelContext create(BufferAllocator bufferAllocator, int x, int maxX) {
        KernelContext kernelContext =  schema.allocate(bufferAllocator);
        kernelContext.x(x);
        kernelContext.maxX(maxX);
        return kernelContext;
    }

}
package hat.buffer;


import hat.ifacemapper.Schema;

import java.lang.invoke.MethodHandles;


public interface KernelContext extends Buffer {
    int x();
    void x(int x);
    int maxX();
    void maxX(int maxX);

    Schema<KernelContext> schema = Schema.of(KernelContext.class, s->s.fields("x","maxX"));

    static KernelContext create(MethodHandles.Lookup lookup,BufferAllocator bufferAllocator, int x, int maxX) {
        KernelContext kernelContext =  schema.allocate(lookup,bufferAllocator);
        kernelContext.x(x);
        kernelContext.maxX(maxX);
        return kernelContext;
    }

}
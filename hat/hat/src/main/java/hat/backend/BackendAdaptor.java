package hat.backend;

import hat.ComputeContext;
import hat.NDRange;
import hat.buffer.Buffer;
import hat.callgraph.KernelCallGraph;
import hat.callgraph.KernelEntrypoint;
import hat.ifacemapper.BoundSchema;
import hat.ifacemapper.SegmentMapper;

import java.lang.foreign.Arena;
import java.lang.reflect.InvocationTargetException;

public abstract class BackendAdaptor implements Backend {
    @Override
    public void computeContextHandoff(ComputeContext computeContext) {

    }

    @Override
    public void dispatchCompute(ComputeContext computeContext, Object... args) {
        try {
            computeContext.computeCallGraph.entrypoint.method.invoke(null, args);
        } catch (IllegalAccessException | InvocationTargetException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public void dispatchKernel(KernelCallGraph kernelCallGraph, NDRange ndRange, Object... args) {
        KernelEntrypoint kernelEntrypoint = kernelCallGraph.entrypoint;
        for (ndRange.kid.x = 0; ndRange.kid.x < ndRange.kid.maxX; ndRange.kid.x++) {
            try {
                args[0] = ndRange.kid;
                kernelEntrypoint.method.invoke(null, args);
            } catch (IllegalAccessException e) {
                throw new RuntimeException(e);
            } catch (InvocationTargetException e) {
                throw new RuntimeException(e);
            }
        }
    }

    @Override
    public <T extends Buffer> T allocate(SegmentMapper<T> segmentMapper, BoundSchema<T> boundSchema) {
        return segmentMapper.allocate(Arena.global(), boundSchema);
    }

}

package hat.backend;


import hat.ComputeContext;
import hat.NDRange;
import hat.callgraph.KernelCallGraph;

public class HIPBackend extends C99NativeBackend {
    public HIPBackend() {
        super("hip_backend");
        getBackend(null);
        info();
    }

    @Override
    public void computeContextHandoff(ComputeContext computeContext) {
        injectBufferTracking(computeContext.computeCallGraph.entrypoint);

    }

    @Override
    public void dispatchKernel(KernelCallGraph kernelCallGraph, NDRange ndRange, Object... args) {
        CompiledKernel compiledKernel = kernelCallGraphCompiledCodeMap.computeIfAbsent(kernelCallGraph, (_) -> {
            String code = createCode(kernelCallGraph, new HIPHatKernelBuilder(), args);
            long programHandle = compileProgram(code);
            if (programOK(programHandle)) {
                long kernelHandle = getKernel(programHandle, kernelCallGraph.entrypoint.method.getName());
                return new CompiledKernel(this, kernelCallGraph, code, kernelHandle, args);
            } else {
                throw new IllegalStateException("HIP failed to compile ");
            }
        });
        compiledKernel.dispatch(ndRange,args);
    }
}

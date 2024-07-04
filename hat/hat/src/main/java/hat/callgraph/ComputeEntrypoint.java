package hat.callgraph;

import hat.optools.FuncOpWrapper;

import java.lang.invoke.MethodHandle;
import java.lang.reflect.Method;

public class ComputeEntrypoint extends ComputeCallGraph.ComputeReachableResolvedMethodCall implements Entrypoint {
    public FuncOpWrapper lowered;
    public MethodHandle mh;

    public ComputeEntrypoint(CallGraph<ComputeEntrypoint> callGraph, Method method, FuncOpWrapper funcOpWrapper) {
        super(callGraph, null, method, funcOpWrapper);
    }
}

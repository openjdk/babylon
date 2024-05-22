package hat.callgraph;

import hat.ComputeContext;
import hat.optools.FuncOpWrapper;

import java.lang.reflect.Method;
import java.lang.reflect.code.type.MethodRef;

public class KernelEntrypoint extends KernelCallGraph.KernelReachableResolvedMethodCall implements Entrypoint {
    public KernelEntrypoint(CallGraph<KernelEntrypoint> callGraph, MethodRef targetMethodRef, Method method, FuncOpWrapper funcOpWrapper) {
        super(callGraph, targetMethodRef, method, funcOpWrapper);
    }
}

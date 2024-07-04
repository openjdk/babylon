package hat.callgraph;

import hat.Accelerator;
import hat.ComputeContext;
import hat.KernelContext;
import hat.buffer.Buffer;
import hat.optools.FuncOpWrapper;
import hat.optools.OpWrapper;
import hat.util.Result;

import java.lang.reflect.Method;
import java.lang.reflect.code.op.CoreOp;
import java.lang.reflect.code.type.JavaType;
import java.lang.reflect.code.type.MethodRef;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;
import java.util.stream.Stream;

public class ComputeCallGraph extends CallGraph<ComputeEntrypoint> {


    public interface ComputeReachable {
    }

    public abstract static class ComputeReachableResolvedMethodCall extends ResolvedMethodCall implements ComputeReachable {
        public ComputeReachableResolvedMethodCall(CallGraph<ComputeEntrypoint> callGraph, MethodRef targetMethodRef, Method method, FuncOpWrapper funcOpWrapper) {
            super(callGraph, targetMethodRef, method, funcOpWrapper);
        }
    }

    public static class ComputeReachableUnresolvedMethodCall extends UnresolvedMethodCall implements ComputeReachable {
        ComputeReachableUnresolvedMethodCall(CallGraph<ComputeEntrypoint> callGraph, MethodRef targetMethodRef, Method method) {
            super(callGraph, targetMethodRef, method);
        }
    }

    public static class ComputeReachableIfaceMappedMethodCall extends ComputeReachableUnresolvedMethodCall {
        ComputeReachableIfaceMappedMethodCall(CallGraph<ComputeEntrypoint> callGraph, MethodRef targetMethodRef, Method method) {
            super(callGraph, targetMethodRef, method);
        }
    }

    public static class ComputeReachableAcceleratorMethodCall extends ComputeReachableUnresolvedMethodCall {
        ComputeReachableAcceleratorMethodCall(CallGraph<ComputeEntrypoint> callGraph, MethodRef targetMethodRef, Method method) {
            super(callGraph, targetMethodRef, method);
        }
    }

    public static class ComputeContextMethodCall extends ComputeReachableUnresolvedMethodCall {
        ComputeContextMethodCall(CallGraph<ComputeEntrypoint> callGraph, MethodRef targetMethodRef, Method method) {
            super(callGraph, targetMethodRef, method);
        }
    }

    public static class OtherComputeReachableResolvedMethodCall extends ComputeReachableResolvedMethodCall {
        OtherComputeReachableResolvedMethodCall(CallGraph<ComputeEntrypoint> callGraph, MethodRef targetMethodRef, Method method, FuncOpWrapper funcOpWrapper) {
            super(callGraph, targetMethodRef, method, funcOpWrapper);
        }
    }

    static boolean isKernelDispatch(Method calledMethod, FuncOpWrapper fow) {
        if (fow.getReturnType() instanceof JavaType javaReturnType && javaReturnType.equals(JavaType.VOID)) {
            if (calledMethod.getParameterTypes() instanceof Class<?>[] parameterTypes && parameterTypes.length > 1) {
                // We check that the proposed kernel first arg is an NDRange.kid and
                // the only other args are primitive or ifacebuffers
                var firstArgIsKid = new Result<Boolean>(false);
                var atLeastOneIfaceBufferParam = new Result<Boolean>(false);
                var hasOnlyPrimitiveAndIfaceBufferParams = new Result<Boolean>(true);
                fow.paramTable().stream().forEach(paramInfo -> {
                    if (paramInfo.idx == 0) {
                        firstArgIsKid.of(parameterTypes[0].isAssignableFrom(KernelContext.class));
                    } else {
                        if (paramInfo.isPrimitive()) {
                            // OK
                        } else if (paramInfo.isIfaceBuffer()) {
                            atLeastOneIfaceBufferParam.of(true);
                        } else {
                            hasOnlyPrimitiveAndIfaceBufferParams.of(false);
                        }
                    }
                });
                return true;
            }
            return false;
        } else {
            return false;
        }
    }

    public final Map<MethodRef, KernelCallGraph> kernelCallGraphMap = new HashMap<>();

    public Stream<KernelCallGraph> kernelCallGraphStream() {
        return kernelCallGraphMap.values().stream();
    }

    //  public ComputeCallGraph(ComputeContext computeContext, ComputeEntrypoint computeEntrypoint) {
    //    super(computeContext, computeEntrypoint);
    // }
    public ComputeCallGraph(ComputeContext computeContext, Method method, FuncOpWrapper funcOpWrapper) {
        super(computeContext, new ComputeEntrypoint(null, method, funcOpWrapper));
        entrypoint.callGraph = this;
    }

    public void updateDag(ComputeReachableResolvedMethodCall computeReachableResolvedMethodCall) {
        /**
         * A ResolvedComputeMethodCall (entrypoint or java  methdod reachable from a compute entrypojnt)  has the following calls
         * <p>
         * 1) java calls to compute class static functions
         * a) we must have the code model available for these and must extend the dag
         * 2) calls to buffer based interface mappings
         * a) getters (return non void)
         * b) setters (return void)
         * 3) calls to the compute context
         * a) kernel dispatches
         * 4) calls through compute context.accelerator;
         * a) range creations (maybe computecontext shuld manage ranges?)
         * 5) References to the dispatched kernels
         * a) We must also have the code models for these and must extend the dag to include these.
         **/


        computeReachableResolvedMethodCall.funcOpWrapper().selectCalls((invokeWrapper) -> {
            var methodRef = invokeWrapper.methodRef();
            Class<?> javaRefClass = invokeWrapper.javaRefClass().orElseThrow();
            Method invokeWrapperCalledMethod = invokeWrapper.method();
            if (Buffer.class.isAssignableFrom(javaRefClass)) {
                // System.out.println("iface mapped buffer call  -> " + methodRef);
                computeReachableResolvedMethodCall.addCall(methodRefToMethodCallMap.computeIfAbsent(methodRef, _ ->
                        new ComputeReachableIfaceMappedMethodCall(this, methodRef, invokeWrapperCalledMethod)
                ));
            } else if (Accelerator.class.isAssignableFrom(javaRefClass)) {
                // System.out.println("call on the accelerator (must be through the computeContext) -> " + methodRef);
                computeReachableResolvedMethodCall.addCall(methodRefToMethodCallMap.computeIfAbsent(methodRef, _ ->
                        new ComputeReachableAcceleratorMethodCall(this, methodRef, invokeWrapperCalledMethod)
                ));

            } else if (ComputeContext.class.isAssignableFrom(javaRefClass)) {
                // System.out.println("call on the computecontext -> " + methodRef);
                computeReachableResolvedMethodCall.addCall(methodRefToMethodCallMap.computeIfAbsent(methodRef, _ ->
                        new ComputeContextMethodCall(this, methodRef, invokeWrapperCalledMethod)
                ));
            } else if (entrypoint.method.getDeclaringClass().equals(javaRefClass)) {
                Optional<CoreOp.FuncOp> optionalFuncOp = invokeWrapperCalledMethod.getCodeModel();
                if (optionalFuncOp.isPresent()) {
                    FuncOpWrapper fow = OpWrapper.wrap(optionalFuncOp.get());
                    if (isKernelDispatch(invokeWrapperCalledMethod, fow)) {
                        // System.out.println("A kernel reference (not a direct call) to a kernel " + methodRef);
                        kernelCallGraphMap.computeIfAbsent(methodRef, _ ->
                                new KernelCallGraph(this, methodRef, invokeWrapperCalledMethod, fow).close()
                        );
                    } else {
                        // System.out.println("A call to a method on the compute class which we have code model for " + methodRef);
                        computeReachableResolvedMethodCall.addCall(methodRefToMethodCallMap.computeIfAbsent(methodRef, _ ->
                                new OtherComputeReachableResolvedMethodCall(this, methodRef, invokeWrapperCalledMethod, fow)
                        ));
                    }
                } else {
                    //  System.out.println("A call to a method on the compute class which we DO NOT have code model for " + methodRef);
                    computeReachableResolvedMethodCall.addCall(methodRefToMethodCallMap.computeIfAbsent(methodRef, _ ->
                            new ComputeReachableUnresolvedMethodCall(this, methodRef, invokeWrapperCalledMethod)
                    ));

                }
            } else {
                //TODO what about ifacenestings?
                // System.out.println("A call to a method on the compute class which we DO NOT have code model for " + methodRef);
                computeReachableResolvedMethodCall.addCall(methodRefToMethodCallMap.computeIfAbsent(methodRef, _ ->
                        new ComputeReachableUnresolvedMethodCall(this, methodRef, invokeWrapperCalledMethod)
                ));
            }
        });

        if (kernelCallGraphMap.isEmpty()) {
            throw new IllegalStateException("entrypoint compute has no kernel references!");
        }

        boolean updated = true;
        computeReachableResolvedMethodCall.closed = true;
        while (updated) {
            updated = false;
            var unclosed = callStream().filter(m -> !m.closed).findFirst();
            if (unclosed.isPresent()) {
                if (unclosed.get() instanceof ComputeReachableResolvedMethodCall reachableResolvedMethodCall) {
                    updateDag(reachableResolvedMethodCall);
                } else {
                    unclosed.get().closed = true;
                }
                updated = true;
            }
        }

    }

    public void close() {
        updateDag(entrypoint);
    }
}

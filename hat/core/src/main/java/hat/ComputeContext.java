/*
 * Copyright (c) 2024, Oracle and/or its affiliates. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.  Oracle designates this
 * particular file as subject to the "Classpath" exception as provided
 * by Oracle in the LICENSE file that accompanied this code.
 *
 * This code is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
 * version 2 for more details (a copy is included in the LICENSE file that
 * accompanied this code).
 *
 * You should have received a copy of the GNU General Public License version
 * 2 along with this work; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
 * or visit www.oracle.com if you need additional information or have any
 * questions.
 */
package hat;

import hat.buffer.DispatchContext;
import optkl.OpHelper;
import optkl.util.carriers.ArenaAndLookupCarrier;
import optkl.ifacemapper.BufferTracker;
import hat.callgraph.ComputeCallGraph;
import hat.callgraph.KernelCallGraph;
import optkl.ifacemapper.MappableIface;
import jdk.incubator.code.dialect.core.CoreOp.FuncOp;
import jdk.incubator.code.Op;
import jdk.incubator.code.Quoted;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.MethodRef;

import java.lang.foreign.Arena;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import java.lang.reflect.Parameter;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

import static optkl.OpHelper.Invoke.invoke;
import static optkl.OpHelper.Lambda.lambda;

/**
 * A ComputeContext is created by an Accelerator to capture and control compute and kernel
 * callgraphs for the work to be performed by the backend.
 * <p/>
 * The Compute closure is created first, by walking the code model of the entrypoint, then transitively
 * visiting all conventional code reachable from this entrypoint.
 * <p/>
 * Generally all user defined methods reachable from the entrypoint (and the entrypoint itself) must be static methods of the same
 * enclosing classes.
 * <p/>
 * We do allow calls on the ComputeContext itself, and on the mapped interface buffers holding non uniform kernel data.
 * <p/>
 * Each request to dispatch a kernel discovered in the compute graph, results in a new Kernel call graph
 * being created with the dispatched kernel as it's entrypoint.
 * <p/>
 * When the ComputeContext is finalized, it is passed to the backend via <a href="Backend.computeClosureHandoff(ComputeContext)"></a>
 *
 * @author Gary Frost
 */
public class ComputeContext implements ArenaAndLookupCarrier, BufferTracker {

    @Override
    public Arena arena() {
        return accelerator.arena();
    }

    @Override
    public MethodHandles.Lookup lookup() {
        return accelerator.lookup();
    }


    public Config config() {
        return accelerator().config();
    }

    public void invokeWithArgs(Object[] args) {
        computeCallGraph.invokeWithArgs(args);
    }

    public enum WRAPPER {
        MUTATE("Mutate"), ACCESS("Access");
        public final MethodRef pre;
        public final MethodRef post;

        WRAPPER(String name) {
            this.pre = MethodRef.method(ComputeContext.class, "pre" + name, void.class, MappableIface.class);
            this.post = MethodRef.method(ComputeContext.class, "post" + name, void.class, MappableIface.class);
        }
    }

    private  final Accelerator accelerator;

    public final  Accelerator accelerator(){
        return accelerator;
    }

    private  final ComputeCallGraph computeCallGraph;

    public final  ComputeCallGraph computeCallGraph(){
        return computeCallGraph;
    }

    /**
     * Called by the Accelerator when the accelerator is passed a compute entrypoint.
     * <p>
     * So given a ComputeClass such as..
     * <pre>
     *  public class MyComputeClass {
     *    @Reflect
     *    public static void addDeltaKernel(KernelContext kc, S32Array arrayOfInt, int delta) {
     *        arrayOfInt.array(kc.x, arrayOfInt.array(kc.x)+delta);
     *    }
     *
     *    @Reflect
     *    static public void doSomeWork(final ComputeContext cc, S32Array arrayOfInt) {
     *        cc.dispatchKernel(KernelContext kc -> addDeltaKernel(kc,arrayOfInt.length(), 5, arrayOfInt);
     *    }
     *  }
     *  </pre>
     *
     * @param accelerator
     * @param computeMethod
     */

    protected ComputeContext(Accelerator accelerator, Method computeMethod) {
        this.accelerator = accelerator;
        Optional<FuncOp> funcOp =  Op.ofMethod(computeMethod);
        if (funcOp.isEmpty()) {
            throw new RuntimeException("Failed to create ComputeCallGraph (did you miss @Reflect annotation?).");
        }
        this.computeCallGraph = new ComputeCallGraph(this, computeMethod, funcOp.get());
        this.accelerator.backend.computeContextHandoff(this);
    }

    public record KernelCallSite(Quoted<JavaOp.LambdaOp> quoted, JavaOp.LambdaOp lambdaOp, MethodRef methodRef, KernelCallGraph kernelCallGraph, Object[] capturedArgs) {}

    private record ConstantArgument(int paramIndex, Class<?> type, Object value) {
        public static ConstantArgument of(int i, Class<?> type, Object capturedValue) {
            // We only specialized for a small set of types
            if (type == int.class && capturedValue instanceof Number value) {
                return new ConstantArgument(i, type, value);
            } else if (type == float.class && capturedValue instanceof Float value) {
                return new ConstantArgument(i, type, value);
            }
            throw new IllegalStateException("Input constant of type: " + type.getName() + " not supported");
        }
    }

    private record SpecializationKey(List<ConstantArgument> arguments) {
        public SpecializationKey {
            arguments = List.copyOf(arguments);
        }

        public static SpecializationKey of(Method kernelMethod, Object[] quotedCapturedValues) {
            Parameter[] parameters = kernelMethod.getParameters();
            List<ConstantArgument> arguments = new ArrayList<>();
            for (int i = 0; i < quotedCapturedValues.length; i++) {
                if (parameters[i].getType().isPrimitive()) {
                    arguments.add(ConstantArgument.of(i, parameters[i].getType(), quotedCapturedValues[i]));
                }
            }
            if (arguments.isEmpty()) {
                return empty();
            }
            return new SpecializationKey(arguments);
        }

        public static SpecializationKey empty() {
            return new SpecializationKey(List.of());
        }
    }

    private final Map<Op.Location, Map<SpecializationKey, KernelCallSite>> kernelCallSiteCache = new ConcurrentHashMap<>();

    static OpHelper.Invoke getTargetInvoke(MethodHandles.Lookup lookup, JavaOp.LambdaOp lambdaOp) {
        return lambdaOp.body().entryBlock().ops().stream()
                .filter(ce -> ce instanceof JavaOp.InvokeOp)
                .map(ce -> (OpHelper.Invoke)invoke(lookup, ce))
                .filter(invoke->!invoke.refIs(ComputeContext.class))
                .findFirst()
                .orElseThrow();
    }

    private static class Dispatcher {
        private final Runnable kernelType;

        private Dispatcher(Runnable kernelType) {
            this.kernelType = kernelType;
        }

        private void dispatch(Map<Op.Location, Map<SpecializationKey, KernelCallSite>> kernelCallSiteCache, MethodHandles.Lookup lookup, ComputeCallGraph computeCallGraph, Accelerator accelerator, NDRange ndRange) {
            Quoted<JavaOp.LambdaOp> quoted = Op.ofLambda(kernelType).orElseThrow();
            JavaOp.LambdaOp lambdaOp = quoted.op();
            var location = quoted.op().location();

            MethodRef method = getTargetInvoke(lookup, lambdaOp).op().invokeReference();
            OpHelper.Lambda lambda1 = lambda(lookup, lambdaOp);
            KernelCallSite kernelCallSite;

            try {
                Object[] quotedCapturedValues = lambda1.getQuotedCapturedValues(quoted, method.resolveToMethod(lookup));
                SpecializationKey key = SpecializationKey.of(method.resolveToMethod(lookup), quotedCapturedValues);
                var m = method.resolveToMethod(lookup);
                if (kernelCallSiteCache.containsKey(location) && kernelCallSiteCache.get(location).containsKey(key)) {
                    var oldKernelCallSite = kernelCallSiteCache.get(location).get(key);
                    kernelCallSite = new KernelCallSite(quoted, oldKernelCallSite.lambdaOp(), oldKernelCallSite.methodRef(), oldKernelCallSite.kernelCallGraph(), oldKernelCallSite.capturedArgs());
                } else {
                    kernelCallSite = kernelCallSiteCache.computeIfAbsent(location, k -> new ConcurrentHashMap<>())
                            .computeIfAbsent(key, _ -> {
                                MethodRef methodRef = getTargetInvoke(lookup, lambdaOp).op().invokeReference();
                                KernelCallGraph kernelCallGraph = computeCallGraph.kernelCallGraphMap.get(methodRef);
                                if (kernelCallGraph == null) {
                                    throw new IllegalStateException("Failed to create KernelCallGraph (did you miss @Reflect annotation?).");
                                }
                                // Create a new KernelCallGraph starting from the original method
                                KernelCallGraph kcg = new KernelCallGraph(kernelCallGraph.computeCallGraph, m, kernelCallGraph.getOriginalKernelFunction());

                                var lambda = lambda(lookup, lambdaOp);
                                Object[] capturedArgs = lambda.getQuotedCapturedValues(quoted, kcg.method());

                                // Compilation happens here!
                                kcg.compile(capturedArgs);
                                return new KernelCallSite(quoted, lambdaOp, method, kcg, capturedArgs);
                            });
                }

            } catch (ReflectiveOperationException e) {
                throw new RuntimeException(e);
            }

            Object[] dispatchContextAndArgs = new Object[kernelCallSite.capturedArgs.length + 1];
            System.arraycopy(kernelCallSite.capturedArgs(), 0, dispatchContextAndArgs, 1, kernelCallSite.capturedArgs().length);
            if (kernelType instanceof Kernel) {
                dispatchContextAndArgs[0] = DispatchContext.createDefaultContext(kernelCallSite.kernelCallGraph().computeCallGraph.computeContext.accelerator());
                accelerator.backend.dispatchKernel(kernelCallSite.kernelCallGraph(), ndRange, dispatchContextAndArgs);
            } else if (kernelType instanceof TileKernel) {
                dispatchContextAndArgs[0] = DispatchContext.createTileContext(kernelCallSite.kernelCallGraph().computeCallGraph.computeContext.accelerator());
                accelerator.backend.dispatchTile(kernelCallSite.kernelCallGraph(), ndRange, dispatchContextAndArgs);
            } else {
                throw new IllegalStateException("Unknown KernelType: "  + kernelType);
            }
        }
    }


    /** Creating the kernel callsite involves
         walking the code model of the lambda
         analyzing the callgraph and transforming to HATDialect
     So we cache the callsite against the location from the lambdaop.
     */
    public void dispatchKernel(NDRange ndRange, Kernel kernel) {
        Dispatcher dispatcher = new Dispatcher(kernel);
        dispatcher.dispatch(kernelCallSiteCache, lookup(), computeCallGraph, accelerator, ndRange);
    }

    /**
     * Function to dispatch a TileKernel in HAT. The dispatch takes the following parameters:
     *
     * @param ndRange    A Tile Range that specified the total number of tiles and the tile-size
     * @param tileKernel The tile kernel of offload and run on the hardware accelerator
     */
    public void dispatchTile(NDRange ndRange, TileKernel tileKernel) {
        Dispatcher dispatcher = new Dispatcher(tileKernel);
        dispatcher.dispatch(kernelCallSiteCache, lookup(), computeCallGraph, accelerator, ndRange);
    }

    @Override
    public void preMutate(MappableIface b) {
        if (accelerator.backend instanceof BufferTracker bufferTracker) {
            bufferTracker.preMutate(b);
        }
    }

    @Override
    public void postMutate(MappableIface b) {
        if (accelerator.backend instanceof BufferTracker bufferTracker) {
            bufferTracker.postMutate(b);
        }

    }

    @Override
    public void preAccess(MappableIface b) {
        if (accelerator.backend instanceof BufferTracker bufferTracker) {
            bufferTracker.preAccess(b);
        }

    }

    @Override
    public void postAccess(MappableIface b) {
        if (accelerator.backend instanceof BufferTracker bufferTracker) {
            bufferTracker.postAccess(b);
        }
    }

    @FunctionalInterface
    public interface Kernel extends Runnable { }

    @FunctionalInterface
    public interface TileKernel extends Runnable { }

}

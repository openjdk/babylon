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

import hat.callgraph.ComputeEntrypoint;
import jdk.incubator.code.Location;
import optkl.util.carriers.LookupCarrier;
import optkl.ifacemapper.BufferAllocator;
import optkl.ifacemapper.BufferTracker;
import hat.callgraph.ComputeCallGraph;
import hat.callgraph.KernelCallGraph;
import optkl.ifacemapper.MappableIface;
import jdk.incubator.code.dialect.core.CoreOp.FuncOp;
import jdk.incubator.code.Reflect;
import jdk.incubator.code.Op;
import jdk.incubator.code.Quoted;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.MethodRef;

import java.lang.foreign.Arena;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import java.util.HashMap;
import java.util.Map;
import java.util.function.Consumer;
import java.util.Optional;

import static optkl.Invoke.getTargetInvoke;
import static optkl.Lambda.lambdaOpHelper;

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
public class ComputeContext implements LookupCarrier,BufferAllocator, BufferTracker {


    @Override
    public Arena arena() {
        return accelerator.arena();
    }

    @Override
    public MethodHandles.Lookup lookup() {
        return accelerator.lookup();
    }

    public ComputeEntrypoint computeEntrypoint() {
        return computeCallGraph.entrypoint;
    }

    public Config config() {
        return accelerator().config();
    }

    public enum WRAPPER {
        MUTATE("Mutate"), ACCESS("Access");//, ESCAPE("Escape");
        final public MethodRef pre;
        final public MethodRef post;

        WRAPPER(String name) {
            this.pre = MethodRef.method(ComputeContext.class, "pre" + name, void.class, MappableIface.class);
            this.post = MethodRef.method(ComputeContext.class, "post" + name, void.class, MappableIface.class);
        }
    }

    private  final Accelerator accelerator;
    final  public  Accelerator accelerator(){
        return accelerator;
    }

    private  final ComputeCallGraph computeCallGraph;
    final  public  ComputeCallGraph computeCallGraph(){
        return computeCallGraph;
    }



    /**
     * Called by the Accelerator when the accelerator is passed a compute entrypoint.
     * <p>
     * So given a ComputeClass such as..
     * <pre>
     *  public class MyComputeClass {
     *    @ Reflect
     *    public static void addDeltaKernel(KernelContext kc, S32Array arrayOfInt, int delta) {
     *        arrayOfInt.array(kc.x, arrayOfInt.array(kc.x)+delta);
     *    }
     *
     *    @ Reflect
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
    record KernelCallSite(Quoted quoted, JavaOp.LambdaOp lambdaOp, MethodRef methodRef, KernelCallGraph kernelCallGraph) {}

    private Map<Location, KernelCallSite> kernelCallSiteCache = new HashMap<>();

    /** Creating the kernel callsite involves
         walking the code model of the lambda
         analysing the callgraph and trsnsforming to HATDielect
     So we cache the callsite against the location from the lambdaop.
     */
    public void dispatchKernel(NDRange<?, ?> ndRange, Kernel kernel) {
        Quoted quoted = Op.ofQuotable(kernel).orElseThrow();

        var location = quoted.op().location();

        var kernelCallSite =  kernelCallSiteCache.computeIfAbsent(location, _-> {
            JavaOp.LambdaOp lambdaOp = (JavaOp.LambdaOp) quoted.op();
            MethodRef methodRef = getTargetInvoke(this.lookup(), lambdaOp, KernelContext.class).op().invokeDescriptor();
            KernelCallGraph kernelCallGraph = computeCallGraph.kernelCallGraphMap.get(methodRef);
            if (kernelCallGraph == null) {
                throw new RuntimeException("Failed to create KernelCallGraph (did you miss @Reflect annotation?).");
            }
            return new KernelCallSite(quoted, lambdaOp, methodRef, kernelCallGraph);
        });
        Object[] args = lambdaOpHelper(lookup(),kernelCallSite.lambdaOp).getQuotedCapturedValues(kernelCallSite.quoted, kernelCallSite.kernelCallGraph.entrypoint.method);
        KernelContext kernelContext = accelerator.range(ndRange);
        args[0] = kernelContext;
        accelerator.backend.dispatchKernel(kernelCallSite.kernelCallGraph, kernelContext, args);
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

    @Reflect
    @FunctionalInterface
    public interface Kernel extends Consumer<KernelContext> { }

}

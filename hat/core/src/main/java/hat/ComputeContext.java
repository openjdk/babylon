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

import hat.buffer.Buffer;
import hat.buffer.BufferAllocator;
import hat.buffer.BufferTracker;
import hat.callgraph.ComputeCallGraph;
import hat.callgraph.KernelCallGraph;
import hat.ifacemapper.BoundSchema;
import hat.ifacemapper.SegmentMapper;
import hat.optools.FuncOpWrapper;
import hat.optools.LambdaOpWrapper;
import hat.optools.OpWrapper;
import jdk.incubator.code.Op;
import jdk.incubator.code.Quotable;
import jdk.incubator.code.Quoted;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.MethodRef;

import java.lang.reflect.Method;
import java.util.function.Consumer;

/**
 * A ComputeContext is created by an Accelerator to capture and control compute and kernel
 * callgraphs for the work to be performed by the backend.
 * <p/>
 * The Compute closure is created first, by walking the code model of the entrypoint, then transitively
 * visiting all conventional code reachable from this entrypoint.
 * <p/>
 * Generally all user defined methods reachable from the entrypoint (and the entrypoint intself) must be static methods of the same
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
public class ComputeContext implements BufferAllocator, BufferTracker {


    public enum WRAPPER {
        MUTATE("Mutate"), ACCESS("Access");//, ESCAPE("Escape");
        final public MethodRef pre;
        final public MethodRef post;

        WRAPPER(String name) {
            this.pre = MethodRef.method(ComputeContext.class, "pre" + name, void.class, Buffer.class);
            this.post = MethodRef.method(ComputeContext.class, "post" + name, void.class, Buffer.class);
        }
    }

    public final Accelerator accelerator;


    public final ComputeCallGraph computeCallGraph;

    /**
     * Called by the Accelerator when the accelerator is passed a compute entrypoint.
     * <p>
     * So given a ComputeClass such as..
     * <pre>
     *  public class MyComputeClass {
     *    @ CodeReflection
     *    public static void addDeltaKernel(KernelContext kc, S32Array arrayOfInt, int delta) {
     *        arrayOfInt.array(kc.x, arrayOfInt.array(kc.x)+delta);
     *    }
     *
     *    @ CodeReflection
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

        //   ModuleOpWrapper module = ModuleOpWrapper.createTransitiveInvokeModule(accelerator.lookup, computeMethod);

        // System.out.println(module.op().toText());

        FuncOpWrapper funcOpWrapper = OpWrapper.wrap(accelerator.lookup,Op.ofMethod(computeMethod).orElseThrow());

        this.computeCallGraph = new ComputeCallGraph(this, computeMethod, funcOpWrapper);

        this.computeCallGraph.close();
        this.accelerator.backend.computeContextHandoff(this);
    }

    /**
     * Called from within compute reachable code to dispatch a kernel.
     *
     * @param range
     * @param quotableKernelContextConsumer
     */

    public void dispatchKernel(int range, QuotableKernelContextConsumer quotableKernelContextConsumer) {
        dispatchKernel(range, 0, 0, 1, quotableKernelContextConsumer);
    }

    public void dispatchKernel(int rangeX, int rangeY, QuotableKernelContextConsumer quotableKernelContextConsumer) {
        dispatchKernel(rangeX, rangeY, 0, 2, quotableKernelContextConsumer);
    }

    public void dispatchKernel(int rangeX, int rangeY, int rangeZ, QuotableKernelContextConsumer quotableKernelContextConsumer) {
        dispatchKernel(rangeX, rangeY, rangeZ, 3, quotableKernelContextConsumer);
    }

    private void dispatchKernel(int rangeX, int rangeY, int rangeZ, int dimNumber, QuotableKernelContextConsumer quotableKernelContextConsumer) {
        Quoted quoted = Op.ofQuotable(quotableKernelContextConsumer).orElseThrow();
        LambdaOpWrapper lambdaOpWrapper = OpWrapper.wrap(computeCallGraph.computeContext.accelerator.lookup,(JavaOp.LambdaOp) quoted.op());
        MethodRef methodRef = lambdaOpWrapper.getQuotableTargetMethodRef();
        KernelCallGraph kernelCallGraph = computeCallGraph.kernelCallGraphMap.get(methodRef);
        try {
            Object[] args = lambdaOpWrapper.getQuotableCapturedValues(quoted, kernelCallGraph.entrypoint.method);
            NDRange ndRange;
            switch (dimNumber) {
                case 1 -> ndRange = accelerator.range(rangeX);
                case 2 -> ndRange = accelerator.range(rangeX, rangeY);
                case 3 -> ndRange = accelerator.range(rangeX, rangeY, rangeZ);
                default -> throw new RuntimeException("[Error] Unexpected dimension value: " + dimNumber + ". Allowed dimensions <1, 2, 3>");
            }
            args[0] = ndRange;
            accelerator.backend.dispatchKernel(kernelCallGraph, ndRange, args);
        } catch (Throwable t) {
            System.out.print("what?" + methodRef + " " + t);
            throw t;
        }
    }

    public void preMutate(Buffer b) {
        if (accelerator.backend instanceof BufferTracker bufferTracker) {
            bufferTracker.preMutate(b);
        }
    }

    @Override
    public void postMutate(Buffer b) {
        if (accelerator.backend instanceof BufferTracker bufferTracker) {
            bufferTracker.postMutate(b);
        }

    }

    @Override
    public void preAccess(Buffer b) {
        if (accelerator.backend instanceof BufferTracker bufferTracker) {
            bufferTracker.preAccess(b);
        }

    }

    @Override
    public void postAccess(Buffer b) {
        if (accelerator.backend instanceof BufferTracker bufferTracker) {
            bufferTracker.postAccess(b);
        }

    }
/*
    @Override
    public void preEscape(Buffer b) {
        if (accelerator.backend instanceof BufferTracker bufferTracker) {
            bufferTracker.preEscape(b);
        }
    }

    @Override
    public void postEscape(Buffer b) {
        if (accelerator.backend instanceof BufferTracker bufferTracker) {
            bufferTracker.postEscape(b);
        }
    } */

    @Override
    public <T extends Buffer> T allocate(SegmentMapper<T> segmentMapper, BoundSchema<T> boundSchema) {
        return accelerator.allocate(segmentMapper, boundSchema);
    }

    public interface QuotableKernelContextConsumer extends Quotable, Consumer<KernelContext> {

    }


}

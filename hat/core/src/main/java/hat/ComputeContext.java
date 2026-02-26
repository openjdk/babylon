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
import hat.optools.OpTk;
import jdk.incubator.code.CodeTransformer;
import jdk.incubator.code.Reflect;
import jdk.incubator.code.Op;
import jdk.incubator.code.Quoted;
import jdk.incubator.code.TypeElement;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.JavaType;
import jdk.incubator.code.dialect.java.MethodRef;
import jdk.incubator.code.dialect.java.PrimitiveType;

import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.util.Optional;
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
        this.computeCallGraph = new ComputeCallGraph(this, computeMethod, Op.ofMethod(computeMethod).orElseThrow());
        this.accelerator.backend.computeContextHandoff(this);
    }

    public void dispatchKernel(NDRange<?, ?> ndRange, Kernel kernel) {
        dispatchKernelWithComputeRange(ndRange, kernel);
    }

    /**
     * Function to dispatch a TileKernel in HAT. The dispatch takes the following parameters:
     * @param tileRange
     *  A Tile Range that specified the total number of tiles and the tile-size
     * @param tileKernel
     *  The tile kernel of offload and run on the hardware accelerator
     */
    public void dispatchTile(TileRange tileRange, Tile tileKernel) {
        Quoted quoted = Op.ofQuotable(tileKernel).orElseThrow();
        JavaOp.LambdaOp lambdaOp = (JavaOp.LambdaOp) quoted.op();
        IO.println("Lambda");
        IO.println(lambdaOp.toText());
        MethodRef methodRef = OpTk.getTargetInvokeOp(lambdaOp).invokeDescriptor();
        try {
            Method method = methodRef.resolveToMethod(accelerator.lookup);
            CoreOp.FuncOp funcOp = Op.ofMethod(method).get();
            IO.println("function: ");
            IO.println(funcOp.toText());

            // Analysis of fields to transform into constants
            funcOp = funcOp.transform((blockBuilder, op) -> {
                if (op instanceof JavaOp.FieldAccessOp.FieldLoadOp fieldLoadOp) {
                    boolean isStaticField = fieldLoadOp.operands().isEmpty();
                    if (isStaticField) {
                        blockBuilder.op(fieldLoadOp);
                        TypeElement typeElement = fieldLoadOp.resultType();
                        if (typeElement instanceof PrimitiveType primitiveType) {
                            JavaType basicType = primitiveType.toBasicType();
                            if (basicType == JavaType.INT) {
                                // Found the int field. we can replace it with a constant value
                                try {
                                    Field field = fieldLoadOp.fieldDescriptor().resolveToField(accelerator.lookup);
                                    IO.println(field);
                                    // We can pass null because, at this point, we know it is a static field
                                    int anInt = field.getInt(null);
                                    CoreOp.ConstantOp c = CoreOp.ConstantOp.constant(basicType, anInt);
                                    Op.Result op1 = blockBuilder.op(c);
                                    c.setLocation(fieldLoadOp.location());
                                    blockBuilder.context().mapValue(fieldLoadOp.result(), op1);
                                } catch (ReflectiveOperationException e) {
                                    throw new RuntimeException(e);
                                }
                            }
                        } else {
                            blockBuilder.op(fieldLoadOp);
                        }
                    } else {
                        blockBuilder.op(fieldLoadOp);
                    }
                } else {
                    blockBuilder.op(op);
                }
                return blockBuilder;
            });

            IO.println("Transformed: " + funcOp.toText());
        } catch (ReflectiveOperationException e) {
            throw new RuntimeException(e);
        }

        // TODO: Dispatch the Tile-Range which include JIT Compilation + Execution

    }

    record CallGraph(Quoted quoted, JavaOp.LambdaOp lambdaOp, MethodRef methodRef, KernelCallGraph kernelCallGraph) {}

    private CallGraph getKernelCallGraph(Kernel kernel) {
        Quoted quoted = Op.ofQuotable(kernel).orElseThrow();
        JavaOp.LambdaOp lambdaOp = (JavaOp.LambdaOp) quoted.op();
        MethodRef methodRef = OpTk.getTargetInvokeOp( lambdaOp).invokeDescriptor();
        KernelCallGraph kernelCallGraph = computeCallGraph.kernelCallGraphMap.get(methodRef);
        if (kernelCallGraph == null){
            throw new RuntimeException("Failed to create KernelCallGraph (did you miss @Reflect annotation?) ");
        }
        return new CallGraph(quoted, lambdaOp, methodRef, kernelCallGraph);
    }

    private void dispatchKernelWithComputeRange(NDRange<?, ?> ndRange, Kernel kernel) {
        CallGraph cg = getKernelCallGraph(kernel);
        try {
            Object[] args = OpTk.getQuotedCapturedValues(cg.lambdaOp,cg.quoted, cg.kernelCallGraph.entrypoint.method);
            KernelContext kernelContext = accelerator.range(ndRange);
            args[0] = kernelContext;
            accelerator.backend.dispatchKernel(cg.kernelCallGraph, kernelContext, args);
        } catch (Throwable t) {
            System.out.print("what?" + cg.methodRef + " " + t);
            t.printStackTrace();
            throw t;
        }
    }

    @Override
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

    @Override
    public <T extends Buffer> T allocate(SegmentMapper<T> segmentMapper, BoundSchema<T> boundSchema) {
        return accelerator.allocate(segmentMapper, boundSchema);
    }

    @Reflect
    @FunctionalInterface
    public interface Kernel extends Consumer<KernelContext> { }

    @Reflect
    @FunctionalInterface
    public interface Tile extends Consumer<TileContext> { }

}

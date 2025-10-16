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
package hat.callgraph;

import hat.BufferTagger;
import hat.buffer.Buffer;
import hat.optools.OpTk;
import hat.phases.HATDialectifyTier;
import jdk.incubator.code.*;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.core.VarType;
import jdk.incubator.code.dialect.java.*;

import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import java.util.*;
import java.util.stream.Stream;

public class KernelCallGraph extends CallGraph<KernelEntrypoint> {
    public final ComputeCallGraph computeCallGraph;
    public final Map<MethodRef, MethodCall> bufferAccessToMethodCallMap = new LinkedHashMap<>();
    public final List<BufferTagger.AccessType> bufferAccessList;
    public boolean usesArrayView;

    public interface KernelReachable {
    }

    public static class KernelReachableResolvedMethodCall extends ResolvedMethodCall implements KernelReachable {
        public KernelReachableResolvedMethodCall(CallGraph<KernelEntrypoint> callGraph, MethodRef targetMethodRef, Method method, CoreOp.FuncOp funcOp) {
            super(callGraph, targetMethodRef, method, funcOp);
        }
    }

    public static class KernelReachableUnresolvedMethodCall extends UnresolvedMethodCall implements KernelReachable {
        KernelReachableUnresolvedMethodCall(CallGraph<KernelEntrypoint> callGraph, MethodRef targetMethodRef, Method method) {
            super(callGraph, targetMethodRef, method);
        }
    }


    public static class KernelReachableUnresolvedIfaceMappedMethodCall extends KernelReachableUnresolvedMethodCall {
        KernelReachableUnresolvedIfaceMappedMethodCall(CallGraph<KernelEntrypoint> callGraph, MethodRef targetMethodRef, Method method) {
            super(callGraph, targetMethodRef, method);
        }
    }

    public static class KidAccessor extends MethodCall {
        KidAccessor(CallGraph<KernelEntrypoint> callGraph, MethodRef targetMethodRef, Method method) {
            super(callGraph, targetMethodRef, method);
        }
    }

    public Stream<KernelReachableResolvedMethodCall> kernelReachableResolvedStream() {
        return methodRefToMethodCallMap.values().stream()
                .filter(call -> call instanceof KernelReachableResolvedMethodCall)
                .map(kernelReachable -> (KernelReachableResolvedMethodCall) kernelReachable);
    }

    KernelCallGraph(ComputeCallGraph computeCallGraph, MethodRef methodRef, Method method, CoreOp.FuncOp funcOp) {
        super(computeCallGraph.computeContext, new KernelEntrypoint(null, methodRef, method, funcOp));
        entrypoint.callGraph = this;
        this.computeCallGraph = computeCallGraph;
        bufferAccessList = BufferTagger.getAccessList(computeContext.accelerator.lookup, entrypoint.funcOp());
        usesArrayView = false;
        setModuleOp(OpTk.createTransitiveInvokeModule(computeContext.accelerator.lookup, entrypoint.funcOp(), this));
    }
    /*
     * A ResolvedKernelMethodCall (entrypoint or java  method reachable from a compute entrypojnt)  has the following calls
     * <p>
     * 1) java calls to compute class static functions provided they follow the kernel restrictions
     *    a) we must have the code model available for these and must extend the dag
     * 2) calls to buffer based interface mappings
     *    a) getters (return non void)
     *    b) setters (return void)
     * 3) calls on the NDRange id
     *
    void oldUpdateDag(KernelReachableResolvedMethodCall kernelReachableResolvedMethodCall) {

        var here = OpTk.CallSite.of(KernelCallGraph.class,"updateDag");
        OpTk.traverse(here, kernelReachableResolvedMethodCall.funcOp(), (map, op) -> {
            if (op instanceof JavaOp.InvokeOp invokeOp) {
              //  MethodRef methodRef = invokeOp.invokeDescriptor();
                Class<?> javaRefTypeClass = OpTk.javaRefClassOrThrow(kernelReachableResolvedMethodCall.callGraph.computeContext.accelerator.lookup,invokeOp);
                Method invokeOpCalledMethod = OpTk.methodOrThrow(kernelReachableResolvedMethodCall.callGraph.computeContext.accelerator.lookup,invokeOp);
                if (Buffer.class.isAssignableFrom(javaRefTypeClass)) {
                        kernelReachableResolvedMethodCall.addCall(methodRefToMethodCallMap.computeIfAbsent(invokeOp.invokeDescriptor(), _ ->
                            new KernelReachableUnresolvedIfaceMappedMethodCall(this, invokeOp.invokeDescriptor(), invokeOpCalledMethod)
                    ));
                } else if (entrypoint.method.getDeclaringClass().equals(javaRefTypeClass)) {
                    Optional<CoreOp.FuncOp> optionalFuncOp = Op.ofMethod(invokeOpCalledMethod);
                    if (optionalFuncOp.isPresent()) {
                             kernelReachableResolvedMethodCall.addCall(methodRefToMethodCallMap.computeIfAbsent(invokeOp.invokeDescriptor(), _ ->
                                new KernelReachableResolvedMethodCall(this, invokeOp.invokeDescriptor(), invokeOpCalledMethod, optionalFuncOp.get()
                                )));
                    } else {
                           kernelReachableResolvedMethodCall.addCall(methodRefToMethodCallMap.computeIfAbsent(invokeOp.invokeDescriptor(), _ ->
                                new KernelReachableUnresolvedMethodCall(this, invokeOp.invokeDescriptor(), invokeOpCalledMethod)
                        ));
                    }
                } else {
                       kernelReachableResolvedMethodCall.addCall(methodRefToMethodCallMap.computeIfAbsent(invokeOp.invokeDescriptor(), _ ->
                            new KernelReachableUnresolvedMethodCall(this, invokeOp.invokeDescriptor(), invokeOpCalledMethod)
                    ));
                    // System.out.println("Were we expecting " + methodRef + " here ");
                }
            }
            return map;
        });

        boolean updated = true;
        kernelReachableResolvedMethodCall.closed = true;
        while (updated) {
            updated = false;
            var unclosed = callStream().filter(m -> !m.closed).findFirst();
            if (unclosed.isPresent()) {
                if (unclosed.get() instanceof KernelReachableResolvedMethodCall reachableResolvedMethodCall) {
                    oldUpdateDag(reachableResolvedMethodCall);
                } else {
                    unclosed.get().closed = true;
                }
                updated = true;
            }
        }
    }*/

   // KernelCallGraph close() {
       // oldUpdateDag(entrypoint);
        // now lets sort the MethodCalls into a dependency list
     //   calls.forEach(m -> m.rank = 0);
       // entrypoint.rankRecurse();
       // throw new RuntimeException("is close ever called");
       // return this;
   // }

  //  KernelCallGraph closeWithModuleOp() {

    //    setModuleOp(OpTk.createTransitiveInvokeModule(computeContext.accelerator.lookup, entrypoint.funcOp(), this));
        //calls.forEach(m -> m.rank = 0);
        //entrypoint.rankRecurse();
     //   return this;
   // }

    @Override
    public boolean filterCalls(CoreOp.FuncOp f, JavaOp.InvokeOp invokeOp, Method method, MethodRef methodRef, Class<?> javaRefTypeClass) {
        if (Buffer.class.isAssignableFrom(javaRefTypeClass)) {
            // TODO this side effect seems scary
            bufferAccessToMethodCallMap.computeIfAbsent(methodRef, _ ->
                    new KernelReachableUnresolvedIfaceMappedMethodCall(this, methodRef, method)
            );
        } else {
            return false;
        }
        return true;
    }

    public void dialectifyToHat() {
        // Analysis Phases to transform the Java Code Model to a HAT Code Model

        // Main kernel
        // TODO we should not need the entrypoint handles seprately. !
        //{
            HATDialectifyTier tier = new HATDialectifyTier(computeContext.accelerator);
            CoreOp.FuncOp f = tier.run(entrypoint.funcOp());
            entrypoint.funcOp(f);
       // }
        // Reachable functions
      //  if (moduleOp != null) {
            List<CoreOp.FuncOp> funcs = new ArrayList<>();
            getModuleOp().functionTable().forEach((_, funcOp) -> {
                // ModuleOp is an Immutable Collection, thus, we need to create a new one from a
                // new list of methods
         //       HATDialectifyTier tier = new HATDialectifyTier(computeContext.accelerator);
                CoreOp.FuncOp fn = tier.run(funcOp);
                funcs.add(fn);
            });
            // TODO: can we just replaced moduleOp here.  What if another side table has a prev reference with non transformed funcOps?
             setModuleOp(CoreOp.module(funcs));
        //} else {
          //  throw new IllegalStateException("moduleOp is null");
          /*  kernelReachableResolvedStream().forEach((kernel) -> {
                HatDialectifyTier tier = new HatDialectifyTier(computeContext.accelerator);
                CoreOp.FuncOp f = tier.run(kernel.funcOp());
                kernel.funcOp(f);
            }); */
        //}
    }

    public void convertArrayView() {
        CoreOp.FuncOp entry = convertArrayViewForFunc(computeContext.accelerator.lookup, entrypoint.funcOp());
        entrypoint.funcOp(entry);

       // if (moduleOp != null) {
            List<CoreOp.FuncOp> funcs = new ArrayList<>();
            getModuleOp().functionTable().forEach((_, kernelOp) -> {
                CoreOp.FuncOp f = convertArrayViewForFunc(computeContext.accelerator.lookup, kernelOp);
                funcs.add(f);
            });
            setModuleOp(CoreOp.module(funcs));
       // } else {
         //   kernelReachableResolvedStream().forEach((method) -> {
           //     CoreOp.FuncOp f = convertArrayViewForFunc(computeContext.accelerator.lookup, method.funcOp());
             //   method.funcOp(f);
            //});
       // }
    }

    public CoreOp.FuncOp convertArrayViewForFunc(MethodHandles.Lookup l, CoreOp.FuncOp entry) {
        if (!OpTk.isArrayView(l, entry)) return entry;
        usesArrayView = true;
        // maps a replaced result to the result it should be replaced by
        Map<Op.Result, Op.Result> replaced = new HashMap<>();
        Map<Op, CoreOp.VarAccessOp.VarLoadOp> bufferVarLoads = new HashMap<>();

        return entry.transform(entry.funcName(), (bb, op) -> {
            switch (op) {
                case JavaOp.InvokeOp iop -> {
                    if (OpTk.isBufferArray(iop) &&
                            OpTk.firstOperand(iop) instanceof Op.Result r) { // ensures we can use iop as key for replaced vvv
                        replaced.put(iop.result(), r);
                        bufferVarLoads.put(((Op.Result) OpTk.firstOperand(r.op())).op(), (CoreOp.VarAccessOp.VarLoadOp) r.op()); // map buffer VarOp to its corresponding VarLoadOp
                        return bb;
                    }
                }
                case CoreOp.VarOp vop -> {
                    if (OpTk.isBufferInitialize(vop) &&
                            OpTk.firstOperand(vop) instanceof Op.Result r) { // makes sure we don't process a new int[] for example
                        Op bufferLoad = replaced.get(r).op(); // gets the VarLoadOp associated w/ og buffer
                        replaced.put(vop.result(), (Op.Result) OpTk.firstOperand(bufferLoad)); // gets VarOp associated w/ og buffer
                        return bb;
                    }
                }
                case CoreOp.VarAccessOp.VarLoadOp vlop -> {
                    if (OpTk.isBufferInitialize(vlop) &&
                            OpTk.firstOperand(vlop) instanceof Op.Result r) {
                        if (r.op() instanceof CoreOp.VarOp) { // if this is the VarLoadOp after the .arrayView() InvokeOp
                            Op.Result replacement = (OpTk.notGlobalVarOp(vlop)) ?
                                    (Op.Result) OpTk.firstOperand(((Op.Result) OpTk.firstOperand(r.op())).op()) :
                                    bufferVarLoads.get(replaced.get(r).op()).result();
                            replaced.put(vlop.result(), replacement);
                        } else { // if this is a VarLoadOp loading in the buffer
                            Value loaded = OpTk.getValue(bb, replaced.get(r));
                            Op.Result newVlop = bb.op(CoreOp.VarAccessOp.varLoad(loaded));
                            bb.context().mapValue(vlop.result(), newVlop);
                            replaced.put(vlop.result(), newVlop);
                        }
                        return bb;
                    }
                }
                // handles only 1D and 2D arrays
                case JavaOp.ArrayAccessOp.ArrayLoadOp alop -> {
                    if (OpTk.isBufferArray(alop) &&
                            OpTk.firstOperand(alop) instanceof Op.Result r) {
                        Op.Result buffer = replaced.getOrDefault(r, r);
                        if (((ArrayType) OpTk.firstOperand(op).type()).dimensions() == 1) { // we ignore the first array[][] load if using 2D arrays
                            if (r.op() instanceof JavaOp.ArrayAccessOp.ArrayLoadOp rowOp) {
                                // idea: we want to calculate the idx for the buffer access
                                // idx = (long) (((long) rowOp.idx * (long) buffer.width()) + alop.idx)
                                Op.Result x = (Op.Result) OpTk.getValue(bb, rowOp.operands().getLast());
                                Op.Result y = (Op.Result) OpTk.getValue(bb, alop.operands().getLast());
                                Op.Result ogBufferLoad = replaced.get((Op.Result) OpTk.firstOperand(rowOp));
                                Op.Result ogBuffer = replaced.getOrDefault((Op.Result) OpTk.firstOperand(ogBufferLoad.op()), (Op.Result) OpTk.firstOperand(ogBufferLoad.op()));
                                Op.Result bufferLoad = bb.op(CoreOp.VarAccessOp.varLoad(OpTk.getValue(bb, ogBuffer)));

                                Class<?> c = (Class<?>) OpTk.classTypeToTypeOrThrow(l, (ClassType) ((VarType) ogBuffer.type()).valueType());
                                MethodRef m = MethodRef.method(c, "width", int.class);
                                Op.Result width = bb.op(JavaOp.invoke(m, OpTk.getValue(bb, bufferLoad)));
                                Op.Result longX = bb.op(JavaOp.conv(JavaType.LONG, x));
                                Op.Result longY = bb.op(JavaOp.conv(JavaType.LONG, y));
                                Op.Result longWidth = bb.op(JavaOp.conv(JavaType.LONG, OpTk.getValue(bb, width)));
                                Op.Result mul = bb.op(JavaOp.mul(OpTk.getValue(bb, longY), OpTk.getValue(bb, longWidth)));
                                Op.Result idx = bb.op(JavaOp.add(OpTk.getValue(bb, longX), OpTk.getValue(bb, mul)));

                                Class<?> storedClass = OpTk.primitiveTypeToClass(alop.result().type());
                                MethodRef arrayMethod = MethodRef.method(c, "array", storedClass, long.class);
                                Op.Result invokeRes = bb.op(JavaOp.invoke(arrayMethod, OpTk.getValue(bb, ogBufferLoad), OpTk.getValue(bb, idx)));
                                bb.context().mapValue(alop.result(), invokeRes);
                            } else {
                                JavaOp.ConvOp conv = JavaOp.conv(JavaType.LONG, OpTk.getValue(bb, alop.operands().get(1)));
                                Op.Result convRes = bb.op(conv);

                                Class<?> c = (Class<?>) OpTk.classTypeToTypeOrThrow(l, (ClassType) buffer.type());
                                Class<?> storedClass = OpTk.primitiveTypeToClass(alop.result().type());
                                MethodRef m = MethodRef.method(c, "array", storedClass, long.class);
                                Op.Result invokeRes = bb.op(JavaOp.invoke(m, OpTk.getValue(bb, buffer), convRes));
                                bb.context().mapValue(alop.result(), invokeRes);
                            }
                        }
                    }
                    return bb;
                }
                // handles only 1D and 2D arrays
                case JavaOp.ArrayAccessOp.ArrayStoreOp asop -> {
                    if (OpTk.isBufferArray( asop) &&
                            OpTk.firstOperand(asop) instanceof Op.Result r) {
                        Op.Result buffer = replaced.getOrDefault(r, r);
                        if (((ArrayType) OpTk.firstOperand(op).type()).dimensions() == 1) { // we ignore the first array[][] load if using 2D arrays
                            if (r.op() instanceof JavaOp.ArrayAccessOp.ArrayLoadOp rowOp) {
                                Op.Result x = (Op.Result) rowOp.operands().getLast();
                                Op.Result y = (Op.Result) asop.operands().get(1);
                                Op.Result ogBufferLoad = replaced.get((Op.Result) OpTk.firstOperand(rowOp));
                                Op.Result ogBuffer = replaced.getOrDefault((Op.Result) OpTk.firstOperand(ogBufferLoad.op()), (Op.Result) OpTk.firstOperand(ogBufferLoad.op()));
                                Op.Result bufferLoad = bb.op(CoreOp.VarAccessOp.varLoad(OpTk.getValue(bb, ogBuffer)));
                                Op.Result computed = (Op.Result) asop.operands().getLast();

                                Class<?> c = (Class<?>) OpTk.classTypeToTypeOrThrow(l, (ClassType) ((VarType) ogBuffer.type()).valueType());
                                MethodRef m = MethodRef.method(c, "width", int.class);
                                Op.Result width = bb.op(JavaOp.invoke(m, OpTk.getValue(bb, bufferLoad)));
                                Op.Result longX = bb.op(JavaOp.conv(JavaType.LONG, OpTk.getValue(bb, x)));
                                Op.Result longY = bb.op(JavaOp.conv(JavaType.LONG, OpTk.getValue(bb, y)));
                                Op.Result longWidth = bb.op(JavaOp.conv(JavaType.LONG, OpTk.getValue(bb, width)));
                                Op.Result mul = bb.op(JavaOp.mul(OpTk.getValue(bb, longY), OpTk.getValue(bb, longWidth)));
                                Op.Result idx = bb.op(JavaOp.add(OpTk.getValue(bb, longX), OpTk.getValue(bb, mul)));

                                MethodRef arrayMethod = MethodRef.method(c, "array", void.class, long.class, int.class);
                                Op.Result invokeRes = bb.op(JavaOp.invoke(arrayMethod, OpTk.getValue(bb, ogBufferLoad), OpTk.getValue(bb, idx), OpTk.getValue(bb, computed)));
                                bb.context().mapValue(asop.result(), invokeRes);
                            } else {
                                Op.Result idx = bb.op(JavaOp.conv(JavaType.LONG, OpTk.getValue(bb, asop.operands().get(1))));
                                Value val = OpTk.getValue(bb, asop.operands().getLast());

                                boolean noRootVlop = (buffer.op() instanceof CoreOp.VarOp);
                                ClassType classType = (noRootVlop) ?
                                        (ClassType) ((CoreOp.VarOp) buffer.op()).varValueType() :
                                        (ClassType) buffer.type();

                                Class<?> c = (Class<?>) OpTk.classTypeToTypeOrThrow(l, classType);
                                Class<?> storedClass = OpTk.primitiveTypeToClass(val.type());
                                MethodRef m = MethodRef.method(c, "array", void.class, long.class, storedClass);
                                Op.Result invokeRes = (noRootVlop) ?
                                        bb.op(JavaOp.invoke(m, OpTk.getValue(bb, r), idx, val)) :
                                        bb.op(JavaOp.invoke(m, OpTk.getValue(bb, buffer), idx, val));
                                bb.context().mapValue(asop.result(), invokeRes);
                            }
                        }
                    }
                    return bb;
                }
                case JavaOp.ArrayLengthOp alen -> {
                    if (OpTk.isBufferArray(alen) &&
                            OpTk.firstOperand(alen) instanceof Op.Result r) {
                        Op.Result buffer = replaced.get(r);
                        Class<?> c = (Class<?>) OpTk.classTypeToTypeOrThrow(l, (ClassType) buffer.type());
                        MethodRef m = MethodRef.method(c, "length", int.class);
                        Op.Result invokeRes = bb.op(JavaOp.invoke(m, OpTk.getValue(bb, buffer)));
                        bb.context().mapValue(alen.result(), invokeRes);
                    }
                    return bb;
                }
                default -> {}
            }
            bb.op(op);
            return bb;
        });
    }
}

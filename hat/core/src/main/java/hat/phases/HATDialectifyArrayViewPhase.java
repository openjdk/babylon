/*
 * Copyright (c) 2025, Oracle and/or its affiliates. All rights reserved.
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
package hat.phases;

import hat.Accelerator;
import hat.ComputeContext;
import hat.dialect.*;
import hat.ifacemapper.MappableIface;
import hat.optools.OpTk;
import jdk.incubator.code.Block;
import jdk.incubator.code.Op;
import jdk.incubator.code.TypeElement;
import jdk.incubator.code.Value;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.core.VarType;
import jdk.incubator.code.dialect.java.*;

import java.lang.invoke.MethodHandles;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

import static hat.optools.OpTk.elements;
import static hat.optools.OpTk.isAssignable;
import static jdk.incubator.code.dialect.core.CoreType.varType;

public class HATDialectifyArrayViewPhase implements HATDialect {

    protected final Accelerator accelerator;
    // TODO: account for different indices
    Map<CoreOp.VarOp, HATVectorVarOp> vectorVarOps = new HashMap<>();
    @Override  public Accelerator accelerator(){
        return this.accelerator;
    }

    public HATDialectifyArrayViewPhase(Accelerator accelerator) {
        this.accelerator = accelerator;
    }

    @Override
    public CoreOp.FuncOp apply(CoreOp.FuncOp funcOp) {
        if (accelerator.backend.config().showCompilationPhases()) {
            System.out.println("[INFO] Code model before HATDialectifyArrayViewPhase: " + funcOp.toText());
        }
        funcOp = run(funcOp);
        if (accelerator.backend.config().showCompilationPhases()) {
            System.out.println("[INFO] Code model after HATDialectifyArrayViewPhase: " + funcOp.toText());
        }
        return funcOp;
    }

    public enum ArrayViewType {
        // GLOBAL("global"),
        // LOCAL("local"),
        PRIVATE("private");

        final String arrayType;
        ArrayViewType(String arrayType) {
            this.arrayType = arrayType;
        }
    }

    public CoreOp.FuncOp run(CoreOp.FuncOp entry) {
        MethodHandles.Lookup l = accelerator.lookup;
        if (!isArrayView(l, entry)) return entry;
        // maps a replaced result to the result it should be replaced by
        Map<Op.Result, Op.Result> replaced = new HashMap<>();
        Map<Op, CoreOp.VarAccessOp.VarLoadOp> bufferVarLoads = new HashMap<>();

        return entry.transform(entry.funcName(), (bb, op) -> {
            // if (bufferAlreadyLoaded(op)) {
            //     return bb;
            // }
            switch (op) {
                case JavaOp.InvokeOp iop -> {
                    if (isVectorOperation(iop)) {
                        // TODO: might need to fix
                        HATVectorViewOp memoryViewOp = buildVectorBinaryOp(iop.invokeDescriptor().name(), iop.externalizeOpName(), iop.resultType(), bb.context().getValues(iop.operands()));
                        Op.Result hatVectorOpResult = bb.op(memoryViewOp);
                        memoryViewOp.setLocation(iop.location());
                        bb.context().mapValue(iop.result(), hatVectorOpResult);
                        replaced.put(iop.result(), hatVectorOpResult);
                        return bb;
                    } else if (isLaneOperation(iop)) {
                        String name = iop.invokeDescriptor().name();
                        Op.Result res = (iop.resultType() == JavaType.VOID) ?
                                vectorSelectStoreOp(iop, name, bb) :
                                vectorSelectLoadOp(iop, name, bb);
                        bb.context().mapValue(iop.result(), res);
                        return bb;
                    } else if (isBufferArray(iop) &&
                                firstOperand(iop) instanceof Op.Result r)
                    { // ensures we can use iop as key for replaced vvv
                        replaced.put(iop.result(), r);
                        bufferVarLoads.put(((Op.Result) firstOperand(r.op())).op(), (CoreOp.VarAccessOp.VarLoadOp) r.op()); // map buffer VarOp to its corresponding VarLoadOp
                        return bb;
                    }
                }
                case CoreOp.VarOp vop -> {
                    if (isBufferInitialize(vop) &&
                            firstOperand(vop) instanceof Op.Result r) { // makes sure we don't process a new int[] for example
                        Op bufferLoad = replaced.get(r).op(); // gets the VarLoadOp associated w/ og buffer
                        replaced.put(vop.result(), (Op.Result) firstOperand(bufferLoad)); // gets VarOp associated w/ og buffer
                        return bb;
                    } else if (isFloat4Op(vop)) {
                        List<Value> inputOperandsVarOp = (vop.operands().isEmpty()) ? List.of() : List.of(firstOperand(vop));
                        HATVectorViewOp memoryViewOp = new HATVectorVarOp(vop.varName(), vop.resultType(), 4, bb.context().getValues(inputOperandsVarOp));
                        Op.Result hatLocalResult = bb.op(memoryViewOp);
                        memoryViewOp.setLocation(vop.location());
                        bb.context().mapValue(vop.result(), hatLocalResult);
                        return bb;
                    }
                }
                case CoreOp.VarAccessOp.VarLoadOp vlop -> {
                    // TODO: clean
                    if ((isBufferInitialize(vlop)) &&
                            firstOperand(vlop) instanceof Op.Result r) {
                        // if (isFloat4Op(vlop)) {
                        //     HATVectorViewOp memoryViewOp = (vectorVarOps.containsKey(vlop.varOp()) ?
                        //             new HATVectorVarLoadOp(vlop.varOp().varName(), vlop.resultType(), List.of(vectorVarOps.get(vlop.varOp()).result())) :
                        //             new HATVectorVarLoadOp(vlop.varOp().varName(), vlop.resultType(), bb.context().getValues(bb.context().getValues(vlop.operands()))));
                        //     Op.Result hatLocalResult = bb.op(memoryViewOp);
                        //     memoryViewOp.setLocation(vlop.location());
                        //     bb.context().mapValue(vlop.result(), hatLocalResult);
                        //     return bb;
                        // }
                        if (r.op() instanceof CoreOp.VarOp) { // if this is the VarLoadOp after the .arrayView() InvokeOp
                            Op.Result replacement = (notGlobalVarOp(vlop)) ?
                                    (Op.Result) firstOperand(((Op.Result) firstOperand(r.op())).op()) :
                                    bufferVarLoads.get(replaced.get(r).op()).result();
                            replaced.put(vlop.result(), replacement);
                        } else { // if this is a VarLoadOp loading in the buffer
                            Value loaded = getValue(bb, replaced.get(r));
                            Op.Result newVlop = bb.op(CoreOp.VarAccessOp.varLoad(loaded));
                            bb.context().mapValue(vlop.result(), newVlop);
                            replaced.put(vlop.result(), newVlop);
                        }
                        return bb;
                    }
                }
                // handles only 1D and 2D arrays
                case JavaOp.ArrayAccessOp.ArrayLoadOp alop -> {
                    if (isBufferArray(alop) &&
                            firstOperand(alop) instanceof Op.Result r) {
                        Op.Result buffer = replaced.getOrDefault(r, r);
                        if (isFloat4Op(alop)) {
                            CoreOp.VarOp vop = (CoreOp.VarOp) ((Op.Result) firstOperand(buffer.op())).op();
                            if (vectorVarOps.containsKey(findVarOp(alop))) {
                                bb.context().mapValue(alop.result(), vectorVarOps.get(findVarOp(alop)).result());
                                return bb;
                            }
                            HATVectorLoadOp vectorLoadOp = new HATVectorLoadOp(
                                    vop.varName(),
                                    varType(((ArrayType) alop.operands().getFirst().type()).componentType()),
                                    ((ArrayType) alop.operands().getFirst().type()).componentType(),
                                    4,
                                    false, //TODO: fix
                                    bb.context().getValues(List.of(buffer, alop.operands().getLast()))
                            );
                            Op.Result hatLocalResult = bb.op(vectorLoadOp);
                            vectorLoadOp.setLocation(alop.location());
                            bb.context().mapValue(alop.result(), hatLocalResult);

                            if (alop.result().uses().stream().noneMatch(
                                    res -> res.op() instanceof CoreOp.VarOp || res.op() instanceof HATVectorBinaryOp ||
                                            (res.op() instanceof JavaOp.InvokeOp iop && isVectorOperation(iop)))) {
                                List<Value> inputOperandsVarOp = List.of(hatLocalResult);
                                HATVectorVarOp memoryViewOp = new HATVectorVarOp(findVarOp(alop).varName(), (VarType) vectorLoadOp.resultType(), 4, inputOperandsVarOp);
                                memoryViewOp.setLocation(alop.location());
                                Op.Result hatLocalResultVop = bb.op(memoryViewOp);
                                vectorVarOps.put(findVarOp(alop), memoryViewOp);
                                bb.context().mapValue(alop.result(), hatLocalResultVop);
                                return bb;
                            }
                            return bb;
                        }
                        if (((ArrayType) firstOperand(op).type()).dimensions() == 1) { // we ignore the first array[][] load if using 2D arrays
                            if (r.op() instanceof JavaOp.ArrayAccessOp.ArrayLoadOp rowOp) {
                                // idea: we want to calculate the idx for the buffer access
                                // idx = (long) (((long) rowOp.idx * (long) buffer.width()) + alop.idx)
                                Op.Result x = (Op.Result) getValue(bb, rowOp.operands().getLast());
                                Op.Result y = (Op.Result) getValue(bb, alop.operands().getLast());
                                Op.Result ogBufferLoad = replaced.get((Op.Result) firstOperand(rowOp));
                                Op.Result ogBuffer = replaced.getOrDefault((Op.Result) firstOperand(ogBufferLoad.op()), (Op.Result) firstOperand(ogBufferLoad.op()));
                                Op.Result bufferLoad = bb.op(CoreOp.VarAccessOp.varLoad(getValue(bb, ogBuffer)));

                                Class<?> c = (Class<?>) OpTk.classTypeToTypeOrThrow(l, (ClassType) ((VarType) ogBuffer.type()).valueType());
                                MethodRef m = MethodRef.method(c, "width", int.class);
                                Op.Result width = bb.op(JavaOp.invoke(m, getValue(bb, bufferLoad)));
                                Op.Result longX = bb.op(JavaOp.conv(JavaType.LONG, x));
                                Op.Result longY = bb.op(JavaOp.conv(JavaType.LONG, y));
                                Op.Result longWidth = bb.op(JavaOp.conv(JavaType.LONG, getValue(bb, width)));
                                Op.Result mul = bb.op(JavaOp.mul(getValue(bb, longY), getValue(bb, longWidth)));
                                Op.Result idx = bb.op(JavaOp.add(getValue(bb, longX), getValue(bb, mul)));

                                Class<?> storedClass = typeElementToClass(l, alop.result().type());
                                MethodRef arrayMethod = MethodRef.method(c, "array", storedClass, long.class);
                                Op.Result invokeRes = bb.op(JavaOp.invoke(arrayMethod, getValue(bb, ogBufferLoad), getValue(bb, idx)));
                                bb.context().mapValue(alop.result(), invokeRes);
                            } else {
                                JavaOp.ConvOp conv = JavaOp.conv(JavaType.LONG, getValue(bb, alop.operands().get(1)));
                                Op.Result convRes = bb.op(conv);

                                Class<?> c = (Class<?>) OpTk.classTypeToTypeOrThrow(l, (ClassType) buffer.type());
                                Class<?> storedClass = typeElementToClass(l, alop.result().type());
                                MethodRef m = MethodRef.method(c, "array", storedClass, long.class);
                                Op.Result invokeRes = bb.op(JavaOp.invoke(m, getValue(bb, buffer), convRes));
                                bb.context().mapValue(alop.result(), invokeRes);
                            }
                        }
                    }
                    return bb;
                }
                // handles only 1D and 2D arrays
                case JavaOp.ArrayAccessOp.ArrayStoreOp asop -> {
                    if (isBufferArray( asop) &&
                            firstOperand(asop) instanceof Op.Result r) {
                        Op.Result buffer = replaced.getOrDefault(r, r);
                        if (isFloat4Op(asop)) {
                            CoreOp.VarOp vop = findVarOp(((Op.Result) asop.operands().getLast()).op());
                            HATVectorStoreView vectorStoreOp = new HATVectorStoreView(
                                    vop.varName(),
                                    vop.resultType(),
                                    4,
                                    HATVectorViewOp.VectorType.FLOAT4,
                                    false, //TODO: fix
                                    bb.context().getValues(List.of(buffer, asop.operands().getLast(), asop.operands().get(1)))
                            );
                            Op.Result hatLocalResult = bb.op(vectorStoreOp);
                            vectorStoreOp.setLocation(asop.location());
                            bb.context().mapValue(asop.result(), hatLocalResult);
                            return bb;
                        }
                        if (((ArrayType) firstOperand(op).type()).dimensions() == 1) { // we ignore the first array[][] load if using 2D arrays
                            if (r.op() instanceof JavaOp.ArrayAccessOp.ArrayLoadOp rowOp) {
                                Op.Result x = (Op.Result) rowOp.operands().getLast();
                                Op.Result y = (Op.Result) asop.operands().get(1);
                                Op.Result ogBufferLoad = replaced.get((Op.Result) firstOperand(rowOp));
                                Op.Result ogBuffer = replaced.getOrDefault((Op.Result) firstOperand(ogBufferLoad.op()), (Op.Result) firstOperand(ogBufferLoad.op()));
                                Op.Result bufferLoad = bb.op(CoreOp.VarAccessOp.varLoad(getValue(bb, ogBuffer)));
                                Op.Result computed = (Op.Result) asop.operands().getLast();

                                Class<?> c = (Class<?>) OpTk.classTypeToTypeOrThrow(l, (ClassType) ((VarType) ogBuffer.type()).valueType());
                                MethodRef m = MethodRef.method(c, "width", int.class);
                                Op.Result width = bb.op(JavaOp.invoke(m, getValue(bb, bufferLoad)));
                                Op.Result longX = bb.op(JavaOp.conv(JavaType.LONG, getValue(bb, x)));
                                Op.Result longY = bb.op(JavaOp.conv(JavaType.LONG, getValue(bb, y)));
                                Op.Result longWidth = bb.op(JavaOp.conv(JavaType.LONG, getValue(bb, width)));
                                Op.Result mul = bb.op(JavaOp.mul(getValue(bb, longY), getValue(bb, longWidth)));
                                Op.Result idx = bb.op(JavaOp.add(getValue(bb, longX), getValue(bb, mul)));

                                MethodRef arrayMethod = MethodRef.method(c, "array", void.class, long.class, int.class);
                                Op.Result invokeRes = bb.op(JavaOp.invoke(arrayMethod, getValue(bb, ogBufferLoad), getValue(bb, idx), getValue(bb, computed)));
                                bb.context().mapValue(asop.result(), invokeRes);
                            } else {
                                Op.Result idx = bb.op(JavaOp.conv(JavaType.LONG, getValue(bb, asop.operands().get(1))));
                                Value val = getValue(bb, asop.operands().getLast());

                                boolean noRootVlop = (buffer.op() instanceof CoreOp.VarOp);
                                ClassType classType = (noRootVlop) ?
                                        (ClassType) ((CoreOp.VarOp) buffer.op()).varValueType() :
                                        (ClassType) buffer.type();

                                Class<?> c = (Class<?>) OpTk.classTypeToTypeOrThrow(l, classType);
                                Class<?> storedClass = typeElementToClass(l, val.type());
                                MethodRef m = MethodRef.method(c, "array", void.class, long.class, storedClass);
                                Op.Result invokeRes = (noRootVlop) ?
                                        bb.op(JavaOp.invoke(m, getValue(bb, r), idx, val)) :
                                        bb.op(JavaOp.invoke(m, getValue(bb, buffer), idx, val));
                                bb.context().mapValue(asop.result(), invokeRes);
                            }
                        }
                    }
                    return bb;
                }
                case JavaOp.ArrayLengthOp alen -> {
                    if (isBufferArray(alen) &&
                            firstOperand(alen) instanceof Op.Result r) {
                        Op.Result buffer = replaced.get(r);
                        Class<?> c = (Class<?>) OpTk.classTypeToTypeOrThrow(l, (ClassType) buffer.type());
                        MethodRef m = MethodRef.method(c, "length", int.class);
                        Op.Result invokeRes = bb.op(JavaOp.invoke(m, getValue(bb, buffer)));
                        bb.context().mapValue(alen.result(), invokeRes);
                    }
                    return bb;
                }
                case HATVectorSelectLoadOp vSelectLoad -> {
                    bb.context().mapValue(vSelectLoad.result(), vectorSelectLoadOp(vSelectLoad, vSelectLoad.mapLane(), bb));
                    return bb;
                }
                case HATVectorSelectStoreOp vSelectStore -> {
                    bb.context().mapValue(vSelectStore.result(), vectorSelectStoreOp(vSelectStore, vSelectStore.mapLane(), bb));
                    return bb;
                }
                default -> {}
            }
            bb.op(op);
            return bb;
        });

        // if (!isArrayView(l, entry)) return entry;
        // // maps a replaced result to the result it should be replaced by
        // Map<Op.Result, Op.Result> replaced = new HashMap<>();
        // Map<Op, CoreOp.VarAccessOp.VarLoadOp> bufferVarLoads = new HashMap<>();
        //
        // return entry.transform(entry.funcName(), (bb, op) -> {
        //     switch (op) {
        //         case JavaOp.InvokeOp iop -> {
        //             if (isBufferArray(iop) &&
        //                     firstOperand(iop) instanceof Op.Result r) { // ensures we can use iop as key for replaced vvv
        //                 replaced.put(iop.result(), r);
        //                 bufferVarLoads.put(((Op.Result) firstOperand(r.op())).op(), (CoreOp.VarAccessOp.VarLoadOp) r.op()); // map buffer VarOp to its corresponding VarLoadOp
        //                 return bb;
        //             }
        //         }
        //         case CoreOp.VarOp vop -> {
        //             if (isBufferInitialize(vop) &&
        //                     firstOperand(vop) instanceof Op.Result r) { // makes sure we don't process a new int[] for example
        //                 Op bufferLoad = replaced.get(r).op(); // gets the VarLoadOp associated w/ og buffer
        //                 replaced.put(vop.result(), (Op.Result) firstOperand(bufferLoad)); // gets VarOp associated w/ og buffer
        //                 return bb;
        //             }
        //         }
        //         case CoreOp.VarAccessOp.VarLoadOp vlop -> {
        //             if (isBufferInitialize(vlop) &&
        //                     firstOperand(vlop) instanceof Op.Result r) {
        //                 if (r.op() instanceof CoreOp.VarOp) { // if this is the VarLoadOp after the .arrayView() InvokeOp
        //                     Op.Result replacement = (notGlobalVarOp(vlop)) ?
        //                             (Op.Result) firstOperand(((Op.Result) firstOperand(r.op())).op()) :
        //                             bufferVarLoads.get(replaced.get(r).op()).result();
        //                     replaced.put(vlop.result(), replacement);
        //                 } else { // if this is a VarLoadOp loading in the buffer
        //                     Value loaded = getValue(bb, replaced.get(r));
        //                     Op.Result newVlop = bb.op(CoreOp.VarAccessOp.varLoad(loaded));
        //                     bb.context().mapValue(vlop.result(), newVlop);
        //                     replaced.put(vlop.result(), newVlop);
        //                 }
        //                 return bb;
        //             }
        //         }
        //         // handles only 1D and 2D arrays
        //         case JavaOp.ArrayAccessOp.ArrayLoadOp alop -> {
        //             if (isBufferArray(alop) &&
        //                     firstOperand(alop) instanceof Op.Result r) {
        //                 Op.Result buffer = replaced.getOrDefault(r, r);
        //                 // if (((Op.Result) firstOperand(op)).op() instanceof JavaOp.ArrayAccessOp) {
        //                 if (((ArrayType) firstOperand(op).type()).dimensions() == 1) {
        //                     List<Value> operands = new ArrayList<>();
        //                     // TODO: add the corresponding ops to operands
        //                     // operands.add(alop.operands().getFirst());
        //                     if (r.op() instanceof JavaOp.ArrayAccessOp.ArrayLoadOp rowOp) {
        //                         // for (Value value : alop.operands()) {
        //                             if (alop.operands().getLast().type() != JavaType.LONG) {
        //                                 JavaOp.ConvOp conv = JavaOp.conv(JavaType.LONG, getValue(bb, alop.operands().get(1)));
        //                                 operands.add(bb.op(conv));
        //                             } else {
        //                                 operands.add(alop.operands().getLast());
        //                             }
        //                         // }
        //                     } else {
        //                         if (alop.operands().getLast().type() != JavaType.LONG) {
        //                             JavaOp.ConvOp conv = JavaOp.conv(JavaType.LONG, getValue(bb, alop.operands().get(1)));
        //                             operands.add(bb.op(conv));
        //                         }
        //                     }
        //                     HATArrayViewLoadOp load = new HATArrayViewLoadOp(
        //                          bufferName(alop),
        //                          buffer.type(),
        //                          operands
        //                     );
        //                     Op.Result res = bb.op(load);
        //                     bb.context().mapValue(alop.result(), res);
        //                 } else {
        //                     System.out.println("aguosiadgoidus");
        //                 }
        //             }
        //             return bb;
        //         }
        //         // handles only 1D and 2D arrays
        //         case JavaOp.ArrayAccessOp.ArrayStoreOp asop -> {
        //             if (isBufferArray(asop) &&
        //                     firstOperand(asop) instanceof Op.Result r) {
        //                 Op.Result buffer = replaced.getOrDefault(r, r);
        //                 if (((ArrayType) firstOperand(op).type()).dimensions() == 1) { // we ignore the first array[][] load if using 2D arrays
        //                     List<Value> operands = new ArrayList<>();
        //                     // TODO: add the corresponding ops to operands
        //                     // operands.add(alop.operands().getFirst());
        //                     if (r.op() instanceof JavaOp.ArrayAccessOp.ArrayLoadOp rowOp) {
        //                         // for (Value value : alop.operands()) {
        //                         if (asop.operands().getLast().type() != JavaType.LONG) {
        //                             JavaOp.ConvOp conv = JavaOp.conv(JavaType.LONG, getValue(bb, asop.operands().get(1)));
        //                             operands.add(bb.op(conv));
        //                         } else {
        //                             operands.add(asop.operands().getLast());
        //                         }
        //                         // }
        //                     } else {
        //                         if (asop.operands().getLast().type() != JavaType.LONG) {
        //                             JavaOp.ConvOp conv = JavaOp.conv(JavaType.LONG, getValue(bb, asop.operands().get(1)));
        //                             operands.add(bb.op(conv));
        //                         }
        //                     }
        //                     HATArrayViewLoadOp load = new HATArrayViewLoadOp(
        //                             bufferName(l, asop),
        //                             buffer.type(),
        //                             operands
        //                     );
        //                     Op.Result res = bb.op(load);
        //                     bb.context().mapValue(asop.result(), res);
        //                 }
        //             }
        //             return bb;
        //         }
        //         case JavaOp.ArrayLengthOp alen -> {
        //             if (isBufferArray(alen) &&
        //                     firstOperand(alen) instanceof Op.Result r) {
        //                 Op.Result buffer = replaced.get(r);
        //
        //                 List<Value> operands = new ArrayList<>();
        //                 HATArrayViewLengthOp load = new HATArrayViewLengthOp(
        //                         ((CoreOp.VarOp) ((Op.Result) firstOperand(buffer.op())).op()).varName(),
        //                         buffer.type(),
        //                         operands
        //                 );
        //
        //                 Op.Result res = bb.op(load);
        //                 bb.context().mapValue(alen.result(), res);
        //
        //                 // Class<?> c = (Class<?>) classTypeToTypeOrThrow(l, (ClassType) buffer.type());
        //                 // MethodRef m = MethodRef.method(c, "length", int.class);
        //                 // Op.Result invokeRes = bb.op(JavaOp.invoke(m, getValue(bb, buffer)));
        //                 // bb.context().mapValue(alen.result(), invokeRes);
        //             }
        //             return bb;
        //         }
        //         default -> {}
        //     }
        //     bb.op(op);
        //     return bb;
        // });
    }

    boolean bufferAlreadyLoaded(Op op) {
        Optional<Op.Result> possibleRemove = op.result().uses().stream().filter(res -> res.op() instanceof JavaOp.ArrayAccessOp.ArrayLoadOp).findFirst();
        if (possibleRemove.isPresent() && vectorVarOps.containsKey(findVarOp(possibleRemove.get().op()))) {
            return true;
        }
        return false;
    }

    Op.Result vectorSelectLoadOp(Op op, String laneName, Block.Builder bb) {
        CoreOp.VarOp vop = findVarOp(((Op.Result) op.operands().getFirst()).op());
        String name = (vop == null) ? "" : vop.varName();
        int lane = getLane(laneName);
        HATVectorViewOp vSelectOp = (vectorVarOps.containsKey(vop)) ?
                new HATVectorSelectLoadOp(name, op.resultType(), lane, List.of(vectorVarOps.get(vop).result())) :
                new HATVectorSelectLoadOp(name, op.resultType(), lane, bb.context().getValues(op.operands()));
        vSelectOp.setLocation(op.location());
        return bb.op(vSelectOp);
    }

    Op.Result vectorSelectStoreOp(Op op, String laneName, Block.Builder bb) {
        CoreOp.VarOp vop = findVarOp(((Op.Result) op.operands().getFirst()).op());
        String name = (vop == null) ? "" : vop.varName();
        int lane = getLane(laneName);
        CoreOp.VarOp resultOp = (((Op.Result) op.operands().getLast()).op() instanceof JavaOp.ArithmeticOperation) ?
                null :
                findVarOp(((Op.Result) bb.context().getValue(op.operands().get(1))).op());
        HATVectorViewOp vSelectOp = (vectorVarOps.containsKey(vop)) ?
                new HATVectorSelectStoreOp(name, op.resultType(), lane, resultOp, List.of(vectorVarOps.get(vop).result(), bb.context().getValue(op.operands().getLast()))) :
                new HATVectorSelectStoreOp(name, op.resultType(), lane, resultOp, bb.context().getValues(op.operands()));
        vSelectOp.setLocation(op.location());
        return bb.op(vSelectOp);
    }

    /*
     * TODO: replace new var methods vvv
     */

    int getLane(String fieldName) {
        return switch (fieldName) {
            case "x" -> 0;
            case "y" -> 1;
            case "z" -> 2;
            case "w" -> 3;
            default -> -1;
        };
    }

    private CoreOp.VarOp findVarOp(Op op) {
        while (!(op instanceof CoreOp.VarOp vop)) {
            if (!op.operands().isEmpty() && firstOperand(op) instanceof Op.Result r) {
                op = r.op();
            } else {
                return null;
            }
        }
        return vop;
    }

    private HATVectorBinaryOp buildVectorBinaryOp(String opType, String varName, TypeElement resultType, List<Value> outputOperands) {
        return switch (opType) {
            case "add" -> new HATVectorAddOp(varName, resultType, outputOperands);
            case "sub" -> new HATVectorSubOp(varName, resultType, outputOperands);
            case "mul" -> new HATVectorMulOp(varName, resultType, outputOperands);
            case "div" -> new HATVectorDivOp(varName, resultType, outputOperands);
            default -> throw new IllegalStateException("Unexpected value: " + opType);
        };
    }

    private boolean isLaneOperation(JavaOp.InvokeOp invokeOp) {
        return OpTk.isIfaceBufferMethod(accelerator.lookup, invokeOp)
                // TODO: CHANGE PLEASE
                && invokeOp.invokeDescriptor().name().length() == 1;
    }

    private boolean isVectorOperation(JavaOp.InvokeOp invokeOp) {
        TypeElement typeElement = invokeOp.resultType();
        boolean isHatVectorType = typeElement.toString().startsWith("hat.buffer.Float");
        return isHatVectorType
                && OpTk.isIfaceBufferMethod(accelerator.lookup, invokeOp)
                // TODO: CHANGE PLEASE
                && invokeOp.invokeDescriptor().name().length() == 3;
    }

    public static boolean isFloat4Op(Op op) {
        return (op.resultType().toString().contains("Float4") || (!op.operands().isEmpty() && op.operands().getFirst().type().toString().contains("Float4")));
    }


    /*
     * ^^^ TODO: replace the methods above
     */


    public static Value firstOperand(Op op) {
        return op.operands().getFirst();
    }

    public static Value getValue(Block.Builder bb, Value value) {
        return bb.context().getValueOrDefault(value, value);
    }

    public static boolean isBufferArray(Op op) {
        // first check if the return is an array type
        //if (op instanceof CoreOp.VarOp vop) {
        //    if (!(vop.varValueType() instanceof ArrayType)) return false;
        //} else if (!(op instanceof JavaOp.ArrayAccessOp)){
        //    if (!(op.resultType() instanceof ArrayType)) return false;
        //}

        // then check if returned array is from a buffer access
        while (!(op instanceof JavaOp.InvokeOp iop)) {
            if (!op.operands().isEmpty() && firstOperand(op) instanceof Op.Result r) {
                op = r.op();
            } else {
                return false;
            }
        }

        //if (iop.invokeDescriptor().refType() instanceof JavaType javaType) {
        //    return isAssignable(l, javaType, MappableIface.class);
        //}
        //return false;
        return iop.invokeDescriptor().name().toLowerCase().contains("arrayview");
    }

    public static boolean notGlobalVarOp(Op op) {
        while (!(op instanceof JavaOp.InvokeOp iop)) {
            if (!op.operands().isEmpty() && firstOperand(op) instanceof Op.Result r) {
                op = r.op();
            } else {
                return false;
            }
        }

        return iop.invokeDescriptor().name().toLowerCase().contains("local") ||
                iop.invokeDescriptor().name().toLowerCase().contains("private");
    }

    public static boolean isBufferInitialize(Op op) {
        // first check if the return is an array type
        if (op instanceof CoreOp.VarOp vop) {
            if (!(vop.varValueType() instanceof ArrayType)) return false;
        } else if (!(op instanceof JavaOp.ArrayAccessOp)){
            if (!(op.resultType() instanceof ArrayType)) return false;
        }

        return isBufferArray(op);
    }

    public static boolean isArrayView(MethodHandles.Lookup lookup, CoreOp.FuncOp entry) {
        var here = OpTk.CallSite.of(HATDialectifyArrayViewPhase.class,"isArrayView");
        return elements(here,entry).anyMatch((element) -> (
                element instanceof JavaOp.InvokeOp iop &&
                        iop.resultType() instanceof ArrayType &&
                        iop.invokeDescriptor().refType() instanceof JavaType javaType &&
                        isAssignable(lookup, javaType, MappableIface.class)));
    }

    public static Class<?> typeElementToClass(MethodHandles.Lookup l, TypeElement type) {
        class PrimitiveHolder {
            static final Map<PrimitiveType, Class<?>> primitiveToClass = Map.of(
                    JavaType.BYTE, byte.class,
                    JavaType.SHORT, short.class,
                    JavaType.INT, int.class,
                    JavaType.LONG, long.class,
                    JavaType.FLOAT, float.class,
                    JavaType.DOUBLE, double.class,
                    JavaType.CHAR, char.class,
                    JavaType.BOOLEAN, boolean.class
            );
        }
        try {
            if (type instanceof PrimitiveType primitiveType) {
                return PrimitiveHolder.primitiveToClass.get(primitiveType);
            } else if (type instanceof ClassType classType) {
                return ((Class<?>) classType.resolve(l));
            } else {
                throw new IllegalArgumentException("given type cannot be converted to class");
            }
        } catch (ReflectiveOperationException e) {
            throw new RuntimeException("given type cannot be converted to class");
        }
    }
}

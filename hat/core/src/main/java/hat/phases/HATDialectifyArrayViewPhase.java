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
import hat.dialect.*;
import hat.ifacemapper.MappableIface;
import hat.optools.OpTk;
import hat.types._V;
import jdk.incubator.code.Block;
import jdk.incubator.code.Op;
import jdk.incubator.code.TypeElement;
import jdk.incubator.code.Value;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.core.CoreType;
import jdk.incubator.code.dialect.core.VarType;
import jdk.incubator.code.dialect.java.*;

import java.lang.invoke.MethodHandles;
import java.util.*;

public class HATDialectifyArrayViewPhase implements HATDialect {

    protected final Accelerator accelerator;
    @Override
    public Accelerator accelerator() {
        return this.accelerator;
    }

    public HATDialectifyArrayViewPhase(Accelerator accelerator) {
        this.accelerator = accelerator;
    }

    @Override
    public CoreOp.FuncOp apply(CoreOp.FuncOp entry) {
        MethodHandles.Lookup l = accelerator.lookup;
        if (!isArrayView(entry)) return entry;

        Map<Op.Result, Op.Result> replaced = new HashMap<>(); // maps a result to the result it should be replaced by
        Map<Op, CoreOp.VarAccessOp.VarLoadOp> bufferVarLoads = new HashMap<>();

        return entry.transform(entry.funcName(), (bb, op) -> {
            switch (op) {
                case JavaOp.InvokeOp iop -> {
                    // catching HATVectorBinaryOps not stored in VarOps
                    if (isVectorBinaryOperation(iop)) {
                        HATVectorBinaryOp vBinaryOp = buildVectorBinaryOp(
                                iop.invokeDescriptor().name(),
                                iop.externalizeOpName(),
                                iop.resultType(),
                                bb.context().getValues(iop.operands())
                        );
                        vBinaryOp.setLocation(iop.location());
                        Op.Result res = bb.op(vBinaryOp);
                        bb.context().mapValue(iop.result(), res);
                        replaced.put(iop.result(), res);
                        return bb;
                    } else if (isBufferArray(iop) &&
                            firstOperand(iop) instanceof Op.Result r) { // ensures we can use iop as key for replaced vvv
                        replaced.put(iop.result(), r);
                        // map buffer VarOp to its corresponding VarLoadOp
                        bufferVarLoads.put(((Op.Result) firstOperand(r.op())).op(), (CoreOp.VarAccessOp.VarLoadOp) r.op());
                        return bb;
                    }
                }
                case CoreOp.VarOp vop -> {
                    if (isBufferInitialize(vop) &&
                            firstOperand(vop) instanceof Op.Result r) { // makes sure we don't process a new int[] for example
                        Op bufferLoad = replaced.get(r).op(); // gets VarLoadOp associated w/ og buffer
                        replaced.put(vop.result(), (Op.Result) firstOperand(bufferLoad)); // gets VarOp associated w/ og buffer
                        return bb;
                    } else if (isVectorOp(vop)) {
                        List<Value> operands = (vop.operands().isEmpty()) ? List.of() : List.of(firstOperand(vop));
                        HATPhaseUtils.VectorMetaData md = HATPhaseUtils.getVectorTypeInfoWithCodeReflection(vop.resultType().valueType());
                        HATVectorVarOp vVarOp = new HATVectorVarOp(
                                vop.varName(),
                                vop.resultType(),
                                md.vectorTypeElement(),
                                md.lanes(),
                                bb.context().getValues(operands)
                        );
                        vVarOp.setLocation(vop.location());
                        Op.Result res = bb.op(vVarOp);
                        bb.context().mapValue(vop.result(), res);
                        return bb;
                    }
                }
                case CoreOp.VarAccessOp.VarLoadOp vlop -> {
                    if ((isBufferInitialize(vlop)) &&
                            firstOperand(vlop) instanceof Op.Result r) {
                        if (r.op() instanceof CoreOp.VarOp) { // if this is the VarLoadOp after the .arrayView() InvokeOp
                            Op.Result replacement = (notGlobalVarOp(vlop)) ?
                                    (Op.Result) firstOperand(((Op.Result) firstOperand(r.op())).op()) :
                                    bufferVarLoads.get(replaced.get(r).op()).result();
                            replaced.put(vlop.result(), replacement);
                        } else { // if this is a VarLoadOp loading the buffer
                            Value loaded = getValue(bb, replaced.get(r));
                            CoreOp.VarAccessOp.VarLoadOp newVarLoad = CoreOp.VarAccessOp.varLoad(loaded);
                            newVarLoad.setLocation(vlop.location());
                            Op.Result res = bb.op(newVarLoad);
                            bb.context().mapValue(vlop.result(), res);
                            replaced.put(vlop.result(), res);
                        }
                        return bb;
                    }
                }
                // TODO: implement more generic array handling for any-dimension arrays
                case JavaOp.ArrayAccessOp.ArrayLoadOp alop -> {
                    if (isBufferArray(alop) &&
                            firstOperand(alop) instanceof Op.Result r) {
                        Op.Result buffer = replaced.getOrDefault(r, r);
                        if (isVectorOp(alop)) {
                            Op vop = ((Op.Result) firstOperand(buffer.op())).op();
                            String name = switch (vop) {
                                case CoreOp.VarOp varOp -> varOp.varName();
                                case HATLocalVarOp hatLocalVarOp -> hatLocalVarOp.varName();
                                case HATPrivateVarOp hatPrivateVarOp -> hatPrivateVarOp.varName();
                                default -> throw new IllegalStateException("Unexpected value: " + vop);
                            };
                            HATPhaseUtils.VectorMetaData md = HATPhaseUtils.getVectorTypeInfoWithCodeReflection(alop.resultType());
                            HATVectorLoadOp vLoadOp = new HATVectorLoadOp(
                                    name,
                                    CoreType.varType(((ArrayType) firstOperand(alop).type()).componentType()),
                                    md.vectorTypeElement(),
                                    md.lanes(),
                                    notGlobalVarOp(alop),
                                    bb.context().getValues(List.of(buffer, alop.operands().getLast()))
                            );
                            vLoadOp.setLocation(alop.location());
                            Op.Result res = bb.op(vLoadOp);
                            bb.context().mapValue(alop.result(), res);
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

                                Class<?> storedClass = typeElementToClass(alop.result().type());
                                MethodRef arrayMethod = MethodRef.method(c, "array", storedClass, long.class);
                                Op.Result invokeRes = bb.op(JavaOp.invoke(arrayMethod, getValue(bb, ogBufferLoad), getValue(bb, idx)));
                                bb.context().mapValue(alop.result(), invokeRes);
                            } else {
                                JavaOp.ConvOp conv = JavaOp.conv(JavaType.LONG, getValue(bb, alop.operands().get(1)));
                                Op.Result convRes = bb.op(conv);

                                Class<?> c = (Class<?>) OpTk.classTypeToTypeOrThrow(l, (ClassType) buffer.type());
                                Class<?> storedClass = typeElementToClass(alop.result().type());
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
                    if (isBufferArray(asop) &&
                            firstOperand(asop) instanceof Op.Result r) {
                        Op.Result buffer = replaced.getOrDefault(r, r);
                        if (isVectorOp(asop)) {
                            Op varOp = findVarOpOrHATVarOP(((Op.Result) asop.operands().getLast()).op());
                            String name = (varOp instanceof HATVectorVarOp) ? ((HATVectorVarOp) varOp).varName() : ((CoreOp.VarOp) varOp).varName();
                            TypeElement resultType = (varOp instanceof HATVectorVarOp) ? (varOp).resultType() : ((CoreOp.VarOp) varOp).resultType();
                            ClassType classType = ((ClassType) ((ArrayType) firstOperand(asop).type()).componentType());
                            HATPhaseUtils.VectorMetaData md = HATPhaseUtils.getVectorTypeInfoWithCodeReflection(classType);
                            HATVectorStoreView vStoreOp = new HATVectorStoreView(
                                    name,
                                    resultType,
                                    md.lanes(),
                                    md.vectorTypeElement(),
                                    notGlobalVarOp(asop),
                                    bb.context().getValues(List.of(buffer, asop.operands().getLast(), asop.operands().get(1)))
                            );
                            vStoreOp.setLocation(asop.location());
                            Op.Result res = bb.op(vStoreOp);
                            bb.context().mapValue(asop.result(), res);
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
                                Class<?> storedClass = typeElementToClass(val.type());
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
                        JavaOp.InvokeOp newInvokeOp = JavaOp.invoke(m, getValue(bb, buffer));
                        newInvokeOp.setLocation(alen.location());
                        Op.Result res = bb.op(newInvokeOp);
                        bb.context().mapValue(alen.result(), res);
                    }
                    return bb;
                }
                case HATVectorSelectLoadOp vSelectLoad -> {
                    CoreOp.VarOp vop = findVarOp(((Op.Result) firstOperand(op)).op());
                    String name = (vop == null) ? "" : vop.varName();
                    HATVectorSelectLoadOp vSelectOp = new HATVectorSelectLoadOp(
                            name,
                            op.resultType(),
                            getLane(vSelectLoad.mapLane()),
                            bb.context().getValues(op.operands())
                    );
                    vSelectOp.setLocation(op.location());
                    bb.context().mapValue(vSelectLoad.result(), bb.op(vSelectOp));
                    return bb;
                }
                case HATVectorSelectStoreOp vSelectStore -> {
                    CoreOp.VarOp vop = findVarOp(((Op.Result) firstOperand(op)).op());
                    String name = (vop == null) ? "" : vop.varName();
                    CoreOp.VarOp resultOp =
                            (((Op.Result) op.operands().getLast()).op() instanceof JavaOp.ArithmeticOperation ||
                                    ((Op.Result) op.operands().getLast()).op() instanceof HATVectorSelectLoadOp) ?
                                    null : findVarOp(((Op.Result) bb.context().getValue(op.operands().get(1))).op());
                    HATVectorSelectStoreOp vSelectOp = new HATVectorSelectStoreOp(
                            name,
                            op.resultType(),
                            getLane(vSelectStore.mapLane()),
                            resultOp,
                            bb.context().getValues(op.operands())
                    );
                    vSelectOp.setLocation(op.location());
                    bb.context().mapValue(vSelectStore.result(), bb.op(vSelectOp));
                    return bb;
                }
                case HATVectorVarLoadOp vVarLoad -> {
                    List<Value> inputOperandsVarLoad = vVarLoad.operands();
                    List<Value> outputOperandsVarLoad = bb.context().getValues(inputOperandsVarLoad);
                    String varLoadName = findVarOp(vVarLoad).varName();
                    HATPhaseUtils.VectorMetaData md = HATPhaseUtils.getVectorTypeInfoWithCodeReflection(vVarLoad.resultType());
                    HATVectorVarLoadOp newVectorVarLoadOp = new HATVectorVarLoadOp(
                            varLoadName,
                            vVarLoad.resultType(),
                            md.vectorTypeElement(),
                            md.lanes(),
                            outputOperandsVarLoad
                    );
                    newVectorVarLoadOp.setLocation(vVarLoad.location());
                    Op.Result res = bb.op(newVectorVarLoadOp);
                    bb.context().mapValue(vVarLoad.result(), res);
                    return bb;
                }
                default -> {
                }
            }
            bb.op(op);
            return bb;
        });
    }

    /*
     * Helper functions:
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

    private HATVectorBinaryOp buildVectorBinaryOp(String opType, String varName, TypeElement resultType, List<Value> outputOperands) {
        HATPhaseUtils.VectorMetaData md = HATPhaseUtils.getVectorTypeInfoWithCodeReflection(resultType);
        return switch (opType) {
            case "add" -> new HATVectorAddOp(varName, resultType, md.vectorTypeElement(), md.lanes(), outputOperands);
            case "sub" -> new HATVectorSubOp(varName, resultType, md.vectorTypeElement(), md.lanes(), outputOperands);
            case "mul" -> new HATVectorMulOp(varName, resultType, md.vectorTypeElement(), md.lanes(), outputOperands);
            case "div" -> new HATVectorDivOp(varName, resultType, md.vectorTypeElement(), md.lanes(), outputOperands);
            default -> throw new IllegalStateException("Unexpected value: " + opType);
        };
    }

    private boolean isVectorBinaryOperation(JavaOp.InvokeOp invokeOp) {
        TypeElement typeElement = invokeOp.resultType();
        boolean isHatVectorType = typeElement.toString().startsWith("hat.buffer.Float");
        return isHatVectorType
                && (invokeOp.invokeDescriptor().name().equalsIgnoreCase("add")
                || invokeOp.invokeDescriptor().name().equalsIgnoreCase("sub")
                || invokeOp.invokeDescriptor().name().equalsIgnoreCase("mul")
                || invokeOp.invokeDescriptor().name().equalsIgnoreCase("div"));
    }

    private Op findVarOpOrHATVarOP(Op op) {
        return searchForOp(op, Set.of(CoreOp.VarOp.class, HATVectorVarOp.class));
    }

    private CoreOp.VarOp findVarOp(Op op) {
        return (CoreOp.VarOp) searchForOp(op, Set.of(CoreOp.VarOp.class));
    }

    public boolean isVectorOp(Op op) {
        if (op.operands().isEmpty()) return false;
        TypeElement type = firstOperand(op).type();
        if (type instanceof ArrayType at) type = at.componentType();
        if (type instanceof ClassType ct) {
            try {
                return _V.class.isAssignableFrom((Class<?>) ct.resolve(accelerator.lookup));
            } catch (ReflectiveOperationException e) {
                throw new RuntimeException(e);
            }
        }
        return false;
    }

    public Value firstOperand(Op op) {
        return op.operands().getFirst();
    }

    public Value getValue(Block.Builder bb, Value value) {
        return bb.context().getValueOrDefault(value, value);
    }

    public boolean isBufferArray(Op op) {
        JavaOp.InvokeOp iop = (JavaOp.InvokeOp) searchForOp(op, Set.of(JavaOp.InvokeOp.class));
        return iop.invokeDescriptor().name().toLowerCase().contains("arrayview");
    }

    public boolean notGlobalVarOp(Op op) {
        JavaOp.InvokeOp iop = (JavaOp.InvokeOp) searchForOp(op, Set.of(JavaOp.InvokeOp.class));
        return iop.invokeDescriptor().name().toLowerCase().contains("local") ||
                iop.invokeDescriptor().name().toLowerCase().contains("private");
    }

    public Op searchForOp(Op op, Set<Class<?>> opClasses) {
        while (!(opClasses.contains(op.getClass()))) {
            if (!op.operands().isEmpty() && firstOperand(op) instanceof Op.Result r) {
                op = r.op();
            } else {
                return null;
            }
        }
        return op;
    }

    public boolean isBufferInitialize(Op op) {
        // first check if the return is an array type
        if (op instanceof CoreOp.VarOp vop) {
            if (!(vop.varValueType() instanceof ArrayType)) return false;
        } else if (!(op instanceof JavaOp.ArrayAccessOp)) {
            if (!(op.resultType() instanceof ArrayType)) return false;
        }

        return isBufferArray(op);
    }

    public boolean isArrayView(CoreOp.FuncOp entry) {
        var here = OpTk.CallSite.of(HATDialectifyArrayViewPhase.class, "isArrayView");
        return OpTk.elements(here, entry).anyMatch((element) -> (
                element instanceof JavaOp.InvokeOp iop &&
                        iop.resultType() instanceof ArrayType &&
                        iop.invokeDescriptor().refType() instanceof JavaType javaType &&
                        OpTk.isAssignable(accelerator.lookup, javaType, MappableIface.class)));
    }

    public Class<?> typeElementToClass(TypeElement type) {
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
                return ((Class<?>) classType.resolve(accelerator.lookup));
            } else {
                throw new IllegalArgumentException("given type cannot be converted to class");
            }
        } catch (ReflectiveOperationException e) {
            throw new RuntimeException("given type cannot be converted to class");
        }
    }
}

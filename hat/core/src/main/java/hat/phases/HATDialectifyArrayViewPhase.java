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
import hat.device.DeviceType;
import hat.dialect.*;
import optkl.LookupCarrier;
import optkl.ifacemapper.MappableIface;
import hat.optools.OpTk;
import hat.types._V;
import jdk.incubator.code.Block;
import jdk.incubator.code.Op;
import jdk.incubator.code.TypeElement;
import jdk.incubator.code.Value;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.core.CoreType;
import jdk.incubator.code.dialect.java.*;

import java.lang.invoke.MethodHandles;
import java.util.*;

public record HATDialectifyArrayViewPhase(LookupCarrier lookupCarrier) implements HATDialect {

    @Override
    public CoreOp.FuncOp apply(CoreOp.FuncOp entry) {
        MethodHandles.Lookup l = lookupCarrier.lookup();
        if (!isArrayView(entry)) return entry;

        Map<Op.Result, Op.Result> replaced = new HashMap<>(); // maps a result to the result it should be replaced by
        Map<Op, CoreOp.VarAccessOp.VarLoadOp> bufferVarLoads = new HashMap<>();

        return entry.transform(entry.funcName(), (bb, op) -> {
            switch (op) {
                case JavaOp.InvokeOp invokeOp -> {
                    if (isVectorBinaryOperation(invokeOp)) {
                        // catching HATVectorBinaryOps not stored in VarOps
                        HATVectorBinaryOp vBinaryOp = buildVectorBinaryOp(
                                invokeOp.invokeDescriptor().name(),
                                obtainVarNameFromInvoke(invokeOp),
                                invokeOp.resultType(),
                                bb.context().getValues(invokeOp.operands())
                        );
                        vBinaryOp.setLocation(invokeOp.location());
                        Op.Result res = bb.op(vBinaryOp);
                        bb.context().mapValue(invokeOp.result(), res);
                        replaced.put(invokeOp.result(), res);
                        return bb;
                    } else if (isBufferArray(invokeOp) &&
                            firstOperand(invokeOp) instanceof Op.Result r) { // ensures we can use iop as key for replaced vvv
                        replaced.put(invokeOp.result(), r);
                        // map buffer VarOp to its corresponding VarLoadOp
                        bufferVarLoads.put((firstOperandAsRes(r.op())).op(), (CoreOp.VarAccessOp.VarLoadOp) r.op());
                        return bb;
                    }
                }
                case CoreOp.VarOp varOp -> {
                    if (isBufferInitialize(varOp) &&
                            firstOperand(varOp) instanceof Op.Result r) { // makes sure we don't process a new int[] for example
                        Op bufferLoad = replaced.get(r).op(); // gets VarLoadOp associated w/ og buffer
                        replaced.put(varOp.result(), firstOperandAsRes(bufferLoad)); // gets VarOp associated w/ og buffer
                        return bb;
                    } else if (isVectorOp(varOp)) {
                        List<Value> operands = (varOp.operands().isEmpty()) ? List.of() : List.of(firstOperand(varOp));
                        HATPhaseUtils.VectorMetaData md = HATPhaseUtils.getVectorTypeInfoWithCodeReflection(varOp.resultType().valueType());
                        HATVectorVarOp vVarOp = new HATVectorVarOp(
                                varOp.varName(),
                                varOp.resultType(),
                                md.vectorTypeElement(),
                                md.lanes(),
                                bb.context().getValues(operands)
                        );
                        vVarOp.setLocation(varOp.location());
                        Op.Result res = bb.op(vVarOp);
                        bb.context().mapValue(varOp.result(), res);
                        return bb;
                    }
                }
                case CoreOp.VarAccessOp.VarLoadOp varLoadOp -> {
                    if ((isBufferInitialize(varLoadOp)) &&
                            firstOperand(varLoadOp) instanceof Op.Result r) {
                        if (r.op() instanceof CoreOp.VarOp) { // if this is the VarLoadOp after the .arrayView() InvokeOp
                            Op.Result replacement = (notGlobalVarOp(varLoadOp)) ?
                                    firstOperandAsRes((firstOperandAsRes(r.op())).op()) :
                                    bufferVarLoads.get(replaced.get(r).op()).result();
                            replaced.put(varLoadOp.result(), replacement);
                        } else { // if this is a VarLoadOp loading the buffer
                            Value loaded = getValue(bb, replaced.get(r));
                            CoreOp.VarAccessOp.VarLoadOp newVarLoad = CoreOp.VarAccessOp.varLoad(loaded);
                            newVarLoad.setLocation(varLoadOp.location());
                            Op.Result res = bb.op(newVarLoad);
                            bb.context().mapValue(varLoadOp.result(), res);
                            replaced.put(varLoadOp.result(), res);
                        }
                        return bb;
                    }
                }
                case JavaOp.ArrayAccessOp.ArrayLoadOp arrayLoadOp -> {
                    if (isBufferArray(arrayLoadOp) &&
                            firstOperand(arrayLoadOp) instanceof Op.Result r) {
                        Op.Result buffer = replaced.getOrDefault(r, r);
                        if (isVectorOp(arrayLoadOp)) {
                            Op vop = (firstOperandAsRes(buffer.op())).op();
                            String name = switch (vop) {
                                case CoreOp.VarOp varOp -> varOp.varName();
                                case HATLocalVarOp hatLocalVarOp -> hatLocalVarOp.varName();
                                case HATPrivateVarOp hatPrivateVarOp -> hatPrivateVarOp.varName();
                                default -> throw new IllegalStateException("Unexpected value: " + vop);
                            };
                            HATPhaseUtils.VectorMetaData md = HATPhaseUtils.getVectorTypeInfoWithCodeReflection(arrayLoadOp.resultType());
                            HATVectorLoadOp vLoadOp = new HATVectorLoadOp(
                                    name,
                                    CoreType.varType(((ArrayType) firstOperand(arrayLoadOp).type()).componentType()),
                                    md.vectorTypeElement(),
                                    md.lanes(),
                                    notGlobalVarOp(arrayLoadOp),
                                    bb.context().getValues(List.of(buffer, arrayLoadOp.operands().getLast()))
                            );
                            vLoadOp.setLocation(arrayLoadOp.location());
                            Op.Result res = bb.op(vLoadOp);
                            bb.context().mapValue(arrayLoadOp.result(), res);
                        } else if (((ArrayType) firstOperand(op).type()).dimensions() == 1) { // we only use the last array load
                            ArrayAccessInfo info = arrayAccessInfo(op.result(), replaced);
                            List<Value> operands = new ArrayList<>();
                            operands.add(info.buffer);
                            operands.addAll(info.indices);
                            HATPtrLoadOp ptrLoadOp = new HATPtrLoadOp(
                                    arrayLoadOp.resultType(),
                                    (Class<?>) OpTk.classTypeToTypeOrThrow(l, (ClassType) info.buffer().type()),
                                    bb.context().getValues(operands)
                            );
                            ptrLoadOp.setLocation(arrayLoadOp.location());
                            Op.Result res = bb.op(ptrLoadOp);
                            bb.context().mapValue(arrayLoadOp.result(), res);
                        }
                    }
                    return bb;
                }
                case JavaOp.ArrayAccessOp.ArrayStoreOp arrayStoreOp -> {
                    if (isBufferArray(arrayStoreOp) &&
                            firstOperand(arrayStoreOp) instanceof Op.Result r) {
                        Op.Result buffer = replaced.getOrDefault(r, r);
                        if (isVectorOp(arrayStoreOp)) {
                            Op varOp = findVarOpOrHATVarOP(((Op.Result) arrayStoreOp.operands().getLast()).op());
                            String name = (varOp instanceof HATVectorVarOp) ? ((HATVectorVarOp) varOp).varName() : ((CoreOp.VarOp) varOp).varName();
                            TypeElement resultType = (varOp instanceof HATVectorVarOp) ? (varOp).resultType() : ((CoreOp.VarOp) varOp).resultType();
                            ClassType classType = ((ClassType) ((ArrayType) firstOperand(arrayStoreOp).type()).componentType());
                            HATPhaseUtils.VectorMetaData md = HATPhaseUtils.getVectorTypeInfoWithCodeReflection(classType);
                            HATVectorStoreView vStoreOp = new HATVectorStoreView(
                                    name,
                                    resultType,
                                    md.lanes(),
                                    md.vectorTypeElement(),
                                    notGlobalVarOp(arrayStoreOp),
                                    bb.context().getValues(List.of(buffer, arrayStoreOp.operands().getLast(), arrayStoreOp.operands().get(1)))
                            );
                            vStoreOp.setLocation(arrayStoreOp.location());
                            Op.Result res = bb.op(vStoreOp);
                            bb.context().mapValue(arrayStoreOp.result(), res);
                        } else if (((ArrayType) firstOperand(op).type()).dimensions() == 1) { // we only use the last array load
                            ArrayAccessInfo info = arrayAccessInfo(op.result(), replaced);
                            List<Value> operands = new ArrayList<>();
                            operands.add(info.buffer());
                            operands.addAll(info.indices);
                            operands.add(arrayStoreOp.operands().getLast());
                            HATPtrStoreOp ptrLoadOp = new HATPtrStoreOp(
                                    arrayStoreOp.resultType(),
                                    (Class<?>) OpTk.classTypeToTypeOrThrow(l, (ClassType) info.buffer().type()),
                                    bb.context().getValues(operands)
                            );
                            ptrLoadOp.setLocation(arrayStoreOp.location());
                            Op.Result res = bb.op(ptrLoadOp);
                            bb.context().mapValue(arrayStoreOp.result(), res);
                        }
                    }
                    return bb;
                }
                case JavaOp.ArrayLengthOp arrayLengthOp -> {
                    if (isBufferArray(arrayLengthOp) &&
                            firstOperand(arrayLengthOp) instanceof Op.Result r) {
                        ArrayAccessInfo info = arrayAccessInfo(op.result(), replaced);
                        HATPtrLengthOp ptrLengthOp = new HATPtrLengthOp(
                                arrayLengthOp.resultType(),
                                (Class<?>) OpTk.classTypeToTypeOrThrow(l, (ClassType) info.buffer().type()),
                                bb.context().getValues(List.of(info.buffer()))
                        );
                        ptrLengthOp.setLocation(arrayLengthOp.location());
                        Op.Result res = bb.op(ptrLengthOp);
                        bb.context().mapValue(arrayLengthOp.result(), res);
                        return bb;
                    }
                }
                default -> {
                }
            }
            bb.op(op);
            return bb;
        });
    }

    record ArrayAccessInfo(Op.Result buffer, List<Op.Result> indices) {};

    record Node<T>(T value, List<Node<T>> edges) {
        ArrayAccessInfo getInfo(Map<Op.Result, Op.Result> replaced) {
            List<Node<T>> wl = new ArrayList<>();
            Set<Node<T>> seen = new HashSet<>();
            Op.Result buffer = null;
            List<Op.Result> indices = new ArrayList<>();
            wl.add(this);
            while (!wl.isEmpty()) {
                Node<T> cur = wl.removeFirst();
                seen.add(cur);
                if (cur.value instanceof Op.Result res) {
                    if (res.op() instanceof JavaOp.ArrayAccessOp || res.op() instanceof JavaOp.ArrayLengthOp) {
                        buffer = res;
                        indices.addFirst(res.op() instanceof JavaOp.ArrayAccessOp ? ((Op.Result) res.op().operands().get(1)) : ((Op.Result) res.op().operands().get(0)));
                    }
                }
                if (!cur.edges().isEmpty()) {
                    Node<T> next = cur.edges().getFirst();
                    if (!seen.contains(next)) wl.add(next);
                }
            }
            buffer = replaced.get((Op.Result) firstOperand(buffer.op()));
            return new ArrayAccessInfo(buffer, indices);
        }
    }

    static ArrayAccessInfo arrayAccessInfo(Value value, Map<Op.Result, Op.Result> replaced) {
        return expressionGraph(value).getInfo(replaced);
    }

    static Node<Value> expressionGraph(Value value) {
        return expressionGraph(new HashMap<>(), value);
    }

    static Node<Value> expressionGraph(Map<Value, Node<Value>> visited, Value value) {
        // If value has already been visited return its node
        if (visited.containsKey(value)) {
            return visited.get(value);
        }

        // Find the expression graphs for each operand
        List<Node<Value>> edges = new ArrayList<>();
        for (Value operand : value.dependsOn()) {
            if (operand instanceof Op.Result res && res.op() instanceof JavaOp.InvokeOp iop && iop.invokeDescriptor().name().toLowerCase().contains("arrayview")) continue;
            edges.add(expressionGraph(operand));
        }
        Node<Value> node = new Node<>(value, edges);
        visited.put(value, node);
        return node;
    }

    /*
     * Helper functions:
     */

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

    public boolean isVectorOp(Op op) {
        if (op.operands().isEmpty()) return false;
        TypeElement type = firstOperand(op).type();
        if (type instanceof ArrayType at) type = at.componentType();
        if (type instanceof ClassType ct) {
            try {
                return _V.class.isAssignableFrom((Class<?>) ct.resolve(lookupCarrier.lookup()));
            } catch (ReflectiveOperationException e) {
                throw new RuntimeException(e);
            }
        }
        return false;
    }

    public static Op.Result firstOperandAsRes(Op op) {
        return (firstOperand(op) instanceof Op.Result res) ? res : null;
    }

    public static Value firstOperand(Op op) {
        return op.operands().getFirst();
    }

    public static Value getValue(Block.Builder bb, Value value) {
        return bb.context().getValueOrDefault(value, value);
    }

    public boolean isBufferArray(Op op) {
        JavaOp.InvokeOp iop = (JavaOp.InvokeOp) searchForOp(op, Set.of(JavaOp.InvokeOp.class));
        return iop.invokeDescriptor().name().toLowerCase().contains("arrayview");
    }

    public boolean notGlobalVarOp(Op op) {
        JavaOp.InvokeOp iop = (JavaOp.InvokeOp) searchForOp(op, Set.of(JavaOp.InvokeOp.class));
        return iop.invokeDescriptor().name().toLowerCase().contains("local") ||
                iop.invokeDescriptor().name().toLowerCase().contains("shared") ||
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
                        (OpTk.isAssignable(lookupCarrier.lookup(), javaType, MappableIface.class)
                                || OpTk.isAssignable(lookupCarrier.lookup(), javaType, DeviceType.class))));
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
                return ((Class<?>) classType.resolve(lookupCarrier.lookup()));
            } else {
                throw new IllegalArgumentException("given type cannot be converted to class");
            }
        } catch (ReflectiveOperationException e) {
            throw new RuntimeException("given type cannot be converted to class");
        }
    }

    private String obtainVarNameFromInvoke(JavaOp.InvokeOp invokeOp) {
        Op.Result invokeResult = invokeOp.result();
        if (!invokeResult.uses().isEmpty()) {
            Op.Result r = invokeResult.uses().stream().toList().getFirst();
            if (r.op() instanceof CoreOp.VarOp varOp) {
                return varOp.varName();
            }
        }
        return invokeOp.externalizeOpName();
    }
}

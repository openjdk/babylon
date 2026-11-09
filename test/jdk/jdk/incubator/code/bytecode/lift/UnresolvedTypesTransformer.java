/*
 * Copyright (c) 2024, 2026, Oracle and/or its affiliates. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.
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

import jdk.incubator.code.*;
import jdk.incubator.code.dialect.core.*;
import jdk.incubator.code.dialect.java.*;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;

/**
 * Resolves unresolved types.
 */
final class UnresolvedTypesTransformer {

    static CoreOp.FuncOp transform(CoreOp.FuncOp func) {
        try {
            return new UnresolvedTypesTransformer().resolve(func);
        } catch (Throwable t) {
            System.out.println(func.toText());
            throw t;
        }
    }

    private final Map<UnresolvedType, JavaType> resolvedMap;

    private UnresolvedTypesTransformer() {
        resolvedMap = new IdentityHashMap<>();
    }

    private CoreOp.FuncOp resolve(CoreOp.FuncOp func) {
        List<Op> ops = func.elements().filter(Op.class::isInstance).map(Op.class::cast).toList();
        boolean changed;
        do {
            changed = false;
            for (Op op : ops) {
                List<CodeType> expected = operandTypes(op);
                for (int i = 0; i < expected.size(); i++) {
                    changed |= resolveTo(op.operands().get(i).type(), expected.get(i));
                }
                changed |= switch (op) {
                    case CoreOp.VarOp vo when !vo.isUninitialized() ->
                        resolveFrom(vo.varValueType(), vo.initOperand().type());
                    case CoreOp.VarAccessOp.VarStoreOp vso ->
                        resolveFrom(vso.varType().valueType(), vso.storeOperand().type());
                    case JavaOp.BinaryOp bo ->
                        resolveFrom(bo.resultType(), bo.operands().get(1).type());
                    case JavaOp.ArrayAccessOp.ArrayLoadOp alo ->
                        resolveFrom(alo.resultType(), knownComponentType(alo.operands().getFirst().type()));
                    case JavaOp.ArrayAccessOp.ArrayStoreOp aso ->
                        resolveFrom(aso.operands().getFirst().type(), knownArrayType(aso.operands().get(2).type()));
                    default -> false;
                };
                for (Block.Reference ref : op.successors()) {
                    for (int i = 0; i < ref.arguments().size(); i++) {
                        CodeType argument = ref.arguments().get(i).type();
                        CodeType parameter = ref.targetBlock().parameters().get(i).type();
                        changed |= resolveTo(argument, parameter) | resolveFrom(parameter, argument);
                    }
                }
            }
        } while (changed);

        return func.transform(new CodeTransformer() {
            @Override
            public void acceptBlock(Block.Builder block, Block b) {
                if (b.isEntryBlock()) {
                    CodeContext cc = block.context();
                    for (Block source : b.ancestorBody().blocks()) {
                        List<CodeType> types = source.parameterTypes().stream()
                                .map(UnresolvedTypesTransformer.this::resolvedType).toList();
                        if (!source.isEntryBlock() && !types.equals(source.parameterTypes())) {
                            Block.Builder target = block.block(types);
                            cc.mapBlock(source, target);
                            cc.mapValues(source.parameters(), target.parameters());
                        }
                    }
                }
                CodeTransformer.super.acceptBlock(block, b);
            }

            @Override
            public Block.Builder acceptOp(Block.Builder block, Op op) {
                CodeContext cc = block.context();
                List<CodeType> expected = operandTypes(op);
                List<Value> operands = new ArrayList<>(cc.getValues(op.operands()));
                boolean changed = !resolvedType(op.resultType()).equals(op.resultType());
                for (int i = 0; i < expected.size(); i++) {
                    CodeType type = expected.get(i);
                    Value value = cc.getValue(op.operands().get(i));
                    if (type instanceof PrimitiveType && value.type() instanceof PrimitiveType && !type.equals(value.type())) {
                        if (value instanceof Op.Result r && r.op() instanceof CoreOp.ConstantOp c) {
                            Object constant = ArithmeticAndConvOpImpls.evaluate(JavaOp.conv(type, value), List.of(c.value()));
                            value = block.add(CoreOp.constant(type, constant));
                        } else  if (type.equals(JavaType.BOOLEAN) && ((PrimitiveType)value.type()).toBasicType().equals(JavaType.INT)) {
                            if (!value.type().equals(JavaType.INT)) {
                                value = block.add(JavaOp.conv(JavaType.INT, value));
                            }
                            value = block.add(JavaOp.neq(
                                    block.add(JavaOp.and(value, block.add(CoreOp.constant(JavaType.INT, 1)))),
                                    block.add(CoreOp.constant(JavaType.INT, 0))));
                        } else {
                            value = block.add(JavaOp.conv(type, value));
                        }
                        changed |= value != operands.set(i, value);
                    }
                }
                if (!changed) {
                    block.add(op);
                    return block;
                }
                Op adaptedCopy = switch (op) {
                    // explicit constant type required
                    case CoreOp.ConstantOp c -> {
                        CodeType type = resolvedType(c.resultType());
                        Object value = type.equals(JavaType.BOOLEAN) && c.value() instanceof Number n
                                ? n.intValue() != 0 : c.value();
                        yield CoreOp.constant(type, value);
                    }
                    // explicit var type required
                    case CoreOp.VarOp vo ->
                        CoreOp.var(vo.varName(),
                                   resolvedType(vo.varValueType()),
                                   vo.isUninitialized() ? null : operands.getFirst());
                    // @@@ copy constructor preserves old component type
                    case JavaOp.ArrayAccessOp.ArrayLoadOp _ ->
                        JavaOp.arrayLoadOp(operands.get(0),
                                           operands.get(1));
                    // repeated operands may need individual type conversions
                    case JavaOp.ArrayAccessOp.ArrayStoreOp _ ->
                        JavaOp.arrayStoreOp(operands.get(0),
                                            operands.get(1),
                                            operands.get(2));
                    // repeated operands may need individual type conversions
                    case JavaOp.InvokeOp io ->
                        JavaOp.invoke(io.invokeKind(),
                                      io.isVarArgs(),
                                      resolvedType(io.resultType()),
                                      io.invokeReference(),
                                      operands);
                    // repeated operands may need individual type conversions
                    case JavaOp.NewOp no ->
                        JavaOp.new_(no.isVarargs(),
                                    resolvedType(no.resultType()),
                                    no.constructorReference(),
                                    operands);
                    default -> {
                        CodeContext opContext = CodeContext.create(cc);
                        opContext.mapValues(op.operands(), operands);
                        if (!operands.stream().map(Value::type).toList().equals(
                                opContext.getValues(op.operands()).stream().map(Value::type).toList())) {
                            // remaining ops should use the same operand types for every occurrence of a given operand
                            throw new IllegalStateException("Conflicting operand types for " + op.getClass().getName());
                        }
                        yield op.transform(opContext, this);
                    }
                };
                adaptedCopy.setLocation(op.location());
                cc.mapValue(op.result(), block.add(adaptedCopy));
                return block;
            }
        });
    }

    private CodeType knownType(CodeType type) {
        return type instanceof UnresolvedType ut ? resolvedMap.get(ut) : type;
    }

    private CodeType resolvedType(CodeType type) {
        return switch (type) {
            case VarType vt ->
                CoreType.varType(resolvedType(vt.valueType()));
            case UnresolvedType.Int uti ->
                resolvedMap.getOrDefault(uti, JavaType.INT);
            case UnresolvedType ut ->
                resolvedMap.getOrDefault(ut, JavaType.J_L_OBJECT);
            default ->
                type;
        };
    }

    private CodeType knownComponentType(CodeType type) {
        return knownType(type) instanceof ArrayType at ? at.componentType() : null;
    }

    private CodeType knownArrayType(CodeType type) {
        return knownType(type) instanceof JavaType jt ? JavaType.array(jt) : null;
    }

    private boolean resolveFrom(CodeType unresolved, CodeType from) {
        CodeType type = knownType(from);
        return (JavaType.BOOLEAN.equals(type) || type instanceof ArrayType) && resolveTo(unresolved, type);
    }

    private boolean resolveTo(CodeType unresolved, CodeType to) {
        return switch (unresolved) {
            case UnresolvedType.Int ui when JavaType.BOOLEAN.equals(knownType(to)) ->
                resolvedMap.putIfAbsent(ui, JavaType.BOOLEAN) == null;
            case UnresolvedType.Ref ur when knownType(to) instanceof JavaType jt && !(jt instanceof PrimitiveType) -> {
                CodeType prev = resolvedMap.get(ur);
                if (!jt.equals(prev) && (prev == null || prev.equals(JavaType.J_L_OBJECT))) {
                    resolvedMap.put(ur, jt);
                    yield true;
                }
                yield false;
            }
            default ->
                false;
        };
    }

    private CodeType arithmeticType(JavaOp.ArithmeticOperation op) {
        CodeType type = resolvedType(op.resultType());
        if (type.equals(JavaType.BOOLEAN)) {
            return type;
        }
        return type instanceof PrimitiveType pt ? pt.toBasicType() : type;
    }

    private List<CodeType> operandTypes(Op op) {
        return switch (op) {
            case JavaOp.CompareOp co -> {
                CodeType type = resolvedType(co.operands().getFirst().type());
                if (type instanceof PrimitiveType pt) type = pt.toBasicType();
                yield List.of(type, type);
            }
            case JavaOp.LshlOp _, JavaOp.LshrOp _, JavaOp.AshrOp _ ->
                List.of(arithmeticType((JavaOp.BinaryOp)op),
                        JavaType.INT);
            case JavaOp.BinaryOp bo -> {
                CodeType type = arithmeticType(bo);
                yield List.of(type, type);
            }
            case JavaOp.NegOp no ->
                List.of(arithmeticType(no));
            case JavaOp.InvokeOp io -> {
                List<CodeType> types = new ArrayList<>();
                if (io.hasReceiver()) types.add(io.invokeReference().refType());
                types.addAll(io.invokeReference().signature().parameterTypes());
                yield types;
            }
            case JavaOp.NewOp no ->
                no.constructorReference().signature().parameterTypes();
            case JavaOp.FieldAccessOp.FieldLoadOp fo when !fo.operands().isEmpty() ->
                List.of(fo.fieldReference().refType());
            case JavaOp.FieldAccessOp.FieldStoreOp fo when fo.operands().size() == 1 ->
                List.of(fo.fieldReference().type());
            case JavaOp.FieldAccessOp.FieldStoreOp fo ->
                List.of(fo.fieldReference().refType(),
                        fo.fieldReference().type());
            case CoreOp.ReturnOp ro when !ro.operands().isEmpty() ->
                List.of(ro.ancestorBody().bodySignature().returnType());
            case CoreOp.VarOp vo when !vo.isUninitialized() ->
                List.of(resolvedType(vo.varValueType()));
            case CoreOp.VarAccessOp.VarStoreOp vo ->
                List.of(vo.varType(),
                        resolvedType(vo.varType().valueType()));
            case JavaOp.ArrayAccessOp.ArrayLoadOp ao ->
                Arrays.asList(knownArrayType(ao.resultType()), // may be null
                              JavaType.INT);
            case JavaOp.ArrayAccessOp.ArrayStoreOp ao ->
                Arrays.asList(ao.operands().getFirst().type(),
                              JavaType.INT,
                              knownComponentType(ao.operands().getFirst().type())); // may be null
            default ->
                List.of();
        };
    }
}

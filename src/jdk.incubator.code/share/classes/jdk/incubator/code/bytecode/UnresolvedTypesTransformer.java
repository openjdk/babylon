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
package jdk.incubator.code.bytecode;

import jdk.incubator.code.Block;
import jdk.incubator.code.CopyContext;
import jdk.incubator.code.Op;
import jdk.incubator.code.OpTransformer;
import jdk.incubator.code.TypeElement;
import jdk.incubator.code.Value;
import jdk.incubator.code.op.CoreOp;
import jdk.incubator.code.type.ArrayType;
import jdk.incubator.code.type.JavaType;
import jdk.incubator.code.type.MethodRef;
import jdk.incubator.code.type.PrimitiveType;
import jdk.incubator.code.type.VarType;
import java.util.ArrayList;
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
        ArrayList<Value> unresolved = func.traverse(new ArrayList<>(), (q, e) -> {
            switch (e) {
                case Block b -> b.parameters().forEach(v -> {
                   if (toResolve(v) != null) q.add(v);
                });
                case Op op when toResolve(op.result()) != null ->
                   q.add(op.result());
                default -> {}
           }
           return q;
        });

        boolean changed = true;
        while (changed) {
            changed = false;
            for (Value v : unresolved) {
                changed |= resolve(v);
            }
        }

        // Remaining types are resolved to defaults
        for (Value v : unresolved) {
            resolvedMap.computeIfAbsent(toResolve(v), ut ->
                switch (ut) {
                    case UnresolvedType.Int _ -> JavaType.INT;
                    case UnresolvedType.Ref _ -> JavaType.J_L_OBJECT;
                });
        }

        return func.transform(blockParamTypesTransformer())
                   .transform(opTypesTransformer())
                   .transform(unifyOperandsTransformer());
    }

    private static UnresolvedType toResolve(Value v) {
        return v == null ? null : switch (v.type()) {
            case UnresolvedType ut -> ut;
            case VarType vt when vt.valueType() instanceof UnresolvedType ut  -> ut;
            default -> null;
        };
    }

    private TypeElement toComponent(TypeElement te) {
        if (te instanceof UnresolvedType ut) {
            te = resolvedMap.get(ut);
        }
        return te instanceof ArrayType at ? at.componentType() : null;
    }

    private TypeElement toArray(TypeElement te) {
        if (te instanceof UnresolvedType ut) {
            te = resolvedMap.get(ut);
        }
        return te instanceof JavaType jt ? JavaType.array(jt) : null;
    }

    private boolean resolve(Value v) {
        UnresolvedType ut = toResolve(v);
        if (ut == null) return false;
        boolean changed = false;
        for (Op.Result useRes : v.uses()) {
            Op op = useRes.op();
            int i = op.operands().indexOf(v);
            if (i >= 0) {
                changed |= switch (op) {
                    case CoreOp.LshlOp _, CoreOp.LshrOp _, CoreOp.AshrOp _ ->
                        i == 0 && resolveTo(ut, op.resultType());
                    case CoreOp.BinaryOp bo ->
                        resolveTo(ut, bo.resultType());
                    case CoreOp.InvokeOp io -> {
                        MethodRef id = io.invokeDescriptor();
                        if (io.hasReceiver()) {
                            if (i == 0) yield resolveTo(ut, id.refType());
                            i--;
                        }
                        yield resolveTo(ut, id.type().parameterTypes().get(i));
                    }
                    case CoreOp.FieldAccessOp fao ->
                        resolveTo(ut, fao.fieldDescriptor().refType());
                    case CoreOp.ReturnOp ro ->
                        resolveTo(ut, ro.ancestorBody().bodyType().returnType());
                    case CoreOp.VarOp vo ->
                        resolveTo(ut, vo.varValueType());
                    case CoreOp.VarAccessOp.VarStoreOp vso ->
                        resolveTo(ut, vso.varType().valueType());
                    case CoreOp.NewOp no ->
                        resolveTo(ut, no.constructorType().parameterTypes().get(i));
                    case CoreOp.ArrayAccessOp.ArrayLoadOp alo ->
                        resolveTo(ut, toArray(alo.resultType()));
                    case CoreOp.ArrayAccessOp.ArrayStoreOp aso ->
                        switch (i) {
                            case 0 -> resolveFrom(ut, toArray(aso.operands().get(2).type()));
                            case 2 -> resolveTo(ut, toComponent(aso.operands().get(0).type()));
                            default -> false;
                        };
                    default -> false;
                };
            }
            // Pull block parameter type when used as block argument
            for (Block.Reference sucRef : useRes.op().successors()) {
                i = sucRef.arguments().indexOf(v);
                if (i >= 0) {
                    changed |= resolveTo(ut, sucRef.targetBlock().parameters().get(i).type());
                }
            }
        }
        if (v instanceof Block.Parameter bp) {
            int bi = bp.index();
            Block b = bp.declaringBlock();
            for (Block pb : b.predecessors()) {
                for (Block.Reference r : pb.successors()) {
                    if (r.targetBlock() == b) {
                        var args = r.arguments();
                        if (args.size() > bi && resolveFrom(ut, args.get(bi).type())) {
                            return true;
                        }
                    }
                }
            }
        } else if (v instanceof Op.Result or) {
            changed |= switch (or.op()) {
                case CoreOp.UnaryOp uo ->
                    resolveFrom(ut, uo.operands().getFirst().type());
                case CoreOp.BinaryOp bo ->
                    resolveFrom(ut, bo.operands().getFirst().type())
                    || resolveFrom(ut, bo.operands().get(1).type());
                case CoreOp.VarAccessOp.VarLoadOp vlo ->
                    resolveFrom(ut, vlo.varType().valueType());
                case CoreOp.VarOp vo ->
                    resolveVarOpType(ut, vo);
                case CoreOp.ArrayAccessOp.ArrayLoadOp alo ->
                    resolveFrom(ut, toComponent(alo.operands().getFirst().type()));
                default -> false;
            };
        }
        return changed;
    }

    private boolean resolveFrom(UnresolvedType unresolved, TypeElement from) {
        TypeElement type = from instanceof UnresolvedType utt ? resolvedMap.get(utt) : from;
        JavaType resolved = resolvedMap.get(unresolved);
        return switch (unresolved) {
            // Only care about arrays
            case UnresolvedType.Ref _ when (resolved == null || resolved.equals(JavaType.J_L_OBJECT)) && type instanceof ArrayType at -> {
                resolvedMap.put(unresolved, at);
                yield true;
            }
            // Only care about booleans
            case UnresolvedType.Int _ when JavaType.BOOLEAN.equals(type) && !JavaType.BOOLEAN.equals(resolved) -> {
                resolvedMap.put(unresolved, JavaType.BOOLEAN);
                yield true;
            }
            default -> false;
        };
    }

    private static final List<PrimitiveType> INT_TYPES = List.of(JavaType.INT, JavaType.CHAR, JavaType.SHORT, JavaType.BYTE, JavaType.BOOLEAN);

    private boolean resolveTo(UnresolvedType unresolved, TypeElement to) {
        TypeElement type = to instanceof UnresolvedType utt ? resolvedMap.get(utt) : to;
        JavaType resolved = resolvedMap.get(unresolved);
        return switch (unresolved) {
            case UnresolvedType.Ref _ when (resolved == null || resolved.equals(JavaType.J_L_OBJECT)) && type instanceof JavaType jt && !jt.equals(resolved) -> {
                resolvedMap.put(unresolved, jt);
                yield true;
            }
            case UnresolvedType.Int _ when type instanceof PrimitiveType pt && (INT_TYPES.indexOf(pt) > (resolved == null ? -1 : INT_TYPES.indexOf(resolved))) -> {
                resolvedMap.put(unresolved, pt);
                yield true;
            }
            default -> false;
        };
    }

    private boolean resolveVarOpType(UnresolvedType ut, CoreOp.VarOp vo) {
        boolean changed = vo.isUninitialized() ? false : resolveFrom(ut, vo.initOperand().type());
        for (Op.Result varUses : vo.result().uses()) {
            changed |= switch (varUses.op()) {
                case CoreOp.VarAccessOp.VarLoadOp vlo ->
                    resolveTo(ut, vlo.resultType());
                case CoreOp.VarAccessOp.VarStoreOp vso ->
                    resolveFrom(ut, vso.storeOperand().type());
                default -> false;
            };
        }
        return changed;
    }

    private Object convertValue(UnresolvedType ut, Object value) {
        return switch (INT_TYPES.indexOf(resolvedMap.get(ut))) {
            case 0 -> toNumber(value).intValue();
            case 1 -> (char)toNumber(value).intValue();
            case 2 -> toNumber(value).shortValue();
            case 3 -> toNumber(value).byteValue();
            case 4 -> value instanceof Number n ? n.intValue() != 0 : (Boolean)value;
            default -> value;
        };
    }

    private static Number toNumber(Object value) {
        return switch (value) {
            case Boolean b -> b ? 1 : 0;
            case Character c -> (int)c;
            case Number n -> n;
            default -> throw new IllegalStateException("Unexpected " + value);
        };
    }

    private OpTransformer blockParamTypesTransformer() {
        return new OpTransformer() {
            @Override
            public void apply(Block.Builder block, Block b) {
                if (block.isEntryBlock()) {
                    CopyContext cc = block.context();
                    List<Block> sourceBlocks = b.parentBody().blocks();

                    // Override blocks with changed parameter types
                    for (int i = 1; i < sourceBlocks.size(); i++) {
                        Block sourceBlock = sourceBlocks.get(i);
                        List<TypeElement> paramTypes = sourceBlock.parameterTypes();
                        if (paramTypes.stream().anyMatch(UnresolvedType.class::isInstance)) {
                            Block.Builder newBlock = block.block(paramTypes.stream()
                                    .map(pt -> pt instanceof UnresolvedType ut  ? resolvedMap.get(ut) : pt)
                                    .toList());
                            cc.mapBlock(sourceBlock, newBlock);
                            cc.mapValues(sourceBlock.parameters(), newBlock.parameters());
                        }
                    }

                }
                OpTransformer.super.apply(block, b);
            }

            @Override
            public Block.Builder apply(Block.Builder block, Op op) {
                block.op(op);
                return block;
            }
        };
    }

    private OpTransformer opTypesTransformer() {
        return (block, op) -> {
            CopyContext cc = block.context();
            switch (op) {
                case CoreOp.ConstantOp cop when op.resultType() instanceof UnresolvedType ut ->
                    cc.mapValue(op.result(), block.op(CoreOp.constant(resolvedMap.get(ut), convertValue(ut, cop.value()))));
                case CoreOp.VarOp vop when vop.varValueType() instanceof UnresolvedType ut ->
                    cc.mapValue(op.result(), block.op(vop.isUninitialized()
                            ? CoreOp.var(vop.varName(), resolvedMap.get(ut))
                            : CoreOp.var(vop.varName(), resolvedMap.get(ut), cc.getValueOrDefault(vop.initOperand(), vop.initOperand()))));
                case CoreOp.ArrayAccessOp.ArrayLoadOp alop when op.resultType() instanceof UnresolvedType -> {
                    List<Value> opers = alop.operands();
                    Value array = opers.getFirst();
                    Value index = opers.getLast();
                    cc.mapValue(op.result(), block.op(CoreOp.arrayLoadOp(cc.getValueOrDefault(array, array), cc.getValueOrDefault(index, index))));
                }
                default ->
                    block.op(op);
            }
            return block;
        };
    }

    private static OpTransformer unifyOperandsTransformer() {
        return (block, op) -> {
            switch (op) {
                case CoreOp.BinaryTestOp _ ->
                    unify(block, op, JavaType.INT, JavaType.INT);
                case CoreOp.LshlOp _, CoreOp.LshrOp _, CoreOp.AshrOp _ ->
                    unify(block, op, op.resultType(), JavaType.INT);
                case CoreOp.BinaryOp _ ->
                    unify(block, op, op.resultType(), op.resultType());
                default ->
                    block.op(op);
            }
            return block;
        };
    }

    private static void unify(Block.Builder block, Op op, TypeElement firstType, TypeElement secondType) {
        List<Value> operands = op.operands();
        CopyContext cc = CopyContext.create(block.context());
        Value first = operands.getFirst();
        boolean changed = false;
        if (first.type() instanceof PrimitiveType && !first.type().equals(firstType)) {
            cc.mapValue(first, block.op(CoreOp.conv(firstType, cc.getValueOrDefault(first, first))));
            changed = true;
        }
        Value second = operands.get(1);
        if (second.type() instanceof PrimitiveType && !second.type().equals(secondType)) {
            cc.mapValue(second, block.op(CoreOp.conv(secondType, cc.getValueOrDefault(second, second))));
            changed = true;
        }
        if (changed) {
            block.context().mapValue(op.result(), block.op(op.copy(cc)));
        } else {
            block.op(op);
        }
    }
}

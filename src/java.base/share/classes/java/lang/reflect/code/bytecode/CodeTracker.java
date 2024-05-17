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
package java.lang.reflect.code.bytecode;

import java.lang.classfile.CodeElement;
import java.lang.classfile.Label;
import java.lang.classfile.MethodModel;
import java.lang.classfile.Opcode;
import java.lang.classfile.TypeKind;
import java.lang.classfile.attribute.StackMapFrameInfo;
import java.lang.classfile.attribute.StackMapFrameInfo.ObjectVerificationTypeInfo;
import java.lang.classfile.attribute.StackMapTableAttribute;
import java.lang.classfile.instruction.*;
import java.lang.constant.ClassDesc;
import java.lang.reflect.AccessFlag;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.stream.Collectors;

import static java.lang.classfile.attribute.StackMapFrameInfo.SimpleVerificationTypeInfo.*;
import java.lang.classfile.attribute.StackMapFrameInfo.UninitializedVerificationTypeInfo;
import java.lang.constant.ConstantDescs;
import static java.lang.constant.ConstantDescs.*;

public final class CodeTracker implements Consumer<CodeElement> {

    public final Map<LoadInstruction, ClassDesc> insMap;

    private final ClassDesc thisClass;
    private final MethodModel mm;
    private List<ClassDesc> stack;
    private List<ClassDesc> locals;

    private final Map<Label, StackMapFrameInfo> stackMap;

    public CodeTracker(MethodModel mm, Optional<StackMapTableAttribute> smta) {
        this.mm = mm;
        this.stack = new ArrayList<>();
        this.locals = new ArrayList<>();
        this.thisClass = mm.parent().orElseThrow().thisClass().asSymbol();
        if (!mm.flags().has(AccessFlag.STATIC)) locals.add(thisClass);
        for (var pt : mm.methodTypeSymbol().parameterList()) {
            locals.add(pt);
            if (TypeKind.from(pt).slotSize() == 2) locals.add(null);
        }
        this.stackMap = smta.map(a -> a.entries().stream().collect(Collectors.toUnmodifiableMap(
                StackMapFrameInfo::target,
                Function.identity()))).orElse(Map.of());
        this.insMap = new HashMap<>();
    }

    private ClassDesc vtiToStackType(StackMapFrameInfo.VerificationTypeInfo vti) {
        return switch (vti) {
            case ITEM_INTEGER -> CD_int;
            case ITEM_FLOAT -> CD_float;
            case ITEM_DOUBLE -> CD_double;
            case ITEM_LONG -> CD_long;
            case ITEM_UNINITIALIZED_THIS -> thisClass;
            case ITEM_NULL -> null;
            case ObjectVerificationTypeInfo ovti -> ovti.classSymbol();
            case UninitializedVerificationTypeInfo _ -> null;
            default -> throw new IllegalArgumentException("Invalid type on stack: " + vti);
        };
    }

    private void push(ClassDesc type) {
        if (!ConstantDescs.CD_void.equals(type)) stack.addLast(type);
    }

    private ClassDesc pop() {
        return stack.removeLast();
    }

    private void pop(int i) {
        while (i-- > 0) pop();
    }

    private void store(int slot, ClassDesc type) {
        for (int i = locals.size(); i <= slot; i++) locals.add(null);
        locals.set(slot, type);
    }

    private ClassDesc load(int slot) {
        return locals.get(slot);
    }

    @Override
    public void accept(CodeElement el) {
        switch (el) {
            case ArrayLoadInstruction _ -> {
                pop(1);push(pop().componentType());
            }
            case ArrayStoreInstruction _ ->
                pop(3);
            case BranchInstruction i -> {
                if (i.opcode() == Opcode.GOTO || i.opcode() == Opcode.GOTO_W) {
                    stack = null;
                    locals = null;
                } else {
                    pop(1);
                }
            }
            case ConstantInstruction i -> {
                var tk = i.typeKind();
                push(ClassDesc.ofDescriptor(tk == TypeKind.ReferenceType ?
                        i.constantValue().getClass().descriptorString() : tk.descriptor()));
            }
            case ConvertInstruction i -> {
                pop(1);push(ClassDesc.ofDescriptor(i.toType().descriptor()));
            }
            case FieldInstruction i -> {
                switch (i.opcode()) {
                    case GETSTATIC ->
                        push(i.typeSymbol());
                    case GETFIELD -> {
                        pop(1);push(i.typeSymbol());
                    }
                    case PUTSTATIC ->
                        pop(1);
                    case PUTFIELD ->
                        pop(2);
                }
            }
            case InvokeDynamicInstruction i -> {
                var type = i.typeSymbol();
                pop(type.parameterCount());
                push(type.returnType());
            }
            case InvokeInstruction i -> {
                var type = i.typeSymbol();
                pop(type.parameterCount());
                if (i.opcode() != Opcode.INVOKESTATIC) pop(1);
                push(type.returnType());
            }
            case IncrementInstruction i -> {}
            case LoadInstruction i -> {
                push(load(i.slot()));
                if (i.typeKind() == TypeKind.ReferenceType) {
                    insMap.put(i, locals.get(i.slot()));
                }
            }
            case StoreInstruction i ->
                store(i.slot(), pop());
            case MonitorInstruction _ ->
                pop(1);
            case NewMultiArrayInstruction i -> {
                pop(i.dimensions());push(i.arrayType().asSymbol());
            }
            case NewObjectInstruction i ->
                push(i.className().asSymbol());
            case NewPrimitiveArrayInstruction i -> {
                pop(1);push(ClassDesc.ofDescriptor(i.typeKind().descriptor()).arrayType());
            }
            case NewReferenceArrayInstruction i -> {
                pop(1);push(i.componentType().asSymbol().arrayType());
            }
            case OperatorInstruction i -> {
                switch (i.opcode()) {
                    case ARRAYLENGTH, INEG, LNEG, FNEG, DNEG -> pop(1);
                    default -> pop(2);
                }
                push(ClassDesc.ofDescriptor(i.typeKind().descriptor()));
            }
            case StackInstruction i -> {
                switch (i.opcode()) {
                    case POP -> pop(1);
                    case POP2 -> {
                        if (TypeKind.from(pop()).slotSize() == 1) pop();
                    }
                    case DUP -> {
                        var v = pop();push(v);push(v);
                    }
                    case DUP2 -> {
                        var v1 = pop();
                        if (TypeKind.from(v1).slotSize() == 1) {
                            var v2 = pop();
                            push(v2);push(v1);
                            push(v2);push(v1);
                        } else {
                            push(v1);push(v1);
                        }
                    }
                    case DUP_X1 -> {
                        var v1 = pop();
                        var v2 = pop();
                        push(v1);push(v2);push(v1);
                    }
                    case DUP_X2 -> {
                        var v1 = pop();
                        var v2 = pop();
                        if (TypeKind.from(v2).slotSize() == 1) {
                            var v3 = pop();
                            push(v1);push(v3);push(v2);push(v1);
                        } else {
                            push(v1);push(v2);push(v1);
                        }
                    }
                    case DUP2_X1 -> {
                        var v1 = pop();
                        var v2 = pop();
                        if (TypeKind.from(v1).slotSize() == 1) {
                            var v3 = pop();
                            push(v2);push(v1);push(v3);push(v2);push(v1);
                        } else {
                            push(v1);push(v2);push(v1);
                        }
                    }
                    case DUP2_X2 -> {
                        var v1 = pop();
                        var v2 = pop();
                        if (TypeKind.from(v1).slotSize() == 1) {
                            var v3 = pop();
                            if (TypeKind.from(v3).slotSize() == 1) {
                                var v4 = pop();
                                push(v2);push(v1);push(v4);push(v3);push(v2);push(v1);
                            } else {
                                push(v2);push(v1);push(v3);push(v2);push(v1);
                            }
                        } else {
                            if (TypeKind.from(v2).slotSize() == 1) {
                                var v3 = pop();
                                push(v1);push(v3);push(v2);push(v1);
                            } else {
                                push(v1);push(v2);push(v1);
                            }
                        }
                    }
                    case SWAP -> {
                        var v1 = pop();
                        var v2 = pop();
                        push(v1);push(v2);
                    }
                }
            }
            case TypeCheckInstruction i -> {
                switch (i.opcode()) {
                    case CHECKCAST -> {
                        pop(1);push(i.type().asSymbol());
                    }
                    case INSTANCEOF -> {
                        pop(1);push(ConstantDescs.CD_int);
                    }
                }
            }
            case LookupSwitchInstruction _, TableSwitchInstruction _, ReturnInstruction _, ThrowInstruction _ -> {
                stack = null;
                locals = null;
            }
            case LabelTarget lt -> {
                var smfi = stackMap.get(lt.label());
                if (smfi != null) {
                    stack = new ArrayList<>();
                    for (var vti : smfi.stack()) {
                        stack.add(vtiToStackType(vti));
                    }
                    locals = new ArrayList<>();
                    // init locals
                    int slot = 0;
                    for (var vti : smfi.locals()) {
                        if (vti != ITEM_TOP) {
                            store(slot, vtiToStackType(vti));
                        }
                        if (vti == ITEM_DOUBLE || vti == ITEM_LONG) {
                            slot++;
                        }
                        slot++;
                    }
                }
            }
            default -> {}
        }
    }
}

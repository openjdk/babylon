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
import java.lang.classfile.Opcode;
import java.lang.classfile.TypeKind;
import java.lang.classfile.attribute.StackMapFrameInfo;
import java.lang.classfile.attribute.StackMapFrameInfo.ObjectVerificationTypeInfo;
import java.lang.classfile.attribute.StackMapTableAttribute;
import java.lang.classfile.instruction.*;
import java.lang.constant.ClassDesc;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.function.Function;
import java.util.stream.Collectors;

import static java.lang.classfile.attribute.StackMapFrameInfo.SimpleVerificationTypeInfo.*;
import java.lang.constant.ConstantDescs;
import static java.lang.constant.ConstantDescs.*;
import java.lang.constant.MethodTypeDesc;
import java.util.IdentityHashMap;

public final class LocalsTypeMapper {

    private final Map<LoadInstruction, ClassDesc> insMap;
    private final ClassDesc thisClass;
    private final List<ClassDesc> stack, locals;
    private final Map<Label, StackMapFrameInfo> stackMap;

    public LocalsTypeMapper(ClassDesc thisClass,
                         MethodTypeDesc methodType,
                         boolean isStatic,
                         Optional<StackMapTableAttribute> stackMapTableAttribute,
                         List<CodeElement> codeElements) {
        this.insMap = new IdentityHashMap<>();
        this.thisClass = thisClass;
        this.stack = new ArrayList<>();
        this.locals = new ArrayList<>();
        if (!isStatic) locals.add(thisClass);
        for (var pt : methodType.parameterList()) {
            locals.add(pt);
            if (pt.equals(CD_long) || pt.equals(CD_double)) locals.add(null);
        }
        this.stackMap = stackMapTableAttribute.map(a -> a.entries().stream().collect(Collectors.toMap(
                StackMapFrameInfo::target,
                Function.identity()))).orElse(Map.of());
        codeElements.forEach(this::accept);
    }

    public ClassDesc getTypeOf(LoadInstruction li) {
        return insMap.get(li);
    }

    private ClassDesc vtiToStackType(StackMapFrameInfo.VerificationTypeInfo vti) {
        return switch (vti) {
            case ITEM_INTEGER -> CD_int;
            case ITEM_FLOAT -> CD_float;
            case ITEM_DOUBLE -> CD_double;
            case ITEM_LONG -> CD_long;
            case ITEM_UNINITIALIZED_THIS -> thisClass;
            case ObjectVerificationTypeInfo ovti -> ovti.classSymbol();
            default -> null;
        };
    }

    private void push(ClassDesc type) {
        if (!ConstantDescs.CD_void.equals(type)) stack.addLast(type);
    }

    private void pushAt(int pos, ClassDesc... types) {
        for (var t : types)
            if (!ConstantDescs.CD_void.equals(t))
                stack.add(stack.size() + pos, t);
    }

    private boolean doubleAt(int pos) {
        var t  = stack.get(stack.size() + pos);
        return t.equals(CD_long) || t.equals(CD_double);
    }

    private ClassDesc pop() {
        return stack.removeLast();
    }

    private ClassDesc get(int pos) {
        return stack.get(stack.size() + pos);
    }

    private ClassDesc top() {
        return stack.getLast();
    }

    private ClassDesc[] top2() {
        return new ClassDesc[] {stack.get(stack.size() - 2), stack.getLast()};
    }

    private LocalsTypeMapper pop(int i) {
        while (i-- > 0) pop();
        return this;
    }

    private void store(int slot, ClassDesc type) {
        for (int i = locals.size(); i <= slot; i++) locals.add(null);
        locals.set(slot, type);
    }

    private ClassDesc load(int slot) {
        return locals.get(slot);
    }

    private void accept(CodeElement el) {
        switch (el) {
            case ArrayLoadInstruction _ ->
                pop(1).push(pop().componentType());
            case ArrayStoreInstruction _ ->
                pop(3);
            case BranchInstruction i when !i.opcode().isUnconditionalBranch() ->
                pop(1);
            case ConstantInstruction i ->
                push(ClassDesc.ofDescriptor(i.typeKind() == TypeKind.ReferenceType ?
                        i.constantValue().getClass().descriptorString() : i.typeKind().descriptor()));
            case ConvertInstruction i ->
                pop(1).push(ClassDesc.ofDescriptor(i.toType().descriptor()));
            case FieldInstruction i -> {
                switch (i.opcode()) {
                    case GETSTATIC ->
                        push(i.typeSymbol());
                    case GETFIELD ->
                        pop(1).push(i.typeSymbol());
                    case PUTSTATIC ->
                        pop(1);
                    case PUTFIELD ->
                        pop(2);
                }
            }
            case InvokeDynamicInstruction i ->
                pop(i.typeSymbol().parameterCount()).push(i.typeSymbol().returnType());
            case InvokeInstruction i ->
                pop(i.typeSymbol().parameterCount() + (i.opcode() == Opcode.INVOKESTATIC ? 0 : 1))
                        .push(i.typeSymbol().returnType());
            case LoadInstruction i -> {
                push(load(i.slot()));
                insMap.put(i, locals.get(i.slot()));
            }
            case StoreInstruction i ->
                store(i.slot(), pop());
            case MonitorInstruction _ ->
                pop(1);
            case NewMultiArrayInstruction i ->
                pop(i.dimensions()).push(i.arrayType().asSymbol());
            case NewObjectInstruction i ->
                push(i.className().asSymbol());
            case NewPrimitiveArrayInstruction i ->
                pop(1).push(ClassDesc.ofDescriptor(i.typeKind().descriptor()).arrayType());
            case NewReferenceArrayInstruction i ->
                pop(1).push(i.componentType().asSymbol().arrayType());
            case OperatorInstruction i ->
                pop(switch (i.opcode()) {
                    case ARRAYLENGTH, INEG, LNEG, FNEG, DNEG -> 1;
                    default -> 2;
                }).push(ClassDesc.ofDescriptor(i.typeKind().descriptor()));
            case StackInstruction i -> {
                switch (i.opcode()) {
                    case POP -> pop(1);
                    case POP2 -> pop(doubleAt(-1) ? 1 : 2);
                    case DUP -> push(top());
                    case DUP2 -> {
                        if (doubleAt(-1)) {
                            push(top());
                        } else {
                            pushAt(-2, top2());
                        }
                    }
                    case DUP_X1 -> pushAt(-2, top());
                    case DUP_X2 -> pushAt(doubleAt(-2) ? -2 : -3, top());
                    case DUP2_X1 -> {
                        if (doubleAt(-1)) {
                            pushAt(-2, top());
                        } else {
                            pushAt(-3, top2());
                        }
                    }
                    case DUP2_X2 -> {
                        if (doubleAt(-1)) {
                            pushAt(doubleAt(-2) ? -2 : -3, top());
                        } else {
                            pushAt(doubleAt(-3) ? -3 : -4, top2());
                        }
                    }
                    case SWAP -> pushAt(-1, pop());
                }
            }
            case TypeCheckInstruction i ->
                pop(1).push(i.opcode() == Opcode.CHECKCAST ? i.type().asSymbol() : ConstantDescs.CD_int);
            case LabelTarget lt -> {
                var smfi = stackMap.get(lt.label());
                if (smfi != null) {
                    stack.clear();
                    for (var vti : smfi.stack()) {
                        push(vtiToStackType(vti));
                    }
                    locals.clear();
                    int slot = 0;
                    for (var vti : smfi.locals()) {
                        store(slot, vtiToStackType(vti));
                        slot += (vti == ITEM_DOUBLE || vti == ITEM_LONG) ? 2 : 1;
                    }
                }
            }
            default -> {}
        }
    }
}

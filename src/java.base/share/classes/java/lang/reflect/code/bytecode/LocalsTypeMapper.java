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
import java.lang.classfile.Instruction;
import java.lang.classfile.Label;
import java.lang.classfile.Opcode;
import java.lang.classfile.attribute.StackMapFrameInfo;
import java.lang.classfile.attribute.StackMapFrameInfo.*;
import java.lang.classfile.attribute.StackMapTableAttribute;
import java.lang.classfile.instruction.*;
import java.lang.constant.ClassDesc;
import java.lang.constant.ConstantDescs;
import java.lang.constant.DirectMethodHandleDesc;
import java.lang.constant.DynamicConstantDesc;
import java.lang.constant.MethodTypeDesc;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.stream.Collectors;

import static java.lang.classfile.attribute.StackMapFrameInfo.SimpleVerificationTypeInfo.*;
import static java.lang.constant.ConstantDescs.*;
import java.lang.reflect.code.Value;
import java.lang.reflect.code.type.JavaType;
import java.util.ArrayDeque;
import java.util.LinkedHashSet;
import java.util.Set;

final class LocalsTypeMapper {

    private record Link(Slot slot, Link other) {}

    static class Variable {
        private ClassDesc type;
        boolean isSingleValue;
        Value value;

        JavaType type() {
            return JavaType.type(type);
        }
    }

    static class Slot {
        ClassDesc type;
        Link up, down;
        Variable var;
        boolean newValue;
    }

    record Frame(List<ClassDesc> stack, List<Slot> locals) {}

    private static final ClassDesc NULL_TYPE = ClassDesc.ofDescriptor(CD_Object.descriptorString());
    private final Map<Integer, Slot> insMap;
    private final Set<Slot> allSlots;
    private final ClassDesc thisClass;
    private final List<ExceptionCatch> exceptionHandlers;
    private final List<ClassDesc> stack;
    private final List<Slot> locals;
    private final Map<Label, Frame> stackMap;
    private final Map<Label, ClassDesc> newMap;
    private boolean frameDirty;
    final List<Slot> slotsToInitialize;

    LocalsTypeMapper(ClassDesc thisClass,
                         List<ClassDesc> initFrameLocals,
                         List<ExceptionCatch> exceptionHandlers,
                         Optional<StackMapTableAttribute> stackMapTableAttribute,
                         List<CodeElement> codeElements) {
        this.insMap = new HashMap<>();
        this.thisClass = thisClass;
        this.exceptionHandlers = exceptionHandlers;
        this.stack = new ArrayList<>();
        this.locals = new ArrayList<>();
        this.allSlots = new LinkedHashSet<>();
        this.newMap = computeNewMap(codeElements);
        this.slotsToInitialize = new ArrayList<>();
        this.stackMap = stackMapTableAttribute.map(a -> a.entries().stream().collect(Collectors.toMap(
                StackMapFrameInfo::target,
                this::toFrame))).orElse(Map.of());
        for (ClassDesc cd : initFrameLocals) {
            slotsToInitialize.add(cd == null ? null : newSlot(cd, true));
        }
        do {
            for (int i = 0; i < initFrameLocals.size(); i++) {
                store(i, slotsToInitialize.get(i), locals);
            }
            this.frameDirty = false;
            for (int i = 0; i < codeElements.size(); i++) {
                accept(i, codeElements.get(i));
            }
            endOfFlow();
        } while (this.frameDirty);

        // Assign variable to slots and calculate var type
        ArrayDeque<Slot> q = new ArrayDeque<>();
        for (Slot slot : allSlots) {
            if (slot.var == null) {
                Variable var = new Variable();
                q.add(slot);
                int sources = 0;
                var.type = slot.type;
                while (!q.isEmpty()) {
                    Slot v = q.pop();
                    if (v.var == null) {
                        if (v.newValue) sources++;
                        v.var = var;
                        Link l = v.up;
                        while (l != null) {
                            if (var.type == NULL_TYPE) var.type = l.slot.type;
                            if (l.slot.var == null) q.add(l.slot);
                            l = l.other;
                        }
                        l = v.down;
                        while (l != null) {
                            if (var.type == NULL_TYPE) var.type = l.slot.type;
                            if (l.slot.var == null) q.add(l.slot);
                            l = l.other;
                        }
                    }
                }
                var.isSingleValue = sources < 2;
            }
        }
    }

    void link(Slot source, Slot target) {
        if (source != target) {
            target.up = new Link(source, target.up);
            source.down = new Link(target, source.down);
        }
    }

    private Frame toFrame(StackMapFrameInfo smfi) {
        List<ClassDesc> fstack = new ArrayList<>(smfi.stack().size());
        List<Slot> flocals = new ArrayList<>(smfi.locals().size() * 2);
        for (var vti : smfi.stack()) {
            fstack.add(vtiToStackType(vti));
        }
        int i = 0;
        for (var vti : smfi.locals()) {
            store(i, vtiToStackType(vti), flocals, false);
            i += vti == ITEM_DOUBLE || vti == ITEM_LONG ? 2 : 1;
        }
        return new Frame(fstack, flocals);
    }

    private static Map<Label, ClassDesc> computeNewMap(List<CodeElement> codeElements) {
        Map<Label, ClassDesc> newMap = new HashMap<>();
        Label lastLabel = null;
        for (int i = 0; i < codeElements.size(); i++) {
            switch (codeElements.get(i)) {
                case LabelTarget lt -> lastLabel = lt.label();
                case NewObjectInstruction newI -> {
                    if (lastLabel != null) {
                        newMap.put(lastLabel, newI.className().asSymbol());
                    }
                }
                case Instruction _ -> lastLabel = null; //invalidate label
                default -> {} //skip
            }
        }
        return newMap;
    }

    Variable getVarOf(int li) {
        return insMap.get(li).var;
    }

    private Slot newSlot(ClassDesc type, boolean newValue) {
        Slot s = new Slot();
        s.type = type;
        s.newValue = newValue;
        allSlots.add(s);
        return s;
    }

    private ClassDesc vtiToStackType(StackMapFrameInfo.VerificationTypeInfo vti) {
        return switch (vti) {
            case ITEM_INTEGER -> CD_int;
            case ITEM_FLOAT -> CD_float;
            case ITEM_DOUBLE -> CD_double;
            case ITEM_LONG -> CD_long;
            case ITEM_UNINITIALIZED_THIS -> thisClass;
            case ITEM_NULL -> NULL_TYPE;
            case ObjectVerificationTypeInfo ovti -> ovti.classSymbol();
            case UninitializedVerificationTypeInfo uvti ->
                newMap.computeIfAbsent(uvti.newTarget(), l -> {
                    throw new IllegalArgumentException("Unitialized type does not point to a new instruction");
                });
            case ITEM_TOP -> null;
        };
    }

    private void push(ClassDesc type) {
        if (!ConstantDescs.CD_void.equals(type)) stack.add(type);
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
        store(slot, type, locals, true);
    }

    private void store(int slot, ClassDesc type, List<Slot> where, boolean newValue) {
        store(slot, type == null ? null : newSlot(type, newValue), where);
    }

    private void store(int slot, Slot s, List<Slot> where) {
        if (s != null) {
            for (int i = where.size(); i <= slot; i++) where.add(null);
            where.set(slot, s);
        }
    }

    private ClassDesc load(int slot) {
        return locals.get(slot).type;
    }

    private void accept(int elIndex, CodeElement el) {
        switch (el) {
            case ArrayLoadInstruction _ ->
                pop(1).push(pop().componentType());
            case ArrayStoreInstruction _ ->
                pop(3);
            case BranchInstruction i -> {
                switch (i.opcode()) {
                    case IFEQ, IFGE, IFGT, IFLE, IFLT, IFNE, IFNONNULL, IFNULL -> {
                        pop();
                        mergeToTargetFrame(i.target());
                    }
                    case IF_ACMPEQ, IF_ACMPNE, IF_ICMPEQ, IF_ICMPGE, IF_ICMPGT, IF_ICMPLE, IF_ICMPLT, IF_ICMPNE -> {
                        pop(2);
                        mergeToTargetFrame(i.target());
                    }
                    case GOTO, GOTO_W -> {
                        mergeToTargetFrame(i.target());
                        endOfFlow();
                    }
                }
            }
            case ConstantInstruction i ->
                push(switch (i.constantValue()) {
                    case null -> NULL_TYPE;
                    case ClassDesc _ -> CD_Class;
                    case Double _ -> CD_double;
                    case Float _ -> CD_float;
                    case Integer _ -> CD_int;
                    case Long _ -> CD_long;
                    case String _ -> CD_String;
                    case DynamicConstantDesc<?> cd when cd.equals(NULL) -> NULL_TYPE;
                    case DynamicConstantDesc<?> cd -> cd.constantType();
                    case DirectMethodHandleDesc _ -> CD_MethodHandle;
                    case MethodTypeDesc _ -> CD_MethodType;
                });
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
            case IncrementInstruction i -> {
                Slot v = locals.get(i.slot());
                store(i.slot(), load(i.slot()));
                link(v, locals.get(i.slot()));
                insMap.put(elIndex, v);
            }
            case InvokeDynamicInstruction i ->
                pop(i.typeSymbol().parameterCount()).push(i.typeSymbol().returnType());
            case InvokeInstruction i ->
                pop(i.typeSymbol().parameterCount() + (i.opcode() == Opcode.INVOKESTATIC ? 0 : 1))
                        .push(i.typeSymbol().returnType());
            case LoadInstruction i -> {
                push(load(i.slot()));
                insMap.put(elIndex, locals.get(i.slot()));
            }
            case StoreInstruction i -> {
                store(i.slot(), pop());
                insMap.put(elIndex, locals.get(i.slot()));
            }
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
                var frame = stackMap.get(lt.label());
                if (frame != null) {
                    if (!stack.isEmpty() || !locals.isEmpty()) {
                        mergeToTargetFrame(lt.label());
                        endOfFlow();
                    }
                    stack.addAll(frame.stack());
                    locals.addAll(frame.locals());
                }
                for (ExceptionCatch ec : exceptionHandlers) {
                    if (lt.label() == ec.tryStart()) {
                        mergeLocalsToTargetFrame(stackMap.get(ec.handler()));
                    }
                }
            }
            case ReturnInstruction _ , ThrowInstruction _ -> {
                endOfFlow();
            }
            case TableSwitchInstruction tsi -> {
                pop();
                mergeToTargetFrame(tsi.defaultTarget());
                for (var c : tsi.cases()) {
                    mergeToTargetFrame(c.target());
                }
                endOfFlow();
            }
            case LookupSwitchInstruction lsi -> {
                pop();
                mergeToTargetFrame(lsi.defaultTarget());
                for (var c : lsi.cases()) {
                    mergeToTargetFrame(c.target());
                }
                endOfFlow();
            }
            default -> {}
        }
    }

    private void endOfFlow() {
        stack.clear();
        locals.clear();
    }

    private void mergeToTargetFrame(Label target) {
        Frame targetFrame = stackMap.get(target);
        // Merge stack
        assert stack.size() == targetFrame.stack.size();
        for (int i = 0; i < targetFrame.stack.size(); i++) {
            ClassDesc se = stack.get(i);
            ClassDesc fe = targetFrame.stack.get(i);
            if (!se.equals(fe)) {
                if (se.isPrimitive() && CD_int.equals(fe)) {
                    targetFrame.stack.set(i, se); // Override int target frame type with more specific int sub-type
                    this.frameDirty = true;
                } else {
                    stack.set(i, fe); // Override stack type with target frame type
                }
            }
        }
        mergeLocalsToTargetFrame(targetFrame);
    }

    private void mergeLocalsToTargetFrame(Frame targetFrame) {
        // Merge locals
        int lSize = Math.min(locals.size(), targetFrame.locals.size());
        for (int i = 0; i < lSize; i++) {
            Slot le = locals.get(i);
            Slot fe = targetFrame.locals.get(i);
            if (le != null && fe != null) {
                link(fe, le); // Link target frame var with its source
                if (!le.type.equals(fe.type)) {
                    if (le.type.isPrimitive() && CD_int.equals(fe.type) ) {
                        fe.type = le.type; // Override int target frame type with more specific int sub-type
                        this.frameDirty = true;
                    } else {
                        le.type = fe.type; // Override var type with target frame type
                    }
                }
            }
        }
    }
}

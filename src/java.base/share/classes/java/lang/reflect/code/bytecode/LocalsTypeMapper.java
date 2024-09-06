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
import java.lang.classfile.TypeKind;
import java.lang.classfile.attribute.CodeAttribute;
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
import java.lang.classfile.components.ClassPrinter;
import static java.lang.constant.ConstantDescs.*;
import java.lang.reflect.code.Value;
import java.lang.reflect.code.type.JavaType;
import java.util.ArrayDeque;
import java.util.Iterator;
import java.util.LinkedHashSet;
import java.util.NoSuchElementException;
import java.util.Set;

final class LocalsTypeMapper {

    static class Variable {
        private ClassDesc type;
        boolean isSingleValue;
        Value value;

        JavaType type() {
            return JavaType.type(type);
        }

        Object defaultValue() {
            return switch (TypeKind.from(type)) {
                case BooleanType -> false;
                case ByteType -> (byte)0;
                case CharType -> (char)0;
                case DoubleType -> 0d;
                case FloatType -> 0f;
                case IntType -> 0;
                case LongType -> 0l;
                case ReferenceType -> null;
                case ShortType -> (short)0;
                default -> throw new IllegalStateException("Invalid type " + type.displayName());
            };
        }

        @Override
        public String toString() {
            return Integer.toHexString(hashCode()).substring(0, 2) + " " + isSingleValue;
        }
    }

    static final class Slot {

        enum Kind {
            STORE, LOAD, FRAME;
        }

        private record Link(Slot slot, Link other) {}

        int bci, sl;
        Kind kind;
        ClassDesc type;
        Variable var;
        private Link up, down;

        void link(Slot target) {
            if (this != target) {
                target.up = new Link(this, target.up);
                this.down = new Link(target, this.down);
            }
        }

        Iterable<Slot> upSlots() {
            return () -> new LinkIterator(up);
        }

        Iterable<Slot> downSlots() {
            return () -> new LinkIterator(down);
        }

        @Override
        public String toString() {
            return "%d: #%d %s %s var:%s".formatted(bci, sl, kind, type.displayName(),  var == null ? null : var.toString());
        }

        static final class LinkIterator implements Iterator<Slot> {
            Link l;
            public LinkIterator(Link l) {
                this.l = l;
            }

            @Override
            public boolean hasNext() {
                return l != null;
            }

            @Override
            public Slot next() {
                if (l == null) throw new NoSuchElementException();
                Slot s = l.slot();
                l = l.other();
                return s;
            }
        }
    }

    record Frame(List<ClassDesc> stack, List<Slot> locals) {}

    private static final ClassDesc NULL_TYPE = ClassDesc.ofDescriptor(CD_Object.descriptorString());
    private final Map<Integer, Slot> insMap;
    private final LinkedHashSet<Slot> allSlots;
    private final ClassDesc thisClass;
    private final List<ExceptionCatch> exceptionHandlers;
    private final Set<ExceptionCatch> handlersStack;
    private final List<ClassDesc> stack;
    private final List<Slot> locals;
    private final Map<Label, Frame> stackMap;
    private final Map<Label, ClassDesc> newMap;
    private final CodeAttribute ca;
    private boolean frameDirty;
    final List<Slot> slotsToInitialize;

    LocalsTypeMapper(ClassDesc thisClass,
                         List<ClassDesc> initFrameLocals,
                         List<ExceptionCatch> exceptionHandlers,
                         Optional<StackMapTableAttribute> stackMapTableAttribute,
                         List<CodeElement> codeElements,
                         CodeAttribute ca) {
        this.insMap = new HashMap<>();
        this.thisClass = thisClass;
        this.exceptionHandlers = exceptionHandlers;
        this.handlersStack = new LinkedHashSet<>();
        this.stack = new ArrayList<>();
        this.locals = new ArrayList<>();
        this.allSlots = new LinkedHashSet<>();
        this.newMap = computeNewMap(codeElements);
        this.slotsToInitialize = new ArrayList<>();
        this.ca = ca;
        this.stackMap = stackMapTableAttribute.map(a -> a.entries().stream().collect(Collectors.toMap(
                StackMapFrameInfo::target,
                this::toFrame))).orElse(Map.of());
        for (ClassDesc cd : initFrameLocals) {
            slotsToInitialize.add(cd == null ? null : newSlot(cd, Slot.Kind.STORE, -1, slotsToInitialize.size()));
        }
        int initSize = allSlots.size();
        do {
            handlersStack.clear();
            // Slot states reset if running additional rounds with adjusted frames
            if (allSlots.size() > initSize) {
                while (allSlots.size() > initSize) allSlots.removeLast();
                allSlots.forEach(sl -> {
                    sl.up = null;
                    sl.down = null;
                    sl.var = null;
                });
            }
            for (int i = 0; i < initFrameLocals.size(); i++) {
                store(i, slotsToInitialize.get(i), locals);
            }
            this.frameDirty = false;
            int bci = 0;
            for (int i = 0; i < codeElements.size(); i++) {
                var ce = codeElements.get(i);
                accept(i, ce, bci);
                if (ce instanceof Instruction ins) bci += ins.sizeInBytes();
            }
            endOfFlow();
        } while (this.frameDirty);

        // Pull LOADs up the FRAMEs
        boolean changed = true;
        while (changed) {
            changed = false;
            for (Slot slot : allSlots) {
                if (slot.kind == Slot.Kind.FRAME) {
                    for (Slot down : slot.downSlots()) {
                        if (down.kind == Slot.Kind.LOAD) {
                            changed = true;
                            slot.kind = Slot.Kind.LOAD;
                            break;
                        }
                    }
                }
            }
        }

        // Assign variable to slots, calculate var type
        Set<Slot> stores = new LinkedHashSet<>();
        ArrayDeque<Slot> q = new ArrayDeque<>();
        Set<Slot> visited = new LinkedHashSet<>();
        for (Slot slot : allSlots) {
            if (slot.var == null && slot.kind != Slot.Kind.FRAME) {
                Variable var = new Variable();
                q.add(slot);
                var.type = slot.type;
                while (!q.isEmpty()) {
                    Slot sl = q.pop();
                    if (sl.var == null) {
                        sl.var = var;
                        for (Slot down : sl.downSlots()) {
                            if (down.kind != Slot.Kind.FRAME) {
                                if (var.type == NULL_TYPE) var.type = down.type;
                                if (down.var == null) q.add(down);
                            }
                        }
                        if (sl.kind == Slot.Kind.LOAD) {
                            for (Slot up : sl.upSlots()) {
                                if (up.kind != Slot.Kind.FRAME) {
                                    if (var.type == NULL_TYPE) var.type = up.type;
                                    if (up.var == null) {
                                        q.add(up);
                                    }
                                }
                            }
                        }
                    }
                    if (sl.var == var && sl.kind == Slot.Kind.STORE) {
                        stores.add(sl);
                    }
                }

                // Detect single value
                var.isSingleValue = stores.size() < 2;

                // Filter initial stores
                for (var it = stores.iterator(); it.hasNext();) {
                    visited.clear();
                    if (isDominantVar(it.next(), var, visited)) {
                        it.remove();
                    }
//                    for (Slot up : it.next().upSlots()) {
//                        if (up.kind != Slot.Kind.FRAME) {
//                            it.remove();
//                            break;
//                        } else
//                    }
                }

                // Insert var initialization if necessary
                if (stores.size() > 1) {
                    // Add synthetic dominant slot, which needs to be initialized with a default value
                    Slot initialSlot = new Slot();
                    initialSlot.var = var;
                    slotsToInitialize.add(initialSlot);
                    if (var.type == CD_long || var.type == CD_double) {
                        slotsToInitialize.add(null);
                    }
                }
                stores.clear();
            }
        }

//        ClassPrinter.toYaml(ca, ClassPrinter.Verbosity.CRITICAL_ATTRIBUTES, System.out::print);
//
//        System.out.println("digraph {");
//        for (Slot s : allSlots) {
//            System.out.println("    S" + Integer.toHexString(s.hashCode()) + " [label=\"" + s.toString() + "\"]");
//        }
//        System.out.println();
//        for (Slot s : allSlots) {
//            var it = s.downSlots().iterator();
//            if (it.hasNext()) {
//                System.out.print("    S" + Integer.toHexString(s.hashCode()) + " -> {S" + Integer.toHexString(it.next().hashCode()));
//                while (it.hasNext()) {
//                    System.out.print(", S" + Integer.toHexString(it.next().hashCode()));
//                }
//                System.out.println("};");
//            }
//        }
//        System.out.println("}");
    }

    private static boolean isDominantVar(Slot slot, Variable var, Set<Slot> visited) {
        if (visited.add(slot)) {
            for (Slot up : slot.upSlots()) {
                if (up.var == null ? up.kind != Slot.Kind.FRAME || !isDominantVar(up, var, visited) : up.var != var) {
                    return false;
                }
            }
        }
        return true;
    }

    private Frame toFrame(StackMapFrameInfo smfi) {
        List<ClassDesc> fstack = new ArrayList<>(smfi.stack().size());
        List<Slot> flocals = new ArrayList<>(smfi.locals().size() * 2);
        for (var vti : smfi.stack()) {
            fstack.add(vtiToStackType(vti));
        }
        int i = 0;
        int bci = ca.labelToBci(smfi.target());
        for (var vti : smfi.locals()) {
            store(i, vtiToStackType(vti), flocals, Slot.Kind.FRAME, bci);
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

    private Slot newSlot(ClassDesc type, Slot.Kind kind, int bci, int sl) {
        Slot s = new Slot();
        s.kind = kind;
        s.type = type;
        s.bci = bci;
        s.sl = sl;
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

    private void store(int slot, ClassDesc type, int bci) {
        store(slot, type, locals, Slot.Kind.STORE, bci);
    }

    private void store(int slot, ClassDesc type, List<Slot> where, Slot.Kind kind, int bci) {
        store(slot, type == null ? null : newSlot(type, kind, bci, slot), where);
    }

    private void store(int slot, Slot s, List<Slot> where) {
        if (s != null) {
            for (int i = where.size(); i <= slot; i++) where.add(null);
            Slot prev = where.set(slot, s);
            if (prev != null) {
                prev.link(s);
            }
        }
    }

    private ClassDesc load(int slot, int bci) {
        Slot sl = locals.get(slot);
        Slot nsl = newSlot(sl.type, Slot.Kind.LOAD, bci, slot);
        sl.link(nsl);
        return sl.type;
    }

    private void accept(int elIndex, CodeElement el, int bci) {
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
                load(i.slot(), bci);
                insMap.put(-elIndex, locals.get(i.slot()));
                store(i.slot(), CD_int, bci);
                insMap.put(elIndex, locals.get(i.slot()));
                for (var ec : handlersStack) {
                    mergeLocalsToTargetFrame(stackMap.get(ec.handler()));
                }
            }
            case InvokeDynamicInstruction i ->
                pop(i.typeSymbol().parameterCount()).push(i.typeSymbol().returnType());
            case InvokeInstruction i ->
                pop(i.typeSymbol().parameterCount() + (i.opcode() == Opcode.INVOKESTATIC ? 0 : 1))
                        .push(i.typeSymbol().returnType());
            case LoadInstruction i -> {
                push(load(i.slot(), bci));
                insMap.put(elIndex, locals.get(i.slot()));
            }
            case StoreInstruction i -> {
                store(i.slot(), pop(), bci);
                insMap.put(elIndex, locals.get(i.slot()));
                for (var ec : handlersStack) {
                    mergeLocalsToTargetFrame(stackMap.get(ec.handler()));
                }
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
                        handlersStack.add(ec);
                        mergeLocalsToTargetFrame(stackMap.get(ec.handler()));
                    }
                    if (lt.label() == ec.tryEnd()) {
                        handlersStack.remove(ec);
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
                le.link(fe); // Link target frame var with its source
                if (!le.type.equals(fe.type)) {
                    if (le.type.isPrimitive() && CD_int.equals(fe.type) ) {
                        fe.type = le.type; // Override int target frame type with more specific int sub-type
                        this.frameDirty = true;
//                    } else {
//                        le.type = fe.type; // Override var type with target frame type
                    }
                }
            }
        }
    }
}

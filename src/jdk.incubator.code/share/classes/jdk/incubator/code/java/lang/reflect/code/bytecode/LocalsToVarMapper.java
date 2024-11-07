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
package jdk.incubator.code.java.lang.reflect.code.bytecode;

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
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Set;
import java.util.Optional;
import java.util.stream.Collectors;

import static java.lang.classfile.attribute.StackMapFrameInfo.SimpleVerificationTypeInfo.*;
import static java.lang.classfile.attribute.StackMapFrameInfo.SimpleVerificationTypeInfo.NULL;
import static java.lang.constant.ConstantDescs.*;

/**
 * LocalsToVarMapper scans bytecode for slot operations, forms oriented flow graphs of the slot operation segments,
 * analyzes the graphs and maps the segments to distinct variables, calculates each variable type and identifies
 * single-assigned variables and variables requiring initialization in the entry block.
 */
final class LocalsToVarMapper {

    /**
     * Variable identity object result of the LocalsToVarMapper analysis.
     */
    public static final class Variable {
        private ClassDesc type;
        private boolean single;

        /**
         * {@return Variable type}
         */
        ClassDesc type() {
            return type;
        }

        /**
         * {@return whether the variable has only single assignement}
         */
        boolean hasSingleAssignment() {
            return single;
        }
    }

    /**
     * Segment of bytecode related to one local slot, it represents a node in the segment graph.
     */
    private static final class Segment {

        /**
         * Categorization of the segment graph nodes.
         */
        enum Kind {

            /**
             * Segment storing a value into the local slot.
             */
            STORE,

            /**
             * Segment requesting to load value from the local slot.
             */
            LOAD,

            /**
             * Segment forming a frame of connection to other segments.
             * This kind of segment is later either resolved as LOAD or it identifies false connection.
             */
            FRAME;
        }

        /**
         * Link between segments.
         */
        record Link(Segment segment, Link other) {}

        /**
         * Kind of segment.
         * The value is not final, {@link Kind.FRAME} segments may be later resolved to {@link Kind.LOAD}.
         */
        Kind kind;

        /**
         * Segment type.
         * The value is not final, int type may be later changed to {@code boolean}, {@code byte}, {@code short} or {@code char}.
         */
        ClassDesc type;

        /**
         * Variable this segment belongs to.
         * The value is calculated later in the process.
         */
        Variable var;


        /**
         * Incoming segments in the flow graph.
         */
        Link from;

        /**
         * Outgoing segments in the flow graph.
         */
        Link to;

        /**
         * Links this segment to an outgoing segment.
         * @param toSegment outgoing segment
         */
        void link(Segment toSegment) {
            if (this != toSegment) {
                toSegment.from = new Link(this, toSegment.from);
                this.to = new Link(toSegment, this.to);
            }
        }

        /**
         * {@return Iterable over incomming segments.}
         */
        Iterable<Segment> fromSegments() {
            return () -> new LinkIterator(from);
        }

        /**
         * {@return Iterable over outgoing segments.}
         */
        Iterable<Segment> toSegments() {
            return () -> new LinkIterator(to);
        }

        private static final class LinkIterator implements Iterator<Segment> {
            Link l;
            public LinkIterator(Link l) {
                this.l = l;
            }

            @Override
            public boolean hasNext() {
                return l != null;
            }

            @Override
            public Segment next() {
                if (l == null) throw new NoSuchElementException();
                Segment s = l.segment();
                l = l.other();
                return s;
            }
        }
    }

    /**
     * Stack map frame
     */
    private record Frame(List<ClassDesc> stack, List<Segment> locals) {}

    /**
     * Specific instance of CD_Object identifying null initialized objects.
     */
    private static final ClassDesc NULL_TYPE = ClassDesc.ofDescriptor(CD_Object.descriptorString());

    /**
     * Map from instruction index to a segment.
     */
    private final Map<Integer, Segment> insMap;

    /**
     * Set of all involved segments.
     */
    private final LinkedHashSet<Segment> allSegments;

    /**
     * This class descriptor.
     */
    private final ClassDesc thisClass;

    /**
     * All exception handlers.
     */
    private final List<ExceptionCatch> exceptionHandlers;

    /**
     * Actual exception handlers stack.
     */
    private final Set<ExceptionCatch> handlersStack;

    /**
     * Actual stack.
     */
    private final List<ClassDesc> stack;

    /**
     * Actual locals.
     */
    private final List<Segment> locals;

    /**
     * Stack map.
     */
    private final Map<Label, Frame> stackMap;

    /**
     * Map of new object types (to resolve uninitialized verification types in the stack map).
     */
    private final Map<Label, ClassDesc> newMap;

    /**
     * Dirty flag indicates modified stack map frame (sub-int adjustments), so the scanning process must restart
     */
    private boolean frameDirty;

    /**
     * Initial set of slots. Static part comes from method arguments.
     * Later phase of the analysis adds synthetic slots (declarations of multiple-assigned variables)
     * with mandatory initialization in the entry block.
     */
    private final List<Segment> initSlots;

    /**
     * Constructor and executor of the LocalsToVarMapper.
     * @param thisClass This class descriptor.
     * @param initFrameLocals Entry frame locals, expanded form of the method receiver and arguments. Second positions of double slots are null.
     * @param exceptionHandlers Exception handlers.
     * @param stackMapTableAttribute Stack map table attribute.
     * @param codeElements Code elements list. Indexes of this list are keys to the {@link #instructionVar(int) } method.
     */
    public LocalsToVarMapper(ClassDesc thisClass,
                     List<ClassDesc> initFrameLocals,
                     List<ExceptionCatch> exceptionHandlers,
                     Optional<StackMapTableAttribute> stackMapTableAttribute,
                     List<CodeElement> codeElements) {
        this.insMap = new HashMap<>();
        this.thisClass = thisClass;
        this.exceptionHandlers = exceptionHandlers;
        this.handlersStack = new LinkedHashSet<>();
        this.stack = new ArrayList<>();
        this.locals = new ArrayList<>();
        this.allSegments = new LinkedHashSet<>();
        this.newMap = computeNewMap(codeElements);
        this.initSlots = new ArrayList<>();
        this.stackMap = stackMapTableAttribute.map(a -> a.entries().stream().collect(Collectors.toMap(
                StackMapFrameInfo::target,
                this::toFrame))).orElse(Map.of());
        for (ClassDesc cd : initFrameLocals) {
            initSlots.add(cd == null ? null : newSegment(cd, Segment.Kind.STORE));
        }
        int initSize = allSegments.size();

        // Main loop of the scan phase
        do {
            // Reset of the exception handler stack
            handlersStack.clear();
            // Slot states reset if running additional rounds (changed stack map frames)
            if (allSegments.size() > initSize) {
                while (allSegments.size() > initSize) allSegments.removeLast();
                allSegments.forEach(sl -> {
                    sl.from = null;
                    sl.to = null;
                    sl.var = null;
                });
            }
            // Initial frame store
            for (int i = 0; i < initFrameLocals.size(); i++) {
                storeLocal(i, initSlots.get(i), locals);
            }
            this.frameDirty = false;
            // Iteration over all code elements
            for (int i = 0; i < codeElements.size(); i++) {
                var ce = codeElements.get(i);
                scan(i, ce);
            }
            endOfFlow();
        } while (this.frameDirty);

        // Segment graph analysis phase
        // First resolve FRAME segments to LOAD segments if directly followed by a LOAD segment
        // Remaining FRAME segments do not form connection with segments of the same variable and will be ignored.
        boolean changed = true;
        while (changed) {
            changed = false;
            for (Segment segment : allSegments) {
                if (segment.kind == Segment.Kind.FRAME) {
                    for (Segment to : segment.toSegments()) {
                        if (to.kind == Segment.Kind.LOAD) {
                            changed = true;
                            segment.kind = Segment.Kind.LOAD;
                            break;
                        }
                    }
                }
            }
        }

        // Assign variable to segments, calculate var type
        Set<Segment> stores = new LinkedHashSet<>(); // Helper set to collect all STORE segments of a variable
        ArrayDeque<Segment> q = new ArrayDeque<>(); // Working queue
        Set<Segment> visited = new LinkedHashSet<>(); // Helper set to traverse segment graph to filter initial stores
        for (Segment segment : allSegments) {
            // Only STORE and LOAD segments without assigned var are computed
            if (segment.var == null && segment.kind != Segment.Kind.FRAME) {
                Variable var = new Variable(); // New variable
                q.add(segment);
                var.type = segment.type; // Initial variable type
                while (!q.isEmpty()) {
                    Segment se = q.pop();
                    if (se.var == null) {
                        se.var = var; // Assign variable to the segment
                        for (Segment to : se.toSegments()) {
                            // All following LOAD segments belong to the same variable
                            if (to.kind == Segment.Kind.LOAD) {
                                if (var.type == NULL_TYPE) {
                                    var.type = to.type; // Initially null type re-assignemnt
                                }
                                if (to.var == null) {
                                    q.add(to);
                                }
                            }
                        }
                        if (se.kind == Segment.Kind.LOAD) {
                            // Segments preceeding LOAD segment also belong to the same variable
                            for (Segment from : se.fromSegments()) {
                                if (from.kind != Segment.Kind.FRAME) { // FRAME segments are ignored
                                    if (var.type == NULL_TYPE) {
                                        var.type = from.type; // Initially null type re-assignemnt
                                    }
                                    if (from.var == null) {
                                        q.add(from);
                                    }
                                }
                            }
                        }
                    }
                    if (se.var == var && se.kind == Segment.Kind.STORE) {
                        stores.add(se); // Collection of all STORE segments of the variable
                    }
                }

                // Single-assigned variable has only one STORE segment
                var.single = stores.size() < 2;

                // Identification of initial STORE segments
                for (var it = stores.iterator(); it.hasNext();) {
                    visited.clear();
                    Segment s = it.next();
                    if (s.from != null && varDominatesOverSegmentPredecessors(s, var, visited)) {
                        // A store preceeding dominantly with segments of the same variable is not initial
                        it.remove();
                    }
                }

                // Remaining stores are all initial.
                if (stores.size() > 1) {
                    // A synthetic default-initialized dominant segment must be inserted to the variable, if there is more than one initial store segment.
                    // It is not necessary to link it with other variable segments, the analysys ends here.
                    Segment initialSegment = new Segment();
                    initialSegment.var = var;
                    initSlots.add(initialSegment);
                    if (var.type == CD_long || var.type == CD_double) {
                        initSlots.add(null); // Do not forget to alocate second slot for double slots.
                    }
                }
                stores.clear();
            }
        }
    }

    /**
     * {@return Number of slots to initialize at entry block (method receiver + arguments + synthetic variable initialization segments).}
     */
    public int slotsToInit() {
        return initSlots.size();
    }

    /**
     * {@return Variable related to the given initial slot or null}
     * @param initSlot initial slot index
     */
    public Variable initSlotVar(int initSlot) {
        Segment s = initSlots.get(initSlot);
        return s == null ? null : s.var;
    }

    /**
     * Method returns relevant {@link Variable} for instructions operating with local slots,
     * such as {@link LoadInstruction}, {@link StoreInstruction} and {@link IncrementInstruction}.
     * For all other elements it returns {@code null}.
     *
     * Instructions are identified by index into the {@code codeElements} list used in the {@link LocalsToVarMapper} initializer.
     *
     * {@link IncrementInstruction} relates to two potentially distinct variables, one variable to load the value from
     * and one variable to store the incremented value into (see: {@link BytecodeLift#liftBody() }).
     *
     * @param codeElementIndex code element index
     * @return Variable related to the given code element index or null
     */
    public Variable instructionVar(int codeElementIndex) {
        return insMap.get(codeElementIndex).var;
    }

    /**
     * Tests if variable dominates over the segment predecessors.
     * All incoming paths to the segment must lead from segments of the given variable and not of any other variable.
     * The paths may pass through {@code FRAME} segments, which do not belong to any variable and their dominance must be computed.
     * Implementation relies on loops-avoiding breadth-first negative search.
     */
    private static boolean varDominatesOverSegmentPredecessors(Segment segment, Variable var, Set<Segment> visited) {
        if (visited.add(segment)) {
            for (Segment pred : segment.fromSegments()) {
                // Breadth-first
                if (pred.kind != Segment.Kind.FRAME && pred.var != var) {
                    return false;
                }
            }
            for (Segment pred : segment.fromSegments()) {
                // Preceeding FRAME segment implies there is no directly preceeding variable and the dominance test must go deeper
                if (pred.kind == Segment.Kind.FRAME && !varDominatesOverSegmentPredecessors(pred, var, visited)) {
                    return false;
                }
            }
        }
        return true;
    }

    /**
     * Cconverts {@link StackMapFrameInfo} to {@code Frame}, where locals are expanded form ({@code null}-filled second slots for double-slots)
     * of {@code FRAME} segments.
     * @param smfi StackMapFrameInfo
     * @return Frame
     */
    private Frame toFrame(StackMapFrameInfo smfi) {
        List<ClassDesc> fstack = new ArrayList<>(smfi.stack().size());
        List<Segment> flocals = new ArrayList<>(smfi.locals().size() * 2);
        for (var vti : smfi.stack()) {
            fstack.add(vtiToStackType(vti));
        }
        int i = 0;
        for (var vti : smfi.locals()) {
            storeLocal(i, vtiToStackType(vti), flocals, Segment.Kind.FRAME);
            i += vti == DOUBLE || vti == LONG ? 2 : 1;
        }
        return new Frame(fstack, flocals);
    }

    /**
     * {@return map of labels immediately preceding {@link NewObjectInstruction} to the object types}
     * The map is important to resolve uninitialized verification types in the stack map.
     * @param codeElements List of code elements to scan
     */
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

    /**
     * {@return new segment and registers it in {@code allSegments} list}
     * @param type class descriptor of segment type
     * @param kind one of the segment kinds: {@code STORE}, {@code LOAD} or {@code FRAME}
     */
    private Segment newSegment(ClassDesc type, Segment.Kind kind) {
        Segment s = new Segment();
        s.kind = kind;
        s.type = type;
        allSegments.add(s);
        return s;
    }

    /**
     * {@return resolved class descriptor of the stack map frame verification type, custom {@code NULL_TYPE} for {@code ITEM_NULL}
     * or {@code null} for {@code ITEM_TOP}}
     * @param vti stack map frame verification type
     */
    private ClassDesc vtiToStackType(StackMapFrameInfo.VerificationTypeInfo vti) {
        return switch (vti) {
            case INTEGER -> CD_int;
            case FLOAT -> CD_float;
            case DOUBLE -> CD_double;
            case LONG -> CD_long;
            case UNINITIALIZED_THIS -> thisClass;
            case NULL -> NULL_TYPE;
            case ObjectVerificationTypeInfo ovti -> ovti.classSymbol();
            case UninitializedVerificationTypeInfo uvti ->
                newMap.computeIfAbsent(uvti.newTarget(), l -> {
                    throw new IllegalArgumentException("Unitialized type does not point to a new instruction");
                });
            case TOP -> null;
        };
    }

    /**
     * Pushes the class descriptor on {@link #stack}, except for {@code void}.
     * @param type class descriptor
     */
    private void push(ClassDesc type) {
        if (!ConstantDescs.CD_void.equals(type)) stack.add(type);
    }

    /**
     * Pushes the class descriptors on the {@link #stack} at the relative position, except for {@code void}.
     * @param pos position relative to the stack tip
     * @param types class descriptors
     */
    private void pushAt(int pos, ClassDesc... types) {
        for (var t : types)
            if (!ConstantDescs.CD_void.equals(t))
                stack.add(stack.size() + pos, t);
    }

    /**
     * {@return if class descriptor on the {@link #stack} at the relative position is {@code long} or {@code double}}
     * @param pos position relative to the stack tip
     */
    private boolean doubleAt(int pos) {
        var t  = stack.get(stack.size() + pos);
        return t.equals(CD_long) || t.equals(CD_double);
    }

    /**
     * {@return class descriptor poped from the {@link #stack}}
     */
    private ClassDesc pop() {
        return stack.removeLast();
    }

    /**
     * {@return class descriptor from the relative position of the {@link #stack}}
     * @param pos position relative to the stack tip
     */
    private ClassDesc get(int pos) {
        return stack.get(stack.size() + pos);
    }

    /**
     * {@return class descriptor from the tip of the {@link #stack}}
     */
    private ClassDesc top() {
        return stack.getLast();
    }

    /**
     * {@return two class descriptors from the tip of the {@link #stack}}
     */
    private ClassDesc[] top2() {
        return new ClassDesc[] {stack.get(stack.size() - 2), stack.getLast()};
    }

    /**
     * Pops given number of class descriptors from the {@link #stack}.
     * @param i number of class descriptors to pop
     * @return this LocalsToVarMapper
     */
    private LocalsToVarMapper pop(int i) {
        while (i-- > 0) pop();
        return this;
    }

    /**
     * Stores class descriptor as a new {@code STORE} {@link Segment} to the {@link #locals}.
     * The new segment is linked with the previous segment on the same slot position (if any).
     * @param slot locals slot number
     * @param type new segment class descriptor
     */
    private void storeLocal(int slot, ClassDesc type) {
        storeLocal(slot, type, locals, Segment.Kind.STORE);
    }

    /**
     * Stores class descriptor as a new {@link Segment} of given kind to the given list .
     * The new segment is linked with the previous segment on the same slot position (if any).
     * @param slot locals slot number
     * @param type new segment class descriptor
     * @param where target list of segments
     * @param kind new segment kind
     */
    private void storeLocal(int slot, ClassDesc type, List<Segment> where, Segment.Kind kind) {
        storeLocal(slot, type == null ? null : newSegment(type, kind), where);
    }

    /**
     * Stores the {@link Segment} to the given list.
     * The new segment is linked with the previous segment on the same slot position (if any).
     * @param slot locals slot number
     * @param segment the segment to store
     * @param where target list of segments
     */
    private void storeLocal(int slot, Segment segment, List<Segment> where) {
        if (segment != null) {
            for (int i = where.size(); i <= slot; i++) where.add(null);
            Segment prev = where.set(slot, segment);
            if (prev != null) {
                prev.link(segment);
            }
        }
    }

    /**
     * Links existing {@link Segment} of the {@link #locals} with a new {@code LOAD} {@link Segment} with inherited type.
     * @param slot slot number to load
     * @return type of the local
     */
    private ClassDesc loadLocal(int slot) {
        Segment segment = locals.get(slot);
        Segment newSegment = newSegment(segment.type, Segment.Kind.LOAD);
        segment.link(newSegment);
        return segment.type;
    }

    /**
     * Main code element scanning method of the scan loop.
     * @param elementIndex element index
     * @param el code element
     */
    private void scan(int elementIndex, CodeElement el) {
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
                pop(1).push(i.toType().upperBound());
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
            case IncrementInstruction i -> { // Increment instruction maps to two segments
                loadLocal(i.slot());
                insMap.put(-elementIndex - 1, locals.get(i.slot())); // source segment is mapped with -elementIndex - 1 key
                storeLocal(i.slot(), CD_int);
                insMap.put(elementIndex, locals.get(i.slot())); // target segment is mapped with elementIndex key
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
                push(loadLocal(i.slot())); // Load instruction segment is mapped with elementIndex key
                insMap.put(elementIndex, locals.get(i.slot()));
            }
            case StoreInstruction i -> {
                storeLocal(i.slot(), pop());
                insMap.put(elementIndex, locals.get(i.slot()));  // Store instruction segment is mapped with elementIndex key
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
                pop(1).push(i.typeKind().upperBound().arrayType());
            case NewReferenceArrayInstruction i ->
                pop(1).push(i.componentType().asSymbol().arrayType());
            case OperatorInstruction i ->
                pop(switch (i.opcode()) {
                    case ARRAYLENGTH, INEG, LNEG, FNEG, DNEG -> 1;
                    default -> 2;
                }).push(i.typeKind().upperBound());
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
                if (frame != null) { // Here we reached a stack map frame, so we merge actual stack and locals into the frame
                    if (!stack.isEmpty() || !locals.isEmpty()) {
                        mergeToTargetFrame(lt.label());
                        endOfFlow();
                    }
                    // Stack and locals are then taken from the frame
                    stack.addAll(frame.stack());
                    locals.addAll(frame.locals());
                }
                for (ExceptionCatch ec : exceptionHandlers) {
                    if (lt.label() == ec.tryStart()) { // Entering a try block
                        handlersStack.add(ec);
                        mergeLocalsToTargetFrame(stackMap.get(ec.handler()));
                    }
                    if (lt.label() == ec.tryEnd()) { // Leaving a try block
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

    /**
     * Merge of the actual {@link #stack} and {@link #locals} to the target stack map frame
     * @param target label of the target stack map frame
     */
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
                    this.frameDirty = true; // This triggers scan loop to run again, as the stack map frame has been adjusted
                } else {
                    stack.set(i, fe); // Override stack type with target frame type
                }
            }
        }
        mergeLocalsToTargetFrame(targetFrame);
    }


    /**
     * Merge of the actual {@link #locals} to the target stack map frame
     * @param targetFrame target stack map frame
     */
    private void mergeLocalsToTargetFrame(Frame targetFrame) {
        // Merge locals
        int lSize = Math.min(locals.size(), targetFrame.locals.size());
        for (int i = 0; i < lSize; i++) {
            Segment le = locals.get(i);
            Segment fe = targetFrame.locals.get(i);
            if (le != null && fe != null) {
                le.link(fe); // Link target frame var with its source
                if (!le.type.equals(fe.type)) {
                    if (le.type.isPrimitive() && CD_int.equals(fe.type) ) {
                        fe.type = le.type; // Override int target frame type with more specific int sub-type
                        this.frameDirty = true; // This triggers scan loop to run again, as the stack map frame has been adjusted
                    }
                }
            }
        }
    }
}

/*
 * Copyright (c) 2024, 2026, Oracle and/or its affiliates. All rights reserved.
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
import jdk.incubator.code.CodeType;
import jdk.incubator.code.Op;
import jdk.incubator.code.Value;
import jdk.incubator.code.bytecode.impl.BytecodeCompactor;
import jdk.incubator.code.bytecode.impl.ConstantLabelSwitchOp;
import jdk.incubator.code.bytecode.impl.DynamicFuncCallOp;
import jdk.incubator.code.bytecode.impl.LoweringTransformer;
import jdk.incubator.code.dialect.core.CoreOp.*;
import jdk.incubator.code.dialect.core.FunctionType;
import jdk.incubator.code.dialect.core.TupleType;
import jdk.incubator.code.dialect.core.VarType;
import jdk.incubator.code.dialect.java.*;

import java.lang.classfile.*;
import java.lang.classfile.instruction.SwitchCase;
import java.lang.constant.*;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.lang.invoke.StringConcatFactory;
import java.lang.reflect.Member;
import java.lang.reflect.Modifier;
import java.util.*;
import java.util.stream.Stream;

import static java.lang.classfile.Opcode.*;
import static java.lang.constant.ConstantDescs.*;
import static jdk.incubator.code.dialect.java.JavaOp.*;

/**
 * Transformer of code models to bytecode.
 */
public final class BytecodeGenerator {

    private static final DirectMethodHandleDesc DMHD_STRING_CONCAT = ofCallsiteBootstrap(
            StringConcatFactory.class.describeConstable().orElseThrow(),
            "makeConcat",
            CD_CallSite);

    private static final MethodTypeDesc MTD_FIND_FIELD =
            MethodTypeDesc.of(CD_MethodHandle, CD_Class, CD_String, CD_Class);
    private static final MethodTypeDesc MTD_FIND_METHOD =
            MethodTypeDesc.of(CD_MethodHandle, CD_Class, CD_String, CD_MethodType);
    private static final MethodTypeDesc MTD_FIND_SPECIAL =
            MethodTypeDesc.of(CD_MethodHandle, CD_Class, CD_String, CD_MethodType, CD_Class);
    private static final MethodTypeDesc MTD_PRIVATE_LOOKUP_IN =
            MethodTypeDesc.of(CD_MethodHandles_Lookup, CD_Class, CD_MethodHandles_Lookup);

    /**
     * Transforms the invokable operation to bytecode encapsulated in a method of hidden class and exposed
     * for invocation via a method handle.
     *
     * @param l the lookup
     * @param iop the invokable operation to transform to bytecode
     * @return the invoking method handle
     * @param <O> the type of the invokable operation
     */
    public static <O extends Op & Op.Invokable> MethodHandle generate(MethodHandles.Lookup l, O iop) {
        String name = iop instanceof FuncOp fop ? fop.funcName() : "m";
        byte[] classBytes = generateClassData(l, name, iop);

        MethodHandles.Lookup hcl;
        try {
            hcl = l.defineHiddenClassWithClassData(classBytes, l, true, MethodHandles.Lookup.ClassOption.NESTMATE);
        } catch (IllegalAccessException e) {
            throw new RuntimeException(e);
        }

        try {
            FunctionType ft = iop.invokableSignature();
            MethodType mt = MethodRef.toNominalDescriptor(ft).resolveConstantDesc(hcl);
            return hcl.findStatic(hcl.lookupClass(), name, mt);
        } catch (ReflectiveOperationException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Transforms the function operation to bytecode encapsulated in a method of a class file.
     * <p>
     * The name of the method is the function operation's {@link FuncOp#funcName() function name}.
     *
     * @param lookup the lookup
     * @param fop the function operation to transform to bytecode
     * @return the class file bytes
     */
    public static byte[] generateClassData(MethodHandles.Lookup lookup, FuncOp fop) {
        return generateClassData(lookup, fop.funcName(), fop);
    }

    /**
     * Transforms the module operation to bytecode encapsulated in methods of a class file.
     *
     * @param lookup the lookup
     * @param clName the name of the generated class file
     * @param mop the module operation to transform to bytecode
     * @return the class file bytes
     */
    public static byte[] generateClassData(MethodHandles.Lookup lookup,
                                           ClassDesc clName,
                                           ModuleOp mop) {
        return generateClassData(lookup, clName, mop.functionTable());
    }

    /**
     * Transforms the invokable operation to bytecode encapsulated in a method of a class file.
     *
     * @param lookup the lookup
     * @param name the name to use for the method of the class file
     * @param iop the invokable operation to transform to bytecode
     * @return the class file bytes
     * @param <O> the type of the invokable operation
     */
    public static <O extends Op & Op.Invokable> byte[] generateClassData(MethodHandles.Lookup lookup,
                                                                         String name,
                                                                         O iop) {
        String packageName = lookup.lookupClass().getPackageName();
        ClassDesc clsName = ClassDesc.of(packageName.isEmpty()
                ? name
                : packageName + "." + name);
        return generateClassData(lookup, clsName, new LinkedHashMap<>(Map.of(name, iop)));
    }

    private static <O extends Op & Op.Invokable> byte[] generateClassData(MethodHandles.Lookup lookup,
                                                                          ClassDesc clName,
                                                                          SequencedMap<String, ? extends O> ops) {
        ModuleOp module = LoweringTransformer.transform(lookup, ops);
        byte[] classBytes = ClassFile.of().build(clName, clb -> {
            for (var e : module.functionTable().sequencedEntrySet()) {
                generateMethod(lookup, clName, e.getKey(), e.getValue(), clb, module.functionTable());
            }
        });

        // Compact locals of the generated bytecode
        return BytecodeCompactor.transform(classBytes);
    }

    private static void generateMethod(MethodHandles.Lookup lookup,
                                       ClassDesc className,
                                       String methodName,
                                       FuncOp fop,
                                       ClassBuilder clb,
                                       SequencedMap<String, FuncOp> functionTable) {
        MethodTypeDesc mtd = MethodRef.toNominalDescriptor(fop.invokableSignature());
        clb.withMethodBody(methodName, mtd, ClassFile.ACC_PUBLIC | ClassFile.ACC_STATIC,
                cob -> new BytecodeGenerator(lookup, className, List.of(), TypeKind.from(mtd.returnType()),
                                             fop.body().blocks(), cob, functionTable).generate());
    }

    private record Slot(int slot, TypeKind typeKind) {}

    private final MethodHandles.Lookup lookup;
    private final ClassDesc className;
    private final List<Value> capturedValues;
    private final TypeKind returnType;
    private final List<Block> blocks;
    private final CodeBuilder cob;
    private final Label[] blockLabels;
    private final Block[][] blocksCatchMap;
    private final CodeType[] catchBlockTypes;
    private final Label[] tryStartLabels;
    private final Map<Value, Slot> slots;
    private final Map<Block.Parameter, Value> singlePredecessorsValues;
    private final Map<String, ? extends Op.Invokable> functionMap;
    private final Map<Op, Boolean> deferCache;
    private final List<Value> oprOnStack;
    private Block[] recentCatchBlocks;

    private BytecodeGenerator(MethodHandles.Lookup lookup,
                              ClassDesc className,
                              List<Value> capturedValues,
                              TypeKind returnType,
                              List<Block> blocks,
                              CodeBuilder cob,
                              Map<String, ? extends Op.Invokable> functionMap) {
        this.lookup = lookup;
        this.className = className;
        this.capturedValues = capturedValues;
        this.returnType = returnType;
        this.blocks = blocks;
        this.cob = cob;
        this.blockLabels = new Label[blocks.size()];
        this.blocksCatchMap = new Block[blocks.size()][];
        this.catchBlockTypes = new CodeType[blocks.size()];
        this.tryStartLabels = new Label[blocks.size()];
        this.slots = new IdentityHashMap<>();
        this.singlePredecessorsValues = new IdentityHashMap<>();
        this.functionMap = functionMap;
        this.deferCache = new IdentityHashMap<>();
        this.oprOnStack = new ArrayList<>();
    }

    private void setCatchStack(Block.Reference target, Block[] activeCatchBlocks) {
        setCatchStack(target.targetBlock().index(), activeCatchBlocks);
    }

    private void setCatchStack(int blockIndex, Block[] activeCatchBlocks) {
        Block[] prevStack = blocksCatchMap[blockIndex];
        if (prevStack == null) {
            blocksCatchMap[blockIndex] = activeCatchBlocks;
        } else {
            assert Arrays.equals(prevStack, activeCatchBlocks);
        }
    }

    private Label getLabel(Block.Reference target) {
        return getLabel(target.targetBlock().index());
    }

    private Label getLabel(int blockIndex) {
        if (blockIndex == blockLabels.length) {
            return cob.endLabel();
        }
        Label l = blockLabels[blockIndex];
        if (l == null) {
            blockLabels[blockIndex] = l = cob.newLabel();
        }
        return l;
    }

    private Slot allocateSlot(Value v) {
        return slots.computeIfAbsent(v, _ -> {
            TypeKind tk = toTypeKind(v.type());
            return new Slot(cob.allocateLocal(tk), tk);
        });
    }

    private void storeIfUsed(Value v) {
        if (!v.uses().isEmpty()) {
            Slot slot = allocateSlot(v);
            cob.storeLocal(slot.typeKind(), slot.slot());
        } else {
            // Only pop results from stack if the value has no further use (no valid slot)
            switch (toTypeKind(v.type()).slotSize()) {
                case 1 -> cob.pop();
                case 2 -> cob.pop2();
            }
        }
    }

    private void storeOperandsOnStack() {
        storeOperandsOnStack(0);
    }

    private void storeOperandsOnStack(List<Value> values) {
        int keep = 0;
        // Make sure none of the values remains on stack
        while (keep < oprOnStack.size() && !values.contains(oprOnStack.get(keep))) {
            keep++;
        }
        storeOperandsOnStack(keep);
    }

    private void storeOperandsOnStack(int keep) {
        while (oprOnStack.size() > keep) {
            storeIfUsed(oprOnStack.removeLast());
        }
    }

    // loads a local or a deferred value only
    private void load(Value v) {
        v = singlePredecessorsValues.getOrDefault(v, v);
        if (v instanceof Op.Result or &&
                or.op() instanceof ConstantOp constantOp &&
                canDefer(constantOp)) {
            cob.loadConstant(switch (constantOp.value()) {
                case null -> null;
                case Boolean b -> {
                    yield b ? 1 : 0;
                }
                case Byte b -> (int)b;
                case Character ch -> (int)ch;
                case Short s -> (int)s;
                case Constable c -> c.describeConstable().orElseThrow();
                default -> throw new IllegalArgumentException("Unexpected constant value: " + constantOp.value());
            });
        } else {
            Slot slot = slots.get(v);
            if (slot == null) {
                if (v instanceof Op.Result or) {
                    // Handling of deferred variables
                    switch (or.op()) {
                        case VarOp vop ->
                            load(vop.initOperand());
                        case VarAccessOp.VarLoadOp vlop ->
                            load(vlop.varOperand());
                        default ->
                            throw new IllegalStateException("Missing slot for: " + or.op());
                    }
                } else {
                    throw new IllegalStateException("Missing slot for: " + v);
                }
            } else {
                cob.loadLocal(slot.typeKind(), slot.slot());
            }
        }
    }

    private void processFirstOperand(Op op) {
        processOperand(op.operands().getFirst());
    }

    private void processOperand(Value operand) {
        processOperands(List.of(operand), true);
    }

    private void processOperands(Op op) {
        processOperands(op.operands(), true);
    }

    private void processOperands(List<Value> operands, boolean keepPrefix) {
        int firstToLoad = Math.min(oprOnStack.size(), operands.size());
        // match stack suffix against the requested operand prefix
        while (firstToLoad > 0 && !oprOnStack.subList(oprOnStack.size() - firstToLoad, oprOnStack.size())
                                             .equals(operands.subList(0, firstToLoad))) {
            firstToLoad--;
        }
        int remainingOnStack = oprOnStack.size() - firstToLoad;
        if (remainingOnStack == 0 || (keepPrefix && Collections.disjoint(oprOnStack.subList(0, remainingOnStack),
                                                                         operands.subList(firstToLoad, operands.size())))) {
            // no operand burried in the remaining stack
            oprOnStack.subList(remainingOnStack, oprOnStack.size()).clear();
        } else if ((keepPrefix || remainingOnStack == 1)
                && operands.size() >= 2 && firstToLoad == 1
                && oprOnStack.get(remainingOnStack - 1) == operands.get(1)
                && Collections.disjoint(oprOnStack, operands.subList(2, operands.size()))) {
            // swap special case
            int topSize = toTypeKind(operands.get(0).type()).slotSize();
            int belowSize = toTypeKind(operands.get(1).type()).slotSize();
            if (topSize == 1 && belowSize == 1) {
                cob.swap();
            } else if (topSize == 1) {
                cob.dup_x2().pop();
            } else if (belowSize == 1) {
                cob.dup2_x1().pop2();
            } else {
                cob.dup2_x2().pop2();
            }
            oprOnStack.subList(remainingOnStack - 1, oprOnStack.size()).clear();
            firstToLoad = 2;
        } else {
            // operands must be digged and made reloadable
            if (keepPrefix) {
                storeOperandsOnStack(operands);
            } else {
                storeOperandsOnStack();
            }
            firstToLoad = 0;
        }
        for (int i = firstToLoad; i < operands.size(); i++) {
            load(operands.get(i));
        }
    }

    // Some of the operations can be deferred
    private boolean canDefer(Op op) {
        Boolean can = deferCache.get(op);
        if (can == null) {
            can = switch (op) {
                case ConstantOp cop -> canDefer(cop);
                case VarOp vop -> canDefer(vop);
                case VarAccessOp.VarLoadOp vlop -> canDefer(vlop);
                default -> false;
            };
            deferCache.put(op, can);
        }
        return can;
    }

    // Class and dynamic constants cannot be deferred, resolution may throw
    // and dynamic constant bootstrap methods may have side effects
    private static boolean canDefer(ConstantOp op) {
        return !op.resultType().equals(JavaType.J_L_CLASS)
                && !(op.value() instanceof DynamicConstantDesc<?>);
    }

    private static boolean canDefer(VarOp op) {
        if (op.isUninitialized()) {
            // Uninitialized var with single store dominating to all its uses can be deferred
            var uses = op.result().uses();
            var storeUses = uses.stream().filter(u -> u.op() instanceof VarAccessOp.VarStoreOp).toList();
            return storeUses.size() == 1 && uses.stream().allMatch(u -> u.isDominatedBy(storeUses.getFirst()));
        } else {
            // Initialized var used only for loads or var with a single-use entry block parameter operand can be deferred
            return op.result().uses().stream().allMatch(u -> u.op() instanceof VarAccessOp.VarLoadOp)
                    || op.initOperand() instanceof Block.Parameter bp
                        && bp.declaringBlock().isEntryBlock()
                        && bp.uses().size() == 1;
        }
    }

    // Var load can be deferred when do not dominate a var store (conservative deferral refusal)
    private boolean canDefer(VarAccessOp.VarLoadOp op) {
        return op.varOperand().uses().stream()
                 .filter(u -> u.op() instanceof VarAccessOp.VarStoreOp)
                 .noneMatch(store -> store.isDominatedBy(op.result()));
    }

    private static boolean isConditionForCondBrOp(CompareOp op) {
        // Result of op has one use as the operand of a CondBrOp op,
        // and both ops are in the same block

        Set<Op.Result> uses = op.result().uses();
        if (uses.size() != 1) {
            return false;
        }
        Op.Result use = uses.iterator().next();

        if (use.declaringBlock() != op.ancestorBlock()) {
            return false;
        }

        // Check if used in successor
        for (Block.Reference s : use.op().successors()) {
            if (s.arguments().contains(op.result())) {
                return false;
            }
        }

        return use.op() instanceof ConditionalBranchOp;
    }

    static ClassDesc toClassDesc(CodeType t) {
        return switch (t) {
            case VarType vt -> toClassDesc(vt.valueType());
            case JavaType jt -> jt.toNominalDescriptor();
            default ->
                throw new IllegalArgumentException("Bad type: " + t);
        };
    }

    static TypeKind toTypeKind(CodeType t) {
        return switch (t) {
            case VarType vt -> toTypeKind(vt.valueType());
            case PrimitiveType pt -> TypeKind.from(pt.toNominalDescriptor());
            case JavaType _ -> TypeKind.REFERENCE;
            default ->
                throw new IllegalArgumentException("Bad type: " + t);
        };
    }

    private void generate() {
        recentCatchBlocks = new Block[0];

        Block entryBlock = blocks.getFirst();
        assert entryBlock.isEntryBlock();

        // Entry block parameters conservatively require slots
        // Some unused parameters might be declared before others that are used
        List<Block.Parameter> parameters = entryBlock.parameters();
        int paramSlot = 0;
        // Captured values prepend parameters in lambda impl methods
        for (Value cv : capturedValues) {
            slots.put(cv, new Slot(cob.parameterSlot(paramSlot++), toTypeKind(cv.type())));
        }
        for (Block.Parameter bp : parameters) {
            slots.put(bp, new Slot(cob.parameterSlot(paramSlot++), toTypeKind(bp.type())));
        }

        blocksCatchMap[entryBlock.index()] = new Block[0];

        // Process blocks in topological order
        // A jump instruction assumes the false successor block is
        // immediately after, in sequence, to the predecessor
        // since the jump instructions branch on a true condition
        // Conditions are inverted when lowered to bytecode
        for (Block b : blocks) {

            Block[] catchBlocks = blocksCatchMap[b.index()];

            // Ignore inaccessible blocks
            if (catchBlocks == null) {
                continue;
            }

            Label blockLabel = getLabel(b.index());
            cob.labelBinding(blockLabel);

            if (!oprOnStack.isEmpty()) {
                throw new IllegalStateException("stack should be empty here");
            }

            exceptionRegionsChange(catchBlocks);

            // If b is a catch block then the exception argument will be represented on the stack
            if (catchBlockTypes[b.index()] != null) {
                // Retain block argument for exception table generation
                push(b.parameters().getFirst());
            }

            List<Op> ops = b.ops();
            for (int i = 0; i < ops.size() - 1; i++) {
                final Op o = ops.get(i);
                final CodeType oprType = o.resultType();
                final TypeKind rvt = toTypeKind(oprType);
                switch (o) {
                    case ConstantOp op -> {
                        if (!canDefer(op)) {
                            // Emit class and dynamic constants at their original position
                            Object v = op.value();
                            if (v instanceof DynamicConstantDesc<?> dcd) {
                                cob.ldc(dcd);
                            } else if (v == null) {
                                cob.aconst_null();
                            } else {
                                cob.ldc(((JavaType)v).toNominalDescriptor());
                            }
                            push(op.result());
                        }
                    }
                    case VarOp op when op.isUninitialized() -> {}
                    case VarOp op -> {
                        //     %1 : Var<int> = var %0 @"i";
                        if (canDefer(op)) {
                            Slot s = slots.get(op.operands().getFirst());
                            if (s != null) {
                                // Var with a single-use entry block parameter can reuse its slot
                                slots.put(op.result(), s);
                            }
                        } else {
                            processFirstOperand(op);
                            storeIfUsed(op.result());
                        }
                    }
                    case VarAccessOp.VarLoadOp op -> {
                        if (canDefer(op)) {
                            // create alias for the source slot
                            slots.computeIfAbsent(op.result(), r -> slots.get(op.operands().getFirst()));
                        } else {
                            load(op.operands().getFirst());
                            push(op.result());
                        }
                    }
                    case VarAccessOp.VarStoreOp op -> {
                        processOperand(op.operands().get(1));
                        Slot slot = allocateSlot(op.operands().getFirst());
                        cob.storeLocal(slot.typeKind(), slot.slot());
                    }
                    case ConvOp op -> {
                        Value first = op.operands().getFirst();
                        processOperand(first);
                        cob.conversion(toTypeKind(first.type()), rvt);
                        push(op.result());
                    }
                    case NegOp op -> {
                        processFirstOperand(op);
                        switch (rvt) { //this can be moved to CodeBuilder::neg(TypeKind)
                            case INT, BOOLEAN, BYTE, SHORT, CHAR -> cob.ineg();
                            case LONG -> cob.lneg();
                            case FLOAT -> cob.fneg();
                            case DOUBLE -> cob.dneg();
                            default -> throw new IllegalArgumentException("Bad type: " + op.resultType());
                        }
                        push(op.result());
                    }
                    case PosOp op -> {
                        processFirstOperand(op);
                        push(op.result());
                    }
                    case ComplOp op -> {
                        // Lower to x ^ -1
                        processFirstOperand(op);
                        switch (rvt) {
                            case INT, BOOLEAN, BYTE, SHORT, CHAR -> {
                                cob.iconst_m1();
                                cob.ixor();
                            }
                            case LONG -> {
                                cob.ldc(-1L);
                                cob.lxor();
                            }
                            default -> throw new IllegalArgumentException("Bad type: " + op.resultType());
                        }
                        push(op.result());
                    }
                    case NotOp op -> {
                        processFirstOperand(op);
                        cob.ifThenElse(CodeBuilder::iconst_0, CodeBuilder::iconst_1);
                        push(op.result());
                    }
                    case AddOp op -> {
                        processOperands(op);
                        switch (rvt) { //this can be moved to CodeBuilder::add(TypeKind)
                            case INT, BOOLEAN, BYTE, SHORT, CHAR -> cob.iadd();
                            case LONG -> cob.ladd();
                            case FLOAT -> cob.fadd();
                            case DOUBLE -> cob.dadd();
                            default -> throw new IllegalArgumentException("Bad type: " + op.resultType());
                        }
                        push(op.result());
                    }
                    case SubOp op -> {
                        processOperands(op);
                        switch (rvt) { //this can be moved to CodeBuilder::sub(TypeKind)
                            case INT, BOOLEAN, BYTE, SHORT, CHAR -> cob.isub();
                            case LONG -> cob.lsub();
                            case FLOAT -> cob.fsub();
                            case DOUBLE -> cob.dsub();
                            default -> throw new IllegalArgumentException("Bad type: " + op.resultType());
                        }
                        push(op.result());
                    }
                    case MulOp op -> {
                        processOperands(op);
                        switch (rvt) { //this can be moved to CodeBuilder::mul(TypeKind)
                            case INT, BOOLEAN, BYTE, SHORT, CHAR -> cob.imul();
                            case LONG -> cob.lmul();
                            case FLOAT -> cob.fmul();
                            case DOUBLE -> cob.dmul();
                            default -> throw new IllegalArgumentException("Bad type: " + op.resultType());
                        }
                        push(op.result());
                    }
                    case DivOp op -> {
                        processOperands(op);
                        switch (rvt) { //this can be moved to CodeBuilder::div(TypeKind)
                            case INT, BOOLEAN, BYTE, SHORT, CHAR -> cob.idiv();
                            case LONG -> cob.ldiv();
                            case FLOAT -> cob.fdiv();
                            case DOUBLE -> cob.ddiv();
                            default -> throw new IllegalArgumentException("Bad type: " + op.resultType());
                        }
                        push(op.result());
                    }
                    case ModOp op -> {
                        processOperands(op);
                        switch (rvt) { //this can be moved to CodeBuilder::rem(TypeKind)
                            case INT, BOOLEAN, BYTE, SHORT, CHAR -> cob.irem();
                            case LONG -> cob.lrem();
                            case FLOAT -> cob.frem();
                            case DOUBLE -> cob.drem();
                            default -> throw new IllegalArgumentException("Bad type: " + op.resultType());
                        }
                        push(op.result());
                    }
                    case AndOp op -> {
                        processOperands(op);
                        switch (rvt) { //this can be moved to CodeBuilder::and(TypeKind)
                            case INT, BOOLEAN, BYTE, SHORT, CHAR -> cob.iand();
                            case LONG -> cob.land();
                            default -> throw new IllegalArgumentException("Bad type: " + op.resultType());
                        }
                        push(op.result());
                    }
                    case OrOp op -> {
                        processOperands(op);
                        switch (rvt) { //this can be moved to CodeBuilder::or(TypeKind)
                            case INT, BOOLEAN, BYTE, SHORT, CHAR -> cob.ior();
                            case LONG -> cob.lor();
                            default -> throw new IllegalArgumentException("Bad type: " + op.resultType());
                        }
                        push(op.result());
                    }
                    case XorOp op -> {
                        processOperands(op);
                        switch (rvt) { //this can be moved to CodeBuilder::xor(TypeKind)
                            case INT, BOOLEAN, BYTE, SHORT, CHAR -> cob.ixor();
                            case LONG -> cob.lxor();
                            default -> throw new IllegalArgumentException("Bad type: " + op.resultType());
                        }
                        push(op.result());
                    }
                    case LshlOp op -> {
                        processOperands(op);
                        adjustRightTypeToInt(op);
                        switch (rvt) { //this can be moved to CodeBuilder::shl(TypeKind)
                            case BYTE, CHAR, INT, SHORT -> cob.ishl();
                            case LONG -> cob.lshl();
                            default -> throw new IllegalArgumentException("Bad type: " + op.resultType());
                        }
                        push(op.result());
                    }
                    case AshrOp op -> {
                        processOperands(op);
                        adjustRightTypeToInt(op);
                        switch (rvt) { //this can be moved to CodeBuilder::shr(TypeKind)
                            case INT, BYTE, SHORT, CHAR -> cob.ishr();
                            case LONG -> cob.lshr();
                            default -> throw new IllegalArgumentException("Bad type: " + op.resultType());
                        }
                        push(op.result());
                    }
                    case LshrOp op -> {
                        processOperands(op);
                        adjustRightTypeToInt(op);
                        switch (rvt) { //this can be moved to CodeBuilder::ushr(TypeKind)
                            case INT, BYTE, SHORT, CHAR -> cob.iushr();
                            case LONG -> cob.lushr();
                            default -> throw new IllegalArgumentException("Bad type: " + op.resultType());
                        }
                        push(op.result());
                    }
                    case ArrayAccessOp.ArrayLoadOp op -> {
                        processOperands(op);
                        cob.arrayLoad(rvt);
                        push(op.result());
                    }
                    case ArrayAccessOp.ArrayStoreOp op -> {
                        processOperands(op);
                        cob.arrayStore(toTypeKind(((ArrayType)op.operands().getFirst().type()).componentType()));
                        push(op.result());
                    }
                    case ArrayLengthOp op -> {
                        processFirstOperand(op);
                        cob.arraylength();
                        push(op.result());
                    }
                    case CompareOp op -> {
                        if (!isConditionForCondBrOp(op)) {
                            Label falseLabel = cob.newLabel();
                            Label endLabel = cob.newLabel();
                            cob.branch(prepareConditionalBranch(op, true), falseLabel)
                               .iconst_1()
                               .goto_(endLabel)
                               .labelBinding(falseLabel)
                               .iconst_0()
                               .labelBinding(endLabel);
                            push(op.result());
                        }
                        // Processing is deferred to the CondBrOp, do not process the op result
                    }
                    case NewOp op -> {
                        switch (op.constructorReference().signature().returnType()) {
                            case ArrayType at -> {
                                processOperands(op);
                                if (at.dimensions() == 1) {
                                    ClassDesc ctd = at.componentType().toNominalDescriptor();
                                    if (ctd.isPrimitive()) {
                                        cob.newarray(TypeKind.from(ctd));
                                    } else {
                                        cob.anewarray(ctd);
                                    }
                                } else {
                                    cob.multianewarray(at.toNominalDescriptor(), op.operands().size());
                                }
                            }
                            case JavaType jt -> {
                                // new and dup must precede the explicit constructor args
                                storeOperandsOnStack(op.operands());
                                cob.new_(jt.toNominalDescriptor())
                                    .dup();
                                if (op.isVarargs()) {
                                    int varargIndex = op.constructorReference().signature().parameterTypes().size() - 1;
                                    var argOperands = op.operands().subList(0, varargIndex);
                                    processOperands(argOperands, true);
                                    var compType = ((ArrayType) op.constructorReference().signature().parameterTypes().getLast()).componentType();
                                    var varArgOperands = op.operands().subList(varargIndex, op.operands().size());
                                    loadArray(compType, varArgOperands);
                                } else {
                                    processOperands(op);
                                }
                                cob.invokespecial(
                                        ((JavaType) op.resultType()).toNominalDescriptor(),
                                        ConstantDescs.INIT_NAME,
                                        MethodRef.toNominalDescriptor(op.constructorReference().signature())
                                                 .changeReturnType(ConstantDescs.CD_void));
                            }
                            default ->
                                throw new IllegalArgumentException("Invalid return type: "
                                                                    + op.constructorReference().signature().returnType());
                        }
                        push(op.result());
                    }
                    case InvokeOp op -> {
                        // loadArray() reloads each element after pushing array/index
                        // any stack operand must be made reloadable now
                        if (op.isVarArgs()) {
                            storeOperandsOnStack(op.varArgOperands());
                        }
                        // Resolve referenced class to determine if interface
                        MethodRef md = op.invokeReference();
                        JavaType refType = (JavaType)md.refType();
                        ClassDesc specialCaller = lookup.lookupClass().describeConstable().get();
                        MethodTypeDesc mDesc = MethodRef.toNominalDescriptor(md.signature());
                        MethodHandles.Lookup protectedAccessLookup = null;
                        if (op.invokeKind() == InvokeOp.InvokeKind.SUPER) {
                            // invokeExact needs the handle below receiver and args
                            // any stack operand must be made reloadable now
                            storeOperandsOnStack(op.operands());
                            // resolve the method handle with lookup.findSpecial using the lookup's class as the specialCaller
                            // and cache it in a dynamic constant
                            // the original lookup is stored in class data.
                            cob.ldc(DynamicConstantDesc.ofNamed(BSM_INVOKE, DEFAULT_NAME, CD_MethodHandle,
                                    MethodHandleDesc.ofMethod(DirectMethodHandleDesc.Kind.VIRTUAL,
                                            CD_MethodHandles_Lookup, "findSpecial", MTD_FIND_SPECIAL),
                                    DynamicConstantDesc.ofNamed(BSM_CLASS_DATA, DEFAULT_NAME, CD_MethodHandles_Lookup),
                                    refType.toNominalDescriptor(), md.name(), mDesc, specialCaller));
                        } else {
                            if (!(refType instanceof ArrayType)) { // array methods (e.g. clone) are excluded
                                protectedAccessLookup = findProtectedAccessLookup(
                                        op.invokeKind() == InvokeOp.InvokeKind.STATIC ? null : receiverClass(op.operands()),
                                        l -> {
                                            var info = l.revealDirect(md.resolveToHandle(l, op.invokeKind()));
                                            return Modifier.isProtected(info.getModifiers())
                                                    && !info.getDeclaringClass().getPackageName().equals(l.lookupClass().getPackageName());
                                        });
                                if (protectedAccessLookup != null) {
                                    specialCaller = protectedAccessLookup.lookupClass().describeConstable().get();
                                    lookupHandle(op.operands(), protectedAccessLookup, op.invokeKind() == InvokeOp.InvokeKind.STATIC ? refType.toNominalDescriptor() : specialCaller,
                                            md.name(), mDesc, op.invokeKind() == InvokeOp.InvokeKind.STATIC ? "findStatic" : "findVirtual");
                                }
                            }
                        }
                        if (op.isVarArgs()) {
                            processOperands(op.argOperands(), true);
                            var varArgOperands = op.varArgOperands();
                            var compType = ((ArrayType) op.invokeReference().signature().parameterTypes().getLast()).componentType();
                            loadArray(compType, varArgOperands);
                        } else {
                            processOperands(op);
                        }
                        Class<?> refClass;
                        try {
                             refClass = (Class<?>)refType.erasure().resolve(lookup);
                        } catch (ReflectiveOperationException e) {
                            throw new IllegalArgumentException(e);
                        }
                        boolean isInterface = refClass.isInterface();
                        switch (op.invokeKind()) {
                            case STATIC -> {
                                if (protectedAccessLookup != null) {
                                    cob.invokevirtual(CD_MethodHandle, "invokeExact", mDesc);
                                } else {
                                    cob.invokestatic(refType.toNominalDescriptor(), md.name(), mDesc, isInterface);
                                }
                            }
                            case INSTANCE -> {
                                if (protectedAccessLookup != null) {
                                    cob.invokevirtual(CD_MethodHandle, "invokeExact", mDesc.insertParameterTypes(0, specialCaller));
                                } else {
                                    cob.invoke(isInterface ? INVOKEINTERFACE : INVOKEVIRTUAL,
                                               refType.toNominalDescriptor(), md.name(), mDesc, isInterface);
                                }
                            }
                            case SUPER ->
                                    cob.invokevirtual(CD_MethodHandle,
                                                      "invokeExact",
                                                      mDesc.insertParameterTypes(0, specialCaller));
                        }
                        ClassDesc ret = toClassDesc(op.resultType());
                        if (!ret.isPrimitive() && !ret.equals(mDesc.returnType())) {
                            // Explicit cast if method return type differs
                            cob.checkcast(ret);
                        }
                        push(op.result());
                    }
                    case FuncCallOp op -> {
                        Op.Invokable fop = functionMap.get(op.funcName());
                        if (fop == null) {
                            throw new IllegalArgumentException("Could not resolve function: " + op.funcName());
                        }
                        processOperands(op);
                        MethodTypeDesc mDesc = MethodRef.toNominalDescriptor(fop.invokableSignature());
                        cob.invoke(
                                INVOKESTATIC,
                                className,
                                op.funcName(),
                                mDesc,
                                false);
                        ClassDesc ret = toClassDesc(op.resultType());
                        if (ret.isClassOrInterface() && !ret.equals(mDesc.returnType())) {
                            // Explicit cast if method return type differs
                            cob.checkcast(ret);
                        }
                        push(op.result());
                    }
                    case FieldAccessOp.FieldLoadOp op -> fieldAccess(op);
                    case FieldAccessOp.FieldStoreOp op -> fieldAccess(op);
                    case InstanceOfOp op -> {
                        processFirstOperand(op);
                        cob.instanceOf(((JavaType) op.targetType()).toNominalDescriptor());
                        push(op.result());
                    }
                    case CastOp op -> {
                        processFirstOperand(op);
                        cob.checkcast(((JavaType) op.targetType()).toNominalDescriptor());
                        push(op.result());
                    }
                    case DynamicFuncCallOp op -> {
                        Op.Invokable fop = functionMap.get(op.funcName());
                        if (fop == null) {
                            throw new IllegalArgumentException("Could not resolve function: " + op.funcName());
                        }
                        processOperands(op);
                        cob.invokedynamic(DynamicCallSiteDesc.of(
                                op.bootstrapMethod(),
                                op.invocationName(),
                                op.invocationType(),
                                op.interfaceMethodType(),
                                MethodHandleDesc.ofMethod(DirectMethodHandleDesc.Kind.STATIC,
                                        className,
                                        op.funcName(),
                                        MethodRef.toNominalDescriptor(fop.invokableSignature())),
                                op.dynamicMethodType()));
                        push(op.result());
                    }
                    case ConcatOp op -> {
                        processOperands(op);
                        cob.invokedynamic(DynamicCallSiteDesc.of(DMHD_STRING_CONCAT, MethodTypeDesc.of(CD_String,
                                toClassDesc(op.operands().get(0).type()),
                                toClassDesc(op.operands().get(1).type()))));
                        push(op.result());
                    }
                    case MonitorOp.MonitorEnterOp op -> {
                        processFirstOperand(op);
                        cob.monitorenter();
                    }
                    case MonitorOp.MonitorExitOp op -> {
                        processFirstOperand(op);
                        cob.monitorexit();
                    }
                    default ->
                        throw new UnsupportedOperationException("Unsupported operation: " + ops.get(i));
                }
            }
            Op top = b.terminatingOp();
            switch (top) {
                case ReturnOp op -> {
                    if (returnType != TypeKind.VOID) {
                        processFirstOperand(op);
                        // @@@ box, unbox, cast here ?
                    }
                    cob.return_(returnType);
                }
                case ThrowOp op -> {
                    processFirstOperand(op);
                    cob.athrow();
                }
                case BranchOp op -> {
                    setCatchStack(op.branch(), recentCatchBlocks);

                    assignBlockArguments(op.branch());
                    cob.goto_(getLabel(op.branch()));
                }
                case ConditionalBranchOp op -> {
                    setCatchStack(op.trueBranch(), recentCatchBlocks);
                    setCatchStack(op.falseBranch(), recentCatchBlocks);

                    if (getConditionForCondBrOp(op) instanceof CompareOp cop) {
                        // Processing of the BinaryTestOp was deferred, so it can be merged with CondBrOp
                        conditionalBranch(prepareConditionalBranch(cop, false), op.trueBranch(), op.falseBranch());
                    } else {
                        processOperands(op.operands(), false);
                        conditionalBranch(IFEQ, op.trueBranch(), op.falseBranch());
                    }
                }
                case ConstantLabelSwitchOp op -> {
                    op.successors().forEach(t -> setCatchStack(t, recentCatchBlocks));
                    var cases = new ArrayList<SwitchCase>();
                    int lo = Integer.MAX_VALUE;
                    int hi = Integer.MIN_VALUE;
                    Label defTarget = null;
                    // each switch edge needs its own argument transfer
                    Map<Block.Reference, Label> transfers = new LinkedHashMap<>();
                    for (int i = 0; i < op.labels().size(); i++) {
                        Integer val = op.labels().get(i);
                        Block.Reference reference = op.successors().get(i);
                        Label target = needToAssignBlockArguments(reference)
                                ? transfers.computeIfAbsent(reference, _ -> cob.newLabel())
                                : getLabel(reference);
                        if (val == null) { // default target has null label value
                            defTarget = target;
                        } else {
                            cases.add(SwitchCase.of(val, target));
                            if (val < lo) lo = val;
                            if (val > hi) hi = val;
                        }
                    }
                    if (defTarget == null) {
                        throw new IllegalArgumentException("Missing default target");
                    }
                    processOperands(op.operands(), false);
                    if (tableSwitchOverLookupSwitch(lo, hi, cases.size())) {
                        cob.tableswitch(defTarget, cases);
                    } else {
                        cob.lookupswitch(defTarget, cases);
                    }
                    for (var transfer : transfers.entrySet()) {
                        cob.labelBinding(transfer.getValue());
                        assignBlockArguments(transfer.getKey());
                        cob.goto_(getLabel(transfer.getKey()));
                    }
                }
                case ExceptionRegionEnter op -> {
                    List<Block.Reference> enteringCatchBlocks = op.catchReferences();
                    List<CodeType> enteringCatchTypes = op.catchTypes();
                    Block[] activeCatchBlocks = Arrays.copyOf(recentCatchBlocks, recentCatchBlocks.length + enteringCatchBlocks.size());
                    int i = recentCatchBlocks.length;
                    int catchIndex = 0;
                    for (Block.Reference catchRef : enteringCatchBlocks) {
                        catchBlockTypes[catchRef.targetBlock().index()] = enteringCatchTypes.get(catchIndex++);
                        activeCatchBlocks[i++] = catchRef.targetBlock();
                        setCatchStack(catchRef, recentCatchBlocks);
                    }
                    setCatchStack(op.startReference(), activeCatchBlocks);

                    assignBlockArguments(op.startReference());
                    cob.goto_(getLabel(op.startReference()));
                }
                case ExceptionRegionExit op -> {
                    List<Block.Reference> exitingCatchBlocks = op.enterOp().catchReferences().reversed();
                    Block[] activeCatchBlocks = Arrays.copyOf(recentCatchBlocks, recentCatchBlocks.length - exitingCatchBlocks.size());
                    setCatchStack(op.endReference(), activeCatchBlocks);

                    // Assert block exits in reverse order
                    int i = recentCatchBlocks.length;
                    for (Block.Reference catchRef : exitingCatchBlocks) {
                        assert catchRef.targetBlock() == recentCatchBlocks[--i];
                    }

                    assignBlockArguments(op.endReference());
                    cob.goto_(getLabel(op.endReference()));
                }
                case UnreachableOp _ ->
                    cob.aconst_null().athrow();
                default ->
                    throw new UnsupportedOperationException("Terminating operation not supported: " + top);
            }
        }
        exceptionRegionsChange(new Block[0]);
    }

    private interface ProtectedAccessPredicate {
        boolean check(MethodHandles.Lookup l) throws ReflectiveOperationException;
    }

    private MethodHandles.Lookup findProtectedAccessLookup(Class<?> receiverClass, ProtectedAccessPredicate predicate) {
        MethodHandles.Lookup l = lookup;
        while (l != null) {
            // Resolving a protected instance member may narrow the handle's receiver
            // to the lookup class. Skip lookups incompatible with the model's receiver
            // and try an enclosing class instead.
            if (receiverClass == null || l.lookupClass().isAssignableFrom(receiverClass)) {
                try {
                    return predicate.check(l) ? l : null;
                } catch (ReflectiveOperationException | IllegalArgumentException _) {}
            }
            Class<?> enclClass = l.lookupClass().getEnclosingClass();
            try {
                l = enclClass != null ? MethodHandles.privateLookupIn(enclClass, lookup) : null;
            } catch (IllegalAccessException _) {
                l = null;
            }
        }
        return null;
    }

    private Class<?> receiverClass(List<Value> operands) {
        try {
            return (Class<?>) ((JavaType) operands.getFirst().type()).erasure().resolve(lookup);
        } catch (ReflectiveOperationException ex) {
            return null;
        }
    }

    private void lookupHandle(List<Value> explicitOperands, MethodHandles.Lookup resLookup, ClassDesc owner, String name, ConstantDesc type, String finder) {
        // handle must precede any operand
        storeOperandsOnStack(explicitOperands);

        // resolve the method handle with the specified finder and lookup and cache it
        // in a dynamic constant
        cob.ldc(DynamicConstantDesc.ofNamed(BSM_INVOKE, DEFAULT_NAME, CD_MethodHandle,
                MethodHandleDesc.ofMethod(DirectMethodHandleDesc.Kind.VIRTUAL, CD_MethodHandles_Lookup,
                        finder, type instanceof MethodTypeDesc ? MTD_FIND_METHOD : MTD_FIND_FIELD),
                resLookup == lookup
                        ? DynamicConstantDesc.ofNamed(BSM_CLASS_DATA, DEFAULT_NAME, CD_MethodHandles_Lookup)
                        : DynamicConstantDesc.ofNamed(BSM_INVOKE, DEFAULT_NAME, CD_MethodHandles_Lookup,
                                MethodHandleDesc.ofMethod(DirectMethodHandleDesc.Kind.STATIC,
                                        CD_MethodHandles, "privateLookupIn", MTD_PRIVATE_LOOKUP_IN),
                                resLookup.lookupClass().describeConstable().orElseThrow(),
                                DynamicConstantDesc.ofNamed(BSM_CLASS_DATA, DEFAULT_NAME, CD_MethodHandles_Lookup)),
                owner, name, type));
    }

    private void fieldAccess(FieldAccessOp op) {
        FieldRef ref = op.fieldReference();
        JavaType refType = (JavaType) ref.refType();
        ClassDesc fieldType = ((JavaType) ref.type()).toNominalDescriptor();
        boolean store = op instanceof FieldAccessOp.FieldStoreOp;
        boolean isStatic = op.operands().size() == (store ? 1 : 0);
        ClassDesc caller = lookup.lookupClass().describeConstable().orElseThrow();
        MethodHandles.Lookup protectedAccessLookup = findProtectedAccessLookup(
                isStatic ? null : receiverClass(op.operands()),
                l -> {
                    Member m = ref.resolveToField(l);
                    return Modifier.isProtected(m.getModifiers())
                            && !m.getDeclaringClass().getPackageName().equals(lookup.lookupClass().getPackageName());
                });
        if (protectedAccessLookup != null) {
            caller = protectedAccessLookup.lookupClass().describeConstable().orElseThrow();
            lookupHandle(op.operands(), protectedAccessLookup, isStatic ? refType.toNominalDescriptor() : caller, ref.name(),
                    fieldType, "find" + (isStatic ? "Static" : "") + (store ? "Setter" : "Getter"));
        }
        processOperands(op);
        if (protectedAccessLookup != null) {
            cob.invokevirtual(CD_MethodHandle,
                              "invokeExact",
                              isStatic ? (store ? MethodTypeDesc.of(CD_void, fieldType)
                                                : MethodTypeDesc.of(fieldType))
                                       : (store ? MethodTypeDesc.of(CD_void, caller, fieldType)
                                                : MethodTypeDesc.of(fieldType, caller)));
        } else {
            cob.fieldAccess(isStatic ? (store ? PUTSTATIC : GETSTATIC)
                                     : (store ? PUTFIELD : GETFIELD),
                            refType.toNominalDescriptor(),
                            ref.name(),
                            fieldType);
        }
        if (!store) {
            ClassDesc ret = toClassDesc(op.resultType());
            if (!ret.isPrimitive() && !ret.equals(fieldType)) {
                // Explicit cast if field type differs
                cob.checkcast(ret);
            }
            push(op.result());
        }
    }

    private void loadArray(JavaType compType, List<Value> array) {
        cob.loadConstant(array.size());
        var compTypeDesc = compType.toNominalDescriptor();
        var typeKind = TypeKind.from(compTypeDesc);
        if (compTypeDesc.isPrimitive()) {
            cob.newarray(typeKind);
        } else {
            cob.anewarray(compTypeDesc);
        }
        for (int j = 0; j < array.size(); j++) {
            // we duplicate array value on the stack to be consumed by arrayStore
            // after completion of this loop the array value will be on top of the stack
            cob.dup();
            cob.loadConstant(j);
            load(array.get(j));
            cob.arrayStore(typeKind);
        }
    }

    private void exceptionRegionsChange(Block[] newCatchBlocks) {
        if (!Arrays.equals(recentCatchBlocks, newCatchBlocks)) {
            int i = recentCatchBlocks.length - 1;
            Label currentLabel = cob.newBoundLabel();
            // Exit catch blocks missing in the newCatchBlocks
            while (i >=0 && (i >= newCatchBlocks.length || recentCatchBlocks[i] != newCatchBlocks[i])) {
                Block catchBlock = recentCatchBlocks[i--];
                CodeType catchType = catchBlockTypes[catchBlock.index()];
                Label tryStart = tryStartLabels[catchBlock.index()];
                Label handler = getLabel(catchBlock.index());
                switch (catchType) {
                    case TupleType tt ->
                        tt.componentTypes().forEach(type ->
                                cob.exceptionCatch(tryStart, currentLabel, handler, ((JavaType) type).toNominalDescriptor()));
                    case ClassType ct ->
                        cob.exceptionCatch(tryStart, currentLabel, handler, ct.toNominalDescriptor());
                    case PrimitiveType pt when pt.equals(JavaType.VOID) ->
                        cob.exceptionCatchAll(tryStart, currentLabel, handler);
                    default ->
                        throw new IllegalArgumentException("Bad catch type: " + catchType);
                }
                tryStartLabels[catchBlock.index()] = null;
            }
            // Fill tryStartLabels for new entries
            while (++i < newCatchBlocks.length) {
                tryStartLabels[newCatchBlocks[i].index()] = currentLabel;
            }
            recentCatchBlocks = newCatchBlocks;
        }
    }

    // Determine whether to issue a tableswitch or a lookupswitch
    // instruction.
    private static boolean tableSwitchOverLookupSwitch(long lo, long hi, long nlabels) {
            long table_space_cost = 4 + (hi - lo + 1); // words
            long table_time_cost = 3; // comparisons
            long lookup_space_cost = 3 + 2 * nlabels;
            long lookup_time_cost = nlabels;
            return
                nlabels > 0 &&
                table_space_cost + 3 * table_time_cost <=
                lookup_space_cost + 3 * lookup_time_cost;
    }

    private void push(Value res) {
        if (res.type().equals(JavaType.VOID)) return;
        var uses = res.uses();
        Op op;
        if (uses.size() == 1
                && uses.getFirst().declaringBlock() == res.declaringBlock()
                && !canDefer(op = uses.getFirst().op())
                // one value can be used in mutliple operands of a single op
                && Stream.concat(op.operands().stream(), op.successors().stream()
                        .flatMap(r -> r.arguments().stream()))
                        .filter(res::equals).limit(2).count() == 1) {
            oprOnStack.add(res);
        } else {
            storeIfUsed(res);
        }
    }

    // the rhs of any shift instruction must be int or smaller -> convert longs
    private void adjustRightTypeToInt(Op op) {
        CodeType right = op.operands().getLast().type();
        if (right.equals(JavaType.LONG)) {
            cob.conversion(toTypeKind(right), TypeKind.INT);
        }
    }

    private static Op getConditionForCondBrOp(ConditionalBranchOp op) {
        Value p = op.predicateOperand();
        if (p.uses().size() != 1) {
            return null;
        }

        if (p.declaringBlock() != op.ancestorBlock()) {
            return null;
        }

        // Check if used in successor
        for (Block.Reference s : op.successors()) {
            if (s.arguments().contains(p)) {
                return null;
            }
        }

        if (p instanceof Op.Result or) {
            return or.op();
        } else {
            return null;
        }
    }

    private void conditionalBranch(Opcode reverseOpcode, Block.Reference trueBlock, Block.Reference falseBlock) {
        if (needToAssignBlockArguments(falseBlock)) {
            Label falseLabel = cob.newLabel();
            cob.branch(reverseOpcode, falseLabel);
            assignBlockArguments(trueBlock);
            cob.goto_(getLabel(trueBlock))
               .labelBinding(falseLabel);
            assignBlockArguments(falseBlock);
            cob.goto_(getLabel(falseBlock));
        } else {
            cob.branch(reverseOpcode, getLabel(falseBlock));
            assignBlockArguments(trueBlock);
            cob.goto_(getLabel(trueBlock));
        }
    }

    private Opcode prepareConditionalBranch(CompareOp op, boolean keepPrefix) {
        Value firstOperand = op.operands().get(0);
        TypeKind typeKind = toTypeKind(firstOperand.type());
        Value secondOperand = op.operands().get(1);
        if (isZeroIntOrNullConstant(secondOperand)) {
            processOperands(List.of(firstOperand), keepPrefix);
            return switch (typeKind) {
                case INT, BOOLEAN, BYTE, SHORT, CHAR ->
                    switch (op) {
                        case EqOp _ -> IFNE;
                        case NeqOp _ -> IFEQ;
                        case GtOp _ -> IFLE;
                        case GeOp _ -> IFLT;
                        case LtOp _ -> IFGE;
                        case LeOp _ -> IFGT;
                        default ->
                            throw new UnsupportedOperationException(op + " on int");
                    };
                case REFERENCE ->
                    switch (op) {
                        case EqOp _ -> IFNONNULL;
                        case NeqOp _ -> IFNULL;
                        default ->
                            throw new UnsupportedOperationException(op + " on Object");
                    };
                default ->
                    throw new UnsupportedOperationException(op + " on " + op.operands().get(0).type());
            };
        }
        processOperands(op.operands(), keepPrefix);
        return switch (typeKind) {
            case INT, BOOLEAN, BYTE, SHORT, CHAR ->
                switch (op) {
                    case EqOp _ -> IF_ICMPNE;
                    case NeqOp _ -> IF_ICMPEQ;
                    case GtOp _ -> IF_ICMPLE;
                    case GeOp _ -> IF_ICMPLT;
                    case LtOp _ -> IF_ICMPGE;
                    case LeOp _ -> IF_ICMPGT;
                    default ->
                        throw new UnsupportedOperationException(op + " on int");
                };
            case REFERENCE ->
                switch (op) {
                    case EqOp _ -> IF_ACMPNE;
                    case NeqOp _ -> IF_ACMPEQ;
                    default ->
                        throw new UnsupportedOperationException(op + " on Object");
                };
            case FLOAT -> {
                // the instruction that come after fcmpg/fcmpl, branch for the false cases
                // f < g -> fcmpg + ifge
                // f <= g -> fcmpg + ifgt
                // f == g -> fcmpg/fcmpl + ifne
                // f > g -> fcmpl + ifle
                // f >= g -> fcmpl + iflt
                if (op instanceof LtOp || op instanceof LeOp) {
                    cob.fcmpg();
                } else {
                    cob.fcmpl();
                }
                yield reverseIfOpcode(op);
            }
            case LONG -> {
                cob.lcmp();
                yield reverseIfOpcode(op);
            }
            case DOUBLE -> {
                if (op instanceof LtOp || op instanceof LeOp) {
                    cob.dcmpg();
                } else {
                    cob.dcmpl();
                }
                yield reverseIfOpcode(op);
            }
            default ->
                throw new UnsupportedOperationException(op + " on " + op.operands().get(0).type());
        };
    }

    private boolean isZeroIntOrNullConstant(Value v) {
        return v instanceof Op.Result or
                && or.op() instanceof ConstantOp cop
                && switch (cop.value()) {
                    case null -> true;
                    case Integer i -> i == 0;
                    case Boolean b -> !b;
                    case Byte b -> b == 0;
                    case Short s -> s == 0;
                    case Character ch -> ch == 0;
                    default -> false;
                };
    }

    private static Opcode reverseIfOpcode(CompareOp op) {
        return switch (op) {
            case EqOp _ -> IFNE;
            case NeqOp _ -> IFEQ;
            case GtOp _ -> IFLE;
            case GeOp _ -> IFLT;
            case LtOp _ -> IFGE;
            case LeOp _ -> IFGT;
            default ->
                throw new UnsupportedOperationException(op.toString());
        };
    }

    private boolean needToAssignBlockArguments(Block.Reference ref) {
        List<Value> sargs = ref.arguments();
        List<Block.Parameter> bargs = ref.targetBlock().parameters();
        boolean need = false;
        for (int i = 0; i < bargs.size(); i++) {
            Block.Parameter barg = bargs.get(i);
            if (!barg.uses().isEmpty() && !barg.equals(sargs.get(i))) {
                need = true;
                allocateSlot(barg);
            }
        }
        return need;
    }

    private void assignBlockArguments(Block.Reference ref) {
        Block target = ref.targetBlock();
        List<Value> sargs = ref.arguments();
        if (catchBlockTypes[target.index()] != null) {
            // Jumping to an exception handler, exception parameter is expected on stack
            processOperands(List.of(sargs.getFirst()), false);
        } else if (target.predecessorReferences().size() > 1) {
            List<Block.Parameter> bargs = target.parameters();
            // First push successor arguments on the stack, then pop and assign
            // so as not to overwrite slots that are reused slots at different argument positions
            List<Value> values = new ArrayList<>();
            for (int i = 0; i < bargs.size(); i++) {
                Block.Parameter barg = bargs.get(i);
                Value value = sargs.get(i);
                if (!barg.equals(value)) {
                    values.add(value);
                }
            }
            processOperands(values, false);
            for (int i = bargs.size() - 1; i >= 0; i--) {
                Block.Parameter barg = bargs.get(i);
                if (!barg.equals(sargs.get(i))) {
                    storeIfUsed(barg);
                }
            }
        } else {
            // Single-predecessor block can just map parameter slots
            storeOperandsOnStack();
            List<Block.Parameter> bargs = ref.targetBlock().parameters();
            for (int i = 0; i < bargs.size(); i++) {
                Value value = sargs.get(i);
                // Map slot of the block argument to slot of the value
                singlePredecessorsValues.put(bargs.get(i), singlePredecessorsValues.getOrDefault(value, value));
            }
        }
    }
}

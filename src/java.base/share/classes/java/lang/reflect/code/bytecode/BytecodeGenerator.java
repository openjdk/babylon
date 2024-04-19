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

import java.lang.classfile.ClassFile;
import java.lang.classfile.CodeBuilder;
import java.lang.classfile.Label;
import java.lang.constant.*;
import java.lang.reflect.code.op.CoreOps.*;

import java.lang.classfile.ClassBuilder;
import java.lang.classfile.Opcode;
import java.lang.classfile.TypeKind;
import java.lang.classfile.attribute.ConstantValueAttribute;
import java.lang.invoke.LambdaMetafactory;
import java.lang.reflect.code.Block;
import java.lang.reflect.code.op.CoreOps;
import java.lang.reflect.code.Op;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.lang.reflect.code.Value;
import java.lang.reflect.code.analysis.Liveness;
import java.lang.reflect.code.type.ArrayType;
import java.lang.reflect.code.type.FieldRef;
import java.lang.reflect.code.type.MethodRef;
import java.lang.reflect.code.type.FunctionType;
import java.lang.reflect.code.type.JavaType;
import java.lang.reflect.code.TypeElement;
import java.lang.reflect.code.type.VarType;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static java.lang.constant.ConstantDescs.*;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.lang.reflect.code.Quotable;
import java.util.stream.Stream;

/**
 * Transformer of code models to bytecode.
 */
public final class BytecodeGenerator {

    private static final DirectMethodHandleDesc DMHD_LAMBDA_METAFACTORY = ofCallsiteBootstrap(
            LambdaMetafactory.class.describeConstable().orElseThrow(),
            "metafactory",
            CD_CallSite, CD_MethodType, CD_MethodHandle, CD_MethodType);

    private static final DirectMethodHandleDesc DMHD_LAMBDA_ALT_METAFACTORY = ofCallsiteBootstrap(
            LambdaMetafactory.class.describeConstable().orElseThrow(),
            "altMetafactory",
            CD_CallSite, CD_Object.arrayType());

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
            hcl = l.defineHiddenClass(classBytes, true);
        } catch (IllegalAccessException e) {
            throw new RuntimeException(e);
        }

        try {
            FunctionType ft = iop.invokableType();
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
        if (!iop.capturedValues().isEmpty()) {
            throw new UnsupportedOperationException("Operation captures values");
        }

        String packageName = lookup.lookupClass().getPackageName();
        ClassDesc className = ClassDesc.of(packageName.isEmpty()
                ? name
                : packageName + "." + name);
        byte[] classBytes = ClassFile.of().build(className, clb -> {
            List<LambdaOp> lambdaSink = new ArrayList<>();
            BitSet quotable = new BitSet();
            generateMethod(lookup, className, name, iop, clb, lambdaSink, quotable);
            for (int i = 0; i < lambdaSink.size(); i++) {
                LambdaOp lop = lambdaSink.get(i);
                if (quotable.get(i)) {
                    clb.withField("lambda$" + i + "$op", CD_String, fb -> fb
                            .withFlags(ClassFile.ACC_STATIC)
                            .with(ConstantValueAttribute.of(quote(lop).toText())));
                }
                generateMethod(lookup, className, "lambda$" + i, lop, clb, lambdaSink, quotable);
            }
        });
        return classBytes;
    }

    private static <O extends Op & Op.Invokable> void generateMethod(MethodHandles.Lookup lookup,
                                                                     ClassDesc className,
                                                                     String methodName,
                                                                     O iop,
                                                                     ClassBuilder clb,
                                                                     List<LambdaOp> lambdaSink,
                                                                     BitSet quotable) {

        List<Value> capturedValues = iop instanceof LambdaOp lop ? lop.capturedValues() : List.of();
        MethodTypeDesc mtd = MethodRef.toNominalDescriptor(
                iop.invokableType()).insertParameterTypes(0, capturedValues.stream()
                        .map(Value::type).map(BytecodeGenerator::toClassDesc).toArray(ClassDesc[]::new));
        clb.withMethodBody(methodName, mtd, ClassFile.ACC_PUBLIC | ClassFile.ACC_STATIC,
                cb -> cb.transforming(new BranchCompactor(), cob ->
                    new BytecodeGenerator(lookup, className, capturedValues, new Liveness(iop),
                                          iop.body().blocks(), cob, lambdaSink, quotable).generate()));
    }

    private record Slot(int slot, TypeKind typeKind) {}
    private record ExceptionRegionWithBlocks(CoreOps.ExceptionRegionEnter ere, BitSet blocks) {}

    private final MethodHandles.Lookup lookup;
    private final ClassDesc className;
    private final List<Value> capturedValues;
    private final List<Block> blocks;
    private final CodeBuilder cob;
    private final Label[] blockLabels;
    private final List<ExceptionRegionWithBlocks> allExceptionRegions;
    private final BitSet[] blocksRegionStack;
    private final BitSet blocksToVisit, catchingBlocks;
    private final Map<Value, Slot> slots;
    private final List<LambdaOp> lambdaSink;
    private final BitSet quotable;
    private Op.Result oprOnStack;

    private BytecodeGenerator(MethodHandles.Lookup lookup,
                              ClassDesc className,
                              List<Value> capturedValues,
                              Liveness liveness,
                              List<Block> blocks,
                              CodeBuilder cob,
                              List<LambdaOp> lambdaSink,
                              BitSet quotable) {
        this.lookup = lookup;
        this.className = className;
        this.capturedValues = capturedValues;
        this.blocks = blocks;
        this.cob = cob;
        this.blockLabels = new Label[blocks.size()];
        this.allExceptionRegions = new ArrayList<>();
        this.blocksRegionStack = new BitSet[blocks.size()];
        this.blocksToVisit = new BitSet(blocks.size());
        this.catchingBlocks = new BitSet();
        this.slots = new HashMap<>();
        this.lambdaSink = lambdaSink;
        this.quotable = quotable;
    }

    private void setExceptionRegionStack(Block.Reference target, BitSet activeRegionStack) {
        setExceptionRegionStack(target.targetBlock().index(), activeRegionStack);
    }

    private void setExceptionRegionStack(int blockIndex, BitSet activeRegionStack) {
        if (blocksRegionStack[blockIndex] == null) {
            blocksToVisit.set(blockIndex);
            blocksRegionStack[blockIndex] = activeRegionStack;
            activeRegionStack.stream().forEach(r -> allExceptionRegions.get(r).blocks.set(blockIndex));
        }
    }

    private Label getLabel(Block.Reference target) {
        return getLabel(target.targetBlock().index());
    }

    private Label getLabel(int blockIndex) {
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
            cob.storeInstruction(slot.typeKind(), slot.slot());
        } else {
            // Only pop results from stack if the value has no further use (no valid slot)
            switch (toTypeKind(v.type()).slotSize()) {
                case 1 -> cob.pop();
                case 2 -> cob.pop2();
            }
        }
    }

    private Slot load(Value v) {
        if (v instanceof Op.Result or &&
                or.op() instanceof CoreOps.ConstantOp constantOp &&
                !constantOp.resultType().equals(JavaType.J_L_CLASS)) {
            cob.constantInstruction(((Constable)constantOp.value()).describeConstable().orElseThrow());
            return null;
        } else {
            Slot slot = slots.get(v);
            cob.loadInstruction(slot.typeKind(), slot.slot());
            return slot;
        }
    }

    private void processFirstOperand(Op op) {
        processOperand(op.operands().getFirst());;
    }

    private void processOperand(Value operand) {
        if (oprOnStack == null) {
            load(operand);
        } else {
            assert oprOnStack == operand;
            oprOnStack = null;
        }
    }

    private void processOperands(Op op) {
        processOperands(op.operands());
    }

    private void processOperands(List<Value> operands) {
        if (oprOnStack == null) {
            operands.forEach(this::load);
        } else {
            assert !operands.isEmpty() && oprOnStack == operands.getFirst();
            oprOnStack = null;
            for (int i = 1; i < operands.size(); i++) {
                load(operands.get(i));
            }
        }
    }

    // Some of the operations can be deferred
    private static boolean canDefer(Op op) {
        return switch (op) {
            case ConstantOp _ ->
                // Loading a class constant may throw an exception so it cannot be deferred
                !op.resultType().equals(JavaType.J_L_CLASS);
            case VarOp _ ->
                // Var with a single-use block parameter operand can be deferred
                op.operands().getFirst() instanceof Block.Parameter bp && bp.uses().size() == 1;
            case VarAccessOp.VarLoadOp _ ->
                // Var load can be deferred when not used as immediate operand
                !isNextUse(op.result());
            default -> false;
        };
    }

    // This method narrows the first operand inconveniences of some operations
    private static boolean isFirstOperand(Op nextOp, Op.Result opr) {
        return switch (nextOp) {
            // When there is no next operation
            case null -> false;
            // New object cannot use first operand from stack, new array fall through to the default
            case NewOp op when !(op.constructorType().returnType() instanceof ArrayType) ->
                false;
            // For lambda the effective operands are captured values
            case LambdaOp op ->
                !op.capturedValues().isEmpty() && op.capturedValues().getFirst() == opr;
            // Conditional branch may delegate to its binary test operation
            case ConditionalBranchOp op when getConditionForCondBrOp(op) instanceof CoreOps.BinaryTestOp bto ->
                isFirstOperand(bto, opr);
            // Var store effective first operand is not the first one
            case VarAccessOp.VarStoreOp op ->
                op.operands().get(1) == opr;
            // regular check of the first operand
            default ->
                !nextOp.operands().isEmpty() && nextOp.operands().getFirst() == opr;
        };
    }

    // Determines if the operation result is immediatelly used by the next operation and so can stay on stack
    private static boolean isNextUse(Op.Result opr) {
        // Pass over deferred operations
        Op nextOp = opr.op();
        do {
            nextOp = opr.declaringBlock().nextOp(nextOp);
        } while (canDefer(nextOp));
        return isFirstOperand(nextOp, opr);
    }

    private static boolean isConditionForCondBrOp(CoreOps.BinaryTestOp op) {
        // Result of op has one use as the operand of a CondBrOp op,
        // and both ops are in the same block

        Set<Op.Result> uses = op.result().uses();
        if (uses.size() != 1) {
            return false;
        }
        Op.Result use = uses.iterator().next();

        if (use.declaringBlock() != op.parentBlock()) {
            return false;
        }

        // Check if used in successor
        for (Block.Reference s : use.op().successors()) {
            if (s.arguments().contains(op.result())) {
                return false;
            }
        }

        return use.op() instanceof CoreOps.ConditionalBranchOp;
    }

    private static ClassDesc toClassDesc(TypeElement t) {
        return switch (t) {
            case VarType vt -> toClassDesc(vt.valueType());
            case JavaType jt -> jt.toNominalDescriptor();
            default ->
                throw new IllegalArgumentException("Bad type: " + t);
        };
    }

    private static TypeKind toTypeKind(TypeElement t) {
        return switch (t) {
            case VarType vt -> toTypeKind(vt.valueType());
            case JavaType jt -> {
                TypeElement bt = jt.toBasicType();
                if (bt.equals(JavaType.VOID)) {
                    yield TypeKind.VoidType;
                } else if (bt.equals(JavaType.INT)) {
                    yield TypeKind.IntType;
                } else if (bt.equals(JavaType.J_L_OBJECT)) {
                    yield TypeKind.ReferenceType;
                } else if (bt.equals(JavaType.LONG)) {
                    yield TypeKind.LongType;
                } else if (bt.equals(JavaType.DOUBLE)) {
                    yield TypeKind.DoubleType;
                } else if (bt.equals(JavaType.BOOLEAN)) {
                    yield TypeKind.BooleanType;
                } else if (bt.equals(JavaType.BYTE)) {
                    yield TypeKind.ByteType;
                } else if (bt.equals(JavaType.CHAR)) {
                    yield TypeKind.CharType;
                } else if (bt.equals(JavaType.FLOAT)) {
                    yield TypeKind.FloatType;
                } else if (bt.equals(JavaType.SHORT)) {
                    yield TypeKind.ShortType;
                } else {
                    throw new IllegalArgumentException("Bad type: " + t);
                }
            }
            default ->
                throw new IllegalArgumentException("Bad type: " + t);
        };
    }

    private void generate() {
        // Compute exception region membership
        setExceptionRegionStack(0, new BitSet());
        int blockIndex;
        while ((blockIndex = blocksToVisit.nextSetBit(0)) >= 0) {
            blocksToVisit.clear(blockIndex);
            BitSet activeRegionStack = blocksRegionStack[blockIndex];
            Block b = blocks.get(blockIndex);
            Op top = b.terminatingOp();
            switch (top) {
                case CoreOps.BranchOp bop ->
                    setExceptionRegionStack(bop.branch(), activeRegionStack);
                case CoreOps.ConditionalBranchOp cop -> {
                    setExceptionRegionStack(cop.falseBranch(), activeRegionStack);
                    setExceptionRegionStack(cop.trueBranch(), activeRegionStack);
                }
                case CoreOps.ExceptionRegionEnter er -> {
                    for (Block.Reference catchBlock : er.catchBlocks().reversed()) {
                        catchingBlocks.set(catchBlock.targetBlock().index());
                        setExceptionRegionStack(catchBlock, activeRegionStack);
                    }
                    activeRegionStack = (BitSet)activeRegionStack.clone();
                    activeRegionStack.set(allExceptionRegions.size());
                    ExceptionRegionWithBlocks newNode = new ExceptionRegionWithBlocks(er, new BitSet());
                    allExceptionRegions.add(newNode);
                    setExceptionRegionStack(er.start(), activeRegionStack);
                }
                case CoreOps.ExceptionRegionExit er -> {
                    activeRegionStack = (BitSet)activeRegionStack.clone();
                    activeRegionStack.clear(activeRegionStack.length() - 1);
                    setExceptionRegionStack(er.end(), activeRegionStack);
                }
                default -> {
                }
            }
        }

        // Declare the exception regions
        for (ExceptionRegionWithBlocks erNode : allExceptionRegions.reversed()) {
            int start  = erNode.blocks.nextSetBit(0);
            while (start >= 0) {
                int end = erNode.blocks.nextClearBit(start);
                Label startLabel = getLabel(start);
                Label endLabel = getLabel(end);
                for (Block.Reference cbr : erNode.ere.catchBlocks()) {
                    List<Block.Parameter> params = cbr.targetBlock().parameters();
                    if (!params.isEmpty()) {
                        JavaType jt = (JavaType) params.get(0).type();
                        cob.exceptionCatch(startLabel, endLabel, getLabel(cbr), jt.toNominalDescriptor());
                    } else {
                        cob.exceptionCatchAll(startLabel, endLabel, getLabel(cbr));
                    }
                }
                start = erNode.blocks.nextSetBit(end);
            }
        }

        // Process blocks in topological order
        // A jump instruction assumes the false successor block is
        // immediately after, in sequence, to the predecessor
        // since the jump instructions branch on a true condition
        // Conditions are inverted when lowered to bytecode
        for (Block b : blocks) {
            // Ignore any non-entry blocks that have no predecessors
            if (!b.isEntryBlock() && b.predecessors().isEmpty()) {
                continue;
            }

            Label blockLabel = getLabel(b.index());
            cob.labelBinding(blockLabel);

            // If b is the entry block then all its parameters conservatively require slots
            // Some unused parameters might be declared before others that are used
            if (b.isEntryBlock()) {
                List<Block.Parameter> parameters = b.parameters();
                int i = 0;
                // Captured values prepend parameters in lambda impl methods
                for (Value cv : capturedValues) {
                    slots.put(cv, new Slot(cob.parameterSlot(i++), toTypeKind(cv.type())));
                }
                for (Block.Parameter bp : parameters) {
                    slots.put(bp, new Slot(cob.parameterSlot(i++), toTypeKind(bp.type())));
                }
            }

            // If b is a catch block then the exception argument will be represented on the stack
            if (catchingBlocks.get(b.index())) {
                // Retain block argument for exception table generation
                storeIfUsed(b.parameters().get(0));
            }

            List<Op> ops = b.ops();
            oprOnStack = null;
            for (int i = 0; i < ops.size() - 1; i++) {
                final Op o = ops.get(i);
                final TypeElement oprType = o.resultType();
                final TypeKind rvt = toTypeKind(oprType);
                switch (o) {
                    case ConstantOp op -> {
                        if (!canDefer(op)) {
                            // Loading a class constant may throw an exception so it cannot be deferred
                            cob.ldc(((JavaType)op.value()).toNominalDescriptor());
                            push(op.result());
                        }
                    }
                    case VarOp op -> {
                        //     %1 : Var<int> = var %0 @"i";
                        if (canDefer(op)) {
                            // Var with a single-use block parameter operand can be deferred
                            slots.put(op.result(), slots.get(op.operands().getFirst()));
                        } else {
                            processOperand(op.operands().getFirst());
                            allocateSlot(op.result());
                            push(op.result());
                        }
                    }
                    case VarAccessOp.VarLoadOp op -> {
                        if (canDefer(op)) {
                            // Var load can be deferred when not used as immediate operand
                            slots.computeIfAbsent(op.result(), r -> slots.get(op.operands().getFirst()));
                        } else {
                            processFirstOperand(op);
                            push(op.result());
                        }
                    }
                    case VarAccessOp.VarStoreOp op -> {
                        processOperand(op.operands().get(1));
                        storeIfUsed(op.operands().get(0));
                    }
                    case ConvOp op -> {
                        Value first = op.operands().getFirst();
                        processOperand(first);
                        TypeKind tk = toTypeKind(first.type());
                        if (tk != rvt) cob.convertInstruction(tk, rvt);
                        push(op.result());
                    }
                    case NegOp op -> {
                        processFirstOperand(op);
                        switch (rvt) { //this can be moved to CodeBuilder::neg(TypeKind)
                            case IntType, BooleanType, ByteType, ShortType, CharType -> cob.ineg();
                            case LongType -> cob.lneg();
                            case FloatType -> cob.fneg();
                            case DoubleType -> cob.dneg();
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
                            case IntType, BooleanType, ByteType, ShortType, CharType -> cob.iadd();
                            case LongType -> cob.ladd();
                            case FloatType -> cob.fadd();
                            case DoubleType -> cob.dadd();
                            default -> throw new IllegalArgumentException("Bad type: " + op.resultType());
                        }
                        push(op.result());
                    }
                    case SubOp op -> {
                        processOperands(op);
                        switch (rvt) { //this can be moved to CodeBuilder::sub(TypeKind)
                            case IntType, BooleanType, ByteType, ShortType, CharType -> cob.isub();
                            case LongType -> cob.lsub();
                            case FloatType -> cob.fsub();
                            case DoubleType -> cob.dsub();
                            default -> throw new IllegalArgumentException("Bad type: " + op.resultType());
                        }
                        push(op.result());
                    }
                    case MulOp op -> {
                        processOperands(op);
                        switch (rvt) { //this can be moved to CodeBuilder::mul(TypeKind)
                            case IntType, BooleanType, ByteType, ShortType, CharType -> cob.imul();
                            case LongType -> cob.lmul();
                            case FloatType -> cob.fmul();
                            case DoubleType -> cob.dmul();
                            default -> throw new IllegalArgumentException("Bad type: " + op.resultType());
                        }
                        push(op.result());
                    }
                    case DivOp op -> {
                        processOperands(op);
                        switch (rvt) { //this can be moved to CodeBuilder::div(TypeKind)
                            case IntType, BooleanType, ByteType, ShortType, CharType -> cob.idiv();
                            case LongType -> cob.ldiv();
                            case FloatType -> cob.fdiv();
                            case DoubleType -> cob.ddiv();
                            default -> throw new IllegalArgumentException("Bad type: " + op.resultType());
                        }
                        push(op.result());
                    }
                    case ModOp op -> {
                        processOperands(op);
                        switch (rvt) { //this can be moved to CodeBuilder::rem(TypeKind)
                            case IntType, BooleanType, ByteType, ShortType, CharType -> cob.irem();
                            case LongType -> cob.lrem();
                            case FloatType -> cob.frem();
                            case DoubleType -> cob.drem();
                            default -> throw new IllegalArgumentException("Bad type: " + op.resultType());
                        }
                        push(op.result());
                    }
                    case AndOp op -> {
                        processOperands(op);
                        switch (rvt) { //this can be moved to CodeBuilder::and(TypeKind)
                            case IntType, BooleanType, ByteType, ShortType, CharType -> cob.iand();
                            case LongType -> cob.land();
                            default -> throw new IllegalArgumentException("Bad type: " + op.resultType());
                        }
                        push(op.result());
                    }
                    case OrOp op -> {
                        processOperands(op);
                        switch (rvt) { //this can be moved to CodeBuilder::or(TypeKind)
                            case IntType, BooleanType, ByteType, ShortType, CharType -> cob.ior();
                            case LongType -> cob.lor();
                            default -> throw new IllegalArgumentException("Bad type: " + op.resultType());
                        }
                        push(op.result());
                    }
                    case XorOp op -> {
                        processOperands(op);
                        switch (rvt) { //this can be moved to CodeBuilder::xor(TypeKind)
                            case IntType, BooleanType, ByteType, ShortType, CharType -> cob.ixor();
                            case LongType -> cob.lxor();
                            default -> throw new IllegalArgumentException("Bad type: " + op.resultType());
                        }
                        push(op.result());
                    }
                    case LshlOp op -> {
                        processOperands(op);
                        adjustRightTypeToInt(op);
                        switch (rvt) { //this can be moved to CodeBuilder::shl(TypeKind)
                            case IntType -> cob.ishl();
                            case LongType -> cob.lshl();
                            default -> throw new IllegalArgumentException("Bad type: " + op.resultType());
                        }
                        push(op.result());
                    }
                    case AshrOp op -> {
                        processOperands(op);
                        adjustRightTypeToInt(op);
                        switch (rvt) { //this can be moved to CodeBuilder::shr(TypeKind)
                            case IntType, ByteType, ShortType, CharType -> cob.ishr();
                            case LongType -> cob.lshr();
                            default -> throw new IllegalArgumentException("Bad type: " + op.resultType());
                        }
                        push(op.result());
                    }
                    case LshrOp op -> {
                        processOperands(op);
                        adjustRightTypeToInt(op);
                        switch (rvt) { //this can be moved to CodeBuilder::ushr(TypeKind)
                            case IntType, ByteType, ShortType, CharType -> cob.iushr();
                            case LongType -> cob.lushr();
                            default -> throw new IllegalArgumentException("Bad type: " + op.resultType());
                        }
                        push(op.result());
                    }
                    case ArrayAccessOp.ArrayLoadOp op -> {
                        processOperands(op);
                        cob.arrayLoadInstruction(rvt);
                        push(op.result());
                    }
                    case ArrayAccessOp.ArrayStoreOp op -> {
                        processOperands(op);
                        cob.arrayStoreInstruction(toTypeKind(op.operands().get(2).type()));
                        push(op.result());
                    }
                    case ArrayLengthOp op -> {
                        processFirstOperand(op);
                        cob.arraylength();
                        push(op.result());
                    }
                    case BinaryTestOp op -> {
                        if (!isConditionForCondBrOp(op)) {
                            processOperands(op);
                            cob.ifThenElse(prepareReverseCondition(op), CodeBuilder::iconst_0, CodeBuilder::iconst_1);
                            push(op.result());
                        }
                        // Processing is deferred to the CondBrOp, do not process the op result
                    }
                    case NewOp op -> {
                        switch (op.constructorType().returnType()) {
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
                                cob.new_(jt.toNominalDescriptor())
                                    .dup();
                                processOperands(op);
                                cob.invokespecial(
                                        ((JavaType) op.resultType()).toNominalDescriptor(),
                                        ConstantDescs.INIT_NAME,
                                        MethodRef.toNominalDescriptor(op.constructorType())
                                                 .changeReturnType(ConstantDescs.CD_void));
                            }
                            default ->
                                throw new IllegalArgumentException("Invalid return type: " + op.constructorType().returnType());
                        }
                        push(op.result());
                    }
                    case InvokeOp op -> {
                        processOperands(op);
                        // @@@ Enhance method descriptor to include how the method is to be invoked
                        // Example result of DirectMethodHandleDesc.toString()
                        //   INTERFACE_VIRTUAL/IntBinaryOperator::applyAsInt(IntBinaryOperator,int,int)int
                        // This will avoid the need to reflectively operate on the descriptor
                        // which may be insufficient in certain cases.
                        DirectMethodHandleDesc.Kind descKind;
                        try {
                            descKind = resolveToMethodHandleDesc(lookup, op.invokeDescriptor()).kind();
                        } catch (ReflectiveOperationException e) {
                            // @@@ Approximate fallback
                            if (op.hasReceiver()) {
                                descKind = DirectMethodHandleDesc.Kind.VIRTUAL;
                            } else {
                                descKind = DirectMethodHandleDesc.Kind.STATIC;
                            }
                        }
                        MethodRef md = op.invokeDescriptor();
                        cob.invokeInstruction(
                                switch (descKind) {
                                    case STATIC, INTERFACE_STATIC   -> Opcode.INVOKESTATIC;
                                    case VIRTUAL                    -> Opcode.INVOKEVIRTUAL;
                                    case INTERFACE_VIRTUAL          -> Opcode.INVOKEINTERFACE;
                                    case SPECIAL, INTERFACE_SPECIAL -> Opcode.INVOKESPECIAL;
                                    default ->
                                        throw new IllegalStateException("Bad method descriptor resolution: "
                                                                        + op.opType() + " > " + op.invokeDescriptor());
                                },
                                ((JavaType) md.refType()).toNominalDescriptor(),
                                md.name(),
                                MethodRef.toNominalDescriptor(md.type()),
                                switch (descKind) {
                                    case INTERFACE_STATIC, INTERFACE_VIRTUAL, INTERFACE_SPECIAL -> true;
                                    default -> false;
                                });

                        push(op.result());
                    }
                    case FieldAccessOp.FieldLoadOp op -> {
                        processOperands(op);
                        FieldRef fd = op.fieldDescriptor();
                        if (op.operands().isEmpty()) {
                            cob.getstatic(
                                    ((JavaType) fd.refType()).toNominalDescriptor(),
                                    fd.name(),
                                    ((JavaType) fd.type()).toNominalDescriptor());
                        } else {
                            cob.getfield(
                                    ((JavaType) fd.refType()).toNominalDescriptor(),
                                    fd.name(),
                                    ((JavaType) fd.type()).toNominalDescriptor());
                        }
                        push(op.result());
                    }
                    case FieldAccessOp.FieldStoreOp op -> {
                        processOperands(op);
                        FieldRef fd = op.fieldDescriptor();
                        if (op.operands().size() == 1) {
                            cob.putstatic(
                                    ((JavaType) fd.refType()).toNominalDescriptor(),
                                    fd.name(),
                                    ((JavaType) fd.type()).toNominalDescriptor());
                        } else {
                            cob.putfield(
                                    ((JavaType) fd.refType()).toNominalDescriptor(),
                                    fd.name(),
                                    ((JavaType) fd.type()).toNominalDescriptor());
                        }
                    }
                    case InstanceOfOp op -> {
                        processFirstOperand(op);
                        cob.instanceof_(((JavaType) op.type()).toNominalDescriptor());
                        push(op.result());
                    }
                    case CastOp op -> {
                        processFirstOperand(op);
                        cob.checkcast(((JavaType) op.type()).toNominalDescriptor());
                        push(op.result());
                    }
                    case LambdaOp op -> {
                        JavaType intfType = (JavaType)op.functionalInterface();
                        MethodTypeDesc mtd = MethodRef.toNominalDescriptor(op.invokableType());
                        try {
                            Class<?> intfClass = intfType.resolve(lookup);
                            processOperands(op.capturedValues());
                            ClassDesc[] captureTypes = op.capturedValues().stream()
                                    .map(Value::type).map(BytecodeGenerator::toClassDesc).toArray(ClassDesc[]::new);
                            int lambdaIndex = lambdaSink.size();
                            if (Quotable.class.isAssignableFrom(intfClass)) {
                                // @@@ double the captured values to enable LambdaMetafactory.FLAG_QUOTABLE
                                for (Value cv : op.capturedValues()) {
                                    load(cv);
                                }
                                cob.invokedynamic(DynamicCallSiteDesc.of(
                                        DMHD_LAMBDA_ALT_METAFACTORY,
                                        funcIntfMethodName(intfClass),
                                        // @@@ double the descriptor parameters
                                        MethodTypeDesc.of(intfType.toNominalDescriptor(),
                                                          Stream.concat(Stream.of(captureTypes),
                                                                        Stream.of(captureTypes)).toList()),
                                        mtd,
                                        MethodHandleDesc.ofMethod(DirectMethodHandleDesc.Kind.STATIC,
                                                                  className,
                                                                  "lambda$" + lambdaIndex,
                                                                  mtd.insertParameterTypes(0, captureTypes)),
                                        mtd,
                                        LambdaMetafactory.FLAG_QUOTABLE,
                                        MethodHandleDesc.ofField(DirectMethodHandleDesc.Kind.STATIC_GETTER,
                                                                 className,
                                                                 "lambda$" + lambdaIndex + "$op",
                                                                 CD_String)));
                                quotable.set(lambdaSink.size());
                            } else {
                                cob.invokedynamic(DynamicCallSiteDesc.of(
                                        DMHD_LAMBDA_METAFACTORY,
                                        funcIntfMethodName(intfClass),
                                        MethodTypeDesc.of(intfType.toNominalDescriptor(), captureTypes),
                                        mtd,
                                        MethodHandleDesc.ofMethod(DirectMethodHandleDesc.Kind.STATIC,
                                                                  className,
                                                                  "lambda$" + lambdaIndex,
                                                                  mtd.insertParameterTypes(0, captureTypes)),
                                        mtd));
                            }
                            lambdaSink.add(op);
                        } catch (ReflectiveOperationException e) {
                            throw new IllegalArgumentException(e);
                        }
                        push(op.result());
                    }
                    default ->
                        throw new UnsupportedOperationException("Unsupported operation: " + ops.get(i));
                }
            }
            Op top = b.terminatingOp();
            switch (top) {
                case CoreOps.ReturnOp op -> {
                    Value a = op.returnValue();
                    if (a == null) {
                        cob.return_();
                    } else {
                        processFirstOperand(op);
                        cob.returnInstruction(toTypeKind(a.type()));
                    }
                }
                case ThrowOp op -> {
                    processFirstOperand(op);
                    cob.athrow();
                }
                case BranchOp op -> {
                    assignBlockArguments(op.branch());
                    cob.goto_(getLabel(op.branch()));
                }
                case ConditionalBranchOp op -> {
                    if (getConditionForCondBrOp(op) instanceof CoreOps.BinaryTestOp btop) {
                        // Processing of the BinaryTestOp was deferred, so it can be merged with CondBrOp
                        processOperands(btop);
                        conditionalBranch(btop, op.trueBranch(), op.falseBranch());
                    } else {
                        processOperands(op);
                        conditionalBranch(Opcode.IFEQ, op, op.trueBranch(), op.falseBranch());
                    }
                }
                case ExceptionRegionEnter op -> {
                    assignBlockArguments(op.start());
                }
                case ExceptionRegionExit op -> {
                    assignBlockArguments(op.end());
                    cob.goto_(getLabel(op.end()));
                }
                default ->
                    throw new UnsupportedOperationException("Terminating operation not supported: " + top);
            }
        }
    }

    private boolean inBlockArgs(Op.Result res) {
        // Check if used in successor
        for (Block.Reference s : res.declaringBlock().successors()) {
            if (s.arguments().contains(res)) {
                return true;
            }
        }
        return false;
    }

    private void push(Op.Result res) {
        assert oprOnStack == null;
        if (res.type().equals(JavaType.VOID)) return;
        if (isNextUse(res)) {
            if (res.uses().size() > 1 || inBlockArgs(res)) {
                switch (toTypeKind(res.type()).slotSize()) {
                    case 1 -> cob.dup();
                    case 2 -> cob.dup2();
                }
                storeIfUsed(res);
            }
            oprOnStack = res;
        } else {
            storeIfUsed(res);
            oprOnStack = null;
        }
    }
    // the rhs of any shift instruction must be int or smaller -> convert longs
    private void adjustRightTypeToInt(Op op) {
        TypeElement right = op.operands().getLast().type();
        if (right.equals(JavaType.LONG)) {
            cob.convertInstruction(toTypeKind(right), TypeKind.IntType);
        }
    }

    private static Op getConditionForCondBrOp(CoreOps.ConditionalBranchOp op) {
        Value p = op.predicate();
        if (p.uses().size() != 1) {
            return null;
        }

        if (p.declaringBlock() != op.parentBlock()) {
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

    private String funcIntfMethodName(Class<?> intfc) {
        String uniqueName = null;
        for (Method m : intfc.getMethods()) {
            // ensure it's SAM interface
            String methodName = m.getName();
            if (Modifier.isAbstract(m.getModifiers())
                    && (m.getReturnType() != String.class
                        || m.getParameterCount() != 0
                        || !methodName.equals("toString"))
                    && (m.getReturnType() != int.class
                        || m.getParameterCount() != 0
                        || !methodName.equals("hashCode"))
                    && (m.getReturnType() != boolean.class
                        || m.getParameterCount() != 1
                        || m.getParameterTypes()[0] != Object.class
                        || !methodName.equals("equals"))) {
                if (uniqueName == null) {
                    uniqueName = methodName;
                } else if (!uniqueName.equals(methodName)) {
                    // too many abstract methods
                    throw new IllegalArgumentException("Not a single-method interface: " + intfc.getName());
                }
            }
        }
        if (uniqueName == null) {
            throw new IllegalArgumentException("No method in: " + intfc.getName());
        }
        return uniqueName;
    }

    private void conditionalBranch(BinaryTestOp op, Block.Reference trueBlock, Block.Reference falseBlock) {
        conditionalBranch(prepareReverseCondition(op), op, trueBlock, falseBlock);
    }

    private void conditionalBranch(Opcode reverseOpcode, Op op, Block.Reference trueBlock, Block.Reference falseBlock) {
        if (!needToAssignBlockArguments(falseBlock)) {
            cob.branchInstruction(reverseOpcode, getLabel(falseBlock));
        } else {
            cob.ifThen(reverseOpcode,
                bb -> {
                    assignBlockArguments(falseBlock);
                    bb.goto_(getLabel(falseBlock));
                });
        }
        assignBlockArguments(trueBlock);
        cob.goto_(getLabel(trueBlock));
    }

    private Opcode prepareReverseCondition(BinaryTestOp op) {
        return switch (toTypeKind(op.operands().get(0).type())) {
            case IntType ->
                switch (op) {
                    case EqOp _ -> Opcode.IF_ICMPNE;
                    case NeqOp _ -> Opcode.IF_ICMPEQ;
                    case GtOp _ -> Opcode.IF_ICMPLE;
                    case GeOp _ -> Opcode.IF_ICMPLT;
                    case LtOp _ -> Opcode.IF_ICMPGE;
                    case LeOp _ -> Opcode.IF_ICMPGT;
                    default ->
                        throw new UnsupportedOperationException(op.opName() + " on int");
                };
            case ReferenceType ->
                switch (op) {
                    case EqOp _ -> Opcode.IF_ACMPNE;
                    case NeqOp _ -> Opcode.IF_ACMPEQ;
                    default ->
                        throw new UnsupportedOperationException(op.opName() + " on Object");
                };
            case FloatType -> {
                cob.fcmpg(); // FCMPL?
                yield reverseIfOpcode(op);
            }
            case LongType -> {
                cob.lcmp();
                yield reverseIfOpcode(op);
            }
            case DoubleType -> {
                cob.dcmpg(); //CMPL?
                yield reverseIfOpcode(op);
            }
            default ->
                throw new UnsupportedOperationException(op.opName() + " on " + op.operands().get(0).type());
        };
    }

    private static Opcode reverseIfOpcode(BinaryTestOp op) {
        return switch (op) {
            case EqOp _ -> Opcode.IFNE;
            case NeqOp _ -> Opcode.IFEQ;
            case GtOp _ -> Opcode.IFLE;
            case GeOp _ -> Opcode.IFLT;
            case LtOp _ -> Opcode.IFGE;
            case LeOp _ -> Opcode.IFGT;
            default ->
                throw new UnsupportedOperationException(op.opName());
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
        List<Value> sargs = ref.arguments();
        List<Block.Parameter> bargs = ref.targetBlock().parameters();
        // First push successor arguments on the stack, then pop and assign
        // so as not to overwrite slots that are reused slots at different argument positions
        for (int i = 0; i < bargs.size(); i++) {
            Block.Parameter barg = bargs.get(i);
            Value value = sargs.get(i);
            if (!barg.uses().isEmpty() && !barg.equals(value)) {
                if (oprOnStack == value) {
                    oprOnStack = null;
                } else {
                    load(value);
                }
                storeIfUsed(barg);
            }
        }
    }

    static DirectMethodHandleDesc resolveToMethodHandleDesc(MethodHandles.Lookup l,
                                                            MethodRef d) throws ReflectiveOperationException {
        MethodHandle mh = d.resolveToHandle(l);

        if (mh.describeConstable().isEmpty()) {
            throw new NoSuchMethodException();
        }

        MethodHandleDesc mhd = mh.describeConstable().get();
        if (!(mhd instanceof DirectMethodHandleDesc dmhd)) {
            throw new NoSuchMethodException();
        }

        return dmhd;
    }

    static CoreOps.FuncOp quote(CoreOps.LambdaOp lop) {
        List<Value> captures = lop.capturedValues();

        // Build the function type
        List<TypeElement> params = captures.stream()
                .map(v -> v.type() instanceof VarType vt ? vt.valueType() : v.type())
                .toList();
        FunctionType ft = FunctionType.functionType(CoreOps.QuotedOp.QUOTED_TYPE, params);

        // Build the function that quotes the lambda
        return CoreOps.func("q", ft).body(b -> {
            // Create variables as needed and obtain the captured values
            // for the copied lambda
            List<Value> outputCaptures = new ArrayList<>();
            for (int i = 0; i < captures.size(); i++) {
                Value c = captures.get(i);
                Block.Parameter p = b.parameters().get(i);
                if (c.type() instanceof VarType _) {
                    Value var = b.op(CoreOps.var(String.valueOf(i), p));
                    outputCaptures.add(var);
                } else {
                    outputCaptures.add(p);
                }
            }

            // Quoted the lambda expression
            Value q = b.op(CoreOps.quoted(b.parentBody(), qb -> {
                // Map the lambda's parent block to the quoted block
                // We are copying lop in the context of the quoted block
                qb.context().mapBlock(lop.parentBlock(), qb);
                // Map the lambda's captured values
                qb.context().mapValues(captures, outputCaptures);
                // Return the lambda to be copied in the quoted operation
                return lop;
            }));
            b.op(CoreOps._return(q));
        });
    }
}

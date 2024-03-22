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

import java.lang.classfile.ClassModel;
import java.lang.classfile.ClassFile;
import java.lang.classfile.CodeBuilder;
import java.lang.classfile.Label;
import java.lang.classfile.components.ClassPrinter;
import java.lang.constant.*;
import java.lang.reflect.code.op.CoreOps.*;

import java.io.File;
import java.io.FileOutputStream;
import java.lang.classfile.Opcode;
import java.lang.classfile.TypeKind;
import java.lang.reflect.code.Block;
import java.lang.reflect.code.Body;
import java.lang.reflect.code.op.CoreOps;
import java.lang.reflect.code.Op;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.lang.reflect.code.Value;
import java.lang.reflect.code.analysis.Liveness;
import java.lang.reflect.code.type.FieldRef;
import java.lang.reflect.code.type.MethodRef;
import java.lang.reflect.code.type.FunctionType;
import java.lang.reflect.code.type.JavaType;
import java.lang.reflect.code.TypeElement;
import java.lang.reflect.code.type.VarType;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.Deque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Transformer of code models to bytecode.
 */
public final class BytecodeGenerator {

    final MethodHandles.Lookup lookup;
    final CodeBuilder cob;
    final Map<Object, Label> labels;
    final Set<Block> catchingBlocks;
    final Map<Value, Slot> slots;

    private BytecodeGenerator(MethodHandles.Lookup lookup, Liveness liveness, CodeBuilder cob) {
        this.lookup = lookup;
        this.cob = cob;
        this.labels = new HashMap<>();
        this.slots = new HashMap<>();
        this.catchingBlocks = new HashSet<>();
    }

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

        {
            print(classBytes);
            try {
                File f = new File("f.class");
                try (FileOutputStream fos = new FileOutputStream(f)) {
                    fos.write(classBytes);
                }
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }

        MethodHandles.Lookup hcl;
        try {
            hcl = l.defineHiddenClass(classBytes, true);
        } catch (IllegalAccessException e) {
            e.printStackTrace();
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

    private static void print(byte[] classBytes) {
        ClassModel cm = ClassFile.of().parse(classBytes);
        ClassPrinter.toYaml(cm, ClassPrinter.Verbosity.CRITICAL_ATTRIBUTES, System.out::print);
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
                                                                         String name, O iop) {
        if (!iop.capturedValues().isEmpty()) {
            throw new UnsupportedOperationException("Operation captures values");
        }

        String packageName = lookup.lookupClass().getPackageName();
        String className = packageName.isEmpty()
                ? name
                : packageName + "." + name;
        Liveness liveness = new Liveness(iop);
        MethodTypeDesc mtd = MethodRef.toNominalDescriptor(iop.invokableType());
        byte[] classBytes = ClassFile.of().build(ClassDesc.of(className), clb ->
                clb.withMethodBody(
                        name,
                        mtd,
                        ClassFile.ACC_PUBLIC | ClassFile.ACC_STATIC,
                        cb -> cb.transforming(new BranchCompactor(), cob ->
                            new BytecodeGenerator(lookup, liveness, cob).generateBody(iop.body()))));
        return classBytes;
    }

    private record Slot(int slot, TypeKind typeKind) {}

    private Label getLabel(Object b) {
        return labels.computeIfAbsent(b, _b -> cob.newLabel());
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
//            System.out.println("Stored " + hash(v) + " in " + slot);
            cob.storeInstruction(slot.typeKind(), slot.slot());
        } else {
//            System.out.println("Popped " + hash(v));
            // Only pop results from stack if the value has no further use (no valid slot)
            switch (toTypeKind(v.type()).slotSize()) {
                case 1 -> cob.pop();
                case 2 -> cob.pop2();
            }
        }
    }

    private static String hash(Value v) {
        return Integer.toHexString(v.hashCode());
    }

    private Slot load(Value v) {
        if (v instanceof Op.Result or &&
                or.op() instanceof CoreOps.ConstantOp constantOp &&
                !constantOp.resultType().equals(JavaType.J_L_CLASS)) {
//            System.out.println("Loaded constant " + hash(v) + " value " + fromValue(constantOp.value()));
            cob.constantInstruction(fromValue(constantOp.value()));
            return null;
        } else {
            Slot slot = slots.get(v);
//            System.out.println("Loaded " + hash(v) + " from " + slot);
            cob.loadInstruction(slot.typeKind(), slot.slot());
            return slot;
        }
    }

    private void processOperands(Op op, boolean isLastOpResultOnStack) {
        for (int i = isLastOpResultOnStack ? 1 : 0; i < op.operands().size(); i++) {
            load(op.operands().get(i));
        }
    }

    private static ConstantDesc fromValue(Object value) {
        return switch (value) {
            case ConstantDesc cd -> cd;
            case JavaType td -> td.toNominalDescriptor();
            default -> throw new IllegalArgumentException("Unsupported constant value: " + value);
        };
    }

    // Determines if the operation result used only by the next operation as the first operand
    private static boolean isResultOnlyUse(Op.Result opr) {
        Set<Op.Result> uses = opr.uses();
        if (uses.size() != 1) {
            return false;
        }

        // Pass over constant operations
        Op.Result use = uses.iterator().next();
        Op nextOp = opr.op();
        do {
            nextOp = opr.declaringBlock().nextOp(nextOp);
        } while (nextOp instanceof CoreOps.ConstantOp);

        if (nextOp == null || use != nextOp.result()) {
            return false;
        }

        // Check if used in successor
        for (Block.Reference s : nextOp.successors()) {
            if (s.arguments().contains(opr)) {
                return false;
            }
        }

        List<Value> operands = nextOp.operands();
        return !operands.isEmpty() && opr == operands.get(0);
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

    private static TypeKind toTypeKind(TypeElement t) {
        return switch (t) {
            case VarType vt -> toTypeKind(vt.valueType());
            case JavaType jt -> {
                TypeElement bt = jt.toBasicType();
                if (bt.equals(JavaType.INT)) {
                    yield TypeKind.IntType;
                } else if (bt.equals(JavaType.LONG)) {
                    yield TypeKind.LongType;
                } else if (bt.equals(JavaType.FLOAT)) {
                    yield TypeKind.FloatType;
                } else if (bt.equals(JavaType.DOUBLE)) {
                    yield TypeKind.DoubleType;
                } else if (bt.equals(JavaType.J_L_OBJECT)) {
                    yield TypeKind.ReferenceType;
                } else {
                    throw new IllegalArgumentException("Bad type: " + t);
                }
            }
            default ->
                throw new IllegalArgumentException("Bad type: " + t);
        };
    }

    private void computeExceptionRegionMembership(Body body) {
        record ExceptionRegionWithBlocks(CoreOps.ExceptionRegionEnter ere, BitSet blocks) {
        }
        // List of all regions
        final List<ExceptionRegionWithBlocks> allRegions = new ArrayList<>();
        class BlockWithActiveExceptionRegions {
            final Block block;
            final BitSet activeRegionStack;
            BlockWithActiveExceptionRegions(Block block, BitSet activeRegionStack) {
                this.block = block;
                this.activeRegionStack = activeRegionStack;
                activeRegionStack.stream().forEach(r -> allRegions.get(r).blocks.set(block.index()));
            }
        }
        final Set<Block> visited = new HashSet<>();
        final Deque<BlockWithActiveExceptionRegions> stack = new ArrayDeque<>();
        stack.push(new BlockWithActiveExceptionRegions(body.entryBlock(), new BitSet()));
        // Compute exception region membership
        while (!stack.isEmpty()) {
            BlockWithActiveExceptionRegions bm = stack.pop();
            Block b = bm.block;
            if (!visited.add(b)) {
                continue;
            }
            Op top = b.terminatingOp();
            switch (top) {
                case CoreOps.BranchOp bop ->
                    stack.push(new BlockWithActiveExceptionRegions(bop.branch().targetBlock(), bm.activeRegionStack));
                case CoreOps.ConditionalBranchOp cop -> {
                    stack.push(new BlockWithActiveExceptionRegions(cop.falseBranch().targetBlock(), bm.activeRegionStack));
                    stack.push(new BlockWithActiveExceptionRegions(cop.trueBranch().targetBlock(), bm.activeRegionStack));
                }
                case CoreOps.ExceptionRegionEnter er -> {
                    for (Block.Reference catchBlock : er.catchBlocks().reversed()) {
                        catchingBlocks.add(catchBlock.targetBlock());
                        stack.push(new BlockWithActiveExceptionRegions(catchBlock.targetBlock(), bm.activeRegionStack));
                    }
                    BitSet activeRegionStack = (BitSet)bm.activeRegionStack.clone();
                    activeRegionStack.set(allRegions.size());
                    ExceptionRegionWithBlocks newNode = new ExceptionRegionWithBlocks(er, new BitSet());
                    allRegions.add(newNode);
                    stack.push(new BlockWithActiveExceptionRegions(er.start().targetBlock(), activeRegionStack));
                }
                case CoreOps.ExceptionRegionExit er -> {
                    BitSet activeRegionStack = (BitSet)bm.activeRegionStack.clone();
                    activeRegionStack.clear(activeRegionStack.length() - 1);
                    stack.push(new BlockWithActiveExceptionRegions(er.end().targetBlock(), activeRegionStack));
                }
                default -> {
                }
            }
        }
        // Declare the exception regions
        final List<Block> blocks = body.blocks();
        for (ExceptionRegionWithBlocks erNode : allRegions.reversed()) {
            int start  = erNode.blocks.nextSetBit(0);
            while (start >= 0) {
                int end = erNode.blocks.nextClearBit(start);
                Label startLabel = getLabel(blocks.get(start));
                Label endLabel = getLabel(blocks.get(end));
                for (Block.Reference cbr : erNode.ere.catchBlocks()) {
                    Block cb = cbr.targetBlock();
                    if (!cb.parameters().isEmpty()) {
                        JavaType jt = (JavaType) cb.parameters().get(0).type();
                        ClassDesc type = jt.toNominalDescriptor();
                        cob.exceptionCatch(startLabel, endLabel, getLabel(cb), type);
                    } else {
                        cob.exceptionCatchAll(startLabel, endLabel, getLabel(cb));
                    }
                }
                start = erNode.blocks.nextSetBit(end);
            }
        }
    }

    private void generateBody(Body body) {
        computeExceptionRegionMembership(body);

        // Process blocks in topological order
        // A jump instruction assumes the false successor block is
        // immediately after, in sequence, to the predecessor
        // since the jump instructions branch on a true condition
        // Conditions are inverted when lowered to bytecode
        List<Block> blocks = body.blocks();
        for (Block b : blocks) {
            // Ignore any non-entry blocks that have no predecessors
            if (body.entryBlock() != b && b.predecessors().isEmpty()) {
                continue;
            }

            Label blockLabel = getLabel(b);
            cob.labelBinding(blockLabel);

            // If b is the entry block then all its parameters conservatively require slots
            // Some unused parameters might be declared before others that are used
            if (b.isEntryBlock()) {
                List<Block.Parameter> parameters = b.parameters();
                for (int i = 0; i < parameters.size(); i++) {
                    Block.Parameter bp = parameters.get(i);
                    slots.put(bp, new Slot(cob.parameterSlot(i), toTypeKind(bp.type())));
                }
            }

            // If b is a catch block then the exception argument will be represented on the stack
            if (catchingBlocks.contains(b)) {
                // Retain block argument for exception table generation
                storeIfUsed(b.parameters().get(0));
            }

            List<Op> ops = b.ops();
            // True if the last result is retained on the stack for use as first operand of current operation
            boolean isLastOpResultOnStack = false;
            Op.Result oprOnStack = null;
            for (int i = 0; i < ops.size() - 1; i++) {
                Op o = ops.get(i);
                TypeElement oprType = o.resultType();
                TypeKind rvt = oprType.equals(JavaType.VOID) ? null : toTypeKind(oprType);
//                System.out.println(o.getClass().getSimpleName() + " result: " + hash(o.result()));
                switch (o) {
                    case ConstantOp op -> {
                        if (op.resultType().equals(JavaType.J_L_CLASS)) {
                            // Loading a class constant may throw an exception so it cannot be deferred
                            cob.ldc(fromValue(op.value()));
                        } else {
                          // Defer process to use, where constants are inlined
                          // This applies to both operands and successor arguments
                          rvt = null;
                        }
                    }
                    case VarOp op -> {
                        //     %1 : Var<int> = var %0 @"i";
                        processOperands(op, isLastOpResultOnStack);
                        isLastOpResultOnStack = false;
                        // Use slot of variable result
                        storeIfUsed(op.result());
                        // Ignore result
                        rvt = null;
                    }
                    case VarAccessOp.VarLoadOp op -> {
                        // Use slot of variable result
                        Slot slot = load(op.operands().get(0));
                        if (slot != null) {
                            slots.putIfAbsent(op.result(), slot);
                        }
                    }
                    case VarAccessOp.VarStoreOp op -> {
                        if (!isLastOpResultOnStack) {
                            load(op.operands().get(1));
                            isLastOpResultOnStack = false;
                        }
                        // Use slot of variable result
                        storeIfUsed(op.operands().get(0));
                    }
                    case ConvOp op -> {
                        processOperands(op, isLastOpResultOnStack);
                        TypeKind tk = toTypeKind(op.operands().get(0).type());
                        if (tk != rvt) cob.convertInstruction(tk, rvt);
                    }
                    case NegOp op -> {
                        processOperands(op, isLastOpResultOnStack);
                        switch (rvt) { //this can be moved to CodeBuilder::neg(TypeKind)
                            case IntType -> cob.ineg();
                            case LongType -> cob.lneg();
                            case FloatType -> cob.fneg();
                            case DoubleType -> cob.dneg();
                            default -> throw new IllegalArgumentException("Bad type: " + op.resultType());
                        }
                    }
                    case NotOp op -> {
                        processOperands(op, isLastOpResultOnStack);
                        cob.ifThenElse(CodeBuilder::iconst_0, CodeBuilder::iconst_1);
                    }
                    case AddOp op -> {
                        processOperands(op, isLastOpResultOnStack);
                        switch (rvt) { //this can be moved to CodeBuilder::add(TypeKind)
                            case IntType -> cob.iadd();
                            case LongType -> cob.ladd();
                            case FloatType -> cob.fadd();
                            case DoubleType -> cob.dadd();
                            default -> throw new IllegalArgumentException("Bad type: " + op.resultType());
                        }
                    }
                    case SubOp op -> {
                        processOperands(op, isLastOpResultOnStack);
                        switch (rvt) { //this can be moved to CodeBuilder::sub(TypeKind)
                            case IntType -> cob.isub();
                            case LongType -> cob.lsub();
                            case FloatType -> cob.fsub();
                            case DoubleType -> cob.dsub();
                            default -> throw new IllegalArgumentException("Bad type: " + op.resultType());
                        }
                    }
                    case MulOp op -> {
                        processOperands(op, isLastOpResultOnStack);
                        switch (rvt) { //this can be moved to CodeBuilder::mul(TypeKind)
                            case IntType -> cob.imul();
                            case LongType -> cob.lmul();
                            case FloatType -> cob.fmul();
                            case DoubleType -> cob.dmul();
                            default -> throw new IllegalArgumentException("Bad type: " + op.resultType());
                        }
                    }
                    case DivOp op -> {
                        processOperands(op, isLastOpResultOnStack);
                        switch (rvt) { //this can be moved to CodeBuilder::div(TypeKind)
                            case IntType -> cob.idiv();
                            case LongType -> cob.ldiv();
                            case FloatType -> cob.fdiv();
                            case DoubleType -> cob.ddiv();
                            default -> throw new IllegalArgumentException("Bad type: " + op.resultType());
                        }
                    }
                    case ModOp op -> {
                        processOperands(op, isLastOpResultOnStack);
                        switch (rvt) { //this can be moved to CodeBuilder::rem(TypeKind)
                            case IntType -> cob.irem();
                            case LongType -> cob.lrem();
                            case FloatType -> cob.frem();
                            case DoubleType -> cob.drem();
                            default -> throw new IllegalArgumentException("Bad type: " + op.resultType());
                        }
                    }
                    case AndOp op -> {
                        processOperands(op, isLastOpResultOnStack);
                        switch (rvt) { //this can be moved to CodeBuilder::and(TypeKind)
                            case IntType, BooleanType -> cob.iand();
                            case LongType -> cob.land();
                            default -> throw new IllegalArgumentException("Bad type: " + op.resultType());
                        }
                    }
                    case OrOp op -> {
                        processOperands(op, isLastOpResultOnStack);
                        switch (rvt) { //this can be moved to CodeBuilder::or(TypeKind)
                            case IntType, BooleanType -> cob.ior();
                            case LongType -> cob.lor();
                            default -> throw new IllegalArgumentException("Bad type: " + op.resultType());
                        }
                    }
                    case XorOp op -> {
                        processOperands(op, isLastOpResultOnStack);
                        switch (rvt) { //this can be moved to CodeBuilder::xor(TypeKind)
                            case IntType, BooleanType -> cob.ixor();
                            case LongType -> cob.lxor();
                            default -> throw new IllegalArgumentException("Bad type: " + op.resultType());
                        }
                    }
                    case ArrayAccessOp.ArrayLoadOp op -> {
                        processOperands(op, isLastOpResultOnStack);
                        cob.arrayLoadInstruction(rvt);
                    }
                    case ArrayAccessOp.ArrayStoreOp op -> {
                        processOperands(op, isLastOpResultOnStack);
                        TypeKind evt = toTypeKind(op.operands().get(2).type());
                        cob.arrayStoreInstruction(evt);
                    }
                    case ArrayLengthOp op -> {
                        processOperands(op, isLastOpResultOnStack);
                        cob.arraylength();
                    }
                    case BinaryTestOp op -> {
                        if (!isConditionForCondBrOp(op)) {
                            processOperands(op, isLastOpResultOnStack);
                            cob.ifThenElse(prepareReverseCondition(op), CodeBuilder::iconst_0, CodeBuilder::iconst_1);
                        } else {
                            // Processing is deferred to the CondBrOp, do not process the op result
                            rvt = null;
                        }
                    }
                    case NewOp op -> {
                        TypeElement t_ = op.constructorType().returnType();
                        JavaType t = (JavaType) t_;
                        switch (t.dimensions()) {
                            case 0 -> {
                                if (isLastOpResultOnStack) {
                                    storeIfUsed(oprOnStack);
                                    isLastOpResultOnStack = false;
                                    oprOnStack = null;
                                }
                                cob.new_(t.toNominalDescriptor())
                                   .dup();
                                processOperands(op, false);
                                cob.invokespecial(
                                        ((JavaType) op.resultType()).toNominalDescriptor(),
                                        ConstantDescs.INIT_NAME,
                                        MethodRef.toNominalDescriptor(op.constructorType())
                                                .changeReturnType(ConstantDescs.CD_void));
                            }
                            case 1 -> {
                                processOperands(op, isLastOpResultOnStack);
                                ClassDesc ctd = t.componentType().toNominalDescriptor();
                                if (ctd.isPrimitive()) {
                                    cob.newarray(TypeKind.from(ctd));
                                } else {
                                    cob.anewarray(ctd);
                                }
                            }
                            default -> {
                                processOperands(op, isLastOpResultOnStack);
                                cob.multianewarray(t.toNominalDescriptor(), op.operands().size());
                            }
                        }
                    }
                    case InvokeOp op -> {
                        processOperands(op, isLastOpResultOnStack);
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
                                        throw new IllegalStateException("Bad method descriptor resolution: " + op.opType() + " > " + op.invokeDescriptor());
                                },
                                ((JavaType) md.refType()).toNominalDescriptor(),
                                md.name(),
                                MethodRef.toNominalDescriptor(md.type()),
                                switch (descKind) {
                                    case INTERFACE_STATIC, INTERFACE_VIRTUAL, INTERFACE_SPECIAL -> true;
                                    default -> false;
                                });

                        if (op.resultType().equals(JavaType.VOID) && !op.operands().isEmpty()) {
                            isLastOpResultOnStack = false;
                        }
                    }
                    case FieldAccessOp.FieldLoadOp op -> {
                        processOperands(op, isLastOpResultOnStack);
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
                    }
                    case FieldAccessOp.FieldStoreOp op -> {
                        processOperands(op, isLastOpResultOnStack);
                        isLastOpResultOnStack = false;
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
                        processOperands(op, isLastOpResultOnStack);
                        cob.instanceof_(((JavaType) op.type()).toNominalDescriptor());
                    }
                    case CastOp op -> {
                        processOperands(op, isLastOpResultOnStack);
                        cob.checkcast(((JavaType) op.type()).toNominalDescriptor());
                    }
                    default ->
                        throw new UnsupportedOperationException("Unsupported operation: " + ops.get(i));
                }
                // Assign slot to operation result
                if (rvt != null) {
                    if (!isResultOnlyUse(o.result())) {
                        isLastOpResultOnStack = false;
                        oprOnStack = null;
                        storeIfUsed(o.result());
                    } else {
                        isLastOpResultOnStack = true;
                        oprOnStack = o.result();
                    }
                }
            }
            Op top = b.terminatingOp();
            switch (top) {
                case CoreOps.ReturnOp op -> {
                    Value a = op.returnValue();
                    if (a == null) {
                        cob.return_();
                    } else {
                        processOperands(op, isLastOpResultOnStack);
                        cob.returnInstruction(toTypeKind(a.type()));
                    }
                }
                case ThrowOp op -> {
                    processOperands(op, isLastOpResultOnStack);
                    cob.athrow();
                }
                case BranchOp op -> {
                    assignBlockArguments(op.branch());
                    cob.goto_(getLabel(op.branch().targetBlock()));
                }
                case ConditionalBranchOp op -> {
                    if (getConditionForCondBrOp(op) instanceof CoreOps.BinaryTestOp btop) {
                        // Processing of the BinaryTestOp was deferred, so it can be merged with CondBrOp
                        processOperands(btop, isLastOpResultOnStack);
                        conditionalBranch(btop, op.trueBranch(), op.falseBranch());
                    } else {
                        processOperands(op, isLastOpResultOnStack);
                        conditionalBranch(Opcode.IFEQ, op, op.trueBranch(), op.falseBranch());
                    }
                }
                case ExceptionRegionEnter op -> {
                    assignBlockArguments(op.start());
                }
                case ExceptionRegionExit op -> {
                    assignBlockArguments(op.end());
                    cob.goto_(getLabel(op.end().targetBlock()));
                }
                default ->
                    throw new UnsupportedOperationException("Terminating operation not supported: " + top);
            }
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

    private void conditionalBranch(BinaryTestOp op, Block.Reference trueBlock, Block.Reference falseBlock) {
        conditionalBranch(prepareReverseCondition(op), op, trueBlock, falseBlock);
    }

    private void conditionalBranch(Opcode reverseOpcode, Op op, Block.Reference trueBlock, Block.Reference falseBlock) {
        if (!needToAssignBlockArguments(falseBlock)) {
            cob.branchInstruction(reverseOpcode, getLabel(falseBlock.targetBlock()));
        } else {
            cob.ifThen(reverseOpcode,
                bb -> {
                    assignBlockArguments(falseBlock);
                    bb.goto_(getLabel(falseBlock.targetBlock()));
                });
        }
        assignBlockArguments(trueBlock);
        cob.goto_(getLabel(trueBlock.targetBlock()));
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
                load(value);
                storeIfUsed(barg);
            }
        }
    }

    static DirectMethodHandleDesc resolveToMethodHandleDesc(MethodHandles.Lookup l, MethodRef d) throws ReflectiveOperationException {
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
}

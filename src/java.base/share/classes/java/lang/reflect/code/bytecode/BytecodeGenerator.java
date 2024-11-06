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

import java.lang.classfile.ClassBuilder;
import java.lang.classfile.ClassFile;
import java.lang.classfile.ClassModel;
import java.lang.classfile.CodeBuilder;
import java.lang.classfile.Label;
import java.lang.classfile.Opcode;
import java.lang.classfile.TypeKind;
import java.lang.classfile.attribute.ConstantValueAttribute;
import java.lang.constant.ClassDesc;
import java.lang.constant.Constable;
import java.lang.constant.ConstantDescs;
import java.lang.constant.DirectMethodHandleDesc;
import java.lang.constant.DynamicCallSiteDesc;
import java.lang.constant.MethodHandleDesc;
import java.lang.constant.MethodTypeDesc;
import java.lang.invoke.LambdaMetafactory;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.lang.invoke.StringConcatFactory;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.lang.reflect.code.Block;
import java.lang.reflect.code.Op;
import java.lang.reflect.code.Quotable;
import java.lang.reflect.code.TypeElement;
import java.lang.reflect.code.Value;
import java.lang.reflect.code.op.CoreOp;
import java.lang.reflect.code.op.CoreOp.*;
import java.lang.reflect.code.type.ArrayType;
import java.lang.reflect.code.type.FieldRef;
import java.lang.reflect.code.type.FunctionType;
import java.lang.reflect.code.type.JavaType;
import java.lang.reflect.code.type.MethodRef;
import java.lang.reflect.code.type.PrimitiveType;
import java.lang.reflect.code.type.VarType;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static java.lang.constant.ConstantDescs.*;

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

    private static final DirectMethodHandleDesc DMHD_STRING_CONCAT = ofCallsiteBootstrap(
            StringConcatFactory.class.describeConstable().orElseThrow(),
            "makeConcat",
            CD_CallSite);

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
            hcl = l.defineHiddenClass(classBytes, true, MethodHandles.Lookup.ClassOption.NESTMATE);
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
        ClassModel generatedModel = ClassFile.of().parse(generateClassData(lookup, fop.funcName(), fop));
        // Compact locals of the generated bytecode
        return ClassFile.of().transformClass(generatedModel, LocalsCompactor.INSTANCE);
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
                    new BytecodeGenerator(lookup, className, capturedValues, TypeKind.from(mtd.returnType()),
                                          iop.body().blocks(), cob, lambdaSink, quotable).generate()));
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
    private final BitSet allCatchBlocks;
    private final Label[] tryStartLabels;
    private final Map<Value, Slot> slots;
    private final Map<Block.Parameter, Value> singlePredecessorsValues;
    private final List<LambdaOp> lambdaSink;
    private final BitSet quotable;
    private final Map<Op, Boolean> deferCache;
    private Value oprOnStack;
    private Block[] recentCatchBlocks;

    private BytecodeGenerator(MethodHandles.Lookup lookup,
                              ClassDesc className,
                              List<Value> capturedValues,
                              TypeKind returnType,
                              List<Block> blocks,
                              CodeBuilder cob,
                              List<LambdaOp> lambdaSink,
                              BitSet quotable) {
        this.lookup = lookup;
        this.className = className;
        this.capturedValues = capturedValues;
        this.returnType = returnType;
        this.blocks = blocks;
        this.cob = cob;
        this.blockLabels = new Label[blocks.size()];
        this.blocksCatchMap = new Block[blocks.size()][];
        this.allCatchBlocks = new BitSet();
        this.tryStartLabels = new Label[blocks.size()];
        this.slots = new IdentityHashMap<>();
        this.singlePredecessorsValues = new IdentityHashMap<>();
        this.lambdaSink = lambdaSink;
        this.quotable = quotable;
        this.deferCache = new IdentityHashMap<>();
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

    private void load(Value v) {
        v = singlePredecessorsValues.getOrDefault(v, v);
        if (v instanceof Op.Result or &&
                or.op() instanceof ConstantOp constantOp &&
                !constantOp.resultType().equals(JavaType.J_L_CLASS)) {
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

    // Constant can be deferred, except for loading of a class constant, which  may throw an exception
    private static boolean canDefer(ConstantOp op) {
        return !op.resultType().equals(JavaType.J_L_CLASS);
    }

    // Single-use var or var with a single-use entry block parameter operand can be deferred
    private static boolean canDefer(VarOp op) {
        return !op.isUninitialized() && (!moreThanOneUse(op.result())
            || op.initOperand() instanceof Block.Parameter bp && bp.declaringBlock().isEntryBlock() && !moreThanOneUse(bp)
            || op.initOperand() instanceof Op.Result or && or.op() instanceof ConstantOp cop && canDefer(cop) && isDefinitelyAssigned(op));
    }

    // Detection if VarOp is definitelly assigned (all its VarLoadOps are dominated by VarStoreOp)
    // VarOp can be deferred in such case
    private static boolean isDefinitelyAssigned(VarOp op) {
        Set<Op.Result> allUses = op.result().uses();
        Set<Op.Result> stores = allUses.stream().filter(r -> r.op() instanceof VarAccessOp.VarStoreOp).collect(Collectors.toSet());
        // All VarLoadOps must be dominated by a VarStoreOp
        for (Op.Result load : allUses) {
            if (load.op() instanceof VarAccessOp.VarLoadOp && !BytecodeUtil.isDominatedBy(load, stores)) {
                return false;
            }
        }
        return true;
    }

    // Var load can be deferred when not used as immediate operand
    private boolean canDefer(VarAccessOp.VarLoadOp op) {
        return !isNextUse(op.result());
    }

    // This method narrows the first operand inconveniences of some operations
    private static boolean isFirstOperand(Op nextOp, Value opr) {
        List<Value> values;
        return switch (nextOp) {
            // When there is no next operation
            case null -> false;
            // New object cannot use first operand from stack, new array fall through to the default
            case NewOp op when !(op.constructorType().returnType() instanceof ArrayType) ->
                false;
            // For lambda the effective operands are captured values
            case LambdaOp op ->
                !(values = op.capturedValues()).isEmpty() && values.getFirst() == opr;
            // Conditional branch may delegate to its binary test operation
            case ConditionalBranchOp op when getConditionForCondBrOp(op) instanceof BinaryTestOp bto ->
                isFirstOperand(bto, opr);
            // Var store effective first operand is not the first one
            case VarAccessOp.VarStoreOp op ->
                op.operands().get(1) == opr;
            // Unconditional branch first target block argument
            case BranchOp op ->
                !(values = op.branch().arguments()).isEmpty() && values.getFirst() == opr;
            // regular check of the first operand
            default ->
                !(values = nextOp.operands()).isEmpty() && values.getFirst() == opr;
        };
    }

    // Determines if the operation result is immediatelly used by the next operation and so can stay on stack
    private boolean isNextUse(Value opr) {
        Op nextOp = switch (opr) {
            case Block.Parameter p -> p.declaringBlock().firstOp();
            case Op.Result r -> r.declaringBlock().nextOp(r.op());
        };
        // Pass over deferred operations
        while (canDefer(nextOp)) {
            nextOp = nextOp.parentBlock().nextOp(nextOp);
        }
        return isFirstOperand(nextOp, opr);
    }

    private static boolean isConditionForCondBrOp(BinaryTestOp op) {
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

        return use.op() instanceof ConditionalBranchOp;
    }

    static ClassDesc toClassDesc(TypeElement t) {
        return switch (t) {
            case VarType vt -> toClassDesc(vt.valueType());
            case JavaType jt -> jt.toNominalDescriptor();
            default ->
                throw new IllegalArgumentException("Bad type: " + t);
        };
    }

    static TypeKind toTypeKind(TypeElement t) {
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

            oprOnStack = null;

            // If b is a catch block then the exception argument will be represented on the stack
            if (allCatchBlocks.get(b.index())) {
                // Retain block argument for exception table generation
                push(b.parameters().getFirst());
            }

            exceptionRegionsChange(catchBlocks);

            List<Op> ops = b.ops();
            for (int i = 0; i < ops.size() - 1; i++) {
                final Op o = ops.get(i);
                final TypeElement oprType = o.resultType();
                final TypeKind rvt = toTypeKind(oprType);
                switch (o) {
                    case ConstantOp op -> {
                        if (!canDefer(op)) {
                            // Constant can be deferred, except for a class constant, which  may throw an exception
                            Object v = op.value();
                            if (v == null) {
                                cob.aconst_null();
                            } else {
                                cob.ldc(((JavaType)v).toNominalDescriptor());
                            }
                            push(op.result());
                        }
                    }
                    case VarOp op -> {
                        //     %1 : Var<int> = var %0 @"i";
                        if (canDefer(op)) {
                            Slot s = slots.get(op.operands().getFirst());
                            if (s != null) {
                                // Var with a single-use entry block parameter can reuse its slot
                                slots.put(op.result(), s);
                            }
                        } else if (!op.isUninitialized()) {
                            processFirstOperand(op);
                            storeIfUsed(op.result());
                        }
                    }
                    case VarAccessOp.VarLoadOp op -> {
                        if (canDefer(op)) {
                            // Var load can be deferred when not used as immediate operand
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
                    case BinaryTestOp op -> {
                        if (!isConditionForCondBrOp(op)) {
                            cob.ifThenElse(prepareConditionalBranch(op), CodeBuilder::iconst_0, CodeBuilder::iconst_1);
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
                                throw new IllegalArgumentException("Invalid return type: "
                                                                    + op.constructorType().returnType());
                        }
                        push(op.result());
                    }
                    case InvokeOp op -> {
                        // @@@ var args
                        processOperands(op);
                        // Resolve referenced class to determine if interface
                        MethodRef md = op.invokeDescriptor();
                        JavaType refType = (JavaType)md.refType();
                        Class<?> refClass;
                        try {
                             refClass = (Class<?>)refType.erasure().resolve(lookup);
                        } catch (ReflectiveOperationException e) {
                            throw new IllegalArgumentException(e);
                        }
                        // Determine invoke opcode
                        final boolean isInterface = refClass.isInterface();
                        if (op.isVarArgs()) {
                            throw new UnsupportedOperationException("invoke varargs unsupported: " + op.invokeDescriptor());
                        }
                        Opcode invokeOpcode = switch (op.invokeKind()) {
                            case STATIC ->
                                    Opcode.INVOKESTATIC;
                            case INSTANCE ->
                                    isInterface ? Opcode.INVOKEINTERFACE : Opcode.INVOKEVIRTUAL;
                            case SUPER ->
                                    // @@@ We cannot generate an invokespecial as it will result in a verify error,
                                    //     since the owner is not assignable to generated hidden class
                                    // @@@ Construct method handle via lookup.findSpecial
                                    //     using the lookup's class as the specialCaller and
                                    //     add that method handle to the to be defined hidden class's constant data
                                    //     Use and ldc+constant dynamic to access the class data,
                                    //     extract the method handle and then invoke it
                                    throw new UnsupportedOperationException("invoke super unsupported: " + op.invokeDescriptor());
                        };
                        MethodTypeDesc mDesc = MethodRef.toNominalDescriptor(md.type());
                        cob.invoke(
                                invokeOpcode,
                                refType.toNominalDescriptor(),
                                md.name(),
                                mDesc,
                                isInterface);
                        ClassDesc ret = toClassDesc(op.resultType());
                        if (ret.isClassOrInterface() && !ret.equals(mDesc.returnType())) {
                            // Explicit cast if method return type differs
                            cob.checkcast(ret);
                        }
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
                        cob.instanceOf(((JavaType) op.type()).toNominalDescriptor());
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
                            Class<?> intfClass = (Class<?>)intfType.erasure().resolve(lookup);
                            processOperands(op.capturedValues());
                            ClassDesc[] captureTypes = op.capturedValues().stream()
                                    .map(Value::type).map(BytecodeGenerator::toClassDesc).toArray(ClassDesc[]::new);
                            int lambdaIndex = lambdaSink.size();
                            if (Quotable.class.isAssignableFrom(intfClass)) {
                                cob.invokedynamic(DynamicCallSiteDesc.of(
                                        DMHD_LAMBDA_ALT_METAFACTORY,
                                        funcIntfMethodName(intfClass),
                                        MethodTypeDesc.of(intfType.toNominalDescriptor(),
                                                          captureTypes),
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

                    if (getConditionForCondBrOp(op) instanceof BinaryTestOp btop) {
                        // Processing of the BinaryTestOp was deferred, so it can be merged with CondBrOp
                        conditionalBranch(prepareConditionalBranch(btop), op.trueBranch(), op.falseBranch());
                    } else {
                        processFirstOperand(op);
                        conditionalBranch(Opcode.IFEQ, op.trueBranch(), op.falseBranch());
                    }
                }
                case ExceptionRegionEnter op -> {
                    List<Block.Reference> enteringCatchBlocks = op.catchBlocks();
                    Block[] activeCatchBlocks = Arrays.copyOf(recentCatchBlocks, recentCatchBlocks.length + enteringCatchBlocks.size());
                    int i = recentCatchBlocks.length;
                    for (Block.Reference catchRef : enteringCatchBlocks) {
                        allCatchBlocks.set(catchRef.targetBlock().index());
                        activeCatchBlocks[i++] = catchRef.targetBlock();
                        setCatchStack(catchRef, recentCatchBlocks);
                    }
                    setCatchStack(op.start(), activeCatchBlocks);

                    assignBlockArguments(op.start());
                    cob.goto_(getLabel(op.start()));
                }
                case ExceptionRegionExit op -> {
                    List<Block.Reference> exitingCatchBlocks = op.catchBlocks();
                    Block[] activeCatchBlocks = Arrays.copyOf(recentCatchBlocks, recentCatchBlocks.length - exitingCatchBlocks.size());
                    setCatchStack(op.end(), activeCatchBlocks);

                    // Assert block exits in reverse order
                    int i = recentCatchBlocks.length;
                    for (Block.Reference catchRef : exitingCatchBlocks) {
                        assert catchRef.targetBlock() == recentCatchBlocks[--i];
                    }

                    assignBlockArguments(op.end());
                    cob.goto_(getLabel(op.end()));
                }
                default ->
                    throw new UnsupportedOperationException("Terminating operation not supported: " + top);
            }
        }
        exceptionRegionsChange(new Block[0]);
    }

    private void exceptionRegionsChange(Block[] newCatchBlocks) {
        if (!Arrays.equals(recentCatchBlocks, newCatchBlocks)) {
            int i = recentCatchBlocks.length - 1;
            Label currentLabel = cob.newBoundLabel();
            // Exit catch blocks missing in the newCatchBlocks
            while (i >=0 && (i >= newCatchBlocks.length || recentCatchBlocks[i] != newCatchBlocks[i])) {
                Block catchBlock = recentCatchBlocks[i--];
                List<Block.Parameter> params = catchBlock.parameters();
                if (!params.isEmpty()) {
                    JavaType jt = (JavaType) params.get(0).type();
                    cob.exceptionCatch(tryStartLabels[catchBlock.index()], currentLabel, getLabel(catchBlock.index()), jt.toNominalDescriptor());
                } else {
                    cob.exceptionCatchAll(tryStartLabels[catchBlock.index()], currentLabel, getLabel(catchBlock.index()));
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

    // Checks if the Op.Result is used more than once in operands and block arguments
    private static boolean moreThanOneUse(Value val) {
        return val.uses().stream().flatMap(u ->
                Stream.concat(
                        u.op().operands().stream(),
                        u.op().successors().stream()
                                .flatMap(r -> r.arguments().stream())))
                .filter(val::equals).limit(2).count() > 1;
    }

    private void push(Value res) {
        assert oprOnStack == null;
        if (res.type().equals(JavaType.VOID)) return;
        if (isNextUse(res)) {
            if (moreThanOneUse(res)) {
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
            cob.conversion(toTypeKind(right), TypeKind.INT);
        }
    }

    private static Op getConditionForCondBrOp(ConditionalBranchOp op) {
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

    private void conditionalBranch(Opcode reverseOpcode, Block.Reference trueBlock, Block.Reference falseBlock) {
        if (!needToAssignBlockArguments(falseBlock)) {
            cob.branch(reverseOpcode, getLabel(falseBlock));
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

    private Opcode prepareConditionalBranch(BinaryTestOp op) {
        Value firstOperand = op.operands().get(0);
        TypeKind typeKind = toTypeKind(firstOperand.type());
        Value secondOperand = op.operands().get(1);
        processOperand(firstOperand);
        if (isZeroIntOrNullConstant(secondOperand)) {
            return switch (typeKind) {
                case INT, BOOLEAN, BYTE, SHORT, CHAR ->
                    switch (op) {
                        case EqOp _ -> Opcode.IFNE;
                        case NeqOp _ -> Opcode.IFEQ;
                        case GtOp _ -> Opcode.IFLE;
                        case GeOp _ -> Opcode.IFLT;
                        case LtOp _ -> Opcode.IFGE;
                        case LeOp _ -> Opcode.IFGT;
                        default ->
                            throw new UnsupportedOperationException(op.opName() + " on int");
                    };
                case REFERENCE ->
                    switch (op) {
                        case EqOp _ -> Opcode.IFNONNULL;
                        case NeqOp _ -> Opcode.IFNULL;
                        default ->
                            throw new UnsupportedOperationException(op.opName() + " on Object");
                    };
                default ->
                    throw new UnsupportedOperationException(op.opName() + " on " + op.operands().get(0).type());
            };
        }
        processOperand(secondOperand);
        return switch (typeKind) {
            case INT, BOOLEAN, BYTE, SHORT, CHAR ->
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
            case REFERENCE ->
                switch (op) {
                    case EqOp _ -> Opcode.IF_ACMPNE;
                    case NeqOp _ -> Opcode.IF_ACMPEQ;
                    default ->
                        throw new UnsupportedOperationException(op.opName() + " on Object");
                };
            case FLOAT -> {
                cob.fcmpg(); // FCMPL?
                yield reverseIfOpcode(op);
            }
            case LONG -> {
                cob.lcmp();
                yield reverseIfOpcode(op);
            }
            case DOUBLE -> {
                cob.dcmpg(); //CMPL?
                yield reverseIfOpcode(op);
            }
            default ->
                throw new UnsupportedOperationException(op.opName() + " on " + op.operands().get(0).type());
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
        Block target = ref.targetBlock();
        List<Value> sargs = ref.arguments();
        if (allCatchBlocks.get(target.index())) {
            // Jumping to an exception handler, exception parameter is expected on stack
            Value value = sargs.getFirst();
            if (oprOnStack == value) {
                oprOnStack = null;
            } else {
                load(value);
            }
        } else if (target.predecessors().size() > 1) {
            List<Block.Parameter> bargs = target.parameters();
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
        } else {
            // Single-predecessor block can just map parameter slots
            List<Block.Parameter> bargs = ref.targetBlock().parameters();
            for (int i = 0; i < bargs.size(); i++) {
                Value value = sargs.get(i);
                if (oprOnStack == value) {
                    storeIfUsed(oprOnStack);
                    oprOnStack = null;
                }
                // Map slot of the block argument to slot of the value
                singlePredecessorsValues.put(bargs.get(i), singlePredecessorsValues.getOrDefault(value, value));
            }
        }
    }

    static FuncOp quote(LambdaOp lop) {
        List<Value> captures = lop.capturedValues();

        // Build the function type
        List<TypeElement> params = captures.stream()
                .map(v -> v.type() instanceof VarType vt ? vt.valueType() : v.type())
                .toList();
        FunctionType ft = FunctionType.functionType(QuotedOp.QUOTED_TYPE, params);

        // Build the function that quotes the lambda
        return CoreOp.func("q", ft).body(b -> {
            // Create variables as needed and obtain the captured values
            // for the copied lambda
            List<Value> outputCaptures = new ArrayList<>();
            for (int i = 0; i < captures.size(); i++) {
                Value c = captures.get(i);
                Block.Parameter p = b.parameters().get(i);
                if (c.type() instanceof VarType _) {
                    Value var = b.op(CoreOp.var(String.valueOf(i), p));
                    outputCaptures.add(var);
                } else {
                    outputCaptures.add(p);
                }
            }

            // Quoted the lambda expression
            Value q = b.op(CoreOp.quoted(b.parentBody(), qb -> {
                // Map the entry block of the lambda's ancestor body to the quoted block
                // We are copying lop in the context of the quoted block, the block mapping
                // ensures the use of captured values are reachable when building
                qb.context().mapBlock(lop.ancestorBody().entryBlock(), qb);
                // Map the lambda's captured values
                qb.context().mapValues(captures, outputCaptures);
                // Return the lambda to be copied in the quoted operation
                return lop;
            }));
            b.op(CoreOp._return(q));
        });
    }
}

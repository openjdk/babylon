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

import jdk.internal.classfile.impl.BytecodeHelpers;

import java.lang.classfile.Attributes;
import java.lang.classfile.ClassFile;
import java.lang.classfile.ClassModel;
import java.lang.classfile.CodeElement;
import java.lang.classfile.CodeModel;
import java.lang.classfile.Instruction;
import java.lang.classfile.Label;
import java.lang.classfile.MethodModel;
import java.lang.classfile.Opcode;
import java.lang.classfile.PseudoInstruction;
import java.lang.classfile.TypeKind;
import java.lang.classfile.attribute.StackMapFrameInfo;
import java.lang.classfile.instruction.*;
import java.lang.constant.ClassDesc;
import java.lang.constant.ConstantDesc;
import java.lang.constant.ConstantDescs;
import java.lang.constant.DirectMethodHandleDesc;
import java.lang.constant.DynamicConstantDesc;
import java.lang.constant.MethodTypeDesc;
import java.lang.invoke.CallSite;
import java.lang.invoke.MethodHandle;
import java.lang.reflect.AccessFlag;
import java.lang.reflect.code.Block;
import java.lang.reflect.code.TypeElement;
import java.lang.reflect.code.op.CoreOp;
import java.lang.reflect.code.Op;
import java.lang.reflect.code.Value;
import java.lang.reflect.code.analysis.NormalizeBlocksTransformer;
import java.lang.reflect.code.type.FieldRef;
import java.lang.reflect.code.type.FunctionType;
import java.lang.reflect.code.type.JavaType;
import java.lang.reflect.code.type.MethodRef;
import java.lang.reflect.code.type.PrimitiveType;
import java.lang.reflect.code.type.VarType;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.Collections;
import java.util.Deque;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import static java.lang.classfile.attribute.StackMapFrameInfo.SimpleVerificationTypeInfo.*;

public final class BytecodeLift {

    private static final ClassDesc CD_LambdaMetafactory = ClassDesc.ofDescriptor("Ljava/lang/invoke/LambdaMetafactory;");
    private static final ClassDesc CD_StringConcatFactory = ClassDesc.ofDescriptor("Ljava/lang/invoke/StringConcatFactory;");
    private static final JavaType MHS_LOOKUP = JavaType.type(ConstantDescs.CD_MethodHandles_Lookup);
    private static final JavaType MH = JavaType.type(ConstantDescs.CD_MethodHandle);
    private static final JavaType MT = JavaType.type(ConstantDescs.CD_MethodType);
    private static final JavaType CLASS_ARRAY = JavaType.array(JavaType.J_L_CLASS);
    private static final MethodRef LCMP = MethodRef.method(JavaType.J_L_LONG, "compare", JavaType.INT, JavaType.LONG, JavaType.LONG);
    private static final MethodRef FCMP = MethodRef.method(JavaType.J_L_FLOAT, "compare", JavaType.INT, JavaType.FLOAT, JavaType.FLOAT);
    private static final MethodRef DCMP = MethodRef.method(JavaType.J_L_DOUBLE, "compare", JavaType.INT, JavaType.DOUBLE, JavaType.DOUBLE);
    private static final MethodRef LOOKUP = MethodRef.method(JavaType.type(ConstantDescs.CD_MethodHandles), "lookup", MHS_LOOKUP);
    private static final MethodRef FIND_STATIC = MethodRef.method(MHS_LOOKUP, "findStatic", MH, JavaType.J_L_CLASS, JavaType.J_L_STRING, MT);
    private static final MethodRef FIND_VIRTUAL = MethodRef.method(MHS_LOOKUP, "findVirtual", MH, JavaType.J_L_CLASS, JavaType.J_L_STRING, MT);
    private static final MethodRef FIND_CONSTRUCTOR = MethodRef.method(MHS_LOOKUP, "findConstructor", MH, JavaType.J_L_CLASS, MT);
    private static final MethodRef FIND_GETTER = MethodRef.method(MHS_LOOKUP, "findGetter", MH, JavaType.J_L_CLASS, JavaType.J_L_STRING, JavaType.J_L_CLASS);
    private static final MethodRef FIND_STATIC_GETTER = MethodRef.method(MHS_LOOKUP, "findStaticGetter", MH, JavaType.J_L_CLASS, JavaType.J_L_STRING, JavaType.J_L_CLASS);
    private static final MethodRef FIND_SETTER = MethodRef.method(MHS_LOOKUP, "findSetter", MH, JavaType.J_L_CLASS, JavaType.J_L_STRING, JavaType.J_L_CLASS);
    private static final MethodRef FIND_STATIC_SETTER = MethodRef.method(MHS_LOOKUP, "findStaticSetter", MH, JavaType.J_L_CLASS, JavaType.J_L_STRING, JavaType.J_L_CLASS);
    private static final MethodRef METHOD_TYPE_0 = MethodRef.method(MT, "methodType", MT, JavaType.J_L_CLASS);
    private static final MethodRef METHOD_TYPE_1 = MethodRef.method(MT, "methodType", MT, JavaType.J_L_CLASS, JavaType.J_L_CLASS);
    private static final MethodRef METHOD_TYPE_L = MethodRef.method(MT, "methodType", MT, JavaType.J_L_CLASS, CLASS_ARRAY);

    private final Block.Builder entryBlock;
    private final ClassModel classModel;
    private final List<Label> exceptionHandlers;
    private final BitSet ereStack;
    private final Map<Label, BitSet> exceptionHandlersMap;
    private final Map<Label, Block.Builder> blockMap;
    private final List<CodeElement> elements;
    private final Deque<Value> stack;
    private final Deque<ClassDesc> newStack;
    private Block.Builder currentBlock;

    private BytecodeLift(Block.Builder entryBlock, ClassModel classModel, CodeModel codeModel, Value... capturedValues) {
        this.entryBlock = entryBlock;
        this.currentBlock = entryBlock;
        this.classModel = classModel;
        this.exceptionHandlers = new ArrayList<>();
        this.ereStack = new BitSet();
        this.newStack = new ArrayDeque<>();
        this.elements = codeModel.elementList();
        this.stack = new ArrayDeque<>();
        this.exceptionHandlersMap = new HashMap<>();
        this.blockMap = codeModel.findAttribute(Attributes.stackMapTable()).map(sma ->
                sma.entries().stream().collect(Collectors.toUnmodifiableMap(
                        StackMapFrameInfo::target,
                        smfi -> entryBlock.block(toBlockParams(smfi.stack()))))).orElseGet(Map::of);
    }

    private List<TypeElement> toBlockParams(List<StackMapFrameInfo.VerificationTypeInfo> vtis) {
        ArrayList<TypeElement> params = new ArrayList<>(vtis.size());
        for (int i = vtis.size() - 1; i >= 0; i--) {
            var vti = vtis.get(i);
            switch (vti) {
                case INTEGER -> params.add(UnresolvedType.unresolvedInt());
                case FLOAT -> params.add(JavaType.FLOAT);
                case DOUBLE -> params.add(JavaType.DOUBLE);
                case LONG -> params.add(JavaType.LONG);
                case NULL -> params.add(UnresolvedType.unresolvedRef());
                case UNINITIALIZED_THIS ->
                    params.add(JavaType.type(classModel.thisClass().asSymbol()));
                case StackMapFrameInfo.ObjectVerificationTypeInfo ovti ->
                    params.add(JavaType.type(ovti.classSymbol()));

                    // Unitialized entry (a new object before its constructor is called)
                    // must be skipped from block parameters because they do not exist in code reflection model
                case StackMapFrameInfo.UninitializedVerificationTypeInfo _ -> {}
                default ->
                    throw new IllegalArgumentException("Unexpected VTI: " + vti);
            }
        }
        return params;
    }

    private Op.Result op(Op op) {
        return currentBlock.op(op);
    }

    // Lift to core dialect
    public static CoreOp.FuncOp lift(byte[] classdata, String methodName) {
        return lift(classdata, methodName, null);
    }

    public static CoreOp.FuncOp lift(byte[] classdata, String methodName, MethodTypeDesc methodType) {
        return lift(ClassFile.of(
                ClassFile.DebugElementsOption.DROP_DEBUG,
                ClassFile.LineNumbersOption.DROP_LINE_NUMBERS).parse(classdata).methods().stream()
                        .filter(mm -> mm.methodName().equalsString(methodName) && (methodType == null || mm.methodTypeSymbol().equals(methodType)))
                        .findFirst().orElseThrow(() -> new IllegalArgumentException("Unknown method: " + methodName)));
    }

    public static CoreOp.FuncOp lift(MethodModel methodModel) {
        ClassModel classModel = methodModel.parent().orElseThrow();
        MethodTypeDesc mDesc = methodModel.methodTypeSymbol();
        if (!methodModel.flags().has(AccessFlag.STATIC)) {
            mDesc = mDesc.insertParameterTypes(0, classModel.thisClass().asSymbol());
        }
        return NormalizeBlocksTransformer.transform(
                UnresolvedTypesTransformer.transform(
                    SlotToVarTransformer.transform(
                        CoreOp.func(methodModel.methodName().stringValue(),
                                    MethodRef.ofNominalDescriptor(mDesc)).body(entryBlock ->
                                            new BytecodeLift(entryBlock,
                                                             classModel,
                                                             methodModel.code().orElseThrow()).liftBody()))));
    }

    private void liftBody() {
        // store entry block
        int slot = 0;
        for (var ep : entryBlock.parameters()) {
            op(SlotOp.store(slot, ep));
            slot += ep.type().equals(JavaType.LONG) || ep.type().equals(JavaType.DOUBLE) ? 2 : 1;
        }

        // fill exceptionHandlers and exceptionHandlersMap
        BitSet eStack = new BitSet();
        var ecs = new ArrayList<ExceptionCatch>();
        for (var e : elements) {
            if (e instanceof ExceptionCatch ec) {
                ecs.add(ec);
                // ecs are squashed by handler
                if (exceptionHandlers.indexOf(ec.handler()) < 0) {
                    exceptionHandlers.add(ec.handler());
                }
            } else if (e instanceof LabelTarget lt) {
                BitSet newEreStack = null;
                for (var er : ecs) {
                    if (lt.label() == er.tryStart() || lt.label() == er.tryEnd()) {
                        if (newEreStack == null) newEreStack = (BitSet)eStack.clone();

                        newEreStack.set(exceptionHandlers.indexOf(er.handler()), lt.label() == er.tryStart());
                    }
                }
                if (newEreStack != null || blockMap.containsKey(lt.label()))  {
                    if (newEreStack != null) eStack = newEreStack;
                    exceptionHandlersMap.put(lt.label(), eStack);
                }
            }
        }

        for (int i = 0; i < elements.size(); i++) {
            switch (elements.get(i)) {
                case ExceptionCatch _ -> {
                    // Exception blocks are inserted by label target (below)
                }
                case LabelTarget lt -> {
                    var newEreStack = exceptionHandlersMap.get(lt.label());
                    if (newEreStack != null) {
                        if (currentBlock != null) {
                            var eresToLeave = (BitSet)ereStack.clone();
                            eresToLeave.andNot(newEreStack);
                            if (!eresToLeave.isEmpty()) {
                                Block.Builder next = entryBlock.block();
                                op(CoreOp.exceptionRegionExit(next.successor(), eresToLeave.stream().mapToObj(ei ->
                                        blockMap.get(exceptionHandlers.get(ei)).successor()).toList()));
                                currentBlock = next;
                            }
                        }
                    }
                    Block.Builder next = blockMap.get(lt.label());
                    if (next != null) {
                        if (currentBlock != null) {
                            op(CoreOp.branch(successorWithStack(next)));
                        }
                        // Stack is reconstructed from block parameters
                        stack.clear();
                        stack.addAll(next.parameters());
                        currentBlock = next;
                    }
                    if (newEreStack != null) {
                        var eresToEnter = (BitSet)newEreStack.clone();
                        eresToEnter.andNot(ereStack);
                        if (!eresToEnter.isEmpty()) {
                            next = entryBlock.block();
                            op(CoreOp.exceptionRegionEnter(next.successor(), eresToEnter.stream().mapToObj(ei ->
                                    blockMap.get(exceptionHandlers.get(ei)).successor()).toList().reversed()));
                            currentBlock = next;
                        }
                        ereStack.clear();
                        ereStack.or(newEreStack);
                    }
                }
                case BranchInstruction inst when BytecodeHelpers.isUnconditionalBranch(inst.opcode()) -> {
                    op(CoreOp.branch(successorWithStack(targetBlockForBranch(inst.target()))));
                    endOfFlow();
                }
                case BranchInstruction inst -> {
                    // Conditional branch
                    Value operand = stack.pop();
                    Op cop = switch (inst.opcode()) {
                        case IFNE -> CoreOp.eq(operand, zero());
                        case IFEQ -> CoreOp.neq(operand, zero());
                        case IFGE -> CoreOp.lt(operand, zero());
                        case IFLE -> CoreOp.gt(operand, zero());
                        case IFGT -> CoreOp.le(operand, zero());
                        case IFLT -> CoreOp.ge(operand, zero());
                        case IFNULL -> CoreOp.neq(operand, liftConstant(null));
                        case IFNONNULL -> CoreOp.eq(operand, liftConstant(null));
                        case IF_ICMPNE -> CoreOp.eq(stack.pop(), operand);
                        case IF_ICMPEQ -> CoreOp.neq(stack.pop(), operand);
                        case IF_ICMPGE -> CoreOp.lt(stack.pop(), operand);
                        case IF_ICMPLE -> CoreOp.gt(stack.pop(), operand);
                        case IF_ICMPGT -> CoreOp.le(stack.pop(), operand);
                        case IF_ICMPLT -> CoreOp.ge(stack.pop(), operand);
                        case IF_ACMPEQ -> CoreOp.neq(stack.pop(), operand);
                        case IF_ACMPNE -> CoreOp.eq(stack.pop(), operand);
                        default -> throw new UnsupportedOperationException("Unsupported branch instruction: " + inst);
                    };
                    Block.Builder branch = targetBlockForBranch(inst.target());
                    Block.Builder next = entryBlock.block();
                    op(CoreOp.conditionalBranch(op(cop),
                            next.successor(),
                            successorWithStack(branch)));
                    currentBlock = next;
                }
                case LookupSwitchInstruction si -> {
                    liftSwitch(si.defaultTarget(), si.cases());
                }
                case TableSwitchInstruction si -> {
                    liftSwitch(si.defaultTarget(), si.cases());
                }
                case ReturnInstruction inst when inst.typeKind() == TypeKind.VOID -> {
                    op(CoreOp._return());
                    endOfFlow();
                }
                case ReturnInstruction _ -> {
                    op(CoreOp._return(stack.pop()));
                    endOfFlow();
                }
                case ThrowInstruction _ -> {
                    op(CoreOp._throw(stack.pop()));
                    endOfFlow();
                }
                case LoadInstruction inst -> {
                    stack.push(op(SlotOp.load(inst.slot(), inst.typeKind())));
                }
                case StoreInstruction inst -> {
                    op(SlotOp.store(inst.slot(), stack.pop()));
                }
                case IncrementInstruction inst -> {
                    op(SlotOp.store(inst.slot(), op(CoreOp.add(op(SlotOp.load(i, TypeKind.INT)), liftConstant(inst.constant())))));
                }
                case ConstantInstruction inst -> {
                    stack.push(liftConstant(inst.constantValue()));
                }
                case ConvertInstruction inst -> {
                    stack.push(op(CoreOp.conv(switch (inst.toType()) {
                        case BYTE -> JavaType.BYTE;
                        case SHORT -> JavaType.SHORT;
                        case INT -> JavaType.INT;
                        case FLOAT -> JavaType.FLOAT;
                        case LONG -> JavaType.LONG;
                        case DOUBLE -> JavaType.DOUBLE;
                        case CHAR -> JavaType.CHAR;
                        case BOOLEAN -> JavaType.BOOLEAN;
                        default ->
                            throw new IllegalArgumentException("Unsupported conversion target: " + inst.toType());
                    }, stack.pop())));
                }
                case OperatorInstruction inst -> {
                    TypeKind tk = inst.typeKind();
                    Value operand = stack.pop();
                    stack.push(op(switch (inst.opcode()) {
                        case IADD, LADD, FADD, DADD ->
                                CoreOp.add(stack.pop(), operand);
                        case ISUB, LSUB, FSUB, DSUB ->
                                CoreOp.sub(stack.pop(), operand);
                        case IMUL, LMUL, FMUL, DMUL ->
                                CoreOp.mul(stack.pop(), operand);
                        case IDIV, LDIV, FDIV, DDIV ->
                                CoreOp.div(stack.pop(), operand);
                        case IREM, LREM, FREM, DREM ->
                                CoreOp.mod(stack.pop(), operand);
                        case INEG, LNEG, FNEG, DNEG ->
                                CoreOp.neg(operand);
                        case ARRAYLENGTH ->
                                CoreOp.arrayLength(operand);
                        case IAND, LAND ->
                                CoreOp.and(stack.pop(), operand);
                        case IOR, LOR ->
                                CoreOp.or(stack.pop(), operand);
                        case IXOR, LXOR ->
                                CoreOp.xor(stack.pop(), operand);
                        case ISHL, LSHL ->
                                CoreOp.lshl(stack.pop(), operand);
                        case ISHR, LSHR ->
                                CoreOp.ashr(stack.pop(), operand);
                        case IUSHR, LUSHR ->
                                CoreOp.lshr(stack.pop(), operand);
                        case LCMP ->
                                CoreOp.invoke(LCMP, stack.pop(), operand);
                        case FCMPL, FCMPG ->
                                CoreOp.invoke(FCMP, stack.pop(), operand);
                        case DCMPL, DCMPG ->
                                CoreOp.invoke(DCMP, stack.pop(), operand);
                        default ->
                            throw new IllegalArgumentException("Unsupported operator opcode: " + inst.opcode());
                    }));
                }
                case FieldInstruction inst -> {
                        FieldRef fd = FieldRef.field(
                                JavaType.type(inst.owner().asSymbol()),
                                inst.name().stringValue(),
                                JavaType.type(inst.typeSymbol()));
                        switch (inst.opcode()) {
                            case GETFIELD ->
                                stack.push(op(CoreOp.fieldLoad(fd, stack.pop())));
                            case GETSTATIC ->
                                stack.push(op(CoreOp.fieldLoad(fd)));
                            case PUTFIELD -> {
                                Value value = stack.pop();
                                op(CoreOp.fieldStore(fd, stack.pop(), value));
                            }
                            case PUTSTATIC ->
                                op(CoreOp.fieldStore(fd, stack.pop()));
                            default ->
                                throw new IllegalArgumentException("Unsupported field opcode: " + inst.opcode());
                        }
                }
                case ArrayStoreInstruction _ -> {
                    Value value = stack.pop();
                    Value index = stack.pop();
                    op(CoreOp.arrayStoreOp(stack.pop(), index, value));
                }
                case ArrayLoadInstruction _ -> {
                    Value index = stack.pop();
                    stack.push(op(CoreOp.arrayLoadOp(stack.pop(), index)));
                }
                case InvokeInstruction inst -> {
                    FunctionType mType = MethodRef.ofNominalDescriptor(inst.typeSymbol());
                    List<Value> operands = new ArrayList<>();
                    for (var _ : mType.parameterTypes()) {
                        operands.add(stack.pop());
                    }
                    MethodRef mDesc = MethodRef.method(
                            JavaType.type(inst.owner().asSymbol()),
                            inst.name().stringValue(),
                            mType);
                    Op.Result result = switch (inst.opcode()) {
                        case INVOKEVIRTUAL, INVOKEINTERFACE -> {
                            operands.add(stack.pop());
                            yield op(CoreOp.invoke(CoreOp.InvokeOp.InvokeKind.INSTANCE, false,
                                    mDesc.type().returnType(), mDesc, operands.reversed()));
                        }
                        case INVOKESTATIC ->
                                op(CoreOp.invoke(CoreOp.InvokeOp.InvokeKind.STATIC, false,
                                        mDesc.type().returnType(), mDesc, operands.reversed()));
                        case INVOKESPECIAL -> {
                            if (inst.owner().asSymbol().equals(newStack.peek()) && inst.name().equalsString(ConstantDescs.INIT_NAME)) {
                                newStack.pop();
                                yield op(CoreOp._new(
                                        FunctionType.functionType(
                                                mDesc.refType(),
                                                mType.parameterTypes()),
                                        operands.reversed()));
                            } else {
                                operands.add(stack.pop());
                                yield op(CoreOp.invoke(CoreOp.InvokeOp.InvokeKind.SUPER, false,
                                        mDesc.type().returnType(), mDesc, operands.reversed()));
                            }
                        }
                        default ->
                            throw new IllegalArgumentException("Unsupported invocation opcode: " + inst.opcode());
                    };
                    if (!result.type().equals(JavaType.VOID)) {
                        stack.push(result);
                    }
                }
                case InvokeDynamicInstruction inst when inst.bootstrapMethod().kind() == DirectMethodHandleDesc.Kind.STATIC -> {
                    DirectMethodHandleDesc bsm = inst.bootstrapMethod();
                    ClassDesc bsmOwner = bsm.owner();
                    if (bsmOwner.equals(CD_LambdaMetafactory)
                        && inst.bootstrapArgs().get(0) instanceof MethodTypeDesc mtd
                        && inst.bootstrapArgs().get(1) instanceof DirectMethodHandleDesc dmhd) {

                        var capturedValues = new Value[dmhd.invocationType().parameterCount() - mtd.parameterCount()];
                        for (int ci = capturedValues.length - 1; ci >= 0; ci--) {
                            capturedValues[ci] = stack.pop();
                        }
                        for (int ci = capturedValues.length; ci < inst.typeSymbol().parameterCount(); ci++) {
                            stack.pop();
                        }
                        MethodTypeDesc mt = dmhd.invocationType();
                        if (capturedValues.length > 0) {
                            mt = mt.dropParameterTypes(0, capturedValues.length);
                        }
                        FunctionType lambdaFunc = FunctionType.functionType(JavaType.type(mt.returnType()),
                                                                            mt.parameterList().stream().map(JavaType::type).toList());
                        CoreOp.LambdaOp.Builder lambda = CoreOp.lambda(currentBlock.parentBody(),
                                                                       lambdaFunc,
                                                                       JavaType.type(inst.typeSymbol().returnType()));
                        if (dmhd.methodName().startsWith("lambda$") && dmhd.owner().equals(classModel.thisClass().asSymbol())) {
                            // inline lambda impl method
                            MethodModel implMethod = classModel.methods().stream().filter(m -> m.methodName().equalsString(dmhd.methodName())).findFirst().orElseThrow();
                            stack.push(op(lambda.body(eb -> new BytecodeLift(eb,
                                                           classModel,
                                                           implMethod.code().orElseThrow(),
                                                           capturedValues).liftBody())));
                        } else {
                            // lambda call to a MH
                            stack.push(op(lambda.body(eb -> {
                                Op.Result ret = eb.op(CoreOp.invoke(
                                        MethodRef.method(JavaType.type(dmhd.owner()),
                                                         dmhd.methodName(),
                                                         lambdaFunc.returnType(),
                                                         lambdaFunc.parameterTypes()),
                                        Stream.concat(Arrays.stream(capturedValues), eb.parameters().stream()).toArray(Value[]::new)));
                                eb.op(ret.type().equals(JavaType.VOID) ? CoreOp._return() : CoreOp._return(ret));
                            })));
                        }
                    } else if (bsmOwner.equals(CD_StringConcatFactory)) {
                        int argsCount = inst.typeSymbol().parameterCount();
                        Deque<Value> args = new ArrayDeque<>(argsCount);
                        for (int ai = 0; ai < argsCount; ai++) {
                            args.push(stack.pop());
                        }
                        Value res = null;
                        if (bsm.methodName().equals("makeConcat")) {
                            for (Value argVal : args) {
                                res = res == null ? argVal : op(CoreOp.concat(res, argVal));
                            }
                        } else {
                            assert bsm.methodName().equals("makeConcatWithConstants");
                            var bsmArgs = inst.bootstrapArgs();
                            String recipe = (String)(bsmArgs.getFirst());
                            int bsmArg = 1;
                            for (int ri = 0; ri < recipe.length(); ri++) {
                                Value argVal = switch (recipe.charAt(ri)) {
                                    case '\u0001' -> args.pop();
                                    case '\u0002' -> liftConstant(bsmArgs.get(bsmArg++));
                                    default -> {
                                        char c;
                                        int start = ri;
                                        while (ri < recipe.length() && (c = recipe.charAt(ri)) != '\u0001' && c != '\u0002') ri++;
                                        yield liftConstant(recipe.substring(start, ri--));
                                    }
                                };
                                res = res == null ? argVal : op(CoreOp.concat(res, argVal));
                            }
                        }
                        if (res != null) stack.push(res);
                    } else {
                        MethodTypeDesc mtd = inst.typeSymbol();

                        //bootstrap
                        MethodTypeDesc bsmDesc = bsm.invocationType();
                        MethodRef bsmRef = MethodRef.method(JavaType.type(bsmOwner),
                                                            bsm.methodName(),
                                                            JavaType.type(bsmDesc.returnType()),
                                                            bsmDesc.parameterList().stream().map(JavaType::type).toArray(TypeElement[]::new));

                        Value[] bootstrapArgs = liftBootstrapArgs(bsmDesc, inst.name().toString(), mtd, inst.bootstrapArgs());
                        Value methodHandle = op(CoreOp.invoke(MethodRef.method(CallSite.class, "dynamicInvoker", MethodHandle.class),
                                                    op(CoreOp.invoke(JavaType.type(ConstantDescs.CD_CallSite), bsmRef, bootstrapArgs))));

                        //invocation
                        List<Value> operands = new ArrayList<>();
                        for (int c = 0; c < mtd.parameterCount(); c++) {
                            operands.add(stack.pop());
                        }
                        operands.add(methodHandle);
                        MethodRef mDesc = MethodRef.method(JavaType.type(ConstantDescs.CD_MethodHandle),
                                                           "invokeExact",
                                                           MethodRef.ofNominalDescriptor(mtd));
                        Op.Result result = op(CoreOp.invoke(mDesc, operands.reversed()));
                        if (!result.type().equals(JavaType.VOID)) {
                            stack.push(result);
                        }
                    }
                }
                case NewObjectInstruction inst -> {
                    // Skip over this and the dup to process the invoke special
                    if (i + 2 < elements.size() - 1
                            && elements.get(i + 1) instanceof StackInstruction dup
                            && dup.opcode() == Opcode.DUP) {
                        i++;
                        newStack.push(inst.className().asSymbol());
                    } else {
                        throw new UnsupportedOperationException("New must be followed by dup");
                    }
                }
                case NewPrimitiveArrayInstruction inst -> {
                    stack.push(op(CoreOp.newArray(
                            switch (inst.typeKind()) {
                                case BOOLEAN -> JavaType.BOOLEAN_ARRAY;
                                case BYTE -> JavaType.BYTE_ARRAY;
                                case CHAR -> JavaType.CHAR_ARRAY;
                                case DOUBLE -> JavaType.DOUBLE_ARRAY;
                                case FLOAT -> JavaType.FLOAT_ARRAY;
                                case INT -> JavaType.INT_ARRAY;
                                case LONG -> JavaType.LONG_ARRAY;
                                case SHORT -> JavaType.SHORT_ARRAY;
                                default ->
                                        throw new UnsupportedOperationException("Unsupported new primitive array type: " + inst.typeKind());
                            },
                            stack.pop())));
                }
                case NewReferenceArrayInstruction inst -> {
                    stack.push(op(CoreOp.newArray(
                            JavaType.type(inst.componentType().asSymbol().arrayType()),
                            stack.pop())));
                }
                case NewMultiArrayInstruction inst -> {
                    stack.push(op(CoreOp._new(
                            FunctionType.functionType(
                                    JavaType.type(inst.arrayType().asSymbol()),
                                    Collections.nCopies(inst.dimensions(), JavaType.INT)),
                            IntStream.range(0, inst.dimensions()).mapToObj(_ -> stack.pop()).toList().reversed())));
                }
                case TypeCheckInstruction inst when inst.opcode() == Opcode.CHECKCAST -> {
                    stack.push(op(CoreOp.cast(JavaType.type(inst.type().asSymbol()), stack.pop())));
                }
                case TypeCheckInstruction inst -> {
                    stack.push(op(CoreOp.instanceOf(JavaType.type(inst.type().asSymbol()), stack.pop())));
                }
                case StackInstruction inst -> {
                    switch (inst.opcode()) {
                        case POP -> {
                            stack.pop();
                        }
                        case POP2 -> {
                            if (isCategory1(stack.pop())) {
                                stack.pop();
                            }
                        }
                        case DUP -> {
                            stack.push(stack.peek());
                        }
                        case DUP_X1 -> {
                            var value1 = stack.pop();
                            var value2 = stack.pop();
                            stack.push(value1);
                            stack.push(value2);
                            stack.push(value1);
                        }
                        case DUP_X2 -> {
                            var value1 = stack.pop();
                            var value2 = stack.pop();
                            if (isCategory1(value2)) {
                                var value3 = stack.pop();
                                stack.push(value1);
                                stack.push(value3);
                            } else {
                                stack.push(value1);
                            }
                            stack.push(value2);
                            stack.push(value1);
                        }
                        case DUP2 -> {
                            var value1 = stack.peek();
                            if (isCategory1(value1)) {
                                stack.pop();
                                var value2 = stack.peek();
                                stack.push(value1);
                                stack.push(value2);
                            }
                            stack.push(value1);
                        }
                        case DUP2_X1 -> {
                            var value1 = stack.pop();
                            var value2 = stack.pop();
                            if (isCategory1(value1)) {
                                var value3 = stack.pop();
                                stack.push(value2);
                                stack.push(value1);
                                stack.push(value3);
                            } else {
                                stack.push(value1);
                            }
                            stack.push(value2);
                            stack.push(value1);
                        }
                        case DUP2_X2 -> {
                            var value1 = stack.pop();
                            var value2 = stack.pop();
                            if (isCategory1(value1)) {
                                var value3 = stack.pop();
                                if (isCategory1(value3)) {
                                    var value4 = stack.pop();
                                    stack.push(value2);
                                    stack.push(value1);
                                    stack.push(value4);
                                } else {
                                    stack.push(value2);
                                    stack.push(value1);
                                }
                                stack.push(value3);
                            } else {
                                if (isCategory1(value2)) {
                                    var value3 = stack.pop();
                                    stack.push(value1);
                                    stack.push(value3);
                                } else {
                                    stack.push(value1);
                                }
                            }
                            stack.push(value2);
                            stack.push(value1);
                        }
                        case SWAP -> {
                            var value1 = stack.pop();
                            var value2 = stack.pop();
                            stack.push(value1);
                            stack.push(value2);
                        }
                        default ->
                            throw new UnsupportedOperationException("Unsupported stack instruction: " + inst);
                    }
                }
                case MonitorInstruction inst -> {
                    var monitor = stack.pop();
                    switch (inst.opcode()) {
                        case MONITORENTER -> op(CoreOp.monitorEnter(monitor));
                        case MONITOREXIT -> op(CoreOp.monitorExit(monitor));
                        default ->
                                throw new UnsupportedOperationException("Unsupported stack instruction: " + inst);
                    }
                }
                case NopInstruction _ -> {}
                case PseudoInstruction _ -> {}
                case Instruction inst ->
                    throw new UnsupportedOperationException("Unsupported instruction: " + inst.opcode().name());
                default ->
                    throw new UnsupportedOperationException("Unsupported code element: " + elements.get(i));
            }
        }
        assert newStack.isEmpty();
    }

    private Op.Result liftConstantsIntoArray(TypeElement arrayType, Object... constants) {
        Op.Result array = op(CoreOp.newArray(arrayType, liftConstant(constants.length)));
        for (int i = 0; i < constants.length; i++) {
            op(CoreOp.arrayStoreOp(array, liftConstant(i), liftConstant(constants[i])));
        }
        return array;
    }

    private Op.Result liftDefaultValue(ClassDesc type) {
        return liftConstant(switch (TypeKind.from(type)) {
            case BOOLEAN -> false;
            case BYTE -> (byte)0;
            case CHAR -> (char)0;
            case DOUBLE -> 0d;
            case FLOAT -> 0f;
            case INT -> 0;
            case LONG -> 0l;
            case REFERENCE -> null;
            case SHORT -> (short)0;
            default -> throw new IllegalStateException("Invalid type " + type.displayName());
        });
    }

    private Op.Result liftConstant(Object c) {
        return switch (c) {
            case null -> op(CoreOp.constant(UnresolvedType.unresolvedRef(), null));
            case ClassDesc cd -> op(CoreOp.constant(JavaType.J_L_CLASS, JavaType.type(cd)));
            case Double d -> op(CoreOp.constant(JavaType.DOUBLE, d));
            case Float f -> op(CoreOp.constant(JavaType.FLOAT, f));
            case Integer ii -> op(CoreOp.constant(UnresolvedType.unresolvedInt(), ii));
            case Long l -> op(CoreOp.constant(JavaType.LONG, l));
            case String s -> op(CoreOp.constant(JavaType.J_L_STRING, s));
            case DirectMethodHandleDesc dmh -> {
                Op.Result lookup = op(CoreOp.invoke(LOOKUP));
                Op.Result owner = liftConstant(dmh.owner());
                Op.Result name = liftConstant(dmh.methodName());
                MethodTypeDesc invDesc = dmh.invocationType();
                yield op(switch (dmh.kind()) {
                    case STATIC, INTERFACE_STATIC  ->
                        CoreOp.invoke(FIND_STATIC, lookup, owner, name, liftConstant(invDesc));
                    case VIRTUAL, INTERFACE_VIRTUAL ->
                        CoreOp.invoke(FIND_VIRTUAL, lookup, owner, name, liftConstant(invDesc.dropParameterTypes(0, 1)));
                    case SPECIAL, INTERFACE_SPECIAL ->
                        //CoreOp.invoke(MethodRef.method(e), "findSpecial", owner, name, liftConstant(invDesc.dropParameterTypes(0, 1)), lookup.lookupClass());
                        throw new UnsupportedOperationException(dmh.toString());
                    case CONSTRUCTOR       ->
                        CoreOp.invoke(FIND_CONSTRUCTOR, lookup, owner, liftConstant(invDesc.changeReturnType(ConstantDescs.CD_Void)));
                    case GETTER            ->
                        CoreOp.invoke(FIND_GETTER, lookup, owner, name, liftConstant(invDesc.returnType()));
                    case STATIC_GETTER     ->
                        CoreOp.invoke(FIND_STATIC_GETTER, lookup, owner, name, liftConstant(invDesc.returnType()));
                    case SETTER            ->
                        CoreOp.invoke(FIND_SETTER, lookup, owner, name, liftConstant(invDesc.parameterType(1)));
                    case STATIC_SETTER     ->
                        CoreOp.invoke(FIND_STATIC_SETTER, lookup, owner, name, liftConstant(invDesc.parameterType(0)));
                });
            }
            case MethodTypeDesc mt -> op(switch (mt.parameterCount()) {
                case 0 -> CoreOp.invoke(METHOD_TYPE_0, liftConstant(mt.returnType()));
                case 1 -> CoreOp.invoke(METHOD_TYPE_1, liftConstant(mt.returnType()), liftConstant(mt.parameterType(0)));
                default -> CoreOp.invoke(METHOD_TYPE_L, liftConstant(mt.returnType()), liftConstantsIntoArray(CLASS_ARRAY, (Object[])mt.parameterArray()));
            });
            case DynamicConstantDesc<?> v when v.bootstrapMethod().owner().equals(ConstantDescs.CD_ConstantBootstraps)
                                         && v.bootstrapMethod().methodName().equals("nullConstant")
                    -> {
                c = null;
                yield liftConstant(null);
            }
            case DynamicConstantDesc<?> dcd -> {
                DirectMethodHandleDesc bsm = dcd.bootstrapMethod();
                MethodTypeDesc bsmDesc = bsm.invocationType();
                Value[] bootstrapArgs = liftBootstrapArgs(bsmDesc, dcd.constantName(), dcd.constantType(), dcd.bootstrapArgsList());
                MethodRef bsmRef = MethodRef.method(JavaType.type(bsm.owner()),
                                                    bsm.methodName(),
                                                    JavaType.type(bsmDesc.returnType()),
                                                    bsmDesc.parameterList().stream().map(JavaType::type).toArray(TypeElement[]::new));
                yield op(CoreOp.invoke(bsmRef, bootstrapArgs));
            }
            case Boolean b -> op(CoreOp.constant(JavaType.BOOLEAN, b));
            case Byte b -> op(CoreOp.constant(JavaType.BYTE, b));
            case Short s -> op(CoreOp.constant(JavaType.SHORT, s));
            case Character ch -> op(CoreOp.constant(JavaType.CHAR, ch));
            default -> throw new UnsupportedOperationException(c.getClass().toString());
        };
    }

    private Value[] liftBootstrapArgs(MethodTypeDesc bsmDesc, String name, ConstantDesc desc, List<ConstantDesc> bsmArgs) {
        Value[] bootstrapArgs = new Value[bsmDesc.parameterCount()];
        bootstrapArgs[0] = op(CoreOp.invoke(LOOKUP));
        bootstrapArgs[1] = liftConstant(name);
        bootstrapArgs[2] = liftConstant(desc);
        ClassDesc lastArgType = bsmDesc.parameterType(bsmDesc.parameterCount() - 1);
        if (lastArgType.isArray()) {
            for (int ai = 0; ai < bootstrapArgs.length - 4; ai++) {
                bootstrapArgs[ai + 3] = liftConstant(bsmArgs.get(ai));
            }
            // Vararg tail of the bootstrap method parameters
            bootstrapArgs[bootstrapArgs.length - 1] =
                    liftConstantsIntoArray(JavaType.type(lastArgType),
                                           bsmArgs.subList(bootstrapArgs.length - 4, bsmArgs.size()).toArray());
        } else {
            for (int ai = 0; ai < bootstrapArgs.length - 3; ai++) {
                bootstrapArgs[ai + 3] = liftConstant(bsmArgs.get(ai));
            }
        }
        return bootstrapArgs;
    }

    private void liftSwitch(Label defaultTarget, List<SwitchCase> cases) {
        Value v = stack.pop();
        if (!valueType(v).equals(PrimitiveType.INT)) {
            v = op(CoreOp.conv(PrimitiveType.INT, v));
        }
        SwitchCase last = cases.getLast();
        Block.Builder def = targetBlockForBranch(defaultTarget);
        for (SwitchCase sc : cases) {
            if (sc == last) {
                op(CoreOp.conditionalBranch(
                        op(CoreOp.eq(v, liftConstant(sc.caseValue()))),
                        successorWithStack(targetBlockForBranch(sc.target())),
                        successorWithStack(def)));
            } else {
                Block.Builder next = entryBlock.block();
                op(CoreOp.conditionalBranch(
                        op(CoreOp.eq(v, liftConstant(sc.caseValue()))),
                        successorWithStack(targetBlockForBranch(sc.target())),
                        next.successor()));
                currentBlock = next;
            }
        }
        endOfFlow();
    }

    private Block.Builder newBlock(List<Block.Parameter> otherBlockParams) {
        return entryBlock.block(otherBlockParams.stream().map(Block.Parameter::type).toList());
    }

    private void endOfFlow() {
        currentBlock = null;
        // Flow discontinued, stack cleared to be ready for the next label target
        stack.clear();
    }

    private Block.Builder targetBlockForBranch(Label targetLabel) {
        Block.Builder targetBlock = blockMap.get(targetLabel);
        var targetEreStack = exceptionHandlersMap.get(targetLabel);
        var eresToEnter = (BitSet)targetEreStack.clone();
        eresToEnter.andNot(ereStack);
        if (!eresToEnter.isEmpty()) {
            // prepend exception region exits
            Block.Builder prev = newBlock(targetBlock.parameters());
            prev.op(CoreOp.exceptionRegionEnter(successorWithStack(targetBlock), eresToEnter.stream().mapToObj(ei ->
                    blockMap.get(exceptionHandlers.get(ei)).successor()).toList().reversed()));
            targetBlock = prev;
        }
        var eresToLeave = (BitSet)ereStack.clone();
        eresToLeave.andNot(targetEreStack);
        if (!eresToLeave.isEmpty()) {
            // prepend exception region enters
            Block.Builder prev = newBlock(targetBlock.parameters());
            prev.op(CoreOp.exceptionRegionExit(successorWithStack(targetBlock), eresToLeave.stream().mapToObj(ei ->
                    blockMap.get(exceptionHandlers.get(ei)).successor()).toList()));
            targetBlock = prev;
        }
        return targetBlock;
    }

    Block.Reference successorWithStack(Block.Builder next) {
        return next.successor(stack.stream().limit(next.parameters().size()).toList());
    }

    private static TypeElement valueType(Value v) {
        var t = v.type();
        while (t instanceof VarType vt) t = vt.valueType();
        return t;
    }

    private Value zero() {
        return op(CoreOp.constant(UnresolvedType.unresolvedInt(), 0));
    }

    private static boolean isCategory1(Value v) {
        return BytecodeGenerator.toTypeKind(v.type()).slotSize() == 1;
    }
}

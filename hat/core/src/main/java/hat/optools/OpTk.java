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
package hat.optools;

import hat.ComputeContext;
import hat.buffer.KernelBufferContext;
import hat.callgraph.CallGraph;
import hat.dialect.HATF16VarOp;
import hat.dialect.HATMemoryOp;
import hat.dialect.HATThreadOp;
import hat.dialect.HATVectorSelectLoadOp;
import hat.dialect.HATVectorAddOp;
import hat.dialect.HATVectorDivOp;
import hat.dialect.HATVectorMulOp;
import hat.dialect.HATVectorSubOp;
import hat.dialect.HATVectorVarOp;
import hat.ifacemapper.MappableIface;
import jdk.incubator.code.Block;
import jdk.incubator.code.CodeElement;
import jdk.incubator.code.CopyContext;
import jdk.incubator.code.Op;
import jdk.incubator.code.OpTransformer;
import jdk.incubator.code.Quoted;
import jdk.incubator.code.TypeElement;
import jdk.incubator.code.Value;
import jdk.incubator.code.analysis.SSA;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.ArrayType;
import jdk.incubator.code.dialect.java.ClassType;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.JavaType;
import jdk.incubator.code.dialect.java.MethodRef;
import jdk.incubator.code.dialect.java.PrimitiveType;

import java.lang.invoke.MethodHandles;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.lang.reflect.Type;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.function.BiFunction;
import java.util.function.Predicate;
import java.util.regex.Pattern;
import java.util.stream.Stream;

public class OpTk {

    public static boolean isKernelContextAccess(JavaOp.FieldAccessOp fieldAccessOp) {
        return fieldAccessOp.fieldDescriptor().refType() instanceof ClassType classType && classType.toClassName().equals("hat.KernelContext");
    }

    public static String fieldName(JavaOp.FieldAccessOp fieldAccessOp) {
        return fieldAccessOp.fieldDescriptor().name();
    }

    public static Object getStaticFinalPrimitiveValue(MethodHandles.Lookup lookup, JavaOp.FieldAccessOp.FieldLoadOp fieldLoadOp) {
        if (fieldLoadOp.fieldDescriptor().refType() instanceof ClassType classType) {
            Class<?> clazz = (Class<?>) classTypeToTypeOrThrow(lookup, classType);
            try {
                Field field = clazz.getField(fieldName(fieldLoadOp));
                field.setAccessible(true);
                return field.get(null);
            } catch (NoSuchFieldException | IllegalAccessException e) {
                throw new RuntimeException(e);
            }
        }
        throw new RuntimeException("Could not find field value" + fieldLoadOp);
    }


    public static Stream<Op> statements(Op.Loop op) {
        var list = new ArrayList<>(statements( op.loopBody().entryBlock()).toList());
        if (list.getLast() instanceof JavaOp.ContinueOp) {
            list.removeLast();
        }
        return list.stream();
    }

    public static Value firstOperand(Op op) {
        return op.operands().getFirst();
    }

    public static Value getValue(Block.Builder bb, Value value) {
        return bb.context().getValueOrDefault(value, value);
    }

    public static boolean isBufferArray( Op op) {
        // first check if the return is an array type
        //if (op instanceof CoreOp.VarOp vop) {
        //    if (!(vop.varValueType() instanceof ArrayType)) return false;
        //} else if (!(op instanceof JavaOp.ArrayAccessOp)){
        //    if (!(op.resultType() instanceof ArrayType)) return false;
        //}

        // then check if returned array is from a buffer access
        while (!(op instanceof JavaOp.InvokeOp iop)) {
            if (!op.operands().isEmpty() && firstOperand(op) instanceof Op.Result r) {
                op = r.op();
            } else {
                return false;
            }
        }

        //if (iop.invokeDescriptor().refType() instanceof JavaType javaType) {
        //    return isAssignable(l, javaType, MappableIface.class);
        //}
        //return false;
        return iop.invokeDescriptor().name().toLowerCase().contains("arrayview");
    }

    public static boolean notGlobalVarOp( Op op) {
        while (!(op instanceof JavaOp.InvokeOp iop)) {
            if (!op.operands().isEmpty() && firstOperand(op) instanceof Op.Result r) {
                op = r.op();
            } else {
                return false;
            }
        }

        return iop.invokeDescriptor().name().toLowerCase().contains("local") ||
                iop.invokeDescriptor().name().toLowerCase().contains("private");
    }

    public static boolean isBufferInitialize( Op op) {
        // first check if the return is an array type
        if (op instanceof CoreOp.VarOp vop) {
            if (!(vop.varValueType() instanceof ArrayType)) return false;
        } else if (!(op instanceof JavaOp.ArrayAccessOp)){
            if (!(op.resultType() instanceof ArrayType)) return false;
        }

        return isBufferArray(op);
    }

    public static boolean isArrayView(MethodHandles.Lookup lookup, CoreOp.FuncOp entry) {
        var here = CallSite.of(OpTk.class,"isArrayView");
        return elements(here,entry).anyMatch((element) -> (
                element instanceof JavaOp.InvokeOp iop &&
                        iop.resultType() instanceof ArrayType &&
                        iop.invokeDescriptor().refType() instanceof JavaType javaType &&
                        isAssignable(lookup, javaType, MappableIface.class)));
    }

    public static CoreOp.ModuleOp createTransitiveInvokeModule(MethodHandles.Lookup lookup,
                                                               CoreOp.FuncOp entry, CallGraph<?> callGraph) {
        LinkedHashSet<MethodRef> funcsVisited = new LinkedHashSet<>();
        List<CoreOp.FuncOp> funcs = new ArrayList<>();
        record RefAndFunc(MethodRef r, CoreOp.FuncOp f) {
        }

        Deque<RefAndFunc> work = new ArrayDeque<>();
        var here = CallSite.of(OpTk.class, "createTransitiveInvokeModule");
        traverse(here, entry, (map, op) -> {
            if (op instanceof JavaOp.InvokeOp invokeOp) {
                Class<?> javaRefTypeClass = javaRefClassOrThrow(callGraph.computeContext.accelerator.lookup, invokeOp);
                try {
                    var method = invokeOp.invokeDescriptor().resolveToMethod(lookup, invokeOp.invokeKind());
                    CoreOp.FuncOp f = Op.ofMethod(method).orElse(null);
                    // TODO filter calls has side effects we may need another call. We might just check the map.

                    if (f != null && !callGraph.filterCalls(f, invokeOp, method, invokeOp.invokeDescriptor(), javaRefTypeClass)) {
                        work.push(new RefAndFunc(invokeOp.invokeDescriptor(),  f));
                    }
                } catch (ReflectiveOperationException _) {
                    throw new IllegalStateException("Could not resolve invokeWrapper to method");
                }
            }
            return map;
        });

        while (!work.isEmpty()) {
            RefAndFunc rf = work.pop();
            if (funcsVisited.add(rf.r)) {
                // TODO:is this really transforming? it seems to be creating a new funcop.. Oh I guess for the new ModuleOp?
                CoreOp.FuncOp tf = rf.f.transform(rf.r.name(), (blockBuilder, op) -> {
                    if (op instanceof JavaOp.InvokeOp iop) {
                        try {
                            Method invokeOpCalledMethod = iop.invokeDescriptor().resolveToMethod(lookup, iop.invokeKind());
                            if (invokeOpCalledMethod instanceof Method m) {
                                CoreOp.FuncOp f = Op.ofMethod(m).orElse(null);
                                if (f != null) {
                                    RefAndFunc call = new RefAndFunc(iop.invokeDescriptor(), f);
                                    work.push(call);
                                    Op.Result result = blockBuilder.op(CoreOp.funcCall(
                                            call.r.name(),
                                            call.f.invokableType(),
                                            blockBuilder.context().getValues(iop.operands())));
                                    blockBuilder.context().mapValue(op.result(), result);
                                    return blockBuilder;
                                }
                            }
                        } catch (ReflectiveOperationException _) {
                            throw new IllegalStateException("Could not resolve invokeWrapper to method");
                        }
                    }
                    blockBuilder.op(op);
                    return blockBuilder;
                });

                funcs.addFirst(tf);
            }
        }

        return CoreOp.module(funcs);
    }

    public static Class<?> primitiveTypeToClass(TypeElement type) {
        assert type != null;
        class PrimitiveHolder {
            static final Map<PrimitiveType, Class<?>> primitiveToClass = Map.of(
                    JavaType.BYTE, byte.class,
                    JavaType.SHORT, short.class,
                    JavaType.INT, int.class,
                    JavaType.LONG, long.class,
                    JavaType.FLOAT, float.class,
                    JavaType.DOUBLE, double.class,
                    JavaType.CHAR, char.class,
                    JavaType.BOOLEAN, boolean.class
            );
        }
        if (type instanceof PrimitiveType primitiveType) {
            return PrimitiveHolder.primitiveToClass.get(primitiveType);
        } else {
            throw new RuntimeException("given type is not a PrimitiveType");
        }
    }

    public static Type classTypeToTypeOrThrow(MethodHandles.Lookup lookup, ClassType classType) {
        try {
            return classType.resolve(lookup);
        } catch (ReflectiveOperationException e) {
            throw new RuntimeException(e);
        }
    }

    public static boolean isAssignable(MethodHandles.Lookup lookup, JavaType javaType, Class<?>... classes) {
        if (javaType instanceof ClassType classType) {
            Type type = classTypeToTypeOrThrow(lookup, classType);
            for (Class<?> clazz : classes) {
                if (clazz.isAssignableFrom((Class<?>) type)) {
                    return true;
                }
            }
        }
        return false;

    }

    public static JavaOp.InvokeOp getQuotableTargetInvokeOpWrapper(JavaOp.LambdaOp lambdaOp) {
        return lambdaOp.body().entryBlock().ops().stream()
                .filter(op -> op instanceof JavaOp.InvokeOp)
                .map(op -> (JavaOp.InvokeOp) op)
                .findFirst().orElseThrow();
    }

    public static Object[] getQuotableCapturedValues(JavaOp.LambdaOp lambdaOp, Quoted quoted, Method method) {
        var block = lambdaOp.body().entryBlock();
        var ops = block.ops();
        Object[] varLoadNames = ops.stream()
                .filter(op -> op instanceof CoreOp.VarAccessOp.VarLoadOp)
                .map(op -> (CoreOp.VarAccessOp.VarLoadOp) op)
                .map(varLoadOp -> (Op.Result) varLoadOp.operands().getFirst())
                .map(varLoadOp -> (CoreOp.VarOp) varLoadOp.op())
                .map(CoreOp.VarOp::varName).toArray();
        Map<String, Object> nameValueMap = new HashMap<>();

        quoted.capturedValues().forEach((k, v) -> {
            if (k instanceof Op.Result result) {
                if (result.op() instanceof CoreOp.VarOp varOp) {
                    nameValueMap.put(varOp.varName(), v);
                }
            }
        });
        Object[] args = new Object[method.getParameterCount()];
        if (args.length != varLoadNames.length) {
            throw new IllegalStateException("Why don't we have enough captures.!! ");
        }
        for (int i = 1; i < args.length; i++) {
            args[i] = nameValueMap.get(varLoadNames[i].toString());
            if (args[i] instanceof CoreOp.Var varbox) {
                args[i] = varbox.value();
            }
        }
        return args;
    }


    // public static Stream<Op> statements(CoreOp.FuncOp op) {
    //   return statements(op.bodies().getFirst().entryBlock());
    // }

    static public Stream<Op> statements(Block block) {
        return block.ops().stream().filter(op->
                (   (op instanceof CoreOp.VarAccessOp.VarStoreOp && op.operands().get(1).uses().size() < 2)
                        || (op instanceof CoreOp.VarOp || op.result().uses().isEmpty())
                        || (op instanceof HATMemoryOp)
                        || (op instanceof HATVectorVarOp)
                        || (op instanceof HATF16VarOp)
                )
                        && !(op instanceof CoreOp.VarOp varOp && paramVar(varOp) != null)
                        && !(op instanceof CoreOp.YieldOp));
    }

    public static JavaType javaRefType(JavaOp.InvokeOp op) {
        return (JavaType) op.invokeDescriptor().refType();
    }

    public static boolean isIfaceBufferMethod(MethodHandles.Lookup lookup, JavaOp.InvokeOp invokeOp) {
        return isAssignable(lookup, javaRefType(invokeOp), MappableIface.class);
    }

    public static boolean isKernelContextMethod(MethodHandles.Lookup lookup, JavaOp.InvokeOp op) {
        return (op.operands().size() > 1 && op.operands().getFirst() instanceof Value value
                && value.type() instanceof JavaType javaType
                && (isAssignable(lookup, javaType, hat.KernelContext.class) || isAssignable(lookup, javaType, KernelBufferContext.class))
        );
    }

    public static boolean isComputeContextMethod(MethodHandles.Lookup lookup, JavaOp.InvokeOp invokeOp) {
        return isAssignable(lookup, javaRefType(invokeOp), ComputeContext.class);
    }

    public static JavaType javaReturnType(JavaOp.InvokeOp invokeOp) {
        return (JavaType) invokeOp.invokeDescriptor().type().returnType();
    }
    public static boolean javaReturnTypeIsVoid(JavaOp.InvokeOp invokeOp) {
        return javaReturnType(invokeOp) instanceof PrimitiveType primitiveType && primitiveType.isVoid();
    }

    public static Method methodOrThrow(MethodHandles.Lookup lookup, JavaOp.InvokeOp op) {
        try {
            return op.invokeDescriptor().resolveToMethod(lookup, op.invokeKind());
        } catch (ReflectiveOperationException e) {
            throw new RuntimeException(e);
        }
    }
/*
    public static Optional<Class<?>> javaReturnClass(MethodHandles.Lookup lookup, JavaOp.InvokeOp op) {
        if (javaReturnType(op) instanceof ClassType classType) {
            return Optional.of((Class<?>) classTypeToTypeOrThrow(lookup, classType));
        } else {
            return Optional.empty();
        }
    }

    public static boolean isIfaceAccessor(MethodHandles.Lookup lookup, JavaOp.InvokeOp invokeOp) {
        if (isIfaceBufferMethod(lookup, invokeOp) && !javaReturnType(invokeOp).equals(JavaType.VOID)) {
            Optional<Class<?>> optionalClazz = javaReturnClass(lookup, invokeOp);
            return optionalClazz.isPresent() && Buffer.class.isAssignableFrom(optionalClazz.get());
        } else {
            return false;
        }
    }
*/

    public static Class<?> javaRefClassOrThrow(MethodHandles.Lookup lookup, JavaOp.InvokeOp op) {
        if (javaRefType(op) instanceof ClassType classType) {
            return (Class<?>) classTypeToTypeOrThrow(lookup, classType);
        } else {
            throw new IllegalStateException(" javaRef class is null");
        }
    }

    /*
       0 =  ()[ ] . -> ++ --
       1 = ++ --+ -! ~ (type) *(deref) &(addressof) sizeof
       2 = * / %
       3 = + -
       4 = << >>
       5 = < <= > >=
       6 = == !=
       7 = &
       8 = ^
       9 = |
       10 = &&
       11 = ||
       12 = ()?:
       13 = += -= *= /= %= &= ^= |= <<= >>=
       14 = ,
    */
    private static int precedenceOf(Op op) {
        return switch (op) {
            case CoreOp.YieldOp o -> 0;
            case JavaOp.InvokeOp o -> 0;
            case CoreOp.FuncCallOp o -> 0;
            case CoreOp.VarOp o -> 13;
            case CoreOp.VarAccessOp.VarStoreOp o -> 13;
            case JavaOp.FieldAccessOp o -> 0;
            case HATThreadOp o -> 0;
            case CoreOp.VarAccessOp.VarLoadOp o -> 0;
            case HATVectorSelectLoadOp o -> 0;      // same as VarLoadOp
            case CoreOp.ConstantOp o -> 0;
            case JavaOp.LambdaOp o -> 0;
            case CoreOp.TupleOp o -> 0;
            case JavaOp.WhileOp o -> 0;
            case JavaOp.ConvOp o -> 1;
            case JavaOp.NegOp  o-> 1;
            case JavaOp.ModOp o -> 2;
            case JavaOp.MulOp o -> 2;
            case HATVectorMulOp o -> 2;
            case JavaOp.DivOp o -> 2;
            case HATVectorDivOp o -> 2;
            case JavaOp.NotOp o -> 2;
            case JavaOp.AddOp o -> 3;
            case HATVectorAddOp o -> 3;
            case HATVectorSubOp o -> 3;
            case JavaOp.SubOp o -> 3;
            case JavaOp.AshrOp o -> 4;
            case JavaOp.LshlOp o -> 4;
            case JavaOp.LshrOp o -> 4;
            case JavaOp.LtOp o -> 5;
            case JavaOp.GtOp o -> 5;
            case JavaOp.LeOp o -> 5;
            case JavaOp.GeOp o -> 5;
            case JavaOp.EqOp o -> 6;
            case JavaOp.NeqOp o -> 6;
            case JavaOp.AndOp o -> 11;
            case JavaOp.XorOp o -> 12;
            case JavaOp.OrOp o -> 13;
            case JavaOp.ConditionalAndOp o -> 14;
            case JavaOp.ConditionalOrOp o -> 15;
            case JavaOp.ConditionalExpressionOp o -> 18;
            case CoreOp.ReturnOp o -> 19;
            default -> throw new IllegalStateException("[Illegal] Precedence Op not registered: " + op.getClass().getSimpleName());
        };
    }
    public static boolean needsParenthesis(Op parent, Op child) {
        return OpTk.precedenceOf(parent) <= OpTk.precedenceOf(child);
    }

    public static Op.Result lhsResult(JavaOp.BinaryOp binaryOp){
        return (Op.Result)binaryOp.operands().get(0);
    }

    public static Op.Result rhsResult(JavaOp.BinaryOp binaryOp){
        return (Op.Result)binaryOp.operands().get(1);
    }

    public static List<Op> ops(JavaOp.JavaConditionalOp javaConditionalOp, int idx){
        return javaConditionalOp.bodies().get(idx).entryBlock().ops();
    }

    public static List<Op> lhsOps(JavaOp.JavaConditionalOp javaConditionalOp){
        return ops(javaConditionalOp,0);
    }

    public static List<Op> rhsOps(JavaOp.JavaConditionalOp javaConditionalOp){
        return ops(javaConditionalOp,1);
    }

    public static Op.Result result(JavaOp.BinaryTestOp binaryTestOp, int idx){
        return (Op.Result)binaryTestOp.operands().get(idx);
    }

    public static Op.Result lhsResult(JavaOp.BinaryTestOp binaryTestOp){
        return result(binaryTestOp,0);
    }

    public static Op.Result rhsResult(JavaOp.BinaryTestOp binaryTestOp){
        return result(binaryTestOp,1);
    }

    public static Op.Result result(JavaOp.ConvOp convOp){
        return (Op.Result)convOp.operands().getFirst();
    }

    public static Op.Result result(CoreOp.ReturnOp returnOp){
        return (Op.Result)returnOp.operands().getFirst();
    }

    public static Block block(JavaOp.ConditionalExpressionOp ternaryOp, int idx){
        return ternaryOp.bodies().get(idx).entryBlock();
    }

    public static Block condBlock(JavaOp.ConditionalExpressionOp ternaryOp){
        return block(ternaryOp,0);
    }

    public static Block thenBlock(JavaOp.ConditionalExpressionOp ternaryOp){
        return block(ternaryOp,1);
    }

    public static Block elseBlock(JavaOp.ConditionalExpressionOp ternaryOp){
        return block(ternaryOp,2);
    }

    public static String funcName(JavaOp.InvokeOp invokeOp) {
        return invokeOp.invokeDescriptor().name();
    }

    public static Value operandOrNull(Op op, int idx) {
        return op.operands().size() > idx?op.operands().get(idx):null;
    }

    public static Op.Result resultOrNull(Op op, int idx) {
        return (operandOrNull(op,idx) instanceof Op.Result result)?result:null;
    }

    public static Block block(JavaOp.ForOp forOp, int idx){
        return forOp.bodies().get(idx).entryBlock();
    }

    public static Block mutateBlock(JavaOp.ForOp forOp){
        return block(forOp,2);
    }

    public static Block loopBlock(JavaOp.ForOp forOp){
        return block(forOp,3);
    }

    public static Block condBlock(JavaOp.ForOp forOp){
        return  forOp.cond().entryBlock();
    }

    public static Block initBlock(JavaOp.ForOp forOp){
        return  forOp.init().entryBlock();
    }

    public static Block block(JavaOp.WhileOp whileOp, int idx){
        return  whileOp.bodies().get(idx).entryBlock();
    }

    public static Block condBlock(JavaOp.WhileOp whileOp){
        return  block(whileOp,0);
    }

    public static Block loopBlock(JavaOp.WhileOp whileOp){
        return  block(whileOp,1);
    }

    public static Block blockOrNull(JavaOp.IfOp ifOp, int idx ){
        return ifOp.bodies().size() > idx?ifOp.bodies().get(idx).entryBlock():null;
    }

    public static boolean fieldNameIs(JavaOp.FieldAccessOp.FieldAccessOp fieldAccessOp, String name) {
        return fieldName(fieldAccessOp).equals(name);
    }
    public static boolean fieldNameMatches(JavaOp.FieldAccessOp.FieldAccessOp fieldAccessOp, Pattern pattern) {
        return pattern.matcher(fieldName(fieldAccessOp)).matches();
    }

    public  record CallSite(Class<?> clazz,String methodName){
        public static CallSite of(Class<?> clazz, String methodName) {
            boolean TRACE = Boolean.getBoolean("TRACE_CALLSITES");

                //System.out.println("TRACE_CALLSITES "+TRACE);

            return TRACE?new CallSite(clazz,methodName):null;
        }

        @Override public  String toString(){
            return clazz.toString()+":"+methodName;
        }
    }
    public static <T> T traverse(CallSite callSite, CoreOp.FuncOp funcOp, BiFunction<T, CodeElement<?,?>,T> bifunc) {
        if (callSite!= null){
            System.out.println(callSite + " traverse is being deprecated!!");
        }
       return  funcOp.traverse(null, bifunc);
    }
    public static CoreOp.FuncOp lower(CallSite callSite, CoreOp.FuncOp funcOp) {
        if (callSite!= null){
            System.out.println(callSite);
        }
        return funcOp.transform(OpTransformer.LOWERING_TRANSFORMER);
    }
    public static Stream<CodeElement<?,?>> elements(CallSite callSite, CoreOp.FuncOp funcOp) {
        if (callSite!= null){
            System.out.println(callSite);
        }
        return funcOp.elements();
    }

    public static CoreOp.FuncOp SSATransformLower(CallSite callSite, CoreOp.FuncOp funcOp){
        if (callSite!= null){
            System.out.println(callSite);
        }
        return  SSA.transform(lower(callSite,funcOp));
    }
    public static CoreOp.FuncOp SSATransform(CallSite callSite, CoreOp.FuncOp funcOp){
        if (callSite!= null){
            System.out.println(callSite);
        }
        return  SSA.transform(funcOp);
    }

    public static CoreOp.FuncOp transform(CallSite callSite, CoreOp.FuncOp funcOp, Predicate<Op> predicate, OpTransformer opTransformer) {
        if (callSite!= null){
            System.out.println(callSite);
        }
        return funcOp.transform((blockBuilder, op) -> {
            if (predicate.test(op)){
                var builder = opTransformer.acceptOp(blockBuilder,op);
                if (builder != blockBuilder){
                    throw new RuntimeException("Where does this builder come from "+builder);
                }
            }else {
                blockBuilder.op(op);
            }
            return blockBuilder;
        });
    }

    public static CoreOp.FuncOp transform(CallSite callSite, CoreOp.FuncOp funcOp, OpTransformer opTransformer) {
        if (callSite!= null){
            System.out.println(callSite);
        }
        return funcOp.transform(opTransformer);
    }

    public record  OpMap(CoreOp.FuncOp fromFuncOp, CoreOp.FuncOp toFuncOp,  Map<Op,Op> fromToOpMap){}

    public  static <InOp extends Op, OutOp extends Op> OutOp replaceOp(Block.Builder blockBuilder, InOp inOp,java.util.function.Function<List<Value>, OutOp> factory) {
        List<Value> inputOperands = inOp.operands();
        CopyContext context = blockBuilder.context();
        List<Value> outputOperands = context.getValues(inputOperands);
        OutOp outOp = factory.apply(outputOperands);
        Op.Result outputResult = blockBuilder.op(outOp);
        Op.Result inputResult = inOp.result();
        outOp.setLocation(inOp.location());
        context.mapValue(inputResult, outputResult);
        return outOp;
    }
    public static < OutOp extends Op> OpMap simpleOpMappingTransform(OpTk.CallSite here, CoreOp.FuncOp fromFuncOp, Predicate<Op> opPredicate,
                                                         java.util.function.Function<List<Value>, OutOp> opFactory){
        Map<Op,Op> fromToOpMap = new LinkedHashMap<>();
        CoreOp.FuncOp toFuncOp =  OpTk.transform(here, fromFuncOp, (blockBuilder, inOp) -> {
            if (opPredicate.test(inOp)) {
                fromToOpMap.put(inOp, replaceOp(blockBuilder, inOp, opFactory));
            }else {
                var r = blockBuilder.op(inOp);
                fromToOpMap.put(inOp,r.op());
            }
            return blockBuilder;
        });
        return new OpMap(fromFuncOp, toFuncOp, fromToOpMap);
    }



    public record ParamVar(CoreOp.VarOp varOp, Block.Parameter parameter, CoreOp.FuncOp funcOp) {
    }

    public static ParamVar paramVar(CoreOp.VarOp varOp) {
        return !varOp.isUninitialized()
                && varOp.operands().getFirst() instanceof Block.Parameter parameter
                && parameter.invokableOperation() instanceof CoreOp.FuncOp funcOp ? new ParamVar(varOp, parameter, funcOp) : null;
    }

    public static boolean returnIsVoid(JavaOp.InvokeOp invokeOp){
        return javaReturnType(invokeOp) instanceof PrimitiveType primitiveType && primitiveType.isVoid();
    }
}

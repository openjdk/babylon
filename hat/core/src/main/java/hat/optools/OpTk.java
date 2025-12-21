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
import hat.KernelContext;
import hat.buffer.HAType;
import hat.callgraph.CallGraph;
import hat.device.DeviceType;
import hat.dialect.*;
import optkl.LookupCarrier;
import optkl.ifacemapper.MappableIface;
import hat.types._V;
import jdk.incubator.code.Block;
import jdk.incubator.code.CodeContext;
import jdk.incubator.code.CodeTransformer;
import jdk.incubator.code.CodeElement;
import jdk.incubator.code.Op;
import jdk.incubator.code.Quoted;
import jdk.incubator.code.TypeElement;
import jdk.incubator.code.Value;
import jdk.incubator.code.analysis.SSA;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.ClassType;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.JavaType;
import jdk.incubator.code.dialect.java.MethodRef;
import jdk.incubator.code.dialect.java.PrimitiveType;
import optkl.ParamVar;

import java.lang.invoke.MethodHandles;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.lang.reflect.Type;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Deque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.function.Function;
import java.util.function.Predicate;
import java.util.regex.Pattern;
import java.util.stream.Stream;

public interface OpTk extends LookupCarrier  {
    Predicate<JavaOp.FieldAccessOp> AnyFieldAccess = _->true;

    static OpTk impl(LookupCarrier lookupCarrier){
        record Impl(MethodHandles.Lookup lookup) implements LookupCarrier,OpTk{}
        return new Impl(lookupCarrier.lookup());
    }


   static boolean isKernelContext(MethodHandles.Lookup lookup,TypeElement typeElement){
       return isAssignable(lookup,typeElement,KernelContext.class);
   }
    default boolean isKernelContext(TypeElement typeElement){
        return isAssignable(lookup(),typeElement,KernelContext.class);
    }

    Predicate<JavaOp.InvokeOp> AnyInvoke = _->true;
    static JavaOp.InvokeOp asKernelContextInvokeOpOrNull(MethodHandles.Lookup lookup, CodeElement<?,?> ce, Predicate<JavaOp.InvokeOp> predicate) {
        if (ce instanceof JavaOp.InvokeOp invokeOp) {
            if (isKernelContext(lookup, invokeOp.invokeDescriptor().refType())) {
                return predicate.test(invokeOp) ? invokeOp : null;
            } else if (invokeOp.operands().size() > 1
                    && invokeOp.operands().getFirst() instanceof Value value
                    && isKernelContext(lookup, value.type())) {
            //    throw new IllegalStateException("did you mean to check if the first arg is KernelContext ?");
            }
        }
        return null;
    }

    static boolean isKernelContextInvokeOp(MethodHandles.Lookup lookup, CodeElement<?,?> ce, Predicate<JavaOp.InvokeOp> predicate) {
        return Objects.nonNull(asKernelContextInvokeOpOrNull(lookup,ce, predicate));
    }


    static boolean isVarAccessFromKernelContextFieldOp(MethodHandles.Lookup lookup,CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
        return isKernelContextFieldAccessOp(lookup, varLoadOp, AnyFieldAccess);//varLoadOp.resultType());
    }
    static JavaOp.FieldAccessOp asKernelContextFieldAccessOrNull(MethodHandles.Lookup lookup, CodeElement<?,?> ce, Predicate<JavaOp.FieldAccessOp> predicate) {
        if (ce instanceof JavaOp.FieldAccessOp fieldAccessOp && isKernelContext(lookup,fieldAccessOp.fieldDescriptor().refType())){
            return predicate.test(fieldAccessOp)?fieldAccessOp:null;
        }
        return null;
    }
    static JavaOp.FieldAccessOp asNamedKernelContextFieldAccessOrNull(MethodHandles.Lookup lookup, CodeElement<?,?> ce, String name) {
        return asKernelContextFieldAccessOrNull(lookup,ce,fieldAccessOp->name.equals(fieldAccessOp.fieldDescriptor().name()));
    }
    static JavaOp.FieldAccessOp asNamedKernelContextFieldAccessOrNull(MethodHandles.Lookup lookup, CodeElement<?,?> ce, Pattern pattern) {
        return asKernelContextFieldAccessOrNull(lookup,ce,fieldAccessOp->pattern.matcher(fieldAccessOp.fieldDescriptor().name()).matches());
    }
    default JavaOp.FieldAccessOp asNamedKernelContextFieldAccessOrNull( CodeElement<?,?> ce, Pattern pattern) {
        return asKernelContextFieldAccessOrNull(lookup(),ce,fieldAccessOp->pattern.matcher(fieldAccessOp.fieldDescriptor().name()).matches());
    }
    static boolean isKernelContextFieldAccessOp(MethodHandles.Lookup lookup,CodeElement<?, ?> ce, Predicate<JavaOp.FieldAccessOp> predicate) {
        return Objects.nonNull(asKernelContextFieldAccessOrNull(lookup,ce, predicate));
    }

    static boolean isKernelContextFieldAccessOp(MethodHandles.Lookup lookup,CodeElement<?, ?> ce) {
        return isKernelContextFieldAccessOp(lookup,ce, AnyFieldAccess);
    }


    static boolean isIfaceBufferMethod(MethodHandles.Lookup lookup, JavaOp.InvokeOp invokeOp) {
        return (isAssignable(lookup, javaRefType(invokeOp), MappableIface.class));
    }

    static boolean isHatType(MethodHandles.Lookup lookup, JavaOp.InvokeOp invokeOp) {
        return (isAssignableTo(lookup, javaRefType(invokeOp), DeviceType.class, MappableIface.class, HAType.class));
    }



    static boolean isComputeContextMethod(MethodHandles.Lookup lookup, JavaOp.InvokeOp invokeOp) {
        return isAssignable(lookup, javaRefType(invokeOp), ComputeContext.class);
    }



    static <F extends Op, T extends Op> T copyLocation(F from, T to ){
        to.setLocation(from.location());
        return to;
    }


    static String fieldName(JavaOp.FieldAccessOp fieldAccessOp) {
        return fieldAccessOp.fieldDescriptor().name();
    }

    static Object getStaticFinalPrimitiveValue(MethodHandles.Lookup lookup, JavaOp.FieldAccessOp.FieldLoadOp fieldLoadOp) {
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


    static Stream<Op> statements(Op.Loop op) {
        var list = new ArrayList<>(statements( op.loopBody().entryBlock()).toList());
        if (list.getLast() instanceof JavaOp.ContinueOp) {
            list.removeLast();
        }
        return list.stream();
    }

    static CoreOp.ModuleOp createTransitiveInvokeModule(MethodHandles.Lookup lookup,
                                                               CoreOp.FuncOp entry, CallGraph<?> callGraph) {
        LinkedHashSet<MethodRef> funcsVisited = new LinkedHashSet<>();
        List<CoreOp.FuncOp> funcs = new ArrayList<>();
        record RefAndFunc(MethodRef r, CoreOp.FuncOp f) {}

        Deque<RefAndFunc> work = new ArrayDeque<>();
        var here = CallSite.of(OpTk.class, "createTransitiveInvokeModule");
        elements(here, entry).forEach(codeElement -> {
            if (codeElement instanceof JavaOp.InvokeOp invokeOp) {
                Class<?> javaRefTypeClass = javaRefClassOrThrow(callGraph.computeContext.accelerator.lookup(), invokeOp);
                try {
                    var method = invokeOp.invokeDescriptor().resolveToMethod(lookup);
                    CoreOp.FuncOp f = Op.ofMethod(method).orElse(null);
                    // TODO filter calls has side effects we may need another call. We might just check the map.

                    if (f != null && !callGraph.filterCalls(f, invokeOp, method, invokeOp.invokeDescriptor(), javaRefTypeClass)) {
                        work.push(new RefAndFunc(invokeOp.invokeDescriptor(),  f));
                    }
                } catch (ReflectiveOperationException _) {
                    throw new IllegalStateException("Could not resolve invokeWrapper to method");
                }
            }
        });

        while (!work.isEmpty()) {
            RefAndFunc rf = work.pop();
            if (funcsVisited.add(rf.r)) {
                // TODO:is this really transforming? it seems to be creating a new funcop.. Oh I guess for the new ModuleOp?
                CoreOp.FuncOp tf = rf.f.transform(rf.r.name(), (blockBuilder, op) -> {
                    if (op instanceof JavaOp.InvokeOp iop) {
                        try {
                            Method invokeOpCalledMethod = iop.invokeDescriptor().resolveToMethod(lookup);
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

    static Type classTypeToTypeOrThrow(MethodHandles.Lookup lookup, ClassType classType) {
        try {
            return classType.resolve(lookup);
        } catch (ReflectiveOperationException e) {
            throw new RuntimeException(e);
        }
    }

    static boolean isAssignable(MethodHandles.Lookup lookup, TypeElement typeElement, Class<?>... classes) {
        if (typeElement instanceof ClassType classType) {
            Type type = classTypeToTypeOrThrow(lookup, classType);
            return Arrays.stream(classes).anyMatch(clazz -> clazz.isAssignableFrom((Class<?>) type));
        }
        return false;

    }

    static boolean isAssignableTo(MethodHandles.Lookup lookup, JavaType javaType, Class<?>... classes) {
        if (javaType instanceof ClassType classType) {
            Type type = classTypeToTypeOrThrow(lookup, classType);
            Class<?> evalKlass = (Class<?>) type;
            return Arrays.stream(classes).anyMatch(evalKlass::isAssignableFrom);
        }
        return false;

    }

    static JavaOp.InvokeOp getTargetInvokeOp(JavaOp.LambdaOp lambdaOp) {
        return lambdaOp.body().entryBlock().ops().stream()
                .filter(op -> op instanceof JavaOp.InvokeOp)
                .map(op -> (JavaOp.InvokeOp) op)
                .findFirst().orElseThrow();
    }

    static Object[] getQuotedCapturedValues(JavaOp.LambdaOp lambdaOp, Quoted quoted, Method method) {
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


    static Stream<Op> statements(Block block) {
        return block.ops().stream().filter(op->
        (
                (op instanceof CoreOp.VarAccessOp.VarStoreOp && op.operands().get(1).uses().size() < 2)
             || (op instanceof CoreOp.VarOp || op.result().uses().isEmpty())
             || (op instanceof HATMemoryVarOp)
             || (op instanceof HATVectorVarOp)
             || (op instanceof HATF16VarOp)
        )
        && !(op instanceof CoreOp.VarOp varOp && ParamVar.of(varOp) != null)
        && !(op instanceof CoreOp.YieldOp));
    }

    static JavaType javaRefType(JavaOp.InvokeOp op) {
        return (JavaType) op.invokeDescriptor().refType();
    }

    static JavaType javaReturnType(JavaOp.InvokeOp invokeOp) {
        return (JavaType) invokeOp.invokeDescriptor().type().returnType();
    }
    static boolean javaReturnTypeIsVoid(JavaOp.InvokeOp invokeOp) {
        return javaReturnType(invokeOp) instanceof PrimitiveType primitiveType && primitiveType.isVoid();
    }

    static Method methodOrThrow(MethodHandles.Lookup lookup, JavaOp.InvokeOp op) {
        try {
            return op.invokeDescriptor().resolveToMethod(lookup);
        } catch (ReflectiveOperationException e) {
            throw new RuntimeException(e);
        }
    }


    static Class<?> javaRefClassOrThrow(MethodHandles.Lookup lookup, JavaOp.InvokeOp op) {
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
            case HATPtrStoreOp o -> 13;
            case HATPtrLengthOp o -> 0;
            case HATPtrLoadOp o -> 0;
            case JavaOp.FieldAccessOp o -> 0;
            case HATThreadOp o -> 0;
            case CoreOp.VarAccessOp.VarLoadOp o -> 0;
            case HATVectorSelectLoadOp o -> 0;      // same as VarLoadOp
            case HATVectorLoadOp o -> 0;
            case HATF16VarLoadOp o -> 0;
            case CoreOp.ConstantOp o -> 0;
            case JavaOp.LambdaOp o -> 0;
            case CoreOp.TupleOp o -> 0;
            case JavaOp.WhileOp o -> 0;
            case JavaOp.ConvOp o -> 1;
            case HATF16ToFloatConvOp o -> 1;
            case JavaOp.NegOp  o-> 1;
            case JavaOp.ModOp o -> 2;
            case JavaOp.MulOp o -> 2;
            case HATVectorMulOp o -> 2;
            case HATF16MulOp o -> 2;
            case JavaOp.DivOp o -> 2;
            case HATVectorDivOp o -> 2;
            case HATF16DivOp o -> 2;
            case JavaOp.NotOp o -> 2;
            case JavaOp.AddOp o -> 3;
            case HATVectorAddOp o -> 3;
            case HATVectorSubOp o -> 3;
            case HATF16AddOp o -> 3;
            case HATF16SubOp o -> 3;
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
    static boolean needsParenthesis(Op parent, Op child) {
        return OpTk.precedenceOf(parent) <= OpTk.precedenceOf(child);
    }

    static Op.Result lhsResult(JavaOp.BinaryOp binaryOp){
        return (Op.Result)binaryOp.operands().get(0);
    }

    static Op.Result rhsResult(JavaOp.BinaryOp binaryOp){
        return (Op.Result)binaryOp.operands().get(1);
    }

    static List<Op> ops(JavaOp.JavaConditionalOp javaConditionalOp, int idx){
        return javaConditionalOp.bodies().get(idx).entryBlock().ops();
    }

    static List<Op> lhsOps(JavaOp.JavaConditionalOp javaConditionalOp){
        return ops(javaConditionalOp,0);
    }

    static List<Op> rhsOps(JavaOp.JavaConditionalOp javaConditionalOp){
        return ops(javaConditionalOp,1);
    }

    static Op.Result result(JavaOp.BinaryTestOp binaryTestOp, int idx){
        return (Op.Result)binaryTestOp.operands().get(idx);
    }

    static Op.Result lhsResult(JavaOp.BinaryTestOp binaryTestOp){
        return result(binaryTestOp,0);
    }

    static Op.Result rhsResult(JavaOp.BinaryTestOp binaryTestOp){
        return result(binaryTestOp,1);
    }

    static Op.Result result(JavaOp.ConvOp convOp){
        return (Op.Result)convOp.operands().getFirst();
    }

    static Op.Result result(CoreOp.ReturnOp returnOp){
        return (Op.Result)returnOp.operands().getFirst();
    }

    static Block block(JavaOp.ConditionalExpressionOp ternaryOp, int idx){
        return ternaryOp.bodies().get(idx).entryBlock();
    }

    static Block condBlock(JavaOp.ConditionalExpressionOp ternaryOp){
        return block(ternaryOp,0);
    }

    static Block thenBlock(JavaOp.ConditionalExpressionOp ternaryOp){
        return block(ternaryOp,1);
    }

    static Block elseBlock(JavaOp.ConditionalExpressionOp ternaryOp){
        return block(ternaryOp,2);
    }

    static String funcName(JavaOp.InvokeOp invokeOp) {
        return invokeOp.invokeDescriptor().name();
    }

    static Value operandOrNull(Op op, int idx) {
        return op.operands().size() > idx?op.operands().get(idx):null;
    }

    static Op.Result resultOrNull(Op op, int idx) {
        return (operandOrNull(op,idx) instanceof Op.Result result)?result:null;
    }

    static Block block(JavaOp.ForOp forOp, int idx){
        return forOp.bodies().get(idx).entryBlock();
    }

    static Block mutateBlock(JavaOp.ForOp forOp){
        return block(forOp,2);
    }

    static Block loopBlock(JavaOp.ForOp forOp){
        return block(forOp,3);
    }

    static Block condBlock(JavaOp.ForOp forOp){
        return  forOp.cond().entryBlock();
    }

    static Block initBlock(JavaOp.ForOp forOp){
        return  forOp.init().entryBlock();
    }

    static Block block(JavaOp.WhileOp whileOp, int idx){
        return  whileOp.bodies().get(idx).entryBlock();
    }

    static Block condBlock(JavaOp.WhileOp whileOp){
        return  block(whileOp,0);
    }

    static Block loopBlock(JavaOp.WhileOp whileOp){
        return  block(whileOp,1);
    }

    static Block blockOrNull(JavaOp.IfOp ifOp, int idx ){
        return ifOp.bodies().size() > idx?ifOp.bodies().get(idx).entryBlock():null;
    }

    static JavaOp.FieldAccessOp fieldAccessOpNameMatches(CodeElement<?,?> codeElement, Predicate<String> namePredicate) {
        return codeElement instanceof JavaOp.FieldAccessOp fieldAccessOp
                && namePredicate.test(fieldName(fieldAccessOp))?fieldAccessOp:null;
    }


    static void inspectNewLevelWhy(Class<?> interfaceClass, Set<Class<?>> interfaceSet) {
        if (interfaceClass != null && interfaceSet.add(interfaceClass)) {
            // only if we add a new interface class, we inspect all interfaces that extends the current inspected class
            Arrays.stream(interfaceClass.getInterfaces())
                    .forEach(superInterface -> inspectNewLevelWhy(superInterface, interfaceSet));
        }
    }
    static boolean  isVectorOperation(JavaOp.InvokeOp invokeOp, Value varValue, Predicate<String> namePredicate) {
        // is Assignble
        if (varValue instanceof Op.Result r && r.op() instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
            TypeElement typeElement = varLoadOp.resultType();
            Set<Class<?>> interfaces = Set.of();
            try {
                Class<?> aClass = Class.forName(typeElement.toString());
                interfaces = OpTk.inspectAllInterfacesWhy(aClass);
            } catch (ClassNotFoundException _) {
            }
            return interfaces.contains(_V.class) && OpTk.isMethod(invokeOp, namePredicate);
        }
        return false;
    }
    static boolean isVectorOperation(JavaOp.InvokeOp invokeOp, boolean laneOk) {
        String typeElement = invokeOp.invokeDescriptor().refType().toString();
        Set<Class<?>> interfaces;
        try {
            Class<?> aClass = Class.forName(typeElement);
            interfaces = OpTk.inspectAllInterfacesWhy(aClass);
        } catch (ClassNotFoundException _) {
            return false;
        }
        return interfaces.contains(_V.class) && laneOk;
    }

    static Set<Class<?>> inspectAllInterfacesWhy(Class<?> klass) {
        Set<Class<?>> interfaceSet = new HashSet<>();
        while (klass != null) {
            Arrays.stream(klass.getInterfaces())
                    .forEach(interfaceClass -> inspectNewLevelWhy(interfaceClass, interfaceSet));
            klass = klass.getSuperclass();
        }
        return interfaceSet;
    }


    static boolean isInvokeDescriptorSubtypeOf(MethodHandles.Lookup lookup,JavaOp.InvokeOp invokeOp, Class<?> klass) {

        var wouldReturn =  (invokeOp.resultType() instanceof JavaType jt && isAssignable(lookup, jt,klass));

        TypeElement typeElement = invokeOp.invokeDescriptor().refType();
        Set<Class<?>> interfaces = Set.of();
        try {
            Class<?> aClass = Class.forName(typeElement.toString());
            interfaces = inspectAllInterfacesWhy(aClass);
        } catch (ClassNotFoundException _) {
        }
        var butReturns =  interfaces.contains(klass);
        if (butReturns != wouldReturn){
           // System.out.print("isInvokeDescriptorSubtypeOf");
        }
        return butReturns;

    }

    static boolean isInvokeDescriptorSubtypeOfAnyMatch(MethodHandles.Lookup lookup,JavaOp.InvokeOp invokeOp, Class<?> ... klasses) {

        boolean wouldReturn=  (invokeOp.resultType() instanceof JavaType jt && isAssignable(lookup, jt,klasses));
       boolean butReturns = false;
        TypeElement typeElement = invokeOp.invokeDescriptor().refType();
        Set<Class<?>> interfaces = Set.of();
        try {
            Class<?> aClass = Class.forName(typeElement.toString());
            interfaces = inspectAllInterfacesWhy(aClass);
        } catch (ClassNotFoundException _) {
        }
        for (Class<?> klass : klasses) {
            if (interfaces.contains(klass)) {
                butReturns =  true;
            }
        }
        if (butReturns != wouldReturn){
         //   System.out.print("isInvokeDescriptorSubtypeOfAnyMatch");
        }
        return butReturns;
    }

    static PrimitiveType asPrimitiveResultOrNull(Value v){
        if (v instanceof Op.Result r){
            if (r.op().resultType() instanceof PrimitiveType primitiveType){
                return primitiveType;
            }
        }
        return null;
    }
    static boolean isPrimitiveResult(Value v){
        return (asPrimitiveResultOrNull(v)!=null);
    }

    static Op.Result asResultOrThrow(Value value) {
        if (value instanceof Op.Result r) {
           return r;
        }else{
            throw new RuntimeException("Value not a result");
        }
    }

    static Stream<Op.Result> operandsAsResults(CodeElement<?,?> codeElement) {
        return codeElement instanceof Op ?
                ((Op)codeElement).operands().stream().filter(o-> o instanceof Op.Result).map(o->(Op.Result)o)
                :Stream.of();
    }
    static Op.Result operandAsResult(CodeElement<?,?> codeElement, int n) {
        return codeElement instanceof Op op  && op.operands().size()>n && op.operands().get(n) instanceof Op.Result result?result:null;
    }
    static Op opFromOperandAsResult(CodeElement<?,?> codeElement, int n) {
        return operandAsResult(codeElement,n) instanceof Op.Result result?result.op():null;
    }

    static Op.Result asResultOrNull(Value operand) {
        return operand instanceof Op.Result result?result:null;
    }
    static boolean isResult(Value operand) {
        return Objects.nonNull(asResultOrNull(operand));
    }

    static Op opOfResultOrNull(Op.Result result) {
        return result.op() instanceof Op op?op:null;
    }

    static TypeElement resultTypeOrNull(CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
        return varLoadOp.resultType() instanceof TypeElement typeElement?typeElement:null;
    }

    static CoreOp.VarAccessOp.VarLoadOp asVarLoadOrNull(Op op) {
        return  op instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp?varLoadOp:null;
    }

    static boolean resultType(MethodHandles.Lookup lookup, CoreOp.VarAccessOp.VarLoadOp varLoadOp, Class<?>... classes) {
        return OpTk.isAssignable(lookup, varLoadOp.resultType(), classes);
    }

    record CallSite(Class<?> clazz,String methodName, boolean tracing){
        public static CallSite of(Class<?> clazz, String methodName) {
            return new CallSite(clazz,methodName, Boolean.getBoolean("TRACE_CALLSITES"));
        }
        public static CallSite of(Class<?> clazz) {
            for (StackTraceElement ste : Thread.currentThread().getStackTrace()) {
                if (ste.getClassName().equals(clazz.getName())) {
                    new CallSite(ste.getClass(),ste.getMethodName(), Boolean.getBoolean("TRACE_CALLSITES"));
                }
            }
            return new CallSite(clazz,"???", Boolean.getBoolean("TRACE_CALLSITES"));
        }

        @Override public  String toString(){
            return clazz.toString()+":"+methodName;
        }
    }
    static CoreOp.FuncOp lower(CallSite callSite, CoreOp.FuncOp funcOp) {
        if (callSite.tracing){
            System.out.println(callSite);
        }
        return funcOp.transform(CodeTransformer.LOWERING_TRANSFORMER);
    }
    static Stream<CodeElement<?,?>> elements(CallSite callSite, CoreOp.FuncOp funcOp) {
        if (callSite.tracing){
            System.out.println(callSite);
        }
        return funcOp.elements();
    }
    static <T extends Op> Stream<T> ops(CallSite callSite, CoreOp.FuncOp funcOp,
                                             Predicate<CodeElement<?,?>> predicate,
                                             Function<CodeElement<?,?>,T> mapper
    ) {
        if (callSite.tracing){
            System.out.println(callSite);
        }
        return funcOp.elements().filter(predicate).map(mapper);
    }
    static <T> Stream<T> opstream(CoreOp.FuncOp funcOp, Function<CodeElement<?,?>,T> mapper) {
        return funcOp.elements().map(mapper).filter(Objects::nonNull);
    }


    static CoreOp.FuncOp SSATransformLower(CallSite callSite, CoreOp.FuncOp funcOp){
        if (callSite.tracing){
            System.out.println(callSite);
        }
        return  SSA.transform(lower(callSite,funcOp));
    }
    static CoreOp.FuncOp SSATransform(CallSite callSite, CoreOp.FuncOp funcOp){
        if (callSite.tracing){
            System.out.println(callSite);
        }
        return  SSA.transform(funcOp);
    }

    static CoreOp.FuncOp transform(CallSite callSite, CoreOp.FuncOp funcOp, Predicate<Op> predicate, CodeTransformer CodeTransformer) {
        if (callSite.tracing){
            System.out.println(callSite);
        }
        return funcOp.transform((blockBuilder, op) -> {
            if (predicate.test(op)){
                var builder = CodeTransformer.acceptOp(blockBuilder,op);
                if (builder != blockBuilder){
                    throw new RuntimeException("Where does this builder come from "+builder);
                }
            }else {
                blockBuilder.op(op);
            }
            return blockBuilder;
        });
    }

    static CoreOp.FuncOp transform(CallSite callSite, CoreOp.FuncOp funcOp, CodeTransformer CodeTransformer) {
        if (callSite.tracing){
            System.out.println(callSite);
        }
        return funcOp.transform(CodeTransformer);
    }

     record  OpMap(CoreOp.FuncOp fromFuncOp, CoreOp.FuncOp toFuncOp,  Map<Op,Op> fromToOpMap){}

    static <InOp extends Op, OutOp extends Op> OutOp replaceOp(Block.Builder blockBuilder, InOp inOp,java.util.function.Function<List<Value>, OutOp> factory) {
        List<Value> inputOperands = inOp.operands();
        CodeContext context = blockBuilder.context();
        List<Value> outputOperands = context.getValues(inputOperands);
        OutOp outOp = factory.apply(outputOperands);
        Op.Result outputResult = blockBuilder.op(outOp);
        Op.Result inputResult = inOp.result();
        outOp.setLocation(inOp.location());
        context.mapValue(inputResult, outputResult);
        return outOp;
    }
    static < OutOp extends Op> OpMap simpleOpMappingTransform(OpTk.CallSite here, CoreOp.FuncOp fromFuncOp, Predicate<Op> opPredicate,
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




    static boolean returnIsVoid(JavaOp.InvokeOp invokeOp){
        return javaReturnType(invokeOp) instanceof PrimitiveType primitiveType && primitiveType.isVoid();
    }

    // IMPORTANT:
    // When we have patterns like:
    //
    // myiFaceArray.array().value(storeAValue);
    //
    // We need to generate extra parenthesis to make the struct pointer accessor "->" correct.
    // This is a common pattern when we have a IFace type that contains a subtype based on
    // struct or union.
    //
    // An example of this is for the type F16Array.
    static boolean needExtraParenthesis(JavaOp.InvokeOp invokeOp) {

        // The following expression checks that the current invokeOp has at least 2 operands:
        // Why 2?
        // - The first one is another invokeOp to load the inner struct from an IFace data structure.
        //   The first operand is also assignable.
        // - The second one is the store value, but this depends on the semantics and definition
        //   of the user code.
        return invokeOp.operands().size() >= 2 && invokeOp.operands().get(0) instanceof Op.Result r1
                && r1.op() instanceof JavaOp.InvokeOp invokeOp2
                && OpTk.javaReturnType(invokeOp2) instanceof ClassType;
    }


    static boolean isMethod(JavaOp.InvokeOp invokeOp, Predicate<String> namePredicate) {
        return namePredicate.test(invokeOp.invokeDescriptor().name());
    }
    static boolean isIfaceBufferInvokeOpWithName(MethodHandles.Lookup lookup, JavaOp.InvokeOp invokeOp, Predicate<String> namePredicate) {
        return OpTk.isIfaceBufferMethod(lookup, invokeOp) && OpTk.isMethod(invokeOp, namePredicate)
                || OpTk.isHatType(lookup, invokeOp) && OpTk.isMethod(invokeOp, namePredicate);
    }

    static  Class<?> typeElementToClass(MethodHandles.Lookup lookup,TypeElement type) {
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
        try {
            if (type instanceof PrimitiveType primitiveType) {
                return PrimitiveHolder.primitiveToClass.get(primitiveType);
            } else if (type instanceof ClassType classType) {
                return ((Class<?>) classType.resolve(lookup));
            } else {
                throw new IllegalArgumentException("given type cannot be converted to class");
            }
        } catch (ReflectiveOperationException e) {
            throw new RuntimeException("given type cannot be converted to class");
        }
    }
    static int dimIdx(String name){
            int dim = name.length()==3?name.charAt(2)-'x':-1;
            if (dim <0||dim>3){
                throw new IllegalStateException();//'x'=1,'y'=2....
            }
            return dim;
    }
    static int dimIdx(JavaOp.FieldAccessOp.FieldLoadOp fieldLoadOp){
        return dimIdx(fieldLoadOp.fieldDescriptor().name());
    }
}

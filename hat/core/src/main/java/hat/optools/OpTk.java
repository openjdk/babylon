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

import hat.callgraph.CallGraph;
import jdk.incubator.code.Op;
import jdk.incubator.code.OpTransformer;
import jdk.incubator.code.Quoted;
import jdk.incubator.code.TypeElement;
import jdk.incubator.code.analysis.SSA;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.ClassType;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.JavaType;
import jdk.incubator.code.dialect.java.MethodRef;

import java.lang.invoke.MethodHandles;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.lang.reflect.Type;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.stream.Stream;

public class OpTk {



    public static Op lhsAsOp(Op op) {
        return ((Op.Result)op.operands().getFirst()).op();
    }

    public static Op rhsAsOp(Op op) {
        return  ((Op.Result)op.operands().get(1)).op();
    }

    public static Stream<OpWrapper<?>> lhsWrappedYieldOpStream(MethodHandles.Lookup lookup,JavaOp.JavaConditionalOp op) {
        return op.bodies().get(0).entryBlock().ops().stream().filter(o->o instanceof CoreOp.YieldOp).map(o-> OpWrapper.wrap(lookup,o));
    }

    public static Stream<OpWrapper<?>> rhsWrappedYieldOpStream(MethodHandles.Lookup lookup, JavaOp.JavaConditionalOp op) {
        return op.bodies().get(1).entryBlock().ops().stream().filter(o->o instanceof CoreOp.YieldOp).map(o-> OpWrapper.wrap(lookup,o));
    }

    public static boolean isKernelContextAccess(JavaOp.FieldAccessOp fieldAccessOp) {
        return fieldAccessOp.fieldDescriptor().refType() instanceof  ClassType classType && classType.toClassName().equals("hat.KernelContext");
    }

    public static TypeElement fieldType(JavaOp.FieldAccessOp fieldAccessOp) {
        return fieldAccessOp.fieldDescriptor().refType();
    }

    public static String fieldName(JavaOp.FieldAccessOp fieldAccessOp) {
        return fieldAccessOp.fieldDescriptor().name();
    }

    public static JavaType javaType(CoreOp.VarOp op) {
        return (JavaType) op.varValueType();
    }

    public static String varName(CoreOp.VarOp op) {
        return op.varName();
    }

    public static Object getStaticFinalPrimitiveValue(MethodHandles.Lookup lookup, JavaOp.FieldAccessOp.FieldLoadOp fieldLoadOp) {
        if (fieldType(fieldLoadOp) instanceof ClassType classType) {
            Class<?> clazz = (Class<?>) classTypeToType(lookup,classType);
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

    public static Stream<OpWrapper<?>> initWrappedYieldOpStream(MethodHandles.Lookup lookup, JavaOp.ForOp op) {
        return  op.init().entryBlock().ops().stream().filter(o->o instanceof CoreOp.YieldOp).map(o->OpWrapper.wrap(lookup,o) );
    }

    // Maybe instead of three of these we can add cond() to Op.Loop?
    public static Stream<OpWrapper<?>> conditionWrappedYieldOpStream(MethodHandles.Lookup lookup, JavaOp.ForOp op ) {
        return  op.cond().entryBlock().ops().stream().filter(o->o instanceof CoreOp.YieldOp).map(o->OpWrapper.wrap(lookup,o) );
    }
    public static Stream<OpWrapper<?>> conditionWrappedYieldOpStream(MethodHandles.Lookup lookup, JavaOp.WhileOp op) {
        // ADD op.cond() to JavaOp.WhileOp  match ForOp?
        return op.bodies().getFirst().entryBlock().ops().stream().filter(o->o instanceof CoreOp.YieldOp).map(o-> OpWrapper.wrap(lookup,o));
    }
    public static Stream<OpWrapper<?>> conditionWrappedYieldOpStream(MethodHandles.Lookup lookup,JavaOp.ConditionalExpressionOp op) {
        // ADD op.cond() to JavaOp.ConditionalExpressionOp match ForOp?
        return op.bodies().getFirst().entryBlock().ops().stream().filter(o->o instanceof CoreOp.YieldOp).map(o-> OpWrapper.wrap(lookup,o));
    }


    public static Stream<OpWrapper<?>> loopWrappedRootOpStream(MethodHandles.Lookup lookup, Op.Loop op) {
        var list = new ArrayList<>(RootSet.rootsWithoutVarFuncDeclarationsOrYields(lookup,op.loopBody().entryBlock()).toList());
        if (list.getLast() instanceof JavaContinueOpWrapper) {
            list.removeLast();
        }
        return list.stream();
    }


    public static Stream<OpWrapper<?>> mutateRootWrappedOpStream(MethodHandles.Lookup lookup, JavaOp.ForOp op) {
        return RootSet.rootsWithoutVarFuncDeclarationsOrYields(lookup,op.bodies().get(2).entryBlock());
    }

    public static Stream<OpWrapper<?>> thenWrappedYieldOpStream(MethodHandles.Lookup lookup, JavaOp.ConditionalExpressionOp op) {
        return op.bodies().get(1).entryBlock().ops().stream().filter(o->o instanceof CoreOp.YieldOp).map(o-> OpWrapper.wrap(lookup,o));
    }

    public static Stream<OpWrapper<?>> elseWrappedYieldOpStream(MethodHandles.Lookup lookup, JavaOp.ConditionalExpressionOp op) {
        return op.bodies().get(2).entryBlock().ops().stream().filter(o->o instanceof CoreOp.YieldOp).map(o-> OpWrapper.wrap(lookup,o));
    }

    public static boolean hasElse(JavaOp.IfOp op,  int idx) {
        return op.bodies().size()>idx && op.bodies().get(idx).entryBlock().ops().size() > 1;
    }



    public static CoreOp.ModuleOp createTransitiveInvokeModule(MethodHandles.Lookup l,
                                                        FuncOpWrapper entry, CallGraph<?> callGraph) {
        LinkedHashSet<MethodRef> funcsVisited = new LinkedHashSet<>();
        List<CoreOp.FuncOp> funcs = new ArrayList<>();
        record RefAndFunc(MethodRef r, FuncOpWrapper f) {}

        Deque<RefAndFunc> work = new ArrayDeque<>();
        entry.op.traverse(null, (map, op) -> {
            if (op instanceof JavaOp.InvokeOp invokeOp) {
                var invokeOpWrapper = (InvokeOpWrapper) OpWrapper.wrap(entry.lookup,invokeOp);
                MethodRef methodRef = invokeOpWrapper.methodRef();
                Method method = null;
                Class<?> javaRefTypeClass = invokeOpWrapper.javaRefClass().orElseThrow();
                try {
                    method = methodRef.resolveToMethod(l, invokeOpWrapper.op.invokeKind());
                } catch (ReflectiveOperationException _) {
                    throw new IllegalStateException("Could not resolve invokeWrapper to method");
                }
                Optional<CoreOp.FuncOp> f = Op.ofMethod(method);
                if (f.isPresent() && !callGraph.filterCalls(f.get(), invokeOpWrapper, method, methodRef, javaRefTypeClass)) {
                    work.push(new RefAndFunc(methodRef, new FuncOpWrapper(l, f.get())));
                }
            }
            return map;
        });

        while (!work.isEmpty()) {
            RefAndFunc rf = work.pop();
            if (!funcsVisited.add(rf.r)) {
                continue;
            }

            CoreOp.FuncOp tf = rf.f.op.transform(rf.r.name(), (blockBuilder, op) -> {
                if (op instanceof JavaOp.InvokeOp iop) {
                    InvokeOpWrapper iopWrapper = OpWrapper.wrap(entry.lookup, iop);
                    MethodRef methodRef = iopWrapper.methodRef();
                    Method invokeOpCalledMethod = null;
                    try {
                        invokeOpCalledMethod = methodRef.resolveToMethod(l, iop.invokeKind());
                    } catch (ReflectiveOperationException _) {
                        throw new IllegalStateException("Could not resolve invokeWrapper to method");
                    }
                    if (invokeOpCalledMethod instanceof Method m) {
                        Optional<CoreOp.FuncOp> f = Op.ofMethod(m);
                        if (f.isPresent()) {
                            RefAndFunc call = new RefAndFunc(methodRef, new FuncOpWrapper(l, f.get()));
                            work.push(call);

                            Op.Result result = blockBuilder.op(CoreOp.funcCall(
                                    call.r.name(),
                                    call.f.op.invokableType(),
                                    blockBuilder.context().getValues(iop.operands())));
                            blockBuilder.context().mapValue(op.result(), result);
                            return blockBuilder;
                        }
                    }
                }
                blockBuilder.op(op);
                return blockBuilder;
            });
            funcs.addFirst(tf);
        }

        return CoreOp.module(funcs);
    }

    public  static Type classTypeToType(MethodHandles.Lookup lookup, ClassType classType) {
        Type javaTypeClass = null;
        try {
            javaTypeClass = classType.resolve(lookup);
        } catch (ReflectiveOperationException e) {
            throw new RuntimeException(e);
        }
        return javaTypeClass;

    }

    public  static boolean isAssignable(MethodHandles.Lookup lookup, JavaType javaType, Class<?> ... classes) {
        if (javaType instanceof ClassType classType) {
            Type type = classTypeToType(lookup,classType);
            for (Class<?> clazz : classes) {
                if (clazz.isAssignableFrom((Class<?>) type)) {
                    return true;
                }
            }
        }
        return false;

    }





    public static InvokeOpWrapper getQuotableTargetInvokeOpWrapper(MethodHandles.Lookup lookup,JavaOp.LambdaOp lambdaOp) {
        return OpWrapper.wrap(lookup, lambdaOp.body().entryBlock().ops().stream()
                .filter(op -> op instanceof JavaOp.InvokeOp)
                .map(op -> (JavaOp.InvokeOp) op)
                .findFirst().get());
    }

    public static MethodRef getQuotableTargetMethodRef(MethodHandles.Lookup lookup, JavaOp.LambdaOp lambdaOp) {
        return getQuotableTargetInvokeOpWrapper(lookup,lambdaOp).methodRef();
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

    public static FuncOpWrapper lower(MethodHandles.Lookup lookup, CoreOp.FuncOp funcOp) {
        return OpWrapper.wrap(lookup,funcOp.transform(OpTransformer.LOWERING_TRANSFORMER));
    }

    public static FuncOpWrapper ssa(MethodHandles.Lookup lookup, CoreOp.FuncOp funcOp) {
        return OpWrapper.wrap(lookup, SSA.transform(funcOp));
    }

    public static Stream<OpWrapper<?>> wrappedRootOpStream(MethodHandles.Lookup lookup, CoreOp.FuncOp op) {
        return RootSet.rootsWithoutVarFuncDeclarationsOrYields(lookup,op.bodies().getFirst().entryBlock());
    }
}

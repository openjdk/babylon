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
import hat.buffer.Buffer;
import hat.buffer.KernelContext;
import hat.callgraph.CallGraph;
import hat.ifacemapper.MappableIface;
import jdk.incubator.code.Block;
import jdk.incubator.code.Op;
import jdk.incubator.code.OpTransformer;
import jdk.incubator.code.Quoted;
import jdk.incubator.code.Value;
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
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.function.Predicate;
import java.util.stream.Collectors;
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


    public static Stream<Op> loopRootOpStream(Op.Loop op) {
        var list = new ArrayList<>(rootsExcludingVarFuncDeclarationsAndYields( op.loopBody().entryBlock()).toList());
        if (list.getLast() instanceof JavaOp.ContinueOp) {
            list.removeLast();
        }
        return list.stream();
    }

    public static CoreOp.ModuleOp createTransitiveInvokeModule(MethodHandles.Lookup l,
                                                               CoreOp.FuncOp entry, CallGraph<?> callGraph) {
        LinkedHashSet<MethodRef> funcsVisited = new LinkedHashSet<>();
        List<CoreOp.FuncOp> funcs = new ArrayList<>();
        record RefAndFunc(MethodRef r, CoreOp.FuncOp f) {
        }

        Deque<RefAndFunc> work = new ArrayDeque<>();
        entry.traverse(null, (map, op) -> {
            if (op instanceof JavaOp.InvokeOp invokeOp) {
                Class<?> javaRefTypeClass = javaRefClassOrThrow(callGraph.computeContext.accelerator.lookup, invokeOp);
                try {
                    var method = invokeOp.invokeDescriptor().resolveToMethod(l, invokeOp.invokeKind());
                    CoreOp.FuncOp f = Op.ofMethod(method).orElse(null);
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
            if (!funcsVisited.add(rf.r)) {
                continue;
            }

            CoreOp.FuncOp tf = rf.f.transform(rf.r.name(), (blockBuilder, op) -> {
                if (op instanceof JavaOp.InvokeOp iop) {
                    try {
                        Method invokeOpCalledMethod = iop.invokeDescriptor().resolveToMethod(l, iop.invokeKind());
                        if (invokeOpCalledMethod instanceof Method m) {
                            CoreOp.FuncOp f = Op.ofMethod(m).orElse(null);
                            if (f!=null) {
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

        return CoreOp.module(funcs);
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


    public static JavaOp.InvokeOp getQuotableTargetInvokeOpWrapper( JavaOp.LambdaOp lambdaOp) {
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

    public static CoreOp.FuncOp lower( CoreOp.FuncOp funcOp) {
        return funcOp.transform(OpTransformer.LOWERING_TRANSFORMER);
    }

    public static Stream<Op> rootOpStream( CoreOp.FuncOp op) {
        return rootsExcludingVarFuncDeclarationsAndYields(op.bodies().getFirst().entryBlock());
    }


    static Predicate<Op> rootFilter = op->
            (   (op instanceof CoreOp.VarAccessOp.VarStoreOp && op.operands().get(1).uses().size() < 2)
              || (op instanceof CoreOp.VarOp || op.result().uses().isEmpty())
            )
            && !(op instanceof CoreOp.VarOp varOp && paramVar(varOp) != null)
            && !(op instanceof CoreOp.YieldOp);

    static public Stream<Op> rootsExcludingVarFuncDeclarationsAndYields(Block block) {
        return block.ops().stream().filter(rootFilter);
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
                && (isAssignable(lookup, javaType, hat.KernelContext.class) || isAssignable(lookup, javaType, KernelContext.class))
        );
    }

    public static boolean isComputeContextMethod(MethodHandles.Lookup lookup, JavaOp.InvokeOp invokeOp) {
        return isAssignable(lookup, javaRefType(invokeOp), ComputeContext.class);
    }

    public static JavaType javaReturnType(JavaOp.InvokeOp op) {
        return (JavaType) op.invokeDescriptor().type().returnType();
    }

    public static Method methodOrThrow(MethodHandles.Lookup lookup, JavaOp.InvokeOp op) {
        try {
            return op.invokeDescriptor().resolveToMethod(lookup, op.invokeKind());
        } catch (ReflectiveOperationException e) {
            throw new RuntimeException(e);
        }
    }

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


    public static Class<?> javaRefClassOrThrow(MethodHandles.Lookup lookup, JavaOp.InvokeOp op) {
        if (javaRefType(op) instanceof ClassType classType) {
            return (Class<?>) classTypeToTypeOrThrow(lookup, classType);
        } else {
            throw new IllegalStateException(" javaRef class is null");
        }
    }


    public record ParamVar(CoreOp.VarOp varOp, Block.Parameter parameter, CoreOp.FuncOp funcOp) {
    }

    public static ParamVar paramVar(CoreOp.VarOp varOp) {
        return !varOp.isUninitialized()
                && varOp.operands().getFirst() instanceof Block.Parameter parameter
                && parameter.invokableOperation() instanceof CoreOp.FuncOp funcOp ? new ParamVar(varOp, parameter, funcOp) : null;
    }

    public static boolean isStructural(Op op){
        return switch (op){
            case JavaOp.ForOp _ -> true;
            case JavaOp.WhileOp _ -> true;
            case JavaOp.IfOp _ -> true;
            case JavaOp.LabeledOp _ ->true;
            case JavaOp.YieldOp _ ->true;
            case CoreOp.TupleOp _ ->true;
            default -> false;
        };
    }
}

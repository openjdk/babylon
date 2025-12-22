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
import hat.types.HAType;
import hat.callgraph.CallGraph;
import hat.device.DeviceType;
import hat.dialect.*;
import optkl.LookupCarrier;
import optkl.OpTkl;
import optkl.ifacemapper.MappableIface;
import hat.types._V;
import jdk.incubator.code.Block;
import jdk.incubator.code.CodeElement;
import jdk.incubator.code.Op;
import jdk.incubator.code.TypeElement;
import jdk.incubator.code.Value;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.JavaType;
import jdk.incubator.code.dialect.java.MethodRef;
import optkl.ParamVar;

import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Deque;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.function.Predicate;
import java.util.regex.Pattern;
import java.util.stream.Stream;

import static optkl.OpTkl.AnyFieldAccess;
import static optkl.OpTkl.elements;
import static optkl.OpTkl.isAssignable;
import static optkl.OpTkl.isAssignableTo;
import static optkl.OpTkl.isMethod;
import static optkl.OpTkl.javaRefClassOrThrow;
import static optkl.OpTkl.javaRefType;

public interface OpTk extends LookupCarrier  {

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


    static JavaOp.InvokeOp asKernelContextInvokeOpOrNull(MethodHandles.Lookup lookup, CodeElement<?,?> ce, Predicate<JavaOp.InvokeOp> predicate) {
        if (ce instanceof JavaOp.InvokeOp invokeOp) {
            if (isKernelContext(lookup, invokeOp.invokeDescriptor().refType())) {
                return predicate.test(invokeOp) ? invokeOp : null;
            } else if (invokeOp.operands().size() > 1
                    && invokeOp.operands().getFirst() instanceof Value value
                    && isKernelContext(lookup, value.type())) {
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





    static boolean isComputeContextMethod(MethodHandles.Lookup lookup, JavaOp.InvokeOp invokeOp) {
        return isAssignable(lookup, javaRefType(invokeOp), ComputeContext.class);
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
        var here = OpTkl.CallSite.of(OpTkl.class, "createTransitiveInvokeModule");
        elements(here, entry).forEach(codeElement -> {
            if (codeElement instanceof JavaOp.InvokeOp invokeOp) {
                Class<?> javaRefTypeClass = javaRefClassOrThrow(callGraph.lookup(), invokeOp);
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
            return interfaces.contains(_V.class) && isMethod(invokeOp, namePredicate);
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

    static boolean isIfaceBufferInvokeOpWithName(MethodHandles.Lookup lookup, JavaOp.InvokeOp invokeOp, Predicate<String> namePredicate) {
        return isIfaceBufferMethod(lookup, invokeOp) && isMethod(invokeOp, namePredicate)
                || isAssignableTo(lookup, javaRefType(invokeOp), DeviceType.class, MappableIface.class, HAType.class)
                && isMethod(invokeOp, namePredicate);
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

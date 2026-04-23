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
package hat.callgraph;

import hat.BufferTagger;
import hat.KernelContext;
import hat.device.NonMappableIface;
import hat.phases.HATTier;
import hat.types.S16ImplOfF16;
import hat.types.Tensor;
import jdk.incubator.code.CodeTransformer;
import jdk.incubator.code.Op;
import jdk.incubator.code.CodeType;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.core.SSA;
import jdk.incubator.code.dialect.java.ClassType;
import optkl.IfaceValue;
import optkl.OpHelper;
import optkl.ifacemapper.AccessType;
import optkl.ifacemapper.MappableIface;
import optkl.util.Mutable;
import optkl.util.carriers.FuncOpCarrier;
import optkl.util.carriers.LookupCarrier;

import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import java.util.HashSet;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Collectors;

import static optkl.OpHelper.Invoke.invoke;

public class KernelCallGraph implements LookupCarrier {
    @Override public MethodHandles.Lookup lookup(){
        return computeCallGraph.lookup();
    }
    public static final boolean  showKernelCallDag = Boolean.getBoolean("showKernelCallDag");
    public static final  boolean  showKernelIfaceDag = Boolean.getBoolean("showKernelIfaceDag");
    public static final boolean  showKernelIfaceDagProposedTypedefs = Boolean.getBoolean("showKernelIfaceDagProposedTypedefs");
    public final ComputeCallGraph computeCallGraph;
    public final MethodCallDag callDag;

    public final IfaceDataDag<MappableIface> ifaceDag;
    public final List<AccessType> bufferAccessList;
    public final Set<CodeType> accessedTypes;
    public final Set<Class<?>> accessedClasses;
    public final Set<Class<? extends IfaceValue>> accessedIfaceClasses;
    public final Set<Class<? extends NonMappableIface>> accessedNonMappableIfaceClasses;
    public final Set<Class<? extends MappableIface>> accessedMappableIfaceClasses;
    public final Set<Class<? extends IfaceValue.vec>> accessedVecClasses;
    public final Set<Class<? extends S16ImplOfF16>> accessedFP16Classes;
    public boolean usesBarrier;
    public boolean useTensors;
    public boolean usesAtomics;
    public final Set<String> accessedKernelContextFields;

    KernelCallGraph(ComputeCallGraph computeCallGraph, Method method, CoreOp.FuncOp e) {

        this.computeCallGraph = computeCallGraph;

        CoreOp.FuncOp ssaFunc =  SSA.transform( e.transform(CodeTransformer.LOWERING_TRANSFORMER)) ;
        var changed  = Mutable.of(true);
        while (changed.get()) { // loop until no more inline-able functions
            changed.set(false);
            ssaFunc = ssaFunc.transform( (blockbuilder, op) -> {
                if (invoke(lookup(), op) instanceof OpHelper.Invoke invoke                         // always but pattern friendly
                        && invoke.resolvedMethodOrNull() instanceof Method m
                        && Op.ofMethod(m) instanceof Optional<CoreOp.FuncOp> optionalFuncOp // always but pattern friendly
                        && optionalFuncOp.isPresent()
                        && optionalFuncOp.get() instanceof CoreOp.FuncOp inline                  // always we just want var in scope
                ){
                    var ssaInline =SSA.transform(inline.transform(CodeTransformer.LOWERING_TRANSFORMER));
                    var exitBlockBuilder = jdk.incubator.code.dialect.core.Inliner.inline(
                            blockbuilder, ssaInline,
                            blockbuilder.context().getValues(invoke.op().operands()), (_, _value) -> {
                                if (_value != null) {
                                    blockbuilder.context().mapValue(invoke.op().result(), _value);
                                }
                            });
                    if (!exitBlockBuilder.parameters().isEmpty()) {
                        blockbuilder.context().mapValue(invoke.op().result(), exitBlockBuilder.parameters().getFirst());
                    }
                    changed.set(true);
                    return exitBlockBuilder.rebind(blockbuilder.context(), blockbuilder.transformer());
                }
                blockbuilder.op(op);
                return blockbuilder;
            });
        }
        var inlinedEntryPoint = ssaFunc;
        this.usesBarrier = OpHelper.Invoke.stream(lookup(), inlinedEntryPoint)
                .anyMatch(invoke -> invoke.refIs(KernelContext.class) && invoke.named("barrier"));
        this.useTensors = OpHelper.Invoke.stream(lookup(), inlinedEntryPoint)
                .anyMatch(invoke -> invoke.refIs(Tensor.class) && invoke.named("load"));
        this.accessedKernelContextFields = new HashSet<>(OpHelper.FieldAccess.stream(lookup(), inlinedEntryPoint)
                .filter(fieldAccess -> fieldAccess.refType(KernelContext.class)).map(OpHelper.FieldAccess::name).toList()
        );
        this.accessedTypes = inlinedEntryPoint.elements()
                .filter(ce -> ce instanceof Op).map(ce -> ((Op) ce).resultType())
                .collect(Collectors.toSet());
        this.accessedClasses = this.accessedTypes.stream()
                .filter(te -> te instanceof ClassType).map(te -> (Class<?>) OpHelper.classTypeToTypeOrThrow(lookup(), (ClassType) te))
                .collect(Collectors.toSet());
        this.accessedIfaceClasses =  this.accessedClasses.stream()
                .filter(c->IfaceValue.class.isAssignableFrom(c)).map(c->(Class<IfaceValue>)c)
                .collect(Collectors.toSet());
        this.accessedMappableIfaceClasses =  this.accessedIfaceClasses.stream()
                .filter(c->MappableIface.class.isAssignableFrom(c)).map(c->(Class<MappableIface>)c)
                .collect(Collectors.toSet());
        this.accessedNonMappableIfaceClasses =  this.accessedIfaceClasses.stream()
                .filter(c->NonMappableIface.class.isAssignableFrom(c)).map(c->(Class<NonMappableIface>)c)
                .collect(Collectors.toSet());
        this.accessedVecClasses =  this.accessedClasses.stream()
                .filter(c->IfaceValue.vec.class.isAssignableFrom(c)).map(c->(Class<IfaceValue.vec>)c)
                .collect(Collectors.toSet());
        this.accessedFP16Classes =  this.accessedClasses.stream()
                .filter(c-> S16ImplOfF16.class.isAssignableFrom(c)).map(c->(Class<S16ImplOfF16>)c)
                .collect(Collectors.toSet());
        this.usesAtomics = OpHelper.Invoke.stream(lookup(), inlinedEntryPoint)
                .anyMatch(invoke ->
                        invoke instanceof OpHelper.Invoke.Virtual
                                && invoke.operandCount() == 1
                                && invoke.returnsInt()
                                && invoke.nameMatchesRegex("(atomic.*)Inc"));




        this.bufferAccessList = BufferTagger.getAccessList(lookup(), inlinedEntryPoint);

        var entrypoint = new FuncOpCarrier.Impl(e);
        HATTier.transform(HATTier.KernelPhases, lookup(), entrypoint, computeCallGraph.computeContext.config().showCompilationPhases());

        this.callDag = new MethodCallDag(lookup(), method, entrypoint.funcOp(), inlinedEntryPoint);
        callDag.rankOrdered.forEach(f ->
                HATTier.transform(HATTier.KernelPhases, lookup(), f, computeCallGraph.computeContext.config().showCompilationPhases())
        );
        if (showKernelCallDag) {
            this.callDag.view("kernelCallDag", n -> n.funcOp().funcName());
        }

        this.ifaceDag = new IfaceDataDag<>(dag->
            entrypoint.funcOp().elements()
                    .filter(ce -> ce instanceof Op).map(ce -> ((Op) ce).resultType())
                    .filter(codeType -> codeType instanceof ClassType).map(codeType -> dag.getNode(lookup(), (ClassType) codeType))
                    .filter(impl -> IfaceValue.class.isAssignableFrom(impl.clazz()))
                    .forEach(iface -> dag.methodsWithIfaceReturnTypes(iface.clazz())
                            .forEach(retType -> dag.addEdge(iface, retType))
                    )
        );
        if (showKernelIfaceDag) {
            this.ifaceDag.view("kernelDataDag", IfaceDataDag.IfaceInfo::dotName);
        }
        if (showKernelIfaceDagProposedTypedefs) {
            ifaceDag.rankOrdered.forEach(ifaceInfo -> System.out.println("create typedef " + ifaceInfo.classType()));
        }
    }
}

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
import hat.phases.HATArrayViewPhase;
import hat.phases.HATTransformer;
import hat.types.S16ImplOfF16;
import hat.types.Tensor;
import jdk.incubator.code.CodeTransformer;
import jdk.incubator.code.Op;
import jdk.incubator.code.CodeType;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.core.SSA;
import jdk.incubator.code.dialect.java.ClassType;
import jdk.incubator.code.dialect.java.JavaOp;
import optkl.IfaceValue;
import optkl.OpHelper;
import optkl.VarTable;
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

    public static final boolean SHOW_KERNEL_CALL_DAG = Boolean.getBoolean("showKernelCallDag");
    public static final  boolean SHOW_KERNEL_IFACE_DAG = Boolean.getBoolean("showKernelIfaceDag");
    public static final boolean SHOW_KERNEL_IFACE_DAG_PROPOSED_TYPEDEFS = Boolean.getBoolean("showKernelIfaceDagProposedTypedefs");

    public final ComputeCallGraph computeCallGraph;
    public final MethodCallDag callDag;
    public final IfaceDataDag<MappableIface> iFaceDag;
    public final List<AccessType> bufferAccessList;
    public final Set<CodeType> accessedTypes;
    public final Set<Class<?>> accessedClasses;
    public final Set<Class<? extends IfaceValue>> accessedIfaceClasses;
    public final Set<Class<? extends NonMappableIface>> accessedNonMappableIfaceClasses;
    public final Set<Class<? extends MappableIface>> accessedMappableIfaceClasses;
    public final Set<Class<? extends IfaceValue.vec>> accessedVecClasses;
    public final Set<Class<? extends S16ImplOfF16>> accessedFP16Classes;
    public final Set<String> accessedKernelContextFields;

    private boolean usesBarrier;
    private boolean usesAtomics;
    private final VarTable varTable;
    private boolean useVectors;
    private final boolean useTensors;

    KernelCallGraph(ComputeCallGraph computeCallGraph, Method method, CoreOp.FuncOp kernelFunction) {
        this.computeCallGraph = computeCallGraph;
        var inlinedEntryPoint = inlineEntryPoint(kernelFunction);
        this.usesBarrier = OpHelper.Invoke.stream(lookup(), inlinedEntryPoint)
                .anyMatch(invoke -> invoke.refIs(KernelContext.class) && invoke.named("barrier"));
        this.accessedKernelContextFields = new HashSet<>(OpHelper.FieldAccess.stream(lookup(), inlinedEntryPoint)
                .filter(fieldAccess -> fieldAccess.refType(KernelContext.class)).map(OpHelper.FieldAccess::name).toList()
        );
        this.accessedTypes = inlinedEntryPoint.elements()
                .filter(Op.class::isInstance).map(ce -> ((Op) ce).resultType())
                .collect(Collectors.toSet());
        this.accessedClasses = this.accessedTypes.stream()
                .filter(ClassType.class::isInstance).map(te -> (Class<?>) OpHelper.classTypeToTypeOrThrow(lookup(), (ClassType) te))
                .collect(Collectors.toSet());
        this.accessedIfaceClasses =  this.accessedClasses.stream()
                .filter(IfaceValue.class::isAssignableFrom).map(c->(Class<IfaceValue>)c)
                .collect(Collectors.toSet());
        this.accessedMappableIfaceClasses =  this.accessedIfaceClasses.stream()
                .filter(MappableIface.class::isAssignableFrom).map(c->(Class<MappableIface>)c)
                .collect(Collectors.toSet());
        this.accessedNonMappableIfaceClasses =  this.accessedIfaceClasses.stream()
                .filter(NonMappableIface.class::isAssignableFrom).map(c->(Class<NonMappableIface>)c)
                .collect(Collectors.toSet());
        this.accessedVecClasses =  this.accessedClasses.stream()
                .filter(IfaceValue.vec.class::isAssignableFrom).map(c->(Class<IfaceValue.vec>)c)
                .collect(Collectors.toSet());
        this.accessedFP16Classes =  this.accessedClasses.stream()
                .filter(S16ImplOfF16.class::isAssignableFrom).map(c->(Class<S16ImplOfF16>)c)
                .collect(Collectors.toSet());
        this.usesAtomics = OpHelper.Invoke.stream(lookup(), inlinedEntryPoint)
                .anyMatch(invoke ->
                        invoke instanceof OpHelper.Invoke.Virtual
                                && invoke.operandCount() == 1
                                && invoke.returnsInt()
                                && invoke.nameMatchesRegex("(atomic.*)Inc"));

        this.bufferAccessList = BufferTagger.getAccessList(lookup(), inlinedEntryPoint);

        // To detect vectors: it could be either because of the use of vector types, or because
        // array views (going through arrayStoreOp/arrayLoadOp)
        this.useVectors = OpHelper.Invoke.stream(lookup(), inlinedEntryPoint)
                .anyMatch(invoke -> invoke.returns(IfaceValue.Vector.class));
        boolean arrayAccess = inlinedEntryPoint.elements().anyMatch(codeElement -> {
            if (codeElement instanceof JavaOp.ArrayAccessOp.ArrayStoreOp arrayStoreOp) {
                return HATArrayViewPhase.isVectorOp(computeCallGraph.lookup(), arrayStoreOp);
            } else return (codeElement instanceof JavaOp.ArrayAccessOp.ArrayLoadOp arrayLoadOp && HATArrayViewPhase.isVectorOp(computeCallGraph.lookup(), arrayLoadOp));
        });
        this.useVectors |= arrayAccess;

        this.useTensors = OpHelper.Invoke.stream(lookup(), inlinedEntryPoint)
                .anyMatch(invoke -> invoke.returns(Tensor.class));

        var entrypoint = new FuncOpCarrier.Impl(kernelFunction);
        this.varTable = new VarTable();
        varTable.addFunction(entrypoint.funcOp().funcName());

        HATTransformer.transform(HATTransformer.KernelPhases, lookup(), entrypoint, varTable, computeCallGraph.computeContext.config().showCompilationPhases());
        checkSSALowering(entrypoint.funcOp());

        this.callDag = new MethodCallDag(lookup(), method, entrypoint.funcOp(), inlinedEntryPoint);
        callDag.rankOrdered.forEach(f -> {
            varTable.addFunction(f.funcOp().funcName());
            HATTransformer.transform(HATTransformer.KernelPhases, lookup(), f, varTable, computeCallGraph.computeContext.config().showCompilationPhases());
            checkSSALowering(f.funcOp());
        });
        if (SHOW_KERNEL_CALL_DAG) {
            this.callDag.view("kernelCallDag", n -> n.funcOp().funcName());
        }

        this.iFaceDag = new IfaceDataDag<>(dag->
            entrypoint.funcOp().elements()
                    .filter(Op.class::isInstance).map(ce -> ((Op) ce).resultType())
                    .filter(ClassType.class::isInstance).map(codeType -> dag.getNode(lookup(), (ClassType) codeType))
                    .filter(impl -> IfaceValue.class.isAssignableFrom(impl.clazz()))
                    .forEach(iface -> dag.methodsWithIfaceReturnTypes(iface.clazz())
                            .forEach(retType -> dag.addEdge(iface, retType))
                    )
        );
        if (SHOW_KERNEL_IFACE_DAG) {
            this.iFaceDag.view("kernelDataDag", IfaceDataDag.IfaceInfo::dotName);
        }
        if (SHOW_KERNEL_IFACE_DAG_PROPOSED_TYPEDEFS) {
            iFaceDag.rankOrdered.forEach(ifaceInfo -> IO.println("create typedef " + ifaceInfo.classType()));
        }
    }

    /**
     * This check for SSA lowering guarantees that the current HAT Code model (dialect) can be
     * lowered to pure SSA representation. Currently, we do not do anything with the lowered
     * code model. However, this could be useful when transforming the accelerator code
     * to other lower-level representations compared to C99-based representations, such as
     * SPIR-V and CUDA PTX. Thus, for sanity check, we keep this check on.
     *
     * <p>It can be enabled with -DCHECK_SSA_LOWERING</p>
     *
     * @param funcOp Function code model
     */
    private void checkSSALowering(CoreOp.FuncOp funcOp) {
        if (computeCallGraph.computeContext.config().checkSSALowering()) {
            CoreOp.FuncOp loweredCodeModel = funcOp.transform(CodeTransformer.LOWERING_TRANSFORMER);
            CoreOp.FuncOp ssaCodeModel = SSA.transform(loweredCodeModel);
            if (ssaCodeModel == null) {
                throw new IllegalStateException("SSA code model is null");
            }
        }
    }

    private CoreOp.FuncOp inlineEntryPoint(CoreOp.FuncOp func) {
        CoreOp.FuncOp ssaFunc =  SSA.transform(func.transform(CodeTransformer.LOWERING_TRANSFORMER)) ;
        var changed  = Mutable.of(true);
        while (changed.get()) { // loop until no more inline-able functions
            changed.set(false);
            ssaFunc = ssaFunc.transform((blockbuilder, op) -> {
                if (invoke(lookup(), op) instanceof OpHelper.Invoke invoke                         // always but pattern friendly
                        && invoke.resolvedMethodOrNull() instanceof Method m
                        && Op.ofMethod(m) instanceof Optional<CoreOp.FuncOp> optionalFuncOp // always but pattern friendly
                        && optionalFuncOp.isPresent()
                        && optionalFuncOp.get() instanceof CoreOp.FuncOp inline                  // always we just want var in scope
                ) {
                    var ssaInline = SSA.transform(inline.transform(CodeTransformer.LOWERING_TRANSFORMER));
                    var exitBlockBuilder = jdk.incubator.code.dialect.core.Inliner.inline(
                            blockbuilder, ssaInline,
                            blockbuilder.context().getValues(invoke.op().operands()), (_, value) -> {
                                if (value != null) {
                                    blockbuilder.context().mapValue(invoke.op().result(), value);
                                }
                            });
                    if (!exitBlockBuilder.parameters().isEmpty()) {
                        blockbuilder.context().mapValue(invoke.op().result(), exitBlockBuilder.parameters().getFirst());
                    }
                    changed.set(true);
                    return exitBlockBuilder.withContextAndTransformer(blockbuilder.context(), blockbuilder.transformer());
                }
                blockbuilder.add(op);
                return blockbuilder;
            });
        }
        return ssaFunc;
    }

    public void setUsesBarrier(boolean useBarrier) {
        this.usesBarrier = useBarrier;
    }

    public void setUsesAtomics(boolean useAtomics) {
        this.usesAtomics = useAtomics;
    }

    public boolean isUsesBarrier() {
        return usesBarrier;
    }

    public boolean isUsesAtomics() {
        return usesAtomics;
    }

    public boolean useVectors() {
        return this.useVectors;
    }

    public boolean useTensors() { return this.useTensors; }

    public VarTable getVarTable() {
        return varTable;
    }

}

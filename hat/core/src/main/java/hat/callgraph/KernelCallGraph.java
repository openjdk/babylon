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
import hat.Inliner;
import hat.KernelContext;
import hat.phases.HATTier;
import hat.types._F16;
import jdk.incubator.code.Op;
import jdk.incubator.code.TypeElement;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.ClassType;
import jdk.incubator.code.dialect.java.MethodRef;
import optkl.IfaceValue;
import optkl.OpHelper;
import optkl.ifacemapper.AccessType;
import optkl.ifacemapper.Buffer;

import java.lang.reflect.Method;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class KernelCallGraph extends CallGraph<KernelEntrypoint> {
    public static final boolean  showKernelCallDag = Boolean.getBoolean("showKernelCallDag");
    public static final  boolean  showKernelIfaceDag = Boolean.getBoolean("showKernelIfaceDag");
    public static final boolean  showKernelIfaceDagProposedTypedefs = Boolean.getBoolean("showKernelIfaceDagProposedTypedefs");
    public final ComputeCallGraph computeCallGraph;
    public final MethodCallDag callDag;
    public final IfaceDataDag ifaceDag;
    public final List<AccessType> bufferAccessList;
    public final Set<TypeElement> accessedTypes;
    public final Set<Class<?>> accessedClassTypes;
    public boolean usesVecTypes;
    public boolean usesFp16;
    public boolean usesBarrier;
    public boolean usesAtomics;
    public final Set<String> accessedKcFields;

    @Override
    public String toString() {
        StringBuilder stringBuilder = new StringBuilder();
        stringBuilder.append("UsesVecTypes:").append(usesVecTypes).append(", ");
        stringBuilder.append("UsesFp16:").append(usesFp16).append(", ");
        stringBuilder.append("UsesAtomics:").append(usesAtomics).append(", ");
        stringBuilder.append("UsesBarrier:").append(usesBarrier).append(", ");
        stringBuilder.append("AccessedKernelContextFields:").append("[").append(String.join(", ", accessedKcFields)).append("]");
        return stringBuilder.toString();
    }

    public interface KernelReachable {
    }

    public static class KernelReachableResolvedMethodCall extends ResolvedMethodCall implements KernelReachable {
        public KernelReachableResolvedMethodCall(CallGraph<KernelEntrypoint> callGraph, Method method, CoreOp.FuncOp funcOp) {
            super(callGraph, method, funcOp);
        }
    }

    KernelCallGraph(ComputeCallGraph computeCallGraph, Method method, CoreOp.FuncOp entry) {
        super(computeCallGraph.computeContext, new KernelEntrypoint(computeCallGraph.computeContext.lookup(), null, method, entry));
        this.entrypoint.callGraph = this;
        this.computeCallGraph = computeCallGraph;
        var inlinedEntryPoint = Inliner.inlineEntrypoint(lookup(), entrypoint.funcOp());
        this.usesBarrier = OpHelper.Invoke.stream(lookup(), inlinedEntryPoint)
                .anyMatch(invoke -> invoke.refIs(KernelContext.class) && invoke.named("barrier"));
        this.accessedKcFields = new HashSet<>(OpHelper.FieldAccess.stream(lookup(), inlinedEntryPoint)
                .filter(fieldAccess -> fieldAccess.refType(KernelContext.class)).map(OpHelper.FieldAccess::name).toList()
        );
        this.accessedTypes = new HashSet<>(inlinedEntryPoint.elements()
                .filter(ce -> ce instanceof Op).map(ce -> ((Op) ce).resultType()).toList()
        );
        this.accessedClassTypes = new HashSet<>(this.accessedTypes.stream()
                .filter(te -> te instanceof ClassType).map(te -> (ClassType) te).map(ct -> (Class<?>) OpHelper.classTypeToTypeOrThrow(lookup(), ct)).toList()
        );
        this.usesVecTypes = this.accessedClassTypes.stream().anyMatch(IfaceValue.vec.class::isAssignableFrom);
        this.usesFp16 = this.accessedClassTypes.stream().anyMatch(_F16.class::isAssignableFrom);
        this.usesAtomics = OpHelper.Invoke.stream(lookup(), inlinedEntryPoint)
                .anyMatch(invoke -> invoke.operandCount() == 1 && invoke.returnsInt() && invoke.nameMatchesRegex("(atomic.*)Inc"));
        this.bufferAccessList = BufferTagger.getAccessList(lookup(), inlinedEntryPoint);

        HATTier.transform(HATTier.KernelPhases, lookup(), entrypoint, config().showCompilationPhases());

        this.callDag = new MethodCallDag(lookup(), method, entrypoint.funcOp(), inlinedEntryPoint);

        callDag.rankOrdered.forEach(f ->
                HATTier.transform(HATTier.KernelPhases, lookup(), f, config().showCompilationPhases())
        );

        if (showKernelCallDag) {
            this.callDag.view("kernelCallDag", n -> n.funcOp().funcName());
        }

        this.ifaceDag = new IfaceDataDag(lookup(), entrypoint.funcOp());
        if (showKernelIfaceDag) {
            this.ifaceDag.view("kernelDataDag", IfaceDataDag.IfaceInfo::dotName);
        }
        if (showKernelIfaceDagProposedTypedefs) {
            ifaceDag.rankOrdered.forEach(ifaceInfo -> System.out.println("create typedef " + ifaceInfo.classType()));
        }
    }
}

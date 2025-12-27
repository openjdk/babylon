/*
 * Copyright (c) 2025, Oracle and/or its affiliates. All rights reserved.
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
package hat.phases;

import hat.callgraph.KernelCallGraph;
import jdk.incubator.code.dialect.core.CoreOp;
import optkl.util.carriers.LookupCarrier;

import java.lang.invoke.MethodHandles;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.function.Function;

public class HATTier implements  LookupCarrier {
    KernelCallGraph kernelCallGraph;
    @Override
    public MethodHandles.Lookup lookup(){
        return kernelCallGraph.lookup();
    }
    private List<HATPhase> hatPhases = new ArrayList<>();

    public HATTier(KernelCallGraph kernelCallGraph) {
        this.kernelCallGraph = kernelCallGraph;
        // barriers
        hatPhases.add(new HATBarrierPhase(kernelCallGraph));

        // array views
        hatPhases.add(new HATArrayViewPhase(kernelCallGraph));

        // Memory
        hatPhases.add(new HATMemoryPhase.LocalMemoryPhase(kernelCallGraph));
        hatPhases.add(new HATMemoryPhase.PrivateMemoryPhase(kernelCallGraph));
        hatPhases.add(new HATMemoryPhase.DeviceTypePhase(kernelCallGraph));

        // ID's /thread access
        hatPhases.add(new HATThreadsPhase.GlobalIdPhase(kernelCallGraph));
        hatPhases.add(new HATThreadsPhase.GlobalSizePhase(kernelCallGraph));
        hatPhases.add(new HATThreadsPhase.LocalIdPhase(kernelCallGraph));
        hatPhases.add(new HATThreadsPhase.LocalSizePhase(kernelCallGraph));
        hatPhases.add(new HATThreadsPhase.BlockPhase(kernelCallGraph));

        // views for vector types
        hatPhases.add(new HATVectorPhase.Float4LoadPhase(kernelCallGraph));
        hatPhases.add(new HATVectorPhase.Float2LoadPhase(kernelCallGraph));
        hatPhases.add(new HATVectorPhase.Float4OfPhase(kernelCallGraph));
        hatPhases.add(new HATVectorPhase.AddPhase(kernelCallGraph));
        hatPhases.add(new HATVectorPhase.SubPhase(kernelCallGraph));
        hatPhases.add(new HATVectorPhase.MulPhase(kernelCallGraph));
        hatPhases.add(new HATVectorPhase.DivPhase(kernelCallGraph));
        hatPhases.add(new HATVectorPhase.MakeMutable(kernelCallGraph));
        hatPhases.add(new HATVectorStorePhase.Float4StorePhase(kernelCallGraph));
        hatPhases.add(new HATVectorStorePhase.Float2StorePhase(kernelCallGraph));

        // Vector Select individual lines
        hatPhases.add(new HATVectorSelectPhase(kernelCallGraph));

        // F16 type
        hatPhases.add(new HATFP16Phase(kernelCallGraph));

    }

    // It computes a set of function code model transformations from FuncOp to FuncOp'.
    public CoreOp.FuncOp apply(CoreOp.FuncOp funcOp) {
        BlockingQueue<Function<CoreOp.FuncOp,CoreOp.FuncOp>> queue = new ArrayBlockingQueue<>(hatPhases.size());
        queue.addAll(hatPhases);

        CoreOp.FuncOp f = funcOp;
        while (!queue.isEmpty()) {
            try {
                // TODO Did we just trash side tables ?
                Function<CoreOp.FuncOp,CoreOp.FuncOp> phase = queue.take();
                f = phase.apply(f);
            } catch (InterruptedException e) {
                throw new RuntimeException(e);
            }
        }
        return f;
    }
}

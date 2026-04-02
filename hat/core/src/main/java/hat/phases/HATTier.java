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
import optkl.util.Mutable;
import optkl.util.carriers.LookupCarrier;

import java.lang.invoke.MethodHandles;
import java.util.List;

public class HATTier  {
    public static final  List<HATPhase> KernelPhases = List.of(
                // barrier
                new HATBarrierPhase(),
                // array views
                new HATArrayViewPhase(),
                // Memory
                new HATMemoryPhase.LocalMemoryPhase(),
                new HATMemoryPhase.PrivateMemoryPhase(),
                new HATMemoryPhase.DeviceTypePhase(),
                // ID's /thread access
                new HATThreadsPhase(),
                // MathLib phase
                new HATMathLibPhase(),
                // views for vector types
                new HATVectorPhase.Float4LoadPhase(),
                new HATVectorPhase.Float2LoadPhase(),
                new HATVectorPhase.Float4OfPhase(),
                new HATVectorPhase.AddPhase(),
                new HATVectorPhase.SubPhase(),
                new HATVectorPhase.MulPhase(),
                new HATVectorPhase.DivPhase(),
                new HATVectorPhase.MakeMutable(),
                new HATVectorStorePhase.Float4StorePhase(),
                new HATVectorStorePhase.Float2StorePhase(),
                // Vector Select individual lines
                new HATVectorSelectPhase(),
                // F16 type
                new HATFP16Phase()
        );

    public static CoreOp.FuncOp transform(List<HATPhase> phases, MethodHandles.Lookup lookup, CoreOp.FuncOp funcOp, boolean showCompilationPhases){
        var mf = Mutable.of(funcOp);
        phases.forEach(phase -> {
            if (showCompilationPhases) {
                System.out.println("Before PHASE" + phase.getClass().getSimpleName() + "\n" + mf.get().toText());
            }
            mf.set(phase.transform(lookup,mf.get()));
            if (showCompilationPhases) {
                System.out.println("After PHASE" + phase.getClass().getSimpleName() + "\n" + mf.get().toText());
            }
        });
        return mf.get();
    }
}

/*
 * Copyright (c) 2025-2026 Oracle and/or its affiliates. All rights reserved.
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

import optkl.util.carriers.FuncOpCarrier;

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

                new HATWarpSizePhase(),
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
                new HATFP16Phase(),

                // Tensors
                new HATTensorsPhase()
        );

    public static void transform(List<HATPhase> phases, MethodHandles.Lookup lookup, FuncOpCarrier funcOpCarrier, boolean showCompilationPhases){
        phases.forEach(phase -> {
            if (showCompilationPhases) {
                IO.println("Before PHASE" + phase.getClass().getSimpleName() + "\n" + funcOpCarrier.funcOp().toText());
            }
            funcOpCarrier.funcOp(phase.transform(lookup,funcOpCarrier.funcOp()));
            if (showCompilationPhases) {
                IO.println("After PHASE" + phase.getClass().getSimpleName() + "\n" + funcOpCarrier.funcOp().toText());
            }
        });
    }

    private HATTier() {
        /* This utility class should not be instantiated */
    }
}

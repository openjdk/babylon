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

import optkl.VarTable;
import optkl.util.carriers.FuncOpCarrier;

import java.lang.invoke.MethodHandles;
import java.util.List;

public class HATTier  {

    public static final  List<HATPhase> KernelPhases = List.of(
                // barrier
                new HATBarrierPhase(),   // Let's keep the dialect for barriers

                // array views
                new HATArrayViewPhase(),  // pending

                // Memory
                new HATMemoryPhase.LocalMemoryPhase(),  // custom nodes removed - OK
                new HATMemoryPhase.PrivateMemoryPhase(),// custom nodes removed - OK
                new HATMemoryPhase.DeviceTypePhase(),   // custom nodes removed - OK

                // ID's /thread access
                new HATThreadsPhase(),  // Let's keep the dialect for barriers

                // MathLib phase
                new HATMathLibPhase(), // custom nodes removed - OK

                // views for vector types
                new HATVectorPhase(), // Pending binOps

                // F16 type
                new HATFP16Phase()  // In progress
        );

    public static void transform(List<HATPhase> phases, MethodHandles.Lookup lookup, FuncOpCarrier funcOpCarrier, VarTable varTable, boolean showCompilationPhases){
        phases.forEach(phase -> {
            if (showCompilationPhases) {
                IO.println("Before PHASE" + phase.getClass().getSimpleName() + "\n" + funcOpCarrier.funcOp().toText());
            }
            funcOpCarrier.funcOp(phase.transform(lookup,funcOpCarrier.funcOp(), varTable));
            if (showCompilationPhases) {
                IO.println("After PHASE" + phase.getClass().getSimpleName() + "\n" + funcOpCarrier.funcOp().toText());
            }
        });
    }

    private HATTier() {
        /* This utility class should not be instantiated */
    }
}

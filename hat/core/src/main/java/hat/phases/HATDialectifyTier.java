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

import hat.Accelerator;
import jdk.incubator.code.dialect.core.CoreOp;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.function.Function;

public class HATDialectifyTier implements Function<CoreOp.FuncOp,CoreOp.FuncOp> {

    private List<HATDialect> hatPhases = new ArrayList<>();

    public HATDialectifyTier(Accelerator accelerator) {
        // barriers
        hatPhases.add(new HATDialectifyBarrierPhase(accelerator));
        // Memory
        hatPhases.add(new HATDialectifyMemoryPhase.SharedPhase(accelerator));
        hatPhases.add(new HATDialectifyMemoryPhase.PrivatePhase(accelerator));

        // ID's /thread access
        hatPhases.add(new HATDialectifyThreadsPhase.GlobalIdPhase(accelerator));
        hatPhases.add(new HATDialectifyThreadsPhase.GlobalSizePhase(accelerator));
        hatPhases.add(new HATDialectifyThreadsPhase.LocalIdPhase(accelerator));
        hatPhases.add(new HATDialectifyThreadsPhase.LocalSizePhase(accelerator));
        hatPhases.add(new HATDialectifyThreadsPhase.BlockPhase(accelerator));

        // views for vector types
        hatPhases.add(new HATDialectifyVectorOpPhase.Float4LoadPhase(accelerator));
        hatPhases.add(new HATDialectifyVectorOpPhase.Float4OfPhase(accelerator));
        hatPhases.add(new HATDialectifyVectorOpPhase.AddPhase(accelerator));
        hatPhases.add(new HATDialectifyVectorOpPhase.SubPhase(accelerator));
        hatPhases.add(new HATDialectifyVectorOpPhase.MulPhase(accelerator));
        hatPhases.add(new HATDialectifyVectorOpPhase.DivPhase(accelerator));
        hatPhases.add(new HATDialectifyVectorOpPhase.MakeMutable(accelerator));
        hatPhases.add(new HATDialectifyVectorStorePhase.Float4StorePhase(accelerator));

        // Vector Select individual lines
        hatPhases.add(new HATDialectifyVectorSelectPhase(accelerator));

        // F16 type
        hatPhases.add(new HATDialectifyFP16Phase(accelerator));

        hatPhases.add(new HATDialectifyArrayViewPhase(accelerator));
    }

    // It computes a set of function code model transformations from FuncOp to FuncOp'.
    @Override
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

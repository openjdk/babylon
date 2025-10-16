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
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;

public class HATDialectifyTier implements HATCompilationTier {

    private List<HATDialectifyPhase> hatPhases = new ArrayList<>();

    public HATDialectifyTier(Accelerator accelerator) {
        hatPhases.add(new HATDialectifyBarrierPhase(accelerator));
        Arrays.stream(HATDialectifyMemoryPhase.Space.values())
                .forEach(space -> hatPhases.add(new HATDialectifyMemoryPhase(accelerator,space)));
        Arrays.stream(HATDialectifyThreadsPhase.ThreadAccess.values())
                .forEach(threadAccess -> hatPhases.add(new HATDialectifyThreadsPhase(accelerator,threadAccess)));
        Arrays.stream(HATDialectifyVectorOpPhase.OpView.values())
                .forEach(vectorOperation -> hatPhases.add(new HATDialectifyVectorOpPhase(accelerator, vectorOperation)));
        Arrays.stream(HATDialectifyVectorStorePhase.StoreView.values())
                .forEach(vectorOperation -> hatPhases.add(new HATDialectifyVectorStorePhase(accelerator, vectorOperation)));
        hatPhases.add(new HATDialectifyVSelectPhase(accelerator));
    }

    // It computes a set of function code model transformations from FuncOp to FuncOp'.
    @Override
    public CoreOp.FuncOp run(CoreOp.FuncOp funcOp) {
        BlockingQueue<HATDialectifyPhase> queue = new ArrayBlockingQueue<>(hatPhases.size());
        queue.addAll(hatPhases);

        CoreOp.FuncOp f = funcOp;
        while (!queue.isEmpty()) {
            try {
                // TODO Did we just trash side tables ?
                HATDialectifyPhase phase = queue.take();
                f = phase.run(f);
            } catch (InterruptedException e) {
                throw new RuntimeException(e);
            }
        }
        return f;
    }
}

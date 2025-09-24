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

import jdk.incubator.code.dialect.core.CoreOp;

import java.lang.invoke.MethodHandles;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;

public class HatDialectifyTier {

    private CoreOp.FuncOp funcOp;
    List<HatDialectifyPhase> hatPhases = new ArrayList<>();

    public HatDialectifyTier(CoreOp.FuncOp funcOp, MethodHandles.Lookup lookup) {
        this.funcOp = funcOp;
        hatPhases.add(new HatDialectifyBarrierPhase());
        for (HatDialectifyMemoryPhase.Space space: HatDialectifyMemoryPhase.Space.values()) {
            hatPhases.add(new HatDialectifyMemoryPhase(space, lookup));
            hatPhases.add(new HatDialectifyMemoryPhase(space, lookup));
        }
        for (HatDilectifyThreadsPhase.ThreadAccess threadAccess: HatDilectifyThreadsPhase.ThreadAccess.values()) {
            hatPhases.add(new HatDilectifyThreadsPhase(threadAccess));
        }
    }

    public CoreOp.FuncOp run() {
        BlockingQueue<HatDialectifyPhase> queue = new ArrayBlockingQueue<>(hatPhases.size());
        queue.addAll(hatPhases);
        CoreOp.FuncOp f = funcOp;
        while (!queue.isEmpty()) {
            HatDialectifyPhase phase;
            try {
                phase = queue.take();
                f = phase.run(f);
            } catch (InterruptedException e) {
                throw new RuntimeException(e);
            }
        }
        return f;
    }
}

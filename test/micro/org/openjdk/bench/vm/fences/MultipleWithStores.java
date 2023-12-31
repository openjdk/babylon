/*
 * Copyright (c) 2021, Red Hat, Inc. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.
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

package org.openjdk.bench.vm.fences;

import org.openjdk.jmh.annotations.*;

import java.lang.invoke.VarHandle;
import java.util.concurrent.TimeUnit;

@Warmup(iterations = 3, time = 1, timeUnit = TimeUnit.SECONDS)
@Measurement(iterations = 3, time = 1, timeUnit = TimeUnit.SECONDS)
@Fork(3)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.NANOSECONDS)
@State(Scope.Thread)
public class MultipleWithStores {

    int x, y, z;

    @Benchmark
    public void plain() {
        x = 1;
        y = 1;
        z = 1;
    }

    @Benchmark
    public void loadLoad() {
        VarHandle.loadLoadFence();
        x = 1;
        VarHandle.loadLoadFence();
        y = 1;
        VarHandle.loadLoadFence();
        z = 1;
        VarHandle.loadLoadFence();
    }

    @Benchmark
    public void storeStore() {
        VarHandle.storeStoreFence();
        x = 1;
        VarHandle.storeStoreFence();
        y = 1;
        VarHandle.storeStoreFence();
        z = 1;
        VarHandle.storeStoreFence();
    }

    @Benchmark
    public void acquire() {
        VarHandle.releaseFence();
        x = 1;
        VarHandle.releaseFence();
        y = 1;
        VarHandle.releaseFence();
        z = 1;
        VarHandle.releaseFence();
    }

    @Benchmark
    public void release() {
        VarHandle.releaseFence();
        x = 1;
        VarHandle.releaseFence();
        y = 1;
        VarHandle.releaseFence();
        z = 1;
        VarHandle.releaseFence();
    }

    @Benchmark
    public void full() {
        VarHandle.fullFence();
        x = 1;
        VarHandle.fullFence();
        y = 1;
        VarHandle.fullFence();
        z = 1;
        VarHandle.fullFence();
    }

}

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

package hat.backend;

import hat.Accelerator;
import hat.KernelContext;
import hat.NDRange;
import hat.callgraph.KernelCallGraph;
import hat.callgraph.KernelEntrypoint;

import java.lang.reflect.InvocationTargetException;
import java.util.Arrays;

public class JavaMultiThreadedBackend extends JavaBackend {


    @Override
    public void dispatchKernel(KernelCallGraph kernelCallGraph, NDRange ndRange, Object... args) {
        KernelEntrypoint kernelEntrypoint = kernelCallGraph.entrypoint;
        instance(ndRange.accelerator).forEachInRange(ndRange, (range) -> {
            Object[] a = Arrays.copyOf(args, args.length); // Annoying.  we need to replace the args[0] but don't want to race other threads.
            try {
                KernelContext c = range.kid;
                a[0] = c;
                kernelEntrypoint.method.invoke(null, a);
            } catch (IllegalAccessException e) {
                throw new RuntimeException(e);
            } catch (InvocationTargetException e) {
                throw new RuntimeException(e);
            }

        });
    }

    WorkStealer workStealer = null;

    synchronized WorkStealer instance(Accelerator accelerator) {
        if (workStealer == null) {
            workStealer = WorkStealer.usingAllProcessors(accelerator);
        }
        return workStealer;
    }


}

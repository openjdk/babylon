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
package wrap.clwrap;

import wrap.Wrap;

import java.io.IOException;
import java.lang.foreign.Arena;

public class Squares {


    public static void main(String[] args) throws IOException {
        try (var arena = Arena.ofConfined()) {
            CLPlatform.CLDevice[] selectedDevice = new CLPlatform.CLDevice[1];
            CLPlatform.platforms(arena).forEach(platform -> {
                System.out.println("Platform Name " + platform.platformName());
                platform.devices.forEach(device -> {
                    System.out.println("   Compute Units     " + device.computeUnits());
                    System.out.println("   Device Name       " + device.deviceName());
                    System.out.println("   Built In Kernels  " + device.builtInKernels());
                    selectedDevice[0] = device;
                });
            });
            var context = selectedDevice[0].createContext();
            var program = context.buildProgram("""
                    __kernel void squares(__global int* in,__global int* out ){
                        int gid = get_global_id(0);
                        out[gid] = in[gid]*in[gid];
                    }
                    """);
            var kernel = program.getKernel("squares");
            var in = Wrap.IntArr.of(arena,512);
            var out = Wrap.IntArr.of(arena,512);
            for (int i = 0; i < 512; i++) {
                in.set(i,i);
            }
            ComputeContext computeContext = new ComputeContext(arena,20);
            var inMem = computeContext.register(in.ptr());
            var outMem = computeContext.register(out.ptr());

            kernel.run(computeContext,512, inMem, outMem);
            for (int i = 0; i < 512; i++) {
                System.out.println(i + " " + out.get(i));
            }
        }
    }

}

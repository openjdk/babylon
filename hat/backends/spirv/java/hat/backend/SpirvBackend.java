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



import hat.ComputeContext;
import hat.callgraph.KernelCallGraph;
import hat.NDRange;
import intel.code.spirv.SpirvModuleGenerator;
import intel.code.spirv.SpirvOps;
import intel.code.spirv.TranslateToSpirvModel;

import java.lang.foreign.MemorySegment;

public class SpirvBackend extends NativeBackend {
    public SpirvBackend()  {
        super("spirv_backend");
        getBackend(null);
    }


    @Override
    public void computeContextHandoff(ComputeContext computeContext){
        System.out.println("Spirv backend recieved closed closure");
        System.out.println("Spirv backend will mutate  "+ computeContext.computeCallGraph.entrypoint + computeContext.computeCallGraph.entrypoint.method);
        injectBufferTracking(computeContext.computeCallGraph.entrypoint);
        boolean doSpirv= false;
        if (doSpirv) {
            TranslateToSpirvModel translateToSpirvModel = new TranslateToSpirvModel();
            computeContext.computeCallGraph.kernelCallGraphStream().forEach(kernelCallGraph -> {
                kernelCallGraph.kernelReachableResolvedStream().forEach(kr -> {
                    String methodName = kr.method.getName();

                    SpirvOps.FuncOp spirvFunc = TranslateToSpirvModel.translateFunction(kr.funcOpWrapper().op());
                    MemorySegment spirvBinary = SpirvModuleGenerator.generateModule(methodName, spirvFunc);

                    System.out.println("\n------- Java Model -------");
                    System.out.println(kr.funcOpWrapper().op().toText());
                    System.out.println("------- SPIR-V Model -------");
                    System.out.println(spirvFunc.toText());
                    System.out.println("------- SPIR-V Module -------");
                    System.out.println(SpirvModuleGenerator.disassembleModule(spirvBinary));
                });

            });

        }

      //  var codeBuilder = new SpirvCodeReflectionBuilder();
      //  Code code = createCode(computeContext, codeBuilder);
       // System.out.println(codeBuilder);
    }
    @Override
    public void dispatchKernel(KernelCallGraph kernelCallGraph, NDRange ndRange, Object... args) {
        System.out.println("implement spirv dispatch kernel");
    }
}

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

package hat.backend.jextracted;

import hat.ComputeContext;
import hat.Config;
import jdk.incubator.code.CodeTransformer;
import hat.callgraph.CallGraph;
import jdk.incubator.code.bytecode.BytecodeGenerator;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.interpreter.Interpreter;

import java.lang.foreign.Arena;
import java.lang.invoke.MethodHandles;

public abstract class JExtractedBackend extends JExtractedBackendDriver {

    public JExtractedBackend(Arena arena, MethodHandles.Lookup lookup,Config config, String libName) {
        super(arena,lookup,config,libName);
    }

    public void dispatchCompute(ComputeContext computeContext, Object... args) {
        if (computeContext.computeEntrypoint().lowered == null) {
            computeContext.computeEntrypoint().lowered =
                    computeContext.computeEntrypoint().funcOp().transform(CodeTransformer.LOWERING_TRANSFORMER);
        }
        boolean interpret = false;
        if (interpret) {
            Interpreter.invoke(computeContext.lookup(), computeContext.computeEntrypoint().lowered, args);
        } else {
            try {
                if (computeContext.computeEntrypoint().mh == null) {
                    computeContext.computeEntrypoint().mh = BytecodeGenerator.generate(computeContext.lookup(), computeContext.computeEntrypoint().lowered);
                }
                computeContext.computeEntrypoint().mh.invokeWithArguments(args);
            } catch (Throwable e) {
                System.out.println(computeContext.computeEntrypoint().lowered.toText());
                throw new RuntimeException(e);
            }
        }
    }

    // This code should be common with ffi-shared probably should be pushed down into another lib?
    protected static CoreOp.FuncOp injectBufferTracking(CallGraph.ResolvedMethodCall computeMethod) {
      //  System.out.println("COMPUTE entrypoint before injecting buffer tracking...");
       // System.out.println(computeMethod.funcOp().toText());
        throw new RuntimeException("implement inject buffer tracking ");
        //return transformedFuncOp;
    }
}

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

package hat.backend.ffi;

import hat.ComputeContext;
import hat.Config;
import jdk.incubator.code.CodeTransformer;
import jdk.incubator.code.bytecode.BytecodeGenerator;
import jdk.incubator.code.interpreter.Interpreter;

import java.lang.foreign.Arena;
import java.lang.invoke.MethodHandles;

public abstract class FFIBackend extends FFIBackendDriver {

    public FFIBackend(Arena arena, MethodHandles.Lookup lookup, String libName, Config config) {
        super(arena, lookup, libName, config);
    }

    public void dispatchCompute(ComputeContext computeContext, Object... args) {
        if (computeContext.computeEntrypoint().lowered == null) {
            computeContext.computeEntrypoint().lowered =
                    computeContext.computeEntrypoint().funcOp().transform(CodeTransformer.LOWERING_TRANSFORMER);
        }
        backendBridge.computeStart();
        if (config().interpret()) {
            Interpreter.invoke(computeContext.lookup(), computeContext.computeEntrypoint().lowered, args);
        } else {
            try {
                if (computeContext.computeEntrypoint().mh == null) {
                    computeContext.computeEntrypoint().mh = BytecodeGenerator.generate(computeContext.lookup(), computeContext.computeEntrypoint().lowered);
                }
                computeContext.computeEntrypoint().mh.invokeWithArguments(args);
            } catch (Throwable e) {
                throw new RuntimeException(e);
            }
        }
        backendBridge.computeEnd();
    }


}

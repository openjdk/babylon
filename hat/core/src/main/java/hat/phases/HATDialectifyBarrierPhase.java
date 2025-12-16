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
import hat.dialect.HATBarrierOp;
import hat.optools.OpTk;
import jdk.incubator.code.dialect.core.CoreOp;

public record HATDialectifyBarrierPhase(Accelerator accelerator) implements HATDialect {

    @Override
    public CoreOp.FuncOp apply(CoreOp.FuncOp fromFuncOp) {
        var here = OpTk.CallSite.of(HATDialectifyBarrierPhase.class, "apply");
        before(here, fromFuncOp);
        // The resulting op map also includes all op mappings (so op -> op') and the to and from funcOp
        // I expect this to be useful for tracking state...

        OpTk.OpMap opMap = OpTk.simpleOpMappingTransform(
                /* for debugging we will remove */ here, fromFuncOp,
                /* filter op                    */ ce -> OpTk.isKernelContextInvokeOp(accelerator.lookup, ce,
                                                    invokeOp->invokeOp.invokeDescriptor().name().equals(HATBarrierOp.INTRINSIC_NAME)),
                /* replace op                   */ HATBarrierOp::new
        );
        after(here, opMap.toFuncOp());
        return opMap.toFuncOp();
    }

}

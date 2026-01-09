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


import hat.KernelContext;
import hat.callgraph.KernelCallGraph;
import hat.dialect.HATThreadOp;
import jdk.incubator.code.CodeElement;
import optkl.Trxfmr;

import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.JavaOp;
import optkl.util.Regex;

import java.util.HashSet;
import java.util.Set;

import static optkl.OpHelper.Named.NamedStaticOrInstance.FieldAccess;
import static optkl.OpHelper.Named.NamedStaticOrInstance.FieldAccess.fieldAccess;
import static optkl.OpHelper.Named.VarAccess;

public record HATThreadsPhase(KernelCallGraph kernelCallGraph) implements HATPhase {

    private static final Regex allfieldNameRegex = Regex.of("[glb][si]([xyz])");

    @Override
    public CoreOp.FuncOp apply(CoreOp.FuncOp funcOp) {
        Set<CodeElement<?, ?>> removeMe = new HashSet<>();
        return Trxfmr.of(this, funcOp)
                .transform(ce -> ce instanceof JavaOp.FieldAccessOp, c -> { // We care about field accesses
                    if (fieldAccess(lookup(), c.op()) instanceof FieldAccess fieldAccess // get a FieldAccessHelper
                            && fieldAccess.refType(KernelContext.class)
                            && fieldAccess.isLoad()
                            && fieldAccess.named(allfieldNameRegex)
                            && fieldAccess.instanceVarAccess() instanceof VarAccess varAccess) {
                        removeMe.add(varAccess.op());// We will remove in the next transform (see removeme)
                        c.replace(HATThreadOp.create(fieldAccess.name()));
                    }
                })
                .remap(removeMe)
                .remove(removeMe::contains)
                .funcOp();
    }
}

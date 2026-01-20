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



import hat.callgraph.KernelCallGraph;
import hat.dialect.HATThreadOp;
import jdk.incubator.code.CodeElement;
import optkl.OpHelper.FieldAccess;
import optkl.Trxfmr;

import jdk.incubator.code.dialect.core.CoreOp;
import hat.phases.KernelContextThreadIdFieldAccessQuery.Match;

import java.util.HashSet;
import java.util.Set;

public record HATThreadsPhase(KernelCallGraph kernelCallGraph) implements HATPhase {
    @Override
    public CoreOp.FuncOp apply(CoreOp.FuncOp funcOp) {
        Set<CodeElement<?, ?>> varAccessesToBeRemoved = new HashSet<>();
        var query = KernelContextThreadIdFieldAccessQuery.create(lookup()); // This Query matches kc->[glb][is][xyz] calls
        return Trxfmr.of(this, funcOp)
                .transform( c -> {
                    if (query.matches(c) instanceof Match match && match.helper() instanceof FieldAccess fieldAccess){
                        varAccessesToBeRemoved.add(fieldAccess.instanceVarAccess().op());  // the var access will be removed the next transform
                        c.replace(switch (fieldAccess.name()){
                            case "gix"->  new HATThreadOp.HAT_GI.HAT_GIX();
                            case "giy"->  new HATThreadOp.HAT_GI.HAT_GIY();
                            case "giz"->  new HATThreadOp.HAT_GI.HAT_GIZ();
                            case "gsx"->  new HATThreadOp.HAT_GS.HAT_GSX();
                            case "gsy"->  new HATThreadOp.HAT_GS.HAT_GSY();
                            case "gsz"->  new HATThreadOp.HAT_GS.HAT_GSZ();
                            case "lix"->  new HATThreadOp.HAT_LI.HAT_LIX();
                            case "liy"->  new HATThreadOp.HAT_LI.HAT_LIY();
                            case "liz"->  new HATThreadOp.HAT_LI.HAT_LIZ();
                            case "lsx"->  new HATThreadOp.HAT_LS.HAT_LSX();
                            case "lsy"->  new HATThreadOp.HAT_LS.HAT_LSY();
                            case "lsz"->  new HATThreadOp.HAT_LS.HAT_LSZ();
                            case "bix"->  new HATThreadOp.HAT_BI.HAT_BIX();
                            case "biy"->  new HATThreadOp.HAT_BI.HAT_BIY();
                            case "biz"->  new HATThreadOp.HAT_BI.HAT_BIZ();
                            default -> throw  new RuntimeException("what is this ?");
                        });
                    }
                })
                .remap(varAccessesToBeRemoved)                // after this transform this set needs to be replaced with new op references
                .remove(varAccessesToBeRemoved::contains)     // now we can transform again and remove everything in the remove me set
                .funcOp();
    }
}

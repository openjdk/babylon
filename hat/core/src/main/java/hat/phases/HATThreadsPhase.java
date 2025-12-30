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
import hat.optools.KernelContextPattern;
import optkl.Trxfmr;

import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.JavaOp;
import optkl.util.CallSite;
import optkl.OpTkl;
import optkl.util.Regex;

import java.util.Objects;
import static optkl.OpTkl.operandsAsResults;

public sealed abstract class HATThreadsPhase implements HATPhase
permits HATThreadsPhase.BlockPhase, HATThreadsPhase.GlobalIdPhase, HATThreadsPhase.GlobalSizePhase, HATThreadsPhase.LocalIdPhase, HATThreadsPhase.LocalSizePhase {
    private final KernelCallGraph kernelCallGraph;
    @Override public KernelCallGraph kernelCallGraph(){
        return kernelCallGraph;
    }

    public HATThreadsPhase(KernelCallGraph kernelCallGraph) {
        this.kernelCallGraph=kernelCallGraph;
    }

    private static final Regex localSizeRegex = Regex.of("ls([xyz])");
    private static final Regex localIdRegex = Regex.of("li([xyz])");
    private static final Regex globalSzRegex = Regex.of("(gs[xyz])");
    private static final Regex blockIdRegex = Regex.of("bi([xyz])");
    private static final Regex globalIdxRegex = Regex.of("(gi[xyz])");
    @Override
    public CoreOp.FuncOp apply(CoreOp.FuncOp funcOp) {
        var txfmr = new Trxfmr(CallSite.of(this.getClass()),funcOp);
        Regex fieldNameRegex  = switch (this){
            case BlockPhase _->blockIdRegex;
            case GlobalIdPhase _->globalIdxRegex;
            case GlobalSizePhase _->globalSzRegex;
            case LocalIdPhase _-> localIdRegex;
            case LocalSizePhase _-> localSizeRegex;
        };
        return txfmr.select(
                ce-> KernelContextPattern.KernelContextFieldAccessPattern.asKernelContextFieldAccessOrNull(
                        lookup(),ce,fieldAccessOp->fieldNameRegex.matches(fieldAccessOp.fieldDescriptor().name()))!=null,(s, o)->
                   operandsAsResults(o)
                     .map(OpTkl::opOfResultOrNull)
                     .map(OpTkl::asVarLoadOrNull)
                     .filter(Objects::nonNull) // ((Result)operand).op()) instanceof VarLoad varload && varload is KernelContext.class
                     .findFirst()
                     .ifPresent(varLoadOp -> s.select(o,varLoadOp))
                ).transform(txfmr.selected::contains, c->{
                   switch (c.op()){
                      case JavaOp.FieldAccessOp.FieldLoadOp $  -> {
                          String name = $.fieldDescriptor().name();
                          int dimIdx = name.length()==3 ?name.charAt(2)-'x' :-1;
                          if (dimIdx <0||dimIdx>3){
                              throw new IllegalStateException();//'x'=1,'y'=2....
                          }
                          c.replace(switch (HATThreadsPhase.this){
                              case BlockPhase _-> HATThreadOp.HATBlockThreadIdOp.of(dimIdx, $.resultType());
                              case GlobalIdPhase _-> HATThreadOp.HATGlobalThreadIdOp.of(dimIdx, $.resultType());
                              case GlobalSizePhase _-> HATThreadOp.HATGlobalSizeOp.of(dimIdx, $.resultType());
                              case LocalIdPhase _-> HATThreadOp.HATLocalThreadIdOp.of(dimIdx, $.resultType());
                              case LocalSizePhase _-> HATThreadOp.HATLocalSizeOp.of(dimIdx,$.resultType());
                          });
                      }
                      case CoreOp.VarAccessOp.VarLoadOp _ -> c.remove();
                      default -> {}
                }
        }).funcOp();
    }

    public static final class BlockPhase extends HATThreadsPhase {
        public BlockPhase(KernelCallGraph kernelCallGraph) {
            super(kernelCallGraph);
        }
    }

    public static final class GlobalIdPhase extends HATThreadsPhase {
        public GlobalIdPhase(KernelCallGraph kernelCallGraph) {
            super(kernelCallGraph);
        }
    }

    public static final class GlobalSizePhase extends HATThreadsPhase {
        public GlobalSizePhase(KernelCallGraph kernelCallGraph) {
            super(kernelCallGraph);
        }
    }

    public static final class LocalIdPhase extends HATThreadsPhase {
        public LocalIdPhase(KernelCallGraph kernelCallGraph) {
            super(kernelCallGraph);
        }
    }

    public static final class LocalSizePhase extends HATThreadsPhase {
        public LocalSizePhase(KernelCallGraph kernelCallGraph) {
            super(kernelCallGraph);
        }
    }
}

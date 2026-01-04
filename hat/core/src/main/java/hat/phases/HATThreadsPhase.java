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
import optkl.FieldAccess;
import optkl.Trxfmr;

import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.JavaOp;
import optkl.OpTkl;
import optkl.util.Regex;

import java.util.HashSet;
import java.util.Objects;
import java.util.Set;

import static optkl.FieldAccess.fieldAccessOpHelper;
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
        Regex fieldNameRegex  = switch (this){
            case BlockPhase _->blockIdRegex;
            case GlobalIdPhase _->globalIdxRegex;
            case GlobalSizePhase _->globalSzRegex;
            case LocalIdPhase _-> localIdRegex;
            case LocalSizePhase _-> localSizeRegex;
        };
        Set<CodeElement<?,?>> removeMe= new HashSet<>();
        return new Trxfmr(funcOp).transform(ce->ce instanceof JavaOp.FieldAccessOp, c->{
                    if (fieldAccessOpHelper(lookup(),c.op()) instanceof FieldAccess fieldAccess
                            && fieldAccess.refType(KernelContext.class)
                            && fieldAccess.op() instanceof JavaOp.FieldAccessOp.FieldLoadOp
                            && fieldAccess.named(fieldNameRegex)) {
                        operandsAsResults(fieldAccess.op())
                                .map(OpTkl::opOfResultOrNull)
                                .map(OpTkl::asVarLoadOrNull)
                                .filter(Objects::nonNull)
                                .findFirst()
                                .ifPresent(varLoadOp -> {
                                    removeMe.add(varLoadOp); // We will need to remove this
                                    int dimIdx = fieldAccess.name().charAt(2) - 'x';
                                    c.replace(switch (HATThreadsPhase.this) {
                                        case BlockPhase _ -> HATThreadOp.HATBlockThreadIdOp.of(dimIdx, fieldAccess.resultType());
                                        case GlobalIdPhase _ -> HATThreadOp.HATGlobalThreadIdOp.of(dimIdx, fieldAccess.resultType());
                                        case GlobalSizePhase _ -> HATThreadOp.HATGlobalSizeOp.of(dimIdx, fieldAccess.resultType());
                                        case LocalIdPhase _ -> HATThreadOp.HATLocalThreadIdOp.of(dimIdx, fieldAccess.resultType());
                                        case LocalSizePhase _ -> HATThreadOp.HATLocalSizeOp.of(dimIdx, fieldAccess.resultType());
                                    });
                                });
                    }
                })
                .remap(removeMe)
                .remove(removeMe::contains)
                .funcOp();
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

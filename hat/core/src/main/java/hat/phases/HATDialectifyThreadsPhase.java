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
import hat.dialect.HATBlockThreadIdOp;
import hat.dialect.HATGlobalSizeOp;
import hat.dialect.HATGlobalThreadIdOp;
import hat.dialect.HATLocalSizeOp;
import hat.dialect.HATLocalThreadIdOp;
import hat.dialect.HATThreadOp;
import hat.optools.Trxfmr;
import hat.optools.OpTk;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.JavaOp;
import optkl.CallSite;
import optkl.OpTkl;
import optkl.Regex;

import java.util.Objects;

import static optkl.OpTkl.operandsAsResults;

public sealed abstract class HATDialectifyThreadsPhase<T extends HATDialectifyThreadsPhase<T,C>,C extends HATThreadOp> implements HATDialectPhase {
    private final KernelCallGraph kernelCallGraph;
    @Override public KernelCallGraph kernelCallGraph(){
        return kernelCallGraph;
    }
    final Class<C> clazz;

    public HATDialectifyThreadsPhase(KernelCallGraph kernelCallGraph, Class<C> clazz) {
        this.kernelCallGraph=kernelCallGraph;
        this.clazz=clazz;
    }

    protected abstract  HATThreadOp factory(JavaOp.FieldAccessOp.FieldLoadOp fieldLoadOp);

    protected abstract Regex regex();

    @Override
    public CoreOp.FuncOp apply(CoreOp.FuncOp funcOp) {
        var txfmr = new Trxfmr(CallSite.of(this.getClass()),funcOp);
        return txfmr.select(
                ce->OpTk.asNamedKernelContextFieldAccessOrNull(lookup(),ce,regex())!=null,(s,o)->
                   operandsAsResults(o)
                     .map(OpTkl::opOfResultOrNull)
                     .map(OpTkl::asVarLoadOrNull)
                     .filter(Objects::nonNull) // ((Result)operand).op()) instanceof VarLoad varload && varload is KernelContext.class
                     .findFirst()
                     .ifPresent(varLoadOp -> s.select(o,varLoadOp))
                ).transform(txfmr.selected::contains, c->{
                   switch (c.op()){
                      case JavaOp.FieldAccessOp.FieldLoadOp $  -> c.replace(factory($));
                      case CoreOp.VarAccessOp.VarLoadOp _ -> c.remove();
                      default -> {}
                }
        }).funcOp();
    }

    public static final class BlockPhase extends HATDialectifyThreadsPhase<BlockPhase,HATBlockThreadIdOp> {
        public BlockPhase(KernelCallGraph kernelCallGraph) {
            super(kernelCallGraph, HATBlockThreadIdOp.class);
        }
        @Override protected Regex regex(){
            return HATBlockThreadIdOp.regex;
        }

        @Override
        public HATThreadOp factory(JavaOp.FieldAccessOp.FieldLoadOp fieldLoadOp){
                return HATBlockThreadIdOp.of(OpTk.dimIdx(fieldLoadOp), fieldLoadOp.resultType());
        }
    }

    public static final class GlobalIdPhase extends HATDialectifyThreadsPhase<GlobalIdPhase,HATGlobalThreadIdOp>  {
        public GlobalIdPhase(KernelCallGraph kernelCallGraph) {
            super(kernelCallGraph, HATGlobalThreadIdOp.class);
        }
        @Override protected Regex regex(){
            return HATGlobalThreadIdOp.regex;
        }
        @Override
        public HATThreadOp factory(JavaOp.FieldAccessOp.FieldLoadOp fieldLoadOp){
                return HATGlobalThreadIdOp.of(OpTk.dimIdx(fieldLoadOp), fieldLoadOp.resultType());
        }
    }

    public static final class GlobalSizePhase extends HATDialectifyThreadsPhase<GlobalSizePhase,HATGlobalSizeOp>  {
        public GlobalSizePhase(KernelCallGraph kernelCallGraph) {
            super(kernelCallGraph, HATGlobalSizeOp.class);
        }
        @Override protected Regex regex(){
            return HATGlobalSizeOp.regex;
        }
        @Override
        public HATThreadOp factory(JavaOp.FieldAccessOp.FieldLoadOp fieldLoadOp){
                return  HATGlobalSizeOp.of(OpTk.dimIdx(fieldLoadOp), fieldLoadOp.resultType());
        }
    }

    public static final class LocalIdPhase extends HATDialectifyThreadsPhase<LocalIdPhase,HATLocalThreadIdOp>  {
        public LocalIdPhase(KernelCallGraph kernelCallGraph) {
            super(kernelCallGraph,HATLocalThreadIdOp.class);
        }
        @Override protected Regex regex(){
            return HATLocalThreadIdOp.regex;
        }
        @Override
        public HATThreadOp factory(JavaOp.FieldAccessOp.FieldLoadOp fieldLoadOp){
                return HATLocalThreadIdOp.of(OpTk.dimIdx(fieldLoadOp), fieldLoadOp.resultType());
        }
    }

    public static final class LocalSizePhase extends HATDialectifyThreadsPhase<LocalSizePhase,HATLocalSizeOp>  {
        public LocalSizePhase(KernelCallGraph kernelCallGraph) {
            super(kernelCallGraph,HATLocalSizeOp.class);
        }
        @Override public Regex regex(){
           return HATLocalSizeOp.regex;
        }
        @Override
        public HATThreadOp factory(JavaOp.FieldAccessOp.FieldLoadOp fieldLoadOp){
            return HATLocalSizeOp.of(OpTk.dimIdx(fieldLoadOp), fieldLoadOp.resultType());
        }
    }
}

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
import hat.KernelContext;
import hat.dialect.HATBlockThreadIdOp;
import hat.dialect.HATGlobalSizeOp;
import hat.dialect.HATGlobalThreadIdOp;
import hat.dialect.HATLocalSizeOp;
import hat.dialect.HATLocalThreadIdOp;
import hat.dialect.HATThreadOp;
import hat.optools.Trxfmr;
import hat.optools.OpTk;
import jdk.incubator.code.CodeContext;
import jdk.incubator.code.CodeElement;
import jdk.incubator.code.Op;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.JavaOp;

import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.Objects;
import java.util.Set;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public abstract class HATDialectifyThreadsPhase implements HATDialect  {
    protected final Accelerator accelerator;
    @Override  public Accelerator accelerator(){
        return this.accelerator;
    }

    public HATDialectifyThreadsPhase(Accelerator accelerator) {
        this.accelerator=accelerator;
    }

    protected abstract  HATThreadOp factory(JavaOp.FieldAccessOp.FieldLoadOp fieldLoadOp);

    protected abstract Pattern pattern();

    @Override

    public CoreOp.FuncOp apply(CoreOp.FuncOp funcOp) {
        var here = OpTk.CallSite.of(this.getClass(), "apply");
        before(here, funcOp);
        Set<Op> ops = new LinkedHashSet<>();
        funcOp.elements()
                .map(ce -> OpTk.asNamedKernelContextFieldAccessOrNull(accelerator.lookup,ce,pattern()))
                .filter(Objects::nonNull)
                .forEach(fieldLoadOp -> fieldLoadOp.operands().stream()
                    .map(OpTk::asResultOrNull)
                    .map(OpTk::opOfResultOrNull)
                    .map(OpTk::asVarLoadOrNull)
                    .filter(Objects::nonNull) // ((Result)operand).op()) instanceof VarLoad varload && varload is KernelContext.class
                    .findFirst()
                    .ifPresent(varLoadOp -> ops.addAll(Set.of(fieldLoadOp,varLoadOp)))
                );

        funcOp = OpTk.transform(here, funcOp, ops::contains, (bb, op) -> {
            CodeContext ctx = bb.context();
            switch (op){
                case CoreOp.VarAccessOp.VarLoadOp  $->
                        ctx.mapValue($.result(), ctx.getValue($.operands().getFirst()));
                case JavaOp.FieldAccessOp.FieldLoadOp $->
                        ctx.mapValue($.result(), bb.op(OpTk.copyLocation($,factory($))).op().result());
                default -> throw new RuntimeException("We should never get here");
            }
            return bb;
        });
        after(here,funcOp);
        return funcOp;
    }
    public CoreOp.FuncOp applyTxform(CoreOp.FuncOp funcOp) {
        return new Trxfmr.Edge.Selector<JavaOp.FieldAccessOp.FieldLoadOp, CoreOp.VarAccessOp.VarLoadOp>()
        .select(funcOp, ce->ce instanceof JavaOp.FieldAccessOp.FieldLoadOp fieldLoadOp
                        && Trxfmr.Edge.kernelContextFieldVarLoad(accelerator.lookup,fieldLoadOp, fieldName->pattern().matcher(fieldName).matches())
                        instanceof Trxfmr.Edge<JavaOp.FieldAccessOp.FieldLoadOp,CoreOp.VarAccessOp.VarLoadOp> e? e:null)
        .transform(funcOp,c->{
            switch (c.op()){
                case JavaOp.FieldAccessOp.FieldLoadOp $  -> c.replace(factory($));
                case CoreOp.VarAccessOp.VarLoadOp _ -> c.remove();
                default -> {}
            }
        });
    }


    //private boolean isMethodFromHatKernelContext(CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
     //   String kernelContextCanonicalName = hat.KernelContext.class.getName();
      //  return varLoadOp.resultType().toString().equals(kernelContextCanonicalName);
   // }


    public static class BlockPhase extends HATDialectifyThreadsPhase  {
        public BlockPhase(Accelerator accelerator) {
            super(accelerator);
        }
        @Override protected Pattern pattern(){
            return Pattern.compile("bi[xyz]");
        }

        @Override
        public HATThreadOp factory(JavaOp.FieldAccessOp.FieldLoadOp fieldLoadOp){
                return new HATBlockThreadIdOp(switch (fieldLoadOp.fieldDescriptor().name()){
                    case "biy"->1;
                    case "biz"->2;
                    default -> 0;
                }, fieldLoadOp.resultType());
        }
    }

    public static class GlobalIdPhase extends HATDialectifyThreadsPhase  {

        public GlobalIdPhase(Accelerator accelerator) {
            super(accelerator);
        }
        @Override protected Pattern pattern(){
            return Pattern.compile("(gi[xyz])");
        }
        @Override
        public HATThreadOp factory(JavaOp.FieldAccessOp.FieldLoadOp fieldLoadOp){
                return new HATGlobalThreadIdOp(switch (fieldLoadOp.fieldDescriptor().name()){
                    case "y", "giy"->1;
                    case "z", "giz"->2;
                    default -> 0;
                }, fieldLoadOp.resultType());
        }
    }

    public static class GlobalSizePhase extends HATDialectifyThreadsPhase  {
        public GlobalSizePhase(Accelerator accelerator) {
            super(accelerator);
        }
        @Override protected Pattern pattern(){
            return Pattern.compile("(gs[xyz])");
        }
        @Override
        public HATThreadOp factory(JavaOp.FieldAccessOp.FieldLoadOp fieldLoadOp){
                return new HATGlobalSizeOp(switch (fieldLoadOp.fieldDescriptor().name()){
                    case "gsy","maxY"->1;
                    case "gsz","maxZ"->2;
                    default -> 0;
                }, fieldLoadOp.resultType());
        }


    }

    public static class LocalIdPhase extends HATDialectifyThreadsPhase  {
        public LocalIdPhase(Accelerator accelerator) {
            super(accelerator);
        }
        @Override protected Pattern pattern(){
            return  Pattern.compile("li([xyz])");
        }
        @Override
        public HATThreadOp factory(JavaOp.FieldAccessOp.FieldLoadOp fieldLoadOp){
                return new HATLocalThreadIdOp(switch (fieldLoadOp.fieldDescriptor().name()){
                    case "liy"->1;
                    case "liz"->2;
                    default -> 0;
                }, fieldLoadOp.resultType());
        }
    }

    public static class LocalSizePhase extends HATDialectifyThreadsPhase  {
        public LocalSizePhase(Accelerator accelerator) {
            super(accelerator);
        }
        @Override protected Pattern pattern(){
            return  Pattern.compile("ls([xyz])");
        }
        @Override
        public HATThreadOp factory(JavaOp.FieldAccessOp.FieldLoadOp fieldLoadOp){
                return new HATLocalSizeOp(switch (fieldLoadOp.fieldDescriptor().name()){
                    case "lsy"->1;
                    case "lsz"->2;
                    default -> 0;
                }, fieldLoadOp.resultType());
        }
    }
}

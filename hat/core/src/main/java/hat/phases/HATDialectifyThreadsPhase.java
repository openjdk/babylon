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
import hat.dialect.HATBlockThreadIdOp;
import hat.dialect.HATGlobalSizeOp;
import hat.dialect.HATGlobalThreadIdOp;
import hat.dialect.HATLocalSizeOp;
import hat.dialect.HATLocalThreadIdOp;
import hat.dialect.HATThreadOp;
import hat.optools.OpTk;
import jdk.incubator.code.CodeElement;
import jdk.incubator.code.CopyContext;
import jdk.incubator.code.Op;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.JavaOp;

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

        Stream<CodeElement<?, ?>> elements = funcOp.elements()
                .filter(codeElement ->codeElement instanceof JavaOp.FieldAccessOp.FieldLoadOp)
                .map(codeElement -> (JavaOp.FieldAccessOp.FieldLoadOp)codeElement)
                .filter(fieldLoadOp -> OpTk.fieldNameMatches(fieldLoadOp,pattern() ))
                .mapMulti((fieldLoadOp, consumer) ->
                        fieldLoadOp.operands().stream()
                                .filter(o->o instanceof Op.Result result && result.op() instanceof CoreOp.VarAccessOp.VarLoadOp)
                                .map(o->( CoreOp.VarAccessOp.VarLoadOp)((Op.Result)o).op())
                                .filter(this::isMethodFromHatKernelContext)
                                .forEach(varLoadOp -> {
                                    consumer.accept(fieldLoadOp);
                                    consumer.accept(varLoadOp);
                                })
                );

        Set<CodeElement<?, ?>> nodesInvolved = elements.collect(Collectors.toSet());


        funcOp = OpTk.transform(here, funcOp, nodesInvolved::contains, (blockBuilder, op) -> {
            CopyContext context = blockBuilder.context();
            if (op instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
                context.mapValue(varLoadOp.result(), context.getValue(varLoadOp.operands().getFirst()));
            } else if (op instanceof JavaOp.FieldAccessOp.FieldLoadOp fieldLoadOp) {
                fieldLoadOp.operands().stream()//does a field Load not have 1 operand?
                        .filter(operand->operand instanceof Op.Result result && result.op() instanceof CoreOp.VarAccessOp.VarLoadOp)
                        .map(operand->(CoreOp.VarAccessOp.VarLoadOp)((Op.Result)operand).op())
                        .forEach(_-> { // why are we looping over all operands ?
                                HATThreadOp threadOp = factory(fieldLoadOp);
                                Op.Result threadResult = blockBuilder.op(threadOp);
                                threadOp.setLocation(fieldLoadOp.location()); // update location
                                context.mapValue(fieldLoadOp.result(), threadResult);
                        });
            }
            return blockBuilder;
        });
        after(here,funcOp);
        return funcOp;
    }

    private boolean isMethodFromHatKernelContext(CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
        String kernelContextCanonicalName = hat.KernelContext.class.getName();
        return varLoadOp.resultType().toString().equals(kernelContextCanonicalName);
    }


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
            return Pattern.compile("([xyz]|gi[xyz])");
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
            return Pattern.compile("(gs[xyz]|max[XYZ])");
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

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
import hat.Config;
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
import jdk.incubator.code.Value;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.JavaOp;

import java.util.List;
import java.util.Set;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.function.Predicate;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class HATDialectifyThreadsPhase implements HATDialect  {

    protected final Accelerator accelerator;
    @Override  public Accelerator accelerator(){
        return this.accelerator;
    }
    public enum ThreadAccess {
        GLOBAL_ID,
        GLOBAL_SIZE,
        LOCAL_ID,
        LOCAL_SIZE,
        BLOCK_ID
    }
    private final ThreadAccess threadAccess;

    public HATDialectifyThreadsPhase(Accelerator accelerator,ThreadAccess threadAccess) {
        this.accelerator=accelerator;
        this.threadAccess =  threadAccess;
    }

    @Override
    public CoreOp.FuncOp apply(CoreOp.FuncOp funcOp) {
        if (accelerator.backend.config().showCompilationPhases()) {
            IO.println("[INFO] Code model before HatDialectifyThreadsPhase: " + funcOp.toText());
        }
        Predicate<JavaOp.FieldAccessOp.FieldLoadOp> isFieldOp = (fieldLoadOp)->
             switch (threadAccess) { // Why not pass threadAccess to isFieldLoadGlobalThreadId? see getDimension style
                case GLOBAL_ID -> isFieldLoadGlobalThreadId(fieldLoadOp);
                case GLOBAL_SIZE -> isFieldLoadGlobalSize(fieldLoadOp);
                case LOCAL_ID -> isFieldLoadThreadId(fieldLoadOp);
                case LOCAL_SIZE -> isFieldLoadThreadSize(fieldLoadOp);
                case BLOCK_ID -> isFieldLoadBlockId(fieldLoadOp);
            };
        Function<JavaOp.FieldAccessOp.FieldLoadOp,HATThreadOp> hatOpFactory = ( fieldLoadOp)-> {
            if (getDimension(threadAccess, fieldLoadOp) instanceof Integer dim && (dim >=0 && dim<3)) {
                return switch (threadAccess) {
                    case GLOBAL_ID -> new HATGlobalThreadIdOp(dim, fieldLoadOp.resultType());
                    case GLOBAL_SIZE -> new HATGlobalSizeOp(dim, fieldLoadOp.resultType());
                    case LOCAL_ID -> new HATLocalThreadIdOp(dim, fieldLoadOp.resultType());
                    case LOCAL_SIZE -> new HATLocalSizeOp(dim, fieldLoadOp.resultType());
                    case BLOCK_ID -> new HATBlockThreadIdOp(dim, fieldLoadOp.resultType());
                };
            }else {
                throw new IllegalStateException("Thread Access can't be below 0!");
            }
        };

        Stream<CodeElement<?, ?>> elements = funcOp.elements()
                .filter(codeElement ->codeElement instanceof JavaOp.FieldAccessOp.FieldLoadOp)
                .map(codeElement -> (JavaOp.FieldAccessOp.FieldLoadOp)codeElement)
                .filter(isFieldOp)
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
      //  if (!nodesInvolved.isEmpty()) {

        var here = OpTk.CallSite.of(HATDialectifyThreadsPhase.class, "run");
        funcOp = OpTk.transform(here, funcOp, nodesInvolved::contains, (blockBuilder, op) -> {
            CopyContext context = blockBuilder.context();
            if (op instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
                context.mapValue(varLoadOp.result(), context.getValue(varLoadOp.operands().getFirst()));
            } else if (op instanceof JavaOp.FieldAccessOp.FieldLoadOp fieldLoadOp) {
                fieldLoadOp.operands().stream()//does a field Load not have 1 operand?
                        .filter(operand->operand instanceof Op.Result result && result.op() instanceof CoreOp.VarAccessOp.VarLoadOp)
                        .map(operand->(CoreOp.VarAccessOp.VarLoadOp)((Op.Result)operand).op())
                        .forEach(_-> { // why are we looping over all operands ?
                                HATThreadOp threadOp = hatOpFactory.apply(fieldLoadOp);
                                Op.Result threadResult = blockBuilder.op(threadOp);
                                threadOp.setLocation(fieldLoadOp.location()); // update location
                                context.mapValue(fieldLoadOp.result(), threadResult);
                        });
            }
            return blockBuilder;
        });

            // No memory nodes involved
          //  return funcOp;
        //}
        if (accelerator.backend.config().showCompilationPhases()) {
            IO.println("[INFO] Code model after HatDialectifyThreadsPhase: " + funcOp.toText());
        }
        return funcOp;
    }


    private int getDimension(ThreadAccess threadAccess, JavaOp.FieldAccessOp.FieldLoadOp fieldLoadOp) {
        String fieldName = fieldLoadOp.fieldDescriptor().name();
        switch (threadAccess) {
            case GLOBAL_ID -> {
                return switch (fieldName){
                    case "y"->1;
                    case "z"->2;
                    default -> 0;
                };
            }
            case GLOBAL_SIZE -> {
                return switch (fieldName){
                    case "gsy"->1;
                    case "gsz"->2;
                    default -> 0;
                };
            }
            case LOCAL_ID -> {
                return switch (fieldName){
                    case "liy"->1;
                    case "liz"->2;
                    default -> 0;
                };
            }
            case LOCAL_SIZE -> {
                return switch (fieldName){
                    case "lsy"->1;
                    case "lsz"->2;
                    default -> 0;
                };
            }
            case BLOCK_ID ->  {
                return switch (fieldName){
                    case "biy"->1;
                    case "biz"->2;
                    default -> 0;
                };
            }
        }
        return -1;
    }


    private boolean isFieldLoadGlobalThreadId(JavaOp.FieldAccessOp.FieldLoadOp fieldLoadOp) {
        return OpTk.fieldNameMatches(fieldLoadOp, Pattern.compile("([xyz]|gi[xyz])"));
    }

    private boolean isFieldLoadGlobalSize(JavaOp.FieldAccessOp.FieldLoadOp fieldLoadOp) {
        return OpTk.fieldNameMatches(fieldLoadOp, Pattern.compile("(gs[xyz]|max[XYZ])"));
    }

    private boolean isFieldLoadThreadId(JavaOp.FieldAccessOp.FieldLoadOp fieldLoadOp) {
        return OpTk.fieldNameMatches(fieldLoadOp, Pattern.compile("li[xyz]"));
    }

    private boolean isFieldLoadThreadSize(JavaOp.FieldAccessOp.FieldLoadOp fieldLoadOp) {
        return OpTk.fieldNameMatches(fieldLoadOp, Pattern.compile("ls[xyz]"));
    }

    private boolean isFieldLoadBlockId(JavaOp.FieldAccessOp.FieldLoadOp fieldLoadOp) {
        return OpTk.fieldNameMatches(fieldLoadOp, Pattern.compile("bi[xyz]"));
    }

    private boolean isMethodFromHatKernelContext(CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
        String kernelContextCanonicalName = hat.KernelContext.class.getName();
        return varLoadOp.resultType().toString().equals(kernelContextCanonicalName);
    }


}

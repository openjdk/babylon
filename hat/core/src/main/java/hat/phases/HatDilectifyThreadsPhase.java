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

import hat.dialect.HatBlockThreadIdOp;
import hat.dialect.HatGlobalSizeOp;
import hat.dialect.HatGlobalThreadIdOp;
import hat.dialect.HatLocalSizeOp;
import hat.dialect.HatLocalThreadIdOp;
import hat.dialect.HatThreadOP;
import jdk.incubator.code.CodeElement;
import jdk.incubator.code.CopyContext;
import jdk.incubator.code.Op;
import jdk.incubator.code.Value;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.JavaOp;

import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class HatDilectifyThreadsPhase implements HatDialectifyPhase {

    private final ThreadAccess threadAccess;

    public HatDilectifyThreadsPhase(ThreadAccess threadAccess) {
        this.threadAccess =  threadAccess;
    }

    @Override
    public CoreOp.FuncOp run(CoreOp.FuncOp funcOp) {
        Stream<CodeElement<?, ?>> elements = funcOp.elements()
                .mapMulti((codeElement, consumer) -> {
                    if (codeElement instanceof JavaOp.FieldAccessOp.FieldLoadOp fieldLoadOp) {
                        List<Value> operands = fieldLoadOp.operands();
                        for (Value inputOperand : operands) {
                            if (inputOperand instanceof Op.Result result) {
                                if (result.op() instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
                                    boolean isThreadIntrinsic = switch (threadAccess) {
                                        case GLOBAL_ID -> isFieldLoadGlobalThreadId(fieldLoadOp);
                                        case GLOBAL_SIZE -> isFieldLoadGlobalSize(fieldLoadOp);
                                        case LOCAL_ID -> isFieldLoadThreadId(fieldLoadOp);
                                        case LOCAL_SIZE -> isFieldLoadThreadSize(fieldLoadOp);
                                        case BLOCK_ID ->  isFieldLoadBlockId(fieldLoadOp);
                                    };
                                    if (isMethodFromHatKernelContext(varLoadOp) && isThreadIntrinsic) {
                                        consumer.accept(fieldLoadOp);
                                        consumer.accept(varLoadOp);
                                    }
                                }
                            }
                        }
                    }
                });

        Set<CodeElement<?, ?>> nodesInvolved = elements.collect(Collectors.toSet());
        if (nodesInvolved.isEmpty()) {
            // No memory nodes involved
            return funcOp;
        }

        funcOp = funcOp.transform((blockBuilder, op) -> {
            CopyContext context = blockBuilder.context();
            if (!nodesInvolved.contains(op)) {
                blockBuilder.op(op);
            } else if (op instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
                // pass value
                context.mapValue(varLoadOp.result(), context.getValue(varLoadOp.operands().getFirst()));
            } else if (op instanceof JavaOp.FieldAccessOp.FieldLoadOp fieldLoadOp) {
                List<Value> operands = fieldLoadOp.operands();
                for (Value operand : operands) {
                    if (operand instanceof Op.Result r && r.op() instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
                        List<Value> varLoadOperands = varLoadOp.operands();
                        List<Value> outputOperands = context.getValues(varLoadOperands);
                        int dim = getDimension(threadAccess, fieldLoadOp);
                        if (dim < 0) {
                            throw new IllegalStateException("Thread Access can't be below 0!");
                        }
                        HatThreadOP threadOP = switch (threadAccess) {
                            case GLOBAL_ID -> new HatGlobalThreadIdOp(dim, fieldLoadOp.resultType(), outputOperands);
                            case GLOBAL_SIZE -> new HatGlobalSizeOp(dim, fieldLoadOp.resultType(), outputOperands);
                            case LOCAL_ID -> new HatLocalThreadIdOp(dim, fieldLoadOp.resultType(), outputOperands);
                            case LOCAL_SIZE -> new HatLocalSizeOp(dim, fieldLoadOp.resultType(), outputOperands);
                            case BLOCK_ID -> new HatBlockThreadIdOp(dim, fieldLoadOp.resultType(), outputOperands);
                        };
                        Op.Result threadResult = blockBuilder.op(threadOP);
                        context.mapValue(fieldLoadOp.result(), threadResult);
                    }
                }
            }
            return blockBuilder;
        });
        //IO.println("[INFO] Code model: " + funcOp.toText());
        //entrypoint.funcOp(funcOp);
        return funcOp;
    }

    public enum ThreadAccess {
        GLOBAL_ID,
        GLOBAL_SIZE,
        LOCAL_ID,
        LOCAL_SIZE,
        BLOCK_ID,
    }

    private int getDimension(ThreadAccess threadAccess, JavaOp.FieldAccessOp.FieldLoadOp fieldLoadOp) {
        String fieldName = fieldLoadOp.fieldDescriptor().name();
        switch (threadAccess) {
            case GLOBAL_ID -> {
                if (fieldName.equals("y")) {
                    return 1;
                } else if (fieldName.equals("z")) {
                    return 2;
                }
                return 0;
            }
            case GLOBAL_SIZE -> {
                if (fieldName.equals("gsy")) {
                    return 1;
                } else if (fieldName.equals("gsz")) {
                    return 2;
                }
                return 0;
            }
            case LOCAL_ID -> {
                if (fieldName.equals("liy")) {
                    return 1;
                } else if (fieldName.equals("lyz")) {
                    return 2;
                }
                return 0;
            }
            case LOCAL_SIZE -> {
                if (fieldName.equals("lsy")) {
                    return 1;
                } else if (fieldName.equals("lsz")) {
                    return 2;
                }
                return 0;
            }
            case BLOCK_ID ->  {
                if (fieldName.equals("biy")) {
                    return 1;
                } else if (fieldName.equals("biz")) {
                    return 2;
                }
                return 0;
            }
        }
        return -1;
    }


    private boolean isFieldLoadGlobalThreadId(JavaOp.FieldAccessOp.FieldLoadOp fieldLoadOp) {
        return fieldLoadOp.fieldDescriptor().name().equals("x")
                || fieldLoadOp.fieldDescriptor().name().equals("y")
                ||  fieldLoadOp.fieldDescriptor().name().equals("z")
                || fieldLoadOp.fieldDescriptor().name().equals("gix")
                || fieldLoadOp.fieldDescriptor().name().equals("giy")
                ||  fieldLoadOp.fieldDescriptor().name().equals("giz");
    }

    private boolean isFieldLoadGlobalSize(JavaOp.FieldAccessOp.FieldLoadOp fieldLoadOp) {
        return fieldLoadOp.fieldDescriptor().name().equals("gsx")
                || fieldLoadOp.fieldDescriptor().name().equals("gsy")
                ||  fieldLoadOp.fieldDescriptor().name().equals("gsz")
                || fieldLoadOp.fieldDescriptor().name().equals("maxX")
                || fieldLoadOp.fieldDescriptor().name().equals("maxY")
                ||  fieldLoadOp.fieldDescriptor().name().equals("maxZ");
    }

    private boolean isFieldLoadThreadId(JavaOp.FieldAccessOp.FieldLoadOp fieldLoadOp) {
        return fieldLoadOp.fieldDescriptor().name().equals("lix")
                || fieldLoadOp.fieldDescriptor().name().equals("liy")
                ||  fieldLoadOp.fieldDescriptor().name().equals("liz");
    }

    private boolean isFieldLoadThreadSize(JavaOp.FieldAccessOp.FieldLoadOp fieldLoadOp) {
        return fieldLoadOp.fieldDescriptor().name().equals("lsx")
                || fieldLoadOp.fieldDescriptor().name().equals("lsy")
                ||  fieldLoadOp.fieldDescriptor().name().equals("lsz");
    }

    private boolean isFieldLoadBlockId(JavaOp.FieldAccessOp.FieldLoadOp fieldLoadOp) {
        return fieldLoadOp.fieldDescriptor().name().equals("bix")
                || fieldLoadOp.fieldDescriptor().name().equals("biy")
                ||  fieldLoadOp.fieldDescriptor().name().equals("biz");
    }

    private boolean isMethodFromHatKernelContext(CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
        String kernelContextCanonicalName = hat.KernelContext.class.getName();
        return varLoadOp.resultType().toString().equals(kernelContextCanonicalName);
    }


}

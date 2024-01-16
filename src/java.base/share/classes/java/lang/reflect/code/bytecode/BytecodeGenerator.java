/*
 * Copyright (c) 2024, Oracle and/or its affiliates. All rights reserved.
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

package java.lang.reflect.code.bytecode;

import java.lang.classfile.ClassModel;
import java.lang.classfile.ClassFile;
import java.lang.classfile.CodeBuilder;
import java.lang.classfile.Label;
import java.lang.classfile.components.ClassPrinter;

import java.io.File;
import java.io.FileOutputStream;
import java.lang.constant.ClassDesc;
import java.lang.reflect.code.Block;
import java.lang.reflect.code.Body;
import java.lang.reflect.code.op.CoreOps;
import java.lang.reflect.code.Op;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.util.ArrayDeque;
import java.util.Deque;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public final class BytecodeGenerator {
    private BytecodeGenerator() {
    }

    public static MethodHandle generate(MethodHandles.Lookup l, CoreOps.FuncOp fop) {
        byte[] classBytes = generateClassData(l, fop);

        {
            print(classBytes);
            try {
                File f = new File("f.class");
                try (FileOutputStream fos = new FileOutputStream(f)) {
                    fos.write(classBytes);
                }
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }

        MethodHandles.Lookup hcl;
        try {
            hcl = l.defineHiddenClass(classBytes, true);
        } catch (IllegalAccessException e) {
            e.printStackTrace();
            throw new RuntimeException(e);
        }

        try {
            MethodType mt = fop.funcDescriptor().resolve(hcl);
            return hcl.findStatic(hcl.lookupClass(), fop.funcName(), mt);
        } catch (ReflectiveOperationException e) {
            throw new RuntimeException(e);
        }
    }

    private static void print(byte[] classBytes) {
        ClassModel cm = ClassFile.of().parse(classBytes);
        ClassPrinter.toYaml(cm, ClassPrinter.Verbosity.CRITICAL_ATTRIBUTES, System.out::print);
    }

    public static byte[] generateClassData(MethodHandles.Lookup l, CoreOps.FuncOp fop) {
        String packageName = l.lookupClass().getPackageName();
        String className = packageName.isEmpty()
                ? fop.funcName()
                : packageName + "." + fop.funcName();
        byte[] classBytes = ClassFile.of().build(ClassDesc.of(className),
                clb -> {
                    clb.withMethodBody(
                            fop.funcName(),
                            fop.funcDescriptor().toNominalDescriptor(),
                            ClassFile.ACC_PUBLIC | ClassFile.ACC_STATIC,
                            cob -> {
                                ConversionContext c = new ConversionContext(cob);
                                generateMethodBody(fop, cob, c);
                            });
                });
        return classBytes;
    }

    /*
        Live list of slot, value, v, and value, r, after which no usage of v dominates r
        i.e. liveness range.
        Free list, once slot goes dead it is added to the free list, so it can be reused.

        Block args need to have a fixed mapping to locals, unless the stack is used.
     */

    static final class ConversionContext implements BytecodeInstructionOps.MethodVisitorContext {
        final CodeBuilder cb;
        final Deque<BytecodeInstructionOps.ExceptionTableStart> labelStack;
        final Map<Object, Label> labels;

        public ConversionContext(CodeBuilder cb) {
            this.cb = cb;
            this.labelStack = new ArrayDeque<>();
            this.labels = new HashMap<>();
        }

        @Override
        public Deque<BytecodeInstructionOps.ExceptionTableStart> exceptionRegionStack() {
            return labelStack;
        }

        @Override
        public Label getLabel(Object b) {
            return labels.computeIfAbsent(b, _b -> cb.newLabel());
        }
    }

    private static void generateMethodBody(CoreOps.FuncOp fop, CodeBuilder mv, ConversionContext c) {
        Body r = fop.body();
        generateFromBody(r, mv, c);
    }

    private static void generateFromBody(Body body, CodeBuilder mv, ConversionContext c) {
        // Process blocks in topological order
        // A jump instruction assumes the false successor block is
        // immediately after, in sequence, to the predecessor
        // since the jump instructions branch on a true condition
        // Conditions are inverted when lowered to bytecode
        List<Block> blocks = body.blocks();
        for (Block b : blocks) {
            // Ignore any non-entry blocks that have no predecessors
            if (body.entryBlock() != b && b.predecessors().isEmpty()) {
                continue;
            }

            Label blockLabel = c.getLabel(b);
            mv.labelBinding(blockLabel);

            List<Op> ops = b.ops();
            for (int i = 0; i < ops.size() - 1; i++) {
                Op op = ops.get(i);
                if (op instanceof BytecodeInstructionOps.InstructionOp inst) {
                    inst.apply(mv, c);
                } else if (op instanceof BytecodeInstructionOps.ControlInstructionOp inst) {
                    inst.apply(mv, c);
                } else {
                    throw new UnsupportedOperationException("Unsupported operation: " + op);
                }
            }

            Op top = b.terminatingOp();
            if (top instanceof BytecodeInstructionOps.GotoInstructionOp inst) {
                Block s = inst.successors().get(0).targetBlock();
                int bi = blocks.indexOf(b);
                int si = blocks.indexOf(s);
                // If successor occurs immediately after this block,
                // then no need for goto instruction
                if (bi != si - 1) {
                    inst.apply(mv, c);
                }
            } else if (top instanceof BytecodeInstructionOps.TerminatingInstructionOp inst) {
                inst.apply(mv, c);
            } else if (top instanceof BytecodeInstructionOps.ControlInstructionOp inst) {
                inst.apply(mv, c);
            } else {
                throw new UnsupportedOperationException("Unsupported operation: " + top.opName());
            }
        }
    }
}

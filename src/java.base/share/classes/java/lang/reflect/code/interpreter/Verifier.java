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

package java.lang.reflect.code.interpreter;

import java.lang.classfile.instruction.BranchInstruction;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.code.*;
import java.lang.reflect.code.op.CoreOp;
import java.lang.reflect.code.type.JavaType;
import java.lang.reflect.code.writer.OpWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public final class Verifier {

    @SuppressWarnings("serial")
    public final class VerifyError extends Error {

        public VerifyError(String message) {
            super(message);
        }

        public String getPrintedContext() {
            return toText(rootOp);
        }
    }

    public static List<Verifier.VerifyError> verify(Op op) {
        return verify(MethodHandles.publicLookup(), op);
    }

    public static List<Verifier.VerifyError> verify(MethodHandles.Lookup l, Op op) {
        var verifier = new Verifier(l, op);
        verifier.verifyOps();
        return verifier.errors == null ? List.of() : Collections.unmodifiableList(verifier.errors);
    }


    private final MethodHandles.Lookup lookup;
    private final Op rootOp;
    private OpWriter.CodeItemNamerOption namerOption;
    private List<Verifier.VerifyError> errors;

    private Verifier(MethodHandles.Lookup lookup, Op rootOp) {
        this.lookup = lookup;
        this.rootOp = rootOp;
    }

    private OpWriter.CodeItemNamerOption getNamer() {
        if (namerOption == null) {
            namerOption = OpWriter.CodeItemNamerOption.of(OpWriter.computeGlobalNames(rootOp));
        }
        return namerOption;
    }

    private String toText(Op op) {
        return OpWriter.toText(op, getNamer());
    }

    private String getName(CodeItem codeItem) {
        return getNamer().namer().apply(codeItem);
    }

    private void error(String message, Object... args) {
        if (errors == null) {
            errors = new ArrayList<>();
        }
        for (int i = 0; i < args.length; i++) {
            var arg = args[i];
            if (arg instanceof Op op) {
                args[i] = toText(op);
            } else if (arg instanceof Block b) {
                args[i] = getName(b);
            } else if (arg instanceof Value v) {
                args[i] = getName(v);
            }
        }
        errors.add(new VerifyError(message.formatted(args)));
    }

    private void verifyOps() {
        rootOp.traverse(null, CodeElement.opVisitor((n, op) -> {
            // Verify operands declaration dominannce
            for (var v : op.operands()) {
                if (!op.result().isDominatedBy(v)) {
                    error("%s %s operand %s is not dominated by its declaration in %s", op.parentBlock(), op, v, v.declaringBlock());
                }
            }

            // Verify individual Ops
            switch (op) {
                case CoreOp.BranchOp br ->
                    verifyBlockReferences(op, br.successors());
                case CoreOp.ArithmeticOperation _, CoreOp.TestOperation _ ->
                    verifyOpHandleExists(op, op.opName());
                case CoreOp.ConvOp _ -> {
                    verifyOpHandleExists(op, op.opName() + "_" + op.opType().returnType());
                }
                default -> {}

            }
            return null;
        }));
    }

    private void verifyBlockReferences(Op op, List<Block.Reference> references) {
        for (Block.Reference r : references) {
            Block b = r.targetBlock();
            List<Value> args = r.arguments();
            List<Block.Parameter> params = r.targetBlock().parameters();
            if (args.size() != params.size()) {
                error("%s %s block reference arguments size to target block parameters size mismatch", b, op);
            } else {
                Block tb = r.targetBlock();
                for (int i = 0; i < args.size(); i++) {
                    if (!isAssignable(params.get(i).type(), args.get(i), tb, b)) {
                        error("%s %s %s is not assignable from %s", op.parentBlock(), op, params.get(i).type(), args.get(i).type());
                    }
                }
            }
        }
    }

    private boolean isAssignable(TypeElement toType, Value fromValue,  Object toContext, Object fromContext) {
        if (toType.equals(fromValue.type())) return true;
        var to = resolveToClass(toType, toContext);
        var from = resolveToClass(fromValue.type(), fromContext);
        if (from.isPrimitive()) {
            // Primitive types assignability
            return to == int.class && (from == byte.class || from == short.class || from == char.class);
        } else {
            // Objects assignability
            return to.isAssignableFrom(from)
                // @@@ null Object assignability ?
                || fromValue instanceof Op.Result or && or.op() instanceof CoreOp.ConstantOp cop && cop.value() == null && !to.isPrimitive();
        }
    }

    public Class<?> resolveToClass(TypeElement d, Object context) {
        try {
            if (d instanceof JavaType jt) {
                return (Class<?>)jt.erasure().resolve(lookup);
            } else {
                error("%s %s is not a Java type", context, d);
            }
        } catch (ReflectiveOperationException e) {
            error("%s %s", context, e.getMessage());
        }
        return Object.class;
    }

    private void verifyOpHandleExists(Op op, String opName) {
        try {
            var mt = Interpreter.resolveToMethodType(lookup, op.opType()).erase();
            MethodHandles.lookup().findStatic(InvokableLeafOps.class, opName, mt);
        } catch (NoSuchMethodException nsme) {
            error("%s %s of type %s is not supported", op.parentBlock(), op, op.opType());
        } catch (IllegalAccessException iae) {
            error("%s %s %s",  op.parentBlock(), op, iae.getMessage());
        }
    }
}

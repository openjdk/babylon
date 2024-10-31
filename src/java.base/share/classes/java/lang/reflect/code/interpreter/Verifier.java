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

import java.lang.invoke.MethodHandles;
import java.lang.reflect.code.*;
import java.lang.reflect.code.op.CoreOp;
import java.lang.reflect.code.type.JavaType;
import java.lang.reflect.code.writer.OpWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public final class Verifier {

    public final class VerifyError {

        private final String message;

        public VerifyError(String message) {
            this.message = message;
        }

        public String getMessage() {
            return message;
        }

        public String getPrintedContext() {
            return toText(rootOp);
        }

        @Override
        public String toString() {
            return getMessage() + " in " + getPrintedContext();
        }
    }

    public static List<Verifier.VerifyError> verify(Op op) {
        return verify(MethodHandles.publicLookup(), op);
    }

    public static List<Verifier.VerifyError> verify(MethodHandles.Lookup l, Op op) {
        var verifier = new Verifier(l, op);
        verifier.verifyOps();
        verifier.verifyExceptionRegions();
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
            args[i] = toText(args[i]);
        }
        errors.add(new VerifyError(message.formatted(args)));
    }

    private String toText(Object arg) {
        return switch (arg) {
            case Op op -> toText(op);
            case Block b -> getName(b);
            case Value v -> getName(v);
            case List<?> l -> l.stream().map(this::toText).toList().toString();
            default -> arg.toString();
        };
    }

    private void verifyOps() {
        rootOp.traverse(null, CodeElement.opVisitor((_, op) -> {
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
                case CoreOp.ConditionalBranchOp cbr ->
                    verifyBlockReferences(op, cbr.successors());
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
            return to.isAssignableFrom(from);
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

    private void verifyExceptionRegions() {
        rootOp.traverse(new HashMap<Block, List<Block>>(), CodeElement.blockVisitor((map, b) -> {
            List<Block> catchBlocks = map.computeIfAbsent(b, _ -> List.of());
            switch (b.terminatingOp()) {
                case CoreOp.BranchOp br ->
                    verifyCatchStack(b, br, br.branch(), catchBlocks, map);
                case CoreOp.ConditionalBranchOp cbr -> {
                    verifyCatchStack(b, cbr, cbr.trueBranch(), catchBlocks, map);
                    verifyCatchStack(b, cbr, cbr.falseBranch(), catchBlocks, map);
                }
                case CoreOp.ExceptionRegionEnter ere -> {
                    List<Block> newCatchBlocks = new ArrayList<>();
                    newCatchBlocks.addAll(catchBlocks);
                    for (Block.Reference cb : ere.catchBlocks()) {
                        newCatchBlocks.add(cb.targetBlock());
                        verifyCatchStack(b, ere, cb, catchBlocks, map);
                    }
                    verifyCatchStack(b, ere, ere.start(), newCatchBlocks, map);
                }
                case CoreOp.ExceptionRegionExit ere -> {
                    List<Block> exitedCatchBlocks = ere.catchBlocks().stream().map(Block.Reference::targetBlock).toList();
                    if (exitedCatchBlocks.size() > catchBlocks.size() || !catchBlocks.reversed().subList(0, exitedCatchBlocks.size()).equals(exitedCatchBlocks)) {
                        error("%s %s exited catch blocks %s does not match actual stack %s", b, ere, exitedCatchBlocks, catchBlocks);
                    } else {
                        verifyCatchStack(b, ere, ere.end(), catchBlocks.subList(0, catchBlocks.size() - exitedCatchBlocks.size()), map);
                    }
                }
                default -> {}
            }
            return map;
        }));
    }

    private void verifyCatchStack(Block b, Op op, Block.Reference target, List<Block> catchBlocks, Map<Block, List<Block>> blockMap) {
        blockMap.compute(target.targetBlock(), (tb, stored) -> {
            if (stored != null && !stored.equals(catchBlocks)) {
                error("%s %s catch stack mismatch at target %s %s vs %s", b, op, tb, stored, catchBlocks);
            }
            return catchBlocks;
        });
    }
}

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

package java.lang.reflect.code.op;

import java.lang.constant.ClassDesc;
import java.lang.reflect.code.*;
import java.lang.reflect.code.type.*;
import java.util.*;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.stream.Stream;

import static java.lang.reflect.code.op.CoreOp.*;
import static java.lang.reflect.code.type.JavaType.*;

/**
 * The top-level operation class for the enclosed set of extended operations.
 * <p>
 * A code model, produced by the Java compiler from Java program source, may consist of extended operations and core
 * operations. Such a model represents the same Java program and preserves the program meaning as defined by the
 * Java Language Specification
 * <p>
 * Extended operations model specific Java language constructs, often those with structured control flow and nested
 * code. Each operation is transformable into a sequence of core operations, commonly referred to as lowering. Those
 * that implement {@link Op.Lowerable} can transform themselves and will transform associated extended operations
 * that are not explicitly lowerable.
 * <p>
 * A code model, produced by the Java compiler from source, and consisting of extended operations and core operations
 * can be transformed to one consisting only of core operations, where all extended operations are lowered. This
 * transformation preserves programing meaning. The resulting lowered code model also represents the same Java program.
 */
public sealed abstract class ExtendedOp extends ExternalizableOp {
    // Split string to ensure the name does not get rewritten
    // when the script copies this source to the jdk.compiler module
    static final String PACKAGE_NAME = "java.lang" + ".reflect.code";

    static final String ExtendedOp_CLASS_NAME = PACKAGE_NAME + "." + ExtendedOp.class.getSimpleName();

    protected ExtendedOp(Op that, CopyContext cc) {
        super(that, cc);
    }

    protected ExtendedOp(String name, List<? extends Value> operands) {
        super(name, operands);
    }

    protected ExtendedOp(ExternalizableOp.ExternalizedOp def) {
        super(def);
    }


    /**
     * The label operation, that can model Java language statements with label identifiers.
     */
    public sealed static abstract class JavaLabelOp extends ExtendedOp
            implements Op.Lowerable, Op.BodyTerminating, JavaStatement {
        JavaLabelOp(ExternalizedOp def) {
            super(def);

            if (def.operands().size() > 1) {
                throw new IllegalArgumentException("Operation must have zero or one operand " + def.name());
            }
        }

        JavaLabelOp(JavaLabelOp that, CopyContext cc) {
            super(that, cc);
        }

        JavaLabelOp(String name, Value label) {
            super(name, checkLabel(label));
        }

        static List<Value> checkLabel(Value label) {
            return label == null ? List.of() : List.of(label);
        }

        Op innerMostEnclosingTarget() {
            /*
                A break statement with no label attempts to transfer control to the
                innermost enclosing switch, while, do, or for statement; this enclosing statement,
                which is called the break target, then immediately completes normally.

                A break statement with label Identifier attempts to transfer control to the
                enclosing labeled statement (14.7) that has the same Identifier as its label;
                this enclosing statement, which is called the break target, then immediately completes normally.
                In this case, the break target need not be a switch, while, do, or for statement.
             */

            // No label
            // Get innermost enclosing loop operation
            Op op = this;
            Body b;
            do {
                b = op.ancestorBody();
                op = b.parentOp();
                if (op == null) {
                    throw new IllegalStateException("No enclosing loop");
                }
            } while (!(op instanceof Op.Loop || op instanceof JavaSwitchStatementOp));

            return switch (op) {
                case Op.Loop lop -> lop.loopBody() == b ? op : null;
                case JavaSwitchStatementOp swStat -> swStat.bodies().contains(b) ? op : null;
                default -> throw new IllegalStateException();
            };
        }

        boolean isUnlabeled() {
            return operands().isEmpty();
        }

        Op target() {
            // If unlabeled then find the nearest enclosing op
            // Otherwise obtain the label target
            if (isUnlabeled()) {
                return innerMostEnclosingTarget();
            }

            Value value = operands().get(0);
            if (value instanceof Result r && r.op().ancestorBody().parentOp() instanceof JavaLabeledOp lop) {
                return lop.target();
            } else {
                throw new IllegalStateException("Bad label value: " + value + " " + ((Result) value).op());
            }
        }

        Block.Builder lower(Block.Builder b, Function<BranchTarget, Block.Builder> f) {
            Op opt = target();
            BranchTarget t = getBranchTarget(b.context(), opt);
            if (t != null) {
                b.op(branch(f.apply(t).successor()));
            } else {
                throw new IllegalStateException("No branch target for operation: " + opt);
            }
            return b;
        }

        @Override
        public TypeElement resultType() {
            return VOID;
        }
    }

    /**
     * The break operation, that can model Java language break statements with label identifiers.
     */
    @OpFactory.OpDeclaration(JavaBreakOp.NAME)
    public static final class JavaBreakOp extends JavaLabelOp {
        public static final String NAME = "java.break";

        public JavaBreakOp(ExternalizedOp def) {
            super(def);
        }

        JavaBreakOp(JavaBreakOp that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public JavaBreakOp transform(CopyContext cc, OpTransformer ot) {
            return new JavaBreakOp(this, cc);
        }

        JavaBreakOp(Value label) {
            super(NAME, label);
        }

        @Override
        public Block.Builder lower(Block.Builder b, OpTransformer opT) {
            return lower(b, BranchTarget::breakBlock);
        }
    }

    /**
     * The continue operation, that can model Java language continue statements with label identifiers.
     */
    @OpFactory.OpDeclaration(JavaContinueOp.NAME)
    public static final class JavaContinueOp extends JavaLabelOp {
        public static final String NAME = "java.continue";

        public JavaContinueOp(ExternalizedOp def) {
            super(def);
        }

        JavaContinueOp(JavaContinueOp that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public JavaContinueOp transform(CopyContext cc, OpTransformer ot) {
            return new JavaContinueOp(this, cc);
        }

        JavaContinueOp(Value label) {
            super(NAME, label);
        }

        @Override
        public Block.Builder lower(Block.Builder b, OpTransformer opT) {
            return lower(b, BranchTarget::continueBlock);
        }
    }

    record BranchTarget(Block.Builder breakBlock, Block.Builder continueBlock) {
    }

    static final String BRANCH_TARGET_MAP_PROPERTY_KEY = "BRANCH_TARGET_MAP";

    static BranchTarget getBranchTarget(CopyContext cc, CodeElement<?, ?> codeElement) {
        @SuppressWarnings("unchecked")
        Map<CodeElement<?, ?>, BranchTarget> m = (Map<CodeElement<?, ?>, BranchTarget>) cc.getProperty(BRANCH_TARGET_MAP_PROPERTY_KEY);
        if (m != null) {
            return m.get(codeElement);
        }
        return null;
    }

    static void setBranchTarget(CopyContext cc, CodeElement<?, ?> codeElement, BranchTarget t) {
        @SuppressWarnings("unchecked")
        Map<CodeElement<?, ?>, BranchTarget> x = (Map<CodeElement<?, ?>, BranchTarget>) cc.computePropertyIfAbsent(
                BRANCH_TARGET_MAP_PROPERTY_KEY, k -> new HashMap<>());
        x.put(codeElement, t);
    }

    /**
     * The yield operation, that can model Java language yield statements.
     */
    @OpFactory.OpDeclaration(JavaYieldOp.NAME)
    public static final class JavaYieldOp extends ExtendedOp
            implements Op.BodyTerminating, JavaStatement, Op.Lowerable {
        public static final String NAME = "java.yield";

        public JavaYieldOp(ExternalizedOp def) {
            super(def);
        }

        JavaYieldOp(JavaYieldOp that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public JavaYieldOp transform(CopyContext cc, OpTransformer ot) {
            return new JavaYieldOp(this, cc);
        }

        JavaYieldOp() {
            super(NAME,
                    List.of());
        }

        JavaYieldOp(Value operand) {
            super(NAME, List.of(operand));
        }

        public Value yieldValue() {
            if (operands().size() == 1) {
                return operands().get(0);
            } else {
                // @@@
                return null;
            }
        }

        @Override
        public TypeElement resultType() {
            return VOID;
        }

        @Override
        public Block.Builder lower(Block.Builder b, OpTransformer opT) {
            // for now, we will use breakBlock field to indicate java.yield target block
            return lower(b, BranchTarget::breakBlock);
        }

        Block.Builder lower(Block.Builder b, Function<BranchTarget, Block.Builder> f) {
            Op opt = target();
            BranchTarget t = getBranchTarget(b.context(), opt);
            if (t != null) {
                b.op(branch(f.apply(t).successor(b.context().getValue(yieldValue()))));
            } else {
                throw new IllegalStateException("No branch target for operation: " + opt);
            }
            return b;
        }

        Op target() {
            return innerMostEnclosingTarget();
        }

        Op innerMostEnclosingTarget() {
            Op op = this;
            Body b;
            do {
                b = op.ancestorBody();
                op = b.parentOp();
                if (op == null) {
                    throw new IllegalStateException("No enclosing switch");
                }
            } while (!(op instanceof JavaSwitchExpressionOp));
            return op;
        }
    }

    /**
     * The block operation, that can model Java language blocks.
     */
    @OpFactory.OpDeclaration(JavaBlockOp.NAME)
    public static final class JavaBlockOp extends ExtendedOp
            implements Op.Nested, Op.Lowerable, JavaStatement {
        public static final String NAME = "java.block";

        final Body body;

        public JavaBlockOp(ExternalizedOp def) {
            super(def);

            if (!def.operands().isEmpty()) {
                throw new IllegalStateException("Operation must have no operands");
            }

            this.body = def.bodyDefinitions().get(0).build(this);
        }

        JavaBlockOp(JavaBlockOp that, CopyContext cc, OpTransformer ot) {
            super(that, cc);

            // Copy body
            this.body = that.body.transform(cc, ot).build(this);
        }

        @Override
        public JavaBlockOp transform(CopyContext cc, OpTransformer ot) {
            return new JavaBlockOp(this, cc, ot);
        }

        JavaBlockOp(Body.Builder bodyC) {
            super(NAME, List.of());

            this.body = bodyC.build(this);
            if (!body.bodyType().returnType().equals(VOID)) {
                throw new IllegalArgumentException("Body should return void: " + body.bodyType());
            }
            if (!body.bodyType().parameterTypes().isEmpty()) {
                throw new IllegalArgumentException("Body should have zero parameters: " + body.bodyType());
            }
        }

        @Override
        public List<Body> bodies() {
            return List.of(body);
        }

        public Body body() {
            return body;
        }

        @Override
        public Block.Builder lower(Block.Builder b, OpTransformer opT) {
            Block.Builder exit = b.block();
            setBranchTarget(b.context(), this, new BranchTarget(exit, null));

            b.transformBody(body, List.of(), opT.andThen((block, op) -> {
                if (op instanceof YieldOp) {
                    block.op(branch(exit.successor()));
                } else {
                    // @@@ Composition of lowerable ops
                    if (op instanceof Lowerable lop) {
                        block = lop.lower(block, opT);
                    } else {
                        block.op(op);
                    }
                }
                return block;
            }));

            return exit;
        }

        @Override
        public TypeElement resultType() {
            return VOID;
        }
    }

    /**
     * The synchronized operation, that can model Java synchronized statements.
     */
    @OpFactory.OpDeclaration(JavaSynchronizedOp.NAME)
    public static final class JavaSynchronizedOp extends ExtendedOp
            implements Op.Nested, Op.Lowerable, JavaStatement {
        public static final String NAME = "java.synchronized";

        final Body expr;
        final Body blockBody;

        public JavaSynchronizedOp(ExternalizedOp def) {
            super(def);

            this.expr = def.bodyDefinitions().get(0).build(this);
            this.blockBody = def.bodyDefinitions().get(1).build(this);
        }

        JavaSynchronizedOp(JavaSynchronizedOp that, CopyContext cc, OpTransformer ot) {
            super(that, cc);

            // Copy bodies
            this.expr = that.expr.transform(cc, ot).build(this);
            this.blockBody = that.blockBody.transform(cc, ot).build(this);
        }

        @Override
        public JavaSynchronizedOp transform(CopyContext cc, OpTransformer ot) {
            return new JavaSynchronizedOp(this, cc, ot);
        }

        JavaSynchronizedOp(Body.Builder exprC, Body.Builder bodyC) {
            super(NAME, List.of());

            this.expr = exprC.build(this);
            if (expr.bodyType().returnType().equals(VOID)) {
                throw new IllegalArgumentException("Expression body should return non-void value: " + expr.bodyType());
            }
            if (!expr.bodyType().parameterTypes().isEmpty()) {
                throw new IllegalArgumentException("Expression body should have zero parameters: " + expr.bodyType());
            }

            this.blockBody = bodyC.build(this);
            if (!blockBody.bodyType().returnType().equals(VOID)) {
                throw new IllegalArgumentException("Block body should return void: " + blockBody.bodyType());
            }
            if (!blockBody.bodyType().parameterTypes().isEmpty()) {
                throw new IllegalArgumentException("Block body should have zero parameters: " + blockBody.bodyType());
            }
        }

        @Override
        public List<Body> bodies() {
            return List.of(expr, blockBody);
        }

        public Body expr() {
            return expr;
        }

        public Body blockBody() {
            return blockBody;
        }

        @Override
        public Block.Builder lower(Block.Builder b, OpTransformer opT) {
            // Lower the expression body, yielding a monitor target
            b = lowerExpr(b, opT);
            Value monitorTarget = b.parameters().get(0);

            // Monitor enter
            b.op(CoreOp.monitorEnter(monitorTarget));

            Block.Builder exit = b.block();
            setBranchTarget(b.context(), this, new BranchTarget(exit, null));

            // Exception region for the body
            Block.Builder syncRegionEnter = b.block();
            Block.Builder catcherFinally = b.block();
            b.op(exceptionRegionEnter(
                    syncRegionEnter.successor(), catcherFinally.successor()));

            OpTransformer syncExitTransformer = opT.compose((block, op) -> {
                if (op instanceof CoreOp.ReturnOp ||
                    (op instanceof ExtendedOp.JavaLabelOp lop && ifExitFromSynchronized(lop))) {
                    // Monitor exit
                    block.op(CoreOp.monitorExit(monitorTarget));
                    // Exit the exception region
                    Block.Builder exitRegion = block.block();
                    block.op(exceptionRegionExit(exitRegion.successor(), catcherFinally.successor()));
                    return exitRegion;
                } else {
                    return block;
                }
            });

            syncRegionEnter.transformBody(blockBody, List.of(), syncExitTransformer.andThen((block, op) -> {
                if (op instanceof YieldOp) {
                    // Monitor exit
                    block.op(CoreOp.monitorExit(monitorTarget));
                    // Exit the exception region
                    block.op(exceptionRegionExit(exit.successor(), catcherFinally.successor()));
                } else {
                    // @@@ Composition of lowerable ops
                    if (op instanceof Lowerable lop) {
                        block = lop.lower(block, syncExitTransformer);
                    } else {
                        block.op(op);
                    }
                }
                return block;
            }));

            // The catcher, with an exception region back branching to itself
            Block.Builder catcherFinallyRegionEnter = b.block();
            catcherFinally.op(exceptionRegionEnter(
                    catcherFinallyRegionEnter.successor(), catcherFinally.successor()));

            // Monitor exit
            catcherFinallyRegionEnter.op(CoreOp.monitorExit(monitorTarget));
            Block.Builder catcherFinallyRegionExit = b.block();
            // Exit the exception region
            catcherFinallyRegionEnter.op(exceptionRegionExit(
                    catcherFinallyRegionExit.successor(), catcherFinally.successor()));
            // Rethrow outside of region
            Block.Parameter t = catcherFinally.parameter(type(Throwable.class));
            catcherFinallyRegionExit.op(_throw(t));

            return exit;
        }

        Block.Builder lowerExpr(Block.Builder b, OpTransformer opT) {
            Block.Builder exprExit = b.block(expr.bodyType().returnType());
            b.transformBody(expr, List.of(), opT.andThen((block, op) -> {
                if (op instanceof YieldOp yop) {
                    Value monitorTarget = block.context().getValue(yop.yieldValue());
                    block.op(branch(exprExit.successor(monitorTarget)));
                } else {
                    // @@@ Composition of lowerable ops
                    if (op instanceof Lowerable lop) {
                        block = lop.lower(block, opT);
                    } else {
                        block.op(op);
                    }
                }
                return block;
            }));
            return exprExit;
        }

        boolean ifExitFromSynchronized(JavaLabelOp lop) {
            Op target = lop.target();
            return target == this || ifAncestorOp(target, this);
        }

        static boolean ifAncestorOp(Op ancestor, Op op) {
            while (op.ancestorBody() != null) {
                op = op.ancestorBody().parentOp();
                if (op == ancestor) {
                    return true;
                }
            }
            return false;
        }

        @Override
        public TypeElement resultType() {
            return VOID;
        }
    }

    /**
     * The labeled operation, that can model Java language labeled statements.
     */
    @OpFactory.OpDeclaration(JavaLabeledOp.NAME)
    public static final class JavaLabeledOp extends ExtendedOp
            implements Op.Nested, Op.Lowerable, JavaStatement {
        public static final String NAME = "java.labeled";

        final Body body;

        public JavaLabeledOp(ExternalizedOp def) {
            super(def);

            if (!def.operands().isEmpty()) {
                throw new IllegalStateException("Operation must have no operands");
            }

            this.body = def.bodyDefinitions().get(0).build(this);
        }

        JavaLabeledOp(JavaLabeledOp that, CopyContext cc, OpTransformer ot) {
            super(that, cc);

            // Copy body
            this.body = that.body.transform(cc, ot).build(this);
        }

        @Override
        public JavaLabeledOp transform(CopyContext cc, OpTransformer ot) {
            return new JavaLabeledOp(this, cc, ot);
        }

        JavaLabeledOp(Body.Builder bodyC) {
            super(NAME, List.of());

            this.body = bodyC.build(this);
            if (!body.bodyType().returnType().equals(VOID)) {
                throw new IllegalArgumentException("Body should return void: " + body.bodyType());
            }
            if (!body.bodyType().parameterTypes().isEmpty()) {
                throw new IllegalArgumentException("Body should have zero parameters: " + body.bodyType());
            }
        }

        @Override
        public List<Body> bodies() {
            return List.of(body);
        }

        public Op label() {
            return body.entryBlock().firstOp();
        }

        public Op target() {
            return body.entryBlock().nextOp(label());
        }

        @Override
        public Block.Builder lower(Block.Builder b, OpTransformer opT) {
            Block.Builder exit = b.block();
            setBranchTarget(b.context(), this, new BranchTarget(exit, null));

            AtomicBoolean first = new AtomicBoolean();
            b.transformBody(body, List.of(), opT.andThen((block, op) -> {
                // Drop first operation that corresponds to the label
                if (!first.get()) {
                    first.set(true);
                    return block;
                }

                if (op instanceof YieldOp) {
                    block.op(branch(exit.successor()));
                } else {
                    // @@@ Composition of lowerable ops
                    if (op instanceof Lowerable lop) {
                        block = lop.lower(block, opT);
                    } else {
                        block.op(op);
                    }
                }
                return block;
            }));

            return exit;
        }

        @Override
        public TypeElement resultType() {
            return VOID;
        }
    }

    /**
     * The if operation, that can model Java language if, if-then, and if-then-else statements.
     */
    @OpFactory.OpDeclaration(JavaIfOp.NAME)
    public static final class JavaIfOp extends ExtendedOp
            implements Op.Nested, Op.Lowerable, JavaStatement {

        static final FunctionType PREDICATE_TYPE = FunctionType.functionType(BOOLEAN);

        static final FunctionType ACTION_TYPE = FunctionType.VOID;

        public static class IfBuilder {
            final Body.Builder ancestorBody;
            final List<Body.Builder> bodies;

            IfBuilder(Body.Builder ancestorBody) {
                this.ancestorBody = ancestorBody;
                this.bodies = new ArrayList<>();
            }

            public ThenBuilder _if(Consumer<Block.Builder> c) {
                Body.Builder body = Body.Builder.of(ancestorBody, PREDICATE_TYPE);
                c.accept(body.entryBlock());
                bodies.add(body);

                return new ThenBuilder(ancestorBody, bodies);
            }
        }

        public static class ThenBuilder {
            final Body.Builder ancestorBody;
            final List<Body.Builder> bodies;

            public ThenBuilder(Body.Builder ancestorBody, List<Body.Builder> bodies) {
                this.ancestorBody = ancestorBody;
                this.bodies = bodies;
            }

            public ElseIfBuilder then(Consumer<Block.Builder> c) {
                Body.Builder body = Body.Builder.of(ancestorBody, ACTION_TYPE);
                c.accept(body.entryBlock());
                bodies.add(body);

                return new ElseIfBuilder(ancestorBody, bodies);
            }

            public ElseIfBuilder then() {
                Body.Builder body = Body.Builder.of(ancestorBody, ACTION_TYPE);
                body.entryBlock().op(_yield());
                bodies.add(body);

                return new ElseIfBuilder(ancestorBody, bodies);
            }
        }

        public static class ElseIfBuilder {
            final Body.Builder ancestorBody;
            final List<Body.Builder> bodies;

            public ElseIfBuilder(Body.Builder ancestorBody, List<Body.Builder> bodies) {
                this.ancestorBody = ancestorBody;
                this.bodies = bodies;
            }

            public ThenBuilder elseif(Consumer<Block.Builder> c) {
                Body.Builder body = Body.Builder.of(ancestorBody, PREDICATE_TYPE);
                c.accept(body.entryBlock());
                bodies.add(body);

                return new ThenBuilder(ancestorBody, bodies);
            }

            public JavaIfOp _else(Consumer<Block.Builder> c) {
                Body.Builder body = Body.Builder.of(ancestorBody, ACTION_TYPE);
                c.accept(body.entryBlock());
                bodies.add(body);

                return new JavaIfOp(bodies);
            }

            public JavaIfOp _else() {
                Body.Builder body = Body.Builder.of(ancestorBody, ACTION_TYPE);
                body.entryBlock().op(_yield());
                bodies.add(body);

                return new JavaIfOp(bodies);
            }
        }

        public static final String NAME = "java.if";

        final List<Body> bodies;

        public JavaIfOp(ExternalizedOp def) {
            super(def);

            if (!def.operands().isEmpty()) {
                throw new IllegalStateException("Operation must have no operands");
            }

            // @@@ Validate

            this.bodies = def.bodyDefinitions().stream().map(bd -> bd.build(this)).toList();
        }

        JavaIfOp(JavaIfOp that, CopyContext cc, OpTransformer ot) {
            super(that, cc);

            // Copy body
            this.bodies = that.bodies.stream()
                    .map(b -> b.transform(cc, ot).build(this)).toList();
        }

        @Override
        public JavaIfOp transform(CopyContext cc, OpTransformer ot) {
            return new JavaIfOp(this, cc, ot);
        }

        JavaIfOp(List<Body.Builder> bodyCs) {
            super(NAME, List.of());

            // Normalize by adding an empty else action
            // @@@ Is this needed?
            if (bodyCs.size() % 2 == 0) {
                bodyCs = new ArrayList<>(bodyCs);
                Body.Builder end = Body.Builder.of(bodyCs.get(0).ancestorBody(),
                        FunctionType.VOID);
                end.entryBlock().op(_yield());
                bodyCs.add(end);
            }

            this.bodies = bodyCs.stream().map(bc -> bc.build(this)).toList();

            if (bodies.size() < 2) {
                throw new IllegalArgumentException("Incorrect number of bodies: " + bodies.size());
            }
            for (int i = 0; i < bodies.size(); i += 2) {
                Body action;
                if (i == bodies.size() - 1) {
                    action = bodies.get(i);
                } else {
                    action = bodies.get(i + 1);
                    Body fromPred = bodies.get(i);
                    if (!fromPred.bodyType().equals(FunctionType.functionType(BOOLEAN))) {
                        throw new IllegalArgumentException("Illegal predicate body descriptor: " + fromPred.bodyType());
                    }
                }
                if (!action.bodyType().equals(FunctionType.VOID)) {
                    throw new IllegalArgumentException("Illegal action body descriptor: " + action.bodyType());
                }
            }
        }

        @Override
        public List<Body> bodies() {
            return bodies;
        }

        @Override
        public Block.Builder lower(Block.Builder b, OpTransformer opT) {
            Block.Builder exit = b.block();
            setBranchTarget(b.context(), this, new BranchTarget(exit, null));

            // Create predicate and action blocks
            List<Block.Builder> builders = new ArrayList<>();
            for (int i = 0; i < bodies.size(); i += 2) {
                if (i == bodies.size() - 1) {
                    builders.add(b.block());
                } else {
                    builders.add(i == 0 ? b : b.block());
                    builders.add(b.block());
                }
            }

            for (int i = 0; i < bodies.size(); i += 2) {
                Body actionBody;
                Block.Builder action;
                if (i == bodies.size() - 1) {
                    actionBody = bodies.get(i);
                    action = builders.get(i);
                } else {
                    Body predBody = bodies.get(i);
                    actionBody = bodies.get(i + 1);

                    Block.Builder pred = builders.get(i);
                    action = builders.get(i + 1);
                    Block.Builder next = builders.get(i + 2);

                    pred.transformBody(predBody, List.of(), opT.andThen((block, op) -> {
                        if (op instanceof YieldOp yo) {
                            block.op(conditionalBranch(block.context().getValue(yo.yieldValue()),
                                    action.successor(), next.successor()));
                        } else if (op instanceof Lowerable lop) {
                            // @@@ Composition of lowerable ops
                            block = lop.lower(block, opT);
                        } else {
                            block.op(op);
                        }
                        return block;
                    }));
                }

                action.transformBody(actionBody, List.of(), opT.andThen((block, op) -> {
                    if (op instanceof YieldOp) {
                        block.op(branch(exit.successor()));
                    } else {
                        // @@@ Composition of lowerable ops
                        if (op instanceof Lowerable lop) {
                            block = lop.lower(block, opT);
                        } else {
                            block.op(op);
                        }
                    }
                    return block;
                }));
            }

            return exit;
        }

        @Override
        public TypeElement resultType() {
            return VOID;
        }
    }

    public abstract static sealed class JavaSwitchOp extends ExtendedOp implements Op.Nested, Op.Lowerable
            permits JavaSwitchStatementOp, JavaSwitchExpressionOp {

        final List<Body> bodies;

        public JavaSwitchOp(ExternalizedOp def) {
            super(def);

            if (def.operands().size() != 1) {
                throw new IllegalStateException("Operation must have one operand");
            }

            // @@@ Validate
            this.bodies = def.bodyDefinitions().stream().map(bd -> bd.build(this)).toList();
        }

        JavaSwitchOp(JavaSwitchOp that, CopyContext cc, OpTransformer ot) {
            super(that, cc);

            // Copy body
            this.bodies = that.bodies.stream()
                    .map(b -> b.transform(cc, ot).build(this)).toList();
        }

        JavaSwitchOp(String name, Value target, List<Body.Builder> bodyCs) {
            super(name, List.of(target));

            // Each case is modelled as a contiguous pair of bodies
            // The first body models the case labels, and the second models the case statements
            // The labels body has a parameter whose type is target operand's type and returns a boolean value
            // The statements body has no parameters and returns void
            this.bodies = bodyCs.stream().map(bc -> bc.build(this)).toList();
        }

        @Override
        public List<Body> bodies() {
            return bodies;
        }

        @Override
        public Block.Builder lower(Block.Builder b, OpTransformer opT) {

            Value selectorExpression = b.context().getValue(operands().get(0));

            // @@@ we can add this during model generation
            // if no case null, add one that throws NPE
            if (!(selectorExpression.type() instanceof PrimitiveType) && !haveNullCase()) {
                Block.Builder throwBlock = b.block();
                throwBlock.op(_throw(
                        throwBlock.op(_new(FunctionType.functionType(JavaType.type(NullPointerException.class))))
                ));

                Block.Builder continueBlock = b.block();

                Result p = b.op(invoke(MethodRef.method(Objects.class, "equals", boolean.class, Object.class, Object.class),
                        selectorExpression, b.op(constant(J_L_OBJECT, null))));
                b.op(conditionalBranch(p, throwBlock.successor(), continueBlock.successor()));

                b = continueBlock;
            }

            List<Block.Builder> blocks = new ArrayList<>();
            for (int i = 0; i < bodies().size(); i++) {
                Block.Builder bb = b.block();
                if (i == 0) {
                    bb = b;
                }
                blocks.add(bb);
            }

            Block.Builder exit;
            if (bodies().isEmpty()) {
                exit = b;
            } else {
                exit = b.block(resultType());
                if (this instanceof JavaSwitchExpressionOp) {
                    exit.context().mapValue(result(), exit.parameters().get(0));
                }
            }

            setBranchTarget(b.context(), this, new BranchTarget(exit, null));
            // map statement body to nextExprBlock
            // this mapping will be used for lowering SwitchFallThroughOp
            for (int i = 1; i < bodies().size() - 2; i+=2) {
                setBranchTarget(b.context(), bodies().get(i), new BranchTarget(null, blocks.get(i + 2)));
            }

            for (int i = 0; i < bodies().size(); i++) {
                boolean isLabelBody = i % 2 == 0;
                Block.Builder curr = blocks.get(i);
                if (isLabelBody) {
                    Block.Builder statement = blocks.get(i + 1);
                    boolean isLastLabel = i == blocks.size() - 2;
                    Block.Builder nextLabel = isLastLabel ? null : blocks.get(i + 2);
                    curr.transformBody(bodies().get(i), List.of(selectorExpression), opT.andThen((block, op) -> {
                        switch (op) {
                            case YieldOp yop when isLastLabel && this instanceof JavaSwitchExpressionOp -> {
                                block.op(branch(statement.successor()));
                            }
                            case YieldOp yop -> block.op(conditionalBranch(
                                    block.context().getValue(yop.yieldValue()),
                                    statement.successor(),
                                    isLastLabel ? exit.successor() : nextLabel.successor()
                            ));
                            case Lowerable lop -> block = lop.lower(block);
                            default -> block.op(op);
                        }
                        return block;
                    }));
                } else { // statement body
                    curr.transformBody(bodies().get(i), blocks.get(i).parameters(), opT.andThen((block, op) -> {
                        switch (op) {
                            case YieldOp yop when this instanceof JavaSwitchStatementOp -> block.op(branch(exit.successor()));
                            case YieldOp yop when this instanceof JavaSwitchExpressionOp -> block.op(branch(exit.successor(block.context().getValue(yop.yieldValue()))));
                            case Lowerable lop -> block = lop.lower(block);
                            default -> block.op(op);
                        }
                        return block;
                    }));
                }
            }

            return exit;
        }

        boolean haveNullCase() {
            /*
            case null is modeled like this:
            (%4 : T)boolean -> {
                %5 : java.lang.Object = constant @null;
                %6 : boolean = invoke %4 %5 @"java.util.Objects::equals(java.lang.Object, java.lang.Object)boolean";
                yield %6;
            }
            * */
            for (int i = 0; i < bodies().size() - 2; i+=2) {
                Body labelBody = bodies().get(i);
                if (labelBody.blocks().size() != 1) {
                    continue; // we skip, for now
                }
                Op terminatingOp = bodies().get(i).entryBlock().terminatingOp();
                //@@@ when op pattern matching is ready, we can use it
                if (terminatingOp instanceof YieldOp yieldOp &&
                        yieldOp.yieldValue() instanceof Op.Result opr &&
                        opr.op() instanceof InvokeOp invokeOp &&
                        invokeOp.invokeDescriptor().equals(MethodRef.method(Objects.class, "equals", boolean.class, Object.class, Object.class)) &&
                        invokeOp.operands().stream().anyMatch(o -> o instanceof Op.Result r && r.op() instanceof ConstantOp cop && cop.value() == null)) {
                    return true;
                }
            }
            return false;
        }
    }

    /**
     * The switch expression operation, that can model Java language switch expressions.
     */
    @OpFactory.OpDeclaration(JavaSwitchExpressionOp.NAME)
    public static final class JavaSwitchExpressionOp extends JavaSwitchOp
            implements JavaExpression {
        public static final String NAME = "java.switch.expression";

        final TypeElement resultType;

        public JavaSwitchExpressionOp(ExternalizedOp def) {
            super(def);

            this.resultType = def.resultType();
        }

        JavaSwitchExpressionOp(JavaSwitchExpressionOp that, CopyContext cc, OpTransformer ot) {
            super(that, cc, ot);

            this.resultType = that.resultType;
        }

        @Override
        public JavaSwitchExpressionOp transform(CopyContext cc, OpTransformer ot) {
            return new JavaSwitchExpressionOp(this, cc, ot);
        }

        JavaSwitchExpressionOp(TypeElement resultType, Value target, List<Body.Builder> bodyCs) {
            super(NAME, target, bodyCs);

            this.resultType = resultType == null ? bodies.get(1).yieldType() : resultType;
        }

        @Override
        public TypeElement resultType() {
            return resultType;
        }
    }

    /**
     * The switch statement operation, that can model Java language switch statement.
     */
    @OpFactory.OpDeclaration(JavaSwitchStatementOp.NAME)
    public static final class JavaSwitchStatementOp extends JavaSwitchOp
            implements JavaStatement {
        public static final String NAME = "java.switch.statement";

        public JavaSwitchStatementOp(ExternalizedOp def) {
            super(def);
        }

        JavaSwitchStatementOp(JavaSwitchStatementOp that, CopyContext cc, OpTransformer ot) {
            super(that, cc, ot);
        }

        @Override
        public JavaSwitchStatementOp transform(CopyContext cc, OpTransformer ot) {
            return new JavaSwitchStatementOp(this, cc, ot);
        }

        JavaSwitchStatementOp(Value target, List<Body.Builder> bodyCs) {
            super(NAME, target, bodyCs);
        }

        @Override
        public TypeElement resultType() {
            return VOID;
        }
    }

    /**
     * The switch fall-through operation, that can model fall-through to the next statement in the switch block after
     * the last statement of the current switch label.
     */
    @OpFactory.OpDeclaration(JavaSwitchFallthroughOp.NAME)
    public static final class JavaSwitchFallthroughOp extends ExtendedOp
            implements Op.BodyTerminating, Op.Lowerable {
        public static final String NAME = "java.switch.fallthrough";

        public JavaSwitchFallthroughOp(ExternalizedOp def) {
            super(def);
        }

        JavaSwitchFallthroughOp(JavaSwitchFallthroughOp that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public JavaSwitchFallthroughOp transform(CopyContext cc, OpTransformer ot) {
            return new JavaSwitchFallthroughOp(this, cc);
        }

        JavaSwitchFallthroughOp() {
            super(NAME, List.of());
        }

        @Override
        public TypeElement resultType() {
            return VOID;
        }

        @Override
        public Block.Builder lower(Block.Builder b, OpTransformer opT) {
            return lower(b, BranchTarget::continueBlock);
        }

        Block.Builder lower(Block.Builder b, Function<BranchTarget, Block.Builder> f) {
            BranchTarget t = getBranchTarget(b.context(), parentBlock().parentBody());
            if (t != null) {
                b.op(branch(f.apply(t).successor()));
            } else {
                throw new IllegalStateException("No branch target for operation: " + this);
            }
            return b;
        }
    }

    /**
     * The for operation, that can model a Java language for statement.
     */
    @OpFactory.OpDeclaration(JavaForOp.NAME)
    public static final class JavaForOp extends ExtendedOp
            implements Op.Loop, Op.Lowerable, JavaStatement {

        public static final class InitBuilder {
            final Body.Builder ancestorBody;
            final List<? extends TypeElement> initTypes;

            InitBuilder(Body.Builder ancestorBody,
                        List<? extends TypeElement> initTypes) {
                this.ancestorBody = ancestorBody;
                this.initTypes = initTypes.stream().map(VarType::varType).toList();
            }

            public JavaForOp.CondBuilder init(Consumer<Block.Builder> c) {
                Body.Builder init = Body.Builder.of(ancestorBody,
                        FunctionType.functionType(TupleType.tupleType(initTypes)));
                c.accept(init.entryBlock());

                return new CondBuilder(ancestorBody, initTypes, init);
            }
        }

        public static final class CondBuilder {
            final Body.Builder ancestorBody;
            final List<? extends TypeElement> initTypes;
            final Body.Builder init;

            public CondBuilder(Body.Builder ancestorBody,
                               List<? extends TypeElement> initTypes,
                               Body.Builder init) {
                this.ancestorBody = ancestorBody;
                this.initTypes = initTypes;
                this.init = init;
            }

            public JavaForOp.UpdateBuilder cond(Consumer<Block.Builder> c) {
                Body.Builder cond = Body.Builder.of(ancestorBody,
                        FunctionType.functionType(BOOLEAN, initTypes));
                c.accept(cond.entryBlock());

                return new UpdateBuilder(ancestorBody, initTypes, init, cond);
            }
        }

        public static final class UpdateBuilder {
            final Body.Builder ancestorBody;
            final List<? extends TypeElement> initTypes;
            final Body.Builder init;
            final Body.Builder cond;

            public UpdateBuilder(Body.Builder ancestorBody,
                                 List<? extends TypeElement> initTypes,
                                 Body.Builder init, Body.Builder cond) {
                this.ancestorBody = ancestorBody;
                this.initTypes = initTypes;
                this.init = init;
                this.cond = cond;
            }

            public JavaForOp.BodyBuilder cond(Consumer<Block.Builder> c) {
                Body.Builder update = Body.Builder.of(ancestorBody,
                        FunctionType.functionType(VOID, initTypes));
                c.accept(update.entryBlock());

                return new BodyBuilder(ancestorBody, initTypes, init, cond, update);
            }

        }

        public static final class BodyBuilder {
            final Body.Builder ancestorBody;
            final List<? extends TypeElement> initTypes;
            final Body.Builder init;
            final Body.Builder cond;
            final Body.Builder update;

            public BodyBuilder(Body.Builder ancestorBody,
                               List<? extends TypeElement> initTypes,
                               Body.Builder init, Body.Builder cond, Body.Builder update) {
                this.ancestorBody = ancestorBody;
                this.initTypes = initTypes;
                this.init = init;
                this.cond = cond;
                this.update = update;
            }

            public JavaForOp body(Consumer<Block.Builder> c) {
                Body.Builder body = Body.Builder.of(ancestorBody,
                        FunctionType.functionType(VOID, initTypes));
                c.accept(body.entryBlock());

                return new JavaForOp(init, cond, update, body);
            }
        }

        static final String NAME = "java.for";

        final Body init;
        final Body cond;
        final Body update;
        final Body body;

        public static JavaForOp create(ExternalizedOp def) {
            return new JavaForOp(def);
        }

        public JavaForOp(ExternalizedOp def) {
            super(def);

            this.init = def.bodyDefinitions().get(0).build(this);
            this.cond = def.bodyDefinitions().get(1).build(this);
            this.update = def.bodyDefinitions().get(2).build(this);
            this.body = def.bodyDefinitions().get(3).build(this);
        }

        JavaForOp(JavaForOp that, CopyContext cc, OpTransformer ot) {
            super(that, cc);

            this.init = that.init.transform(cc, ot).build(this);
            this.cond = that.cond.transform(cc, ot).build(this);
            this.update = that.update.transform(cc, ot).build(this);
            this.body = that.body.transform(cc, ot).build(this);
        }

        @Override
        public JavaForOp transform(CopyContext cc, OpTransformer ot) {
            return new JavaForOp(this, cc, ot);
        }

        JavaForOp(Body.Builder initC,
                  Body.Builder condC,
                  Body.Builder updateC,
                  Body.Builder bodyC) {
            super(NAME, List.of());

            this.init = initC.build(this);

            this.cond = condC.build(this);

            this.update = updateC.build(this);
            if (!update.bodyType().returnType().equals(VOID)) {
                throw new IllegalArgumentException("Update should return void: " + update.bodyType());
            }

            this.body = bodyC.build(this);
            if (!body.bodyType().returnType().equals(VOID)) {
                throw new IllegalArgumentException("Body should return void: " + body.bodyType());
            }
        }

        @Override
        public List<Body> bodies() {
            return List.of(init, cond, update, body);
        }

        public Body init() {
            return init;
        }

        public Body cond() {
            return cond;
        }

        public Body update() {
            return update;
        }

        @Override
        public Body loopBody() {
            return body;
        }

        @Override
        public Block.Builder lower(Block.Builder b, OpTransformer opT) {
            Block.Builder header = b.block();
            Block.Builder body = b.block();
            Block.Builder update = b.block();
            Block.Builder exit = b.block();

            List<Value> initValues = new ArrayList<>();
            // @@@ Init body has one yield operation yielding
            //  void, a single variable, or a tuple of one or more variables
            b.transformBody(init, List.of(), opT.andThen((block, op) -> {
                if (op instanceof CoreOp.TupleOp) {
                    // Drop Tuple if a yielded
                    boolean isResult = op.result().uses().size() == 1 &&
                            op.result().uses().stream().allMatch(r -> r.op() instanceof YieldOp);
                    if (!isResult) {
                        block.op(op);
                    }
                } else if (op instanceof YieldOp yop) {
                    if (yop.yieldValue() == null) {
                        block.op(branch(header.successor()));
                        return block;
                    } else if (yop.yieldValue() instanceof Result or) {
                        if (or.op() instanceof CoreOp.TupleOp top) {
                            initValues.addAll(block.context().getValues(top.operands()));
                        } else {
                            initValues.addAll(block.context().getValues(yop.operands()));
                        }
                        block.op(branch(header.successor()));
                        return block;
                    }

                    throw new IllegalStateException("Bad yield operation");
                } else {
                    // @@@ Composition of lowerable ops
                    block.op(op);
                }
                return block;
            }));

            header.transformBody(cond, initValues, opT.andThen((block, op) -> {
                if (op instanceof YieldOp yo) {
                    block.op(conditionalBranch(block.context().getValue(yo.yieldValue()),
                            body.successor(), exit.successor()));
                } else if (op instanceof Lowerable lop) {
                    // @@@ Composition of lowerable ops
                    block = lop.lower(block, opT);
                } else {
                    block.op(op);
                }
                return block;
            }));

            setBranchTarget(b.context(), this, new BranchTarget(exit, update));

            body.transformBody(this.body, initValues, opT.andThen((block, op) -> {
                // @@@ Composition of lowerable ops
                if (op instanceof Lowerable lop) {
                    block = lop.lower(block, opT);
                } else {
                    block.op(op);
                }
                return block;
            }));

            update.transformBody(this.update, initValues, opT.andThen((block, op) -> {
                if (op instanceof YieldOp) {
                    block.op(branch(header.successor()));
                } else {
                    // @@@ Composition of lowerable ops
                    block.op(op);
                }
                return block;
            }));

            return exit;
        }

        @Override
        public TypeElement resultType() {
            return VOID;
        }
    }

    /**
     * The enhanced for operation, that can model a Java language enhanced for statement.
     */
    @OpFactory.OpDeclaration(JavaEnhancedForOp.NAME)
    public static final class JavaEnhancedForOp extends ExtendedOp
            implements Op.Loop, Op.Lowerable, JavaStatement {

        public static final class ExpressionBuilder {
            final Body.Builder ancestorBody;
            final TypeElement iterableType;
            final TypeElement elementType;

            ExpressionBuilder(Body.Builder ancestorBody,
                              TypeElement iterableType, TypeElement elementType) {
                this.ancestorBody = ancestorBody;
                this.iterableType = iterableType;
                this.elementType = elementType;
            }

            public DefinitionBuilder expression(Consumer<Block.Builder> c) {
                Body.Builder expression = Body.Builder.of(ancestorBody,
                        FunctionType.functionType(iterableType));
                c.accept(expression.entryBlock());

                return new DefinitionBuilder(ancestorBody, elementType, expression);
            }
        }

        public static final class DefinitionBuilder {
            final Body.Builder ancestorBody;
            final TypeElement elementType;
            final Body.Builder expression;

            DefinitionBuilder(Body.Builder ancestorBody,
                              TypeElement elementType, Body.Builder expression) {
                this.ancestorBody = ancestorBody;
                this.elementType = elementType;
                this.expression = expression;
            }

            public BodyBuilder definition(Consumer<Block.Builder> c) {
                return definition(elementType, c);
            }

            public BodyBuilder definition(TypeElement bodyElementType, Consumer<Block.Builder> c) {
                Body.Builder definition = Body.Builder.of(ancestorBody,
                        FunctionType.functionType(bodyElementType, elementType));
                c.accept(definition.entryBlock());

                return new BodyBuilder(ancestorBody, elementType, expression, definition);
            }
        }

        public static final class BodyBuilder {
            final Body.Builder ancestorBody;
            final TypeElement elementType;
            final Body.Builder expression;
            final Body.Builder definition;

            BodyBuilder(Body.Builder ancestorBody,
                        TypeElement elementType, Body.Builder expression, Body.Builder definition) {
                this.ancestorBody = ancestorBody;
                this.elementType = elementType;
                this.expression = expression;
                this.definition = definition;
            }

            public JavaEnhancedForOp body(Consumer<Block.Builder> c) {
                Body.Builder body = Body.Builder.of(ancestorBody,
                        FunctionType.functionType(VOID, elementType));
                c.accept(body.entryBlock());

                return new JavaEnhancedForOp(expression, definition, body);
            }
        }

        static final String NAME = "java.enhancedFor";

        final Body expression;
        final Body init;
        final Body body;

        public static JavaEnhancedForOp create(ExternalizedOp def) {
            return new JavaEnhancedForOp(def);
        }

        public JavaEnhancedForOp(ExternalizedOp def) {
            super(def);

            this.expression = def.bodyDefinitions().get(0).build(this);
            this.init = def.bodyDefinitions().get(1).build(this);
            this.body = def.bodyDefinitions().get(2).build(this);
        }

        JavaEnhancedForOp(JavaEnhancedForOp that, CopyContext cc, OpTransformer ot) {
            super(that, cc);

            this.expression = that.expression.transform(cc, ot).build(this);
            this.init = that.init.transform(cc, ot).build(this);
            this.body = that.body.transform(cc, ot).build(this);
        }

        @Override
        public JavaEnhancedForOp transform(CopyContext cc, OpTransformer ot) {
            return new JavaEnhancedForOp(this, cc, ot);
        }

        JavaEnhancedForOp(Body.Builder expressionC, Body.Builder initC, Body.Builder bodyC) {
            super(NAME, List.of());

            this.expression = expressionC.build(this);
            if (expression.bodyType().returnType().equals(VOID)) {
                throw new IllegalArgumentException("Expression should return non-void value: " + expression.bodyType());
            }
            if (!expression.bodyType().parameterTypes().isEmpty()) {
                throw new IllegalArgumentException("Expression should have zero parameters: " + expression.bodyType());
            }

            this.init = initC.build(this);
            if (init.bodyType().returnType().equals(VOID)) {
                throw new IllegalArgumentException("Initialization should return non-void value: " + init.bodyType());
            }
            if (init.bodyType().parameterTypes().size() != 1) {
                throw new IllegalArgumentException("Initialization should have one parameter: " + init.bodyType());
            }

            this.body = bodyC.build(this);
            if (!body.bodyType().returnType().equals(VOID)) {
                throw new IllegalArgumentException("Body should return void: " + body.bodyType());
            }
            if (body.bodyType().parameterTypes().size() != 1) {
                throw new IllegalArgumentException("Body should have one parameter: " + body.bodyType());
            }
        }

        @Override
        public List<Body> bodies() {
            return List.of(expression, init, body);
        }

        public Body expression() {
            return expression;
        }

        public Body initialization() {
            return init;
        }

        @Override
        public Body loopBody() {
            return body;
        }

        static final MethodRef ITERABLE_ITERATOR = MethodRef.method(Iterable.class, "iterator", Iterator.class);
        static final MethodRef ITERATOR_HAS_NEXT = MethodRef.method(Iterator.class, "hasNext", boolean.class);
        static final MethodRef ITERATOR_NEXT = MethodRef.method(Iterator.class, "next", Object.class);

        @Override
        public Block.Builder lower(Block.Builder b, OpTransformer opT) {
            JavaType elementType = (JavaType) init.entryBlock().parameters().get(0).type();
            boolean isArray = expression.bodyType().returnType() instanceof ArrayType;

            Block.Builder preHeader = b.block(expression.bodyType().returnType());
            Block.Builder header = b.block(isArray ? List.of(INT) : List.of());
            Block.Builder init = b.block();
            Block.Builder body = b.block();
            Block.Builder exit = b.block();

            b.transformBody(expression, List.of(), opT.andThen((block, op) -> {
                if (op instanceof YieldOp yop) {
                    Value loopSource = block.context().getValue(yop.yieldValue());
                    block.op(branch(preHeader.successor(loopSource)));
                } else {
                    // @@@ Composition of lowerable ops
                    block.op(op);
                }
                return block;
            }));

            if (isArray) {
                Value array = preHeader.parameters().get(0);
                Value arrayLength = preHeader.op(arrayLength(array));
                Value i = preHeader.op(constant(INT, 0));
                preHeader.op(branch(header.successor(i)));

                i = header.parameters().get(0);
                Value p = header.op(lt(i, arrayLength));
                header.op(conditionalBranch(p, init.successor(), exit.successor()));

                Value e = init.op(arrayLoadOp(array, i));
                List<Value> initValues = new ArrayList<>();
                // @@@ Init body has one yield operation yielding a single variable
                init.transformBody(this.init, List.of(e), (block, op) -> {
                    if (op instanceof YieldOp yop) {
                        initValues.addAll(block.context().getValues(yop.operands()));
                        block.op(branch(body.successor()));
                    } else {
                        // @@@ Composition of lowerable ops
                        block.op(op);
                    }
                    return block;
                });

                Block.Builder update = b.block();
                setBranchTarget(b.context(), this, new BranchTarget(exit, update));

                body.transformBody(this.body, initValues, opT.andThen((block, op) -> {
                    // @@@ Composition of lowerable ops
                    if (op instanceof Lowerable lop) {
                        block = lop.lower(block, opT);
                    } else {
                        block.op(op);
                    }
                    return block;
                }));

                i = update.op(add(i, update.op(constant(INT, 1))));
                update.op(branch(header.successor(i)));
            } else {
                JavaType iterable = parameterized(type(Iterator.class), elementType);
                Value iterator = preHeader.op(CoreOp.invoke(iterable, ITERABLE_ITERATOR, preHeader.parameters().get(0)));
                preHeader.op(branch(header.successor()));

                Value p = header.op(CoreOp.invoke(ITERATOR_HAS_NEXT, iterator));
                header.op(conditionalBranch(p, init.successor(), exit.successor()));

                Value e = init.op(CoreOp.invoke(elementType, ITERATOR_NEXT, iterator));
                List<Value> initValues = new ArrayList<>();
                init.transformBody(this.init, List.of(e), opT.andThen((block, op) -> {
                    if (op instanceof YieldOp yop) {
                        initValues.addAll(block.context().getValues(yop.operands()));
                        block.op(branch(body.successor()));
                    } else {
                        // @@@ Composition of lowerable ops
                        block.op(op);
                    }
                    return block;
                }));

                setBranchTarget(b.context(), this, new BranchTarget(exit, header));

                body.transformBody(this.body, initValues, opT.andThen((block, op) -> {
                    // @@@ Composition of lowerable ops
                    if (op instanceof Lowerable lop) {
                        block = lop.lower(block, opT);
                    } else {
                        block.op(op);
                    }
                    return block;
                }));
            }

            return exit;
        }

        @Override
        public TypeElement resultType() {
            return VOID;
        }
    }

    /**
     * The while operation, that can model a Java language while statement.
     */
    @OpFactory.OpDeclaration(JavaWhileOp.NAME)
    public static final class JavaWhileOp extends ExtendedOp
            implements Op.Loop, Op.Lowerable, JavaStatement {

        public static class PredicateBuilder {
            final Body.Builder ancestorBody;

            PredicateBuilder(Body.Builder ancestorBody) {
                this.ancestorBody = ancestorBody;
            }

            public JavaWhileOp.BodyBuilder predicate(Consumer<Block.Builder> c) {
                Body.Builder body = Body.Builder.of(ancestorBody, FunctionType.functionType(BOOLEAN));
                c.accept(body.entryBlock());

                return new JavaWhileOp.BodyBuilder(ancestorBody, body);
            }
        }

        public static class BodyBuilder {
            final Body.Builder ancestorBody;
            private final Body.Builder predicate;

            BodyBuilder(Body.Builder ancestorBody, Body.Builder predicate) {
                this.ancestorBody = ancestorBody;
                this.predicate = predicate;
            }

            public JavaWhileOp body(Consumer<Block.Builder> c) {
                Body.Builder body = Body.Builder.of(ancestorBody, FunctionType.VOID);
                c.accept(body.entryBlock());

                return new JavaWhileOp(List.of(predicate, body));
            }
        }

        private static final String NAME = "java.while";

        private final List<Body> bodies;

        public JavaWhileOp(ExternalizedOp def) {
            super(def);

            // @@@ Validate
            this.bodies = def.bodyDefinitions().stream().map(bd -> bd.build(this)).toList();
        }

        JavaWhileOp(List<Body.Builder> bodyCs) {
            super(NAME, List.of());

            this.bodies = bodyCs.stream().map(bc -> bc.build(this)).toList();
        }

        JavaWhileOp(Body.Builder predicate, Body.Builder body) {
            super(NAME, List.of());

            Objects.requireNonNull(body);

            this.bodies = Stream.of(predicate, body).filter(Objects::nonNull)
                    .map(bc -> bc.build(this)).toList();

            // @@@ This will change with pattern bindings
            if (!bodies.get(0).bodyType().equals(FunctionType.functionType(BOOLEAN))) {
                throw new IllegalArgumentException(
                        "Predicate body descriptor should be " + FunctionType.functionType(BOOLEAN) +
                                " but is " + bodies.get(0).bodyType());
            }
            if (!bodies.get(1).bodyType().equals(FunctionType.VOID)) {
                throw new IllegalArgumentException(
                        "Body descriptor should be " + FunctionType.functionType(VOID) +
                                " but is " + bodies.get(1).bodyType());
            }
        }

        JavaWhileOp(JavaWhileOp that, CopyContext cc, OpTransformer ot) {
            super(that, cc);

            this.bodies = that.bodies.stream()
                    .map(b -> b.transform(cc, ot).build(this)).toList();
        }

        @Override
        public JavaWhileOp transform(CopyContext cc, OpTransformer ot) {
            return new JavaWhileOp(this, cc, ot);
        }

        @Override
        public List<Body> bodies() {
            return bodies;
        }

        public Body predicateBody() {
            return bodies.get(0);
        }

        @Override
        public Body loopBody() {
            return bodies.get(1);
        }

        @Override
        public Block.Builder lower(Block.Builder b, OpTransformer opT) {
            Block.Builder header = b.block();
            Block.Builder body = b.block();
            Block.Builder exit = b.block();

            b.op(branch(header.successor()));

            header.transformBody(predicateBody(), List.of(), opT.andThen((block, op) -> {
                if (op instanceof CoreOp.YieldOp yo) {
                    block.op(conditionalBranch(block.context().getValue(yo.yieldValue()),
                            body.successor(), exit.successor()));
                } else if (op instanceof Lowerable lop) {
                    // @@@ Composition of lowerable ops
                    block = lop.lower(block, opT);
                } else {
                    block.op(op);
                }
                return block;
            }));

            setBranchTarget(b.context(), this, new BranchTarget(exit, header));

            body.transformBody(loopBody(), List.of(), opT.andThen((block, op) -> {
                // @@@ Composition of lowerable ops
                if (op instanceof Lowerable lop) {
                    block = lop.lower(block, opT);
                } else {
                    block.op(op);
                }
                return block;
            }));

            return exit;
        }

        @Override
        public TypeElement resultType() {
            return VOID;
        }
    }

    /**
     * The do-while operation, that can model a Java language do statement.
     */
    // @@@ Unify JavaDoWhileOp and JavaWhileOp with common abstract superclass
    @OpFactory.OpDeclaration(JavaDoWhileOp.NAME)
    public static final class JavaDoWhileOp extends ExtendedOp
            implements Op.Loop, Op.Lowerable, JavaStatement {

        public static class PredicateBuilder {
            final Body.Builder ancestorBody;
            private final Body.Builder body;

            PredicateBuilder(Body.Builder ancestorBody, Body.Builder body) {
                this.ancestorBody = ancestorBody;
                this.body = body;
            }

            public JavaDoWhileOp predicate(Consumer<Block.Builder> c) {
                Body.Builder predicate = Body.Builder.of(ancestorBody, FunctionType.functionType(BOOLEAN));
                c.accept(predicate.entryBlock());

                return new JavaDoWhileOp(List.of(body, predicate));
            }
        }

        public static class BodyBuilder {
            final Body.Builder ancestorBody;

            BodyBuilder(Body.Builder ancestorBody) {
                this.ancestorBody = ancestorBody;
            }

            public JavaDoWhileOp.PredicateBuilder body(Consumer<Block.Builder> c) {
                Body.Builder body = Body.Builder.of(ancestorBody, FunctionType.VOID);
                c.accept(body.entryBlock());

                return new JavaDoWhileOp.PredicateBuilder(ancestorBody, body);
            }
        }

        private static final String NAME = "java.do.while";

        private final List<Body> bodies;

        public JavaDoWhileOp(ExternalizedOp def) {
            super(def);

            // @@@ Validate
            this.bodies = def.bodyDefinitions().stream().map(bd -> bd.build(this)).toList();
        }

        JavaDoWhileOp(List<Body.Builder> bodyCs) {
            super(NAME, List.of());

            this.bodies = bodyCs.stream().map(bc -> bc.build(this)).toList();
        }

        JavaDoWhileOp(Body.Builder body, Body.Builder predicate) {
            super(NAME, List.of());

            Objects.requireNonNull(body);

            this.bodies = Stream.of(body, predicate).filter(Objects::nonNull)
                    .map(bc -> bc.build(this)).toList();

            if (!bodies.get(0).bodyType().equals(FunctionType.VOID)) {
                throw new IllegalArgumentException(
                        "Body descriptor should be " + FunctionType.functionType(VOID) +
                                " but is " + bodies.get(1).bodyType());
            }
            if (!bodies.get(1).bodyType().equals(FunctionType.functionType(BOOLEAN))) {
                throw new IllegalArgumentException(
                        "Predicate body descriptor should be " + FunctionType.functionType(BOOLEAN) +
                                " but is " + bodies.get(0).bodyType());
            }
        }

        JavaDoWhileOp(JavaDoWhileOp that, CopyContext cc, OpTransformer ot) {
            super(that, cc);

            this.bodies = that.bodies.stream()
                    .map(b -> b.transform(cc, ot).build(this)).toList();
        }

        @Override
        public JavaDoWhileOp transform(CopyContext cc, OpTransformer ot) {
            return new JavaDoWhileOp(this, cc, ot);
        }

        @Override
        public List<Body> bodies() {
            return bodies;
        }

        public Body predicateBody() {
            return bodies.get(1);
        }

        @Override
        public Body loopBody() {
            return bodies.get(0);
        }

        @Override
        public Block.Builder lower(Block.Builder b, OpTransformer opT) {
            Block.Builder body = b.block();
            Block.Builder header = b.block();
            Block.Builder exit = b.block();

            b.op(branch(body.successor()));

            setBranchTarget(b.context(), this, new BranchTarget(exit, header));

            body.transformBody(loopBody(), List.of(), opT.andThen((block, op) -> {
                // @@@ Composition of lowerable ops
                if (op instanceof Lowerable lop) {
                    block = lop.lower(block, opT);
                } else {
                    block.op(op);
                }
                return block;
            }));

            header.transformBody(predicateBody(), List.of(), opT.andThen((block, op) -> {
                if (op instanceof CoreOp.YieldOp yo) {
                    block.op(conditionalBranch(block.context().getValue(yo.yieldValue()),
                            body.successor(), exit.successor()));
                } else if (op instanceof Lowerable lop) {
                    // @@@ Composition of lowerable ops
                    block = lop.lower(block, opT);
                } else {
                    block.op(op);
                }
                return block;
            }));

            return exit;
        }

        @Override
        public TypeElement resultType() {
            return VOID;
        }
    }

    /**
     * The conditional-and-or operation, that can model Java language condition-or or conditional-and expressions.
     */
    public sealed static abstract class JavaConditionalOp extends ExtendedOp
            implements Op.Nested, Op.Lowerable, JavaExpression {
        final List<Body> bodies;

        public JavaConditionalOp(ExternalizedOp def) {
            super(def);

            if (!def.operands().isEmpty()) {
                throw new IllegalStateException("Operation must have no operands");
            }

            // @@@ Validate

            this.bodies = def.bodyDefinitions().stream().map(bd -> bd.build(this)).toList();
        }

        JavaConditionalOp(JavaConditionalOp that, CopyContext cc, OpTransformer ot) {
            super(that, cc);

            // Copy body
            this.bodies = that.bodies.stream().map(b -> b.transform(cc, ot).build(this)).toList();
        }

        JavaConditionalOp(String name, List<Body.Builder> bodyCs) {
            super(name, List.of());

            if (bodyCs.isEmpty()) {
                throw new IllegalArgumentException();
            }

            this.bodies = bodyCs.stream().map(bc -> bc.build(this)).toList();
            for (Body b : bodies) {
                if (!b.bodyType().equals(FunctionType.functionType(BOOLEAN))) {
                    throw new IllegalArgumentException("Body conditional body descriptor: " + b.bodyType());
                }
            }
        }

        @Override
        public List<Body> bodies() {
            return bodies;
        }

        static Block.Builder lower(Block.Builder startBlock, OpTransformer opT, JavaConditionalOp cop) {
            List<Body> bodies = cop.bodies();

            Block.Builder exit = startBlock.block();
            TypeElement oprType = cop.result().type();
            Block.Parameter arg = exit.parameter(oprType);
            startBlock.context().mapValue(cop.result(), arg);

            // Transform bodies in reverse order
            // This makes available the blocks to be referenced as successors in prior blocks

            Block.Builder pred = null;
            for (int i = bodies.size() - 1; i >= 0; i--) {
                OpTransformer opt;
                if (i == bodies.size() - 1) {
                    opt = (block, op) -> {
                        if (op instanceof CoreOp.YieldOp yop) {
                            Value p = block.context().getValue(yop.yieldValue());
                            block.op(branch(exit.successor(p)));
                        } else if (op instanceof Lowerable lop) {
                            // @@@ Composition of lowerable ops
                            block = lop.lower(block, opT);
                        } else {
                            // Copy
                            block.apply(op);
                        }
                        return block;
                    };
                } else {
                    Block.Builder nextPred = pred;
                    opt = (block, op) -> {
                        if (op instanceof CoreOp.YieldOp yop) {
                            Value p = block.context().getValue(yop.yieldValue());
                            if (cop instanceof JavaConditionalAndOp) {
                                block.op(conditionalBranch(p, nextPred.successor(), exit.successor(p)));
                            } else {
                                block.op(conditionalBranch(p, exit.successor(p), nextPred.successor()));
                            }
                        } else if (op instanceof Lowerable lop) {
                            // @@@ Composition of lowerable ops
                            block = lop.lower(block, opT);
                        } else {
                            // Copy
                            block.apply(op);
                        }
                        return block;
                    };
                }

                Body fromPred = bodies.get(i);
                if (i == 0) {
                    startBlock.transformBody(fromPred, List.of(), opt);
                } else {
                    pred = startBlock.block(fromPred.bodyType().parameterTypes());
                    pred.transformBody(fromPred, pred.parameters(), opT.andThen(opt));
                }
            }

            return exit;
        }

        @Override
        public TypeElement resultType() {
            return BOOLEAN;
        }
    }

    /**
     * The conditional-and operation, that can model Java language conditional-and expressions.
     */
    @OpFactory.OpDeclaration(JavaConditionalAndOp.NAME)
    public static final class JavaConditionalAndOp extends JavaConditionalOp {

        public static class Builder {
            final Body.Builder ancestorBody;
            final List<Body.Builder> bodies;

            Builder(Body.Builder ancestorBody, Consumer<Block.Builder> lhs, Consumer<Block.Builder> rhs) {
                this.ancestorBody = ancestorBody;
                this.bodies = new ArrayList<>();
                and(lhs);
                and(rhs);
            }

            public Builder and(Consumer<Block.Builder> c) {
                Body.Builder body = Body.Builder.of(ancestorBody, FunctionType.functionType(BOOLEAN));
                c.accept(body.entryBlock());
                bodies.add(body);

                return this;
            }

            public JavaConditionalAndOp build() {
                return new JavaConditionalAndOp(bodies);
            }
        }

        public static final String NAME = "java.cand";

        public JavaConditionalAndOp(ExternalizedOp def) {
            super(def);
        }

        JavaConditionalAndOp(JavaConditionalAndOp that, CopyContext cc, OpTransformer ot) {
            super(that, cc, ot);
        }

        @Override
        public JavaConditionalAndOp transform(CopyContext cc, OpTransformer ot) {
            return new JavaConditionalAndOp(this, cc, ot);
        }

        JavaConditionalAndOp(List<Body.Builder> bodyCs) {
            super(NAME, bodyCs);
        }

        @Override
        public Block.Builder lower(Block.Builder b, OpTransformer opT) {
            return lower(b, opT, this);
        }
    }

    /**
     * The conditional-or operation, that can model Java language conditional-or expressions.
     */
    @OpFactory.OpDeclaration(JavaConditionalOrOp.NAME)
    public static final class JavaConditionalOrOp extends JavaConditionalOp {

        public static class Builder {
            final Body.Builder ancestorBody;
            final List<Body.Builder> bodies;

            Builder(Body.Builder ancestorBody, Consumer<Block.Builder> lhs, Consumer<Block.Builder> rhs) {
                this.ancestorBody = ancestorBody;
                this.bodies = new ArrayList<>();
                or(lhs);
                or(rhs);
            }

            public Builder or(Consumer<Block.Builder> c) {
                Body.Builder body = Body.Builder.of(ancestorBody, FunctionType.functionType(BOOLEAN));
                c.accept(body.entryBlock());
                bodies.add(body);

                return this;
            }

            public JavaConditionalOrOp build() {
                return new JavaConditionalOrOp(bodies);
            }
        }

        public static final String NAME = "java.cor";

        public JavaConditionalOrOp(ExternalizedOp def) {
            super(def);
        }

        JavaConditionalOrOp(JavaConditionalOrOp that, CopyContext cc, OpTransformer ot) {
            super(that, cc, ot);
        }

        @Override
        public JavaConditionalOrOp transform(CopyContext cc, OpTransformer ot) {
            return new JavaConditionalOrOp(this, cc, ot);
        }

        JavaConditionalOrOp(List<Body.Builder> bodyCs) {
            super(NAME, bodyCs);
        }

        @Override
        public Block.Builder lower(Block.Builder b, OpTransformer opT) {
            return lower(b, opT, this);
        }
    }

    /**
     * The conditional operation, that can model Java language conditional operator {@code ?} expressions.
     */
    @OpFactory.OpDeclaration(JavaConditionalExpressionOp.NAME)
    public static final class JavaConditionalExpressionOp extends ExtendedOp
            implements Op.Nested, Op.Lowerable, JavaExpression {

        public static final String NAME = "java.cexpression";

        final TypeElement resultType;
        // {cond, truepart, falsepart}
        final List<Body> bodies;

        public JavaConditionalExpressionOp(ExternalizedOp def) {
            super(def);

            if (!def.operands().isEmpty()) {
                throw new IllegalStateException("Operation must have no operands");
            }

            // @@@ Validate

            this.bodies = def.bodyDefinitions().stream().map(bd -> bd.build(this)).toList();
            this.resultType = def.resultType();
        }

        JavaConditionalExpressionOp(JavaConditionalExpressionOp that, CopyContext cc, OpTransformer ot) {
            super(that, cc);

            // Copy body
            this.bodies = that.bodies.stream()
                    .map(b -> b.transform(cc, ot).build(this)).toList();
            this.resultType = that.resultType;
        }

        @Override
        public JavaConditionalExpressionOp transform(CopyContext cc, OpTransformer ot) {
            return new JavaConditionalExpressionOp(this, cc, ot);
        }

        JavaConditionalExpressionOp(TypeElement expressionType, List<Body.Builder> bodyCs) {
            super(NAME, List.of());

            this.bodies = bodyCs.stream().map(bc -> bc.build(this)).toList();
            // @@@ when expressionType is null, we assume truepart and falsepart have the same yieldType
            this.resultType = expressionType == null ? bodies.get(1).yieldType() : expressionType;

            if (bodies.size() < 3) {
                throw new IllegalArgumentException("Incorrect number of bodies: " + bodies.size());
            }

            Body cond = bodies.get(0);
            if (!cond.bodyType().equals(FunctionType.functionType(BOOLEAN))) {
                throw new IllegalArgumentException("Illegal cond body descriptor: " + cond.bodyType());
            }
        }

        @Override
        public List<Body> bodies() {
            return bodies;
        }

        @Override
        public Block.Builder lower(Block.Builder b, OpTransformer opT) {
            Block.Builder exit = b.block(resultType());
            exit.context().mapValue(result(), exit.parameters().get(0));

            setBranchTarget(b.context(), this, new BranchTarget(exit, null));

            List<Block.Builder> builders = List.of(b.block(), b.block());
            b.transformBody(bodies.get(0), List.of(), opT.andThen((block, op) -> {
                if (op instanceof YieldOp yo) {
                    block.op(conditionalBranch(block.context().getValue(yo.yieldValue()),
                            builders.get(0).successor(), builders.get(1).successor()));
                } else if (op instanceof Lowerable lop) {
                    // @@@ Composition of lowerable ops
                    block = lop.lower(block, opT);
                } else {
                    block.op(op);
                }
                return block;
            }));

            for (int i = 0; i < 2; i++) {
                builders.get(i).transformBody(bodies.get(i + 1), List.of(), opT.andThen((block, op) -> {
                    if (op instanceof YieldOp yop) {
                        block.op(branch(exit.successor(block.context().getValue(yop.yieldValue()))));
                    } else if (op instanceof Lowerable lop) {
                        // @@@ Composition of lowerable ops
                        block = lop.lower(block, opT);
                    } else {
                        block.op(op);
                    }
                    return block;
                }));
            }

            return exit;
        }

        @Override
        public TypeElement resultType() {
            return resultType;
        }
    }

    /**
     * The try operation, that can model Java language try statements.
     */
    @OpFactory.OpDeclaration(JavaTryOp.NAME)
    public static final class JavaTryOp extends ExtendedOp
            implements Op.Nested, Op.Lowerable, JavaStatement {

        public static final class BodyBuilder {
            final Body.Builder ancestorBody;
            final List<? extends TypeElement> resourceTypes;
            final Body.Builder resources;

            BodyBuilder(Body.Builder ancestorBody, List<? extends TypeElement> resourceTypes, Body.Builder resources) {
                this.ancestorBody = ancestorBody;
                this.resourceTypes = resourceTypes;
                this.resources = resources;
            }

            public CatchBuilder body(Consumer<Block.Builder> c) {
                Body.Builder body = Body.Builder.of(ancestorBody,
                        FunctionType.functionType(VOID, resourceTypes));
                c.accept(body.entryBlock());

                return new CatchBuilder(ancestorBody, resources, body);
            }
        }

        public static final class CatchBuilder {
            final Body.Builder ancestorBody;
            final Body.Builder resources;
            final Body.Builder body;
            final List<Body.Builder> catchers;

            CatchBuilder(Body.Builder ancestorBody, Body.Builder resources, Body.Builder body) {
                this.ancestorBody = ancestorBody;
                this.resources = resources;
                this.body = body;
                this.catchers = new ArrayList<>();
            }

            // @@@ multi-catch
            public CatchBuilder _catch(TypeElement exceptionType, Consumer<Block.Builder> c) {
                Body.Builder _catch = Body.Builder.of(ancestorBody,
                        FunctionType.functionType(VOID, exceptionType));
                c.accept(_catch.entryBlock());
                catchers.add(_catch);

                return this;
            }

            public JavaTryOp _finally(Consumer<Block.Builder> c) {
                Body.Builder _finally = Body.Builder.of(ancestorBody, FunctionType.VOID);
                c.accept(_finally.entryBlock());

                return new JavaTryOp(resources, body, catchers, _finally);
            }

            public JavaTryOp noFinalizer() {
                return new JavaTryOp(resources, body, catchers, null);
            }
        }

        static final String NAME = "java.try";

        final Body resources;
        final Body body;
        final List<Body> catchers;
        final Body finalizer;

        public static JavaTryOp create(ExternalizedOp def) {
            return new JavaTryOp(def);
        }

        public JavaTryOp(ExternalizedOp def) {
            super(def);

            List<Body> bodies = def.bodyDefinitions().stream().map(b -> b.build(this)).toList();
            Body first = bodies.get(0);
            if (first.bodyType().returnType().equals(VOID)) {
                this.resources = null;
                this.body = first;
            } else {
                this.resources = first;
                this.body = bodies.get(1);
            }

            Body last = bodies.get(bodies.size() - 1);
            if (last.bodyType().parameterTypes().isEmpty()) {
                this.finalizer = last;
            } else {
                this.finalizer = null;
            }
            this.catchers = bodies.subList(
                    resources == null ? 1 : 2,
                    bodies.size() - (finalizer == null ? 0 : 1));
        }

        JavaTryOp(JavaTryOp that, CopyContext cc, OpTransformer ot) {
            super(that, cc);

            if (that.resources != null) {
                this.resources = that.resources.transform(cc, ot).build(this);
            } else {
                this.resources = null;
            }
            this.body = that.body.transform(cc, ot).build(this);
            this.catchers = that.catchers.stream()
                    .map(b -> b.transform(cc, ot).build(this))
                    .toList();
            if (that.finalizer != null) {
                this.finalizer = that.finalizer.transform(cc, ot).build(this);
            } else {
                this.finalizer = null;
            }
        }

        @Override
        public JavaTryOp transform(CopyContext cc, OpTransformer ot) {
            return new JavaTryOp(this, cc, ot);
        }

        JavaTryOp(Body.Builder resourcesC,
                  Body.Builder bodyC,
                  List<Body.Builder> catchersC,
                  Body.Builder finalizerC) {
            super(NAME, List.of());

            if (resourcesC != null) {
                this.resources = resourcesC.build(this);
                if (resources.bodyType().returnType().equals(VOID)) {
                    throw new IllegalArgumentException("Resources should not return void: " + resources.bodyType());
                }
                if (!resources.bodyType().parameterTypes().isEmpty()) {
                    throw new IllegalArgumentException("Resources should have zero parameters: " + resources.bodyType());
                }
            } else {
                this.resources = null;
            }

            this.body = bodyC.build(this);
            if (!body.bodyType().returnType().equals(VOID)) {
                throw new IllegalArgumentException("Try should return void: " + body.bodyType());
            }

            this.catchers = catchersC.stream().map(c -> c.build(this)).toList();
            for (Body _catch : catchers) {
                if (!_catch.bodyType().returnType().equals(VOID)) {
                    throw new IllegalArgumentException("Catch should return void: " + _catch.bodyType());
                }
                if (_catch.bodyType().parameterTypes().size() != 1) {
                    throw new IllegalArgumentException("Catch should have zero parameters: " + _catch.bodyType());
                }
            }

            if (finalizerC != null) {
                this.finalizer = finalizerC.build(this);
                if (!finalizer.bodyType().returnType().equals(VOID)) {
                    throw new IllegalArgumentException("Finally should return void: " + finalizer.bodyType());
                }
                if (!finalizer.bodyType().parameterTypes().isEmpty()) {
                    throw new IllegalArgumentException("Finally should have zero parameters: " + finalizer.bodyType());
                }
            } else {
                this.finalizer = null;
            }
        }

        @Override
        public List<Body> bodies() {
            ArrayList<Body> bodies = new ArrayList<>();
            if (resources != null) {
                bodies.add(resources);
            }
            bodies.add(body);
            bodies.addAll(catchers);
            if (finalizer != null) {
                bodies.add(finalizer);
            }
            return bodies;
        }

        public Body resources() {
            return resources;
        }

        public Body body() {
            return body;
        }

        public List<Body> catchers() {
            return catchers;
        }

        public Body finalizer() {
            return finalizer;
        }

        @Override
        public Block.Builder lower(Block.Builder b, OpTransformer opT) {
            if (resources != null) {
                throw new UnsupportedOperationException("Lowering of try-with-resources is unsupported");
            }

            Block.Builder exit = b.block();
            setBranchTarget(b.context(), this, new BranchTarget(exit, null));

            // Simple case with no catch and finally bodies
            if (catchers.isEmpty() && finalizer == null) {
                b.transformBody(body, List.of(), (block, op) -> {
                    if (op instanceof YieldOp) {
                        block.op(branch(exit.successor()));
                    } else {
                        // @@@ Composition of lowerable ops
                        if (op instanceof Lowerable lop) {
                            block = lop.lower(block, opT);
                        } else {
                            block.op(op);
                        }
                    }
                    return block;
                });
                return exit;
            }

            Block.Builder tryRegionEnter = b.block();
            Block.Builder tryRegionExit = b.block();

            // Construct the catcher block builders
            List<Block.Builder> catchers = catchers().stream()
                    .map(catcher -> b.block())
                    .toList();
            Block.Builder catcherFinally;
            if (finalizer == null) {
                catcherFinally = null;
            } else {
                catcherFinally = b.block();
                catchers = new ArrayList<>(catchers);
                catchers.add(catcherFinally);
            }

            // Enter the try exception region
            List<Block.Reference> exitHandlers = catchers.stream()
                    .map(Block.Builder::successor)
                    .toList();
            b.op(exceptionRegionEnter(tryRegionEnter.successor(), exitHandlers.reversed()));

            OpTransformer tryExitTransformer;
            if (finalizer != null) {
                tryExitTransformer = opT.compose((block, op) -> {
                    if (op instanceof CoreOp.ReturnOp ||
                            (op instanceof ExtendedOp.JavaLabelOp lop && ifExitFromTry(lop))) {
                        return inlineFinalizer(block, exitHandlers, opT);
                    } else {
                        return block;
                    }
                });
            } else {
                tryExitTransformer = opT.compose((block, op) -> {
                    if (op instanceof CoreOp.ReturnOp ||
                            (op instanceof ExtendedOp.JavaLabelOp lop && ifExitFromTry(lop))) {
                        Block.Builder tryRegionReturnExit = block.block();
                        block.op(exceptionRegionExit(tryRegionReturnExit.successor(), exitHandlers));
                        return tryRegionReturnExit;
                    } else {
                        return block;
                    }
                });
            }
            // Inline the try body
            AtomicBoolean hasTryRegionExit = new AtomicBoolean();
            tryRegionEnter.transformBody(body, List.of(), tryExitTransformer.andThen((block, op) -> {
                if (op instanceof YieldOp) {
                    hasTryRegionExit.set(true);
                    block.op(branch(tryRegionExit.successor()));
                } else {
                    // @@@ Composition of lowerable ops
                    if (op instanceof Lowerable lop) {
                        block = lop.lower(block, tryExitTransformer);
                    } else {
                        block.op(op);
                    }
                }
                return block;
            }));

            Block.Builder finallyEnter = null;
            if (finalizer != null) {
                finallyEnter = b.block();
                if (hasTryRegionExit.get()) {
                    // Exit the try exception region
                    tryRegionExit.op(exceptionRegionExit(finallyEnter.successor(), exitHandlers));
                }
            } else if (hasTryRegionExit.get()) {
                // Exit the try exception region
                tryRegionExit.op(exceptionRegionExit(exit.successor(), exitHandlers));
            }

            // Inline the catch bodies
            for (int i = 0; i < this.catchers.size(); i++) {
                Block.Builder catcher = catchers.get(i);
                Body catcherBody = this.catchers.get(i);
                // Create the throwable argument
                Block.Parameter t = catcher.parameter(catcherBody.bodyType().parameterTypes().get(0));

                if (finalizer != null) {
                    Block.Builder catchRegionEnter = b.block();
                    Block.Builder catchRegionExit = b.block();

                    // Enter the catch exception region
                    Result catchExceptionRegion = catcher.op(
                            exceptionRegionEnter(catchRegionEnter.successor(), catcherFinally.successor()));

                    OpTransformer catchExitTransformer = opT.compose((block, op) -> {
                        if (op instanceof CoreOp.ReturnOp) {
                            return inlineFinalizer(block, List.of(catcherFinally.successor()), opT);
                        } else if (op instanceof ExtendedOp.JavaLabelOp lop && ifExitFromTry(lop)) {
                            return inlineFinalizer(block, List.of(catcherFinally.successor()), opT);
                        } else {
                            return block;
                        }
                    });
                    // Inline the catch body
                    AtomicBoolean hasCatchRegionExit = new AtomicBoolean();
                    catchRegionEnter.transformBody(catcherBody, List.of(t), catchExitTransformer.andThen((block, op) -> {
                        if (op instanceof YieldOp) {
                            hasCatchRegionExit.set(true);
                            block.op(branch(catchRegionExit.successor()));
                        } else {
                            // @@@ Composition of lowerable ops
                            if (op instanceof Lowerable lop) {
                                block = lop.lower(block, catchExitTransformer);
                            } else {
                                block.op(op);
                            }
                        }
                        return block;
                    }));

                    // Exit the catch exception region
                    if (hasCatchRegionExit.get()) {
                        hasTryRegionExit.set(true);
                        catchRegionExit.op(exceptionRegionExit(finallyEnter.successor(), catcherFinally.successor()));
                    }
                } else {
                    // Inline the catch body
                    catcher.transformBody(catcherBody, List.of(t), opT.andThen((block, op) -> {
                        if (op instanceof YieldOp) {
                            block.op(branch(exit.successor()));
                        } else {
                            // @@@ Composition of lowerable ops
                            if (op instanceof Lowerable lop) {
                                block = lop.lower(block, opT);
                            } else {
                                block.op(op);
                            }
                        }
                        return block;
                    }));
                }
            }

            if (finalizer != null && hasTryRegionExit.get()) {
                // Inline the finally body
                finallyEnter.transformBody(finalizer, List.of(), opT.andThen((block, op) -> {
                    if (op instanceof YieldOp) {
                        block.op(branch(exit.successor()));
                    } else {
                        // @@@ Composition of lowerable ops
                        if (op instanceof Lowerable lop) {
                            block = lop.lower(block, opT);
                        } else {
                            block.op(op);
                        }
                    }
                    return block;
                }));
            }

            // Inline the finally body as a catcher of Throwable and adjusting to throw
            if (finalizer != null) {
                // Create the throwable argument
                Block.Parameter t = catcherFinally.parameter(type(Throwable.class));

                catcherFinally.transformBody(finalizer, List.of(), opT.andThen((block, op) -> {
                    if (op instanceof YieldOp) {
                        block.op(_throw(t));
                    } else {
                        // @@@ Composition of lowerable ops
                        if (op instanceof Lowerable lop) {
                            block = lop.lower(block, opT);
                        } else {
                            block.op(op);
                        }
                    }
                    return block;
                }));
            }
            return exit;
        }

        boolean ifExitFromTry(JavaLabelOp lop) {
            Op target = lop.target();
            return target == this || ifAncestorOp(target, this);
        }

        static boolean ifAncestorOp(Op ancestor, Op op) {
            while (op.ancestorBody() != null) {
                op = op.ancestorBody().parentOp();
                if (op == ancestor) {
                    return true;
                }
            }
            return false;
        }

        Block.Builder inlineFinalizer(Block.Builder block1, List<Block.Reference> tryHandlers, OpTransformer opT) {
            Block.Builder finallyEnter = block1.block();
            Block.Builder finallyExit = block1.block();

            block1.op(exceptionRegionExit(finallyEnter.successor(), tryHandlers));

            // Inline the finally body
            finallyEnter.transformBody(finalizer, List.of(), opT.andThen((block2, op2) -> {
                if (op2 instanceof YieldOp) {
                    block2.op(branch(finallyExit.successor()));
                } else {
                    // @@@ Composition of lowerable ops
                    if (op2 instanceof Lowerable lop2) {
                        block2 = lop2.lower(block2, opT);
                    } else {
                        block2.op(op2);
                    }
                }
                return block2;
            }));

            return finallyExit;
        }

        @Override
        public TypeElement resultType() {
            return VOID;
        }
    }

    //
    // Patterns

    static final String Pattern_CLASS_NAME = ExtendedOp_CLASS_NAME + "$" + Pattern.class.getSimpleName();

    // Reified pattern nodes

    /**
     * Synthetic pattern types
     * // @@@ Replace with types extending from TypeElement
     */
    public sealed interface Pattern {

        /**
         * Synthetic type pattern type.
         *
         * @param <T> the type of values that are bound
         */
        final class Type<T> implements Pattern {
            Type() {
            }
        }

        /**
         * Synthetic record pattern type.
         *
         * @param <T> the type of records that are bound
         */
        final class Record<T> implements Pattern {
            Record() {
            }
        }

        final class MatchAll implements Pattern {
            MatchAll() {
            }
        }

        // @@@ Pattern types

        JavaType PATTERN_BINDING_TYPE = JavaType.type(ClassDesc.of(Pattern_CLASS_NAME +
                "$" + Type.class.getSimpleName()));
        JavaType PATTERN_RECORD_TYPE = JavaType.type(ClassDesc.of(Pattern_CLASS_NAME +
                "$" + Pattern.Record.class.getSimpleName()));

        JavaType PATTERN_MATCH_ALL_TYPE = JavaType.type(ClassDesc.of(Pattern_CLASS_NAME +
                "$" + Pattern.MatchAll.class.getSimpleName()));

        static JavaType bindingType(TypeElement t) {
            return parameterized(PATTERN_BINDING_TYPE, (JavaType) t);
        }

        static JavaType recordType(TypeElement t) {
            return parameterized(PATTERN_RECORD_TYPE, (JavaType) t);
        }

        static JavaType matchAllType() {
            return PATTERN_MATCH_ALL_TYPE;
        }

        static TypeElement targetType(TypeElement t) {
            return ((ClassType) t).typeArguments().get(0);
        }
    }

    /**
     * Pattern operations.
     */
    public static final class PatternOps {
        PatternOps() {
        }

        /**
         * The pattern operation.
         */
        public sealed static abstract class PatternOp extends ExtendedOp implements Op.Pure {
            PatternOp(ExternalizedOp def) {
                super(def);
            }

            PatternOp(PatternOp that, CopyContext cc) {
                super(that, cc);
            }

            PatternOp(String name, List<Value> operands) {
                super(name, operands);
            }
        }

        /**
         * The binding pattern operation, that can model Java language type patterns.
         */
        @OpFactory.OpDeclaration(TypePatternOp.NAME)
        public static final class TypePatternOp extends PatternOp {
            public static final String NAME = "pattern.type";

            public static final String ATTRIBUTE_BINDING_NAME = NAME + ".binding.name";

            final TypeElement resultType;
            final String bindingName;

            public static TypePatternOp create(ExternalizedOp def) {
                String name = def.extractAttributeValue(ATTRIBUTE_BINDING_NAME, true,
                        v -> switch (v) {
                            case String s -> s;
                            case null -> null;
                            default -> throw new UnsupportedOperationException("Unsupported pattern binding name value:" + v);
                        });
                return new TypePatternOp(def, name);
            }

            TypePatternOp(ExternalizedOp def, String bindingName) {
                super(def);

                this.bindingName = bindingName;
                this.resultType = def.resultType();
            }

            TypePatternOp(TypePatternOp that, CopyContext cc) {
                super(that, cc);

                this.bindingName = that.bindingName;
                this.resultType = that.resultType;
            }

            @Override
            public TypePatternOp transform(CopyContext cc, OpTransformer ot) {
                return new TypePatternOp(this, cc);
            }

            TypePatternOp(TypeElement targetType, String bindingName) {
                super(NAME, List.of());

                this.bindingName = bindingName;
                this.resultType = Pattern.bindingType(targetType);
            }

            @Override
            public Map<String, Object> attributes() {
                HashMap<String, Object> attrs = new HashMap<>(super.attributes());
                if (bindingName != null) {
                    attrs.put("", bindingName);
                }
                return attrs;
            }

            public String bindingName() {
                return bindingName;
            }

            public TypeElement targetType() {
                return Pattern.targetType(resultType());
            }

            @Override
            public TypeElement resultType() {
                return resultType;
            }
        }

        /**
         * The record pattern operation, that can model Java language record patterns.
         */
        @OpFactory.OpDeclaration(RecordPatternOp.NAME)
        public static final class RecordPatternOp extends PatternOp {
            public static final String NAME = "pattern.record";

            public static final String ATTRIBUTE_RECORD_DESCRIPTOR = NAME + ".descriptor";

            final RecordTypeRef recordDescriptor;

            public static RecordPatternOp create(ExternalizedOp def) {
                RecordTypeRef recordDescriptor = def.extractAttributeValue(ATTRIBUTE_RECORD_DESCRIPTOR, true,
                        v -> switch (v) {
                            case String s -> RecordTypeRef.ofString(s);
                            case RecordTypeRef rtd -> rtd;
                            case null, default ->
                                    throw new UnsupportedOperationException("Unsupported record type descriptor value:" + v);
                        });

                return new RecordPatternOp(def, recordDescriptor);
            }

            RecordPatternOp(ExternalizedOp def, RecordTypeRef recordDescriptor) {
                super(def);

                this.recordDescriptor = recordDescriptor;
            }

            RecordPatternOp(RecordPatternOp that, CopyContext cc) {
                super(that, cc);

                this.recordDescriptor = that.recordDescriptor;
            }

            @Override
            public RecordPatternOp transform(CopyContext cc, OpTransformer ot) {
                return new RecordPatternOp(this, cc);
            }

            RecordPatternOp(RecordTypeRef recordDescriptor, List<Value> nestedPatterns) {
                // The type of each value is a subtype of Pattern
                // The number of values corresponds to the number of components of the record
                super(NAME, List.copyOf(nestedPatterns));

                this.recordDescriptor = recordDescriptor;
            }

            @Override
            public Map<String, Object> attributes() {
                HashMap<String, Object> m = new HashMap<>(super.attributes());
                m.put("", recordDescriptor);
                return Collections.unmodifiableMap(m);
            }

            public RecordTypeRef recordDescriptor() {
                return recordDescriptor;
            }

            public TypeElement targetType() {
                return Pattern.targetType(resultType());
            }

            @Override
            public TypeElement resultType() {
                return Pattern.recordType(recordDescriptor.recordType());
            }
        }

        @OpFactory.OpDeclaration(MatchAllPatternOp.NAME)
        public static final class MatchAllPatternOp extends PatternOp {

            // @@@ we may need to add info about the type of the record component
            // this info can be used when lowering

            public static final String NAME = "pattern.match.all";

            public MatchAllPatternOp(ExternalizedOp def) {
                super(def);
            }

            MatchAllPatternOp(MatchAllPatternOp that, CopyContext cc) {
                super(that, cc);
            }

            MatchAllPatternOp() {
                super(NAME, List.of());
            }

            @Override
            public Op transform(CopyContext cc, OpTransformer ot) {
                return new MatchAllPatternOp(this, cc);
            }

            @Override
            public TypeElement resultType() {
                return Pattern.matchAllType();
            }
        }

        /**
         * The match operation, that can model Java language pattern matching.
         */
        @OpFactory.OpDeclaration(MatchOp.NAME)
        public static final class MatchOp extends ExtendedOp implements Op.Isolated, Op.Lowerable {
            public static final String NAME = "pattern.match";

            final Body pattern;
            final Body match;

            public MatchOp(ExternalizedOp def) {
                super(def);

                this.pattern = def.bodyDefinitions().get(0).build(this);
                this.match = def.bodyDefinitions().get(1).build(this);
            }

            MatchOp(MatchOp that, CopyContext cc, OpTransformer ot) {
                super(that, cc);

                this.pattern = that.pattern.transform(cc, ot).build(this);
                this.match = that.match.transform(cc, ot).build(this);
            }

            @Override
            public MatchOp transform(CopyContext cc, OpTransformer ot) {
                return new MatchOp(this, cc, ot);
            }

            MatchOp(Value target, Body.Builder patternC, Body.Builder matchC) {
                super(NAME,
                        List.of(target));

                this.pattern = patternC.build(this);
                this.match = matchC.build(this);
            }

            @Override
            public List<Body> bodies() {
                return List.of(pattern, match);
            }

            public Body pattern() {
                return pattern;
            }

            public Body match() {
                return match;
            }

            public Value target() {
                return operands().get(0);
            }

            @Override
            public Block.Builder lower(Block.Builder b, OpTransformer opT) {
                // No match block
                Block.Builder endNoMatchBlock = b.block();
                // Match block
                Block.Builder endMatchBlock = b.block();
                // End block
                Block.Builder endBlock = b.block();
                Block.Parameter matchResult = endBlock.parameter(resultType());
                // Map match operation result
                b.context().mapValue(result(), matchResult);

                List<Value> patternValues = new ArrayList<>();
                Op patternYieldOp = pattern.entryBlock().terminatingOp();
                Op.Result rootPatternValue = (Op.Result) patternYieldOp.operands().get(0);
                Block.Builder currentBlock = lower(endNoMatchBlock, b,
                        patternValues,
                        rootPatternValue.op(),
                        b.context().getValue(target()));
                currentBlock.op(branch(endMatchBlock.successor()));

                // No match block
                // Pass false
                endNoMatchBlock.op(branch(endBlock.successor(
                        endNoMatchBlock.op(constant(BOOLEAN, false)))));

                // Match block
                // Lower match body and pass true
                endMatchBlock.transformBody(match, patternValues, opT.andThen((block, op) -> {
                    if (op instanceof YieldOp) {
                        block.op(branch(endBlock.successor(
                                block.op(constant(BOOLEAN, true)))));
                    } else if (op instanceof Lowerable lop) {
                        // @@@ Composition of lowerable ops
                        block = lop.lower(block, opT);
                    } else {
                        block.op(op);
                    }
                    return block;
                }));

                return endBlock;
            }

            static Block.Builder lower(Block.Builder endNoMatchBlock, Block.Builder currentBlock,
                                       List<Value> bindings,
                                       Op pattern, Value target) {
                return switch (pattern) {
                    case RecordPatternOp rp -> lowerRecordPattern(endNoMatchBlock, currentBlock, bindings, rp, target);
                    case TypePatternOp tp -> lowerTypePattern(endNoMatchBlock, currentBlock, bindings, tp, target);
                    case MatchAllPatternOp map -> lowerMatchAllPattern(currentBlock);
                    case null, default -> throw new UnsupportedOperationException("Unknown pattern op: " + pattern);
                };
            }

            static Block.Builder lowerRecordPattern(Block.Builder endNoMatchBlock, Block.Builder currentBlock,
                                                    List<Value> bindings,
                                                    ExtendedOp.PatternOps.RecordPatternOp rpOp, Value target) {
                TypeElement targetType = rpOp.targetType();

                Block.Builder nextBlock = currentBlock.block();

                // Check if instance of target type
                Op.Result isInstance = currentBlock.op(CoreOp.instanceOf(targetType, target));
                currentBlock.op(conditionalBranch(isInstance, nextBlock.successor(), endNoMatchBlock.successor()));

                currentBlock = nextBlock;

                target = currentBlock.op(CoreOp.cast(targetType, target));

                // Access component values of record and match on each as nested target
                List<Value> dArgs = rpOp.operands();
                for (int i = 0; i < dArgs.size(); i++) {
                    Op.Result nestedPattern = (Op.Result) dArgs.get(i);
                    // @@@ Handle exceptions?
                    Value nestedTarget = currentBlock.op(CoreOp.invoke(rpOp.recordDescriptor().methodForComponent(i), target));

                    currentBlock = lower(endNoMatchBlock, currentBlock, bindings, nestedPattern.op(), nestedTarget);
                }

                return currentBlock;
            }

            static Block.Builder lowerTypePattern(Block.Builder endNoMatchBlock, Block.Builder currentBlock,
                                                  List<Value> bindings,
                                                  TypePatternOp tpOp, Value target) {
                TypeElement targetType = tpOp.targetType();

                // Check if instance of target type
                Op p = null; // op that perform type check
                Op c; // op that perform conversion
                TypeElement s = target.type();
                TypeElement t = targetType;
                if (t instanceof PrimitiveType pt) {
                    if (s instanceof ClassType cs) {
                        // unboxing conversions
                        ClassType box;
                        if (cs.unbox().isEmpty()) { // s not a boxed type
                            // e.g. Number -> int, narrowing + unboxing
                            box = pt.box().orElseThrow();
                            p = CoreOp.instanceOf(box, target);
                        } else {
                            // e.g. Float -> float, unboxing
                            // e.g. Integer -> long, unboxing + widening
                            box = cs;
                        }
                        c = invoke(MethodRef.method(box, t + "Value", t), target);
                    } else {
                        // primitive to primitive conversion
                        PrimitiveType ps = ((PrimitiveType) s);
                        if (isNarrowingPrimitiveConv(ps, pt) || isWideningPrimitiveConvWithCheck(ps, pt)
                                || isWideningAndNarrowingPrimitiveConv(ps, pt)) {
                            // e.g. int -> byte, narrowing
                            // e,g. int -> float, widening with check
                            // e.g. byte -> char, widening and narrowing
                            MethodRef mref = convMethodRef(s, t);
                            p = invoke(mref, target);
                        }
                        c = CoreOp.conv(targetType, target);
                    }
                } else if (s instanceof PrimitiveType ps) {
                    // boxing conversions
                    ClassType ct = (ClassType) t;
                    ClassType box;
                    if (ct.unbox().isEmpty()) {
                        // e.g. int -> Number, boxing + widening
                        box = ps.box().orElseThrow();
                    } else {
                        // e.g. byte -> Byte, boxing
                        box = ct;
                    }
                    c = invoke(MethodRef.method(box, "valueOf", box, ps), target);
                } else {
                    // e.g. Number -> Double, ...
                    p = CoreOp.instanceOf(targetType, target);
                    c = CoreOp.cast(targetType, target);
                }

                if (p != null) {
                    // p != null, we need to perform type check at runtime
                    Block.Builder nextBlock = currentBlock.block();
                    currentBlock.op(conditionalBranch(currentBlock.op(p), nextBlock.successor(), endNoMatchBlock.successor()));
                    currentBlock = nextBlock;
                }

                target = currentBlock.op(c);

                bindings.add(target);

                return currentBlock;
            }

            private static boolean isWideningAndNarrowingPrimitiveConv(PrimitiveType s, PrimitiveType t) {
                return BYTE.equals(s) && CHAR.equals(t);
            }

            private static boolean isWideningPrimitiveConvWithCheck(PrimitiveType s, PrimitiveType t) {
                return (INT.equals(s) && FLOAT.equals(t))
                        || (LONG.equals(s) && FLOAT.equals(t))
                        || (LONG.equals(s) && DOUBLE.equals(t));
            }

            // s -> t is narrowing if order(t) <= order(s)
            private final static Map<PrimitiveType, Integer> narrowingOrder = Map.of(
                    BYTE, 1,
                    SHORT, 2,
                    CHAR, 2,
                    INT, 3,
                    LONG, 4,
                    FLOAT, 5,
                    DOUBLE, 6
            );
            private static boolean isNarrowingPrimitiveConv(PrimitiveType s, PrimitiveType t) {
                return narrowingOrder.get(t) <= narrowingOrder.get(s);
            }

            private static MethodRef convMethodRef(TypeElement s, TypeElement t) {
                if (BYTE.equals(s) || SHORT.equals(s) || CHAR.equals(s)) {
                    s = INT;
                }
                String sn = capitalize(s.toString());
                String tn = capitalize(t.toString());
                String mn = "is%sTo%sExact".formatted(sn, tn);
                JavaType exactConversionSupport = JavaType.type(ClassDesc.of("java.lang.runtime.ExactConversionsSupport"));
                return MethodRef.method(exactConversionSupport, mn, BOOLEAN, s);
            }

            private static String capitalize(String s) {
                return s.substring(0, 1).toUpperCase() + s.substring(1);
            }

            static Block.Builder lowerMatchAllPattern(Block.Builder currentBlock) {
                return currentBlock;
            }

            @Override
            public TypeElement resultType() {
                return BOOLEAN;
            }
        }
    }


    /**
     * A factory for extended and core operations.
     */
    // @@@ Compute lazily
    public static final OpFactory FACTORY = CoreOp.FACTORY.andThen(OpFactory.OP_FACTORY.get(ExtendedOp.class));


    /**
     * Creates a continue operation.
     *
     * @return the continue operation
     */
    public static JavaContinueOp _continue() {
        return _continue(null);
    }

    /**
     * Creates a continue operation.
     *
     * @param label the value associated with where to continue from
     * @return the continue operation
     */
    public static JavaContinueOp _continue(Value label) {
        return new JavaContinueOp(label);
    }

    /**
     * Creates a break operation.
     *
     * @return the break operation
     */
    public static JavaBreakOp _break() {
        return _break(null);
    }

    /**
     * Creates a break operation.
     *
     * @param label the value associated with where to continue from
     * @return the break operation
     */
    public static JavaBreakOp _break(Value label) {
        return new JavaBreakOp(label);
    }

    /**
     * Creates a yield operation.
     *
     * @return the yield operation
     */
    public static JavaYieldOp java_yield() {
        return new JavaYieldOp();
    }

    /**
     * Creates a yield operation.
     *
     * @param operand the value to yield
     * @return the yield operation
     */
    public static JavaYieldOp java_yield(Value operand) {
        return new JavaYieldOp(operand);
    }

    /**
     * Creates a block operation.
     *
     * @param body the body builder of the operation to be built and become its child
     * @return the block operation
     */
    public static JavaBlockOp block(Body.Builder body) {
        return new JavaBlockOp(body);
    }

    /**
     * Creates a synchronized operation.
     *
     * @param expr the expression body builder of the operation to be built and become its child
     * @param blockBody the block body builder of the operation to be built and become its child
     * @return the synchronized operation
     */
    public static JavaSynchronizedOp synchronized_(Body.Builder expr, Body.Builder blockBody) {
        return new JavaSynchronizedOp(expr, blockBody);
    }

    /**
     * Creates a labeled operation.
     *
     * @param body the body builder of the operation to be built and become its child
     * @return the block operation
     */
    public static JavaLabeledOp labeled(Body.Builder body) {
        return new JavaLabeledOp(body);
    }

    /**
     * Creates an if operation builder.
     *
     * @param ancestorBody the nearest ancestor body builder from which to construct
     *                     body builders for this operation
     * @return the if operation builder
     */
    public static JavaIfOp.IfBuilder _if(Body.Builder ancestorBody) {
        return new JavaIfOp.IfBuilder(ancestorBody);
    }

    // Pairs of
    //   predicate ()boolean, body ()void
    // And one optional body ()void at the end

    /**
     * Creates an if operation.
     *
     * @param bodies the body builders of operation to be built and become its children
     * @return the if operation
     */
    public static JavaIfOp _if(List<Body.Builder> bodies) {
        return new JavaIfOp(bodies);
    }

    /**
     * Creates a switch expression operation.
     * <p>
     * The result type of the operation will be derived from the yield type of the second body
     *
     * @param target the switch target value
     * @param bodies the body builders of the operation to be built and become its children
     * @return the switch expression operation
     */
    public static JavaSwitchExpressionOp switchExpression(Value target, List<Body.Builder> bodies) {
        return new JavaSwitchExpressionOp(null, target, bodies);
    }

    /**
     * Creates a switch expression operation.
     *
     * @param resultType the result type of the expression
     * @param target     the switch target value
     * @param bodies     the body builders of the operation to be built and become its children
     * @return the switch expression operation
     */
    public static JavaSwitchExpressionOp switchExpression(TypeElement resultType, Value target,
                                                          List<Body.Builder> bodies) {
        Objects.requireNonNull(resultType);
        return new JavaSwitchExpressionOp(resultType, target, bodies);
    }

    /**
     * Creates a switch statement operation.
     * @param target the switch target value
     * @param bodies the body builders of the operation to be built and become its children
     * @return the switch statement operation
     */
    public static JavaSwitchStatementOp switchStatement(Value target, List<Body.Builder> bodies) {
        return new JavaSwitchStatementOp(target, bodies);
    }

    /**
     * Creates a switch fallthrough operation.
     *
     * @return the switch fallthrough operation
     */
    public static JavaSwitchFallthroughOp switchFallthroughOp() {
        return new JavaSwitchFallthroughOp();
    }

    /**
     * Creates a for operation builder.
     *
     * @param ancestorBody the nearest ancestor body builder from which to construct
     *                     body builders for this operation
     * @param initTypes    the types of initialized variables
     * @return the for operation builder
     */
    public static JavaForOp.InitBuilder _for(Body.Builder ancestorBody, TypeElement... initTypes) {
        return _for(ancestorBody, List.of(initTypes));
    }

    /**
     * Creates a for operation builder.
     *
     * @param ancestorBody the nearest ancestor body builder from which to construct
     *                     body builders for this operation
     * @param initTypes    the types of initialized variables
     * @return the for operation builder
     */
    public static JavaForOp.InitBuilder _for(Body.Builder ancestorBody, List<? extends TypeElement> initTypes) {
        return new JavaForOp.InitBuilder(ancestorBody, initTypes);
    }


    /**
     * Creates a for operation.
     *
     * @param init   the init body builder of the operation to be built and become its child
     * @param cond   the cond body builder of the operation to be built and become its child
     * @param update the update body builder of the operation to be built and become its child
     * @param body   the main body builder of the operation to be built and become its child
     * @return the for operation
     */
    // init ()Tuple<Var<T1>, Var<T2>, ..., Var<TN>>, or init ()void
    // cond (Var<T1>, Var<T2>, ..., Var<TN>)boolean
    // update (Var<T1>, Var<T2>, ..., Var<TN>)void
    // body (Var<T1>, Var<T2>, ..., Var<TN>)void
    public static JavaForOp _for(Body.Builder init,
                                 Body.Builder cond,
                                 Body.Builder update,
                                 Body.Builder body) {
        return new JavaForOp(init, cond, update, body);
    }

    /**
     * Creates an enhanced for operation builder.
     *
     * @param ancestorBody the nearest ancestor body builder from which to construct
     *                     body builders for this operation
     * @param iterableType the iterable type
     * @param elementType  the element type
     * @return the enhanced for operation builder
     */
    public static JavaEnhancedForOp.ExpressionBuilder enhancedFor(Body.Builder ancestorBody,
                                                                  TypeElement iterableType, TypeElement elementType) {
        return new JavaEnhancedForOp.ExpressionBuilder(ancestorBody, iterableType, elementType);
    }

    // expression ()I<E>
    // init (E )Var<T>
    // body (Var<T> )void

    /**
     * Creates an enhanced for operation.
     *
     * @param expression the expression body builder of the operation to be built and become its child
     * @param init       the init body builder of the operation to be built and become its child
     * @param body       the main body builder of the operation to be built and become its child
     * @return the enhanced for operation
     */
    public static JavaEnhancedForOp enhancedFor(Body.Builder expression,
                                                Body.Builder init,
                                                Body.Builder body) {
        return new JavaEnhancedForOp(expression, init, body);
    }

    /**
     * Creates a while operation builder.
     *
     * @param ancestorBody the nearest ancestor body builder from which to construct
     *                     body builders for this operation
     * @return the while operation builder
     */
    public static JavaWhileOp.PredicateBuilder _while(Body.Builder ancestorBody) {
        return new JavaWhileOp.PredicateBuilder(ancestorBody);
    }

    /**
     * Creates a while operation.
     *
     * @param predicate the predicate body builder of the operation to be built and become its child
     * @param body      the main body builder of the operation to be built and become its child
     * @return the while operation
     */
    // predicate, ()boolean, may be null for predicate returning true
    // body, ()void
    public static JavaWhileOp _while(Body.Builder predicate, Body.Builder body) {
        return new JavaWhileOp(predicate, body);
    }

    /**
     * Creates a do operation builder.
     *
     * @param ancestorBody the nearest ancestor body builder from which to construct
     *                     body builders for this operation
     * @return the do operation builder
     */
    public static JavaDoWhileOp.BodyBuilder doWhile(Body.Builder ancestorBody) {
        return new JavaDoWhileOp.BodyBuilder(ancestorBody);
    }

    /**
     * Creates a do operation.
     *
     * @param predicate the predicate body builder of the operation to be built and become its child
     * @param body      the main body builder of the operation to be built and become its child
     * @return the do operation
     */
    public static JavaDoWhileOp doWhile(Body.Builder body, Body.Builder predicate) {
        return new JavaDoWhileOp(body, predicate);
    }

    /**
     * Creates a conditional-and operation builder.
     *
     * @param ancestorBody the nearest ancestor body builder from which to construct
     *                     body builders for this operation
     * @param lhs          a consumer that builds the left-hand side body
     * @param rhs          a consumer that builds the right-hand side body
     * @return the conditional-and operation builder
     */
    public static JavaConditionalAndOp.Builder conditionalAnd(Body.Builder ancestorBody,
                                                              Consumer<Block.Builder> lhs, Consumer<Block.Builder> rhs) {
        return new JavaConditionalAndOp.Builder(ancestorBody, lhs, rhs);
    }

    /**
     * Creates a conditional-or operation builder.
     *
     * @param ancestorBody the nearest ancestor body builder from which to construct
     *                     body builders for this operation
     * @param lhs          a consumer that builds the left-hand side body
     * @param rhs          a consumer that builds the right-hand side body
     * @return the conditional-or operation builder
     */
    public static JavaConditionalOrOp.Builder conditionalOr(Body.Builder ancestorBody,
                                                            Consumer<Block.Builder> lhs, Consumer<Block.Builder> rhs) {
        return new JavaConditionalOrOp.Builder(ancestorBody, lhs, rhs);
    }

    /**
     * Creates a conditional-and operation
     *
     * @param bodies the body builders of operation to be built and become its children
     * @return the conditional-and operation
     */
    // predicates, ()boolean
    public static JavaConditionalAndOp conditionalAnd(List<Body.Builder> bodies) {
        return new JavaConditionalAndOp(bodies);
    }

    /**
     * Creates a conditional-or operation
     *
     * @param bodies the body builders of operation to be built and become its children
     * @return the conditional-or operation
     */
    // predicates, ()boolean
    public static JavaConditionalOrOp conditionalOr(List<Body.Builder> bodies) {
        return new JavaConditionalOrOp(bodies);
    }

    /**
     * Creates a conditional operation
     *
     * @param expressionType the result type of the expression
     * @param bodies         the body builders of operation to be built and become its children
     * @return the conditional operation
     */
    public static JavaConditionalExpressionOp conditionalExpression(TypeElement expressionType,
                                                                    List<Body.Builder> bodies) {
        Objects.requireNonNull(expressionType);
        return new JavaConditionalExpressionOp(expressionType, bodies);
    }

    /**
     * Creates a conditional operation
     * <p>
     * The result type of the operation will be derived from the yield type of the second body
     *
     * @param bodies the body builders of operation to be built and become its children
     * @return the conditional operation
     */
    public static JavaConditionalExpressionOp conditionalExpression(List<Body.Builder> bodies) {
        return new JavaConditionalExpressionOp(null, bodies);
    }

    /**
     * Creates try operation builder.
     *
     * @param ancestorBody the nearest ancestor body builder from which to construct
     *                     body builders for this operation
     * @param c            a consumer that builds the try body
     * @return the try operation builder
     */
    public static JavaTryOp.CatchBuilder _try(Body.Builder ancestorBody, Consumer<Block.Builder> c) {
        Body.Builder _try = Body.Builder.of(ancestorBody, FunctionType.VOID);
        c.accept(_try.entryBlock());
        return new JavaTryOp.CatchBuilder(ancestorBody, null, _try);
    }

    /**
     * Creates try-with-resources operation builder.
     *
     * @param ancestorBody the nearest ancestor body builder from which to construct
     *                     body builders for this operation
     * @param c            a consumer that builds the resources body
     * @return the try-with-resources operation builder
     */
    public static JavaTryOp.BodyBuilder tryWithResources(Body.Builder ancestorBody,
                                                         List<? extends TypeElement> resourceTypes,
                                                         Consumer<Block.Builder> c) {
        resourceTypes = resourceTypes.stream().map(VarType::varType).toList();
        Body.Builder resources = Body.Builder.of(ancestorBody,
                FunctionType.functionType(TupleType.tupleType(resourceTypes)));
        c.accept(resources.entryBlock());
        return new JavaTryOp.BodyBuilder(ancestorBody, resourceTypes, resources);
    }

    // resources ()Tuple<Var<R1>, Var<R2>, ..., Var<RN>>, or null
    // try (Var<R1>, Var<R2>, ..., Var<RN>)void, or try ()void
    // catch (E )void, where E <: Throwable
    // finally ()void, or null

    /**
     * Creates a try or try-with-resources operation.
     *
     * @param resources the try body builder of the operation to be built and become its child,
     *                  may be null
     * @param body      the try body builder of the operation to be built and become its child
     * @param catchers  the catch body builders of the operation to be built and become its children
     * @param finalizer the finalizer body builder of the operation to be built and become its child
     * @return the try or try-with-resources operation
     */
    public static JavaTryOp _try(Body.Builder resources,
                                 Body.Builder body,
                                 List<Body.Builder> catchers,
                                 Body.Builder finalizer) {
        return new JavaTryOp(resources, body, catchers, finalizer);
    }

    //
    // Patterns

    /**
     * Creates a pattern match operation.
     *
     * @param target  the target value
     * @param pattern the pattern body builder of the operation to be built and become its child
     * @param match   the match body builder of the operation to be built and become its child
     * @return the pattern match operation
     */
    public static PatternOps.MatchOp match(Value target,
                                           Body.Builder pattern, Body.Builder match) {
        return new PatternOps.MatchOp(target, pattern, match);
    }

    /**
     * Creates a pattern binding operation.
     *
     * @param type        the type of value to be bound
     * @param bindingName the binding name
     * @return the pattern binding operation
     */
    public static PatternOps.TypePatternOp typePattern(TypeElement type, String bindingName) {
        return new PatternOps.TypePatternOp(type, bindingName);
    }

    /**
     * Creates a record pattern operation.
     *
     * @param recordDescriptor the record descriptor
     * @param nestedPatterns   the nested pattern values
     * @return the record pattern operation
     */
    public static PatternOps.RecordPatternOp recordPattern(RecordTypeRef recordDescriptor, Value... nestedPatterns) {
        return recordPattern(recordDescriptor, List.of(nestedPatterns));
    }

    /**
     * Creates a record pattern operation.
     *
     * @param recordDescriptor the record descriptor
     * @param nestedPatterns   the nested pattern values
     * @return the record pattern operation
     */
    public static PatternOps.RecordPatternOp recordPattern(RecordTypeRef recordDescriptor, List<Value> nestedPatterns) {
        return new PatternOps.RecordPatternOp(recordDescriptor, nestedPatterns);
    }

    public static PatternOps.MatchAllPatternOp matchAllPattern() {
        return new PatternOps.MatchAllPatternOp();
    }

}

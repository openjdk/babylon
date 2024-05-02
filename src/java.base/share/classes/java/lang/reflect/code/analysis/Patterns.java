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

package java.lang.reflect.code.analysis;

import java.lang.reflect.code.Block;
import java.lang.reflect.code.Op;
import java.lang.reflect.code.CodeElement;
import java.lang.reflect.code.Value;
import java.lang.reflect.code.op.CoreOp;
import java.util.*;
import java.util.function.BiFunction;
import java.util.function.Predicate;

/**
 * A simple and experimental pattern match mechanism on values and operations.
 * <p>
 * When the language has support for pattern matching with matcher methods we should be able to express
 * matching on values and operations more powerfully and concisely.
 */
public final class Patterns {

    private Patterns() {
    }


    /**
     * Traverses this operation and its descendant operations and returns the set of operations that are unused
     * (have no uses) and are pure (are instances of {@code Op.Pure} and thus have no side effects).
     *
     * @param op the operation to traverse
     * @return the set of used and pure operations.
     */
    public static Set<Op> matchUnusedPureOps(Op op) {
        return matchUnusedPureOps(op, o -> o instanceof Op.Pure);
    }

    /**
     * Traverses this operation and its descendant operations and returns the set of operations that are unused
     * (have no uses) and are pure (according to the given predicate).
     *
     * @param op       the operation to traverse
     * @param testPure the predicate to test if an operation is pure
     * @return the set of used and pure operations.
     */
    public static Set<Op> matchUnusedPureOps(Op op, Predicate<Op> testPure) {
        return match(
                new HashSet<>(),
                op, opP(o -> isDeadOp(o, testPure)),
                (ms, deadOps) -> {
                    deadOps.add(ms.op());

                    // Dependent dead ops
                    matchDependentDeadOps(ms.op(), deadOps, testPure);
                    // @@@ No means to control traversal and only go deeper when
                    // there is only one user
//                    ms.op().traverseOperands(null, (_a, arg) -> {
//                        if (arg.result().users().size() == 1) {
//                            deadOps.add(arg);
//                        }
//
//                        return null;
//                    });

                    return deadOps;
                });
    }

    static boolean isDeadOp(Op op, Predicate<Op> testPure) {
        if (op instanceof Op.Terminating) {
            return false;
        }

        return op.result() != null && op.result().uses().isEmpty() && testPure.test(op);
    }

    // @@@ this could be made generic with a method traversing backwards
    static void matchDependentDeadOps(Op op, Set<Op> deadOps, Predicate<Op> testPure) {
        for (Value arg : op.operands()) {
            if (arg instanceof Op.Result or) {
                if (arg.uses().size() == 1 && testPure.test(or.op())) {
                    deadOps.add(or.op());

                    // Traverse only when a single user
                    matchDependentDeadOps(or.op(), deadOps, testPure);
                }
            }
        }
    }


    // Matching of patterns

    /**
     * The state of a successful match of an operation with matched operands (if any)
     */
    public static final class MatchState {
        final Op op;
        final List<Value> matchedOperands;

        MatchState(Op op, List<Value> matchedOperands) {
            this.op = op;
            this.matchedOperands = matchedOperands;
        }

        /**
         * {@return the matched operation}
         */
        public Op op() {
            return op;
        }

        /**
         * {@return the matched operands}
         */
        public List<Value> matchedOperands() {
            return matchedOperands;
        }
    }

    record PatternAndFunction<R>(OpPattern p, BiFunction<MatchState, R, R> f) {
    }

    // Visiting op pattern matcher
    static class OpPatternMatcher<R> implements BiFunction<R, Op, R> {
        final List<PatternAndFunction<R>> patterns;
        final PatternState state;
        final Map<MatchState, BiFunction<MatchState, R, R>> matches;

        OpPatternMatcher(OpPattern p, BiFunction<MatchState, R, R> f) {
            this(List.of(new PatternAndFunction<>(p, f)));
        }

        OpPatternMatcher(List<PatternAndFunction<R>> patterns) {
            this.patterns = patterns;
            this.state = new PatternState();
            this.matches = new HashMap<>();
        }

        @Override
        public R apply(R r, Op op) {
            for (PatternAndFunction<R> pf : patterns) {
                if (pf.p.match(op, state)) {
                    MatchState ms = new MatchState(op, state.resetOnMatch());

                    r = pf.f.apply(ms, r);
                } else {
                    state.resetOnNoMatch();
                }
            }

            return r;
        }
    }

    /**
     * A match builder for declaring matching for one or more groups of operation patterns against a given traversable
     * and descendant operations (in order).
     * @param <R> the match result type
     */
    public static final class MultiMatchBuilder<R> {
        final CodeElement<?, ?> o;
        final R r;
        List<PatternAndFunction<R>> patterns;

        MultiMatchBuilder(CodeElement<?, ?> o, R r) {
            this.o = o;
            this.r = r;
            this.patterns = new ArrayList<>();
        }

        /**
         * Declares a first of possibly other operation patterns in a group.
         *
         * @param p the operation pattern
         * @return a builder to declare further patterns in the group.
         */
        public MultiMatchCaseBuilder pattern(OpPattern p) {
            return new MultiMatchCaseBuilder(p);
        }

        public R matchThenApply() {
            OpPatternMatcher<R> opm = new OpPatternMatcher<>(patterns);
            return o.traverse(r, CodeElement.opVisitor(opm));
        }

        /**
         * A builder to declare further operation patterns in a group or to associate a
         * target function to be applied if any of the patterns in the group match.
         */
        public final class MultiMatchCaseBuilder {
            List<OpPattern> patterns;

            MultiMatchCaseBuilder(OpPattern p) {
                this.patterns = new ArrayList<>();
                patterns.add(p);
            }

            /**
             * Declares an operation pattern in the group.
             *
             * @param p the operation pattern
             * @return this builder.
             */
            public MultiMatchCaseBuilder pattern(OpPattern p) {
                patterns.add(p);
                return this;
            }

            /**
             * Declares the target function to be applied if any of the operation patterns on the group match.
             *
             * @param f the target function.
             * @return the match builder to build further groups.
             */
            public MultiMatchBuilder<R> target(BiFunction<MatchState, R, R> f) {
                patterns.stream().map(p -> new PatternAndFunction<>(p, f)).forEach(MultiMatchBuilder.this.patterns::add);
                return MultiMatchBuilder.this;
            }
        }
    }

    /**
     * Constructs a match builder from which to declare matching for one or more groups of operation patterns against a
     * given traversable and descendant operations (in order).
     *
     * @param r   the initial match result
     * @param t   the traversable
     * @param <R> the match result type
     * @return the match builder
     */
    public static <R> MultiMatchBuilder<R> multiMatch(R r, CodeElement<?, ?> t) {
        return new MultiMatchBuilder<>(t, r);
    }

    /**
     * Matches an operation pattern against the given traversable and descendant operations (in order).
     *
     * @param r         the initial match result
     * @param t         the traversable
     * @param opPattern the operation pattern
     * @param matcher   the function to be applied with a match state and the current match result when an
     *                  encountered operation matches the operation pattern
     * @param <R>       the match result type
     * @return the match result
     */
    public static <R> R match(R r, CodeElement<?, ?> t, OpPattern opPattern,
                              BiFunction<MatchState, R, R> matcher) {
        OpPatternMatcher<R> opm = new OpPatternMatcher<>(opPattern, matcher);
        return t.traverse(r, CodeElement.opVisitor(opm));
    }


    // Pattern classes

    static final class PatternState {
        List<Value> matchedOperands;

        void addOperand(Value v) {
            if (matchedOperands == null) {
                matchedOperands = new ArrayList<>();
            }
            matchedOperands.add(v);
        }

        List<Value> resetOnMatch() {
            if (matchedOperands != null) {
                List<Value> r = matchedOperands;
                matchedOperands = null;
                return r;
            } else {
                return List.of();
            }
        }

        void resetOnNoMatch() {
            if (matchedOperands != null) {
                matchedOperands.clear();
            }
        }
    }

    /**
     * A pattern matching against a value or operation.
     */
    public sealed static abstract class Pattern {
        Pattern() {
        }

        abstract boolean match(Value v, PatternState state);
    }

    /**
     * A pattern matching against an operation.
     */
    public static final class OpPattern extends Pattern {
        final Predicate<Op> opTest;
        final List<Pattern> operandPatterns;

        OpPattern(Predicate<Op> opTest, List<Pattern> operandPatterns) {
            this.opTest = opTest;
            this.operandPatterns = List.copyOf(operandPatterns);
        }

        @Override
        boolean match(Value v, PatternState state) {
            if (v instanceof Op.Result or) {
                return match(or.op(), state);
            } else {
                return false;
            }
        }

        boolean match(Op op, PatternState state) {
            // Test does not match
            if (!opTest.test(op)) {
                return false;
            }

            if (!operandPatterns.isEmpty()) {
                // Arity does not match
                if (op.operands().size() != operandPatterns.size()) {
                    return false;
                }

                // Match all arguments
                for (int i = 0; i < operandPatterns.size(); i++) {
                    Pattern p = operandPatterns.get(i);
                    Value v = op.operands().get(i);

                    if (!p.match(v, state)) {
                        return false;
                    }
                }
            }

            return true;
        }
    }

    /**
     * A pattern that unconditionally matches a value which is captured. If the value is an operation result of an
     * operation, then an operation pattern (if any) is further matched against the operation.
     */
    // @@@ type?
    static final class ValuePattern extends Pattern {
        final OpPattern opMatcher;

        ValuePattern() {
            this(null);
        }

        public ValuePattern(OpPattern opMatcher) {
            this.opMatcher = opMatcher;
        }

        @Override
        boolean match(Value v, PatternState state) {
            // Capture the operand
            state.addOperand(v);

            // Match on operation on nested pattern, if any
            return opMatcher == null || opMatcher.match(v, state);
        }
    }

    /**
     * A pattern that conditionally matches an operation result which is captured,  then an operation pattern (if any)
     * is further matched against the result's operation.
     */
    static final class OpResultPattern extends Pattern {
        final OpPattern opMatcher;

        OpResultPattern() {
            this(null);
        }

        public OpResultPattern(OpPattern opMatcher) {
            this.opMatcher = opMatcher;
        }

        @Override
        boolean match(Value v, PatternState state) {
            if (!(v instanceof Op.Result)) {
                return false;
            }

            // Capture the operand
            state.addOperand(v);

            // Match on operation on nested pattern, if any
            return opMatcher == null || opMatcher.match(v, state);
        }
    }

    /**
     * A pattern that conditionally matches a block parameter which is captured.
     */
    static final class BlockParameterPattern extends Pattern {
        BlockParameterPattern() {
        }

        @Override
        boolean match(Value v, PatternState state) {
            if (!(v instanceof Block.Parameter)) {
                return false;
            }

            // Capture the operand
            state.addOperand(v);

            return true;
        }
    }

    /**
     * A pattern matching any value or operation.
     */
    static final class AnyPattern extends Pattern {
        AnyPattern() {
        }

        @Override
        boolean match(Value v, PatternState state) {
            return true;
        }
    }


    // Pattern factories

    /**
     * Creates an operation pattern that tests against an operation by applying it to the predicate, and if
     * {@code true}, matches operand patterns against the operation's operands (in order) .
     * This operation pattern matches an operation if the test returns {@code true} and all operand patterns match
     * against the operation's operands.
     *
     * @param opTest the predicate
     * @param patterns the operand patterns
     * @return the operation pattern
     */
    public static OpPattern opP(Predicate<Op> opTest, Pattern... patterns) {
        return opP(opTest, List.of(patterns));
    }

    /**
     * Creates an operation pattern that tests against an operation by applying it to the predicate, and if
     * {@code true}, matches operand patterns against the operation's operands (in order) .
     * This operation pattern matches an operation if the test returns {@code true} and all operand patterns match
     * against the operation's operands.
     *
     * @param opTest the predicate
     * @param patterns the operand patterns
     * @return the operation pattern
     */
    public static OpPattern opP(Predicate<Op> opTest, List<Pattern> patterns) {
        return new OpPattern(opTest, patterns);
    }

    /**
     * Creates an operation pattern that tests if the operation is an instance of the class, and if
     * {@code true}, matches operand patterns against the operation's operands (in order) .
     * This operation pattern matches an operation if the test returns {@code true} and all operand patterns match
     * against the operation's operands.
     *
     * @param opClass the operation class
     * @param patterns the operand patterns
     * @return the operation pattern
     */
    public static OpPattern opP(Class<?> opClass, Pattern... patterns) {
        return opP(opClass::isInstance, patterns);
    }

    /**
     * Creates an operation pattern that tests if the operation is a {@link CoreOp.ConstantOp constant} operation
     * and whose constant value is equal to the given value.
     * This operation pattern matches an operation if the test returns {@code true}.
     *
     * @param value the value
     * @return the operation pattern.
     */
    public static OpPattern constantP(Object value) {
        return opP(op -> {
            if (op instanceof CoreOp.ConstantOp cop) {
                return Objects.equals(value, cop.value());
            }

            return false;
        });
    }

    /**
     * Creates a value pattern that unconditionally matches any value and captures the value in match state.
     *
     * @return the value pattern.
     */
    public static Pattern valueP() {
        return new ValuePattern();
    }

    /**
     * Creates a value pattern that unconditionally matches any value and captures the value in match state, and
     * if the value is an operation result of an operation, then the operation pattern is matched against that
     * operation.
     * This value pattern matches value if value is not an operation result, or otherwise matches if the operation
     * pattern matches.
     *
     * @param opMatcher the operation pattern
     * @return the value pattern.
     */
    public static Pattern valueP(OpPattern opMatcher) {
        return new ValuePattern(opMatcher);
    }

    /**
     * Creates an operation result pattern that conditionally matches an operation result and captures it in match state.
     *
     * @return the operation result.
     */
    public static Pattern opResultP() {
        return new OpResultPattern();
    }

    /**
     * Creates an operation result pattern that conditionally matches an operation result and captures it in match state,
     * then the operation pattern is matched against the result's operation.
     *
     * @param opMatcher the operation pattern
     * @return the operation result.
     */
    public static Pattern opResultP(OpPattern opMatcher) {
        return new OpResultPattern(opMatcher);
    }

    /**
     * Creates a block parameter result pattern that conditionally matches a block parameter and captures it in match state.
     *
     * @return the block parameter.
     */
    public static Pattern blockParameterP() {
        return new BlockParameterPattern();
    }

    /**
     * Creates a pattern that unconditionally matches any value or operation.
     *
     * @return the value pattern.
     */
    public static Pattern _P() {
        return new AnyPattern();
    }
}

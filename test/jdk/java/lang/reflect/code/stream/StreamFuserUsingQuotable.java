/*
 * Copyright (c) 2024, Oracle and/or its affiliates. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.
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

import jdk.incubator.code.*;
import jdk.incubator.code.analysis.Inliner;
import jdk.incubator.code.dialect.core.CoreType;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.JavaOp.EnhancedForOp;
import jdk.incubator.code.dialect.java.ClassType;
import jdk.incubator.code.dialect.java.JavaType;
import java.util.ArrayList;
import java.util.List;
import java.util.function.*;

import static jdk.incubator.code.dialect.core.CoreOp.*;
import static jdk.incubator.code.dialect.java.JavaOp.continue_;
import static jdk.incubator.code.dialect.java.JavaOp.enhancedFor;
import static jdk.incubator.code.dialect.java.JavaType.parameterized;
import static jdk.incubator.code.dialect.java.JavaType.type;

public final class StreamFuserUsingQuotable {

    // Quotable functional interfaces

    public interface QuotablePredicate<T> extends Quotable, Predicate<T> {
    }

    public interface QuotableFunction<T, R> extends Quotable, Function<T, R> {
    }

    public interface QuotableSupplier<T> extends Quotable, Supplier<T> {
    }

    public interface QuotableConsumer<T> extends Quotable, Consumer<T> {
    }

    public interface QuotableBiConsumer<T, U> extends Quotable, BiConsumer<T, U> {
    }


    StreamFuserUsingQuotable() {}

    public static <T> StreamExprBuilder<T> fromList(Class<T> elementClass) {
        JavaType elementType = type(elementClass);
        // java.util.List<E>
        JavaType listType = parameterized(type(List.class), elementType);
        return new StreamExprBuilder<>(listType, elementType,
                (b, v) -> StreamExprBuilder.enhancedForLoop(b, elementType, v)::body);
    }

    public static class StreamExprBuilder<T> {
        static class StreamOp {
            final JavaOp.LambdaOp lambdaOp;

            StreamOp(Quotable quotedLambda) {
                if (!(Op.ofQuotable(quotedLambda).get().op() instanceof JavaOp.LambdaOp lambdaOp)) {
                    throw new IllegalArgumentException("Quotable operation is not lambda operation");
                }
                if (!(Op.ofQuotable(quotedLambda).get().capturedValues().isEmpty())) {
                    throw new IllegalArgumentException("Quotable operation captures values");
                }
                this.lambdaOp = lambdaOp;
            }

            JavaOp.LambdaOp op() {
                return lambdaOp;
            }
        }

        static class MapStreamOp extends StreamOp {
            public MapStreamOp(Quotable quotedLambda) {
                super(quotedLambda);
            }
        }

        static class FlatMapStreamOp extends StreamOp {
            public FlatMapStreamOp(Quotable quotedLambda) {
                super(quotedLambda);
            }
        }

        static class FilterStreamOp extends StreamOp {
            public FilterStreamOp(Quotable quotedLambda) {
                super(quotedLambda);
            }
        }

        final JavaType sourceType;
        final JavaType sourceElementType;
        final BiFunction<Body.Builder, Value, Function<Consumer<Block.Builder>, Op>> loopSupplier;
        final List<StreamOp> streamOps;

        StreamExprBuilder(JavaType sourceType, JavaType sourceElementType,
                          BiFunction<Body.Builder, Value, Function<Consumer<Block.Builder>, Op>> loopSupplier) {
            this.sourceType = sourceType;
            this.sourceElementType = sourceElementType;
            this.loopSupplier = loopSupplier;
            this.streamOps = new ArrayList<>();
        }

        static EnhancedForOp.BodyBuilder enhancedForLoop(Body.Builder ancestorBody, JavaType elementType,
                                                         Value iterable) {
            return enhancedFor(ancestorBody, iterable.type(), elementType)
                    .expression(b -> {
                        b.op(core_yield(iterable));
                    })
                    .definition(b -> {
                        b.op(core_yield(b.parameters().get(0)));
                    });
        }

        @SuppressWarnings("unchecked")
        public <R> StreamExprBuilder<R> map(QuotableFunction<T, R> f) {
            streamOps.add(new MapStreamOp(f));
            return (StreamExprBuilder<R>) this;
        }

        @SuppressWarnings("unchecked")
        public <R> StreamExprBuilder<R> flatMap(QuotableFunction<T, Iterable<R>> f) {
            streamOps.add(new FlatMapStreamOp(f));
            return (StreamExprBuilder<R>) this;
        }

        public StreamExprBuilder<T> filter(QuotablePredicate<T> f) {
            streamOps.add(new FilterStreamOp(f));
            return this;
        }

        void fuseIntermediateOperations(Block.Builder body, BiConsumer<Block.Builder, Value> terminalConsumer) {
            fuseIntermediateOperation(0, body, body.parameters().get(0), null, terminalConsumer);
        }

        void fuseIntermediateOperation(int i, Block.Builder body, Value element, Block.Builder continueBlock,
                                       BiConsumer<Block.Builder, Value> terminalConsumer) {
            if (i == streamOps.size()) {
                terminalConsumer.accept(body, element);
                return;
            }

            StreamOp sop = streamOps.get(i);
            if (sop instanceof MapStreamOp) {
                Inliner.inline(body, sop.op(), List.of(element), (block, value) -> {
                    fuseIntermediateOperation(i + 1, block, value, continueBlock, terminalConsumer);
                });
            } else if (sop instanceof FilterStreamOp) {
                Inliner.inline(body, sop.op(), List.of(element), (block, p) -> {
                    Block.Builder _if = block.block();
                    Block.Builder _else = continueBlock;
                    if (continueBlock == null) {
                        _else = block.block();
                        _else.op(JavaOp.continue_());
                    }

                    block.op(conditionalBranch(p, _if.successor(), _else.successor()));

                    fuseIntermediateOperation(i + 1, _if, element, _else, terminalConsumer);
                });
            } else if (sop instanceof FlatMapStreamOp) {
                Inliner.inline(body, sop.op(), List.of(element), (block, iterable) -> {
                    EnhancedForOp forOp = enhancedFor(block.parentBody(),
                            iterable.type(), ((ClassType) iterable.type()).typeArguments().get(0))
                            .expression(b -> {
                                b.op(core_yield(iterable));
                            })
                            .definition(b -> {
                                b.op(core_yield(b.parameters().get(0)));
                            })
                            .body(b -> {
                                fuseIntermediateOperation(i + 1,
                                        b,
                                        b.parameters().get(0),
                                        null, terminalConsumer);
                            });

                    block.op(forOp);
                    block.op(JavaOp.continue_());
                });
            }
        }

        public FuncOp forEach(QuotableConsumer<T> quotableConsumer) {
            if (!(Op.ofQuotable(quotableConsumer).get().op() instanceof JavaOp.LambdaOp consumer)) {
                throw new IllegalArgumentException("Quotable consumer is not lambda operation");
            }
            if (!(Op.ofQuotable(quotableConsumer).get().capturedValues().isEmpty())) {
                throw new IllegalArgumentException("Quotable consumer captures values");
            }

            return func("fused.forEach", CoreType.functionType(JavaType.VOID, sourceType))
                    .body(b -> {
                        Value source = b.parameters().get(0);

                        Op sourceLoop = loopSupplier.apply(b.parentBody(), source)
                                .apply(loopBlock -> {
                                    fuseIntermediateOperations(loopBlock, (terminalBlock, resultValue) -> {
                                        Inliner.inline(terminalBlock, consumer, List.of(resultValue),
                                                (_, _) -> {
                                                });
                                        terminalBlock.op(JavaOp.continue_());
                                    });

                                });
                        b.op(sourceLoop);
                        b.op(return_());
                    });
        }

        public <C> FuncOp collect(QuotableSupplier<C> quotableSupplier, QuotableBiConsumer<C, T> quotableAccumulator) {
            if (!(Op.ofQuotable(quotableSupplier).get().op() instanceof JavaOp.LambdaOp supplier)) {
                throw new IllegalArgumentException("Quotable supplier is not lambda operation");
            }
            if (!(Op.ofQuotable(quotableSupplier).get().capturedValues().isEmpty())) {
                throw new IllegalArgumentException("Quotable supplier captures values");
            }
            if (!(Op.ofQuotable(quotableAccumulator).get().op() instanceof JavaOp.LambdaOp accumulator)) {
                throw new IllegalArgumentException("Quotable accumulator is not lambda operation");
            }
            if (!(Op.ofQuotable(quotableAccumulator).get().capturedValues().isEmpty())) {
                throw new IllegalArgumentException("Quotable accumulator captures values");
            }

            JavaType collectType = (JavaType) supplier.invokableType().returnType();
            return func("fused.collect", CoreType.functionType(collectType, sourceType))
                    .body(b -> {
                        Value source = b.parameters().get(0);

                        Inliner.inline(b, supplier, List.of(), (block, collect) -> {
                            Op sourceLoop = loopSupplier.apply(block.parentBody(), source)
                                    .apply(loopBlock -> {
                                        fuseIntermediateOperations(loopBlock, (terminalBlock, resultValue) -> {
                                            Inliner.inline(terminalBlock, accumulator, List.of(collect, resultValue),
                                                    (_, _) -> {
                                                    });
                                            terminalBlock.op(JavaOp.continue_());
                                        });
                                    });
                            block.op(sourceLoop);
                            block.op(return_(collect));
                        });
                    });
        }

    }
}


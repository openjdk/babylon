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

import java.lang.reflect.code.*;
import java.lang.reflect.code.op.ExtendedOp.JavaEnhancedForOp;
import java.lang.reflect.code.type.ClassType;
import java.lang.reflect.code.type.FunctionType;
import java.lang.reflect.code.type.JavaType;
import java.util.ArrayList;
import java.util.List;
import java.util.function.*;

import static java.lang.reflect.code.op.CoreOp.*;
import static java.lang.reflect.code.op.ExtendedOp._continue;
import static java.lang.reflect.code.op.ExtendedOp.enhancedFor;
import static java.lang.reflect.code.type.JavaType.type;

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
        JavaType listType = type(type(List.class), elementType);
        return new StreamExprBuilder<>(listType, elementType,
                (b, v) -> StreamExprBuilder.enhancedForLoop(b, elementType, v)::body);
    }

    public static class StreamExprBuilder<T> {
        static class StreamOp {
            final LambdaOp lambdaOp;

            StreamOp(Quotable quotedLambda) {
                if (!(quotedLambda.quoted().op() instanceof LambdaOp lambdaOp)) {
                    throw new IllegalArgumentException("Quotable operation is not lambda operation");
                }
                if (!(quotedLambda.quoted().capturedValues().isEmpty())) {
                    throw new IllegalArgumentException("Quotable operation captures values");
                }
                this.lambdaOp = lambdaOp;
            }

            LambdaOp op() {
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

        static JavaEnhancedForOp.BodyBuilder enhancedForLoop(Body.Builder ancestorBody, JavaType elementType,
                                                             Value iterable) {
            return enhancedFor(ancestorBody, iterable.type(), elementType)
                    .expression(b -> {
                        b.op(_yield(iterable));
                    })
                    .definition(b -> {
                        b.op(_yield(b.parameters().get(0)));
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
                body.inline(sop.op(), List.of(element), (block, value) -> {
                    fuseIntermediateOperation(i + 1, block, value, continueBlock, terminalConsumer);
                });
            } else if (sop instanceof FilterStreamOp) {
                body.inline(sop.op(), List.of(element), (block, p) -> {
                    Block.Builder _if = block.block();
                    Block.Builder _else = continueBlock;
                    if (continueBlock == null) {
                        _else = block.block();
                        _else.op(_continue());
                    }

                    block.op(conditionalBranch(p, _if.successor(), _else.successor()));

                    fuseIntermediateOperation(i + 1, _if, element, _else, terminalConsumer);
                });
            } else if (sop instanceof FlatMapStreamOp) {
                body.inline(sop.op(), List.of(element), (block, iterable) -> {
                    JavaEnhancedForOp forOp = enhancedFor(block.parentBody(),
                            iterable.type(), ((ClassType) iterable.type()).typeArguments().get(0))
                            .expression(b -> {
                                b.op(_yield(iterable));
                            })
                            .definition(b -> {
                                b.op(_yield(b.parameters().get(0)));
                            })
                            .body(b -> {
                                fuseIntermediateOperation(i + 1,
                                        b,
                                        b.parameters().get(0),
                                        null, terminalConsumer);
                            });

                    block.op(forOp);
                    block.op(_continue());
                });
            }
        }

        public FuncOp forEach(QuotableConsumer<T> quotableConsumer) {
            if (!(quotableConsumer.quoted().op() instanceof LambdaOp consumer)) {
                throw new IllegalArgumentException("Quotable consumer is not lambda operation");
            }
            if (!(quotableConsumer.quoted().capturedValues().isEmpty())) {
                throw new IllegalArgumentException("Quotable consumer captures values");
            }

            return func("fused.forEach", FunctionType.functionType(JavaType.VOID, sourceType))
                    .body(b -> {
                        Value source = b.parameters().get(0);

                        Op sourceLoop = loopSupplier.apply(b.parentBody(), source)
                                .apply(loopBlock -> {
                                    fuseIntermediateOperations(loopBlock, (terminalBlock, resultValue) -> {
                                        terminalBlock.inline(consumer, List.of(resultValue),
                                                (_, _) -> {
                                                });
                                        terminalBlock.op(_continue());
                                    });

                                });
                        b.op(sourceLoop);
                        b.op(_return());
                    });
        }

        public <C> FuncOp collect(QuotableSupplier<C> quotableSupplier, QuotableBiConsumer<C, T> quotableAccumulator) {
            if (!(quotableSupplier.quoted().op() instanceof LambdaOp supplier)) {
                throw new IllegalArgumentException("Quotable supplier is not lambda operation");
            }
            if (!(quotableSupplier.quoted().capturedValues().isEmpty())) {
                throw new IllegalArgumentException("Quotable supplier captures values");
            }
            if (!(quotableAccumulator.quoted().op() instanceof LambdaOp accumulator)) {
                throw new IllegalArgumentException("Quotable accumulator is not lambda operation");
            }
            if (!(quotableAccumulator.quoted().capturedValues().isEmpty())) {
                throw new IllegalArgumentException("Quotable accumulator captures values");
            }

            JavaType collectType = (JavaType) supplier.invokableType().returnType();
            return func("fused.collect", FunctionType.functionType(collectType, sourceType))
                    .body(b -> {
                        Value source = b.parameters().get(0);

                        b.inline(supplier, List.of(), (block, collect) -> {
                            Op sourceLoop = loopSupplier.apply(block.parentBody(), source)
                                    .apply(loopBlock -> {
                                        fuseIntermediateOperations(loopBlock, (terminalBlock, resultValue) -> {
                                            terminalBlock.inline(accumulator, List.of(collect, resultValue),
                                                    (_, _) -> {
                                                    });
                                            terminalBlock.op(_continue());
                                        });
                                    });
                            block.op(sourceLoop);
                            block.op(_return(collect));
                        });
                    });
        }

    }
}


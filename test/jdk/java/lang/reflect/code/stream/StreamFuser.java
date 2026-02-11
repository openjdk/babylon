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
import jdk.incubator.code.dialect.core.Inliner;
import jdk.incubator.code.dialect.core.CoreType;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.JavaOp.EnhancedForOp;
import jdk.incubator.code.dialect.java.ClassType;
import jdk.incubator.code.dialect.java.JavaType;
import java.util.ArrayList;
import java.util.List;
import java.util.function.*;

import static jdk.incubator.code.dialect.core.CoreOp.*;
import static jdk.incubator.code.dialect.java.JavaOp.enhancedFor;
import static jdk.incubator.code.dialect.java.JavaType.parameterized;
import static jdk.incubator.code.dialect.java.JavaType.type;

public final class StreamFuser {

    StreamFuser() {}

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

            StreamOp(Object reflectableLambda) {
                Quoted<JavaOp.LambdaOp> quotedLambda = Op.ofLambda(reflectableLambda).orElseThrow();
                if (!(quotedLambda.capturedValues().isEmpty())) {
                    throw new IllegalArgumentException("Reflectable lambda captures values");
                }
                this.lambdaOp = quotedLambda.op();
            }

            JavaOp.LambdaOp op() {
                return lambdaOp;
            }
        }

        static class MapStreamOp extends StreamOp {
            public MapStreamOp(Object reflectableLambda) {
                super(reflectableLambda);
            }
        }

        static class FlatMapStreamOp extends StreamOp {
            public FlatMapStreamOp(Object reflectableLambda) {
                super(reflectableLambda);
            }
        }

        static class FilterStreamOp extends StreamOp {
            public FilterStreamOp(Object reflectableLambda) {
                super(reflectableLambda);
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
        public <R> StreamExprBuilder<R> map(Function<T, R> f) {
            streamOps.add(new MapStreamOp(f));
            return (StreamExprBuilder<R>) this;
        }

        @SuppressWarnings("unchecked")
        public <R> StreamExprBuilder<R> flatMap(Function<T, Iterable<R>> f) {
            streamOps.add(new FlatMapStreamOp(f));
            return (StreamExprBuilder<R>) this;
        }

        public StreamExprBuilder<T> filter(Predicate<T> f) {
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

        public FuncOp forEach(Consumer<T> reflectableConsumer) {
            Quoted<JavaOp.LambdaOp> quotedConsumer = Op.ofLambda(reflectableConsumer).orElseThrow();
            if (!(quotedConsumer.capturedValues().isEmpty())) {
                throw new IllegalArgumentException("Reflectable consumer captures values");
            }
            JavaOp.LambdaOp consumer = quotedConsumer.op();

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

        public <C> FuncOp collect(Supplier<C> reflectableSupplier, BiConsumer<C, T> reflectableAccumulator) {
            Quoted<JavaOp.LambdaOp> quotedSupplier = Op.ofLambda(reflectableSupplier).orElseThrow();
            if (!(quotedSupplier.capturedValues().isEmpty())) {
                throw new IllegalArgumentException("Reflectable supplier captures values");
            }
            JavaOp.LambdaOp supplier = quotedSupplier.op();
            Quoted<JavaOp.LambdaOp> quotedAccumulator = Op.ofLambda(reflectableAccumulator).orElseThrow();
            if (!(quotedAccumulator.capturedValues().isEmpty())) {
                throw new IllegalArgumentException("Reflectable accumulator captures values");
            }
            JavaOp.LambdaOp accumulator = quotedAccumulator.op();

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


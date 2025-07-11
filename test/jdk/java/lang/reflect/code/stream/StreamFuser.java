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
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.core.CoreType;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.JavaOp.EnhancedForOp;
import jdk.incubator.code.dialect.java.ClassType;
import jdk.incubator.code.dialect.java.JavaType;
import java.util.ArrayList;
import java.util.List;
import java.util.function.BiConsumer;
import java.util.function.BiFunction;
import java.util.function.Consumer;
import java.util.function.Function;

import static jdk.incubator.code.dialect.core.CoreOp.*;
import static jdk.incubator.code.dialect.java.JavaOp.continue_;
import static jdk.incubator.code.dialect.java.JavaOp.enhancedFor;
import static jdk.incubator.code.dialect.java.JavaType.parameterized;
import static jdk.incubator.code.dialect.java.JavaType.type;

public final class StreamFuser {

    StreamFuser() {}

    public static StreamExprBuilder fromList(JavaType elementType) {
        // java.util.List<E>
        JavaType listType = parameterized(type(List.class), elementType);
        return new StreamExprBuilder(listType, elementType,
                (b, v) -> StreamExprBuilder.enhancedForLoop(b, elementType, v)::body);
    }

    public static class StreamExprBuilder {
        static class StreamOp {
            final Quoted quotedClosure;

            StreamOp(Quoted quotedClosure) {
                if (!(quotedClosure.op() instanceof CoreOp.ClosureOp)) {
                    throw new IllegalArgumentException("Quoted operation is not closure operation");
                }
                this.quotedClosure = quotedClosure;
            }

            CoreOp.ClosureOp op() {
                return (CoreOp.ClosureOp) quotedClosure.op();
            }
        }

        static class MapStreamOp extends StreamOp {
            public MapStreamOp(Quoted quotedClosure) {
                super(quotedClosure);
                // @@@ Check closure signature
            }
        }

        static class FlatMapStreamOp extends StreamOp {
            public FlatMapStreamOp(Quoted quotedClosure) {
                super(quotedClosure);
                // @@@ Check closure signature
            }
        }

        static class FilterStreamOp extends StreamOp {
            public FilterStreamOp(Quoted quotedClosure) {
                super(quotedClosure);
                // @@@ Check closure signature
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

        public StreamExprBuilder map(Quoted f) {
            streamOps.add(new MapStreamOp(f));
            return this;
        }

        public StreamExprBuilder flatMap(Quoted f) {
            streamOps.add(new FlatMapStreamOp(f));
            return this;
        }

        public StreamExprBuilder filter(Quoted f) {
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

        public FuncOp forEach(Quoted quotedConsumer) {
            if (!(quotedConsumer.op() instanceof CoreOp.ClosureOp consumer)) {
                throw new IllegalArgumentException("Quoted consumer is not closure operation");
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

        // Supplier<C> supplier, BiConsumer<C, T> accumulator
        public FuncOp collect(Quoted quotedSupplier, Quoted quotedAccumulator) {
            if (!(quotedSupplier.op() instanceof CoreOp.ClosureOp supplier)) {
                throw new IllegalArgumentException("Quoted supplier is not closure operation");
            }
            if (!(quotedAccumulator.op() instanceof CoreOp.ClosureOp accumulator)) {
                throw new IllegalArgumentException("Quoted accumulator is not closure operation");
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

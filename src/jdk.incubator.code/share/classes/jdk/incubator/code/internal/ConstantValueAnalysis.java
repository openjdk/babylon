/*
 * Copyright (c) 2026, Oracle and/or its affiliates. All rights reserved.
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

package jdk.incubator.code.internal;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.function.Function;

import jdk.incubator.code.Block;
import jdk.incubator.code.Body;
import jdk.incubator.code.Op;
import jdk.incubator.code.Value;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.JavaType;
import jdk.incubator.code.dialect.java.PrimitiveType;

public final class ConstantValueAnalysis {
    private final Function<Block, List<List<Value>>> incomingArguments;
    private final Map<Value, Optional<Object>> cache = new HashMap<>();

    public ConstantValueAnalysis(Function<Block, List<List<Value>>> incomingArguments) {
        this.incomingArguments = incomingArguments;
    }

    void clearUnknowns() {
        cache.values().removeIf(Optional::isEmpty);
    }

    public Optional<Object> evaluate(Value value) {
        if (value == null
                || !(value.type() instanceof PrimitiveType && !value.type().equals(JavaType.VOID)
                || value.type().equals(JavaType.J_L_STRING))) {
            return Optional.empty();
        }
        Optional<Object> cached = cache.get(value);
        if (cached != null) {
            return cached;
        }
        cache.put(value, Optional.empty());
        Optional<Object> resultValue = switch (value) {
            case Block.Parameter parameter -> {
                List<List<Value>> incoming = incomingArguments.apply(parameter.declaringBlock());
                if (incoming.isEmpty()) {
                    yield Optional.empty();
                }
                Object agreed = null;
                for (List<Value> arguments : incoming) {
                    Optional<Object> constant = evaluate(arguments.get(parameter.index()));
                    if (constant.isEmpty()) {
                        yield Optional.empty();
                    }
                    Object candidate = constant.orElseThrow();
                    if (agreed != null && !agreed.equals(candidate)) {
                        yield Optional.empty();
                    }
                    agreed = candidate;
                }
                yield Optional.of(agreed);
            }
            case Op.Result result ->
                switch (result.op()) {
                    case CoreOp.ConstantOp constant ->
                            Optional.ofNullable(constant.value());
                    case JavaOp.ArithmeticOperation _, JavaOp.ConvOp _ -> {
                        List<Object> operands = new ArrayList<>(result.op().operands().size());
                        for (Value operand : result.op().operands()) {
                            Optional<Object> constant = evaluate(operand);
                            if (constant.isEmpty()) {
                                yield Optional.empty();
                            }
                            operands.add(constant.orElseThrow());
                        }
                        yield Optional.of(ArithmeticAndConvOpImpls.evaluate(result.op(), operands));
                    }
                    case JavaOp.CastOp co when value.type().equals(JavaType.J_L_STRING) ->
                        evaluate(co.operands().getFirst()).filter(String.class::isInstance);
                    case JavaOp.ConditionalExpressionOp co ->
                        yieldValue(co.predicateBody()).filter(Boolean.class::isInstance)
                                .map(v -> yieldValue((Boolean) v ? co.trueBody() : co.falseBody()).orElse(null));
                    case JavaOp.ConditionalOrOp co ->
                        co.bodies().stream().map(this::yieldValue)
                                .filter(v -> v.isEmpty() || v.get().equals(true))
                                .findFirst().orElse(Optional.of(false));
                    case JavaOp.ConditionalAndOp co ->
                        co.bodies().stream().map(this::yieldValue)
                                .filter(v -> v.isEmpty() || v.get().equals(false))
                                .findFirst().orElse(Optional.of(true));
                    default ->  Optional.empty();
                };
        };
        cache.put(value, resultValue);
        return resultValue;
    }

    public Boolean booleanValue(Value value) {
        return evaluate(value).filter(Boolean.class::isInstance).map(Boolean.class::cast).orElse(null);
    }

    private Optional<Object> yieldValue(Body body) {
        return body.blocks().size() == 1 && body.entryBlock().terminatingOp() instanceof CoreOp.YieldOp yield
                ? evaluate(yield.yieldValue())
                : Optional.empty();
    }
}

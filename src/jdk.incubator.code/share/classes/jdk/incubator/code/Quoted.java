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

package jdk.incubator.code;

import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.core.FunctionType;
import jdk.incubator.code.dialect.core.VarType;

import java.util.*;
import java.util.function.Consumer;
import java.util.stream.Stream;

/**
 * The quoted form of an operation.
 * <p>
 * The quoted form is utilized when the code model of some code is to be obtained rather than obtaining the result of
 * executing that code. For example passing the of a lambda expression in quoted form rather than the expression being
 * targeted to a functional interface from which it can be invoked.
 */
public final class Quoted {
    private final Op op;
    private final SequencedMap<Value, Object> capturedValues;

    static final SequencedMap<Value, Object> EMPTY_SEQUENCED_MAP = new LinkedHashMap<>();
    /**
     * Constructs the quoted form of a given operation.
     *
     * @param op the invokable operation.
     */
    public Quoted(Op op) {
        this(op, EMPTY_SEQUENCED_MAP);
    }

    /**
     * Constructs the quoted form of a given operation.
     * <p>
     * The captured values key set must have the same elements and same encounter order as
     * operation's captured values, specifically the following expression should evaluate to true:
     * {@snippet lang=java :
     * op.capturedValues().equals(new ArrayList<>(capturedValues.keySet()));
     * }
     *
     * @param op             the operation.
     * @param capturedValues the captured values referred to by the operation
     * @see Op#capturedValues()
     */
    public Quoted(Op op, SequencedMap<Value, Object> capturedValues) {
        // @@@ This check is potentially expensive, remove or keep as assert?
        // @@@ Or make Quoted an interface, with a module private implementation?
        assert Stream.concat(op.operands().stream(), op.capturedValues().stream()).toList()
                .equals(new ArrayList<>(capturedValues.keySet()));
        this.op = op;
        this.capturedValues = Collections.unmodifiableSequencedMap(new LinkedHashMap<>(capturedValues));
    }

    /**
     * Returns the operation.
     *
     * @return the operation.
     */
    public Op op() {
        return op;
    }

    /**
     * Returns the captured values.
     * <p>
     * The captured values key set has the same elements and same encounter order as
     * operation's captured values, specifically the following expression evaluates to true:
     * {@snippet lang=java :
     * op().capturedValues().equals(new ArrayList<>(capturedValues().keySet()));
     * }
     *
     * @return the captured values, as an unmodifiable map.
     */
    public SequencedMap<Value, Object> capturedValues() {
        return capturedValues;
    }

    public static CoreOp.FuncOp quoteOp(Op op) {

        if (op.result() == null) {
            throw new IllegalArgumentException("Op not bound");
        }

        List<Value> inputOperandsAndCaptures = Stream.concat(op.operands().stream(), op.capturedValues().stream()).toList();

        // Build the function type
        List<TypeElement> params = inputOperandsAndCaptures.stream()
                .map(v -> v.type() instanceof VarType vt ? vt.valueType() : v.type())
                .toList();
        FunctionType ft = FunctionType.functionType(CoreOp.QuotedOp.QUOTED_TYPE, params);

        // Build the function that quotes the lambda
        return CoreOp.func("q", ft).body(b -> {
            // Create variables as needed and obtain the operands and captured values for the copied lambda
            List<Value> outputOperandsAndCaptures = new ArrayList<>();
            for (int i = 0; i < inputOperandsAndCaptures.size(); i++) {
                Value inputValue = inputOperandsAndCaptures.get(i);
                Value outputValue = b.parameters().get(i);
                if (inputValue.type() instanceof VarType) {
                    outputValue = b.op(CoreOp.var(String.valueOf(i), outputValue));
                }
                outputOperandsAndCaptures.add(outputValue);
            }

            // Quoted the lambda expression
            Value q = b.op(CoreOp.quoted(b.parentBody(), qb -> {
                // Map the entry block of the op's ancestor body to the quoted block
                // We are copying op in the context of the quoted block, the block mapping
                // ensures the use of operands and captured values are reachable when building
                qb.context().mapBlock(op.ancestorBody().entryBlock(), qb);
                // Map the op's operands and captured values
                qb.context().mapValues(inputOperandsAndCaptures, outputOperandsAndCaptures);
                // Return the op to be copied in the quoted operation
                return op;
            }));
            b.op(CoreOp._return(q));
        });
    }

    public static Quoted quotedOp(CoreOp.FuncOp funcOp, Object[] args) {

        if (funcOp.body().blocks().size() != 1) {
            throw new IllegalArgumentException("Argument operation has more then one block");
        }
        Block fblock = funcOp.body().entryBlock();

        if (fblock.ops().size() < 2) {
            throw new IllegalArgumentException();
        }

        if (!(fblock.ops().get(fblock.ops().size() - 2) instanceof CoreOp.QuotedOp qop)) {
            throw new IllegalArgumentException("Before last operation is not a QuotedOp");
        }

        if (!(fblock.ops().getLast() instanceof CoreOp.ReturnOp returnOp)) {
            throw new IllegalArgumentException("Last operation not a ReturnOp");
        }
        if (returnOp.returnValue() == null) {
            throw new IllegalArgumentException();
        }
        if (!returnOp.returnValue().equals(qop.result())) {
            throw new IllegalArgumentException("Argument operation doesn't return the result of QuotedOp");
        }

        Op op = qop.quotedOp();

        SequencedSet<Value> operandsAndCaptures = new LinkedHashSet<>();
        operandsAndCaptures.addAll(op.operands());
        operandsAndCaptures.addAll(op.capturedValues());

        // validation rule of block params and constant op result
        Consumer<Value> validate = v -> {
            if (v.uses().isEmpty()) {
                throw new IllegalArgumentException();
            } else if (v.uses().size() == 1
                    && !(v.uses().iterator().next().op() instanceof CoreOp.VarOp vop && vop.result().uses().size() >= 1
                    && vop.result().uses().stream().noneMatch(u -> u.op().parentBlock() == fblock))
                    && !operandsAndCaptures.contains(v)) {
                throw new IllegalArgumentException();
            } else if (v.uses().size() > 1 && v.uses().stream().anyMatch(u -> u.op().parentBlock() == fblock)) {
                throw new IllegalArgumentException();
            }
        };

        for (Block.Parameter p : fblock.parameters()) {
            validate.accept(p);
        }

        List<Op> ops = fblock.ops().subList(0, fblock.ops().size() - 2);
        for (Op o : ops) {
            switch (o) {
                case CoreOp.VarOp varOp -> {
                    if (varOp.isUninitialized()) {
                        throw new IllegalArgumentException();
                    }
                    if (varOp.initOperand() instanceof Op.Result opr && !(opr.op() instanceof CoreOp.ConstantOp)) {
                        throw new IllegalArgumentException("VarOp initial value came from an operation that's not a ConstantOp");
                    }
                }
                case CoreOp.ConstantOp cop -> validate.accept(cop.result());
                default -> throw new IllegalArgumentException("Operation not a VarOp nor a ConstantOp, " + o);
            }
        }

        // map captured values to their corresponding runtime values
        // captured value can be:
        // 1- block param
        // 2- result of VarOp whose initial value is constant
        // 3- result of VarOp whose initial value is block param
        // 4- result of ConstantOp
        List<Block.Parameter> params = funcOp.parameters();
        if (params.size() != args.length) {
            throw new IllegalArgumentException();
        }
        SequencedMap<Value, Object> m = new LinkedHashMap<>();
        for (Value v : operandsAndCaptures) {
            switch (v) {
                case Block.Parameter p -> {
                    Object rv = args[p.index()];
                    m.put(v, rv);
                }
                case Op.Result opr when opr.op() instanceof CoreOp.VarOp varOp -> {
                    if (varOp.initOperand() instanceof Op.Result r && r.op() instanceof CoreOp.ConstantOp cop) {
                        m.put(v, CoreOp.Var.of(cop.value()));
                    } else if (varOp.initOperand() instanceof Block.Parameter p) {
                        Object rv = args[p.index()];
                        m.put(v, CoreOp.Var.of(rv));
                    }
                }
                case Op.Result opr when opr.op() instanceof CoreOp.ConstantOp cop -> {
                    m.put(v, cop.value());
                }
                default -> throw new IllegalArgumentException();
            }
        }

        return new Quoted(op, m);
    }
}

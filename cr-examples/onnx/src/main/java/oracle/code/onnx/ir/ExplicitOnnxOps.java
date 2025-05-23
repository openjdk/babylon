/*
 * Copyright (c) 2025, Oracle and/or its affiliates. All rights reserved.
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

package oracle.code.onnx.ir;

import java.util.*;
import jdk.incubator.code.*;
import jdk.incubator.code.Op.Nested;
import jdk.incubator.code.op.ExternalizableOp;
import jdk.incubator.code.op.OpFactory;

public sealed class ExplicitOnnxOps permits OnnxOps {

    // @@@ this should be generated from contrib operators
    @OpFactory.OpDeclaration(GroupQueryAttention.NAME)
    public static final class GroupQueryAttention extends OnnxOp {
        public static final String NAME = "com.microsoft.GroupQueryAttention";

        public enum Attribute implements OnnxAttribute {
            do_rotary(Long.class, true, 0),
            kv_num_heads(Long.class, false, null),
            local_window_size(Long.class, true, -1),
            num_heads(Long.class, false, null),
            rotary_interleaved(Long.class, true, 0),
            scale(Float.class, true, null), // @@@ Default value is 1/sqrt(head_size)
            ;

                final Class<?> t;
                final boolean optional;
                final Object defaultValue;

                Attribute(Class<?> type, boolean optional, Object defaultValue) {
                    this.t = type;
                    this.optional = optional;
                    this.defaultValue = defaultValue;
                    assert optional || defaultValue == null;
                }

                public Class<?> type() {
                    return t;
                }

                public boolean isOptional() {
                    return optional;
                }

                public Object defaultValue() {
                    return defaultValue;
                }
        }

        public enum TypeConstraint implements OnnxTypeConstraint {
            T(new OnnxType.TypeVariable("T", List.of(OnnxType.tensor(OnnxType.float16()), OnnxType.tensor(OnnxType.bfloat16()), OnnxType.tensor(OnnxType.float32())))),
            M(new OnnxType.TypeVariable("M", List.of(OnnxType.tensor(OnnxType.int32())))),
            ;

            final OnnxType.TypeVariable typeVariable;

            TypeConstraint(OnnxType.TypeVariable typeVariable) {
                assert typeVariable.name().equals(name());
                this.typeVariable = typeVariable;
            }

            @Override
            public OnnxType.TypeVariable typeVariable() {
                return typeVariable;
            }
        }

        public enum InputParameter implements OnnxParameter {
            query(TypeConstraint.T.typeVariable(), Quantifier.REQUIRED),
            key(TypeConstraint.T.typeVariable(), Quantifier.OPTIONAL),
            value(TypeConstraint.T.typeVariable(), Quantifier.OPTIONAL),
            past_key(TypeConstraint.T.typeVariable(), Quantifier.OPTIONAL),
            past_value(TypeConstraint.T.typeVariable(), Quantifier.OPTIONAL),
            seqlens_k(TypeConstraint.M.typeVariable(), Quantifier.REQUIRED),
            total_sequence_length(TypeConstraint.M.typeVariable(), Quantifier.REQUIRED),
            cos_cache(TypeConstraint.T.typeVariable(), Quantifier.OPTIONAL),
            sin_cache(TypeConstraint.T.typeVariable(), Quantifier.OPTIONAL),
            ;

            final OnnxType type;
            final Quantifier quantifier;

            InputParameter(OnnxType type, Quantifier quantifier) {
                this.type = type;
                this.quantifier = quantifier;
            }

            @Override
            public OnnxType type() {
                return type;
            }

            @Override
            public Quantifier quantifier() {
                return quantifier;
            }
        }

        public enum OutputParameter implements OnnxParameter {
            output(TypeConstraint.T.typeVariable(), Quantifier.REQUIRED),
            present_key(TypeConstraint.T.typeVariable(), Quantifier.REQUIRED),
            present_value(TypeConstraint.T.typeVariable(), Quantifier.REQUIRED),
            ;

            final OnnxType type;
            final Quantifier quantifier;

            OutputParameter(OnnxType type, Quantifier quantifier) {
                this.type = type;
                this.quantifier = quantifier;
            }

            @Override
            public OnnxType type() {
                return type;
            }

            @Override
            public Quantifier quantifier() {
                return quantifier;
            }
        }

        public static final OnnxSchema SCHEMA = new OnnxSchemaRecord(
                NAME,
                List.of(Attribute.values()),
                List.of(TypeConstraint.values()),
                List.of(InputParameter.values()),
                List.of(OutputParameter.values())
        );

        public GroupQueryAttention(ExternalizedOp def) {
            super(SCHEMA, def);
        }

        GroupQueryAttention(GroupQueryAttention that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public GroupQueryAttention transform(CopyContext cc, OpTransformer ot) {
            return new GroupQueryAttention(this, cc);
        }

        GroupQueryAttention(TypeElement resultType, Value query, java.util.Optional<Value> key, java.util.Optional<Value> value, java.util.Optional<Value> past_key, java.util.Optional<Value> past_value, Value seqlens_k, Value total_sequence_length, java.util.Optional<Value> cos_cache, java.util.Optional<Value> sin_cache, java.util.Optional<Value> do_rotary, Value kv_num_heads, java.util.Optional<Value> local_window_size, Value num_heads, java.util.Optional<Value> rotary_interleaved, java.util.Optional<Value> scale) {
            super(SCHEMA, resultType, Collections.emptySet(), List.of(query, key, value, past_key, past_value, seqlens_k, total_sequence_length, cos_cache, sin_cache), List.of(do_rotary, kv_num_heads, local_window_size, num_heads, rotary_interleaved, scale));
        }

        @Override
        public SequencedSet<OnnxParameter> onnxOutputs() {
            return onnxOutputs(SCHEMA);
        }

        @Override
        public SequencedMap<OnnxParameter, Object> onnxInputs() {
            return onnxInputs(SCHEMA, List.of(query(), key(), value(), past_key(), past_value(), seqlens_k(), total_sequence_length(), cos_cache(), sin_cache()));
        }

        public Value query() {
            return operands().get(0);
        }

        public java.util.Optional<Value> key() {
            int i = optionalInputArguments.indexOf(InputParameter.key);
            return i != -1 ? java.util.Optional.of(operands().get(1 + i)) : java.util.Optional.empty();
        }

        public java.util.Optional<Value> value() {
            int i = optionalInputArguments.indexOf(InputParameter.value);
            return i != -1 ? java.util.Optional.of(operands().get(1 + i)) : java.util.Optional.empty();
        }

        public java.util.Optional<Value> past_key() {
            int i = optionalInputArguments.indexOf(InputParameter.past_key);
            return i != -1 ? java.util.Optional.of(operands().get(1 + i)) : java.util.Optional.empty();
        }

        public java.util.Optional<Value> past_value() {
            int i = optionalInputArguments.indexOf(InputParameter.past_value);
            return i != -1 ? java.util.Optional.of(operands().get(1 + i)) : java.util.Optional.empty();
        }

        private int skipOptional() {
            for (int i = optionalInputArguments.size() - 1; i >= 0; i--) {
                var opt = optionalInputArguments.get(i);
                if (opt != InputParameter.cos_cache && opt != InputParameter.sin_cache) return i;
            }
            return -1;
        }

        public Value seqlens_k() {
            return operands().get(skipOptional() + 2);
        }

        public Value total_sequence_length() {
            return operands().get(skipOptional() + 3);
        }

        public java.util.Optional<Value> cos_cache() {
            int i = optionalInputArguments.indexOf(InputParameter.cos_cache);
            return i != -1 ? java.util.Optional.of(operands().get(3 + i)) : java.util.Optional.empty();
        }

        public java.util.Optional<Value> sin_cache() {
            int i = optionalInputArguments.indexOf(InputParameter.sin_cache);
            return i != -1 ? java.util.Optional.of(operands().get(3 + i)) : java.util.Optional.empty();
        }
    }

    public static GroupQueryAttention GroupQueryAttention(TypeElement resultType, Value query, java.util.Optional<Value> key, java.util.Optional<Value> value, java.util.Optional<Value> past_key, java.util.Optional<Value> past_value, Value seqlens_k, Value total_sequence_length, java.util.Optional<Value> cos_cache, java.util.Optional<Value> sin_cache, java.util.Optional<Value> do_rotary, Value kv_num_heads, java.util.Optional<Value> local_window_size, Value num_heads, java.util.Optional<Value> rotary_interleaved, java.util.Optional<Value> scale) {
        return new GroupQueryAttention(resultType, query, key, value, past_key, past_value, seqlens_k, total_sequence_length, cos_cache, sin_cache, do_rotary, kv_num_heads, local_window_size, num_heads, rotary_interleaved, scale);
    }

    // @@@ this should be generated from contrib operators
    @OpFactory.OpDeclaration(MatMulNBits.NAME)
    public static final class MatMulNBits extends OnnxOp {
        public static final String NAME = "com.microsoft.MatMulNBits";

        public enum Attribute implements OnnxAttribute {
            K(Long.class, false, null),
            N(Long.class, false, null),
            accuracy_level(Long.class, true, 0),
            bits(Long.class, false, null),
            block_size(Long.class, false, null),
            ;

                final Class<?> t;
                final boolean optional;
                final Object defaultValue;

                Attribute(Class<?> type, boolean optional, Object defaultValue) {
                    this.t = type;
                    this.optional = optional;
                    this.defaultValue = defaultValue;
                    assert optional || defaultValue == null;
                }

                public Class<?> type() {
                    return t;
                }

                public boolean isOptional() {
                    return optional;
                }

                public Object defaultValue() {
                    return defaultValue;
                }
        }

        public enum TypeConstraint implements OnnxTypeConstraint {
            T1(new OnnxType.TypeVariable("T1", List.of(OnnxType.tensor(OnnxType.float32()), OnnxType.tensor(OnnxType.float16())))),
            T2(new OnnxType.TypeVariable("T2", List.of(OnnxType.tensor(OnnxType.uint8()), OnnxType.tensor(OnnxType.int32())))),
            T3(new OnnxType.TypeVariable("T3", List.of(OnnxType.tensor(OnnxType.uint8()), OnnxType.tensor(OnnxType.int32()), OnnxType.tensor(OnnxType.float16()), OnnxType.tensor(OnnxType.float32())))),
            T4(new OnnxType.TypeVariable("T4", List.of(OnnxType.tensor(OnnxType.int32())))),
            ;

            final OnnxType.TypeVariable typeVariable;

            TypeConstraint(OnnxType.TypeVariable typeVariable) {
                assert typeVariable.name().equals(name());
                this.typeVariable = typeVariable;
            }

            @Override
            public OnnxType.TypeVariable typeVariable() {
                return typeVariable;
            }
        }

        public enum InputParameter implements OnnxParameter {
            A(TypeConstraint.T1.typeVariable(), Quantifier.REQUIRED),
            B(TypeConstraint.T2.typeVariable(), Quantifier.REQUIRED),
            scales(TypeConstraint.T1.typeVariable(), Quantifier.REQUIRED),
            zero_points(TypeConstraint.T3.typeVariable(), Quantifier.OPTIONAL),
            g_idx(TypeConstraint.T4.typeVariable(), Quantifier.OPTIONAL),
            bias(TypeConstraint.T1.typeVariable(), Quantifier.OPTIONAL),
            ;

            final OnnxType type;
            final Quantifier quantifier;

            InputParameter(OnnxType type, Quantifier quantifier) {
                this.type = type;
                this.quantifier = quantifier;
            }

            @Override
            public OnnxType type() {
                return type;
            }

            @Override
            public Quantifier quantifier() {
                return quantifier;
            }
        }

        public enum OutputParameter implements OnnxParameter {
            Y(TypeConstraint.T1.typeVariable(), Quantifier.REQUIRED),
            ;

            final OnnxType type;
            final Quantifier quantifier;

            OutputParameter(OnnxType type, Quantifier quantifier) {
                this.type = type;
                this.quantifier = quantifier;
            }

            @Override
            public OnnxType type() {
                return type;
            }

            @Override
            public Quantifier quantifier() {
                return quantifier;
            }
        }

        public static final OnnxSchema SCHEMA = new OnnxSchemaRecord(
                NAME,
                List.of(Attribute.values()),
                List.of(TypeConstraint.values()),
                List.of(InputParameter.values()),
                List.of(OutputParameter.values())
        );

        public MatMulNBits(ExternalizedOp def) {
            super(SCHEMA, def);
        }

        MatMulNBits(MatMulNBits that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public MatMulNBits transform(CopyContext cc, OpTransformer ot) {
            return new MatMulNBits(this, cc);
        }

        MatMulNBits(TypeElement resultType, Value a, Value b, Value scales, java.util.Optional<Value> zero_points, java.util.Optional<Value> g_idx, java.util.Optional<Value> bias, Value K, Value N, java.util.Optional<Value> accuracy_level, Value bits, Value block_size) {
            super(SCHEMA, resultType, Collections.emptySet(), List.of(a, b, scales, zero_points, g_idx, bias), List.of(K, N, accuracy_level, bits, block_size));
        }

        @Override
        public SequencedSet<OnnxParameter> onnxOutputs() {
            return onnxOutputs(SCHEMA);
        }

        @Override
        public SequencedMap<OnnxParameter, Object> onnxInputs() {
                    return onnxInputs(SCHEMA, List.of(a(), b(), scales(), zero_points(), g_idx(), bias()));
        }

        public Value a() {
            return operands().get(0);
        }

        public Value b() {
            return operands().get(1);
        }

        public Value scales() {
            return operands().get(2);
        }

        public java.util.Optional<Value> zero_points() {
            int i = optionalInputArguments.indexOf(InputParameter.zero_points);
            return i != -1 ? java.util.Optional.of(operands().get(3 + i)) : java.util.Optional.empty();
        }

        public java.util.Optional<Value> g_idx() {
            int i = optionalInputArguments.indexOf(InputParameter.g_idx);
            return i != -1 ? java.util.Optional.of(operands().get(3 + i)) : java.util.Optional.empty();
        }

        public java.util.Optional<Value> bias() {
            int i = optionalInputArguments.indexOf(InputParameter.bias);
            return i != -1 ? java.util.Optional.of(operands().get(3 + i)) : java.util.Optional.empty();
        }
    }

    public static MatMulNBits MatMulNBits(TypeElement resultType, Value a, Value b, Value scales, java.util.Optional<Value> zero_points, java.util.Optional<Value> g_idx, java.util.Optional<Value> bias, Value K, Value N, java.util.Optional<Value> accuracy_level, Value bits, Value block_size) {
        return new MatMulNBits(resultType, a, b, scales, zero_points, g_idx, bias, K, N, accuracy_level, bits, block_size);
    }

    // @@@ this should be generated from contrib operators
    @OpFactory.OpDeclaration(SkipSimplifiedLayerNormalization.NAME)
    public static final class SkipSimplifiedLayerNormalization extends OnnxOp {
        public static final String NAME = "com.microsoft.SkipSimplifiedLayerNormalization";

        public enum Attribute implements OnnxAttribute {
            epsilon(Float.class, true, null),
            ;

                final Class<?> t;
                final boolean optional;
                final Object defaultValue;

                Attribute(Class<?> type, boolean optional, Object defaultValue) {
                    this.t = type;
                    this.optional = optional;
                    this.defaultValue = defaultValue;
                    assert optional || defaultValue == null;
                }

                public Class<?> type() {
                    return t;
                }

                public boolean isOptional() {
                    return optional;
                }

                public Object defaultValue() {
                    return defaultValue;
                }
        }

        public enum TypeConstraint implements OnnxTypeConstraint {
            T(new OnnxType.TypeVariable("T", List.of(OnnxType.tensor(OnnxType.float32()), OnnxType.tensor(OnnxType.float16())))),
            ;

            final OnnxType.TypeVariable typeVariable;

            TypeConstraint(OnnxType.TypeVariable typeVariable) {
                assert typeVariable.name().equals(name());
                this.typeVariable = typeVariable;
            }

            @Override
            public OnnxType.TypeVariable typeVariable() {
                return typeVariable;
            }
        }

        public enum InputParameter implements OnnxParameter {
            input(TypeConstraint.T.typeVariable(), Quantifier.REQUIRED),
            skip(TypeConstraint.T.typeVariable(), Quantifier.REQUIRED),
            gamma(TypeConstraint.T.typeVariable(), Quantifier.REQUIRED),
            bias(TypeConstraint.T.typeVariable(), Quantifier.OPTIONAL),
            ;

            final OnnxType type;
            final Quantifier quantifier;

            InputParameter(OnnxType type, Quantifier quantifier) {
                this.type = type;
                this.quantifier = quantifier;
            }

            @Override
            public OnnxType type() {
                return type;
            }

            @Override
            public Quantifier quantifier() {
                return quantifier;
            }
        }

        public enum OutputParameter implements OnnxParameter {
            output(TypeConstraint.T.typeVariable(), Quantifier.REQUIRED),
            mean(OnnxType.TENSOR_FLOAT32, Quantifier.OPTIONAL),
            inv_std_var(OnnxType.TENSOR_FLOAT32, Quantifier.OPTIONAL),
            input_skip_bias_sum(OnnxType.TENSOR_FLOAT32, Quantifier.OPTIONAL),
            ;

            final OnnxType type;
            final Quantifier quantifier;

            OutputParameter(OnnxType type, Quantifier quantifier) {
                this.type = type;
                this.quantifier = quantifier;
            }

            @Override
            public OnnxType type() {
                return type;
            }

            @Override
            public Quantifier quantifier() {
                return quantifier;
            }
        }

        public static final OnnxSchema SCHEMA = new OnnxSchemaRecord(
                NAME,
                List.of(Attribute.values()),
                List.of(TypeConstraint.values()),
                List.of(InputParameter.values()),
                List.of(OutputParameter.values())
        );

        public SkipSimplifiedLayerNormalization(ExternalizedOp def) {
            super(SCHEMA, def);
        }

        SkipSimplifiedLayerNormalization(SkipSimplifiedLayerNormalization that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public SkipSimplifiedLayerNormalization transform(CopyContext cc, OpTransformer ot) {
            return new SkipSimplifiedLayerNormalization(this, cc);
        }

        SkipSimplifiedLayerNormalization(TypeElement resultType, Value input, Value skip, Value gamma, java.util.Optional<Value> bias, java.util.Optional<Value> epsilon) {
            super(SCHEMA, resultType, Collections.emptySet(), List.of(input, skip, gamma, bias), List.of(epsilon));
        }

        @Override
        public SequencedSet<OnnxParameter> onnxOutputs() {
            return onnxOutputs(SCHEMA);
        }

        @Override
        public SequencedMap<OnnxParameter, Object> onnxInputs() {
            return onnxInputs(SCHEMA, List.of(input(), skip(), gamma(), bias()));
        }

        public Value input() {
            return operands().get(0);
        }

        public Value skip() {
            return operands().get(1);
        }

        public Value gamma() {
            return operands().get(2);
        }

        public java.util.Optional<Value> bias() {
            int i = optionalInputArguments.indexOf(InputParameter.bias);
            return i != -1 ? java.util.Optional.of(operands().get(3 + i)) : java.util.Optional.empty();
        }
    }

    public static SkipSimplifiedLayerNormalization SkipSimplifiedLayerNormalization(TypeElement resultType, Value input, Value skip, Value gamma, java.util.Optional<Value> bias, java.util.Optional<Value> epsilon) {
        return new SkipSimplifiedLayerNormalization(resultType, input, skip, gamma, bias, epsilon);
    }




    @OpFactory.OpDeclaration(If.NAME)
    public static final class If extends OnnxOp implements Nested {
        public static final String NAME = "If";

        final Body thenBody, elseBody;

        // @@@ make or fake elseBody as "else_branch" attribute and thenBody as "then_branch" attribute
        public enum Attribute implements OnnxOp.OnnxAttribute.None { }

        public enum TypeConstraint implements OnnxOp.OnnxTypeConstraint {
            V(new OnnxType.TypeVariable("V", List.of(OnnxType.tensor(OnnxType.uint8()), OnnxType.tensor(OnnxType.uint16()), OnnxType.tensor(OnnxType.uint32()), OnnxType.tensor(OnnxType.uint64()), OnnxType.tensor(OnnxType.int8()), OnnxType.tensor(OnnxType.int16()), OnnxType.tensor(OnnxType.int32()), OnnxType.tensor(OnnxType.int64()), OnnxType.tensor(OnnxType.bfloat16()), OnnxType.tensor(OnnxType.float16()), OnnxType.tensor(OnnxType.float32()), OnnxType.tensor(OnnxType.float64()), OnnxType.tensor(OnnxType.bool())))),
            B(new OnnxType.TypeVariable("B", List.of(OnnxType.tensor(OnnxType.bool())))),
            ;

            final OnnxType.TypeVariable typeVariable;

            TypeConstraint(OnnxType.TypeVariable typeVariable) {
                assert typeVariable.name().equals(name());
                this.typeVariable = typeVariable;
            }

            @Override
            public OnnxType.TypeVariable typeVariable() {
                return typeVariable;
            }
        }

        public enum InputParameter implements OnnxOp.OnnxParameter {
            cond(TypeConstraint.B.typeVariable(), OnnxOp.OnnxParameter.Quantifier.REQUIRED),
            ;

            final OnnxType type;
            final OnnxOp.OnnxParameter.Quantifier quantifier;

            InputParameter(OnnxType type, OnnxOp.OnnxParameter.Quantifier quantifier) {
                this.type = type;
                this.quantifier = quantifier;
            }

            @Override
            public OnnxType type() {
                return type;
            }

            @Override
            public OnnxOp.OnnxParameter.Quantifier quantifier() {
                return quantifier;
            }
        }

        public enum OutputParameter implements OnnxOp.OnnxParameter {
            output(TypeConstraint.V.typeVariable(), OnnxOp.OnnxParameter.Quantifier.VARIADIC),
            ;

            final OnnxType type;
            final OnnxOp.OnnxParameter.Quantifier quantifier;

            OutputParameter(OnnxType type, OnnxOp.OnnxParameter.Quantifier quantifier) {
                this.type = type;
                this.quantifier = quantifier;
            }

            @Override
            public OnnxType type() {
                return type;
            }

            @Override
            public OnnxOp.OnnxParameter.Quantifier quantifier() {
                return quantifier;
            }
        }

        public static final OnnxOp.OnnxSchema SCHEMA = new OnnxSchemaRecord(
                NAME,
                List.of(Attribute.values()),
                List.of(TypeConstraint.values()),
                List.of(InputParameter.values()),
                List.of(OutputParameter.values())
        );

        public If(ExternalizableOp.ExternalizedOp def) {
            super(SCHEMA, def);

            this.thenBody = def.bodyDefinitions().get(0).build(this);
            this.elseBody = def.bodyDefinitions().get(1).build(this);
        }

        If(If that, CopyContext cc, OpTransformer ot) {
            super(that, cc);

            this.thenBody = that.thenBody.transform(cc, ot).build(this);
            this.elseBody = that.elseBody.transform(cc, ot).build(this);
        }

        @Override
        public If transform(CopyContext cc, OpTransformer ot) {
            return new If(this, cc, ot);
        }

        If(TypeElement resultType, Value cond, Body.Builder thenBranch, Body.Builder elseBranch) {
            super(SCHEMA, resultType, Set.of(), List.of(cond), List.of());

            this.thenBody = thenBranch.build(this);
            this.elseBody = elseBranch.build(this);
        }

        @Override
        public List<Body> bodies() {
            return List.of(thenBody, elseBody);
        }

        @Override
        public SequencedSet<OnnxOp.OnnxParameter> onnxOutputs() {
            return onnxOutputs(SCHEMA);
        }

        @Override
        public SequencedMap<OnnxOp.OnnxParameter, Object> onnxInputs() {
            return onnxInputs(SCHEMA, List.of(cond()));
        }

        public Value cond() {
            return operands().get(0);
        }

        public Body elseBranch() {
            return elseBody;
        }

        public Body thenBranch() {
            return thenBody;
        }
    }

    public static If If(TypeElement resultType, Value cond, Body.Builder thenBody, Body.Builder elseBody) {
        return new If(resultType, cond, thenBody, elseBody);
    }

    @OpFactory.OpDeclaration(Loop.NAME)
    public static final class Loop extends OnnxOp implements Op.Loop {
        public static final String NAME = "Loop";

        final Body body;

        // @@@ make or fake body
        public enum Attribute implements OnnxOp.OnnxAttribute.None { }

        public enum TypeConstraint implements OnnxOp.OnnxTypeConstraint {
            V(new OnnxType.TypeVariable("V", List.of(OnnxType.tensor(OnnxType.uint8()), OnnxType.tensor(OnnxType.uint16()), OnnxType.tensor(OnnxType.uint32()), OnnxType.tensor(OnnxType.uint64()), OnnxType.tensor(OnnxType.int8()), OnnxType.tensor(OnnxType.int16()), OnnxType.tensor(OnnxType.int32()), OnnxType.tensor(OnnxType.int64()), OnnxType.tensor(OnnxType.bfloat16()), OnnxType.tensor(OnnxType.float16()), OnnxType.tensor(OnnxType.float32()), OnnxType.tensor(OnnxType.float64()), OnnxType.tensor(OnnxType.bool())))),
            I(new OnnxType.TypeVariable("I", List.of(OnnxType.tensor(OnnxType.int64())))),
            B(new OnnxType.TypeVariable("B", List.of(OnnxType.tensor(OnnxType.bool())))),
            ;

            final OnnxType.TypeVariable typeVariable;

            TypeConstraint(OnnxType.TypeVariable typeVariable) {
                assert typeVariable.name().equals(name());
                this.typeVariable = typeVariable;
            }

            @Override
            public OnnxType.TypeVariable typeVariable() {
                return typeVariable;
            }
        }

        public enum InputParameter implements OnnxOp.OnnxParameter {
            // @@@ Onnx spec declares the input parameters as optional, however it is causing problems
            M(TypeConstraint.I.typeVariable(), OnnxOp.OnnxParameter.Quantifier.REQUIRED),
            cond(TypeConstraint.B.typeVariable(), OnnxOp.OnnxParameter.Quantifier.REQUIRED),
            v_initial(TypeConstraint.V.typeVariable(), OnnxOp.OnnxParameter.Quantifier.VARIADIC),
            ;

            final OnnxType type;
            final OnnxOp.OnnxParameter.Quantifier quantifier;

            InputParameter(OnnxType type, OnnxOp.OnnxParameter.Quantifier quantifier) {
                this.type = type;
                this.quantifier = quantifier;
            }

            @Override
            public OnnxType type() {
                return type;
            }

            @Override
            public OnnxOp.OnnxParameter.Quantifier quantifier() {
                return quantifier;
            }
        }

        public enum OutputParameter implements OnnxOp.OnnxParameter {
            v_final_and_scan_outputs(TypeConstraint.V.typeVariable(), OnnxOp.OnnxParameter.Quantifier.VARIADIC),
            ;

            final OnnxType type;
            final OnnxOp.OnnxParameter.Quantifier quantifier;

            OutputParameter(OnnxType type, OnnxOp.OnnxParameter.Quantifier quantifier) {
                this.type = type;
                this.quantifier = quantifier;
            }

            @Override
            public OnnxType type() {
                return type;
            }

            @Override
            public OnnxOp.OnnxParameter.Quantifier quantifier() {
                return quantifier;
            }
        }

        public static final OnnxOp.OnnxSchema SCHEMA = new OnnxSchemaRecord(
                NAME,
                List.of(Attribute.values()),
                List.of(TypeConstraint.values()),
                List.of(InputParameter.values()),
                List.of(OutputParameter.values())
        );

        public Loop(ExternalizableOp.ExternalizedOp def) {
            super(SCHEMA, def);

            this.body = def.bodyDefinitions().get(0).build(this);
        }

        Loop(ExplicitOnnxOps.Loop that, CopyContext cc, OpTransformer ot) {
            super(that, cc);

            this.body = that.body.transform(cc, ot).build(this);
        }

        @Override
        public ExplicitOnnxOps.Loop transform(CopyContext cc, OpTransformer ot) {
            return new ExplicitOnnxOps.Loop(this, cc, ot);
        }

        Loop(TypeElement resultType, Value m, Value cond, Object v_initial, Body.Builder body) {
            super(SCHEMA, resultType, Set.of(), List.of(m, cond, v_initial), List.of());

            this.body = body.build(this);
        }

        @Override
        public List<Body> bodies() {
            return List.of(body);
        }

        @Override
        public SequencedSet<OnnxOp.OnnxParameter> onnxOutputs() {
            return onnxOutputs(SCHEMA);
        }

        @Override
        public SequencedMap<OnnxOp.OnnxParameter, Object> onnxInputs() {
            return onnxInputs(SCHEMA, List.of(cond()));
        }

        public Value max() {
            return operands().get(0);
        }

        public Value cond() {
            return operands().get(1);
        }

        public List<Value> v_initial() {
            return operands().subList(2, operands().size());
        }

        @Override
        public Body loopBody() {
            return body;
        }
    }

    public static Loop Loop(TypeElement resultType, Value m, Value cond, Object v_initial, Body.Builder body) {
        return new Loop(resultType, m, cond, v_initial, body);
    }
}

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

    @OpFactory.OpDeclaration(If.NAME)
    public static final class If extends OnnxOp implements Nested {
        public static final String NAME = "If";

        final Body elseBody, thenBody;

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

            this.elseBody = def.bodyDefinitions().get(0).build(this);
            this.thenBody = def.bodyDefinitions().get(1).build(this);
        }

        If(If that, CopyContext cc, OpTransformer ot) {
            super(that, cc);

            this.elseBody = that.elseBody.transform(cc, ot).build(this);
            this.thenBody = that.thenBody.transform(cc, ot).build(this);
        }

        @Override
        public If transform(CopyContext cc, OpTransformer ot) {
            return new If(this, cc, ot);
        }

        If(TypeElement resultType, Value cond, Body.Builder elseBranch, Body.Builder thenBranch) {
            super(SCHEMA, resultType, Set.of(), List.of(cond), List.of());

            this.elseBody = elseBranch.build(this);
            this.thenBody = thenBranch.build(this);
        }

        @Override
        public List<Body> bodies() {
            return List.of(elseBody, thenBody);
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

    public static If If(TypeElement resultType, Value cond, Body.Builder elseBody, Body.Builder thenBody) {
        return new If(resultType, cond, elseBody, thenBody);
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

        Loop(TypeElement resultType, Value m, Value cond, List<Value> v_initial, Body.Builder body) {
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

        public Optional<Value> m() {
            int i = optionalInputArguments.indexOf(InputParameter.M);
            return i != -1 ? Optional.of(operands().get(1 + i)) : Optional.empty();
        }

        public Optional<Value> cond() {
            int i = optionalInputArguments.indexOf(InputParameter.cond);
            return i != -1 ? Optional.of(operands().get(1 + i)) : Optional.empty();
        }

        public List<Value> v_initial() {
            return operands().subList(1, operands().size());
        }

        @Override
        public Body loopBody() {
            return body;
        }
    }

    public static Loop Loop(TypeElement resultType, Value m, Value cond, List<Value> v_initial, Body.Builder body) {
        return new Loop(resultType, m, cond, v_initial, body);
    }
}

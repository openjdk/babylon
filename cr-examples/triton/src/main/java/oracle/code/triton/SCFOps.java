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

package oracle.code.triton;

import java.lang.reflect.code.*;
import java.lang.reflect.code.descriptor.MethodTypeDesc;
import java.lang.reflect.code.descriptor.TypeDesc;
import java.lang.reflect.code.op.*;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;

public class SCFOps {

    @OpDeclaration(ForOp.NAME)
    public static final class ForOp extends OpWithDefinition implements Op.Loop {

        public static class Builder {
            final Body.Builder ancestorBody;
            final List<Value> range;
            final MethodTypeDesc loopDescriptor;

            Builder(Body.Builder ancestorBody, List<Value> range, MethodTypeDesc loopDescriptor) {
                this.ancestorBody = ancestorBody;
                this.range = range;
                this.loopDescriptor = loopDescriptor;
            }

            public ForOp body(Consumer<Block.Builder> c) {
                Body.Builder body = Body.Builder.of(ancestorBody, loopDescriptor);
                c.accept(body.entryBlock());
                return new ForOp(range, body);
            }

            public ForOp body(CopyContext cc, Consumer<Block.Builder> c) {
                Body.Builder body = Body.Builder.of(ancestorBody, loopDescriptor, cc);
                c.accept(body.entryBlock());
                return new ForOp(range, body);
            }
        }

        public static final String NAME = "scf.for";

        final Body body;

        public ForOp(OpDefinition def) {
            super(def);

            this.body = def.bodyDefinitions().get(0).build(this);
        }

        ForOp(ForOp that, CopyContext cc, OpTransformer ot) {
            super(that, cc);

            this.body = that.body.transform(cc, ot).build(this);
        }

        @Override
        public ForOp transform(CopyContext cc, OpTransformer ot) {
            return new ForOp(this, cc, ot);
        }

        ForOp(List<Value> range, Body.Builder bodyBuilder) {
            super(NAME, range);

            this.body = bodyBuilder.build(this);
        }

        @Override
        public TypeDesc resultType() {
            return body.yieldType();
        }

        @Override
        public List<Body> bodies() {
            return List.of(body);
        }

        @Override
        public Body loopBody() {
            return body;
        }
    }

    @OpDeclaration(YieldOp.NAME)
    public static class YieldOp extends OpWithDefinition implements Op.Terminating {
        public static final String NAME = "scf.yield";

        public YieldOp(OpDefinition def) {
            super(def);
        }

        YieldOp(YieldOp that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public YieldOp transform(CopyContext cc, OpTransformer ot) {
            return new YieldOp(this, cc);
        }

        YieldOp(List<Value> values) {
            super(NAME, values);
        }

        @Override
        public TypeDesc resultType() {
            return TypeDesc.VOID;
        }

        static TypeDesc yieldType(List<Value> values) {
            if (values.size() == 1) {
                return values.get(0).type();
            } else {
                return CoreOps.Tuple.typeFromValues(values);
            }
        }
    }

    static public ForOp.Builder for_(Body.Builder ancestorBody,
                                     Value start, Value end, Value step,
                                     List<Value> iterValues) {
        TypeDesc yieldType = (iterValues.size() == 1)
                ? iterValues.get(0).type()
                : CoreOps.Tuple.typeFromValues(iterValues);

        List<TypeDesc> bodyParameterTypes = new ArrayList<>();
        bodyParameterTypes.add(start.type());
        bodyParameterTypes.addAll(iterValues.stream().map(Value::type).toList());
        MethodTypeDesc bodyType = MethodTypeDesc.methodType(yieldType, bodyParameterTypes);

        List<Value> operands = new ArrayList<>();
        operands.addAll(List.of(start, end, step));
        operands.addAll(iterValues);
        return new ForOp.Builder(ancestorBody, operands, bodyType);
    }


    public static final OpFactory FACTORY = OpFactory.OP_FACTORY.get(SCFOps.class);

    static public YieldOp yield_(Value... values) {
        return yield_(List.of(values));
    }

    static public YieldOp yield_(List<Value> values) {
        return new YieldOp(values);
    }
}

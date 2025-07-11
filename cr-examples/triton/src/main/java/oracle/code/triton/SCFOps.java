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

import jdk.incubator.code.*;
import jdk.incubator.code.extern.ExternalizedOp;
import jdk.incubator.code.extern.OpFactory;
import jdk.incubator.code.dialect.core.CoreType;
import jdk.incubator.code.dialect.core.FunctionType;
import jdk.incubator.code.dialect.java.JavaType;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;

public class SCFOps {

    @OpFactoryHelper.OpDeclaration(ForOp.NAME)
    public static final class ForOp extends Op implements Op.Loop {

        public static class Builder {
            final Body.Builder ancestorBody;
            final List<Value> range;
            final FunctionType loopType;

            Builder(Body.Builder ancestorBody, List<Value> range, FunctionType loopType) {
                this.ancestorBody = ancestorBody;
                this.range = range;
                this.loopType = loopType;
            }

            public ForOp body(Consumer<Block.Builder> c) {
                Body.Builder body = Body.Builder.of(ancestorBody, loopType);
                c.accept(body.entryBlock());
                return new ForOp(range, body);
            }

            public ForOp body(CopyContext cc, Consumer<Block.Builder> c) {
                Body.Builder body = Body.Builder.of(ancestorBody, loopType, cc);
                c.accept(body.entryBlock());
                return new ForOp(range, body);
            }
        }

        public static final String NAME = "scf.for";

        final Body body;

        public ForOp(ExternalizedOp def) {
            super(def.name(), def.operands());;

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
        public TypeElement resultType() {
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

    @OpFactoryHelper.OpDeclaration(YieldOp.NAME)
    public static class YieldOp extends Op implements Op.Terminating {
        public static final String NAME = "scf.yield";

        public YieldOp(ExternalizedOp def) {
            super(def.name(), def.operands());
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
        public TypeElement resultType() {
            return JavaType.VOID;
        }

        static TypeElement yieldType(List<Value> values) {
            if (values.size() == 1) {
                return values.get(0).type();
            } else {
                return CoreType.tupleTypeFromValues(values);
            }
        }
    }

    static public ForOp.Builder for_(Body.Builder ancestorBody,
                                     Value start, Value end, Value step,
                                     List<Value> iterValues) {
        TypeElement yieldType = (iterValues.size() == 1)
                ? iterValues.get(0).type()
                : CoreType.tupleTypeFromValues(iterValues);

        List<TypeElement> bodyParameterTypes = new ArrayList<>();
        bodyParameterTypes.add(start.type());
        bodyParameterTypes.addAll(iterValues.stream().map(Value::type).toList());
        FunctionType bodyType = CoreType.functionType(yieldType, bodyParameterTypes);

        List<Value> operands = new ArrayList<>();
        operands.addAll(List.of(start, end, step));
        operands.addAll(iterValues);
        return new ForOp.Builder(ancestorBody, operands, bodyType);
    }


    public static final OpFactory OP_FACTORY = OpFactoryHelper.OP_FACTORY.get(SCFOps.class);

    static public YieldOp yield_(Value... values) {
        return yield_(List.of(values));
    }

    static public YieldOp yield_(List<Value> values) {
        return new YieldOp(values);
    }
}

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

package java.lang.reflect.code.op;

import java.io.IOException;
import java.io.StringWriter;
import java.io.Writer;
import java.lang.reflect.code.*;
import java.lang.reflect.code.type.FunctionType;
import java.util.*;
import java.util.function.Consumer;

import static java.lang.reflect.code.op.OpFactory.*;

public final class AnfDialect {

    private AnfDialect() {
    }

    @OpDeclaration(AnfLetOp.NAME)
    public static final class AnfLetOp extends ExternalizableOp implements Op.Terminating, Op.Nested {
        public static final String NAME = "anf.let";

        public static class Builder {
            final Body.Builder ancestorBody;
            final TypeElement yieldType;

            Builder(Body.Builder ancestorBody, TypeElement yieldType) {
                this.ancestorBody = ancestorBody;
                this.yieldType = yieldType;
            }

            public AnfLetOp body(Consumer<Block.Builder> c) {
                Body.Builder body = Body.Builder.of(ancestorBody, FunctionType.functionType(yieldType));
                c.accept(body.entryBlock());
                return new AnfLetOp(body);
            }
        }

        // Terminating operation is the in expression that yields the result for this operation
        // Any operation result in the entry block used by the terminating operation
        // or a descendant operation is a let binding
        final Body bindings;

        public AnfLetOp(ExternalizedOp def) {
            super(def);

            this.bindings = def.bodyDefinitions().get(0).build(this);
        }

        public AnfLetOp(AnfLetOp that, CopyContext cc, OpTransformer ot) {
            super(that, cc);

            this.bindings = that.bindings.transform(cc, ot).build(this);
        }

        @Override
        public Op transform(CopyContext cc, OpTransformer ot) {
            return new AnfLetOp(this, cc, ot);
        }

        public AnfLetOp(Body.Builder bodyBuilder) {
            super(NAME, List.of());

            this.bindings = bodyBuilder.build(this);
        }

        @Override
        public List<Body> bodies() {
            return List.of(bindings);
        }

        @Override
        public TypeElement resultType() {
            return this.bindings.yieldType();
        }
    }


    @OpDeclaration(AnfLetRecOp.NAME)
    public static final class AnfLetRecOp extends ExternalizableOp implements Op.Terminating, Op.Nested {
        public static final String NAME = "anf.letrec";

        public static class Builder {
            final Body.Builder ancestorBody;
            final TypeElement yieldType;

            Builder(Body.Builder ancestorBody, TypeElement yieldType) {
                this.ancestorBody = ancestorBody;
                this.yieldType = yieldType;
            }

            public AnfLetRecOp body(Consumer<Block.Builder> c) {
                Body.Builder body = Body.Builder.of(ancestorBody, FunctionType.functionType(yieldType));
                c.accept(body.entryBlock());
                return new AnfLetRecOp(body);
            }
        }

        // Terminating operation is the in expression that yields the result for this operation
        // Any operation result in the entry block used by the terminating operation
        // or a descendant operation is a letrec binding
        final Body bindings;

        public AnfLetRecOp(ExternalizedOp def) {
            super(def);

            this.bindings = def.bodyDefinitions().get(0).build(this);
        }

        public AnfLetRecOp(AnfLetRecOp that, CopyContext cc, OpTransformer ot) {
            super(that, cc);

            this.bindings = that.bindings.transform(cc, ot).build(this);
        }

        @Override
        public Op transform(CopyContext cc, OpTransformer ot) {
            return new AnfLetRecOp(this, cc, ot);
        }

        public AnfLetRecOp(Body.Builder bodyBuilder) {
            super(AnfLetRecOp.NAME, List.of());

            this.bindings = bodyBuilder.build(this);
        }

        @Override
        public List<Body> bodies() {
            return List.of(bindings);
        }

        @Override
        public TypeElement resultType() {
            return this.bindings.yieldType();
        }

        public List<CoreOp.FuncOp> funcOps() {
            List<Op> ops = bindings.entryBlock().ops();
            return ops.subList(0, ops.size() - 1).stream()
                    .<CoreOp.FuncOp>mapMulti((op, objectConsumer) -> {
                        if (op instanceof CoreOp.FuncOp fop) {
                            objectConsumer.accept(fop);
                        }
                    }).toList();
        }
    }

    @OpDeclaration(AnfIfOp.NAME)
    public static final class AnfIfOp extends ExternalizableOp implements Op.Terminating, Op.Nested {
        public static final String NAME = "anf.if";

        public static class ThenBuilder {
            final Body.Builder ancestorBody;
            final TypeElement yieldType;
            final Value test;

            ThenBuilder(Body.Builder ancestorBody, TypeElement yieldType, Value test) {
                this.ancestorBody = ancestorBody;
                this.yieldType = yieldType;
                this.test = test;
            }

            public ElseBuilder if_(Consumer<Block.Builder> c) {
                Body.Builder then_ = Body.Builder.of(ancestorBody,
                        FunctionType.functionType(yieldType));
                c.accept(then_.entryBlock());

                return new ElseBuilder(this, then_);
            }
        }

        public static class ElseBuilder {
            final ThenBuilder thenBuilder;
            final Body.Builder then_;

            public ElseBuilder(ThenBuilder thenBuilder, Body.Builder then_) {
                this.thenBuilder = thenBuilder;
                this.then_ = then_;
            }

            public AnfIfOp else_(Consumer<Block.Builder> c) {
                Body.Builder else_ = Body.Builder.of(thenBuilder.ancestorBody,
                        FunctionType.functionType(thenBuilder.yieldType));
                c.accept(else_.entryBlock());

                return new AnfIfOp(thenBuilder.test, then_, else_);
            }
        }

        final Body then_, else_;

        public AnfIfOp(ExternalizableOp.ExternalizedOp def) {
            super(def);

            this.then_ = def.bodyDefinitions().get(0).build(this);
            this.else_ = def.bodyDefinitions().get(1).build(this);
        }

        public AnfIfOp(AnfIfOp that, CopyContext cc, OpTransformer ot) {
            super(that, cc);

            this.then_ = that.then_.transform(cc, ot).build(this);
            this.else_ = that.else_.transform(cc, ot).build(this);
        }

        @Override
        public Op transform(CopyContext cc, OpTransformer ot) {
            return new AnfIfOp(this, cc, ot);
        }

        AnfIfOp(Value test, Body.Builder thenBodyBuilder, Body.Builder elseBodyBuilder) {
            super(NAME, List.of(test));

            this.then_ = thenBodyBuilder.build(this);
            this.else_ = elseBodyBuilder.build(this);
        }

        public Value getTest() {
            return this.operands().get(0);
        }

        public Body _then() {
            return then_;
        }

        public Body _else() {
            return else_;
        }

        @Override
        public List<Body> bodies() {
            return List.of(then_, else_);
        }

        @Override
        public TypeElement resultType() {
            return this.then_.yieldType();
        }
    }

    @OpFactory.OpDeclaration(AnfFuncOp.NAME)
    public static final class AnfFuncOp extends ExternalizableOp implements Op.Nested {

        public static class Builder {
            final Body.Builder ancestorBody;
            final String funcName;
            final FunctionType funcType;

            Builder(Body.Builder ancestorBody, String funcName, FunctionType funcType) {
                this.ancestorBody = ancestorBody;
                this.funcName = funcName;
                this.funcType = funcType;
            }

            public AnfFuncOp body(Consumer<Block.Builder> c) {
                Body.Builder body = Body.Builder.of(ancestorBody, funcType);
                c.accept(body.entryBlock());
                return new AnfFuncOp(funcName, body);
            }
        }

        public static final String NAME = "anf.func";
        public static final String ATTRIBUTE_FUNC_NAME = NAME + ".name";

        final String funcName;
        final Body body;

        public static AnfFuncOp create(ExternalizedOp def) {
            if (!def.operands().isEmpty()) {
                throw new IllegalStateException("Bad op " + def.name());
            }

            String funcName = def.extractAttributeValue(ATTRIBUTE_FUNC_NAME, true,
                    v -> switch (v) {
                        case String s -> s;
                        case null, default -> throw new UnsupportedOperationException("Unsupported func name value:" + v);
                    });
            return new AnfFuncOp(def, funcName);
        }

        AnfFuncOp(ExternalizedOp def, String funcName) {
            super(def);

            this.funcName = funcName;
            this.body = def.bodyDefinitions().get(0).build(this);
        }

        AnfFuncOp(AnfFuncOp that, CopyContext cc, OpTransformer oa) {
            this(that, that.funcName, cc, oa);
        }

        AnfFuncOp(AnfFuncOp that, String funcName, CopyContext cc, OpTransformer ot) {
            super(that, cc);

            this.funcName = funcName;
            this.body = that.body.transform(cc, ot).build(this);
        }

        @Override
        public AnfFuncOp transform(CopyContext cc, OpTransformer ot) {
            return new AnfFuncOp(this, cc, ot);
        }

        AnfFuncOp(String funcName, Body.Builder bodyBuilder) {
            super(NAME,
                    List.of());

            this.funcName = funcName;
            this.body = bodyBuilder.build(this);
        }

        @Override
        public List<Body> bodies() {
            return List.of(body);
        }

        @Override
        public Map<String, Object> attributes() {
            HashMap<String, Object> m = new HashMap<>(super.attributes());
            m.put("", funcName);
            return Collections.unmodifiableMap(m);
        }

        public FunctionType invokableType() {
            return body.bodyType();
        }

        public String funcName() {
            return funcName;
        }

        public Body body() {
            return body;
        }

        @Override
        public TypeElement resultType() {
            return invokableType();
        }
    }

    @OpFactory.OpDeclaration(AnfApply.NAME)
    public static final class AnfApply extends ExternalizableOp implements Op.Terminating {
        public static final String NAME = "anf.apply";

        public AnfApply(ExternalizedOp def) {
            super(def);
        }

        public AnfApply(AnfApply that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public Op transform(CopyContext cc, OpTransformer ot) {
            return new AnfApply(this, cc);
        }

        @Override
        public TypeElement resultType() {
            FunctionType ft = (FunctionType) operands().get(0).type();
            return ft.returnType();
        }

        public AnfApply(List<Value> arguments) {
            super(AnfApply.NAME, arguments);

            // First argument is func value
            // Subsequent arguments are func arguments
        }


        public List<Value> args() {
            return operands().subList(1, this.operands().size());
        }
    }

    @OpFactory.OpDeclaration(AnfApplyStub.NAME)
    public static final class AnfApplyStub extends ExternalizableOp implements Op.Terminating {
        public static final String NAME = "anf.apply.stub";
        public static final String ATTRIBUTE_RESULT_TYPE = ".resultType";
        public static final String ATTRIBUTE_CALLSITE_NAME = ".callsiteName";

        public final String callSiteName;
        public final TypeElement resultType;

        public static AnfApplyStub create(ExternalizedOp def) {
            if (!def.operands().isEmpty()) {
                throw new IllegalStateException("Bad op " + def.name());
            }

            String callsiteName = def.extractAttributeValue(ATTRIBUTE_CALLSITE_NAME, true,
                    v -> switch (v) {
                        case String s -> s;
                        case null, default -> throw new UnsupportedOperationException("Unsupported func name value:" + v);
                    });
            return new AnfApplyStub(def,callsiteName,def.resultType());
        }

        public AnfApplyStub(ExternalizedOp def, String name, TypeElement resultType) {
            super(def);
            this.resultType = resultType;
            this.callSiteName = name;
        }

        public AnfApplyStub(AnfApplyStub that, CopyContext cc) {
            super(that, cc);
            this.callSiteName = that.callSiteName;
            this.resultType = that.resultType;
        }

        @Override
        public Map<String, Object> attributes() {
            HashMap<String, Object> m = new HashMap<>(super.attributes());
            m.put("", callSiteName);
            return Collections.unmodifiableMap(m);
        }

        @Override
        public Op transform(CopyContext cc, OpTransformer ot) {
            return new AnfApplyStub(this, cc);
        }

        public AnfApplyStub(String callSiteName, List<Value> arguments, TypeElement resultType) {
            super(AnfApplyStub.NAME, arguments);
            this.resultType = resultType;
            this.callSiteName = callSiteName;

            // First argument is func value
            // Subsequent arguments are func arguments
        }

        @Override
        public TypeElement resultType() {
            return this.resultType;
        }

        public List<Value> args() {
            return operands().subList(1, this.operands().size());
        }
    }

    public static AnfLetRecOp.Builder letrec(Body.Builder ancestorBody, TypeElement yieldType) {
        return new AnfLetRecOp.Builder(ancestorBody, yieldType);
    }

    public static AnfLetRecOp letrec(Body.Builder body) {
        return new AnfLetRecOp(body);
    }

    public static AnfLetOp.Builder let(Body.Builder ancestorBody, TypeElement yieldType) {
        return new AnfLetOp.Builder(ancestorBody, yieldType);
    }

    public static AnfLetOp let(Body.Builder body) {
        return new AnfLetOp(body);
    }

    public static AnfIfOp.ThenBuilder if_(Body.Builder ancestorBody, TypeElement yieldType, Value test) {
        return new AnfIfOp.ThenBuilder(ancestorBody, yieldType, test);
    }

    public static AnfIfOp if_(Body.Builder then_, Body.Builder else_, Value test) {
        return new AnfIfOp(test, then_, else_);
    }

    public static AnfFuncOp.Builder func(Body.Builder ancestorBody, String funcName, FunctionType funcType) {
        List<TypeElement> params = new ArrayList<>();
        params.add(funcType);
        params.addAll(funcType.parameterTypes());
        return new AnfFuncOp.Builder(ancestorBody, funcName, FunctionType.functionType(funcType.returnType(), params));
    }

    public static AnfFuncOp func(String funcName, Body.Builder body) {
        return new AnfFuncOp(funcName, body);
    }

    public static AnfApply apply(List<Value> arguments) {
        return new AnfApply(arguments);
    }
    //public static AnfApplyStub applyStub(String name, List<Value> arguments, TypeElement type) { return new AnfApplyStub(name, arguments, type);}
}

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

import java.lang.constant.ClassDesc;
import java.lang.reflect.code.*;
import java.lang.reflect.code.type.FieldRef;
import java.lang.reflect.code.type.MethodRef;
import java.lang.reflect.code.type.*;
import java.util.*;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.function.Predicate;

/**
 * The top-level operation class for the set of enclosed core operations.
 * <p>
 * A code model, produced by the Java compiler from Java program source, may consist of extended operations and core
 * operations. Such a model represents the same Java program and preserves the program meaning as defined by the
 * Java Language Specification
 */
public sealed abstract class CoreOp extends ExternalizableOp {
    /**
     * An operation that models a Java expression
     */
    sealed interface JavaExpression permits
            ArithmeticOperation,
            ArrayAccessOp.ArrayLoadOp,
            ArrayAccessOp.ArrayStoreOp,
            ArrayLengthOp,
            CastOp,
            ConvOp,
            ClosureOp,
            ConcatOp,
            ConstantOp,
            FieldAccessOp.FieldLoadOp,
            FieldAccessOp.FieldStoreOp,
            InstanceOfOp,
            InvokeOp,
            LambdaOp,
            NewOp,
            TestOperation,
            VarAccessOp.VarLoadOp,
            VarAccessOp.VarStoreOp,
            ExtendedOp.JavaConditionalExpressionOp,
            ExtendedOp.JavaConditionalOp,
            ExtendedOp.JavaSwitchExpressionOp {
    }

    /**
     * An operation that models a Java statement
     */
    sealed interface JavaStatement permits
            ArrayAccessOp.ArrayStoreOp,
            AssertOp,
            FieldAccessOp.FieldStoreOp,
            InvokeOp,
            NewOp,
            ReturnOp,
            ThrowOp,
            VarAccessOp.VarStoreOp,
            VarOp,
            ExtendedOp.JavaBlockOp,
            ExtendedOp.JavaDoWhileOp,
            ExtendedOp.JavaEnhancedForOp,
            ExtendedOp.JavaForOp,
            ExtendedOp.JavaIfOp,
            ExtendedOp.JavaLabelOp,
            ExtendedOp.JavaLabeledOp,
            ExtendedOp.JavaTryOp,
            ExtendedOp.JavaWhileOp,
            ExtendedOp.JavaYieldOp {
    }

    // Split string to ensure the name does not get rewritten
    // when the script copies this source to the jdk.compiler module
    static final String PACKAGE_NAME = "java.lang" + ".reflect.code";

    static final String CoreOp_CLASS_NAME = PACKAGE_NAME + "." + CoreOp.class.getSimpleName();

    protected CoreOp(Op that, CopyContext cc) {
        super(that, cc);
    }

    protected CoreOp(String name, List<? extends Value> operands) {
        super(name, operands);
    }

    protected CoreOp(ExternalizedOp def) {
        super(def);
    }

    /**
     * The function operation, that can model a Java method declaration.
     */
    @OpFactory.OpDeclaration(FuncOp.NAME)
    public static final class FuncOp extends CoreOp
            implements Op.Invokable, Op.Isolated, Op.Lowerable {

        public static class Builder {
            final Body.Builder ancestorBody;
            final String funcName;
            final FunctionType funcType;

            Builder(Body.Builder ancestorBody, String funcName, FunctionType funcType) {
                this.ancestorBody = ancestorBody;
                this.funcName = funcName;
                this.funcType = funcType;
            }

            public FuncOp body(Consumer<Block.Builder> c) {
                Body.Builder body = Body.Builder.of(ancestorBody, funcType);
                c.accept(body.entryBlock());
                return new FuncOp(funcName, body);
            }
        }

        public static final String NAME = "func";
        public static final String ATTRIBUTE_FUNC_NAME = NAME + ".name";

        final String funcName;
        final Body body;

        public static FuncOp create(ExternalizedOp def) {
            if (!def.operands().isEmpty()) {
                throw new IllegalStateException("Bad op " + def.name());
            }

            String funcName = def.extractAttributeValue(ATTRIBUTE_FUNC_NAME, true,
                    v -> switch (v) {
                        case String s -> s;
                        default -> throw new UnsupportedOperationException("Unsupported func name value:" + v);
                    });
            return new FuncOp(def, funcName);
        }

        FuncOp(ExternalizedOp def, String funcName) {
            super(def);

            this.funcName = funcName;
            this.body = def.bodyDefinitions().get(0).build(this);
        }

        FuncOp(FuncOp that, CopyContext cc, OpTransformer oa) {
            this(that, that.funcName, cc, oa);
        }

        FuncOp(FuncOp that, String funcName, CopyContext cc, OpTransformer ot) {
            super(that, cc);

            this.funcName = funcName;
            this.body = that.body.transform(cc, ot).build(this);
        }

        @Override
        public FuncOp transform(CopyContext cc, OpTransformer ot) {
            return new FuncOp(this, cc, ot);
        }

        public FuncOp transform(OpTransformer ot) {
            return new FuncOp(this, CopyContext.create(), ot);
        }

        public FuncOp transform(String funcName, OpTransformer ot) {
            return new FuncOp(this, funcName, CopyContext.create(), ot);
        }

        FuncOp(String funcName, Body.Builder bodyBuilder) {
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

        @Override
        public FunctionType invokableType() {
            return body.bodyType();
        }

        public String funcName() {
            return funcName;
        }

        @Override
        public Body body() {
            return body;
        }

        @Override
        public Block.Builder lower(Block.Builder b, OpTransformer _ignore) {
            // Isolate body with respect to ancestor transformations
            b.op(this, OpTransformer.LOWERING_TRANSFORMER);
            return b;
        }

        @Override
        public TypeElement resultType() {
            return JavaType.VOID;
        }
    }

    /**
     * The function call operation, that models a call to a function, by name, declared in the module op that is also an
     * ancestor of this operation.
     */
    // @@@ stack effects equivalent to the call operation as if the function were a Java method?
    @OpFactory.OpDeclaration(FuncCallOp.NAME)
    public static final class FuncCallOp extends CoreOp {
        public static final String NAME = "func.call";
        public static final String ATTRIBUTE_FUNC_NAME = NAME + ".name";

        final String funcName;
        final TypeElement resultType;

        public static FuncCallOp create(ExternalizedOp def) {
            String funcName = def.extractAttributeValue(ATTRIBUTE_FUNC_NAME, true,
                    v -> switch (v) {
                        case String s -> s;
                        default -> throw new UnsupportedOperationException("Unsupported func name value:" + v);
                    });

            return new FuncCallOp(def, funcName);
        }

        FuncCallOp(ExternalizedOp def, String funcName) {
            super(def);

            this.funcName = funcName;
            this.resultType = def.resultType();
        }

        FuncCallOp(FuncCallOp that, CopyContext cc) {
            super(that, cc);

            this.funcName = that.funcName;
            this.resultType = that.resultType;
        }

        @Override
        public FuncCallOp transform(CopyContext cc, OpTransformer ot) {
            return new FuncCallOp(this, cc);
        }

        FuncCallOp(String funcName, TypeElement resultType, List<Value> args) {
            super(NAME, args);

            this.funcName = funcName;
            this.resultType = resultType;
        }

        @Override
        public Map<String, Object> attributes() {
            HashMap<String, Object> m = new HashMap<>(super.attributes());
            m.put("", funcName);
            return Collections.unmodifiableMap(m);
        }

        public String funcName() {
            return funcName;
        }

        @Override
        public TypeElement resultType() {
            return resultType;
        }
    }

    /**
     * The module operation, modeling a collection of functions,
     * and creating a symbol table of function name to function
     */
    @OpFactory.OpDeclaration(ModuleOp.NAME)
    public static final class ModuleOp extends CoreOp
            implements Op.Isolated, Op.Lowerable {

        public static final String NAME = "module";

        final SequencedMap<String, FuncOp> table;
        final Body body;

        public static ModuleOp create(ExternalizedOp def) {
            if (!def.operands().isEmpty()) {
                throw new IllegalStateException("Bad op " + def.name());
            }

            return new ModuleOp(def);
        }

        ModuleOp(ExternalizedOp def) {
            super(def);

            this.body = def.bodyDefinitions().get(0).build(this);
            this.table = createTable(body);
        }

        ModuleOp(ModuleOp that, CopyContext cc, OpTransformer ot) {
            super(that, cc);

            this.body = that.body.transform(cc, ot).build(this);
            this.table = createTable(body);
        }

        static SequencedMap<String, FuncOp> createTable(Body body) {
            SequencedMap<String, FuncOp> table = new LinkedHashMap<>();
            for (var op : body.entryBlock().ops()) {
                if (op instanceof FuncOp fop) {
                    table.put(fop.funcName(), fop);
                } else if (!(op instanceof Op.Terminating)) {
                    throw new IllegalArgumentException("Bad operation in module: " + op);
                }
            }
            return Collections.unmodifiableSequencedMap(table);
        }

        @Override
        public ModuleOp transform(CopyContext cc, OpTransformer ot) {
            return new ModuleOp(this, cc, ot);
        }

        public ModuleOp transform(OpTransformer ot) {
            return new ModuleOp(this, CopyContext.create(), ot);
        }

        ModuleOp(List<FuncOp> functions) {
            super(NAME,
                    List.of());

            Body.Builder bodyC = Body.Builder.of(null, FunctionType.VOID);
            Block.Builder entryBlock = bodyC.entryBlock();
            SequencedMap<String, FuncOp> table = new LinkedHashMap<>();
            for (FuncOp f : functions) {
                entryBlock.op(f);
                table.put(f.funcName(), f);
            }
            entryBlock.op(CoreOp.unreachable());
            this.table = Collections.unmodifiableSequencedMap(table);
            this.body = bodyC.build(this);
        }

        @Override
        public List<Body> bodies() {
            return List.of(body);
        }

        public SequencedMap<String, FuncOp> functionTable() {
            return table;
        }

        @Override
        public TypeElement resultType() {
            return JavaType.VOID;
        }

        @Override
        public Block.Builder lower(Block.Builder b, OpTransformer _ignore) {
            b.op(this, OpTransformer.LOWERING_TRANSFORMER);
            return b;
        }
    }

    /**
     * The quoted operation, that models the quoting of an operation.
     */
    @OpFactory.OpDeclaration(QuotedOp.NAME)
    public static final class QuotedOp extends CoreOp
            implements Op.Nested, Op.Lowerable, Op.Pure {
        public static final String NAME = "quoted";

        // Type name must be the same in the java.base and jdk.compiler module
        static final String Quoted_CLASS_NAME = PACKAGE_NAME +
                "." + Quoted.class.getSimpleName();
        public static final JavaType QUOTED_TYPE = JavaType.type(ClassDesc.of(Quoted_CLASS_NAME));

        final Body quotedBody;

        final Op quotedOp;

        public QuotedOp(ExternalizedOp def) {
            super(def);

            this.quotedBody = def.bodyDefinitions().get(0).build(this);

            if (quotedBody.entryBlock().terminatingOp() instanceof YieldOp brk &&
                    brk.yieldValue() instanceof Result quotedOpResult) {
                this.quotedOp = quotedOpResult.op();
            } else {
                throw new IllegalArgumentException();
            }
        }

        QuotedOp(QuotedOp that, CopyContext cc, OpTransformer ot) {
            super(that, cc);

            this.quotedBody = that.quotedBody.transform(cc, ot).build(this);
            this.quotedOp = that.quotedOp;
        }

        @Override
        public QuotedOp transform(CopyContext cc, OpTransformer ot) {
            return new QuotedOp(this, cc, ot);
        }

        QuotedOp(Body.Builder bodyC) {
            super(NAME,
                    List.of());

            this.quotedBody = bodyC.build(this);
            if (quotedBody.blocks().size() > 1) {
                throw new IllegalArgumentException();
            }
            if (!(quotedBody.entryBlock().terminatingOp() instanceof YieldOp yop)) {
                throw new IllegalArgumentException();
            }
            if (!(yop.yieldValue() instanceof Result r)) {
                throw new IllegalArgumentException();
            }
            this.quotedOp = r.op();
        }

        @Override
        public List<Body> bodies() {
            return List.of(quotedBody);
        }

        public Op quotedOp() {
            return quotedOp;
        }

        @Override
        public List<Value> capturedValues() {
            return quotedBody.capturedValues();
        }

        @Override
        public Block.Builder lower(Block.Builder b, OpTransformer _ignore) {
            // Isolate body with respect to ancestor transformations
            // and copy directly without lowering descendant operations
            b.op(this, OpTransformer.COPYING_TRANSFORMER);
            return b;
        }

        @Override
        public TypeElement resultType() {
            return QUOTED_TYPE;
        }
    }

    /**
     * The lambda operation, that can model a Java lambda expression.
     */
    @OpFactory.OpDeclaration(LambdaOp.NAME)
    public static final class LambdaOp extends CoreOp
            implements Op.Invokable, Op.Lowerable, JavaExpression {

        public static class Builder {
            final Body.Builder ancestorBody;
            final FunctionType funcType;
            final TypeElement functionalInterface;

            Builder(Body.Builder ancestorBody, FunctionType funcType, TypeElement functionalInterface) {
                this.ancestorBody = ancestorBody;
                this.funcType = funcType;
                this.functionalInterface = functionalInterface;
            }

            public LambdaOp body(Consumer<Block.Builder> c) {
                Body.Builder body = Body.Builder.of(ancestorBody, funcType);
                c.accept(body.entryBlock());
                return new LambdaOp(functionalInterface, body);
            }
        }

        public static final String NAME = "lambda";

        final TypeElement functionalInterface;
        final Body body;

        public LambdaOp(ExternalizedOp def) {
            super(def);

            this.functionalInterface = def.resultType();
            this.body = def.bodyDefinitions().get(0).build(this);
        }

        LambdaOp(LambdaOp that, CopyContext cc, OpTransformer ot) {
            super(that, cc);

            this.functionalInterface = that.functionalInterface;
            this.body = that.body.transform(cc, ot).build(this);
        }

        @Override
        public LambdaOp transform(CopyContext cc, OpTransformer ot) {
            return new LambdaOp(this, cc, ot);
        }

        LambdaOp(TypeElement functionalInterface, Body.Builder bodyC) {
            super(NAME,
                    List.of());

            this.functionalInterface = functionalInterface;
            this.body = bodyC.build(this);
        }

        @Override
        public List<Body> bodies() {
            return List.of(body);
        }

        @Override
        public FunctionType invokableType() {
            return body.bodyType();
        }

        public TypeElement functionalInterface() {
            return functionalInterface;
        }

        @Override
        public Body body() {
            return body;
        }

        @Override
        public List<Value> capturedValues() {
            return body.capturedValues();
        }

        @Override
        public Block.Builder lower(Block.Builder b, OpTransformer _ignore) {
            // Isolate body with respect to ancestor transformations
            b.op(this, OpTransformer.LOWERING_TRANSFORMER);
            return b;
        }

        @Override
        public TypeElement resultType() {
            return functionalInterface();
        }

        /**
         * Determines if this lambda operation could have originated from a
         * method reference declared in Java source code.
         * <p>
         * Such a lambda operation is one with the following constraints:
         * <ol>
         *     <li>Zero or one captured value (assuming correspondence to the {@code this} variable).
         *     <li>A body with only one (entry) block that contains only variable declaration
         *     operations, variable load operations, invoke operations to box or unbox
         *     primitive values, a single invoke operation to the method that is
         *     referenced, and a return operation.
         *     <li>if the return operation returns a non-void result then that result is,
         *     or uniquely depends on, the result of the referencing invoke operation.
         *     <li>If the lambda operation captures one value then the first operand corresponds
         *     to captured the value, and subsequent operands of the referencing invocation
         *     operation are, or uniquely depend on, the lambda operation's parameters, in order.
         *     Otherwise, the first and subsequent operands of the referencing invocation
         *     operation are, or uniquely depend on, the lambda operation's parameters, in order.
         * </ol>
         * A value, V2, uniquely depends on another value, V1, if the graph of what V2 depends on
         * contains only nodes with single edges terminating in V1, and the graph of what depends on V1
         * is bidirectionally equal to the graph of what V2 depends on.
         *
         * @return the invocation operation to the method referenced by the lambda
         * operation, otherwise empty.
         */
        public Optional<InvokeOp> methodReference() {
            // Single block
            if (body().blocks().size() > 1) {
                return Optional.empty();
            }

            // Zero or one (this) capture
            List<Value> cvs = capturedValues();
            if (cvs.size() > 1) {
                return Optional.empty();
            }

            Map<Value, Value> valueMapping = new HashMap<>();
            CoreOp.InvokeOp methodRefInvokeOp = extractMethodInvoke(valueMapping, body().entryBlock().ops());
            if (methodRefInvokeOp == null) {
                return Optional.empty();
            }

            // Lambda's parameters map in encounter order with the invocation's operands
            List<Value> lambdaParameters = new ArrayList<>();
            if (cvs.size() == 1) {
                lambdaParameters.add(cvs.getFirst());
            }
            lambdaParameters.addAll(parameters());
            List<Value> methodRefOperands = methodRefInvokeOp.operands().stream().map(valueMapping::get).toList();
            if (!lambdaParameters.equals(methodRefOperands)) {
                return Optional.empty();
            }

            return Optional.of(methodRefInvokeOp);
        }

        static CoreOp.InvokeOp extractMethodInvoke(Map<Value, Value> valueMapping, List<Op> ops) {
            CoreOp.InvokeOp methodRefInvokeOp = null;
            for (Op op : ops) {
                switch (op) {
                    case CoreOp.VarOp varOp -> {
                        if (isValueUsedWithOp(varOp.result(), o -> o instanceof CoreOp.VarAccessOp.VarStoreOp)) {
                            return null;
                        }
                    }
                    case CoreOp.VarAccessOp.VarLoadOp varLoadOp -> {
                        Value v = varLoadOp.varOp().operands().getFirst();
                        valueMapping.put(varLoadOp.result(), valueMapping.getOrDefault(v, v));
                    }
                    case CoreOp.InvokeOp iop when isBoxOrUnboxInvocation(iop) -> {
                        Value v = iop.operands().getFirst();
                        valueMapping.put(iop.result(), valueMapping.getOrDefault(v, v));
                    }
                    case CoreOp.InvokeOp iop -> {
                        if (methodRefInvokeOp != null) {
                            return null;
                        }

                        for (Value o : iop.operands()) {
                            valueMapping.put(o, valueMapping.getOrDefault(o, o));
                        }
                        methodRefInvokeOp = iop;
                    }
                    case CoreOp.ReturnOp rop -> {
                        if (methodRefInvokeOp == null) {
                            return null;
                        }
                        Value r = rop.returnValue();
                        if (!(valueMapping.getOrDefault(r, r) instanceof Op.Result invokeResult)) {
                            return null;
                        }
                        if (invokeResult.op() != methodRefInvokeOp) {
                            return null;
                        }
                        assert methodRefInvokeOp.result().uses().size() == 1;
                    }
                    default -> {
                        return null;
                    }
                }
            }

            return methodRefInvokeOp;
        }

        private static boolean isValueUsedWithOp(Value value, Predicate<Op> opPredicate) {
            for (Op.Result user : value.uses()) {
                if (opPredicate.test(user.op())) {
                    return true;
                }
            }
            return false;
        }

        // @@@ Move to functionality on JavaType(s)
        static final Set<String> UNBOX_NAMES = Set.of(
                "byteValue",
                "shortValue",
                "charValue",
                "intValue",
                "longValue",
                "floatValue",
                "doubleValue",
                "booleanValue");

        private static boolean isBoxOrUnboxInvocation(CoreOp.InvokeOp iop) {
            MethodRef mr = iop.invokeDescriptor();
            return mr.refType() instanceof ClassType ct && ct.unbox().isPresent() &&
                    (UNBOX_NAMES.contains(mr.name()) || mr.name().equals("valueOf"));
        }
    }

    /**
     * The closure operation, that can model a structured Java lambda expression
     * that has no target type (a functional interface).
     */
    @OpFactory.OpDeclaration(ClosureOp.NAME)
    public static final class ClosureOp extends CoreOp
            implements Op.Invokable, Op.Lowerable, JavaExpression {

        public static class Builder {
            final Body.Builder ancestorBody;
            final FunctionType funcType;

            Builder(Body.Builder ancestorBody, FunctionType funcType) {
                this.ancestorBody = ancestorBody;
                this.funcType = funcType;
            }

            public ClosureOp body(Consumer<Block.Builder> c) {
                Body.Builder body = Body.Builder.of(ancestorBody, funcType);
                c.accept(body.entryBlock());
                return new ClosureOp(body);
            }
        }

        public static final String NAME = "closure";

        final Body body;

        public ClosureOp(ExternalizedOp def) {
            super(def);

            this.body = def.bodyDefinitions().get(0).build(this);
        }

        ClosureOp(ClosureOp that, CopyContext cc, OpTransformer ot) {
            super(that, cc);

            this.body = that.body.transform(cc, ot).build(this);
        }

        @Override
        public ClosureOp transform(CopyContext cc, OpTransformer ot) {
            return new ClosureOp(this, cc, ot);
        }

        ClosureOp(Body.Builder bodyC) {
            super(NAME,
                    List.of());

            this.body = bodyC.build(this);
        }

        @Override
        public List<Body> bodies() {
            return List.of(body);
        }

        @Override
        public FunctionType invokableType() {
            return body.bodyType();
        }

        @Override
        public Body body() {
            return body;
        }

        @Override
        public List<Value> capturedValues() {
            return body.capturedValues();
        }

        @Override
        public Block.Builder lower(Block.Builder b, OpTransformer _ignore) {
            // Isolate body with respect to ancestor transformations
            b.op(this, OpTransformer.LOWERING_TRANSFORMER);
            return b;
        }

        @Override
        public TypeElement resultType() {
            return body.bodyType();
        }
    }

    /**
     * The closure call operation, that models a call to a closure, by reference
     */
//  @@@ stack effects equivalent to the invocation of an SAM of on an instance of an anonymous functional interface
//  that is the target of the closures lambda expression.
    @OpFactory.OpDeclaration(ClosureCallOp.NAME)
    public static final class ClosureCallOp extends CoreOp {
        public static final String NAME = "closure.call";

        public ClosureCallOp(ExternalizedOp def) {
            super(def);
        }

        ClosureCallOp(ClosureCallOp that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public ClosureCallOp transform(CopyContext cc, OpTransformer ot) {
            return new ClosureCallOp(this, cc);
        }

        ClosureCallOp(List<Value> args) {
            super(NAME, args);
        }

        @Override
        public TypeElement resultType() {
            FunctionType ft = (FunctionType) operands().getFirst().type();
            return ft.returnType();
        }
    }

    /**
     * The terminating return operation, that can model the Java language return statement.
     * <p>
     * This operation exits an isolated body.
     */
    @OpFactory.OpDeclaration(ReturnOp.NAME)
    public static final class ReturnOp extends CoreOp
            implements Op.BodyTerminating, JavaStatement {
        public static final String NAME = "return";

        public ReturnOp(ExternalizedOp def) {
            super(def);

            if (def.operands().size() > 1) {
                throw new IllegalArgumentException("Operation must have zero or one operand " + def.name());
            }
        }

        ReturnOp(ReturnOp that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public ReturnOp transform(CopyContext cc, OpTransformer ot) {
            return new ReturnOp(this, cc);
        }

        ReturnOp() {
            super(NAME, List.of());
        }

        ReturnOp(Value operand) {
            super(NAME, List.of(operand));
        }

        public Value returnValue() {
            if (operands().size() == 1) {
                return operands().get(0);
            } else {
                // @@@
                return null;
            }
        }

        @Override
        public TypeElement resultType() {
            return JavaType.VOID;
        }
    }

    /**
     * The terminating throw operation, that can model the Java language throw statement.
     */
    @OpFactory.OpDeclaration(ThrowOp.NAME)
    public static final class ThrowOp extends CoreOp
            implements Op.BodyTerminating, JavaStatement {
        public static final String NAME = "throw";

        public ThrowOp(ExternalizedOp def) {
            super(def);

            if (def.operands().size() != 1) {
                throw new IllegalArgumentException("Operation must have one operand " + def.name());
            }
        }

        ThrowOp(ThrowOp that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public ThrowOp transform(CopyContext cc, OpTransformer ot) {
            return new ThrowOp(this, cc);
        }

        ThrowOp(Value e) {
            super(NAME, List.of(e));
        }

        public Value argument() {
            return operands().get(0);
        }

        @Override
        public TypeElement resultType() {
            return JavaType.VOID;
        }
    }

    /**
     * The assertion operation. Supporting assertions in statement form.
     */
    @OpFactory.OpDeclaration(AssertOp.NAME)
    public static final class AssertOp extends CoreOp
            implements Op.Nested, JavaStatement {
        public static final String NAME = "assert";
        public final List<Body> bodies;

        public AssertOp(ExternalizedOp def) {
            super(def);
            var bodies = def.bodyDefinitions().stream().map(b -> b.build(this)).toList();
            checkBodies(bodies);
            this.bodies = bodies;
        }

        public AssertOp(List<Body.Builder> bodies) {
            super(NAME, List.of());
            checkBodies(bodies);
            this.bodies = bodies.stream().map(b -> b.build(this)).toList();
        }

        AssertOp(AssertOp that, CopyContext cc, OpTransformer ot) {

            super(that, cc);
            this.bodies = that.bodies.stream().map(b -> b.transform(cc, ot).build(this)).toList();
        }

        private void checkBodies(List<?> bodies) {
            if (bodies.size() != 1 && bodies.size() != 2) {
                throw new IllegalArgumentException("Assert must have one or two bodies.");
            }
        }

        @Override
        public Op transform(CopyContext cc, OpTransformer ot) {
            return new AssertOp(this, cc, ot);
        }

        @Override
        public TypeElement resultType() {
            return JavaType.VOID;
        }

        @Override
        public List<Body> bodies() {
            return this.bodies;
        }
    }

    /**
     * The terminating unreachable operation.
     * <p>
     * This operation models termination that is unreachable.
     */
    @OpFactory.OpDeclaration(UnreachableOp.NAME)
    public static final class UnreachableOp extends CoreOp
            implements Op.BodyTerminating {
        public static final String NAME = "unreachable";

        public UnreachableOp(ExternalizedOp def) {
            super(def);

            if (!def.operands().isEmpty()) {
                throw new IllegalArgumentException("Operation must zero operands " + def.name());
            }
        }

        UnreachableOp(UnreachableOp that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public UnreachableOp transform(CopyContext cc, OpTransformer ot) {
            return new UnreachableOp(this, cc);
        }

        UnreachableOp() {
            super(NAME, List.of());
        }

        @Override
        public TypeElement resultType() {
            return JavaType.VOID;
        }
    }

    /**
     * The terminating yield operation.
     * <p>
     * This operation models exits from its parent body, yielding at most one value (zero value for yielding unit
     * or void)
     */
    @OpFactory.OpDeclaration(YieldOp.NAME)
    public static final class YieldOp extends CoreOp
            implements Op.BodyTerminating {
        public static final String NAME = "yield";

        public YieldOp(ExternalizedOp def) {
            super(def);

            if (def.operands().size() > 1) {
                throw new IllegalArgumentException("Operation must have zero or one operand " + def.name());
            }
        }

        YieldOp(YieldOp that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public YieldOp transform(CopyContext cc, OpTransformer ot) {
            return new YieldOp(this, cc);
        }

        YieldOp() {
            super(NAME, List.of());
        }

        YieldOp(List<Value> operands) {
            super(NAME, operands);
        }

        public Value yieldValue() {
            if (operands().size() == 1) {
                return operands().get(0);
            } else {
                // @@@
                return null;
            }
        }

        @Override
        public TypeElement resultType() {
            return JavaType.VOID;
        }
    }

    /**
     * The terminating unconditional branch operation.
     * <p>
     * This operation accepts a successor to the next block to branch to.
     */
    @OpFactory.OpDeclaration(BranchOp.NAME)
    public static final class BranchOp extends CoreOp
            implements Op.BlockTerminating {
        public static final String NAME = "branch";

        final Block.Reference b;

        public BranchOp(ExternalizedOp def) {
            super(def);

            if (!def.operands().isEmpty() || def.successors().size() != 1) {
                throw new IllegalArgumentException("Operation must have zero arguments and one successor" + def.name());
            }

            this.b = def.successors().get(0);
        }

        BranchOp(BranchOp that, CopyContext cc) {
            super(that, cc);

            this.b = cc.getSuccessorOrCreate(that.b);
        }

        @Override
        public BranchOp transform(CopyContext cc, OpTransformer ot) {
            return new BranchOp(this, cc);
        }

        BranchOp(Block.Reference successor) {
            super(NAME, List.of());

            this.b = successor;
        }

        @Override
        public List<Block.Reference> successors() {
            return List.of(b);
        }

        public Block.Reference branch() {
            return b;
        }

        @Override
        public TypeElement resultType() {
            return JavaType.VOID;
        }
    }

    /**
     * The terminating conditional branch operation.
     * <p>
     * This operation accepts a boolean operand and two successors, the true successor and false successor.
     * When the operand is true the  true successor is selected, otherwise the false successor is selected.
     * The selected successor refers to the next block to branch to.
     */
    @OpFactory.OpDeclaration(ConditionalBranchOp.NAME)
    public static final class ConditionalBranchOp extends CoreOp
            implements Op.BlockTerminating {
        public static final String NAME = "cbranch";

        final Block.Reference t;
        final Block.Reference f;

        public ConditionalBranchOp(ExternalizedOp def) {
            super(def);

            if (def.operands().size() != 1 || def.successors().size() != 2) {
                throw new IllegalArgumentException("Operation must one operand and two successors" + def.name());
            }

            this.t = def.successors().get(0);
            this.f = def.successors().get(1);
        }

        ConditionalBranchOp(ConditionalBranchOp that, CopyContext cc) {
            super(that, cc);

            this.t = cc.getSuccessorOrCreate(that.t);
            this.f = cc.getSuccessorOrCreate(that.f);
        }

        @Override
        public ConditionalBranchOp transform(CopyContext cc, OpTransformer ot) {
            return new ConditionalBranchOp(this, cc);
        }

        ConditionalBranchOp(Value p, Block.Reference t, Block.Reference f) {
            super(NAME, List.of(p));

            this.t = t;
            this.f = f;
        }

        @Override
        public List<Block.Reference> successors() {
            return List.of(t, f);
        }

        public Value predicate() {
            return operands().get(0);
        }

        public Block.Reference trueBranch() {
            return t;
        }

        public Block.Reference falseBranch() {
            return f;
        }

        @Override
        public TypeElement resultType() {
            return JavaType.VOID;
        }
    }

    /**
     * The constant operation, that can model Java language literal and constant expressions.
     */
    @OpFactory.OpDeclaration(ConstantOp.NAME)
    public static final class ConstantOp extends CoreOp
            implements Op.Pure, JavaExpression {
        public static final String NAME = "constant";

        public static final String ATTRIBUTE_CONSTANT_VALUE = NAME + ".value";

        final Object value;
        final TypeElement type;

        public static ConstantOp create(ExternalizedOp def) {
            if (!def.operands().isEmpty()) {
                throw new IllegalArgumentException("Operation must have zero operands");
            }

            Object value = def.extractAttributeValue(ATTRIBUTE_CONSTANT_VALUE, true,
                    v -> processConstantValue(def.resultType(), v));
            return new ConstantOp(def, value);
        }

        static Object processConstantValue(TypeElement t, Object value) {
            if (t.equals(JavaType.BOOLEAN)) {
                if (value instanceof String s) {
                    return Boolean.valueOf(s);
                } else if (value instanceof Boolean) {
                    return value;
                }
            } else if (t.equals(JavaType.BYTE)) {
                if (value instanceof String s) {
                    return Byte.valueOf(s);
                } else if (value instanceof Number n) {
                    return n.byteValue();
                }
            } else if (t.equals(JavaType.SHORT)) {
                if (value instanceof String s) {
                    return Short.valueOf(s);
                } else if (value instanceof Number n) {
                    return n.shortValue();
                }
            } else if (t.equals(JavaType.CHAR)) {
                if (value instanceof String s) {
                    return s.charAt(0);
                } else if (value instanceof Character) {
                    return value;
                }
            } else if (t.equals(JavaType.INT)) {
                if (value instanceof String s) {
                    return Integer.valueOf(s);
                } else if (value instanceof Number n) {
                    return n.intValue();
                }
            } else if (t.equals(JavaType.LONG)) {
                if (value instanceof String s) {
                    return Long.valueOf(s);
                } else if (value instanceof Number n) {
                    return n.longValue();
                }
            } else if (t.equals(JavaType.FLOAT)) {
                if (value instanceof String s) {
                    return Float.valueOf(s);
                } else if (value instanceof Number n) {
                    return n.floatValue();
                }
            } else if (t.equals(JavaType.DOUBLE)) {
                if (value instanceof String s) {
                    return Double.valueOf(s);
                } else if (value instanceof Number n) {
                    return n.doubleValue();
                }
            } else if (t.equals(JavaType.J_L_STRING)) {
                return value == NULL_ATTRIBUTE_VALUE ? null :
                        value.toString();
            } else if (t.equals(JavaType.J_L_CLASS)) {
                return value == NULL_ATTRIBUTE_VALUE ? null : JavaType.ofString(value.toString());
            } else if (value == NULL_ATTRIBUTE_VALUE) {
                return null; // null constant
            }

            throw new UnsupportedOperationException("Unsupported constant type and value: " + t + " " + value);
        }

        ConstantOp(ExternalizedOp def, Object value) {
            super(def);

            this.type = def.resultType();
            this.value = value;
        }

        ConstantOp(ConstantOp that, CopyContext cc) {
            super(that, cc);

            this.type = that.type;
            this.value = that.value;
        }

        @Override
        public ConstantOp transform(CopyContext cc, OpTransformer ot) {
            return new ConstantOp(this, cc);
        }

        ConstantOp(TypeElement type, Object value) {
            super(NAME, List.of());

            this.type = type;
            this.value = value;
        }

        @Override
        public Map<String, Object> attributes() {
            HashMap<String, Object> attrs = new HashMap<>(super.attributes());
            attrs.put("", value == null ? NULL_ATTRIBUTE_VALUE : value);
            return attrs;
        }

        public Object value() {
            return value;
        }

        @Override
        public TypeElement resultType() {
            return type;
        }
    }

    /**
     * An operation characteristic indicating the operation's behavior may be emulated using Java reflection.
     * A descriptor is derived from or declared by the operation that can be resolved at runtime to
     * an instance of a reflective handle or member. That handle or member can be operated on to
     * emulate the operation's behavior, specifically as bytecode behavior.
     */
    public sealed interface ReflectiveOp {
    }

    /**
     * The invoke operation, that can model Java language method invocation expressions.
     */
    @OpFactory.OpDeclaration(InvokeOp.NAME)
    public static final class InvokeOp extends CoreOp
            implements ReflectiveOp, JavaExpression, JavaStatement {
        public static final String NAME = "invoke";
        public static final String ATTRIBUTE_INVOKE_DESCRIPTOR = NAME + ".descriptor";

        final MethodRef invokeDescriptor;
        final TypeElement resultType;

        public static InvokeOp create(ExternalizedOp def) {
            MethodRef invokeDescriptor = def.extractAttributeValue(ATTRIBUTE_INVOKE_DESCRIPTOR,
                    true, v -> switch (v) {
                        case String s -> MethodRef.ofString(s);
                        case MethodRef md -> md;
                        default -> throw new UnsupportedOperationException("Unsupported invoke descriptor value:" + v);
                    });

            return new InvokeOp(def, invokeDescriptor);
        }

        InvokeOp(ExternalizedOp def, MethodRef invokeDescriptor) {
            super(def);

            this.invokeDescriptor = invokeDescriptor;
            this.resultType = def.resultType();
        }

        InvokeOp(InvokeOp that, CopyContext cc) {
            super(that, cc);

            this.invokeDescriptor = that.invokeDescriptor;
            this.resultType = that.resultType;
        }

        @Override
        public InvokeOp transform(CopyContext cc, OpTransformer ot) {
            return new InvokeOp(this, cc);
        }

        InvokeOp(MethodRef invokeDescriptor, List<Value> args) {
            this(invokeDescriptor.type().returnType(), invokeDescriptor, args);
        }

        InvokeOp(TypeElement resultType, MethodRef invokeDescriptor, List<Value> args) {
            super(NAME, args);

            this.invokeDescriptor = invokeDescriptor;
            this.resultType = resultType;
        }

        @Override
        public Map<String, Object> attributes() {
            HashMap<String, Object> m = new HashMap<>(super.attributes());
            m.put("", invokeDescriptor);
            return Collections.unmodifiableMap(m);
        }

        public MethodRef invokeDescriptor() {
            return invokeDescriptor;
        }

        public boolean hasReceiver() {
            return operands().size() != invokeDescriptor().type().parameterTypes().size();
        }

        @Override
        public TypeElement resultType() {
            return resultType;
        }
    }

    /**
     * The conversion operation, that can model Java language cast expressions
     * for numerical conversion, or such implicit conversion.
     */
    @OpFactory.OpDeclaration(ConvOp.NAME)
    public static final class ConvOp extends CoreOp
            implements Op.Pure, JavaExpression {
        public static final String NAME = "conv";

        final TypeElement resultType;

        public ConvOp(ExternalizedOp def) {
            super(def);

            this.resultType = def.resultType();
        }

        ConvOp(ConvOp that, CopyContext cc) {
            super(that, cc);

            this.resultType = that.resultType;
        }

        @Override
        public Op transform(CopyContext cc, OpTransformer ot) {
            return new ConvOp(this, cc);
        }

        ConvOp(TypeElement resultType, Value arg) {
            super(NAME, List.of(arg));

            this.resultType = resultType;
        }

        @Override
        public TypeElement resultType() {
            return resultType;
        }
    }

    /**
     * The new operation, that can models Java language instance creation expressions.
     */
    @OpFactory.OpDeclaration(NewOp.NAME)
    public static final class NewOp extends CoreOp
            implements ReflectiveOp, JavaExpression, JavaStatement {
        public static final String NAME = "new";
        public static final String ATTRIBUTE_NEW_DESCRIPTOR = NAME + ".descriptor";

        final FunctionType constructorType;
        final TypeElement resultType;

        public static NewOp create(ExternalizedOp def) {
            FunctionType constructorType = def.extractAttributeValue(ATTRIBUTE_NEW_DESCRIPTOR, true,
                    v -> switch (v) {
                        case String s -> {
                            TypeElement te = CoreTypeFactory.CORE_TYPE_FACTORY
                                    .constructType(TypeElement.ExternalizedTypeElement.ofString(s));
                            if (!(te instanceof FunctionType ft)) {
                                throw new UnsupportedOperationException("Unsupported new descriptor value:" + v);
                            }
                            yield ft;
                        }
                        case FunctionType ct -> ct;
                        default -> throw new UnsupportedOperationException("Unsupported new descriptor value:" + v);
                    });
            return new NewOp(def, constructorType);
        }

        NewOp(ExternalizedOp def, FunctionType constructorType) {
            super(def);

            this.constructorType = constructorType;
            this.resultType = def.resultType();
        }

        NewOp(NewOp that, CopyContext cc) {
            super(that, cc);

            this.constructorType = that.constructorType;
            this.resultType = that.resultType;
        }

        @Override
        public NewOp transform(CopyContext cc, OpTransformer ot) {
            return new NewOp(this, cc);
        }

        NewOp(FunctionType constructorType, List<Value> args) {
            this(constructorType.returnType(), constructorType, args);
        }

        NewOp(TypeElement resultType, FunctionType constructorType, List<Value> args) {
            super(NAME, args);

            this.constructorType = constructorType;
            this.resultType = resultType;
        }

        @Override
        public Map<String, Object> attributes() {
            HashMap<String, Object> m = new HashMap<>(super.attributes());
            m.put("", constructorType);
            return Collections.unmodifiableMap(m);
        }

        public TypeElement type() {
            return opType().returnType();
        }

        public FunctionType constructorType() {
            return constructorType;
        }

        @Override
        public TypeElement resultType() {
            return resultType;
        }
    }

    /**
     * An operation that performs access.
     */
    public sealed interface AccessOp {
    }

    /**
     * A field access operation, that can model Java langauge field access expressions.
     */
    public sealed abstract static class FieldAccessOp extends CoreOp
            implements AccessOp, ReflectiveOp {
        public static final String ATTRIBUTE_FIELD_DESCRIPTOR = "field.descriptor";

        final FieldRef fieldDescriptor;

        FieldAccessOp(ExternalizedOp def, FieldRef fieldDescriptor) {
            super(def);

            this.fieldDescriptor = fieldDescriptor;
        }

        FieldAccessOp(FieldAccessOp that, CopyContext cc) {
            super(that, cc);

            this.fieldDescriptor = that.fieldDescriptor;
        }

        FieldAccessOp(String name, List<Value> operands,
                      FieldRef fieldDescriptor) {
            super(name, operands);

            this.fieldDescriptor = fieldDescriptor;
        }

        @Override
        public Map<String, Object> attributes() {
            HashMap<String, Object> m = new HashMap<>(super.attributes());
            m.put("", fieldDescriptor);
            return Collections.unmodifiableMap(m);
        }

        public final FieldRef fieldDescriptor() {
            return fieldDescriptor;
        }

        /**
         * The field load operation, that can model Java language field access expressions combined with load access to
         * field instance variables.
         */
        @OpFactory.OpDeclaration(FieldLoadOp.NAME)
        public static final class FieldLoadOp extends FieldAccessOp
                implements Op.Pure, JavaExpression {
            public static final String NAME = "field.load";

            final TypeElement resultType;

            public static FieldLoadOp create(ExternalizedOp def) {
                if (def.operands().size() > 1) {
                    throw new IllegalArgumentException("Operation must accept zero or one operand");
                }

                FieldRef fieldDescriptor = def.extractAttributeValue(ATTRIBUTE_FIELD_DESCRIPTOR, true,
                        v -> switch (v) {
                            case String s -> FieldRef.ofString(s);
                            case FieldRef fd -> fd;
                            default ->
                                    throw new UnsupportedOperationException("Unsupported field descriptor value:" + v);
                        });
                return new FieldLoadOp(def, fieldDescriptor);
            }

            FieldLoadOp(ExternalizedOp opdef, FieldRef fieldDescriptor) {
                super(opdef, fieldDescriptor);

                resultType = opdef.resultType();
            }

            FieldLoadOp(FieldLoadOp that, CopyContext cc) {
                super(that, cc);

                resultType = that.resultType();
            }

            @Override
            public FieldLoadOp transform(CopyContext cc, OpTransformer ot) {
                return new FieldLoadOp(this, cc);
            }

            // instance
            FieldLoadOp(TypeElement resultType, FieldRef descriptor, Value receiver) {
                super(NAME, List.of(receiver), descriptor);

                this.resultType = resultType;
            }

            // static
            FieldLoadOp(TypeElement resultType, FieldRef descriptor) {
                super(NAME, List.of(), descriptor);

                this.resultType = resultType;
            }

            @Override
            public TypeElement resultType() {
                return resultType;
            }
        }

        /**
         * The field store operation, that can model Java language field access expressions combined with store access
         * to field instance variables.
         */
        @OpFactory.OpDeclaration(FieldStoreOp.NAME)
        public static final class FieldStoreOp extends FieldAccessOp
                implements JavaExpression, JavaStatement {
            public static final String NAME = "field.store";

            public static FieldStoreOp create(ExternalizedOp def) {
                if (def.operands().size() > 2) {
                    throw new IllegalArgumentException("Operation must accept one or two operands");
                }

                FieldRef fieldDescriptor = def.extractAttributeValue(ATTRIBUTE_FIELD_DESCRIPTOR, true,
                        v -> switch (v) {
                            case String s -> FieldRef.ofString(s);
                            case FieldRef fd -> fd;
                            default ->
                                    throw new UnsupportedOperationException("Unsupported field descriptor value:" + v);
                        });
                return new FieldStoreOp(def, fieldDescriptor);
            }

            FieldStoreOp(ExternalizedOp opdef, FieldRef fieldDescriptor) {
                super(opdef, fieldDescriptor);
            }

            FieldStoreOp(FieldStoreOp that, CopyContext cc) {
                super(that, cc);
            }

            @Override
            public FieldStoreOp transform(CopyContext cc, OpTransformer ot) {
                return new FieldStoreOp(this, cc);
            }

            // instance
            FieldStoreOp(FieldRef descriptor, Value receiver, Value v) {
                super(NAME,
                        List.of(receiver, v), descriptor);
            }

            // static
            FieldStoreOp(FieldRef descriptor, Value v) {
                super(NAME,
                        List.of(v), descriptor);
            }

            @Override
            public TypeElement resultType() {
                return JavaType.VOID;
            }
        }
    }

    /**
     * The array length operation, that can model Java language field access expressions to the length field of an
     * array.
     */
    @OpFactory.OpDeclaration(ArrayLengthOp.NAME)
    public static final class ArrayLengthOp extends CoreOp
            implements ReflectiveOp, JavaExpression {
        public static final String NAME = "array.length";

        public ArrayLengthOp(ExternalizedOp def) {
            super(def);
        }

        ArrayLengthOp(ArrayLengthOp that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public ArrayLengthOp transform(CopyContext cc, OpTransformer ot) {
            return new ArrayLengthOp(this, cc);
        }

        ArrayLengthOp(Value array) {
            super(NAME, List.of(array));
        }

        @Override
        public TypeElement resultType() {
            return JavaType.INT;
        }
    }

    /**
     * The array access operation, that can model Java language array access expressions.
     */
    public sealed abstract static class ArrayAccessOp extends CoreOp
            implements AccessOp, ReflectiveOp {
        ArrayAccessOp(ExternalizedOp def) {
            super(def);

            if (def.operands().size() != 2 && def.operands().size() != 3) {
                throw new IllegalArgumentException("Operation must have 2 or 3 operands");
            }

            // @@@ validate first operand is an array
        }

        ArrayAccessOp(ArrayAccessOp that, CopyContext cc) {
            this(that, cc.getValues(that.operands()));
        }

        ArrayAccessOp(ArrayAccessOp that, List<Value> operands) {
            super(that.opName(), operands);
        }

        ArrayAccessOp(String name,
                      Value array, Value index, Value v) {
            super(name, operands(array, index, v));
        }

        static List<Value> operands(Value array, Value index, Value v) {
            return v == null
                    ? List.of(array, index)
                    : List.of(array, index, v);
        }

        static TypeElement resultType(Value array, Value v) {
            if (!(array.type() instanceof ArrayType arrayType)) {
                throw new IllegalArgumentException("Type is not an array type: " + array.type());
            }

            // @@@ restrict to indexes of int?
            TypeElement componentType = arrayType.componentType();
            if (v == null) {
                return componentType;
            } else {
                return JavaType.VOID;
            }
        }

        /**
         * The array load operation, that can model Java language array expressions combined with load access to the
         * components of an array.
         */
        @OpFactory.OpDeclaration(ArrayLoadOp.NAME)
        public static final class ArrayLoadOp extends ArrayAccessOp
                implements Op.Pure, JavaExpression {
            public static final String NAME = "array.load";

            public ArrayLoadOp(ExternalizedOp def) {
                super(def);
            }

            ArrayLoadOp(ArrayLoadOp that, CopyContext cc) {
                super(that, cc);
            }

            @Override
            public ArrayLoadOp transform(CopyContext cc, OpTransformer ot) {
                return new ArrayLoadOp(this, cc);
            }

            ArrayLoadOp(Value array, Value index) {
                super(NAME, array, index, null);
            }

            @Override
            public TypeElement resultType() {
                Value array = operands().get(0);
                ArrayType t = (ArrayType) array.type();
                return t.componentType();
            }
        }

        /**
         * The array store operation, that can model Java language array expressions combined with store access to the
         * components of an array.
         */
        @OpFactory.OpDeclaration(ArrayStoreOp.NAME)
        public static final class ArrayStoreOp extends ArrayAccessOp
                implements JavaExpression, JavaStatement {
            public static final String NAME = "array.store";

            public ArrayStoreOp(ExternalizedOp def) {
                super(def);
            }

            ArrayStoreOp(ArrayStoreOp that, CopyContext cc) {
                super(that, cc);
            }

            @Override
            public ArrayStoreOp transform(CopyContext cc, OpTransformer ot) {
                return new ArrayStoreOp(this, cc);
            }

            ArrayStoreOp(Value array, Value index, Value v) {
                super(NAME, array, index, v);
            }

            @Override
            public TypeElement resultType() {
                return JavaType.VOID;
            }
        }
    }

    /**
     * The instanceof operation, that can model Java language instanceof expressions when the instanceof keyword is a
     * type comparison operator.
     */
    @OpFactory.OpDeclaration(InstanceOfOp.NAME)
    public static final class InstanceOfOp extends CoreOp
            implements Op.Pure, ReflectiveOp, JavaExpression {
        public static final String NAME = "instanceof";
        public static final String ATTRIBUTE_TYPE_DESCRIPTOR = NAME + ".descriptor";

        final TypeElement typeDescriptor;

        public static InstanceOfOp create(ExternalizedOp def) {
            if (def.operands().size() != 1) {
                throw new IllegalArgumentException("Operation must have one operand " + def.name());
            }

            TypeElement typeDescriptor = def.extractAttributeValue(ATTRIBUTE_TYPE_DESCRIPTOR, true,
                    v -> switch (v) {
                        case String s -> JavaType.ofString(s);
                        case JavaType td -> td;
                        default -> throw new UnsupportedOperationException("Unsupported type descriptor value:" + v);
                    });
            return new InstanceOfOp(def, typeDescriptor);
        }

        InstanceOfOp(ExternalizedOp def, TypeElement typeDescriptor) {
            super(def);

            this.typeDescriptor = typeDescriptor;
        }

        InstanceOfOp(InstanceOfOp that, CopyContext cc) {
            super(that, cc);

            this.typeDescriptor = that.typeDescriptor;
        }

        @Override
        public InstanceOfOp transform(CopyContext cc, OpTransformer ot) {
            return new InstanceOfOp(this, cc);
        }

        InstanceOfOp(TypeElement t, Value v) {
            super(NAME,
                    List.of(v));

            this.typeDescriptor = t;
        }

        @Override
        public Map<String, Object> attributes() {
            HashMap<String, Object> m = new HashMap<>(super.attributes());
            m.put("", typeDescriptor);
            return Collections.unmodifiableMap(m);
        }

        public TypeElement type() {
            return typeDescriptor;
        }

        @Override
        public TypeElement resultType() {
            return JavaType.BOOLEAN;
        }
    }

    /**
     * The cast operation, that can model Java language cast expressions for reference types.
     */
    @OpFactory.OpDeclaration(CastOp.NAME)
    public static final class CastOp extends CoreOp
            implements Op.Pure, ReflectiveOp, JavaExpression {
        public static final String NAME = "cast";
        public static final String ATTRIBUTE_TYPE_DESCRIPTOR = NAME + ".descriptor";

        final TypeElement resultType;
        final TypeElement typeDescriptor;

        public static CastOp create(ExternalizedOp def) {
            if (def.operands().size() != 1) {
                throw new IllegalArgumentException("Operation must have one operand " + def.name());
            }

            TypeElement type = def.extractAttributeValue(ATTRIBUTE_TYPE_DESCRIPTOR, true,
                    v -> switch (v) {
                        case String s -> JavaType.ofString(s);
                        case JavaType td -> td;
                        default -> throw new UnsupportedOperationException("Unsupported type descriptor value:" + v);
                    });
            return new CastOp(def, type);
        }

        CastOp(ExternalizedOp def, TypeElement typeDescriptor) {
            super(def);

            this.resultType = def.resultType();
            this.typeDescriptor = typeDescriptor;
        }

        CastOp(CastOp that, CopyContext cc) {
            super(that, cc);

            this.resultType = that.resultType;
            this.typeDescriptor = that.typeDescriptor;
        }

        @Override
        public CastOp transform(CopyContext cc, OpTransformer ot) {
            return new CastOp(this, cc);
        }

        CastOp(TypeElement resultType, TypeElement t, Value v) {
            super(NAME, List.of(v));

            this.resultType = resultType;
            this.typeDescriptor = t;
        }

        @Override
        public Map<String, Object> attributes() {
            HashMap<String, Object> m = new HashMap<>(super.attributes());
            m.put("", typeDescriptor);
            return Collections.unmodifiableMap(m);
        }

        public TypeElement type() {
            return typeDescriptor;
        }

        @Override
        public TypeElement resultType() {
            return resultType;
        }
    }


    /**
     * A runtime representation of a variable.
     *
     * @param <T> the type of the var's value.
     * @@@ Ideally should never be exposed
     * @@@ Move to interpreter?
     */
    public interface Var<T> {
        /**
         * {@return the value of a var}
         */
        T value();

        /**
         * Constructs an instance of a var.
         *
         * @param value the initial value of the var.
         * @param <T>   the type of the var's value.
         * @return the var
         */
        static <T> Var<T> of(T value) {
            return () -> value;
        }
    }

    /**
     * The variable operation, that can model declarations of Java language local variables, method parameters, or
     * lambda parameters.
     */
    @OpFactory.OpDeclaration(VarOp.NAME)
    public static final class VarOp extends CoreOp
            implements JavaStatement {
        public static final String NAME = "var";
        public static final String ATTRIBUTE_NAME = NAME + ".name";

        final String varName;
        final VarType resultType;

        public static VarOp create(ExternalizedOp def) {
            if (def.operands().size() != 1) {
                throw new IllegalStateException("Operation must have one operand");
            }

            String name = def.extractAttributeValue(ATTRIBUTE_NAME, true,
                    v -> switch (v) {
                        case String s -> s;
                        default -> throw new UnsupportedOperationException("Unsupported var name value:" + v);
                    });
            return new VarOp(def, name);
        }

        VarOp(ExternalizedOp def, String varName) {
            super(def);

            this.varName = varName;
            this.resultType = (VarType) def.resultType();
        }

        VarOp(VarOp that, CopyContext cc) {
            super(that, cc);

            this.varName = that.varName;
            this.resultType = that.isResultTypeOverridable()
                    ? VarType.varType(initOperand().type()) : that.resultType;
        }

        boolean isResultTypeOverridable() {
            return resultType().valueType().equals(initOperand().type());
        }

        @Override
        public VarOp transform(CopyContext cc, OpTransformer ot) {
            return new VarOp(this, cc);
        }

        VarOp(String varName, Value init) {
            this(varName, init.type(), init);
        }

        VarOp(String varName, TypeElement type, Value init) {
            super(NAME, List.of(init));

            this.varName = varName;
            this.resultType = VarType.varType(type);
        }

        @Override
        public Map<String, Object> attributes() {
            if (varName == null) {
                return super.attributes();
            }

            HashMap<String, Object> m = new HashMap<>(super.attributes());
            m.put("", varName);
            return Collections.unmodifiableMap(m);
        }

        public Value initOperand() {
            return operands().getFirst();
        }

        public String varName() {
            return varName;
        }

        public TypeElement varValueType() {
            return resultType.valueType();
        }

        @Override
        public VarType resultType() {
            return resultType;
        }
    }

    /**
     * The var access operation, that can model access to Java language local variables, method parameters, or
     * lambda parameters.
     */
    public sealed abstract static class VarAccessOp extends CoreOp
            implements AccessOp {
        VarAccessOp(ExternalizedOp opdef) {
            super(opdef);
        }

        VarAccessOp(VarAccessOp that, CopyContext cc) {
            super(that, cc);
        }

        VarAccessOp(String name, List<Value> operands) {
            super(name, operands);
        }

        public Value varOperand() {
            return operands().getFirst();
        }

        public VarType varType() {
            return (VarType) varOperand().type();
        }

        public VarOp varOp() {
            if (!(varOperand() instanceof Result varValue)) {
                throw new IllegalStateException("Variable access to block parameter: " + varOperand());
            }

            // At a high-level a variable value can be a BlockArgument.
            // Lowering should remove such cases and the var declaration should emerge
            // This method is primarily used when transforming to pure SSA
            return (VarOp) varValue.op();
        }

        static void checkIsVarOp(Value varValue) {
            if (!(varValue.type() instanceof VarType)) {
                throw new IllegalArgumentException("Value's type is not a variable type: " + varValue);
            }
        }

        /**
         * The variable load operation, that models a reading variable.
         */
        @OpFactory.OpDeclaration(VarLoadOp.NAME)
        public static final class VarLoadOp extends VarAccessOp
                implements JavaExpression {
            public static final String NAME = "var.load";

            public VarLoadOp(ExternalizedOp opdef) {
                super(opdef);

                if (opdef.operands().size() != 1) {
                    throw new IllegalArgumentException("Operation must have one operand");
                }
                checkIsVarOp(opdef.operands().get(0));
            }

            VarLoadOp(VarLoadOp that, CopyContext cc) {
                super(that, cc);
            }

            VarLoadOp(List<Value> varValue) {
                super(NAME, varValue);
            }

            @Override
            public VarLoadOp transform(CopyContext cc, OpTransformer ot) {
                return new VarLoadOp(this, cc);
            }

            // (Variable)VarType
            VarLoadOp(Value varValue) {
                super(NAME, List.of(varValue));
            }

            @Override
            public TypeElement resultType() {
                return varType().valueType();
            }
        }

        /**
         * The variable store operation, that can model a variable assignment.
         */
        @OpFactory.OpDeclaration(VarStoreOp.NAME)
        public static final class VarStoreOp extends VarAccessOp
                implements JavaExpression, JavaStatement {
            public static final String NAME = "var.store";

            public VarStoreOp(ExternalizedOp opdef) {
                super(opdef);

                if (opdef.operands().size() != 2) {
                    throw new IllegalArgumentException("Operation must have two operands");
                }
                checkIsVarOp(opdef.operands().get(0));
            }

            VarStoreOp(VarStoreOp that, CopyContext cc) {
                super(that, cc);
            }

            VarStoreOp(List<Value> values) {
                super(NAME,
                        values);
            }

            @Override
            public VarStoreOp transform(CopyContext cc, OpTransformer ot) {
                return new VarStoreOp(this, cc);
            }

            // (Variable, VarType)void
            VarStoreOp(Value varValue, Value v) {
                super(NAME,
                        List.of(varValue, v));
            }

            public Value storeOperand() {
                return operands().get(1);
            }

            @Override
            public TypeElement resultType() {
                return JavaType.VOID;
            }
        }
    }

    // Tuple operations, for modelling return with multiple values

    /**
     * The tuple operation. A tuple contain a fixed set of values accessible by their component index.
     */
    @OpFactory.OpDeclaration(TupleOp.NAME)
    public static final class TupleOp extends CoreOp {
        public static final String NAME = "tuple";

        public TupleOp(ExternalizedOp def) {
            super(def);
        }

        TupleOp(TupleOp that, CopyContext cc) {
            this(cc.getValues(that.operands()));
        }

        @Override
        public TupleOp transform(CopyContext cc, OpTransformer ot) {
            return new TupleOp(this, cc);
        }

        TupleOp(List<? extends Value> componentValues) {
            super(NAME, componentValues);
        }

        @Override
        public TypeElement resultType() {
            return TupleType.tupleTypeFromValues(operands());
        }
    }

    /**
     * The tuple component load operation, that access the component of a tuple at a given, constant, component index.
     */
    @OpFactory.OpDeclaration(TupleLoadOp.NAME)
    public static final class TupleLoadOp extends CoreOp {
        public static final String NAME = "tuple.load";
        public static final String ATTRIBUTE_INDEX = NAME + ".index";

        final int index;

        public static TupleLoadOp create(ExternalizedOp def) {
            if (def.operands().size() != 1) {
                throw new IllegalStateException("Operation must have one operand");
            }

            int index = def.extractAttributeValue(ATTRIBUTE_INDEX, true,
                    v -> switch (v) {
                        case String s -> Integer.valueOf(s);
                        case Integer i -> i;
                        default -> throw new UnsupportedOperationException("Unsupported tuple index value:" + v);
                    });
            return new TupleLoadOp(def, index);
        }

        TupleLoadOp(ExternalizedOp def, int index) {
            super(def);

            // @@@ Validate tuple type and index
            this.index = index;
        }

        TupleLoadOp(TupleLoadOp that, CopyContext cc) {
            this(that, cc.getValues(that.operands()));
        }

        TupleLoadOp(TupleLoadOp that, List<Value> values) {
            super(NAME, values);

            this.index = that.index;
        }

        @Override
        public TupleLoadOp transform(CopyContext cc, OpTransformer ot) {
            return new TupleLoadOp(this, cc);
        }

        TupleLoadOp(Value tupleValue, int index) {
            super(NAME, List.of(tupleValue));

            this.index = index;
        }

        @Override
        public Map<String, Object> attributes() {
            HashMap<String, Object> m = new HashMap<>(super.attributes());
            m.put("", index);
            return Collections.unmodifiableMap(m);
        }

        public int index() {
            return index;
        }

        @Override
        public TypeElement resultType() {
            Value tupleValue = operands().get(0);
            TupleType t = (TupleType) tupleValue.type();
            return t.componentTypes().get(index);
        }
    }

    /**
     * The tuple component set operation, that access the component of a tuple at a given, constant, component index.
     */
    @OpFactory.OpDeclaration(TupleWithOp.NAME)
    public static final class TupleWithOp extends CoreOp {
        public static final String NAME = "tuple.with";
        public static final String ATTRIBUTE_INDEX = NAME + ".index";

        final int index;

        public static TupleWithOp create(ExternalizedOp def) {
            if (def.operands().size() != 2) {
                throw new IllegalStateException("Operation must have two operands");
            }

            int index = def.extractAttributeValue(ATTRIBUTE_INDEX, true,
                    v -> switch (v) {
                        case String s -> Integer.valueOf(s);
                        case Integer i -> i;
                        default -> throw new UnsupportedOperationException("Unsupported tuple index value:" + v);
                    });
            return new TupleWithOp(def, index);
        }

        TupleWithOp(ExternalizedOp def, int index) {
            super(def);

            // @@@ Validate tuple type and index
            this.index = index;
        }

        TupleWithOp(TupleWithOp that, CopyContext cc) {
            this(that, cc.getValues(that.operands()));
        }

        TupleWithOp(TupleWithOp that, List<Value> values) {
            super(NAME, values);

            this.index = that.index;
        }

        @Override
        public TupleWithOp transform(CopyContext cc, OpTransformer ot) {
            return new TupleWithOp(this, cc);
        }

        TupleWithOp(Value tupleValue, int index, Value value) {
            super(NAME, List.of(tupleValue, value));

            // @@@ Validate tuple type and index
            this.index = index;
        }

        @Override
        public Map<String, Object> attributes() {
            HashMap<String, Object> m = new HashMap<>(super.attributes());
            m.put("", index);
            return Collections.unmodifiableMap(m);
        }

        public int index() {
            return index;
        }

        @Override
        public TypeElement resultType() {
            Value tupleValue = operands().get(0);
            TupleType tupleType = (TupleType) tupleValue.type();
            Value value = operands().get(2);

            List<TypeElement> tupleComponentTypes = new ArrayList<>(tupleType.componentTypes());
            tupleComponentTypes.set(index, value.type());
            return TupleType.tupleType(tupleComponentTypes);
        }
    }

    // @@@ Sealed
    // Synthetic/hidden type that is the result type of an ExceptionRegionStart operation
    // and is an operand of an ExceptionRegionEnd operation

    /**
     * A synthetic exception region type, that is the operation result-type of an exception region
     * start operation.
     */
    // @@@: Create as new type element
    public interface ExceptionRegion {
        TypeElement EXCEPTION_REGION_TYPE = JavaType.type(ExceptionRegion.class);
    }

    /**
     * The exception region start operation.
     */
    @OpFactory.OpDeclaration(ExceptionRegionEnter.NAME)
    public static final class ExceptionRegionEnter extends CoreOp
            implements Op.BlockTerminating {
        public static final String NAME = "exception.region.enter";

        // First successor is the non-exceptional successor whose target indicates
        // the first block in the exception region.
        // One or more subsequent successors target the exception catching blocks
        // each of which have one block argument whose type is an exception type.
        final List<Block.Reference> s;

        public ExceptionRegionEnter(ExternalizedOp def) {
            super(def);

            if (def.successors().size() < 2) {
                throw new IllegalArgumentException("Operation must have two or more successors" + def.name());
            }

            this.s = List.copyOf(def.successors());
        }

        ExceptionRegionEnter(ExceptionRegionEnter that, CopyContext cc) {
            super(that, cc);

            this.s = that.s.stream().map(cc::getSuccessorOrCreate).toList();
        }

        @Override
        public ExceptionRegionEnter transform(CopyContext cc, OpTransformer ot) {
            return new ExceptionRegionEnter(this, cc);
        }

        ExceptionRegionEnter(List<Block.Reference> s) {
            super(NAME, List.of());

            if (s.size() < 2) {
                throw new IllegalArgumentException("Operation must have two or more successors" + opName());
            }

            this.s = List.copyOf(s);
        }

        @Override
        public List<Block.Reference> successors() {
            return s;
        }

        public Block.Reference start() {
            return s.get(0);
        }

        public List<Block.Reference> catchBlocks() {
            return s.subList(1, s.size());
        }

        @Override
        public TypeElement resultType() {
            return ExceptionRegion.EXCEPTION_REGION_TYPE;
        }
    }

    /**
     * The exception region end operation.
     */
    @OpFactory.OpDeclaration(ExceptionRegionExit.NAME)
    public static final class ExceptionRegionExit extends CoreOp
            implements Op.BlockTerminating {
        public static final String NAME = "exception.region.exit";

        final Block.Reference end;

        public ExceptionRegionExit(ExternalizedOp def) {
            super(def);

            if (def.operands().size() != 1) {
                throw new IllegalArgumentException("Operation must have one operand" + def.name());
            }

            if (def.successors().size() != 1) {
                throw new IllegalArgumentException("Operation must have one successor" + def.name());
            }

            this.end = def.successors().get(0);
        }

        ExceptionRegionExit(ExceptionRegionExit that, CopyContext cc) {
            super(that, cc);

            this.end = cc.getSuccessorOrCreate(that.end);
        }

        @Override
        public ExceptionRegionExit transform(CopyContext cc, OpTransformer ot) {
            return new ExceptionRegionExit(this, cc);
        }

        ExceptionRegionExit(Value exceptionRegion, Block.Reference end) {
            super(NAME, checkValue(exceptionRegion));

            this.end = end;
        }

        static List<Value> checkValue(Value er) {
            if (!(er instanceof Result or && or.op() instanceof ExceptionRegionEnter)) {
                throw new IllegalArgumentException(
                        "Operand not the result of an exception.region.start operation: " + er);
            }

            return List.of(er);
        }

        @Override
        public List<Block.Reference> successors() {
            return List.of(end);
        }

        public Block.Reference end() {
            return end;
        }

        public ExceptionRegionEnter regionStart() {
            if (operands().get(0) instanceof Result or &&
                    or.op() instanceof ExceptionRegionEnter ers) {
                return ers;
            }
            throw new InternalError("Should not reach here");
        }

        @Override
        public TypeElement resultType() {
            return JavaType.VOID;
        }
    }

    /**
     * The String Concatenation Operation
     */

    @OpFactory.OpDeclaration(ConcatOp.NAME)
    public static final class ConcatOp extends CoreOp
            implements Op.Pure, JavaExpression {
        public static final String NAME = "concat";

        public ConcatOp(ConcatOp that, CopyContext cc) {
            super(that, cc);
        }

        public ConcatOp(ExternalizedOp def) {
            super(def);
            if (def.operands().size() != 2) {
                throw new IllegalArgumentException("Concatenation Operation must have two operands.");
            }
        }

        public ConcatOp(Value lhs, Value rhs) {
            super(ConcatOp.NAME, List.of(lhs, rhs));
        }

        @Override
        public Op transform(CopyContext cc, OpTransformer ot) {
            return new ConcatOp(this, cc);
        }

        @Override
        public TypeElement resultType() {
            return JavaType.J_L_STRING;
        }
    }

    //
    // Arithmetic ops

    /**
     * The arithmetic operation.
     */
    public sealed static abstract class ArithmeticOperation extends CoreOp
            implements Op.Pure, JavaExpression {
        protected ArithmeticOperation(ExternalizedOp def) {
            super(def);

            if (def.operands().isEmpty()) {
                throw new IllegalArgumentException("Operation must have one or more operands");
            }
        }

        protected ArithmeticOperation(ArithmeticOperation that, CopyContext cc) {
            super(that, cc);
        }

        protected ArithmeticOperation(String name, List<Value> operands) {
            super(name, operands);
        }
    }

    /**
     * The test operation.
     */
    public sealed static abstract class TestOperation extends CoreOp
            implements Op.Pure, JavaExpression {
        protected TestOperation(ExternalizedOp def) {
            super(def);

            if (def.operands().isEmpty()) {
                throw new IllegalArgumentException("Operation must have one or more operands");
            }
        }

        protected TestOperation(TestOperation that, CopyContext cc) {
            super(that, cc);
        }

        protected TestOperation(String name, List<Value> operands) {
            super(name, operands);
        }
    }

    /**
     * The binary arithmetic operation.
     */
    public sealed static abstract class BinaryOp extends ArithmeticOperation {
        protected BinaryOp(ExternalizedOp def) {
            super(def);

            if (def.operands().size() != 2) {
                throw new IllegalArgumentException("Number of operands must be 2: " + def.operands().size());
            }
        }

        protected BinaryOp(BinaryOp that, CopyContext cc) {
            super(that, cc);
        }

        protected BinaryOp(String name, Value lhs, Value rhs) {
            super(name, List.of(lhs, rhs));
        }

        @Override
        public TypeElement resultType() {
            return operands().get(0).type();
        }
    }

    /**
     * The unary arithmetic operation.
     */
    public sealed static abstract class UnaryOp extends ArithmeticOperation {
        protected UnaryOp(ExternalizedOp def) {
            super(def);

            if (def.operands().size() != 1) {
                throw new IllegalArgumentException("Number of operands must be 1: " + def.operands().size());
            }
        }

        protected UnaryOp(UnaryOp that, CopyContext cc) {
            super(that, cc);
        }

        protected UnaryOp(String name, Value v) {
            super(name, List.of(v));
        }

        @Override
        public TypeElement resultType() {
            return operands().get(0).type();
        }
    }

    /**
     * The binary test operation.
     */
    public sealed static abstract class BinaryTestOp extends TestOperation {
        protected BinaryTestOp(ExternalizedOp def) {
            super(def);

            if (def.operands().size() != 2) {
                throw new IllegalArgumentException("Number of operands must be 2: " + def.operands().size());
            }
        }

        protected BinaryTestOp(BinaryTestOp that, CopyContext cc) {
            super(that, cc);
        }

        protected BinaryTestOp(String name, Value lhs, Value rhs) {
            super(name, List.of(lhs, rhs));
        }

        @Override
        public TypeElement resultType() {
            return JavaType.BOOLEAN;
        }
    }

    /**
     * The add operation, that can model the Java language binary {@code +} operator for numeric types
     */
    @OpFactory.OpDeclaration(AddOp.NAME)
    public static final class AddOp extends BinaryOp {
        public static final String NAME = "add";

        public AddOp(ExternalizedOp def) {
            super(def);
        }

        AddOp(AddOp that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public AddOp transform(CopyContext cc, OpTransformer ot) {
            return new AddOp(this, cc);
        }

        AddOp(Value lhs, Value rhs) {
            super(NAME, lhs, rhs);
        }
    }

    /**
     * The sub operation, that can model the Java language binary {@code -} operator for numeric types
     */
    @OpFactory.OpDeclaration(SubOp.NAME)
    public static final class SubOp extends BinaryOp {
        public static final String NAME = "sub";

        public SubOp(ExternalizedOp opdef) {
            super(opdef);
        }

        SubOp(SubOp that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public SubOp transform(CopyContext cc, OpTransformer ot) {
            return new SubOp(this, cc);
        }

        SubOp(Value lhs, Value rhs) {
            super(NAME, lhs, rhs);
        }
    }

    /**
     * The mul operation, that can model the Java language binary {@code *} operator for numeric types
     */
    @OpFactory.OpDeclaration(MulOp.NAME)
    public static final class MulOp extends BinaryOp {
        public static final String NAME = "mul";

        public MulOp(ExternalizedOp opdef) {
            super(opdef);
        }

        MulOp(MulOp that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public MulOp transform(CopyContext cc, OpTransformer ot) {
            return new MulOp(this, cc);
        }

        MulOp(Value lhs, Value rhs) {
            super(NAME, lhs, rhs);
        }
    }

    /**
     * The div operation, that can model the Java language binary {@code /} operator for numeric types
     */
    @OpFactory.OpDeclaration(DivOp.NAME)
    public static final class DivOp extends BinaryOp {
        public static final String NAME = "div";

        public DivOp(ExternalizedOp opdef) {
            super(opdef);
        }

        DivOp(DivOp that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public DivOp transform(CopyContext cc, OpTransformer ot) {
            return new DivOp(this, cc);
        }

        DivOp(Value lhs, Value rhs) {
            super(NAME, lhs, rhs);
        }
    }

    /**
     * The mod operation, that can model the Java language binary {@code %} operator for numeric types
     */
    @OpFactory.OpDeclaration(ModOp.NAME)
    public static final class ModOp extends BinaryOp {
        public static final String NAME = "mod";

        public ModOp(ExternalizedOp opdef) {
            super(opdef);
        }

        ModOp(ModOp that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public ModOp transform(CopyContext cc, OpTransformer ot) {
            return new ModOp(this, cc);
        }

        ModOp(Value lhs, Value rhs) {
            super(NAME, lhs, rhs);
        }
    }

    /**
     * The bitwise/logical or operation, that can model the Java language binary {@code |} operator for integral types
     * and booleans
     */
    @OpFactory.OpDeclaration(OrOp.NAME)
    public static final class OrOp extends BinaryOp {
        public static final String NAME = "or";

        public OrOp(ExternalizedOp opdef) {
            super(opdef);
        }

        OrOp(OrOp that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public OrOp transform(CopyContext cc, OpTransformer ot) {
            return new OrOp(this, cc);
        }

        OrOp(Value lhs, Value rhs) {
            super(NAME, lhs, rhs);
        }
    }

    /**
     * The bitwise/logical and operation, that can model the Java language binary {@code &} operator for integral types
     * and booleans
     */
    @OpFactory.OpDeclaration(AndOp.NAME)
    public static final class AndOp extends BinaryOp {
        public static final String NAME = "and";

        public AndOp(ExternalizedOp opdef) {
            super(opdef);
        }

        AndOp(AndOp that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public AndOp transform(CopyContext cc, OpTransformer ot) {
            return new AndOp(this, cc);
        }

        AndOp(Value lhs, Value rhs) {
            super(NAME, lhs, rhs);
        }
    }

    /**
     * The xor operation, that can model the Java language binary {@code ^} operator for integral types
     * and booleans
     */
    @OpFactory.OpDeclaration(XorOp.NAME)
    public static final class XorOp extends BinaryOp {
        public static final String NAME = "xor";

        public XorOp(ExternalizedOp opdef) {
            super(opdef);
        }

        XorOp(XorOp that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public XorOp transform(CopyContext cc, OpTransformer ot) {
            return new XorOp(this, cc);
        }

        XorOp(Value lhs, Value rhs) {
            super(NAME, lhs, rhs);
        }
    }

    /**
     * The (logical) shift left operation, that can model the Java language binary {@code <<} operator for integral types
     */
    @OpFactory.OpDeclaration(LshlOp.NAME)
    public static final class LshlOp extends BinaryOp {
        public static final String NAME = "lshl";

        public LshlOp(ExternalizedOp opdef) {
            super(opdef);
        }

        LshlOp(LshlOp that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public LshlOp transform(CopyContext cc, OpTransformer ot) {
            return new LshlOp(this, cc);
        }

        LshlOp(Value lhs, Value rhs) {
            super(NAME, lhs, rhs);
        }
    }

    /**
     * The (arithmetic) shift right operation, that can model the Java language binary {@code >>} operator for integral types
     */
    @OpFactory.OpDeclaration(AshrOp.NAME)
    public static final class AshrOp extends CoreOp.BinaryOp {
        public static final String NAME = "ashr";

        public AshrOp(ExternalizedOp opdef) {
            super(opdef);
        }

        AshrOp(AshrOp that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public AshrOp transform(CopyContext cc, OpTransformer ot) {
            return new AshrOp(this, cc);
        }

        AshrOp(Value lhs, Value rhs) {
            super(NAME, lhs, rhs);
        }
    }

    /**
     * The unsigned (logical) shift right operation, that can model the Java language binary {@code >>>} operator for integral types
     */
    @OpFactory.OpDeclaration(LshrOp.NAME)
    public static final class LshrOp extends CoreOp.BinaryOp {
        public static final String NAME = "lshr";

        public LshrOp(ExternalizedOp opdef) {
            super(opdef);
        }

        LshrOp(LshrOp that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public LshrOp transform(CopyContext cc, OpTransformer ot) {
            return new LshrOp(this, cc);
        }

        LshrOp(Value lhs, Value rhs) {
            super(NAME, lhs, rhs);
        }
    }

    /**
     * The neg operation, that can model the Java language unary {@code -} operator for numeric types
     */
    @OpFactory.OpDeclaration(NegOp.NAME)
    public static final class NegOp extends UnaryOp {
        public static final String NAME = "neg";

        public NegOp(ExternalizedOp opdef) {
            super(opdef);
        }

        NegOp(NegOp that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public NegOp transform(CopyContext cc, OpTransformer ot) {
            return new NegOp(this, cc);
        }

        NegOp(Value v) {
            super(NAME, v);
        }
    }

    /**
     * The not operation, that can model the Java language unary {@code !} operator for boolean types
     */
    @OpFactory.OpDeclaration(NotOp.NAME)
    public static final class NotOp extends UnaryOp {
        public static final String NAME = "not";

        public NotOp(ExternalizedOp opdef) {
            super(opdef);
        }

        NotOp(NotOp that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public NotOp transform(CopyContext cc, OpTransformer ot) {
            return new NotOp(this, cc);
        }

        NotOp(Value v) {
            super(NAME, v);
        }
    }

    /**
     * The equals operation, that can model the Java language equality {@code ==} operator for numeric, boolean
     * and reference types
     */
    @OpFactory.OpDeclaration(EqOp.NAME)
    public static final class EqOp extends BinaryTestOp {
        public static final String NAME = "eq";

        public EqOp(ExternalizedOp opdef) {
            super(opdef);
        }

        EqOp(EqOp that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public EqOp transform(CopyContext cc, OpTransformer ot) {
            return new EqOp(this, cc);
        }

        EqOp(Value lhs, Value rhs) {
            super(NAME, lhs, rhs);
        }
    }

    /**
     * The not equals operation, that can model the Java language equality {@code !=} operator for numeric, boolean
     * and reference types
     */
    @OpFactory.OpDeclaration(NeqOp.NAME)
    public static final class NeqOp extends BinaryTestOp {
        public static final String NAME = "neq";

        public NeqOp(ExternalizedOp opdef) {
            super(opdef);
        }

        NeqOp(NeqOp that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public NeqOp transform(CopyContext cc, OpTransformer ot) {
            return new NeqOp(this, cc);
        }

        NeqOp(Value lhs, Value rhs) {
            super(NAME, lhs, rhs);
        }
    }

    /**
     * The greater than operation, that can model the Java language relational {@code >} operator for numeric types
     */
    @OpFactory.OpDeclaration(GtOp.NAME)
    public static final class GtOp extends BinaryTestOp {
        public static final String NAME = "gt";

        public GtOp(ExternalizedOp opdef) {
            super(opdef);
        }

        GtOp(GtOp that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public GtOp transform(CopyContext cc, OpTransformer ot) {
            return new GtOp(this, cc);
        }

        GtOp(Value lhs, Value rhs) {
            super(NAME, lhs, rhs);
        }
    }

    /**
     * The greater than or equal to operation, that can model the Java language relational {@code >=} operator for
     * numeric types
     */
    @OpFactory.OpDeclaration(GeOp.NAME)
    public static final class GeOp extends BinaryTestOp {
        public static final String NAME = "ge";

        public GeOp(ExternalizedOp opdef) {
            super(opdef);
        }

        GeOp(GeOp that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public GeOp transform(CopyContext cc, OpTransformer ot) {
            return new GeOp(this, cc);
        }

        GeOp(Value lhs, Value rhs) {
            super(NAME, lhs, rhs);
        }
    }

    /**
     * The less than operation, that can model the Java language relational {@code <} operator for
     * numeric types
     */
    @OpFactory.OpDeclaration(LtOp.NAME)
    public static final class LtOp extends BinaryTestOp {
        public static final String NAME = "lt";

        public LtOp(ExternalizedOp opdef) {
            super(opdef);
        }

        LtOp(LtOp that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public LtOp transform(CopyContext cc, OpTransformer ot) {
            return new LtOp(this, cc);
        }

        LtOp(Value lhs, Value rhs) {
            super(NAME, lhs, rhs);
        }
    }

    /**
     * The less than or equal to operation, that can model the Java language relational {@code <=} operator for
     * numeric types
     */
    @OpFactory.OpDeclaration(LeOp.NAME)
    public static final class LeOp extends BinaryTestOp {
        public static final String NAME = "le";

        public LeOp(ExternalizedOp opdef) {
            super(opdef);
        }

        LeOp(LeOp that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public LeOp transform(CopyContext cc, OpTransformer ot) {
            return new LeOp(this, cc);
        }

        LeOp(Value lhs, Value rhs) {
            super(NAME, lhs, rhs);
        }
    }


    /**
     * A factory for core operations.
     */
    // @@@ Compute lazily
    public static final OpFactory FACTORY = OpFactory.OP_FACTORY.get(CoreOp.class);

    /**
     * Creates a function operation builder
     *
     * @param funcName the function name
     * @param funcType the function type
     * @return the function operation builder
     */
    public static FuncOp.Builder func(String funcName, FunctionType funcType) {
        return new FuncOp.Builder(null, funcName, funcType);
    }

    /**
     * Creates a function operation
     *
     * @param funcName the function name
     * @param body     the function body
     * @return the function operation
     */
    public static FuncOp func(String funcName, Body.Builder body) {
        return new FuncOp(funcName, body);
    }

    /**
     * Creates a function call operation
     *
     * @param funcName the name of the function operation
     * @param funcType the function type
     * @param args     the function arguments
     * @return the function call operation
     */
    public static FuncCallOp funcCall(String funcName, FunctionType funcType, Value... args) {
        return funcCall(funcName, funcType, List.of(args));
    }

    /**
     * Creates a function call operation
     *
     * @param funcName the name of the function operation
     * @param funcType the function type
     * @param args     the function arguments
     * @return the function call operation
     */
    public static FuncCallOp funcCall(String funcName, FunctionType funcType, List<Value> args) {
        return new FuncCallOp(funcName, funcType.returnType(), args);
    }

    /**
     * Creates a function call operation
     *
     * @param func the target function
     * @param args the function arguments
     * @return the function call operation
     */
    public static FuncCallOp funcCall(FuncOp func, Value... args) {
        return funcCall(func, List.of(args));
    }

    /**
     * Creates a function call operation
     *
     * @param func the target function
     * @param args the function argments
     * @return the function call operation
     */
    public static FuncCallOp funcCall(FuncOp func, List<Value> args) {
        return new FuncCallOp(func.funcName(), func.invokableType().returnType(), args);
    }

    /**
     * Creates a module operation.
     *
     * @param functions the functions of the module operation
     * @return the module operation
     */
    public static ModuleOp module(FuncOp... functions) {
        return module(List.of(functions));
    }

    /**
     * Creates a module operation.
     *
     * @param functions the functions of the module operation
     * @return the module operation
     */
    public static ModuleOp module(List<FuncOp> functions) {
        return new ModuleOp(List.copyOf(functions));
    }

    /**
     * Creates a quoted operation.
     *
     * @param ancestorBody the ancestor of the body of the quoted operation
     * @param opFunc       a function that accepts the body of the quoted operation and returns the operation to be quoted
     * @return the quoted operation
     */
    public static QuotedOp quoted(Body.Builder ancestorBody,
                                  Function<Block.Builder, Op> opFunc) {
        Body.Builder body = Body.Builder.of(ancestorBody, FunctionType.VOID);
        Block.Builder block = body.entryBlock();
        block.op(_yield(
                block.op(opFunc.apply(block))));
        return new QuotedOp(body);
    }

    /**
     * Creates a quoted operation.
     *
     * @param body quoted operation body
     * @return the quoted operation
     */
    public static QuotedOp quoted(Body.Builder body) {
        return new QuotedOp(body);
    }

    /**
     * Creates a lambda operation.
     *
     * @param ancestorBody        the ancestor of the body of the lambda operation
     * @param funcType            the lambda operation's function type
     * @param functionalInterface the lambda operation's functional interface type
     * @return the lambda operation
     */
    public static LambdaOp.Builder lambda(Body.Builder ancestorBody,
                                          FunctionType funcType, TypeElement functionalInterface) {
        return new LambdaOp.Builder(ancestorBody, funcType, functionalInterface);
    }

    /**
     * Creates a lambda operation.
     *
     * @param functionalInterface the lambda operation's functional interface type
     * @param body                the body of the lambda operation
     * @return the lambda operation
     */
    public static LambdaOp lambda(TypeElement functionalInterface, Body.Builder body) {
        return new LambdaOp(functionalInterface, body);
    }

    /**
     * Creates a closure operation.
     *
     * @param ancestorBody the ancestor of the body of the closure operation
     * @param funcType     the closure operation's function type
     * @return the closure operation
     */
    public static ClosureOp.Builder closure(Body.Builder ancestorBody,
                                            FunctionType funcType) {
        return new ClosureOp.Builder(ancestorBody, funcType);
    }

    /**
     * Creates a closure operation.
     *
     * @param body the body of the closure operation
     * @return the closure operation
     */
    public static ClosureOp closure(Body.Builder body) {
        return new ClosureOp(body);
    }

    /**
     * Creates a closure call operation.
     *
     * @param args the closure arguments. The first argument is the closure operation to be called
     * @return the closure call operation
     */
    // @@@: Is this the right signature?
    public static ClosureCallOp closureCall(Value... args) {
        return closureCall(List.of(args));
    }

    /**
     * Creates a closure call operation.
     *
     * @param args the closure arguments. The first argument is the closure operation to be called
     * @return the closure call operation
     */
    // @@@: Is this the right signature?
    public static ClosureCallOp closureCall(List<Value> args) {
        return new ClosureCallOp(args);
    }

    /**
     * Creates an exception region enter operation
     *
     * @param start    the exception region block
     * @param catchers the blocks handling exceptions thrown by the region block
     * @return the exception region enter operation
     */
    public static ExceptionRegionEnter exceptionRegionEnter(Block.Reference start, Block.Reference... catchers) {
        return exceptionRegionEnter(start, List.of(catchers));
    }

    /**
     * Creates an exception region enter operation
     *
     * @param start    the exception region block
     * @param catchers the blocks handling exceptions thrown by the region block
     * @return the exception region enter operation
     */
    public static ExceptionRegionEnter exceptionRegionEnter(Block.Reference start, List<Block.Reference> catchers) {
        List<Block.Reference> s = new ArrayList<>();
        s.add(start);
        s.addAll(catchers);
        return new ExceptionRegionEnter(s);
    }

    /**
     * Creates an exception region exit operation
     *
     * @param exceptionRegion the exception region to be exited
     * @param end             the block to which control is transferred after the exception region is exited
     * @return the exception region exit operation
     */
    public static ExceptionRegionExit exceptionRegionExit(Value exceptionRegion, Block.Reference end) {
        return new ExceptionRegionExit(exceptionRegion, end);
    }

    /**
     * Creates a return operation.
     *
     * @return the return operation
     */
    public static ReturnOp _return() {
        return new ReturnOp();
    }

    /**
     * Creates a return operation.
     *
     * @param returnValue the return value
     * @return the return operation
     */
    public static ReturnOp _return(Value returnValue) {
        return new ReturnOp(returnValue);
    }

    /**
     * Creates a throw operation.
     *
     * @param exceptionValue the thrown value
     * @return the throw operation
     */
    public static ThrowOp _throw(Value exceptionValue) {
        return new ThrowOp(exceptionValue);
    }

    /**
     * Creates an unreachable operation.
     *
     * @return the unreachable operation
     */
    public static UnreachableOp unreachable() {
        return new UnreachableOp();
    }

    /**
     * Creates a yield operation.
     *
     * @return the yield operation
     */
    public static YieldOp _yield() {
        return new YieldOp();
    }

    /**
     * Creates a yield operation.
     *
     * @param yieldValue the yielded value
     * @return the yield operation
     */
    public static YieldOp _yield(Value yieldValue) {
        return new YieldOp(List.of(yieldValue));
    }

    /**
     * Creates an assert operation.
     *
     * @param bodies the nested bodies
     * @return the assert operation
     */
    public static AssertOp _assert(List<Body.Builder> bodies) {
        return new AssertOp(bodies);
    }

    /**
     * Creates an unconditional break operation.
     *
     * @param target the jump target
     * @return the unconditional break operation
     */
    public static BranchOp branch(Block.Reference target) {
        return new BranchOp(target);
    }

    /**
     * Creates a conditional break operation.
     *
     * @param condValue   the test value of the conditional break operation
     * @param trueTarget  the jump target when the test value evaluates to true
     * @param falseTarget the jump target when the test value evaluates to false
     * @return the conditional break operation
     */
    public static ConditionalBranchOp conditionalBranch(Value condValue,
                                                        Block.Reference trueTarget, Block.Reference falseTarget) {
        return new ConditionalBranchOp(condValue, trueTarget, falseTarget);
    }

    /**
     * Creates a constant operation.
     *
     * @param type  the constant type
     * @param value the constant value
     * @return the constant operation
     */
    public static ConstantOp constant(TypeElement type, Object value) {
        return new ConstantOp(type, value);
    }

    /**
     * Creates an invoke operation.
     *
     * @param invokeDescriptor the invocation descriptor
     * @param args             the invoke parameters
     * @return the invoke operation
     */
    public static InvokeOp invoke(MethodRef invokeDescriptor, Value... args) {
        return new InvokeOp(invokeDescriptor, List.of(args));
    }

    /**
     * Creates an invoke operation.
     *
     * @param invokeDescriptor the invocation descriptor
     * @param args             the invoke parameters
     * @return the invoke operation
     */
    public static InvokeOp invoke(MethodRef invokeDescriptor, List<Value> args) {
        return new InvokeOp(invokeDescriptor, args);
    }

    /**
     * Creates an invoke operation.
     *
     * @param returnType       the invocation return type
     * @param invokeDescriptor the invocation descriptor
     * @param args             the invoke parameters
     * @return the invoke operation
     */
    public static InvokeOp invoke(TypeElement returnType, MethodRef invokeDescriptor, Value... args) {
        return new InvokeOp(returnType, invokeDescriptor, List.of(args));
    }

    /**
     * Creates an invoke operation.
     *
     * @param returnType       the invocation return type
     * @param invokeDescriptor the invocation descriptor
     * @param args             the invoke parameters
     * @return the invoke operation
     */
    public static InvokeOp invoke(TypeElement returnType, MethodRef invokeDescriptor, List<Value> args) {
        return new InvokeOp(returnType, invokeDescriptor, args);
    }

    /**
     * Creates a conversion operation.
     *
     * @param to   the conversion target type
     * @param from the value to be converted
     * @return the conversion operation
     */
    public static ConvOp conv(TypeElement to, Value from) {
        return new ConvOp(to, from);
    }

    /**
     * Creates an instance creation operation.
     *
     * @param constructorType the constructor type
     * @param args            the constructor arguments
     * @return the instance creation operation
     */
    public static NewOp _new(FunctionType constructorType, Value... args) {
        return _new(constructorType, List.of(args));
    }

    /**
     * Creates an instance creation operation.
     *
     * @param constructorType the constructor type
     * @param args            the constructor arguments
     * @return the instance creation operation
     */
    public static NewOp _new(FunctionType constructorType, List<Value> args) {
        return new NewOp(constructorType, args);
    }

    /**
     * Creates an instance creation operation.
     *
     * @param returnType      the instance type
     * @param constructorType the constructor type
     * @param args            the constructor arguments
     * @return the instance creation operation
     */
    public static NewOp _new(TypeElement returnType, FunctionType constructorType,
                             Value... args) {
        return _new(returnType, constructorType, List.of(args));
    }

    /**
     * Creates an instance creation operation.
     *
     * @param returnType      the instance type
     * @param constructorType the constructor type
     * @param args            the constructor arguments
     * @return the instance creation operation
     */
    public static NewOp _new(TypeElement returnType, FunctionType constructorType,
                             List<Value> args) {
        return new NewOp(returnType, constructorType, args);
    }

    /**
     * Creates an array creation operation.
     *
     * @param arrayType the array type
     * @param length    the array size
     * @return the array creation operation
     */
    public static NewOp newArray(TypeElement arrayType, Value length) {
        return _new(FunctionType.functionType(arrayType, JavaType.INT), length);
    }

    // @@@ Add field load/store overload with explicit fieldType

    /**
     * Creates a field load operation to a non-static field.
     *
     * @param descriptor the field descriptor
     * @param receiver   the receiver value
     * @return the field load operation
     */
    public static FieldAccessOp.FieldLoadOp fieldLoad(FieldRef descriptor, Value receiver) {
        return new FieldAccessOp.FieldLoadOp(descriptor.type(), descriptor, receiver);
    }

    /**
     * Creates a field load operation to a non-static field.
     *
     * @param resultType the result type of the operation
     * @param descriptor the field descriptor
     * @param receiver   the receiver value
     * @return the field load operation
     */
    public static FieldAccessOp.FieldLoadOp fieldLoad(TypeElement resultType, FieldRef descriptor, Value receiver) {
        return new FieldAccessOp.FieldLoadOp(resultType, descriptor, receiver);
    }

    /**
     * Creates a field load operation to a static field.
     *
     * @param descriptor the field descriptor
     * @return the field load operation
     */
    public static FieldAccessOp.FieldLoadOp fieldLoad(FieldRef descriptor) {
        return new FieldAccessOp.FieldLoadOp(descriptor.type(), descriptor);
    }

    /**
     * Creates a field load operation to a static field.
     *
     * @param resultType the result type of the operation
     * @param descriptor the field descriptor
     * @return the field load operation
     */
    public static FieldAccessOp.FieldLoadOp fieldLoad(TypeElement resultType, FieldRef descriptor) {
        return new FieldAccessOp.FieldLoadOp(resultType, descriptor);
    }

    /**
     * Creates a field store operation to a non-static field.
     *
     * @param descriptor the field descriptor
     * @param receiver   the receiver value
     * @param v          the value to store
     * @return the field store operation
     */
    public static FieldAccessOp.FieldStoreOp fieldStore(FieldRef descriptor, Value receiver, Value v) {
        return new FieldAccessOp.FieldStoreOp(descriptor, receiver, v);
    }

    /**
     * Creates a field load operation to a static field.
     *
     * @param descriptor the field descriptor
     * @param v          the value to store
     * @return the field store operation
     */
    public static FieldAccessOp.FieldStoreOp fieldStore(FieldRef descriptor, Value v) {
        return new FieldAccessOp.FieldStoreOp(descriptor, v);
    }

    /**
     * Creates an array length operation.
     *
     * @param array the array value
     * @return the array length operation
     */
    public static ArrayLengthOp arrayLength(Value array) {
        return new ArrayLengthOp(array);
    }

    /**
     * Creates an array load operation.
     *
     * @param array the array value
     * @param index the index value
     * @return the array load operation
     */
    public static ArrayAccessOp.ArrayLoadOp arrayLoadOp(Value array, Value index) {
        return new ArrayAccessOp.ArrayLoadOp(array, index);
    }

    /**
     * Creates an array store operation.
     *
     * @param array the array value
     * @param index the index value
     * @param v     the value to store
     * @return the array store operation
     */
    public static ArrayAccessOp.ArrayStoreOp arrayStoreOp(Value array, Value index, Value v) {
        return new ArrayAccessOp.ArrayStoreOp(array, index, v);
    }

    /**
     * Creates an instanceof operation.
     *
     * @param t the type to test against
     * @param v the value to test
     * @return the instanceof operation
     */
    public static InstanceOfOp instanceOf(TypeElement t, Value v) {
        return new InstanceOfOp(t, v);
    }

    /**
     * Creates a cast operation.
     *
     * @param resultType the result type of the operation
     * @param v          the value to cast
     * @return the cast operation
     */
    public static CastOp cast(TypeElement resultType, Value v) {
        return new CastOp(resultType, resultType, v);
    }

    /**
     * Creates a cast operation.
     *
     * @param resultType the result type of the operation
     * @param t          the type to cast to
     * @param v          the value to cast
     * @return the cast operation
     */
    public static CastOp cast(TypeElement resultType, JavaType t, Value v) {
        return new CastOp(resultType, t, v);
    }

    /**
     * Creates a var operation.
     *
     * @param init the initial value of the var
     * @return the var operation
     */
    public static VarOp var(Value init) {
        return var(null, init);
    }

    /**
     * Creates a var operation.
     *
     * @param name the name of the var
     * @param init the initial value of the var
     * @return the var operation
     */
    public static VarOp var(String name, Value init) {
        return new VarOp(name, init);
    }

    /**
     * Creates a var operation.
     *
     * @param name the name of the var
     * @param type the type of the var's value
     * @param init the initial value of the var
     * @return the var operation
     */
    public static VarOp var(String name, TypeElement type, Value init) {
        return new VarOp(name, type, init);
    }

    /**
     * Creates a var load operation.
     *
     * @param varValue the var value
     * @return the var load operation
     */
    public static VarAccessOp.VarLoadOp varLoad(Value varValue) {
        return new VarAccessOp.VarLoadOp(varValue);
    }

    /**
     * Creates a var store operation.
     *
     * @param varValue the var value
     * @param v        the value to store in the var
     * @return the var store operation
     */
    public static VarAccessOp.VarStoreOp varStore(Value varValue, Value v) {
        return new VarAccessOp.VarStoreOp(varValue, v);
    }

    /**
     * Creates a tuple operation.
     *
     * @param componentValues the values of tuple (in order)
     * @return the tuple operation
     */
    public static TupleOp tuple(Value... componentValues) {
        return tuple(List.of(componentValues));
    }

    /**
     * Creates a tuple operation.
     *
     * @param componentValues the values of tuple (in order)
     * @return the tuple operation
     */
    public static TupleOp tuple(List<? extends Value> componentValues) {
        return new TupleOp(componentValues);
    }

    /**
     * Creates a tuple load operation.
     *
     * @param tuple the tuple value
     * @param index the component index value
     * @return the tuple load operation
     */
    public static TupleLoadOp tupleLoad(Value tuple, int index) {
        return new TupleLoadOp(tuple, index);
    }

    /**
     * Creates a tuple with operation.
     *
     * @param tuple the tuple value
     * @param index the component index value
     * @param value the component value
     * @return the tuple with operation
     */
    public static TupleWithOp tupleWith(Value tuple, int index, Value value) {
        return new TupleWithOp(tuple, index, value);
    }

    //
    // Arithmetic ops

    /**
     * Creates an add operation.
     *
     * @param lhs the first operand
     * @param rhs the second operand
     * @return the add operation
     */
    public static BinaryOp add(Value lhs, Value rhs) {
        return new AddOp(lhs, rhs);
    }

    /**
     * Creates a sub operation.
     *
     * @param lhs the first operand
     * @param rhs the second operand
     * @return the sub operation
     */
    public static BinaryOp sub(Value lhs, Value rhs) {
        return new SubOp(lhs, rhs);
    }

    /**
     * Creates a mul operation.
     *
     * @param lhs the first operand
     * @param rhs the second operand
     * @return the mul operation
     */
    public static BinaryOp mul(Value lhs, Value rhs) {
        return new MulOp(lhs, rhs);
    }

    /**
     * Creates a div operation.
     *
     * @param lhs the first operand
     * @param rhs the second operand
     * @return the div operation
     */
    public static BinaryOp div(Value lhs, Value rhs) {
        return new DivOp(lhs, rhs);
    }

    /**
     * Creates a mod operation.
     *
     * @param lhs the first operand
     * @param rhs the second operand
     * @return the mod operation
     */
    public static BinaryOp mod(Value lhs, Value rhs) {
        return new ModOp(lhs, rhs);
    }

    /**
     * Creates a bitwise/logical or operation.
     *
     * @param lhs the first operand
     * @param rhs the second operand
     * @return the or operation
     */
    public static BinaryOp or(Value lhs, Value rhs) {
        return new OrOp(lhs, rhs);
    }

    /**
     * Creates a bitwise/logical and operation.
     *
     * @param lhs the first operand
     * @param rhs the second operand
     * @return the and operation
     */
    public static BinaryOp and(Value lhs, Value rhs) {
        return new AndOp(lhs, rhs);
    }

    /**
     * Creates a bitwise/logical xor operation.
     *
     * @param lhs the first operand
     * @param rhs the second operand
     * @return the xor operation
     */
    public static BinaryOp xor(Value lhs, Value rhs) {
        return new XorOp(lhs, rhs);
    }

    /**
     * Creates a left shift operation.
     *
     * @param lhs the first operand
     * @param rhs the second operand
     * @return the xor operation
     */
    public static BinaryOp lshl(Value lhs, Value rhs) {
        return new LshlOp(lhs, rhs);
    }

    /**
     * Creates a right shift operation.
     *
     * @param lhs the first operand
     * @param rhs the second operand
     * @return the xor operation
     */
    public static BinaryOp ashr(Value lhs, Value rhs) {
        return new AshrOp(lhs, rhs);
    }

    /**
     * Creates an unsigned right shift operation.
     *
     * @param lhs the first operand
     * @param rhs the second operand
     * @return the xor operation
     */
    public static BinaryOp lshr(Value lhs, Value rhs) {
        return new LshrOp(lhs, rhs);
    }

    /**
     * Creates a neg operation.
     *
     * @param v the operand
     * @return the neg operation
     */
    public static UnaryOp neg(Value v) {
        return new NegOp(v);
    }

    /**
     * Creates a not operation.
     *
     * @param v the operand
     * @return the not operation
     */
    public static UnaryOp not(Value v) {
        return new NotOp(v);
    }


    /**
     * Creates an equals comparison operation.
     *
     * @param lhs the first operand
     * @param rhs the second operand
     * @return the equals comparison operation
     */
    public static BinaryTestOp eq(Value lhs, Value rhs) {
        return new EqOp(lhs, rhs);
    }

    /**
     * Creates a not equals comparison operation.
     *
     * @param lhs the first operand
     * @param rhs the second operand
     * @return the not equals comparison operation
     */
    public static BinaryTestOp neq(Value lhs, Value rhs) {
        return new NeqOp(lhs, rhs);
    }

    /**
     * Creates a greater than comparison operation.
     *
     * @param lhs the first operand
     * @param rhs the second operand
     * @return the greater than comparison operation
     */
    public static BinaryTestOp gt(Value lhs, Value rhs) {
        return new GtOp(lhs, rhs);
    }

    /**
     * Creates a greater than or equals to comparison operation.
     *
     * @param lhs the first operand
     * @param rhs the second operand
     * @return the greater than or equals to comparison operation
     */
    public static BinaryTestOp ge(Value lhs, Value rhs) {
        return new GeOp(lhs, rhs);
    }

    /**
     * Creates a less than comparison operation.
     *
     * @param lhs the first operand
     * @param rhs the second operand
     * @return the less than comparison operation
     */
    public static BinaryTestOp lt(Value lhs, Value rhs) {
        return new LtOp(lhs, rhs);
    }

    /**
     * Creates a less than or equals to comparison operation.
     *
     * @param lhs the first operand
     * @param rhs the second operand
     * @return the less than or equals to comparison operation
     */
    public static BinaryTestOp le(Value lhs, Value rhs) {
        return new LeOp(lhs, rhs);
    }

    /**
     * Creates a string concatenation operation.
     *
     * @param lhs the first operand
     * @param rhs the second operand
     * @return the string concatenation operation
     */
    public static ConcatOp concat(Value lhs, Value rhs) {
        return new ConcatOp(lhs, rhs);
    }
}

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

package jdk.incubator.code.dialect.java;

import java.lang.constant.ClassDesc;
import jdk.incubator.code.*;
import jdk.incubator.code.extern.DialectFactory;
import jdk.incubator.code.dialect.core.*;
import jdk.incubator.code.extern.ExternalizedOp;
import jdk.incubator.code.extern.OpFactory;
import jdk.incubator.code.internal.OpDeclaration;

import java.util.*;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.function.Predicate;
import java.util.stream.Stream;

import static jdk.incubator.code.dialect.core.CoreOp.*;
import static jdk.incubator.code.dialect.java.JavaType.*;

/**
 * The top-level operation class for Java operations.
 * <p>
 * A code model, produced by the Java compiler from Java program source, may consist of core operations and Java
 * operations. Such a model represents the same Java program and preserves the program meaning as defined by the
 * Java Language Specification
 * <p>
 * Java operations model specific Java language constructs or Java program behaviour. Some Java operations model
 * structured control flow and nested code. These operations are transformable, commonly referred to as lowering, into
 * a sequence of other core or Java operations. Those that implement {@link Op.Lowerable} can transform themselves and
 * will transform associated operations that are not explicitly lowerable.
 * <p>
 * A code model, produced by the Java compiler from source, and consisting of core operations and Java operations
 * can be transformed to one consisting only of non-lowerable operations, where all lowerable operations are lowered.
 * This transformation preserves programing meaning. The resulting lowered code model also represents the same Java
 * program.
 */
public sealed abstract class JavaOp extends Op {

    protected JavaOp(Op that, CopyContext cc) {
        super(that, cc);
    }

    protected JavaOp(String name, List<? extends Value> operands) {
        super(name, operands);
    }

    /**
     * An operation that models a Java expression
     */
    public sealed interface JavaExpression permits
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
            ConditionalExpressionOp,
            JavaConditionalOp,
            SwitchExpressionOp {
    }

    /**
     * An operation that models a Java statement
     */
    public sealed interface JavaStatement permits
            ArrayAccessOp.ArrayStoreOp,
            AssertOp,
            FieldAccessOp.FieldStoreOp,
            InvokeOp,
            NewOp,
            ReturnOp,
            ThrowOp,
            VarAccessOp.VarStoreOp,
            VarOp,
            BlockOp,
            DoWhileOp,
            EnhancedForOp,
            ForOp,
            IfOp,
            JavaLabelOp,
            LabeledOp,
            SynchronizedOp,
            TryOp,
            WhileOp,
            YieldOp,
            SwitchStatementOp {
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
     * An operation that performs access.
     */
    public sealed interface AccessOp permits
        CoreOp.VarAccessOp,
        FieldAccessOp,
        ArrayAccessOp {
    }



    /**
     * The lambda operation, that can model a Java lambda expression.
     */
    @OpDeclaration(LambdaOp.NAME)
    public static final class LambdaOp extends JavaOp
            implements Invokable, Lowerable, JavaExpression {

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

        static final String NAME = "lambda";

        final TypeElement functionalInterface;
        final Body body;

        LambdaOp(ExternalizedOp def) {
            this(def.resultType(), def.bodyDefinitions().get(0));
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
            InvokeOp methodRefInvokeOp = extractMethodInvoke(valueMapping, body().entryBlock().ops());
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

        static InvokeOp extractMethodInvoke(Map<Value, Value> valueMapping, List<Op> ops) {
            InvokeOp methodRefInvokeOp = null;
            for (Op op : ops) {
                switch (op) {
                    case VarOp varOp -> {
                        if (isValueUsedWithOp(varOp.result(), o -> o instanceof VarAccessOp.VarStoreOp)) {
                            return null;
                        }
                    }
                    case VarAccessOp.VarLoadOp varLoadOp -> {
                        Value v = varLoadOp.varOp().operands().getFirst();
                        valueMapping.put(varLoadOp.result(), valueMapping.getOrDefault(v, v));
                    }
                    case InvokeOp iop when isBoxOrUnboxInvocation(iop) -> {
                        Value v = iop.operands().getFirst();
                        valueMapping.put(iop.result(), valueMapping.getOrDefault(v, v));
                    }
                    case InvokeOp iop -> {
                        if (methodRefInvokeOp != null) {
                            return null;
                        }

                        for (Value o : iop.operands()) {
                            valueMapping.put(o, valueMapping.getOrDefault(o, o));
                        }
                        methodRefInvokeOp = iop;
                    }
                    case ReturnOp rop -> {
                        if (methodRefInvokeOp == null) {
                            return null;
                        }
                        Value r = rop.returnValue();
                        if (!(valueMapping.getOrDefault(r, r) instanceof Result invokeResult)) {
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
            for (Result user : value.uses()) {
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

        private static boolean isBoxOrUnboxInvocation(InvokeOp iop) {
            MethodRef mr = iop.invokeDescriptor();
            return mr.refType() instanceof ClassType ct && ct.unbox().isPresent() &&
                    (UNBOX_NAMES.contains(mr.name()) || mr.name().equals("valueOf"));
        }
    }

    /**
     * The terminating throw operation, that can model the Java language throw statement.
     */
    @OpDeclaration(ThrowOp.NAME)
    public static final class ThrowOp extends JavaOp
            implements BodyTerminating, JavaStatement {
        static final String NAME = "throw";

        ThrowOp(ExternalizedOp def) {
            if (def.operands().size() != 1) {
                throw new IllegalArgumentException("Operation must have one operand " + def.name());
            }

            this(def.operands().get(0));
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
            return VOID;
        }
    }

    /**
     * The assertion operation. Supporting assertions in statement form.
     */
    @OpDeclaration(AssertOp.NAME)
    public static final class AssertOp extends JavaOp
            implements Nested, JavaStatement {
        static final String NAME = "assert";
        public final List<Body> bodies;

        AssertOp(ExternalizedOp def) {
            this(def.bodyDefinitions());
        }

        public AssertOp(List<Body.Builder> bodies) {
            super(NAME, List.of());

            if (bodies.size() != 1 && bodies.size() != 2) {
                throw new IllegalArgumentException("Assert must have one or two bodies.");
            }
            this.bodies = bodies.stream().map(b -> b.build(this)).toList();
        }

        AssertOp(AssertOp that, CopyContext cc, OpTransformer ot) {
            super(that, cc);
            this.bodies = that.bodies.stream().map(b -> b.transform(cc, ot).build(this)).toList();
        }

        @Override
        public Op transform(CopyContext cc, OpTransformer ot) {
            return new AssertOp(this, cc, ot);
        }

        @Override
        public TypeElement resultType() {
            return VOID;
        }

        @Override
        public List<Body> bodies() {
            return this.bodies;
        }
    }

    /**
     * A monitor operation.
     */
    public sealed abstract static class MonitorOp extends JavaOp {
        MonitorOp(MonitorOp that, CopyContext cc) {
            super(that, cc);
        }

        MonitorOp(String name, Value monitor) {
            super(name, List.of(monitor));
        }

        public Value monitorValue() {
            return operands().getFirst();
        }

        @Override
        public TypeElement resultType() {
            return VOID;
        }

        /**
         * The monitor enter operation.
         */
        @OpDeclaration(MonitorEnterOp.NAME)
        public static final class MonitorEnterOp extends MonitorOp {
            static final String NAME = "monitor.enter";

            MonitorEnterOp(ExternalizedOp def) {
                if (def.operands().size() != 1) {
                    throw new IllegalArgumentException("Operation must have one operand " + def.name());
                }

                this(def.operands().get(0));
            }

            MonitorEnterOp(MonitorEnterOp that, CopyContext cc) {
                super(that, cc);
            }

            @Override
            public MonitorEnterOp transform(CopyContext cc, OpTransformer ot) {
                return new MonitorEnterOp(this, cc);
            }

            MonitorEnterOp(Value monitor) {
                super(NAME, monitor);
            }
        }

        /**
         * The monitor exit operation.
         */
        @OpDeclaration(MonitorExitOp.NAME)
        public static final class MonitorExitOp extends MonitorOp {
            static final String NAME = "monitor.exit";

            MonitorExitOp(ExternalizedOp def) {
                if (def.operands().size() != 1) {
                    throw new IllegalArgumentException("Operation must have one operand " + def.name());
                }

                this(def.operands().get(0));
            }

            MonitorExitOp(MonitorExitOp that, CopyContext cc) {
                super(that, cc);
            }

            @Override
            public MonitorExitOp transform(CopyContext cc, OpTransformer ot) {
                return new MonitorExitOp(this, cc);
            }

            MonitorExitOp(Value monitor) {
                super(NAME, monitor);
            }
        }
    }

    /**
     * The invoke operation, that can model Java language method invocation expressions.
     */
    @OpDeclaration(InvokeOp.NAME)
    public static final class InvokeOp extends JavaOp
            implements ReflectiveOp, JavaExpression, JavaStatement {

        /**
         * The kind of invocation.
         */
        public enum InvokeKind {
            /**
             * An invocation on a class (static) method.
             */
            STATIC,
            /**
             * An invocation on an instance method.
             */
            INSTANCE,
            /**
             * A super invocation on an instance method.
             */
            SUPER
        }

        static final String NAME = "invoke";
        public static final String ATTRIBUTE_INVOKE_DESCRIPTOR = NAME + ".descriptor";
        public static final String ATTRIBUTE_INVOKE_KIND = NAME + ".kind";
        public static final String ATTRIBUTE_INVOKE_VARARGS = NAME + ".varargs";

        final InvokeKind invokeKind;
        final boolean isVarArgs;
        final MethodRef invokeDescriptor;
        final TypeElement resultType;

        static InvokeOp create(ExternalizedOp def) {
            // Required attribute
            MethodRef invokeDescriptor = def.extractAttributeValue(ATTRIBUTE_INVOKE_DESCRIPTOR,
                    true, v -> switch (v) {
                        case MethodRef md -> md;
                        case null, default ->
                                throw new UnsupportedOperationException("Unsupported invoke descriptor value:" + v);
                    });

            // If not present defaults to false
            boolean isVarArgs = def.extractAttributeValue(ATTRIBUTE_INVOKE_VARARGS,
                    false, v -> switch (v) {
                        case Boolean b -> b;
                        case null, default -> false;
                    });

            // If not present and is not varargs defaults to class or instance invocation
            // based on number of operands and parameters
            InvokeKind ik = def.extractAttributeValue(ATTRIBUTE_INVOKE_KIND,
                    false, v -> switch (v) {
                        case String s -> InvokeKind.valueOf(s);
                        case InvokeKind k -> k;
                        case null, default -> {
                            if (isVarArgs) {
                                // If varargs then we cannot infer invoke kind
                                throw new UnsupportedOperationException("Unsupported invoke kind value:" + v);
                            }
                            int paramCount = invokeDescriptor.type().parameterTypes().size();
                            int argCount = def.operands().size();
                            yield (argCount == paramCount + 1)
                                    ? InvokeKind.INSTANCE
                                    : InvokeKind.STATIC;
                        }
                    });


            return new InvokeOp(ik, isVarArgs, def.resultType(), invokeDescriptor, def.operands());
        }

        InvokeOp(InvokeOp that, CopyContext cc) {
            super(that, cc);

            this.invokeKind = that.invokeKind;
            this.isVarArgs = that.isVarArgs;
            this.invokeDescriptor = that.invokeDescriptor;
            this.resultType = that.resultType;
        }

        @Override
        public InvokeOp transform(CopyContext cc, OpTransformer ot) {
            return new InvokeOp(this, cc);
        }

        InvokeOp(InvokeKind invokeKind, boolean isVarArgs, TypeElement resultType, MethodRef invokeDescriptor, List<Value> args) {
            super(NAME, args);

            validateArgCount(invokeKind, isVarArgs, invokeDescriptor, args);

            this.invokeKind = invokeKind;
            this.isVarArgs = isVarArgs;
            this.invokeDescriptor = invokeDescriptor;
            this.resultType = resultType;
        }

        static void validateArgCount(InvokeKind invokeKind, boolean isVarArgs, MethodRef invokeDescriptor, List<Value> operands) {
            int paramCount = invokeDescriptor.type().parameterTypes().size();
            int argCount = operands.size() - (invokeKind == InvokeKind.STATIC ? 0 : 1);
            if ((!isVarArgs && argCount != paramCount)
                    || argCount < paramCount - 1) {
                throw new IllegalArgumentException(invokeKind + " " + isVarArgs + " " + invokeDescriptor);
            }
        }

        @Override
        public Map<String, Object> externalize() {
            HashMap<String, Object> m = new HashMap<>();
            m.put("", invokeDescriptor);
            if (isVarArgs) {
                // If varargs then we need to declare the invoke.kind attribute
                // Given a method `A::m(A... more)` and an invocation with one
                // operand, we don't know if that operand corresponds to the
                // receiver or a method argument
                m.put(ATTRIBUTE_INVOKE_KIND, invokeKind);
                m.put(ATTRIBUTE_INVOKE_VARARGS, isVarArgs);
            } else if (invokeKind == InvokeKind.SUPER) {
                m.put(ATTRIBUTE_INVOKE_KIND, invokeKind);
            }
            return Collections.unmodifiableMap(m);
        }

        public InvokeKind invokeKind() {
            return invokeKind;
        }

        public boolean isVarArgs() {
            return isVarArgs;
        }

        public MethodRef invokeDescriptor() {
            return invokeDescriptor;
        }

        // @@@ remove?
        public boolean hasReceiver() {
            return invokeKind != InvokeKind.STATIC;
        }

        public List<Value> varArgOperands() {
            if (!isVarArgs) {
                return null;
            }

            int operandCount = operands().size();
            int argCount = operandCount - (invokeKind == InvokeKind.STATIC ? 0 : 1);
            int paramCount = invokeDescriptor.type().parameterTypes().size();
            int varArgCount = argCount - (paramCount - 1);
            return operands().subList(operandCount - varArgCount, operandCount);
        }

        public List<Value> argOperands() {
            if (!isVarArgs) {
                return operands();
            }
            int paramCount = invokeDescriptor().type().parameterTypes().size();
            int argOperandsCount = paramCount - (invokeKind() == InvokeKind.STATIC ? 1 : 0);
            return operands().subList(0, argOperandsCount);
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
    @OpDeclaration(ConvOp.NAME)
    public static final class ConvOp extends JavaOp
            implements Pure, JavaExpression {
        static final String NAME = "conv";

        final TypeElement resultType;

        ConvOp(ExternalizedOp def) {
            this(def.resultType(), def.operands().get(0));
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
    @OpDeclaration(NewOp.NAME)
    public static final class NewOp extends JavaOp
            implements ReflectiveOp, JavaExpression, JavaStatement {

        static final String NAME = "new";
        public static final String ATTRIBUTE_NEW_DESCRIPTOR = NAME + ".descriptor";
        public static final String ATTRIBUTE_NEW_VARARGS = NAME + ".varargs";

        final boolean isVarArgs;
        final ConstructorRef constructorDescriptor;
        final TypeElement resultType;

        static NewOp create(ExternalizedOp def) {
            // Required attribute
            ConstructorRef constructorDescriptor = def.extractAttributeValue(ATTRIBUTE_NEW_DESCRIPTOR,
                    true, v -> switch (v) {
                        case ConstructorRef cd -> cd;
                        case null, default ->
                                throw new UnsupportedOperationException("Unsupported constructor descriptor value:" + v);
                    });

            // If not present defaults to false
            boolean isVarArgs = def.extractAttributeValue(ATTRIBUTE_NEW_VARARGS,
                    false, v -> switch (v) {
                        case Boolean b -> b;
                        case null, default -> false;
                    });

            return new NewOp(isVarArgs, def.resultType(), constructorDescriptor, def.operands());
        }

        NewOp(NewOp that, CopyContext cc) {
            super(that, cc);

            this.isVarArgs = that.isVarArgs;
            this.constructorDescriptor = that.constructorDescriptor;
            this.resultType = that.resultType;
        }

        @Override
        public NewOp transform(CopyContext cc, OpTransformer ot) {
            return new NewOp(this, cc);
        }

        NewOp(boolean isVarargs, TypeElement resultType, ConstructorRef constructorDescriptor, List<Value> args) {
            super(NAME, args);

            validateArgCount(isVarargs, constructorDescriptor, args);

            this.isVarArgs = isVarargs;
            this.constructorDescriptor = constructorDescriptor;
            this.resultType = resultType;
        }

        static void validateArgCount(boolean isVarArgs, ConstructorRef constructorDescriptor, List<Value> operands) {
            int paramCount = constructorDescriptor.type().parameterTypes().size();
            int argCount = operands.size();
            if ((!isVarArgs && argCount != paramCount)
                    || argCount < paramCount - 1) {
                throw new IllegalArgumentException(isVarArgs + " " + constructorDescriptor);
            }
        }

        @Override
        public Map<String, Object> externalize() {
            HashMap<String, Object> m = new HashMap<>();
            m.put("", constructorDescriptor);
            if (isVarArgs) {
                m.put(ATTRIBUTE_NEW_VARARGS, isVarArgs);
            }
            return Collections.unmodifiableMap(m);
        }

        public boolean isVarargs() {
            return isVarArgs;
        }

        public TypeElement type() {
            return opType().returnType();
        } // @@@ duplication, same as resultType()

        public ConstructorRef constructorDescriptor() {
            return constructorDescriptor;
        }

        @Override
        public TypeElement resultType() {
            return resultType;
        }
    }

    /**
     * A field access operation, that can model Java langauge field access expressions.
     */
    public sealed abstract static class FieldAccessOp extends JavaOp
            implements AccessOp, ReflectiveOp {
        public static final String ATTRIBUTE_FIELD_DESCRIPTOR = "field.descriptor";

        final FieldRef fieldDescriptor;

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
        public Map<String, Object> externalize() {
            return Map.of("", fieldDescriptor);
        }

        public final FieldRef fieldDescriptor() {
            return fieldDescriptor;
        }

        /**
         * The field load operation, that can model Java language field access expressions combined with load access to
         * field instance variables.
         */
        @OpDeclaration(FieldLoadOp.NAME)
        public static final class FieldLoadOp extends FieldAccessOp
                implements Pure, JavaExpression {
            static final String NAME = "field.load";

            final TypeElement resultType;

            static FieldLoadOp create(ExternalizedOp def) {
                if (def.operands().size() > 1) {
                    throw new IllegalArgumentException("Operation must accept zero or one operand");
                }

                FieldRef fieldDescriptor = def.extractAttributeValue(ATTRIBUTE_FIELD_DESCRIPTOR, true,
                        v -> switch (v) {
                            case FieldRef fd -> fd;
                            case null, default ->
                                    throw new UnsupportedOperationException("Unsupported field descriptor value:" + v);
                        });
                if (def.operands().isEmpty()) {
                    return new FieldLoadOp(def.resultType(), fieldDescriptor);
                } else {
                    return new FieldLoadOp(def.resultType(), fieldDescriptor, def.operands().get(0));
                }
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
        @OpDeclaration(FieldStoreOp.NAME)
        public static final class FieldStoreOp extends FieldAccessOp
                implements JavaExpression, JavaStatement {
            static final String NAME = "field.store";

            static FieldStoreOp create(ExternalizedOp def) {
                if (def.operands().isEmpty() || def.operands().size() > 2) {
                    throw new IllegalArgumentException("Operation must accept one or two operands");
                }

                FieldRef fieldDescriptor = def.extractAttributeValue(ATTRIBUTE_FIELD_DESCRIPTOR, true,
                        v -> switch (v) {
                            case FieldRef fd -> fd;
                            case null, default ->
                                    throw new UnsupportedOperationException("Unsupported field descriptor value:" + v);
                        });
                if (def.operands().size() == 1) {
                    return new FieldStoreOp(fieldDescriptor, def.operands().get(0));
                } else {
                    return new FieldStoreOp(fieldDescriptor, def.operands().get(0), def.operands().get(1));
                }
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
                super(NAME, List.of(receiver, v), descriptor);
            }

            // static
            FieldStoreOp(FieldRef descriptor, Value v) {
                super(NAME, List.of(v), descriptor);
            }

            @Override
            public TypeElement resultType() {
                return VOID;
            }
        }
    }

    /**
     * The array length operation, that can model Java language field access expressions to the length field of an
     * array.
     */
    @OpDeclaration(ArrayLengthOp.NAME)
    public static final class ArrayLengthOp extends JavaOp
            implements ReflectiveOp, JavaExpression {
        static final String NAME = "array.length";

        ArrayLengthOp(ExternalizedOp def) {
            this(def.operands().get(0));
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
            return INT;
        }
    }

    /**
     * The array access operation, that can model Java language array access expressions.
     */
    public sealed abstract static class ArrayAccessOp extends JavaOp
            implements AccessOp, ReflectiveOp {

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

        /**
         * The array load operation, that can model Java language array expressions combined with load access to the
         * components of an array.
         */
        @OpDeclaration(ArrayLoadOp.NAME)
        public static final class ArrayLoadOp extends ArrayAccessOp
                implements Pure, JavaExpression {
            static final String NAME = "array.load";
            final TypeElement componentType;

            ArrayLoadOp(ExternalizedOp def) {
                if (def.operands().size() != 2) {
                    throw new IllegalArgumentException("Operation must have two operands");
                }

                this(def.operands().get(0), def.operands().get(1), def.resultType());
            }

            ArrayLoadOp(ArrayLoadOp that, CopyContext cc) {
                super(that, cc);
                this.componentType = that.componentType;
            }

            @Override
            public ArrayLoadOp transform(CopyContext cc, OpTransformer ot) {
                return new ArrayLoadOp(this, cc);
            }

            ArrayLoadOp(Value array, Value index) {
                // @@@ revisit this when the component type is not explicitly given (see VarOp.resultType as an example)
                this(array, index, ((ArrayType)array.type()).componentType());
            }

            ArrayLoadOp(Value array, Value index, TypeElement componentType) {
                super(NAME, array, index, null);
                this.componentType = componentType;
            }

            @Override
            public TypeElement resultType() {
                return componentType;
            }
        }

        /**
         * The array store operation, that can model Java language array expressions combined with store access to the
         * components of an array.
         */
        @OpDeclaration(ArrayStoreOp.NAME)
        public static final class ArrayStoreOp extends ArrayAccessOp
                implements JavaExpression, JavaStatement {
            static final String NAME = "array.store";

            ArrayStoreOp(ExternalizedOp def) {
                if (def.operands().size() != 3) {
                    throw new IllegalArgumentException("Operation must have two operands");
                }

                this(def.operands().get(0), def.operands().get(1), def.operands().get(2));
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
                return VOID;
            }
        }
    }

    /**
     * The instanceof operation, that can model Java language instanceof expressions when the instanceof keyword is a
     * type comparison operator.
     */
    @OpDeclaration(InstanceOfOp.NAME)
    public static final class InstanceOfOp extends JavaOp
            implements Pure, ReflectiveOp, JavaExpression {
        static final String NAME = "instanceof";
        public static final String ATTRIBUTE_TYPE_DESCRIPTOR = NAME + ".descriptor";

        final TypeElement typeDescriptor;

        static InstanceOfOp create(ExternalizedOp def) {
            if (def.operands().size() != 1) {
                throw new IllegalArgumentException("Operation must have one operand " + def.name());
            }

            TypeElement typeDescriptor = def.extractAttributeValue(ATTRIBUTE_TYPE_DESCRIPTOR, true,
                    v -> switch (v) {
                        case JavaType td -> td;
                        case null, default -> throw new UnsupportedOperationException("Unsupported type descriptor value:" + v);
                    });
            return new InstanceOfOp(typeDescriptor, def.operands().get(0));
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
            super(NAME, List.of(v));

            this.typeDescriptor = t;
        }

        @Override
        public Map<String, Object> externalize() {
            return Map.of("", typeDescriptor);
        }

        public TypeElement type() {
            return typeDescriptor;
        }

        @Override
        public TypeElement resultType() {
            return BOOLEAN;
        }
    }

    /**
     * The cast operation, that can model Java language cast expressions for reference types.
     */
    @OpDeclaration(CastOp.NAME)
    public static final class CastOp extends JavaOp
            implements Pure, ReflectiveOp, JavaExpression {
        static final String NAME = "cast";
        public static final String ATTRIBUTE_TYPE_DESCRIPTOR = NAME + ".descriptor";

        final TypeElement resultType;
        final TypeElement typeDescriptor;

        static CastOp create(ExternalizedOp def) {
            if (def.operands().size() != 1) {
                throw new IllegalArgumentException("Operation must have one operand " + def.name());
            }

            TypeElement type = def.extractAttributeValue(ATTRIBUTE_TYPE_DESCRIPTOR, true,
                    v -> switch (v) {
                        case JavaType td -> td;
                        case null, default -> throw new UnsupportedOperationException("Unsupported type descriptor value:" + v);
                    });
            return new CastOp(def.resultType(), type, def.operands().get(0));
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
        public Map<String, Object> externalize() {
            return Map.of("", typeDescriptor);
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
     * The exception region start operation.
     */
    @OpDeclaration(ExceptionRegionEnter.NAME)
    public static final class ExceptionRegionEnter extends JavaOp
            implements BlockTerminating {
        static final String NAME = "exception.region.enter";

        // First successor is the non-exceptional successor whose target indicates
        // the first block in the exception region.
        // One or more subsequent successors target the exception catching blocks
        // each of which have one block argument whose type is an exception type.
        final List<Block.Reference> s;

        ExceptionRegionEnter(ExternalizedOp def) {
            this(def.successors());
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
            return VOID;
        }
    }

    /**
     * The exception region end operation.
     */
    @OpDeclaration(ExceptionRegionExit.NAME)
    public static final class ExceptionRegionExit extends JavaOp
            implements BlockTerminating {
        static final String NAME = "exception.region.exit";

        // First successor is the non-exceptional successor whose target indicates
        // the first block following the exception region.
        final List<Block.Reference> s;

        ExceptionRegionExit(ExternalizedOp def) {
            this(def.successors());
        }

        ExceptionRegionExit(ExceptionRegionExit that, CopyContext cc) {
            super(that, cc);

            this.s = that.s.stream().map(cc::getSuccessorOrCreate).toList();
        }

        @Override
        public ExceptionRegionExit transform(CopyContext cc, OpTransformer ot) {
            return new ExceptionRegionExit(this, cc);
        }

        ExceptionRegionExit(List<Block.Reference> s) {
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

        public Block.Reference end() {
            return s.get(0);
        }

        public List<Block.Reference> catchBlocks() {
            return s.subList(1, s.size());
        }

        @Override
        public TypeElement resultType() {
            return VOID;
        }
    }

    /**
     * The String Concatenation Operation
     */

    @OpDeclaration(ConcatOp.NAME)
    public static final class ConcatOp extends JavaOp
            implements Pure, JavaExpression {
        static final String NAME = "concat";

        public ConcatOp(ConcatOp that, CopyContext cc) {
            super(that, cc);
        }

        ConcatOp(ExternalizedOp def) {
            if (def.operands().size() != 2) {
                throw new IllegalArgumentException("Concatenation Operation must have two operands.");
            }

            this(def.operands().get(0), def.operands().get(1));
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
            return J_L_STRING;
        }
    }

    /**
     * The arithmetic operation.
     */
    public sealed static abstract class ArithmeticOperation extends JavaOp
            implements Pure, JavaExpression {
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
    public sealed static abstract class TestOperation extends JavaOp
            implements Pure, JavaExpression {
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
        protected BinaryTestOp(BinaryTestOp that, CopyContext cc) {
            super(that, cc);
        }

        protected BinaryTestOp(String name, Value lhs, Value rhs) {
            super(name, List.of(lhs, rhs));
        }

        @Override
        public TypeElement resultType() {
            return BOOLEAN;
        }
    }

    /**
     * The add operation, that can model the Java language binary {@code +} operator for numeric types
     */
    @OpDeclaration(AddOp.NAME)
    public static final class AddOp extends BinaryOp {
        static final String NAME = "add";

        AddOp(ExternalizedOp def) {
            this(def.operands().get(0), def.operands().get(1));
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
    @OpDeclaration(SubOp.NAME)
    public static final class SubOp extends BinaryOp {
        static final String NAME = "sub";

        SubOp(ExternalizedOp def) {
            this(def.operands().get(0), def.operands().get(1));
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
    @OpDeclaration(MulOp.NAME)
    public static final class MulOp extends BinaryOp {
        static final String NAME = "mul";

        MulOp(ExternalizedOp def) {
            this(def.operands().get(0), def.operands().get(1));
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
    @OpDeclaration(DivOp.NAME)
    public static final class DivOp extends BinaryOp {
        static final String NAME = "div";

        DivOp(ExternalizedOp def) {
            this(def.operands().get(0), def.operands().get(1));
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
    @OpDeclaration(ModOp.NAME)
    public static final class ModOp extends BinaryOp {
        static final String NAME = "mod";

        ModOp(ExternalizedOp def) {
            this(def.operands().get(0), def.operands().get(1));
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
    @OpDeclaration(OrOp.NAME)
    public static final class OrOp extends BinaryOp {
        static final String NAME = "or";

        OrOp(ExternalizedOp def) {
            this(def.operands().get(0), def.operands().get(1));
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
    @OpDeclaration(AndOp.NAME)
    public static final class AndOp extends BinaryOp {
        static final String NAME = "and";

        AndOp(ExternalizedOp def) {
            this(def.operands().get(0), def.operands().get(1));
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
    @OpDeclaration(XorOp.NAME)
    public static final class XorOp extends BinaryOp {
        static final String NAME = "xor";

        XorOp(ExternalizedOp def) {
            this(def.operands().get(0), def.operands().get(1));
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
    @OpDeclaration(LshlOp.NAME)
    public static final class LshlOp extends BinaryOp {
        static final String NAME = "lshl";

        LshlOp(ExternalizedOp def) {
            this(def.operands().get(0), def.operands().get(1));
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
    @OpDeclaration(AshrOp.NAME)
    public static final class AshrOp extends JavaOp.BinaryOp {
        static final String NAME = "ashr";

        AshrOp(ExternalizedOp def) {
            this(def.operands().get(0), def.operands().get(1));
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
    @OpDeclaration(LshrOp.NAME)
    public static final class LshrOp extends JavaOp.BinaryOp {
        static final String NAME = "lshr";

        LshrOp(ExternalizedOp def) {
            this(def.operands().get(0), def.operands().get(1));
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
    @OpDeclaration(NegOp.NAME)
    public static final class NegOp extends UnaryOp {
        static final String NAME = "neg";

        NegOp(ExternalizedOp def) {
            this(def.operands().get(0));
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
     * The bitwise complement operation, that can model the Java language unary {@code ~} operator for integral types
     */
    @OpDeclaration(ComplOp.NAME)
    public static final class ComplOp extends UnaryOp {
        static final String NAME = "compl";

        ComplOp(ExternalizedOp def) {
            this(def.operands().get(0));
        }

        ComplOp(ComplOp that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public ComplOp transform(CopyContext cc, OpTransformer ot) {
            return new ComplOp(this, cc);
        }

        ComplOp(Value v) {
            super(NAME, v);
        }
    }

    /**
     * The not operation, that can model the Java language unary {@code !} operator for boolean types
     */
    @OpDeclaration(NotOp.NAME)
    public static final class NotOp extends UnaryOp {
        static final String NAME = "not";

        NotOp(ExternalizedOp def) {
            this(def.operands().get(0));
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
    @OpDeclaration(EqOp.NAME)
    public static final class EqOp extends BinaryTestOp {
        static final String NAME = "eq";

        EqOp(ExternalizedOp def) {
            this(def.operands().get(0), def.operands().get(1));
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
    @OpDeclaration(NeqOp.NAME)
    public static final class NeqOp extends BinaryTestOp {
        static final String NAME = "neq";

        NeqOp(ExternalizedOp def) {
            this(def.operands().get(0), def.operands().get(1));
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
    @OpDeclaration(GtOp.NAME)
    public static final class GtOp extends BinaryTestOp {
        static final String NAME = "gt";

        GtOp(ExternalizedOp def) {
            this(def.operands().get(0), def.operands().get(1));
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
    @OpDeclaration(GeOp.NAME)
    public static final class GeOp extends BinaryTestOp {
        static final String NAME = "ge";

        GeOp(ExternalizedOp def) {
            this(def.operands().get(0), def.operands().get(1));
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
    @OpDeclaration(LtOp.NAME)
    public static final class LtOp extends BinaryTestOp {
        static final String NAME = "lt";

        LtOp(ExternalizedOp def) {
            this(def.operands().get(0), def.operands().get(1));
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
    @OpDeclaration(LeOp.NAME)
    public static final class LeOp extends BinaryTestOp {
        static final String NAME = "le";

        LeOp(ExternalizedOp def) {
            this(def.operands().get(0), def.operands().get(1));
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
     * The label operation, that can model Java language statements with label identifiers.
     */
    public sealed static abstract class JavaLabelOp extends JavaOp
            implements Op.Lowerable, Op.BodyTerminating, JavaStatement {
        JavaLabelOp(JavaLabelOp that, CopyContext cc) {
            super(that, cc);
        }

        JavaLabelOp(String name, Value label) {
            super(name, checkLabel(label));
        }

        static List<Value> checkLabel(Value label) {
            return label == null ? List.of() : List.of(label);
        }

        Op innerMostEnclosingTarget() {
            /*
                A break statement with no label attempts to transfer control to the
                innermost enclosing switch, while, do, or for statement; this enclosing statement,
                which is called the break target, then immediately completes normally.

                A break statement with label Identifier attempts to transfer control to the
                enclosing labeled statement (14.7) that has the same Identifier as its label;
                this enclosing statement, which is called the break target, then immediately completes normally.
                In this case, the break target need not be a switch, while, do, or for statement.
             */

            // No label
            // Get innermost enclosing loop operation
            Op op = this;
            Body b;
            do {
                b = op.ancestorBody();
                op = b.parentOp();
                if (op == null) {
                    throw new IllegalStateException("No enclosing loop");
                }
            } while (!(op instanceof Op.Loop || op instanceof SwitchStatementOp));

            return switch (op) {
                case Op.Loop lop -> lop.loopBody() == b ? op : null;
                case SwitchStatementOp swStat -> swStat.bodies().contains(b) ? op : null;
                default -> throw new IllegalStateException();
            };
        }

        boolean isUnlabeled() {
            return operands().isEmpty();
        }

        Op target() {
            // If unlabeled then find the nearest enclosing op
            // Otherwise obtain the label target
            if (isUnlabeled()) {
                return innerMostEnclosingTarget();
            }

            Value value = operands().get(0);
            if (value instanceof Result r && r.op().ancestorBody().parentOp() instanceof LabeledOp lop) {
                return lop.target();
            } else {
                throw new IllegalStateException("Bad label value: " + value + " " + ((Result) value).op());
            }
        }

        Block.Builder lower(Block.Builder b, Function<BranchTarget, Block.Builder> f) {
            Op opt = target();
            BranchTarget t = getBranchTarget(b.context(), opt);
            if (t != null) {
                b.op(branch(f.apply(t).successor()));
            } else {
                throw new IllegalStateException("No branch target for operation: " + opt);
            }
            return b;
        }

        @Override
        public TypeElement resultType() {
            return VOID;
        }
    }

    /**
     * The break operation, that can model Java language break statements with label identifiers.
     */
    @OpDeclaration(BreakOp.NAME)
    public static final class BreakOp extends JavaLabelOp {
        static final String NAME = "java.break";

        BreakOp(ExternalizedOp def) {
            this(def.operands().isEmpty() ? null : def.operands().get(0));
        }

        BreakOp(BreakOp that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public BreakOp transform(CopyContext cc, OpTransformer ot) {
            return new BreakOp(this, cc);
        }

        BreakOp(Value label) {
            super(NAME, label);
        }

        @Override
        public Block.Builder lower(Block.Builder b, OpTransformer opT) {
            return lower(b, BranchTarget::breakBlock);
        }
    }

    /**
     * The continue operation, that can model Java language continue statements with label identifiers.
     */
    @OpDeclaration(ContinueOp.NAME)
    public static final class ContinueOp extends JavaLabelOp {
        static final String NAME = "java.continue";

        ContinueOp(ExternalizedOp def) {
            this(def.operands().isEmpty() ? null : def.operands().get(0));
        }

        ContinueOp(ContinueOp that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public ContinueOp transform(CopyContext cc, OpTransformer ot) {
            return new ContinueOp(this, cc);
        }

        ContinueOp(Value label) {
            super(NAME, label);
        }

        @Override
        public Block.Builder lower(Block.Builder b, OpTransformer opT) {
            return lower(b, BranchTarget::continueBlock);
        }
    }

    record BranchTarget(Block.Builder breakBlock, Block.Builder continueBlock) {
    }

    static final String BRANCH_TARGET_MAP_PROPERTY_KEY = "BRANCH_TARGET_MAP";

    static BranchTarget getBranchTarget(CopyContext cc, CodeElement<?, ?> codeElement) {
        @SuppressWarnings("unchecked")
        Map<CodeElement<?, ?>, BranchTarget> m = (Map<CodeElement<?, ?>, BranchTarget>) cc.getProperty(BRANCH_TARGET_MAP_PROPERTY_KEY);
        if (m != null) {
            return m.get(codeElement);
        }
        return null;
    }

    static void setBranchTarget(CopyContext cc, CodeElement<?, ?> codeElement, BranchTarget t) {
        @SuppressWarnings("unchecked")
        Map<CodeElement<?, ?>, BranchTarget> x = (Map<CodeElement<?, ?>, BranchTarget>) cc.computePropertyIfAbsent(
                BRANCH_TARGET_MAP_PROPERTY_KEY, k -> new HashMap<>());
        x.put(codeElement, t);
    }

    /**
     * The yield operation, that can model Java language yield statements.
     */
    @OpDeclaration(YieldOp.NAME)
    public static final class YieldOp extends JavaOp
            implements Op.BodyTerminating, JavaStatement, Op.Lowerable {
        static final String NAME = "java.yield";

        YieldOp(ExternalizedOp def) {
            if (def.operands().size() > 1) {
                throw new IllegalArgumentException("Operation must have zero or one operand " + def.name());
            }

            this(def.operands().isEmpty() ? null : def.operands().get(0));
        }

        YieldOp(YieldOp that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public YieldOp transform(CopyContext cc, OpTransformer ot) {
            return new YieldOp(this, cc);
        }

        YieldOp(Value operand) {
            super(NAME, operand == null ? List.of() : List.of(operand));
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
            return VOID;
        }

        @Override
        public Block.Builder lower(Block.Builder b, OpTransformer opT) {
            // for now, we will use breakBlock field to indicate java.yield target block
            return lower(b, BranchTarget::breakBlock);
        }

        Block.Builder lower(Block.Builder b, Function<BranchTarget, Block.Builder> f) {
            Op opt = target();
            BranchTarget t = getBranchTarget(b.context(), opt);
            if (t != null) {
                b.op(branch(f.apply(t).successor(b.context().getValue(yieldValue()))));
            } else {
                throw new IllegalStateException("No branch target for operation: " + opt);
            }
            return b;
        }

        Op target() {
            return innerMostEnclosingTarget();
        }

        Op innerMostEnclosingTarget() {
            Op op = this;
            Body b;
            do {
                b = op.ancestorBody();
                op = b.parentOp();
                if (op == null) {
                    throw new IllegalStateException("No enclosing switch");
                }
            } while (!(op instanceof SwitchExpressionOp));
            return op;
        }
    }

    /**
     * The block operation, that can model Java language blocks.
     */
    @OpDeclaration(BlockOp.NAME)
    public static final class BlockOp extends JavaOp
            implements Op.Nested, Op.Lowerable, JavaStatement {
        static final String NAME = "java.block";

        final Body body;

        BlockOp(ExternalizedOp def) {
            if (!def.operands().isEmpty()) {
                throw new IllegalStateException("Operation must have no operands");
            }

            this(def.bodyDefinitions().get(0));
        }

        BlockOp(BlockOp that, CopyContext cc, OpTransformer ot) {
            super(that, cc);

            // Copy body
            this.body = that.body.transform(cc, ot).build(this);
        }

        @Override
        public BlockOp transform(CopyContext cc, OpTransformer ot) {
            return new BlockOp(this, cc, ot);
        }

        BlockOp(Body.Builder bodyC) {
            super(NAME, List.of());

            this.body = bodyC.build(this);
            if (!body.bodyType().returnType().equals(VOID)) {
                throw new IllegalArgumentException("Body should return void: " + body.bodyType());
            }
            if (!body.bodyType().parameterTypes().isEmpty()) {
                throw new IllegalArgumentException("Body should have zero parameters: " + body.bodyType());
            }
        }

        @Override
        public List<Body> bodies() {
            return List.of(body);
        }

        public Body body() {
            return body;
        }

        @Override
        public Block.Builder lower(Block.Builder b, OpTransformer opT) {
            Block.Builder exit = b.block();
            setBranchTarget(b.context(), this, new BranchTarget(exit, null));

            b.transformBody(body, List.of(), opT.andThen((block, op) -> {
                if (op instanceof CoreOp.YieldOp) {
                    block.op(branch(exit.successor()));
                } else {
                    // @@@ Composition of lowerable ops
                    if (op instanceof Lowerable lop) {
                        block = lop.lower(block, opT);
                    } else {
                        block.op(op);
                    }
                }
                return block;
            }));

            return exit;
        }

        @Override
        public TypeElement resultType() {
            return VOID;
        }
    }

    /**
     * The synchronized operation, that can model Java synchronized statements.
     */
    @OpDeclaration(SynchronizedOp.NAME)
    public static final class SynchronizedOp extends JavaOp
            implements Op.Nested, Op.Lowerable, JavaStatement {
        static final String NAME = "java.synchronized";

        final Body expr;
        final Body blockBody;

        SynchronizedOp(ExternalizedOp def) {
            this(def.bodyDefinitions().get(0), def.bodyDefinitions().get(1));
        }

        SynchronizedOp(SynchronizedOp that, CopyContext cc, OpTransformer ot) {
            super(that, cc);

            // Copy bodies
            this.expr = that.expr.transform(cc, ot).build(this);
            this.blockBody = that.blockBody.transform(cc, ot).build(this);
        }

        @Override
        public SynchronizedOp transform(CopyContext cc, OpTransformer ot) {
            return new SynchronizedOp(this, cc, ot);
        }

        SynchronizedOp(Body.Builder exprC, Body.Builder bodyC) {
            super(NAME, List.of());

            this.expr = exprC.build(this);
            if (expr.bodyType().returnType().equals(VOID)) {
                throw new IllegalArgumentException("Expression body should return non-void value: " + expr.bodyType());
            }
            if (!expr.bodyType().parameterTypes().isEmpty()) {
                throw new IllegalArgumentException("Expression body should have zero parameters: " + expr.bodyType());
            }

            this.blockBody = bodyC.build(this);
            if (!blockBody.bodyType().returnType().equals(VOID)) {
                throw new IllegalArgumentException("Block body should return void: " + blockBody.bodyType());
            }
            if (!blockBody.bodyType().parameterTypes().isEmpty()) {
                throw new IllegalArgumentException("Block body should have zero parameters: " + blockBody.bodyType());
            }
        }

        @Override
        public List<Body> bodies() {
            return List.of(expr, blockBody);
        }

        public Body expr() {
            return expr;
        }

        public Body blockBody() {
            return blockBody;
        }

        @Override
        public Block.Builder lower(Block.Builder b, OpTransformer opT) {
            // Lower the expression body, yielding a monitor target
            b = lowerExpr(b, opT);
            Value monitorTarget = b.parameters().get(0);

            // Monitor enter
            b.op(monitorEnter(monitorTarget));

            Block.Builder exit = b.block();
            setBranchTarget(b.context(), this, new BranchTarget(exit, null));

            // Exception region for the body
            Block.Builder syncRegionEnter = b.block();
            Block.Builder catcherFinally = b.block();
            b.op(exceptionRegionEnter(
                    syncRegionEnter.successor(), catcherFinally.successor()));

            OpTransformer syncExitTransformer = opT.compose((block, op) -> {
                if (op instanceof CoreOp.ReturnOp ||
                    (op instanceof JavaOp.JavaLabelOp lop && ifExitFromSynchronized(lop))) {
                    // Monitor exit
                    block.op(monitorExit(monitorTarget));
                    // Exit the exception region
                    Block.Builder exitRegion = block.block();
                    block.op(exceptionRegionExit(exitRegion.successor(), catcherFinally.successor()));
                    return exitRegion;
                } else {
                    return block;
                }
            });

            syncRegionEnter.transformBody(blockBody, List.of(), syncExitTransformer.andThen((block, op) -> {
                if (op instanceof CoreOp.YieldOp) {
                    // Monitor exit
                    block.op(monitorExit(monitorTarget));
                    // Exit the exception region
                    block.op(exceptionRegionExit(exit.successor(), catcherFinally.successor()));
                } else {
                    // @@@ Composition of lowerable ops
                    if (op instanceof Lowerable lop) {
                        block = lop.lower(block, syncExitTransformer);
                    } else {
                        block.op(op);
                    }
                }
                return block;
            }));

            // The catcher, with an exception region back branching to itself
            Block.Builder catcherFinallyRegionEnter = b.block();
            catcherFinally.op(exceptionRegionEnter(
                    catcherFinallyRegionEnter.successor(), catcherFinally.successor()));

            // Monitor exit
            catcherFinallyRegionEnter.op(monitorExit(monitorTarget));
            Block.Builder catcherFinallyRegionExit = b.block();
            // Exit the exception region
            catcherFinallyRegionEnter.op(exceptionRegionExit(
                    catcherFinallyRegionExit.successor(), catcherFinally.successor()));
            // Rethrow outside of region
            Block.Parameter t = catcherFinally.parameter(type(Throwable.class));
            catcherFinallyRegionExit.op(throw_(t));

            return exit;
        }

        Block.Builder lowerExpr(Block.Builder b, OpTransformer opT) {
            Block.Builder exprExit = b.block(expr.bodyType().returnType());
            b.transformBody(expr, List.of(), opT.andThen((block, op) -> {
                if (op instanceof CoreOp.YieldOp yop) {
                    Value monitorTarget = block.context().getValue(yop.yieldValue());
                    block.op(branch(exprExit.successor(monitorTarget)));
                } else {
                    // @@@ Composition of lowerable ops
                    if (op instanceof Lowerable lop) {
                        block = lop.lower(block, opT);
                    } else {
                        block.op(op);
                    }
                }
                return block;
            }));
            return exprExit;
        }

        boolean ifExitFromSynchronized(JavaLabelOp lop) {
            Op target = lop.target();
            return target == this || ifAncestorOp(target, this);
        }

        static boolean ifAncestorOp(Op ancestor, Op op) {
            while (op.ancestorBody() != null) {
                op = op.ancestorBody().parentOp();
                if (op == ancestor) {
                    return true;
                }
            }
            return false;
        }

        @Override
        public TypeElement resultType() {
            return VOID;
        }
    }

    /**
     * The labeled operation, that can model Java language labeled statements.
     */
    @OpDeclaration(LabeledOp.NAME)
    public static final class LabeledOp extends JavaOp
            implements Op.Nested, Op.Lowerable, JavaStatement {
        static final String NAME = "java.labeled";

        final Body body;

        LabeledOp(ExternalizedOp def) {
            if (!def.operands().isEmpty()) {
                throw new IllegalStateException("Operation must have no operands");
            }

            this(def.bodyDefinitions().get(0));
        }

        LabeledOp(LabeledOp that, CopyContext cc, OpTransformer ot) {
            super(that, cc);

            // Copy body
            this.body = that.body.transform(cc, ot).build(this);
        }

        @Override
        public LabeledOp transform(CopyContext cc, OpTransformer ot) {
            return new LabeledOp(this, cc, ot);
        }

        LabeledOp(Body.Builder bodyC) {
            super(NAME, List.of());

            this.body = bodyC.build(this);
            if (!body.bodyType().returnType().equals(VOID)) {
                throw new IllegalArgumentException("Body should return void: " + body.bodyType());
            }
            if (!body.bodyType().parameterTypes().isEmpty()) {
                throw new IllegalArgumentException("Body should have zero parameters: " + body.bodyType());
            }
        }

        @Override
        public List<Body> bodies() {
            return List.of(body);
        }

        public Op label() {
            return body.entryBlock().firstOp();
        }

        public Op target() {
            return body.entryBlock().nextOp(label());
        }

        @Override
        public Block.Builder lower(Block.Builder b, OpTransformer opT) {
            Block.Builder exit = b.block();
            setBranchTarget(b.context(), this, new BranchTarget(exit, null));

            AtomicBoolean first = new AtomicBoolean();
            b.transformBody(body, List.of(), opT.andThen((block, op) -> {
                // Drop first operation that corresponds to the label
                if (!first.get()) {
                    first.set(true);
                    return block;
                }

                if (op instanceof CoreOp.YieldOp) {
                    block.op(branch(exit.successor()));
                } else {
                    // @@@ Composition of lowerable ops
                    if (op instanceof Lowerable lop) {
                        block = lop.lower(block, opT);
                    } else {
                        block.op(op);
                    }
                }
                return block;
            }));

            return exit;
        }

        @Override
        public TypeElement resultType() {
            return VOID;
        }
    }

    /**
     * The if operation, that can model Java language if, if-then, and if-then-else statements.
     */
    @OpDeclaration(IfOp.NAME)
    public static final class IfOp extends JavaOp
            implements Op.Nested, Op.Lowerable, JavaStatement {

        static final FunctionType PREDICATE_TYPE = CoreType.functionType(BOOLEAN);

        static final FunctionType ACTION_TYPE = CoreType.FUNCTION_TYPE_VOID;

        public static class IfBuilder {
            final Body.Builder ancestorBody;
            final List<Body.Builder> bodies;

            IfBuilder(Body.Builder ancestorBody) {
                this.ancestorBody = ancestorBody;
                this.bodies = new ArrayList<>();
            }

            public ThenBuilder if_(Consumer<Block.Builder> c) {
                Body.Builder body = Body.Builder.of(ancestorBody, PREDICATE_TYPE);
                c.accept(body.entryBlock());
                bodies.add(body);

                return new ThenBuilder(ancestorBody, bodies);
            }
        }

        public static class ThenBuilder {
            final Body.Builder ancestorBody;
            final List<Body.Builder> bodies;

            public ThenBuilder(Body.Builder ancestorBody, List<Body.Builder> bodies) {
                this.ancestorBody = ancestorBody;
                this.bodies = bodies;
            }

            public ElseIfBuilder then(Consumer<Block.Builder> c) {
                Body.Builder body = Body.Builder.of(ancestorBody, ACTION_TYPE);
                c.accept(body.entryBlock());
                bodies.add(body);

                return new ElseIfBuilder(ancestorBody, bodies);
            }

            public ElseIfBuilder then() {
                Body.Builder body = Body.Builder.of(ancestorBody, ACTION_TYPE);
                body.entryBlock().op(core_yield());
                bodies.add(body);

                return new ElseIfBuilder(ancestorBody, bodies);
            }
        }

        public static class ElseIfBuilder {
            final Body.Builder ancestorBody;
            final List<Body.Builder> bodies;

            public ElseIfBuilder(Body.Builder ancestorBody, List<Body.Builder> bodies) {
                this.ancestorBody = ancestorBody;
                this.bodies = bodies;
            }

            public ThenBuilder elseif(Consumer<Block.Builder> c) {
                Body.Builder body = Body.Builder.of(ancestorBody, PREDICATE_TYPE);
                c.accept(body.entryBlock());
                bodies.add(body);

                return new ThenBuilder(ancestorBody, bodies);
            }

            public IfOp else_(Consumer<Block.Builder> c) {
                Body.Builder body = Body.Builder.of(ancestorBody, ACTION_TYPE);
                c.accept(body.entryBlock());
                bodies.add(body);

                return new IfOp(bodies);
            }

            public IfOp else_() {
                Body.Builder body = Body.Builder.of(ancestorBody, ACTION_TYPE);
                body.entryBlock().op(core_yield());
                bodies.add(body);

                return new IfOp(bodies);
            }
        }

        static final String NAME = "java.if";

        final List<Body> bodies;

        IfOp(ExternalizedOp def) {
            if (!def.operands().isEmpty()) {
                throw new IllegalStateException("Operation must have no operands");
            }

            this(def.bodyDefinitions());
        }

        IfOp(IfOp that, CopyContext cc, OpTransformer ot) {
            super(that, cc);

            // Copy body
            this.bodies = that.bodies.stream()
                    .map(b -> b.transform(cc, ot).build(this)).toList();
        }

        @Override
        public IfOp transform(CopyContext cc, OpTransformer ot) {
            return new IfOp(this, cc, ot);
        }

        IfOp(List<Body.Builder> bodyCs) {
            super(NAME, List.of());

            // Normalize by adding an empty else action
            // @@@ Is this needed?
            if (bodyCs.size() % 2 == 0) {
                bodyCs = new ArrayList<>(bodyCs);
                Body.Builder end = Body.Builder.of(bodyCs.get(0).ancestorBody(),
                        CoreType.FUNCTION_TYPE_VOID);
                end.entryBlock().op(core_yield());
                bodyCs.add(end);
            }

            this.bodies = bodyCs.stream().map(bc -> bc.build(this)).toList();

            if (bodies.size() < 2) {
                throw new IllegalArgumentException("Incorrect number of bodies: " + bodies.size());
            }
            for (int i = 0; i < bodies.size(); i += 2) {
                Body action;
                if (i == bodies.size() - 1) {
                    action = bodies.get(i);
                } else {
                    action = bodies.get(i + 1);
                    Body fromPred = bodies.get(i);
                    if (!fromPred.bodyType().equals(CoreType.functionType(BOOLEAN))) {
                        throw new IllegalArgumentException("Illegal predicate body descriptor: " + fromPred.bodyType());
                    }
                }
                if (!action.bodyType().equals(CoreType.FUNCTION_TYPE_VOID)) {
                    throw new IllegalArgumentException("Illegal action body descriptor: " + action.bodyType());
                }
            }
        }

        @Override
        public List<Body> bodies() {
            return bodies;
        }

        @Override
        public Block.Builder lower(Block.Builder b, OpTransformer opT) {
            Block.Builder exit = b.block();
            setBranchTarget(b.context(), this, new BranchTarget(exit, null));

            // Create predicate and action blocks
            List<Block.Builder> builders = new ArrayList<>();
            for (int i = 0; i < bodies.size(); i += 2) {
                if (i == bodies.size() - 1) {
                    builders.add(b.block());
                } else {
                    builders.add(i == 0 ? b : b.block());
                    builders.add(b.block());
                }
            }

            for (int i = 0; i < bodies.size(); i += 2) {
                Body actionBody;
                Block.Builder action;
                if (i == bodies.size() - 1) {
                    actionBody = bodies.get(i);
                    action = builders.get(i);
                } else {
                    Body predBody = bodies.get(i);
                    actionBody = bodies.get(i + 1);

                    Block.Builder pred = builders.get(i);
                    action = builders.get(i + 1);
                    Block.Builder next = builders.get(i + 2);

                    pred.transformBody(predBody, List.of(), opT.andThen((block, op) -> {
                        if (op instanceof CoreOp.YieldOp yo) {
                            block.op(conditionalBranch(block.context().getValue(yo.yieldValue()),
                                    action.successor(), next.successor()));
                        } else if (op instanceof Lowerable lop) {
                            // @@@ Composition of lowerable ops
                            block = lop.lower(block, opT);
                        } else {
                            block.op(op);
                        }
                        return block;
                    }));
                }

                action.transformBody(actionBody, List.of(), opT.andThen((block, op) -> {
                    if (op instanceof CoreOp.YieldOp) {
                        block.op(branch(exit.successor()));
                    } else {
                        // @@@ Composition of lowerable ops
                        if (op instanceof Lowerable lop) {
                            block = lop.lower(block, opT);
                        } else {
                            block.op(op);
                        }
                    }
                    return block;
                }));
            }

            return exit;
        }

        @Override
        public TypeElement resultType() {
            return VOID;
        }
    }

    public abstract static sealed class JavaSwitchOp extends JavaOp implements Op.Nested, Op.Lowerable
            permits SwitchStatementOp, SwitchExpressionOp {

        final List<Body> bodies;

        JavaSwitchOp(JavaSwitchOp that, CopyContext cc, OpTransformer ot) {
            super(that, cc);

            // Copy body
            this.bodies = that.bodies.stream()
                    .map(b -> b.transform(cc, ot).build(this)).toList();
        }

        JavaSwitchOp(String name, Value target, List<Body.Builder> bodyCs) {
            super(name, List.of(target));

            // Each case is modelled as a contiguous pair of bodies
            // The first body models the case labels, and the second models the case statements
            // The labels body has a parameter whose type is target operand's type and returns a boolean value
            // The statements body has no parameters and returns void
            this.bodies = bodyCs.stream().map(bc -> bc.build(this)).toList();
        }

        @Override
        public List<Body> bodies() {
            return bodies;
        }

        @Override
        public Block.Builder lower(Block.Builder b, OpTransformer opT) {

            Value selectorExpression = b.context().getValue(operands().get(0));

            // @@@ we can add this during model generation
            // if no case null, add one that throws NPE
            if (!(selectorExpression.type() instanceof PrimitiveType) && !haveNullCase()) {
                Block.Builder throwBlock = b.block();
                throwBlock.op(throw_(
                        throwBlock.op(new_(ConstructorRef.constructor(NullPointerException.class)))
                ));

                Block.Builder continueBlock = b.block();

                Result p = b.op(invoke(MethodRef.method(Objects.class, "equals", boolean.class, Object.class, Object.class),
                        selectorExpression, b.op(constant(J_L_OBJECT, null))));
                b.op(conditionalBranch(p, throwBlock.successor(), continueBlock.successor()));

                b = continueBlock;
            }

            List<Block.Builder> blocks = new ArrayList<>();
            for (int i = 0; i < bodies().size(); i++) {
                Block.Builder bb = b.block();
                if (i == 0) {
                    bb = b;
                }
                blocks.add(bb);
            }

            Block.Builder exit;
            if (bodies().isEmpty()) {
                exit = b;
            } else {
                exit = b.block(resultType());
                if (this instanceof SwitchExpressionOp) {
                    exit.context().mapValue(result(), exit.parameters().get(0));
                }
            }

            setBranchTarget(b.context(), this, new BranchTarget(exit, null));
            // map statement body to nextExprBlock
            // this mapping will be used for lowering SwitchFallThroughOp
            for (int i = 1; i < bodies().size() - 2; i+=2) {
                setBranchTarget(b.context(), bodies().get(i), new BranchTarget(null, blocks.get(i + 2)));
            }

            for (int i = 0; i < bodies().size(); i++) {
                boolean isLabelBody = i % 2 == 0;
                Block.Builder curr = blocks.get(i);
                if (isLabelBody) {
                    Block.Builder statement = blocks.get(i + 1);
                    boolean isLastLabel = i == blocks.size() - 2;
                    Block.Builder nextLabel = isLastLabel ? null : blocks.get(i + 2);
                    curr.transformBody(bodies().get(i), List.of(selectorExpression), opT.andThen((block, op) -> {
                        switch (op) {
                            case CoreOp.YieldOp yop when isLastLabel && this instanceof SwitchExpressionOp -> {
                                block.op(branch(statement.successor()));
                            }
                            case CoreOp.YieldOp yop -> block.op(conditionalBranch(
                                    block.context().getValue(yop.yieldValue()),
                                    statement.successor(),
                                    isLastLabel ? exit.successor() : nextLabel.successor()
                            ));
                            case Lowerable lop -> block = lop.lower(block);
                            default -> block.op(op);
                        }
                        return block;
                    }));
                } else { // statement body
                    curr.transformBody(bodies().get(i), blocks.get(i).parameters(), opT.andThen((block, op) -> {
                        switch (op) {
                            case CoreOp.YieldOp yop when this instanceof SwitchStatementOp -> block.op(branch(exit.successor()));
                            case CoreOp.YieldOp yop when this instanceof SwitchExpressionOp -> block.op(branch(exit.successor(block.context().getValue(yop.yieldValue()))));
                            case Lowerable lop -> block = lop.lower(block);
                            default -> block.op(op);
                        }
                        return block;
                    }));
                }
            }

            return exit;
        }

        boolean haveNullCase() {
            /*
            case null is modeled like this:
            (%4 : T)boolean -> {
                %5 : java.lang.Object = constant @null;
                %6 : boolean = invoke %4 %5 @"java.util.Objects::equals(java.lang.Object, java.lang.Object)boolean";
                yield %6;
            }
            * */
            for (int i = 0; i < bodies().size() - 2; i+=2) {
                Body labelBody = bodies().get(i);
                if (labelBody.blocks().size() != 1) {
                    continue; // we skip, for now
                }
                Op terminatingOp = bodies().get(i).entryBlock().terminatingOp();
                //@@@ when op pattern matching is ready, we can use it
                if (terminatingOp instanceof CoreOp.YieldOp yieldOp &&
                        yieldOp.yieldValue() instanceof Op.Result opr &&
                        opr.op() instanceof InvokeOp invokeOp &&
                        invokeOp.invokeDescriptor().equals(MethodRef.method(Objects.class, "equals", boolean.class, Object.class, Object.class)) &&
                        invokeOp.operands().stream().anyMatch(o -> o instanceof Op.Result r && r.op() instanceof ConstantOp cop && cop.value() == null)) {
                    return true;
                }
            }
            return false;
        }
    }

    /**
     * The switch expression operation, that can model Java language switch expressions.
     */
    @OpDeclaration(SwitchExpressionOp.NAME)
    public static final class SwitchExpressionOp extends JavaSwitchOp
            implements JavaExpression {
        static final String NAME = "java.switch.expression";

        final TypeElement resultType;

        SwitchExpressionOp(ExternalizedOp def) {
            this(def.resultType(), def.operands().get(0), def.bodyDefinitions());
        }

        SwitchExpressionOp(SwitchExpressionOp that, CopyContext cc, OpTransformer ot) {
            super(that, cc, ot);

            this.resultType = that.resultType;
        }

        @Override
        public SwitchExpressionOp transform(CopyContext cc, OpTransformer ot) {
            return new SwitchExpressionOp(this, cc, ot);
        }

        SwitchExpressionOp(TypeElement resultType, Value target, List<Body.Builder> bodyCs) {
            super(NAME, target, bodyCs);

            this.resultType = resultType == null ? bodies.get(1).yieldType() : resultType;
        }

        @Override
        public TypeElement resultType() {
            return resultType;
        }
    }

    /**
     * The switch statement operation, that can model Java language switch statement.
     */
    @OpDeclaration(SwitchStatementOp.NAME)
    public static final class SwitchStatementOp extends JavaSwitchOp
            implements JavaStatement {
        static final String NAME = "java.switch.statement";

        SwitchStatementOp(ExternalizedOp def) {
            this(def.operands().get(0), def.bodyDefinitions());
        }

        SwitchStatementOp(SwitchStatementOp that, CopyContext cc, OpTransformer ot) {
            super(that, cc, ot);
        }

        @Override
        public SwitchStatementOp transform(CopyContext cc, OpTransformer ot) {
            return new SwitchStatementOp(this, cc, ot);
        }

        SwitchStatementOp(Value target, List<Body.Builder> bodyCs) {
            super(NAME, target, bodyCs);
        }

        @Override
        public TypeElement resultType() {
            return VOID;
        }
    }

    /**
     * The switch fall-through operation, that can model fall-through to the next statement in the switch block after
     * the last statement of the current switch label.
     */
    @OpDeclaration(SwitchFallthroughOp.NAME)
    public static final class SwitchFallthroughOp extends JavaOp
            implements Op.BodyTerminating, Op.Lowerable {
        static final String NAME = "java.switch.fallthrough";

        SwitchFallthroughOp(ExternalizedOp def) {
            this();
        }

        SwitchFallthroughOp(SwitchFallthroughOp that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public SwitchFallthroughOp transform(CopyContext cc, OpTransformer ot) {
            return new SwitchFallthroughOp(this, cc);
        }

        SwitchFallthroughOp() {
            super(NAME, List.of());
        }

        @Override
        public TypeElement resultType() {
            return VOID;
        }

        @Override
        public Block.Builder lower(Block.Builder b, OpTransformer opT) {
            return lower(b, BranchTarget::continueBlock);
        }

        Block.Builder lower(Block.Builder b, Function<BranchTarget, Block.Builder> f) {
            BranchTarget t = getBranchTarget(b.context(), parentBlock().parentBody());
            if (t != null) {
                b.op(branch(f.apply(t).successor()));
            } else {
                throw new IllegalStateException("No branch target for operation: " + this);
            }
            return b;
        }
    }

    /**
     * The for operation, that can model a Java language for statement.
     */
    @OpDeclaration(ForOp.NAME)
    public static final class ForOp extends JavaOp
            implements Op.Loop, Op.Lowerable, JavaStatement {

        public static final class InitBuilder {
            final Body.Builder ancestorBody;
            final List<? extends TypeElement> initTypes;

            InitBuilder(Body.Builder ancestorBody,
                        List<? extends TypeElement> initTypes) {
                this.ancestorBody = ancestorBody;
                this.initTypes = initTypes.stream().map(CoreType::varType).toList();
            }

            public ForOp.CondBuilder init(Consumer<Block.Builder> c) {
                Body.Builder init = Body.Builder.of(ancestorBody,
                        CoreType.functionType(CoreType.tupleType(initTypes)));
                c.accept(init.entryBlock());

                return new CondBuilder(ancestorBody, initTypes, init);
            }
        }

        public static final class CondBuilder {
            final Body.Builder ancestorBody;
            final List<? extends TypeElement> initTypes;
            final Body.Builder init;

            public CondBuilder(Body.Builder ancestorBody,
                               List<? extends TypeElement> initTypes,
                               Body.Builder init) {
                this.ancestorBody = ancestorBody;
                this.initTypes = initTypes;
                this.init = init;
            }

            public ForOp.UpdateBuilder cond(Consumer<Block.Builder> c) {
                Body.Builder cond = Body.Builder.of(ancestorBody,
                        CoreType.functionType(BOOLEAN, initTypes));
                c.accept(cond.entryBlock());

                return new UpdateBuilder(ancestorBody, initTypes, init, cond);
            }
        }

        public static final class UpdateBuilder {
            final Body.Builder ancestorBody;
            final List<? extends TypeElement> initTypes;
            final Body.Builder init;
            final Body.Builder cond;

            public UpdateBuilder(Body.Builder ancestorBody,
                                 List<? extends TypeElement> initTypes,
                                 Body.Builder init, Body.Builder cond) {
                this.ancestorBody = ancestorBody;
                this.initTypes = initTypes;
                this.init = init;
                this.cond = cond;
            }

            public ForOp.BodyBuilder cond(Consumer<Block.Builder> c) {
                Body.Builder update = Body.Builder.of(ancestorBody,
                        CoreType.functionType(VOID, initTypes));
                c.accept(update.entryBlock());

                return new BodyBuilder(ancestorBody, initTypes, init, cond, update);
            }

        }

        public static final class BodyBuilder {
            final Body.Builder ancestorBody;
            final List<? extends TypeElement> initTypes;
            final Body.Builder init;
            final Body.Builder cond;
            final Body.Builder update;

            public BodyBuilder(Body.Builder ancestorBody,
                               List<? extends TypeElement> initTypes,
                               Body.Builder init, Body.Builder cond, Body.Builder update) {
                this.ancestorBody = ancestorBody;
                this.initTypes = initTypes;
                this.init = init;
                this.cond = cond;
                this.update = update;
            }

            public ForOp body(Consumer<Block.Builder> c) {
                Body.Builder body = Body.Builder.of(ancestorBody,
                        CoreType.functionType(VOID, initTypes));
                c.accept(body.entryBlock());

                return new ForOp(init, cond, update, body);
            }
        }

        static final String NAME = "java.for";

        final Body init;
        final Body cond;
        final Body update;
        final Body body;

        static ForOp create(ExternalizedOp def) {
            return new ForOp(def);
        }

        ForOp(ExternalizedOp def) {
            this(def.bodyDefinitions().get(0),
                    def.bodyDefinitions().get(1),
                    def.bodyDefinitions().get(2),
                    def.bodyDefinitions().get(3));
        }

        ForOp(ForOp that, CopyContext cc, OpTransformer ot) {
            super(that, cc);

            this.init = that.init.transform(cc, ot).build(this);
            this.cond = that.cond.transform(cc, ot).build(this);
            this.update = that.update.transform(cc, ot).build(this);
            this.body = that.body.transform(cc, ot).build(this);
        }

        @Override
        public ForOp transform(CopyContext cc, OpTransformer ot) {
            return new ForOp(this, cc, ot);
        }

        ForOp(Body.Builder initC,
              Body.Builder condC,
              Body.Builder updateC,
              Body.Builder bodyC) {
            super(NAME, List.of());

            this.init = initC.build(this);

            this.cond = condC.build(this);

            this.update = updateC.build(this);
            if (!update.bodyType().returnType().equals(VOID)) {
                throw new IllegalArgumentException("Update should return void: " + update.bodyType());
            }

            this.body = bodyC.build(this);
            if (!body.bodyType().returnType().equals(VOID)) {
                throw new IllegalArgumentException("Body should return void: " + body.bodyType());
            }
        }

        @Override
        public List<Body> bodies() {
            return List.of(init, cond, update, body);
        }

        public Body init() {
            return init;
        }

        public Body cond() {
            return cond;
        }

        public Body update() {
            return update;
        }

        @Override
        public Body loopBody() {
            return body;
        }

        @Override
        public Block.Builder lower(Block.Builder b, OpTransformer opT) {
            Block.Builder header = b.block();
            Block.Builder body = b.block();
            Block.Builder update = b.block();
            Block.Builder exit = b.block();

            List<Value> initValues = new ArrayList<>();
            // @@@ Init body has one yield operation yielding
            //  void, a single variable, or a tuple of one or more variables
            b.transformBody(init, List.of(), opT.andThen((block, op) -> {
                if (op instanceof CoreOp.TupleOp) {
                    // Drop Tuple if a yielded
                    boolean isResult = op.result().uses().size() == 1 &&
                            op.result().uses().stream().allMatch(r -> r.op() instanceof CoreOp.YieldOp);
                    if (!isResult) {
                        block.op(op);
                    }
                } else if (op instanceof CoreOp.YieldOp yop) {
                    if (yop.yieldValue() == null) {
                        block.op(branch(header.successor()));
                        return block;
                    } else if (yop.yieldValue() instanceof Result or) {
                        if (or.op() instanceof CoreOp.TupleOp top) {
                            initValues.addAll(block.context().getValues(top.operands()));
                        } else {
                            initValues.addAll(block.context().getValues(yop.operands()));
                        }
                        block.op(branch(header.successor()));
                        return block;
                    }

                    throw new IllegalStateException("Bad yield operation");
                } else {
                    // @@@ Composition of lowerable ops
                    block.op(op);
                }
                return block;
            }));

            header.transformBody(cond, initValues, opT.andThen((block, op) -> {
                if (op instanceof CoreOp.YieldOp yo) {
                    block.op(conditionalBranch(block.context().getValue(yo.yieldValue()),
                            body.successor(), exit.successor()));
                } else if (op instanceof Lowerable lop) {
                    // @@@ Composition of lowerable ops
                    block = lop.lower(block, opT);
                } else {
                    block.op(op);
                }
                return block;
            }));

            setBranchTarget(b.context(), this, new BranchTarget(exit, update));

            body.transformBody(this.body, initValues, opT.andThen((block, op) -> {
                // @@@ Composition of lowerable ops
                if (op instanceof Lowerable lop) {
                    block = lop.lower(block, opT);
                } else {
                    block.op(op);
                }
                return block;
            }));

            update.transformBody(this.update, initValues, opT.andThen((block, op) -> {
                if (op instanceof CoreOp.YieldOp) {
                    block.op(branch(header.successor()));
                } else {
                    // @@@ Composition of lowerable ops
                    block.op(op);
                }
                return block;
            }));

            return exit;
        }

        @Override
        public TypeElement resultType() {
            return VOID;
        }
    }

    /**
     * The enhanced for operation, that can model a Java language enhanced for statement.
     */
    @OpDeclaration(EnhancedForOp.NAME)
    public static final class EnhancedForOp extends JavaOp
            implements Op.Loop, Op.Lowerable, JavaStatement {

        public static final class ExpressionBuilder {
            final Body.Builder ancestorBody;
            final TypeElement iterableType;
            final TypeElement elementType;

            ExpressionBuilder(Body.Builder ancestorBody,
                              TypeElement iterableType, TypeElement elementType) {
                this.ancestorBody = ancestorBody;
                this.iterableType = iterableType;
                this.elementType = elementType;
            }

            public DefinitionBuilder expression(Consumer<Block.Builder> c) {
                Body.Builder expression = Body.Builder.of(ancestorBody,
                        CoreType.functionType(iterableType));
                c.accept(expression.entryBlock());

                return new DefinitionBuilder(ancestorBody, elementType, expression);
            }
        }

        public static final class DefinitionBuilder {
            final Body.Builder ancestorBody;
            final TypeElement elementType;
            final Body.Builder expression;

            DefinitionBuilder(Body.Builder ancestorBody,
                              TypeElement elementType, Body.Builder expression) {
                this.ancestorBody = ancestorBody;
                this.elementType = elementType;
                this.expression = expression;
            }

            public BodyBuilder definition(Consumer<Block.Builder> c) {
                return definition(elementType, c);
            }

            public BodyBuilder definition(TypeElement bodyElementType, Consumer<Block.Builder> c) {
                Body.Builder definition = Body.Builder.of(ancestorBody,
                        CoreType.functionType(bodyElementType, elementType));
                c.accept(definition.entryBlock());

                return new BodyBuilder(ancestorBody, elementType, expression, definition);
            }
        }

        public static final class BodyBuilder {
            final Body.Builder ancestorBody;
            final TypeElement elementType;
            final Body.Builder expression;
            final Body.Builder definition;

            BodyBuilder(Body.Builder ancestorBody,
                        TypeElement elementType, Body.Builder expression, Body.Builder definition) {
                this.ancestorBody = ancestorBody;
                this.elementType = elementType;
                this.expression = expression;
                this.definition = definition;
            }

            public EnhancedForOp body(Consumer<Block.Builder> c) {
                Body.Builder body = Body.Builder.of(ancestorBody,
                        CoreType.functionType(VOID, elementType));
                c.accept(body.entryBlock());

                return new EnhancedForOp(expression, definition, body);
            }
        }

        static final String NAME = "java.enhancedFor";

        final Body expression;
        final Body init;
        final Body body;

        static EnhancedForOp create(ExternalizedOp def) {
            return new EnhancedForOp(def);
        }

        EnhancedForOp(ExternalizedOp def) {
            this(def.bodyDefinitions().get(0),
                    def.bodyDefinitions().get(1),
                    def.bodyDefinitions().get(2));
        }

        EnhancedForOp(EnhancedForOp that, CopyContext cc, OpTransformer ot) {
            super(that, cc);

            this.expression = that.expression.transform(cc, ot).build(this);
            this.init = that.init.transform(cc, ot).build(this);
            this.body = that.body.transform(cc, ot).build(this);
        }

        @Override
        public EnhancedForOp transform(CopyContext cc, OpTransformer ot) {
            return new EnhancedForOp(this, cc, ot);
        }

        EnhancedForOp(Body.Builder expressionC, Body.Builder initC, Body.Builder bodyC) {
            super(NAME, List.of());

            this.expression = expressionC.build(this);
            if (expression.bodyType().returnType().equals(VOID)) {
                throw new IllegalArgumentException("Expression should return non-void value: " + expression.bodyType());
            }
            if (!expression.bodyType().parameterTypes().isEmpty()) {
                throw new IllegalArgumentException("Expression should have zero parameters: " + expression.bodyType());
            }

            this.init = initC.build(this);
            if (init.bodyType().returnType().equals(VOID)) {
                throw new IllegalArgumentException("Initialization should return non-void value: " + init.bodyType());
            }
            if (init.bodyType().parameterTypes().size() != 1) {
                throw new IllegalArgumentException("Initialization should have one parameter: " + init.bodyType());
            }

            this.body = bodyC.build(this);
            if (!body.bodyType().returnType().equals(VOID)) {
                throw new IllegalArgumentException("Body should return void: " + body.bodyType());
            }
            if (body.bodyType().parameterTypes().size() != 1) {
                throw new IllegalArgumentException("Body should have one parameter: " + body.bodyType());
            }
        }

        @Override
        public List<Body> bodies() {
            return List.of(expression, init, body);
        }

        public Body expression() {
            return expression;
        }

        public Body initialization() {
            return init;
        }

        @Override
        public Body loopBody() {
            return body;
        }

        static final MethodRef ITERABLE_ITERATOR = MethodRef.method(Iterable.class, "iterator", Iterator.class);
        static final MethodRef ITERATOR_HAS_NEXT = MethodRef.method(Iterator.class, "hasNext", boolean.class);
        static final MethodRef ITERATOR_NEXT = MethodRef.method(Iterator.class, "next", Object.class);

        @Override
        public Block.Builder lower(Block.Builder b, OpTransformer opT) {
            JavaType elementType = (JavaType) init.entryBlock().parameters().get(0).type();
            boolean isArray = expression.bodyType().returnType() instanceof ArrayType;

            Block.Builder preHeader = b.block(expression.bodyType().returnType());
            Block.Builder header = b.block(isArray ? List.of(INT) : List.of());
            Block.Builder init = b.block();
            Block.Builder body = b.block();
            Block.Builder exit = b.block();

            b.transformBody(expression, List.of(), opT.andThen((block, op) -> {
                if (op instanceof CoreOp.YieldOp yop) {
                    Value loopSource = block.context().getValue(yop.yieldValue());
                    block.op(branch(preHeader.successor(loopSource)));
                } else {
                    // @@@ Composition of lowerable ops
                    block.op(op);
                }
                return block;
            }));

            if (isArray) {
                Value array = preHeader.parameters().get(0);
                Value arrayLength = preHeader.op(arrayLength(array));
                Value i = preHeader.op(constant(INT, 0));
                preHeader.op(branch(header.successor(i)));

                i = header.parameters().get(0);
                Value p = header.op(lt(i, arrayLength));
                header.op(conditionalBranch(p, init.successor(), exit.successor()));

                Value e = init.op(arrayLoadOp(array, i));
                List<Value> initValues = new ArrayList<>();
                // @@@ Init body has one yield operation yielding a single variable
                init.transformBody(this.init, List.of(e), (block, op) -> {
                    if (op instanceof CoreOp.YieldOp yop) {
                        initValues.addAll(block.context().getValues(yop.operands()));
                        block.op(branch(body.successor()));
                    } else {
                        // @@@ Composition of lowerable ops
                        block.op(op);
                    }
                    return block;
                });

                Block.Builder update = b.block();
                setBranchTarget(b.context(), this, new BranchTarget(exit, update));

                body.transformBody(this.body, initValues, opT.andThen((block, op) -> {
                    // @@@ Composition of lowerable ops
                    if (op instanceof Lowerable lop) {
                        block = lop.lower(block, opT);
                    } else {
                        block.op(op);
                    }
                    return block;
                }));

                i = update.op(add(i, update.op(constant(INT, 1))));
                update.op(branch(header.successor(i)));
            } else {
                JavaType iterable = parameterized(type(Iterator.class), elementType);
                Value iterator = preHeader.op(invoke(iterable, ITERABLE_ITERATOR, preHeader.parameters().get(0)));
                preHeader.op(branch(header.successor()));

                Value p = header.op(invoke(ITERATOR_HAS_NEXT, iterator));
                header.op(conditionalBranch(p, init.successor(), exit.successor()));

                Value e = init.op(invoke(elementType, ITERATOR_NEXT, iterator));
                List<Value> initValues = new ArrayList<>();
                init.transformBody(this.init, List.of(e), opT.andThen((block, op) -> {
                    if (op instanceof CoreOp.YieldOp yop) {
                        initValues.addAll(block.context().getValues(yop.operands()));
                        block.op(branch(body.successor()));
                    } else {
                        // @@@ Composition of lowerable ops
                        block.op(op);
                    }
                    return block;
                }));

                setBranchTarget(b.context(), this, new BranchTarget(exit, header));

                body.transformBody(this.body, initValues, opT.andThen((block, op) -> {
                    // @@@ Composition of lowerable ops
                    if (op instanceof Lowerable lop) {
                        block = lop.lower(block, opT);
                    } else {
                        block.op(op);
                    }
                    return block;
                }));
            }

            return exit;
        }

        @Override
        public TypeElement resultType() {
            return VOID;
        }
    }

    /**
     * The while operation, that can model a Java language while statement.
     */
    @OpDeclaration(WhileOp.NAME)
    public static final class WhileOp extends JavaOp
            implements Op.Loop, Op.Lowerable, JavaStatement {

        public static class PredicateBuilder {
            final Body.Builder ancestorBody;

            PredicateBuilder(Body.Builder ancestorBody) {
                this.ancestorBody = ancestorBody;
            }

            public WhileOp.BodyBuilder predicate(Consumer<Block.Builder> c) {
                Body.Builder body = Body.Builder.of(ancestorBody, CoreType.functionType(BOOLEAN));
                c.accept(body.entryBlock());

                return new WhileOp.BodyBuilder(ancestorBody, body);
            }
        }

        public static class BodyBuilder {
            final Body.Builder ancestorBody;
            private final Body.Builder predicate;

            BodyBuilder(Body.Builder ancestorBody, Body.Builder predicate) {
                this.ancestorBody = ancestorBody;
                this.predicate = predicate;
            }

            public WhileOp body(Consumer<Block.Builder> c) {
                Body.Builder body = Body.Builder.of(ancestorBody, CoreType.FUNCTION_TYPE_VOID);
                c.accept(body.entryBlock());

                return new WhileOp(List.of(predicate, body));
            }
        }

        private static final String NAME = "java.while";

        private final List<Body> bodies;

        WhileOp(ExternalizedOp def) {
            this(def.bodyDefinitions());
        }

        WhileOp(List<Body.Builder> bodyCs) {
            super(NAME, List.of());

            this.bodies = bodyCs.stream().map(bc -> bc.build(this)).toList();
        }

        WhileOp(Body.Builder predicate, Body.Builder body) {
            super(NAME, List.of());

            Objects.requireNonNull(body);

            this.bodies = Stream.of(predicate, body).filter(Objects::nonNull)
                    .map(bc -> bc.build(this)).toList();

            // @@@ This will change with pattern bindings
            if (!bodies.get(0).bodyType().equals(CoreType.functionType(BOOLEAN))) {
                throw new IllegalArgumentException(
                        "Predicate body descriptor should be " + CoreType.functionType(BOOLEAN) +
                                " but is " + bodies.get(0).bodyType());
            }
            if (!bodies.get(1).bodyType().equals(CoreType.FUNCTION_TYPE_VOID)) {
                throw new IllegalArgumentException(
                        "Body descriptor should be " + CoreType.functionType(VOID) +
                                " but is " + bodies.get(1).bodyType());
            }
        }

        WhileOp(WhileOp that, CopyContext cc, OpTransformer ot) {
            super(that, cc);

            this.bodies = that.bodies.stream()
                    .map(b -> b.transform(cc, ot).build(this)).toList();
        }

        @Override
        public WhileOp transform(CopyContext cc, OpTransformer ot) {
            return new WhileOp(this, cc, ot);
        }

        @Override
        public List<Body> bodies() {
            return bodies;
        }

        public Body predicateBody() {
            return bodies.get(0);
        }

        @Override
        public Body loopBody() {
            return bodies.get(1);
        }

        @Override
        public Block.Builder lower(Block.Builder b, OpTransformer opT) {
            Block.Builder header = b.block();
            Block.Builder body = b.block();
            Block.Builder exit = b.block();

            b.op(branch(header.successor()));

            header.transformBody(predicateBody(), List.of(), opT.andThen((block, op) -> {
                if (op instanceof CoreOp.YieldOp yo) {
                    block.op(conditionalBranch(block.context().getValue(yo.yieldValue()),
                            body.successor(), exit.successor()));
                } else if (op instanceof Lowerable lop) {
                    // @@@ Composition of lowerable ops
                    block = lop.lower(block, opT);
                } else {
                    block.op(op);
                }
                return block;
            }));

            setBranchTarget(b.context(), this, new BranchTarget(exit, header));

            body.transformBody(loopBody(), List.of(), opT.andThen((block, op) -> {
                // @@@ Composition of lowerable ops
                if (op instanceof Lowerable lop) {
                    block = lop.lower(block, opT);
                } else {
                    block.op(op);
                }
                return block;
            }));

            return exit;
        }

        @Override
        public TypeElement resultType() {
            return VOID;
        }
    }

    /**
     * The do-while operation, that can model a Java language do statement.
     */
    // @@@ Unify JavaDoWhileOp and JavaWhileOp with common abstract superclass
    @OpDeclaration(DoWhileOp.NAME)
    public static final class DoWhileOp extends JavaOp
            implements Op.Loop, Op.Lowerable, JavaStatement {

        public static class PredicateBuilder {
            final Body.Builder ancestorBody;
            private final Body.Builder body;

            PredicateBuilder(Body.Builder ancestorBody, Body.Builder body) {
                this.ancestorBody = ancestorBody;
                this.body = body;
            }

            public DoWhileOp predicate(Consumer<Block.Builder> c) {
                Body.Builder predicate = Body.Builder.of(ancestorBody, CoreType.functionType(BOOLEAN));
                c.accept(predicate.entryBlock());

                return new DoWhileOp(List.of(body, predicate));
            }
        }

        public static class BodyBuilder {
            final Body.Builder ancestorBody;

            BodyBuilder(Body.Builder ancestorBody) {
                this.ancestorBody = ancestorBody;
            }

            public DoWhileOp.PredicateBuilder body(Consumer<Block.Builder> c) {
                Body.Builder body = Body.Builder.of(ancestorBody, CoreType.FUNCTION_TYPE_VOID);
                c.accept(body.entryBlock());

                return new DoWhileOp.PredicateBuilder(ancestorBody, body);
            }
        }

        private static final String NAME = "java.do.while";

        private final List<Body> bodies;

        DoWhileOp(ExternalizedOp def) {
            this(def.bodyDefinitions());
        }

        DoWhileOp(List<Body.Builder> bodyCs) {
            super(NAME, List.of());

            this.bodies = bodyCs.stream().map(bc -> bc.build(this)).toList();
        }

        DoWhileOp(Body.Builder body, Body.Builder predicate) {
            super(NAME, List.of());

            Objects.requireNonNull(body);

            this.bodies = Stream.of(body, predicate).filter(Objects::nonNull)
                    .map(bc -> bc.build(this)).toList();

            if (!bodies.get(0).bodyType().equals(CoreType.FUNCTION_TYPE_VOID)) {
                throw new IllegalArgumentException(
                        "Body descriptor should be " + CoreType.functionType(VOID) +
                                " but is " + bodies.get(1).bodyType());
            }
            if (!bodies.get(1).bodyType().equals(CoreType.functionType(BOOLEAN))) {
                throw new IllegalArgumentException(
                        "Predicate body descriptor should be " + CoreType.functionType(BOOLEAN) +
                                " but is " + bodies.get(0).bodyType());
            }
        }

        DoWhileOp(DoWhileOp that, CopyContext cc, OpTransformer ot) {
            super(that, cc);

            this.bodies = that.bodies.stream()
                    .map(b -> b.transform(cc, ot).build(this)).toList();
        }

        @Override
        public DoWhileOp transform(CopyContext cc, OpTransformer ot) {
            return new DoWhileOp(this, cc, ot);
        }

        @Override
        public List<Body> bodies() {
            return bodies;
        }

        public Body predicateBody() {
            return bodies.get(1);
        }

        @Override
        public Body loopBody() {
            return bodies.get(0);
        }

        @Override
        public Block.Builder lower(Block.Builder b, OpTransformer opT) {
            Block.Builder body = b.block();
            Block.Builder header = b.block();
            Block.Builder exit = b.block();

            b.op(branch(body.successor()));

            setBranchTarget(b.context(), this, new BranchTarget(exit, header));

            body.transformBody(loopBody(), List.of(), opT.andThen((block, op) -> {
                // @@@ Composition of lowerable ops
                if (op instanceof Lowerable lop) {
                    block = lop.lower(block, opT);
                } else {
                    block.op(op);
                }
                return block;
            }));

            header.transformBody(predicateBody(), List.of(), opT.andThen((block, op) -> {
                if (op instanceof CoreOp.YieldOp yo) {
                    block.op(conditionalBranch(block.context().getValue(yo.yieldValue()),
                            body.successor(), exit.successor()));
                } else if (op instanceof Lowerable lop) {
                    // @@@ Composition of lowerable ops
                    block = lop.lower(block, opT);
                } else {
                    block.op(op);
                }
                return block;
            }));

            return exit;
        }

        @Override
        public TypeElement resultType() {
            return VOID;
        }
    }

    /**
     * The conditional-and-or operation, that can model Java language condition-or or conditional-and expressions.
     */
    public sealed static abstract class JavaConditionalOp extends JavaOp
            implements Op.Nested, Op.Lowerable, JavaExpression {
        final List<Body> bodies;

        JavaConditionalOp(JavaConditionalOp that, CopyContext cc, OpTransformer ot) {
            super(that, cc);

            // Copy body
            this.bodies = that.bodies.stream().map(b -> b.transform(cc, ot).build(this)).toList();
        }

        JavaConditionalOp(String name, List<Body.Builder> bodyCs) {
            super(name, List.of());

            if (bodyCs.isEmpty()) {
                throw new IllegalArgumentException();
            }

            this.bodies = bodyCs.stream().map(bc -> bc.build(this)).toList();
            for (Body b : bodies) {
                if (!b.bodyType().equals(CoreType.functionType(BOOLEAN))) {
                    throw new IllegalArgumentException("Body conditional body descriptor: " + b.bodyType());
                }
            }
        }

        @Override
        public List<Body> bodies() {
            return bodies;
        }

        static Block.Builder lower(Block.Builder startBlock, OpTransformer opT, JavaConditionalOp cop) {
            List<Body> bodies = cop.bodies();

            Block.Builder exit = startBlock.block();
            TypeElement oprType = cop.result().type();
            Block.Parameter arg = exit.parameter(oprType);
            startBlock.context().mapValue(cop.result(), arg);

            // Transform bodies in reverse order
            // This makes available the blocks to be referenced as successors in prior blocks

            Block.Builder pred = null;
            for (int i = bodies.size() - 1; i >= 0; i--) {
                OpTransformer opt;
                if (i == bodies.size() - 1) {
                    opt = (block, op) -> {
                        if (op instanceof CoreOp.YieldOp yop) {
                            Value p = block.context().getValue(yop.yieldValue());
                            block.op(branch(exit.successor(p)));
                        } else if (op instanceof Lowerable lop) {
                            // @@@ Composition of lowerable ops
                            block = lop.lower(block, opT);
                        } else {
                            // Copy
                            block.apply(op);
                        }
                        return block;
                    };
                } else {
                    Block.Builder nextPred = pred;
                    opt = (block, op) -> {
                        if (op instanceof CoreOp.YieldOp yop) {
                            Value p = block.context().getValue(yop.yieldValue());
                            if (cop instanceof ConditionalAndOp) {
                                block.op(conditionalBranch(p, nextPred.successor(), exit.successor(p)));
                            } else {
                                block.op(conditionalBranch(p, exit.successor(p), nextPred.successor()));
                            }
                        } else if (op instanceof Lowerable lop) {
                            // @@@ Composition of lowerable ops
                            block = lop.lower(block, opT);
                        } else {
                            // Copy
                            block.apply(op);
                        }
                        return block;
                    };
                }

                Body fromPred = bodies.get(i);
                if (i == 0) {
                    startBlock.transformBody(fromPred, List.of(), opt);
                } else {
                    pred = startBlock.block(fromPred.bodyType().parameterTypes());
                    pred.transformBody(fromPred, pred.parameters(), opT.andThen(opt));
                }
            }

            return exit;
        }

        @Override
        public TypeElement resultType() {
            return BOOLEAN;
        }
    }

    /**
     * The conditional-and operation, that can model Java language conditional-and expressions.
     */
    @OpDeclaration(ConditionalAndOp.NAME)
    public static final class ConditionalAndOp extends JavaConditionalOp {

        public static class Builder {
            final Body.Builder ancestorBody;
            final List<Body.Builder> bodies;

            Builder(Body.Builder ancestorBody, Consumer<Block.Builder> lhs, Consumer<Block.Builder> rhs) {
                this.ancestorBody = ancestorBody;
                this.bodies = new ArrayList<>();
                and(lhs);
                and(rhs);
            }

            public Builder and(Consumer<Block.Builder> c) {
                Body.Builder body = Body.Builder.of(ancestorBody, CoreType.functionType(BOOLEAN));
                c.accept(body.entryBlock());
                bodies.add(body);

                return this;
            }

            public ConditionalAndOp build() {
                return new ConditionalAndOp(bodies);
            }
        }

        static final String NAME = "java.cand";

        ConditionalAndOp(ExternalizedOp def) {
            this(def.bodyDefinitions());
        }

        ConditionalAndOp(ConditionalAndOp that, CopyContext cc, OpTransformer ot) {
            super(that, cc, ot);
        }

        @Override
        public ConditionalAndOp transform(CopyContext cc, OpTransformer ot) {
            return new ConditionalAndOp(this, cc, ot);
        }

        ConditionalAndOp(List<Body.Builder> bodyCs) {
            super(NAME, bodyCs);
        }

        @Override
        public Block.Builder lower(Block.Builder b, OpTransformer opT) {
            return lower(b, opT, this);
        }
    }

    /**
     * The conditional-or operation, that can model Java language conditional-or expressions.
     */
    @OpDeclaration(ConditionalOrOp.NAME)
    public static final class ConditionalOrOp extends JavaConditionalOp {

        public static class Builder {
            final Body.Builder ancestorBody;
            final List<Body.Builder> bodies;

            Builder(Body.Builder ancestorBody, Consumer<Block.Builder> lhs, Consumer<Block.Builder> rhs) {
                this.ancestorBody = ancestorBody;
                this.bodies = new ArrayList<>();
                or(lhs);
                or(rhs);
            }

            public Builder or(Consumer<Block.Builder> c) {
                Body.Builder body = Body.Builder.of(ancestorBody, CoreType.functionType(BOOLEAN));
                c.accept(body.entryBlock());
                bodies.add(body);

                return this;
            }

            public ConditionalOrOp build() {
                return new ConditionalOrOp(bodies);
            }
        }

        static final String NAME = "java.cor";

        ConditionalOrOp(ExternalizedOp def) {
            this(def.bodyDefinitions());
        }

        ConditionalOrOp(ConditionalOrOp that, CopyContext cc, OpTransformer ot) {
            super(that, cc, ot);
        }

        @Override
        public ConditionalOrOp transform(CopyContext cc, OpTransformer ot) {
            return new ConditionalOrOp(this, cc, ot);
        }

        ConditionalOrOp(List<Body.Builder> bodyCs) {
            super(NAME, bodyCs);
        }

        @Override
        public Block.Builder lower(Block.Builder b, OpTransformer opT) {
            return lower(b, opT, this);
        }
    }

    /**
     * The conditional operation, that can model Java language conditional operator {@code ?} expressions.
     */
    @OpDeclaration(ConditionalExpressionOp.NAME)
    public static final class ConditionalExpressionOp extends JavaOp
            implements Op.Nested, Op.Lowerable, JavaExpression {

        static final String NAME = "java.cexpression";

        final TypeElement resultType;
        // {cond, truepart, falsepart}
        final List<Body> bodies;

        ConditionalExpressionOp(ExternalizedOp def) {
            if (!def.operands().isEmpty()) {
                throw new IllegalStateException("Operation must have no operands");
            }

            this(def.resultType(), def.bodyDefinitions());
        }

        ConditionalExpressionOp(ConditionalExpressionOp that, CopyContext cc, OpTransformer ot) {
            super(that, cc);

            // Copy body
            this.bodies = that.bodies.stream()
                    .map(b -> b.transform(cc, ot).build(this)).toList();
            this.resultType = that.resultType;
        }

        @Override
        public ConditionalExpressionOp transform(CopyContext cc, OpTransformer ot) {
            return new ConditionalExpressionOp(this, cc, ot);
        }

        ConditionalExpressionOp(TypeElement expressionType, List<Body.Builder> bodyCs) {
            super(NAME, List.of());

            this.bodies = bodyCs.stream().map(bc -> bc.build(this)).toList();
            // @@@ when expressionType is null, we assume truepart and falsepart have the same yieldType
            this.resultType = expressionType == null ? bodies.get(1).yieldType() : expressionType;

            if (bodies.size() < 3) {
                throw new IllegalArgumentException("Incorrect number of bodies: " + bodies.size());
            }

            Body cond = bodies.get(0);
            if (!cond.bodyType().equals(CoreType.functionType(BOOLEAN))) {
                throw new IllegalArgumentException("Illegal cond body descriptor: " + cond.bodyType());
            }
        }

        @Override
        public List<Body> bodies() {
            return bodies;
        }

        @Override
        public Block.Builder lower(Block.Builder b, OpTransformer opT) {
            Block.Builder exit = b.block(resultType());
            exit.context().mapValue(result(), exit.parameters().get(0));

            setBranchTarget(b.context(), this, new BranchTarget(exit, null));

            List<Block.Builder> builders = List.of(b.block(), b.block());
            b.transformBody(bodies.get(0), List.of(), opT.andThen((block, op) -> {
                if (op instanceof CoreOp.YieldOp yo) {
                    block.op(conditionalBranch(block.context().getValue(yo.yieldValue()),
                            builders.get(0).successor(), builders.get(1).successor()));
                } else if (op instanceof Lowerable lop) {
                    // @@@ Composition of lowerable ops
                    block = lop.lower(block, opT);
                } else {
                    block.op(op);
                }
                return block;
            }));

            for (int i = 0; i < 2; i++) {
                builders.get(i).transformBody(bodies.get(i + 1), List.of(), opT.andThen((block, op) -> {
                    if (op instanceof CoreOp.YieldOp yop) {
                        block.op(branch(exit.successor(block.context().getValue(yop.yieldValue()))));
                    } else if (op instanceof Lowerable lop) {
                        // @@@ Composition of lowerable ops
                        block = lop.lower(block, opT);
                    } else {
                        block.op(op);
                    }
                    return block;
                }));
            }

            return exit;
        }

        @Override
        public TypeElement resultType() {
            return resultType;
        }
    }

    /**
     * The try operation, that can model Java language try statements.
     */
    @OpDeclaration(TryOp.NAME)
    public static final class TryOp extends JavaOp
            implements Op.Nested, Op.Lowerable, JavaStatement {

        public static final class BodyBuilder {
            final Body.Builder ancestorBody;
            final List<? extends TypeElement> resourceTypes;
            final Body.Builder resources;

            BodyBuilder(Body.Builder ancestorBody, List<? extends TypeElement> resourceTypes, Body.Builder resources) {
                this.ancestorBody = ancestorBody;
                this.resourceTypes = resourceTypes;
                this.resources = resources;
            }

            public CatchBuilder body(Consumer<Block.Builder> c) {
                Body.Builder body = Body.Builder.of(ancestorBody,
                        CoreType.functionType(VOID, resourceTypes));
                c.accept(body.entryBlock());

                return new CatchBuilder(ancestorBody, resources, body);
            }
        }

        public static final class CatchBuilder {
            final Body.Builder ancestorBody;
            final Body.Builder resources;
            final Body.Builder body;
            final List<Body.Builder> catchers;

            CatchBuilder(Body.Builder ancestorBody, Body.Builder resources, Body.Builder body) {
                this.ancestorBody = ancestorBody;
                this.resources = resources;
                this.body = body;
                this.catchers = new ArrayList<>();
            }

            // @@@ multi-catch
            public CatchBuilder catch_(TypeElement exceptionType, Consumer<Block.Builder> c) {
                Body.Builder _catch = Body.Builder.of(ancestorBody,
                        CoreType.functionType(VOID, exceptionType));
                c.accept(_catch.entryBlock());
                catchers.add(_catch);

                return this;
            }

            public TryOp finally_(Consumer<Block.Builder> c) {
                Body.Builder _finally = Body.Builder.of(ancestorBody, CoreType.FUNCTION_TYPE_VOID);
                c.accept(_finally.entryBlock());

                return new TryOp(resources, body, catchers, _finally);
            }

            public TryOp noFinalizer() {
                return new TryOp(resources, body, catchers, null);
            }
        }

        static final String NAME = "java.try";

        final Body resources;
        final Body body;
        final List<Body> catchers;
        final Body finalizer;

        static TryOp create(ExternalizedOp def) {
            return new TryOp(def);
        }

        TryOp(ExternalizedOp def) {
            List<Body.Builder> bodies = def.bodyDefinitions();
            Body.Builder first = bodies.getFirst();
            Body.Builder resources;
            Body.Builder body;
            if (first.bodyType().returnType().equals(VOID)) {
                resources = null;
                body = first;
            } else {
                resources = first;
                body = bodies.get(1);
            }

            Body.Builder last = bodies.getLast();
            Body.Builder finalizer;
            if (last.bodyType().parameterTypes().isEmpty()) {
                finalizer = last;
            } else {
                finalizer = null;
            }
            List<Body.Builder> catchers = bodies.subList(
                    resources == null ? 1 : 2,
                    bodies.size() - (finalizer == null ? 0 : 1));

            this(resources, body, catchers, finalizer);
        }

        TryOp(TryOp that, CopyContext cc, OpTransformer ot) {
            super(that, cc);

            if (that.resources != null) {
                this.resources = that.resources.transform(cc, ot).build(this);
            } else {
                this.resources = null;
            }
            this.body = that.body.transform(cc, ot).build(this);
            this.catchers = that.catchers.stream()
                    .map(b -> b.transform(cc, ot).build(this))
                    .toList();
            if (that.finalizer != null) {
                this.finalizer = that.finalizer.transform(cc, ot).build(this);
            } else {
                this.finalizer = null;
            }
        }

        @Override
        public TryOp transform(CopyContext cc, OpTransformer ot) {
            return new TryOp(this, cc, ot);
        }

        TryOp(Body.Builder resourcesC,
              Body.Builder bodyC,
              List<Body.Builder> catchersC,
              Body.Builder finalizerC) {
            super(NAME, List.of());

            if (resourcesC != null) {
                this.resources = resourcesC.build(this);
                if (resources.bodyType().returnType().equals(VOID)) {
                    throw new IllegalArgumentException("Resources should not return void: " + resources.bodyType());
                }
                if (!resources.bodyType().parameterTypes().isEmpty()) {
                    throw new IllegalArgumentException("Resources should have zero parameters: " + resources.bodyType());
                }
            } else {
                this.resources = null;
            }

            this.body = bodyC.build(this);
            if (!body.bodyType().returnType().equals(VOID)) {
                throw new IllegalArgumentException("Try should return void: " + body.bodyType());
            }

            this.catchers = catchersC.stream().map(c -> c.build(this)).toList();
            for (Body _catch : catchers) {
                if (!_catch.bodyType().returnType().equals(VOID)) {
                    throw new IllegalArgumentException("Catch should return void: " + _catch.bodyType());
                }
                if (_catch.bodyType().parameterTypes().size() != 1) {
                    throw new IllegalArgumentException("Catch should have zero parameters: " + _catch.bodyType());
                }
            }

            if (finalizerC != null) {
                this.finalizer = finalizerC.build(this);
                if (!finalizer.bodyType().returnType().equals(VOID)) {
                    throw new IllegalArgumentException("Finally should return void: " + finalizer.bodyType());
                }
                if (!finalizer.bodyType().parameterTypes().isEmpty()) {
                    throw new IllegalArgumentException("Finally should have zero parameters: " + finalizer.bodyType());
                }
            } else {
                this.finalizer = null;
            }
        }

        @Override
        public List<Body> bodies() {
            ArrayList<Body> bodies = new ArrayList<>();
            if (resources != null) {
                bodies.add(resources);
            }
            bodies.add(body);
            bodies.addAll(catchers);
            if (finalizer != null) {
                bodies.add(finalizer);
            }
            return bodies;
        }

        public Body resources() {
            return resources;
        }

        public Body body() {
            return body;
        }

        public List<Body> catchers() {
            return catchers;
        }

        public Body finalizer() {
            return finalizer;
        }

        @Override
        public Block.Builder lower(Block.Builder b, OpTransformer opT) {
            if (resources != null) {
                throw new UnsupportedOperationException("Lowering of try-with-resources is unsupported");
            }

            Block.Builder exit = b.block();
            setBranchTarget(b.context(), this, new BranchTarget(exit, null));

            // Simple case with no catch and finally bodies
            if (catchers.isEmpty() && finalizer == null) {
                b.transformBody(body, List.of(), (block, op) -> {
                    if (op instanceof CoreOp.YieldOp) {
                        block.op(branch(exit.successor()));
                    } else {
                        // @@@ Composition of lowerable ops
                        if (op instanceof Lowerable lop) {
                            block = lop.lower(block, opT);
                        } else {
                            block.op(op);
                        }
                    }
                    return block;
                });
                return exit;
            }

            Block.Builder tryRegionEnter = b.block();
            Block.Builder tryRegionExit = b.block();

            // Construct the catcher block builders
            List<Block.Builder> catchers = catchers().stream()
                    .map(catcher -> b.block())
                    .toList();
            Block.Builder catcherFinally;
            if (finalizer == null) {
                catcherFinally = null;
            } else {
                catcherFinally = b.block();
                catchers = new ArrayList<>(catchers);
                catchers.add(catcherFinally);
            }

            // Enter the try exception region
            List<Block.Reference> exitHandlers = catchers.stream()
                    .map(Block.Builder::successor)
                    .toList();
            b.op(exceptionRegionEnter(tryRegionEnter.successor(), exitHandlers.reversed()));

            OpTransformer tryExitTransformer;
            if (finalizer != null) {
                tryExitTransformer = opT.compose((block, op) -> {
                    if (op instanceof CoreOp.ReturnOp ||
                            (op instanceof JavaOp.JavaLabelOp lop && ifExitFromTry(lop))) {
                        return inlineFinalizer(block, exitHandlers, opT);
                    } else {
                        return block;
                    }
                });
            } else {
                tryExitTransformer = opT.compose((block, op) -> {
                    if (op instanceof CoreOp.ReturnOp ||
                            (op instanceof JavaOp.JavaLabelOp lop && ifExitFromTry(lop))) {
                        Block.Builder tryRegionReturnExit = block.block();
                        block.op(exceptionRegionExit(tryRegionReturnExit.successor(), exitHandlers));
                        return tryRegionReturnExit;
                    } else {
                        return block;
                    }
                });
            }
            // Inline the try body
            AtomicBoolean hasTryRegionExit = new AtomicBoolean();
            tryRegionEnter.transformBody(body, List.of(), tryExitTransformer.andThen((block, op) -> {
                if (op instanceof CoreOp.YieldOp) {
                    hasTryRegionExit.set(true);
                    block.op(branch(tryRegionExit.successor()));
                } else {
                    // @@@ Composition of lowerable ops
                    if (op instanceof Lowerable lop) {
                        block = lop.lower(block, tryExitTransformer);
                    } else {
                        block.op(op);
                    }
                }
                return block;
            }));

            Block.Builder finallyEnter = null;
            if (finalizer != null) {
                finallyEnter = b.block();
                if (hasTryRegionExit.get()) {
                    // Exit the try exception region
                    tryRegionExit.op(exceptionRegionExit(finallyEnter.successor(), exitHandlers));
                }
            } else if (hasTryRegionExit.get()) {
                // Exit the try exception region
                tryRegionExit.op(exceptionRegionExit(exit.successor(), exitHandlers));
            }

            // Inline the catch bodies
            for (int i = 0; i < this.catchers.size(); i++) {
                Block.Builder catcher = catchers.get(i);
                Body catcherBody = this.catchers.get(i);
                // Create the throwable argument
                Block.Parameter t = catcher.parameter(catcherBody.bodyType().parameterTypes().get(0));

                if (finalizer != null) {
                    Block.Builder catchRegionEnter = b.block();
                    Block.Builder catchRegionExit = b.block();

                    // Enter the catch exception region
                    Result catchExceptionRegion = catcher.op(
                            exceptionRegionEnter(catchRegionEnter.successor(), catcherFinally.successor()));

                    OpTransformer catchExitTransformer = opT.compose((block, op) -> {
                        if (op instanceof CoreOp.ReturnOp) {
                            return inlineFinalizer(block, List.of(catcherFinally.successor()), opT);
                        } else if (op instanceof JavaOp.JavaLabelOp lop && ifExitFromTry(lop)) {
                            return inlineFinalizer(block, List.of(catcherFinally.successor()), opT);
                        } else {
                            return block;
                        }
                    });
                    // Inline the catch body
                    AtomicBoolean hasCatchRegionExit = new AtomicBoolean();
                    catchRegionEnter.transformBody(catcherBody, List.of(t), catchExitTransformer.andThen((block, op) -> {
                        if (op instanceof CoreOp.YieldOp) {
                            hasCatchRegionExit.set(true);
                            block.op(branch(catchRegionExit.successor()));
                        } else {
                            // @@@ Composition of lowerable ops
                            if (op instanceof Lowerable lop) {
                                block = lop.lower(block, catchExitTransformer);
                            } else {
                                block.op(op);
                            }
                        }
                        return block;
                    }));

                    // Exit the catch exception region
                    if (hasCatchRegionExit.get()) {
                        hasTryRegionExit.set(true);
                        catchRegionExit.op(exceptionRegionExit(finallyEnter.successor(), catcherFinally.successor()));
                    }
                } else {
                    // Inline the catch body
                    catcher.transformBody(catcherBody, List.of(t), opT.andThen((block, op) -> {
                        if (op instanceof CoreOp.YieldOp) {
                            block.op(branch(exit.successor()));
                        } else {
                            // @@@ Composition of lowerable ops
                            if (op instanceof Lowerable lop) {
                                block = lop.lower(block, opT);
                            } else {
                                block.op(op);
                            }
                        }
                        return block;
                    }));
                }
            }

            if (finalizer != null && hasTryRegionExit.get()) {
                // Inline the finally body
                finallyEnter.transformBody(finalizer, List.of(), opT.andThen((block, op) -> {
                    if (op instanceof CoreOp.YieldOp) {
                        block.op(branch(exit.successor()));
                    } else {
                        // @@@ Composition of lowerable ops
                        if (op instanceof Lowerable lop) {
                            block = lop.lower(block, opT);
                        } else {
                            block.op(op);
                        }
                    }
                    return block;
                }));
            }

            // Inline the finally body as a catcher of Throwable and adjusting to throw
            if (finalizer != null) {
                // Create the throwable argument
                Block.Parameter t = catcherFinally.parameter(type(Throwable.class));

                catcherFinally.transformBody(finalizer, List.of(), opT.andThen((block, op) -> {
                    if (op instanceof CoreOp.YieldOp) {
                        block.op(throw_(t));
                    } else {
                        // @@@ Composition of lowerable ops
                        if (op instanceof Lowerable lop) {
                            block = lop.lower(block, opT);
                        } else {
                            block.op(op);
                        }
                    }
                    return block;
                }));
            }
            return exit;
        }

        boolean ifExitFromTry(JavaLabelOp lop) {
            Op target = lop.target();
            return target == this || ifAncestorOp(target, this);
        }

        static boolean ifAncestorOp(Op ancestor, Op op) {
            while (op.ancestorBody() != null) {
                op = op.ancestorBody().parentOp();
                if (op == ancestor) {
                    return true;
                }
            }
            return false;
        }

        Block.Builder inlineFinalizer(Block.Builder block1, List<Block.Reference> tryHandlers, OpTransformer opT) {
            Block.Builder finallyEnter = block1.block();
            Block.Builder finallyExit = block1.block();

            block1.op(exceptionRegionExit(finallyEnter.successor(), tryHandlers));

            // Inline the finally body
            finallyEnter.transformBody(finalizer, List.of(), opT.andThen((block2, op2) -> {
                if (op2 instanceof CoreOp.YieldOp) {
                    block2.op(branch(finallyExit.successor()));
                } else {
                    // @@@ Composition of lowerable ops
                    if (op2 instanceof Lowerable lop2) {
                        block2 = lop2.lower(block2, opT);
                    } else {
                        block2.op(op2);
                    }
                }
                return block2;
            }));

            return finallyExit;
        }

        @Override
        public TypeElement resultType() {
            return VOID;
        }
    }

    //
    // Patterns

    // Reified pattern nodes

    /**
     * Synthetic pattern types
     * // @@@ Replace with types extending from TypeElement
     */
    public sealed interface Pattern {

        /**
         * Synthetic type pattern type.
         *
         * @param <T> the type of values that are bound
         */
        final class Type<T> implements Pattern {
            Type() {
            }
        }

        /**
         * Synthetic record pattern type.
         *
         * @param <T> the type of records that are bound
         */
        final class Record<T> implements Pattern {
            Record() {
            }
        }

        final class MatchAll implements Pattern {
            MatchAll() {
            }
        }

        // @@@ Pattern types

        JavaType PATTERN_BINDING_TYPE = JavaType.type(Type.class);

        JavaType PATTERN_RECORD_TYPE = JavaType.type(Record.class);

        JavaType PATTERN_MATCH_ALL_TYPE = JavaType.type(MatchAll.class);

        static JavaType bindingType(TypeElement t) {
            return parameterized(PATTERN_BINDING_TYPE, (JavaType) t);
        }

        static JavaType recordType(TypeElement t) {
            return parameterized(PATTERN_RECORD_TYPE, (JavaType) t);
        }

        static JavaType matchAllType() {
            return PATTERN_MATCH_ALL_TYPE;
        }

        static TypeElement targetType(TypeElement t) {
            return ((ClassType) t).typeArguments().get(0);
        }
    }

    /**
     * Pattern operations.
     */
    public static final class PatternOps {
        PatternOps() {
        }

        /**
         * The pattern operation.
         */
        public sealed static abstract class PatternOp extends JavaOp implements Op.Pure {
            PatternOp(PatternOp that, CopyContext cc) {
                super(that, cc);
            }

            PatternOp(String name, List<Value> operands) {
                super(name, operands);
            }
        }

        /**
         * The binding pattern operation, that can model Java language type patterns.
         */
        @OpDeclaration(TypePatternOp.NAME)
        public static final class TypePatternOp extends PatternOp {
            static final String NAME = "pattern.type";

            public static final String ATTRIBUTE_BINDING_NAME = NAME + ".binding.name";

            final TypeElement resultType;
            final String bindingName;

            TypePatternOp(ExternalizedOp def) {
                super(NAME, List.of());

                this.bindingName = def.extractAttributeValue(ATTRIBUTE_BINDING_NAME, true,
                        v -> switch (v) {
                            case String s -> s;
                            case null -> null;
                            default -> throw new UnsupportedOperationException("Unsupported pattern binding name value:" + v);
                        });
                // @@@ Cannot use canonical constructor because it wraps the given type
                this.resultType = def.resultType();
            }

            TypePatternOp(TypePatternOp that, CopyContext cc) {
                super(that, cc);

                this.bindingName = that.bindingName;
                this.resultType = that.resultType;
            }

            @Override
            public TypePatternOp transform(CopyContext cc, OpTransformer ot) {
                return new TypePatternOp(this, cc);
            }

            TypePatternOp(TypeElement targetType, String bindingName) {
                super(NAME, List.of());

                this.bindingName = bindingName;
                this.resultType = Pattern.bindingType(targetType);
            }

            @Override
            public Map<String, Object> externalize() {
                return bindingName == null ? Map.of() : Map.of("", bindingName);
            }

            public String bindingName() {
                return bindingName;
            }

            public TypeElement targetType() {
                return Pattern.targetType(resultType());
            }

            @Override
            public TypeElement resultType() {
                return resultType;
            }
        }

        /**
         * The record pattern operation, that can model Java language record patterns.
         */
        @OpDeclaration(RecordPatternOp.NAME)
        public static final class RecordPatternOp extends PatternOp {
            static final String NAME = "pattern.record";

            public static final String ATTRIBUTE_RECORD_DESCRIPTOR = NAME + ".descriptor";

            final RecordTypeRef recordDescriptor;

            static RecordPatternOp create(ExternalizedOp def) {
                RecordTypeRef recordDescriptor = def.extractAttributeValue(ATTRIBUTE_RECORD_DESCRIPTOR, true,
                        v -> switch (v) {
                            case RecordTypeRef rtd -> rtd;
                            case null, default ->
                                    throw new UnsupportedOperationException("Unsupported record type descriptor value:" + v);
                        });

                return new RecordPatternOp(recordDescriptor, def.operands());
            }

            RecordPatternOp(RecordPatternOp that, CopyContext cc) {
                super(that, cc);

                this.recordDescriptor = that.recordDescriptor;
            }

            @Override
            public RecordPatternOp transform(CopyContext cc, OpTransformer ot) {
                return new RecordPatternOp(this, cc);
            }

            RecordPatternOp(RecordTypeRef recordDescriptor, List<Value> nestedPatterns) {
                // The type of each value is a subtype of Pattern
                // The number of values corresponds to the number of components of the record
                super(NAME, List.copyOf(nestedPatterns));

                this.recordDescriptor = recordDescriptor;
            }

            @Override
            public Map<String, Object> externalize() {
                return Map.of("", recordDescriptor());
            }

            public RecordTypeRef recordDescriptor() {
                return recordDescriptor;
            }

            public TypeElement targetType() {
                return Pattern.targetType(resultType());
            }

            @Override
            public TypeElement resultType() {
                return Pattern.recordType(recordDescriptor.recordType());
            }
        }

        @OpDeclaration(MatchAllPatternOp.NAME)
        public static final class MatchAllPatternOp extends PatternOp {

            // @@@ we may need to add info about the type of the record component
            // this info can be used when lowering

            static final String NAME = "pattern.match.all";

            MatchAllPatternOp(ExternalizedOp def) {
                this();
            }

            MatchAllPatternOp(MatchAllPatternOp that, CopyContext cc) {
                super(that, cc);
            }

            MatchAllPatternOp() {
                super(NAME, List.of());
            }

            @Override
            public Op transform(CopyContext cc, OpTransformer ot) {
                return new MatchAllPatternOp(this, cc);
            }

            @Override
            public TypeElement resultType() {
                return Pattern.matchAllType();
            }
        }

        /**
         * The match operation, that can model Java language pattern matching.
         */
        @OpDeclaration(MatchOp.NAME)
        public static final class MatchOp extends JavaOp implements Op.Isolated, Op.Lowerable {
            static final String NAME = "pattern.match";

            final Body pattern;
            final Body match;

            MatchOp(ExternalizedOp def) {
                this(def.operands().get(0),
                        def.bodyDefinitions().get(0), def.bodyDefinitions().get(1));
            }

            MatchOp(MatchOp that, CopyContext cc, OpTransformer ot) {
                super(that, cc);

                this.pattern = that.pattern.transform(cc, ot).build(this);
                this.match = that.match.transform(cc, ot).build(this);
            }

            @Override
            public MatchOp transform(CopyContext cc, OpTransformer ot) {
                return new MatchOp(this, cc, ot);
            }

            MatchOp(Value target, Body.Builder patternC, Body.Builder matchC) {
                super(NAME,
                        List.of(target));

                this.pattern = patternC.build(this);
                this.match = matchC.build(this);
            }

            @Override
            public List<Body> bodies() {
                return List.of(pattern, match);
            }

            public Body pattern() {
                return pattern;
            }

            public Body match() {
                return match;
            }

            public Value target() {
                return operands().get(0);
            }

            @Override
            public Block.Builder lower(Block.Builder b, OpTransformer opT) {
                // No match block
                Block.Builder endNoMatchBlock = b.block();
                // Match block
                Block.Builder endMatchBlock = b.block();
                // End block
                Block.Builder endBlock = b.block();
                Block.Parameter matchResult = endBlock.parameter(resultType());
                // Map match operation result
                b.context().mapValue(result(), matchResult);

                List<Value> patternValues = new ArrayList<>();
                Op patternYieldOp = pattern.entryBlock().terminatingOp();
                Op.Result rootPatternValue = (Op.Result) patternYieldOp.operands().get(0);
                Block.Builder currentBlock = lower(endNoMatchBlock, b,
                        patternValues,
                        rootPatternValue.op(),
                        b.context().getValue(target()));
                currentBlock.op(branch(endMatchBlock.successor()));

                // No match block
                // Pass false
                endNoMatchBlock.op(branch(endBlock.successor(
                        endNoMatchBlock.op(constant(BOOLEAN, false)))));

                // Match block
                // Lower match body and pass true
                endMatchBlock.transformBody(match, patternValues, opT.andThen((block, op) -> {
                    if (op instanceof CoreOp.YieldOp) {
                        block.op(branch(endBlock.successor(
                                block.op(constant(BOOLEAN, true)))));
                    } else if (op instanceof Lowerable lop) {
                        // @@@ Composition of lowerable ops
                        block = lop.lower(block, opT);
                    } else {
                        block.op(op);
                    }
                    return block;
                }));

                return endBlock;
            }

            static Block.Builder lower(Block.Builder endNoMatchBlock, Block.Builder currentBlock,
                                       List<Value> bindings,
                                       Op pattern, Value target) {
                return switch (pattern) {
                    case RecordPatternOp rp -> lowerRecordPattern(endNoMatchBlock, currentBlock, bindings, rp, target);
                    case TypePatternOp tp -> lowerTypePattern(endNoMatchBlock, currentBlock, bindings, tp, target);
                    case MatchAllPatternOp map -> lowerMatchAllPattern(currentBlock);
                    case null, default -> throw new UnsupportedOperationException("Unknown pattern op: " + pattern);
                };
            }

            static Block.Builder lowerRecordPattern(Block.Builder endNoMatchBlock, Block.Builder currentBlock,
                                                    List<Value> bindings,
                                                    JavaOp.PatternOps.RecordPatternOp rpOp, Value target) {
                TypeElement targetType = rpOp.targetType();

                Block.Builder nextBlock = currentBlock.block();

                // Check if instance of target type
                Op.Result isInstance = currentBlock.op(instanceOf(targetType, target));
                currentBlock.op(conditionalBranch(isInstance, nextBlock.successor(), endNoMatchBlock.successor()));

                currentBlock = nextBlock;

                target = currentBlock.op(cast(targetType, target));

                // Access component values of record and match on each as nested target
                List<Value> dArgs = rpOp.operands();
                for (int i = 0; i < dArgs.size(); i++) {
                    Op.Result nestedPattern = (Op.Result) dArgs.get(i);
                    // @@@ Handle exceptions?
                    Value nestedTarget = currentBlock.op(invoke(rpOp.recordDescriptor().methodForComponent(i), target));

                    currentBlock = lower(endNoMatchBlock, currentBlock, bindings, nestedPattern.op(), nestedTarget);
                }

                return currentBlock;
            }

            static Block.Builder lowerTypePattern(Block.Builder endNoMatchBlock, Block.Builder currentBlock,
                                                  List<Value> bindings,
                                                  TypePatternOp tpOp, Value target) {
                TypeElement targetType = tpOp.targetType();

                // Check if instance of target type
                Op p; // op that perform type check
                Op c; // op that perform conversion
                TypeElement s = target.type();
                TypeElement t = targetType;
                if (t instanceof PrimitiveType pt) {
                    if (s instanceof ClassType cs) {
                        // unboxing conversions
                        ClassType box;
                        if (cs.unbox().isEmpty()) { // s not a boxed type
                            // e.g. Number -> int, narrowing + unboxing
                            box = pt.box().orElseThrow();
                            p = instanceOf(box, target);
                        } else {
                            // e.g. Float -> float, unboxing
                            // e.g. Integer -> long, unboxing + widening
                            box = cs;
                            p = null;
                        }
                        c = invoke(MethodRef.method(box, t + "Value", t), target);
                    } else {
                        // primitive to primitive conversion
                        PrimitiveType ps = ((PrimitiveType) s);
                        if (isNarrowingPrimitiveConv(ps, pt) || isWideningPrimitiveConvWithCheck(ps, pt)
                                || isWideningAndNarrowingPrimitiveConv(ps, pt)) {
                            // e.g. int -> byte, narrowing
                            // e,g. int -> float, widening with check
                            // e.g. byte -> char, widening and narrowing
                            MethodRef mref = convMethodRef(s, t);
                            p = invoke(mref, target);
                        } else {
                            p = null;
                        }
                        c = conv(targetType, target);
                    }
                } else if (s instanceof PrimitiveType ps) {
                    // boxing conversions
                    // e.g. int -> Number, boxing + widening
                    // e.g. byte -> Byte, boxing
                    p = null;
                    ClassType box = ps.box().orElseThrow();
                    c = invoke(MethodRef.method(box, "valueOf", box, ps), target);
                } else if (!s.equals(t)) {
                    // reference to reference, but not identity
                    // e.g. Number -> Double, narrowing
                    // e.g. Short -> Object, widening
                    p = instanceOf(targetType, target);
                    c = cast(targetType, target);
                } else {
                    // identity reference
                    // e.g. Character -> Character
                    p = null;
                    c = null;
                }

                if (c != null) {
                    if (p != null) {
                        // p != null, we need to perform type check at runtime
                        Block.Builder nextBlock = currentBlock.block();
                        currentBlock.op(conditionalBranch(currentBlock.op(p), nextBlock.successor(), endNoMatchBlock.successor()));
                        currentBlock = nextBlock;
                    }
                    target = currentBlock.op(c);
                }

                bindings.add(target);

                return currentBlock;
            }

            private static boolean isWideningAndNarrowingPrimitiveConv(PrimitiveType s, PrimitiveType t) {
                return BYTE.equals(s) && CHAR.equals(t);
            }

            private static boolean isWideningPrimitiveConvWithCheck(PrimitiveType s, PrimitiveType t) {
                return (INT.equals(s) && FLOAT.equals(t))
                        || (LONG.equals(s) && FLOAT.equals(t))
                        || (LONG.equals(s) && DOUBLE.equals(t));
            }

            // s -> t is narrowing if order(t) <= order(s)
            private final static Map<PrimitiveType, Integer> narrowingOrder = Map.of(
                    BYTE, 1,
                    SHORT, 2,
                    CHAR, 2,
                    INT, 3,
                    LONG, 4,
                    FLOAT, 5,
                    DOUBLE, 6
            );
            private static boolean isNarrowingPrimitiveConv(PrimitiveType s, PrimitiveType t) {
                return narrowingOrder.get(t) <= narrowingOrder.get(s);
            }

            private static MethodRef convMethodRef(TypeElement s, TypeElement t) {
                if (BYTE.equals(s) || SHORT.equals(s) || CHAR.equals(s)) {
                    s = INT;
                }
                String sn = capitalize(s.toString());
                String tn = capitalize(t.toString());
                String mn = "is%sTo%sExact".formatted(sn, tn);
                JavaType exactConversionSupport = JavaType.type(ClassDesc.of("java.lang.runtime.ExactConversionsSupport"));
                return MethodRef.method(exactConversionSupport, mn, BOOLEAN, s);
            }

            private static String capitalize(String s) {
                return s.substring(0, 1).toUpperCase() + s.substring(1);
            }

            static Block.Builder lowerMatchAllPattern(Block.Builder currentBlock) {
                return currentBlock;
            }

            @Override
            public TypeElement resultType() {
                return BOOLEAN;
            }
        }
    }

    static Op createOp(ExternalizedOp def) {
        Op op = switch (def.name()) {
            case "add" -> new AddOp(def);
            case "and" -> new AndOp(def);
            case "array.length" -> new ArrayLengthOp(def);
            case "array.load" -> new ArrayAccessOp.ArrayLoadOp(def);
            case "array.store" -> new ArrayAccessOp.ArrayStoreOp(def);
            case "ashr" -> new AshrOp(def);
            case "assert" -> new AssertOp(def);
            case "cast" -> CastOp.create(def);
            case "compl" -> new ComplOp(def);
            case "concat" -> new ConcatOp(def);
            case "conv" -> new ConvOp(def);
            case "div" -> new DivOp(def);
            case "eq" -> new EqOp(def);
            case "exception.region.enter" -> new ExceptionRegionEnter(def);
            case "exception.region.exit" -> new ExceptionRegionExit(def);
            case "field.load" -> FieldAccessOp.FieldLoadOp.create(def);
            case "field.store" -> FieldAccessOp.FieldStoreOp.create(def);
            case "ge" -> new GeOp(def);
            case "gt" -> new GtOp(def);
            case "instanceof" -> InstanceOfOp.create(def);
            case "invoke" -> InvokeOp.create(def);
            case "java.block" -> new BlockOp(def);
            case "java.break" -> new BreakOp(def);
            case "java.cand" -> new ConditionalAndOp(def);
            case "java.cexpression" -> new ConditionalExpressionOp(def);
            case "java.continue" -> new ContinueOp(def);
            case "java.cor" -> new ConditionalOrOp(def);
            case "java.do.while" -> new DoWhileOp(def);
            case "java.enhancedFor" -> EnhancedForOp.create(def);
            case "java.for" -> ForOp.create(def);
            case "java.if" -> new IfOp(def);
            case "java.labeled" -> new LabeledOp(def);
            case "java.switch.expression" -> new SwitchExpressionOp(def);
            case "java.switch.fallthrough" -> new SwitchFallthroughOp(def);
            case "java.switch.statement" -> new SwitchStatementOp(def);
            case "java.synchronized" -> new SynchronizedOp(def);
            case "java.try" -> TryOp.create(def);
            case "java.while" -> new WhileOp(def);
            case "java.yield" -> new YieldOp(def);
            case "lambda" -> new LambdaOp(def);
            case "le" -> new LeOp(def);
            case "lshl" -> new LshlOp(def);
            case "lshr" -> new LshrOp(def);
            case "lt" -> new LtOp(def);
            case "mod" -> new ModOp(def);
            case "monitor.enter" -> new MonitorOp.MonitorEnterOp(def);
            case "monitor.exit" -> new MonitorOp.MonitorExitOp(def);
            case "mul" -> new MulOp(def);
            case "neg" -> new NegOp(def);
            case "neq" -> new NeqOp(def);
            case "new" -> NewOp.create(def);
            case "not" -> new NotOp(def);
            case "or" -> new OrOp(def);
            case "pattern.match" -> new PatternOps.MatchOp(def);
            case "pattern.match.all" -> new PatternOps.MatchAllPatternOp(def);
            case "pattern.record" -> PatternOps.RecordPatternOp.create(def);
            case "pattern.type" -> new PatternOps.TypePatternOp(def);
            case "sub" -> new SubOp(def);
            case "throw" -> new ThrowOp(def);
            case "xor" -> new XorOp(def);
            default -> null;
        };
        if (op != null) {
            op.setLocation(def.location());
        }
        return op;
    }

    /**
     * An operation factory for core operations composed with Java operations.
     */
    public static final OpFactory JAVA_OP_FACTORY = CoreOp.CORE_OP_FACTORY.andThen(JavaOp::createOp);

    /**
     * A Java dialect factory, for constructing core and Java operations and constructing
     * core type and Java type elements, where the core type elements can refer to Java
     * type elements.
     */
    public static final DialectFactory JAVA_DIALECT_FACTORY = new DialectFactory(
            JAVA_OP_FACTORY,
            JAVA_TYPE_FACTORY);

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
     * @param end             the block to which control is transferred after the exception region is exited
     * @param catchers the blocks handling exceptions thrown by the region block
     * @return the exception region exit operation
     */
    public static ExceptionRegionExit exceptionRegionExit(Block.Reference end, Block.Reference... catchers) {
        return exceptionRegionExit(end, List.of(catchers));
    }

    /**
     * Creates an exception region exit operation
     *
     * @param end             the block to which control is transferred after the exception region is exited
     * @param catchers the blocks handling exceptions thrown by the region block
     * @return the exception region exit operation
     */
    public static ExceptionRegionExit exceptionRegionExit(Block.Reference end, List<Block.Reference> catchers) {
        List<Block.Reference> s = new ArrayList<>();
        s.add(end);
        s.addAll(catchers);
        return new ExceptionRegionExit(s);
    }

    /**
     * Creates a throw operation.
     *
     * @param exceptionValue the thrown value
     * @return the throw operation
     */
    public static ThrowOp throw_(Value exceptionValue) {
        return new ThrowOp(exceptionValue);
    }

    /**
     * Creates an assert operation.
     *
     * @param bodies the nested bodies
     * @return the assert operation
     */
    public static AssertOp assert_(List<Body.Builder> bodies) {
        return new AssertOp(bodies);
    }

    public static MonitorOp.MonitorEnterOp monitorEnter(Value monitor) {
        return new MonitorOp.MonitorEnterOp(monitor);
    }

    public static MonitorOp.MonitorExitOp monitorExit(Value monitor) {
        return new MonitorOp.MonitorExitOp(monitor);
    }

    /**
     * Creates an invoke operation modeling an invocation to an
     * instance or static (class) method with no variable arguments.
     * <p>
     * The invoke kind of the invoke operation is determined by
     * comparing the argument count with the invoke descriptor's
     * parameter count. If they are equal then the invoke kind is
     * {@link InvokeOp.InvokeKind#STATIC static}. If the parameter count
     * plus one is equal to the argument count then the invoke kind
     * is {@link InvokeOp.InvokeKind#STATIC instance}.
     * <p>
     * The invoke return type is the invoke descriptors return type.
     *
     * @param invokeDescriptor the invoke descriptor
     * @param args             the invoke parameters
     * @return the invoke operation
     */
    public static InvokeOp invoke(MethodRef invokeDescriptor, Value... args) {
        return invoke(invokeDescriptor, List.of(args));
    }

    /**
     * Creates an invoke operation modeling an invocation to an
     * instance or static (class) method with no variable arguments.
     * <p>
     * The invoke kind of the invoke operation is determined by
     * comparing the argument count with the invoke descriptor's
     * parameter count. If they are equal then the invoke kind is
     * {@link InvokeOp.InvokeKind#STATIC static}. If the parameter count
     * plus one is equal to the argument count then the invoke kind
     * is {@link InvokeOp.InvokeKind#STATIC instance}.
     * <p>
     * The invoke return type is the invoke descriptors return type.
     *
     * @param invokeDescriptor the invoke descriptor
     * @param args             the invoke arguments
     * @return the invoke operation
     */
    public static InvokeOp invoke(MethodRef invokeDescriptor, List<Value> args) {
        return invoke(invokeDescriptor.type().returnType(), invokeDescriptor, args);
    }

    /**
     * Creates an invoke operation modeling an invocation to an
     * instance or static (class) method with no variable arguments.
     * <p>
     * The invoke kind of the invoke operation is determined by
     * comparing the argument count with the invoke descriptor's
     * parameter count. If they are equal then the invoke kind is
     * {@link InvokeOp.InvokeKind#STATIC static}. If the parameter count
     * plus one is equal to the argument count then the invoke kind
     * is {@link InvokeOp.InvokeKind#STATIC instance}.
     *
     * @param returnType       the invoke return type
     * @param invokeDescriptor the invoke descriptor
     * @param args             the invoke arguments
     * @return the invoke operation
     */
    public static InvokeOp invoke(TypeElement returnType, MethodRef invokeDescriptor, Value... args) {
        return invoke(returnType, invokeDescriptor, List.of(args));
    }

    /**
     * Creates an invoke operation modeling an invocation to an
     * instance or static (class) method with no variable arguments.
     * <p>
     * The invoke kind of the invoke operation is determined by
     * comparing the argument count with the invoke descriptor's
     * parameter count. If they are equal then the invoke kind is
     * {@link InvokeOp.InvokeKind#STATIC static}. If the parameter count
     * plus one is equal to the argument count then the invoke kind
     * is {@link InvokeOp.InvokeKind#STATIC instance}.
     *
     * @param returnType       the invoke return type
     * @param invokeDescriptor the invoke descriptor
     * @param args             the invoke arguments
     * @return the invoke super operation
     */
    public static InvokeOp invoke(TypeElement returnType, MethodRef invokeDescriptor, List<Value> args) {
        int paramCount = invokeDescriptor.type().parameterTypes().size();
        int argCount = args.size();
        InvokeOp.InvokeKind ik = (argCount == paramCount + 1)
                ? InvokeOp.InvokeKind.INSTANCE
                : InvokeOp.InvokeKind.STATIC;
        return new InvokeOp(ik, false, returnType, invokeDescriptor, args);
    }

    /**
     * Creates an invoke operation modelling an invocation to a method.
     *
     * @param invokeKind       the invoke kind
     * @param isVarArgs        true if an invocation to a variable argument method
     * @param returnType       the return type
     * @param invokeDescriptor the invoke descriptor
     * @param args             the invoke arguments
     * @return the invoke operation
     * @throws IllegalArgumentException if there is a mismatch between the argument count
     *                                  and the invoke descriptors parameter count.
     */
    public static InvokeOp invoke(InvokeOp.InvokeKind invokeKind, boolean isVarArgs,
                                  TypeElement returnType, MethodRef invokeDescriptor, List<Value> args) {
        return new InvokeOp(invokeKind, isVarArgs, returnType, invokeDescriptor, args);
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
     * @param constructorDescriptor the constructor descriptor
     * @param args            the constructor arguments
     * @return the instance creation operation
     */
    public static NewOp new_(ConstructorRef constructorDescriptor, Value... args) {
        return new_(constructorDescriptor, List.of(args));
    }

    /**
     * Creates an instance creation operation.
     *
     * @param constructorDescriptor the constructor descriptor
     * @param args            the constructor arguments
     * @return the instance creation operation
     */
    public static NewOp new_(ConstructorRef constructorDescriptor, List<Value> args) {
        return new NewOp(false, constructorDescriptor.refType(), constructorDescriptor, args);
    }

    /**
     * Creates an instance creation operation.
     *
     * @param returnType      the instance type
     * @param constructorDescriptor the constructor descriptor
     * @param args            the constructor arguments
     * @return the instance creation operation
     */
    public static NewOp new_(TypeElement returnType, ConstructorRef constructorDescriptor,
                             Value... args) {
        return new_(returnType, constructorDescriptor, List.of(args));
    }

    /**
     * Creates an instance creation operation.
     *
     * @param returnType      the instance type
     * @param constructorDescriptor the constructor descriptor
     * @param args            the constructor arguments
     * @return the instance creation operation
     */
    public static NewOp new_(TypeElement returnType, ConstructorRef constructorDescriptor,
                             List<Value> args) {
        return new NewOp(false, returnType, constructorDescriptor, args);
    }

    /**
     * Creates an instance creation operation.
     *
     * @param returnType      the instance type
     * @param constructorDescriptor the constructor descriptor
     * @param args            the constructor arguments
     * @return the instance creation operation
     */
    public static NewOp new_(boolean isVarargs, TypeElement returnType, ConstructorRef constructorDescriptor,
                             List<Value> args) {
        return new NewOp(isVarargs, returnType, constructorDescriptor, args);
    }

    /**
     * Creates an array creation operation.
     *
     * @param arrayType the array type
     * @param length    the array size
     * @return the array creation operation
     */
    public static NewOp newArray(TypeElement arrayType, Value length) {
        ConstructorRef constructorDescriptor = ConstructorRef.constructor(arrayType, INT);
        return new_(constructorDescriptor, length);
    }

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
     * Creates an array load operation.
     *
     * @param array the array value
     * @param index the index value
     * @param componentType type of the array component
     * @return the array load operation
     */
    public static ArrayAccessOp.ArrayLoadOp arrayLoadOp(Value array, Value index, TypeElement componentType) {
        return new ArrayAccessOp.ArrayLoadOp(array, index, componentType);
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
     * Creates a bitwise complement operation.
     *
     * @param v the operand
     * @return the bitwise complement operation
     */
    public static UnaryOp compl(Value v) {
        return new ComplOp(v);
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

    /**
     * Creates a continue operation.
     *
     * @return the continue operation
     */
    public static ContinueOp continue_() {
        return continue_(null);
    }

    /**
     * Creates a continue operation.
     *
     * @param label the value associated with where to continue from
     * @return the continue operation
     */
    public static ContinueOp continue_(Value label) {
        return new ContinueOp(label);
    }

    /**
     * Creates a break operation.
     *
     * @return the break operation
     */
    public static BreakOp break_() {
        return break_(null);
    }

    /**
     * Creates a break operation.
     *
     * @param label the value associated with where to continue from
     * @return the break operation
     */
    public static BreakOp break_(Value label) {
        return new BreakOp(label);
    }

    /**
     * Creates a yield operation.
     *
     * @return the yield operation
     */
    public static YieldOp java_yield() {
        return java_yield(null);
    }

    /**
     * Creates a yield operation.
     *
     * @param operand the value to yield
     * @return the yield operation
     */
    public static YieldOp java_yield(Value operand) {
        return new YieldOp(operand);
    }

    /**
     * Creates a block operation.
     *
     * @param body the body builder of the operation to be built and become its child
     * @return the block operation
     */
    public static BlockOp block(Body.Builder body) {
        return new BlockOp(body);
    }

    /**
     * Creates a synchronized operation.
     *
     * @param expr the expression body builder of the operation to be built and become its child
     * @param blockBody the block body builder of the operation to be built and become its child
     * @return the synchronized operation
     */
    public static SynchronizedOp synchronized_(Body.Builder expr, Body.Builder blockBody) {
        return new SynchronizedOp(expr, blockBody);
    }

    /**
     * Creates a labeled operation.
     *
     * @param body the body builder of the operation to be built and become its child
     * @return the block operation
     */
    public static LabeledOp labeled(Body.Builder body) {
        return new LabeledOp(body);
    }

    /**
     * Creates an if operation builder.
     *
     * @param ancestorBody the nearest ancestor body builder from which to construct
     *                     body builders for this operation
     * @return the if operation builder
     */
    public static IfOp.IfBuilder if_(Body.Builder ancestorBody) {
        return new IfOp.IfBuilder(ancestorBody);
    }

    // Pairs of
    //   predicate ()boolean, body ()void
    // And one optional body ()void at the end

    /**
     * Creates an if operation.
     *
     * @param bodies the body builders of operation to be built and become its children
     * @return the if operation
     */
    public static IfOp if_(List<Body.Builder> bodies) {
        return new IfOp(bodies);
    }

    /**
     * Creates a switch expression operation.
     * <p>
     * The result type of the operation will be derived from the yield type of the second body
     *
     * @param target the switch target value
     * @param bodies the body builders of the operation to be built and become its children
     * @return the switch expression operation
     */
    public static SwitchExpressionOp switchExpression(Value target, List<Body.Builder> bodies) {
        return new SwitchExpressionOp(null, target, bodies);
    }

    /**
     * Creates a switch expression operation.
     *
     * @param resultType the result type of the expression
     * @param target     the switch target value
     * @param bodies     the body builders of the operation to be built and become its children
     * @return the switch expression operation
     */
    public static SwitchExpressionOp switchExpression(TypeElement resultType, Value target,
                                                      List<Body.Builder> bodies) {
        Objects.requireNonNull(resultType);
        return new SwitchExpressionOp(resultType, target, bodies);
    }

    /**
     * Creates a switch statement operation.
     * @param target the switch target value
     * @param bodies the body builders of the operation to be built and become its children
     * @return the switch statement operation
     */
    public static SwitchStatementOp switchStatement(Value target, List<Body.Builder> bodies) {
        return new SwitchStatementOp(target, bodies);
    }

    /**
     * Creates a switch fallthrough operation.
     *
     * @return the switch fallthrough operation
     */
    public static SwitchFallthroughOp switchFallthroughOp() {
        return new SwitchFallthroughOp();
    }

    /**
     * Creates a for operation builder.
     *
     * @param ancestorBody the nearest ancestor body builder from which to construct
     *                     body builders for this operation
     * @param initTypes    the types of initialized variables
     * @return the for operation builder
     */
    public static ForOp.InitBuilder for_(Body.Builder ancestorBody, TypeElement... initTypes) {
        return for_(ancestorBody, List.of(initTypes));
    }

    /**
     * Creates a for operation builder.
     *
     * @param ancestorBody the nearest ancestor body builder from which to construct
     *                     body builders for this operation
     * @param initTypes    the types of initialized variables
     * @return the for operation builder
     */
    public static ForOp.InitBuilder for_(Body.Builder ancestorBody, List<? extends TypeElement> initTypes) {
        return new ForOp.InitBuilder(ancestorBody, initTypes);
    }


    /**
     * Creates a for operation.
     *
     * @param init   the init body builder of the operation to be built and become its child
     * @param cond   the cond body builder of the operation to be built and become its child
     * @param update the update body builder of the operation to be built and become its child
     * @param body   the main body builder of the operation to be built and become its child
     * @return the for operation
     */
    // init ()Tuple<Var<T1>, Var<T2>, ..., Var<TN>>, or init ()void
    // cond (Var<T1>, Var<T2>, ..., Var<TN>)boolean
    // update (Var<T1>, Var<T2>, ..., Var<TN>)void
    // body (Var<T1>, Var<T2>, ..., Var<TN>)void
    public static ForOp for_(Body.Builder init,
                             Body.Builder cond,
                             Body.Builder update,
                             Body.Builder body) {
        return new ForOp(init, cond, update, body);
    }

    /**
     * Creates an enhanced for operation builder.
     *
     * @param ancestorBody the nearest ancestor body builder from which to construct
     *                     body builders for this operation
     * @param iterableType the iterable type
     * @param elementType  the element type
     * @return the enhanced for operation builder
     */
    public static EnhancedForOp.ExpressionBuilder enhancedFor(Body.Builder ancestorBody,
                                                              TypeElement iterableType, TypeElement elementType) {
        return new EnhancedForOp.ExpressionBuilder(ancestorBody, iterableType, elementType);
    }

    // expression ()I<E>
    // init (E )Var<T>
    // body (Var<T> )void

    /**
     * Creates an enhanced for operation.
     *
     * @param expression the expression body builder of the operation to be built and become its child
     * @param init       the init body builder of the operation to be built and become its child
     * @param body       the main body builder of the operation to be built and become its child
     * @return the enhanced for operation
     */
    public static EnhancedForOp enhancedFor(Body.Builder expression,
                                            Body.Builder init,
                                            Body.Builder body) {
        return new EnhancedForOp(expression, init, body);
    }

    /**
     * Creates a while operation builder.
     *
     * @param ancestorBody the nearest ancestor body builder from which to construct
     *                     body builders for this operation
     * @return the while operation builder
     */
    public static WhileOp.PredicateBuilder while_(Body.Builder ancestorBody) {
        return new WhileOp.PredicateBuilder(ancestorBody);
    }

    /**
     * Creates a while operation.
     *
     * @param predicate the predicate body builder of the operation to be built and become its child
     * @param body      the main body builder of the operation to be built and become its child
     * @return the while operation
     */
    // predicate, ()boolean, may be null for predicate returning true
    // body, ()void
    public static WhileOp while_(Body.Builder predicate, Body.Builder body) {
        return new WhileOp(predicate, body);
    }

    /**
     * Creates a do operation builder.
     *
     * @param ancestorBody the nearest ancestor body builder from which to construct
     *                     body builders for this operation
     * @return the do operation builder
     */
    public static DoWhileOp.BodyBuilder doWhile(Body.Builder ancestorBody) {
        return new DoWhileOp.BodyBuilder(ancestorBody);
    }

    /**
     * Creates a do operation.
     *
     * @param predicate the predicate body builder of the operation to be built and become its child
     * @param body      the main body builder of the operation to be built and become its child
     * @return the do operation
     */
    public static DoWhileOp doWhile(Body.Builder body, Body.Builder predicate) {
        return new DoWhileOp(body, predicate);
    }

    /**
     * Creates a conditional-and operation builder.
     *
     * @param ancestorBody the nearest ancestor body builder from which to construct
     *                     body builders for this operation
     * @param lhs          a consumer that builds the left-hand side body
     * @param rhs          a consumer that builds the right-hand side body
     * @return the conditional-and operation builder
     */
    public static ConditionalAndOp.Builder conditionalAnd(Body.Builder ancestorBody,
                                                          Consumer<Block.Builder> lhs, Consumer<Block.Builder> rhs) {
        return new ConditionalAndOp.Builder(ancestorBody, lhs, rhs);
    }

    /**
     * Creates a conditional-or operation builder.
     *
     * @param ancestorBody the nearest ancestor body builder from which to construct
     *                     body builders for this operation
     * @param lhs          a consumer that builds the left-hand side body
     * @param rhs          a consumer that builds the right-hand side body
     * @return the conditional-or operation builder
     */
    public static ConditionalOrOp.Builder conditionalOr(Body.Builder ancestorBody,
                                                        Consumer<Block.Builder> lhs, Consumer<Block.Builder> rhs) {
        return new ConditionalOrOp.Builder(ancestorBody, lhs, rhs);
    }

    /**
     * Creates a conditional-and operation
     *
     * @param bodies the body builders of operation to be built and become its children
     * @return the conditional-and operation
     */
    // predicates, ()boolean
    public static ConditionalAndOp conditionalAnd(List<Body.Builder> bodies) {
        return new ConditionalAndOp(bodies);
    }

    /**
     * Creates a conditional-or operation
     *
     * @param bodies the body builders of operation to be built and become its children
     * @return the conditional-or operation
     */
    // predicates, ()boolean
    public static ConditionalOrOp conditionalOr(List<Body.Builder> bodies) {
        return new ConditionalOrOp(bodies);
    }

    /**
     * Creates a conditional operation
     *
     * @param expressionType the result type of the expression
     * @param bodies         the body builders of operation to be built and become its children
     * @return the conditional operation
     */
    public static ConditionalExpressionOp conditionalExpression(TypeElement expressionType,
                                                                List<Body.Builder> bodies) {
        Objects.requireNonNull(expressionType);
        return new ConditionalExpressionOp(expressionType, bodies);
    }

    /**
     * Creates a conditional operation
     * <p>
     * The result type of the operation will be derived from the yield type of the second body
     *
     * @param bodies the body builders of operation to be built and become its children
     * @return the conditional operation
     */
    public static ConditionalExpressionOp conditionalExpression(List<Body.Builder> bodies) {
        return new ConditionalExpressionOp(null, bodies);
    }

    /**
     * Creates try operation builder.
     *
     * @param ancestorBody the nearest ancestor body builder from which to construct
     *                     body builders for this operation
     * @param c            a consumer that builds the try body
     * @return the try operation builder
     */
    public static TryOp.CatchBuilder try_(Body.Builder ancestorBody, Consumer<Block.Builder> c) {
        Body.Builder _try = Body.Builder.of(ancestorBody, CoreType.FUNCTION_TYPE_VOID);
        c.accept(_try.entryBlock());
        return new TryOp.CatchBuilder(ancestorBody, null, _try);
    }

    /**
     * Creates try-with-resources operation builder.
     *
     * @param ancestorBody the nearest ancestor body builder from which to construct
     *                     body builders for this operation
     * @param c            a consumer that builds the resources body
     * @return the try-with-resources operation builder
     */
    public static TryOp.BodyBuilder tryWithResources(Body.Builder ancestorBody,
                                                     List<? extends TypeElement> resourceTypes,
                                                     Consumer<Block.Builder> c) {
        resourceTypes = resourceTypes.stream().map(CoreType::varType).toList();
        Body.Builder resources = Body.Builder.of(ancestorBody,
                CoreType.functionType(CoreType.tupleType(resourceTypes)));
        c.accept(resources.entryBlock());
        return new TryOp.BodyBuilder(ancestorBody, resourceTypes, resources);
    }

    // resources ()Tuple<Var<R1>, Var<R2>, ..., Var<RN>>, or null
    // try (Var<R1>, Var<R2>, ..., Var<RN>)void, or try ()void
    // catch (E )void, where E <: Throwable
    // finally ()void, or null

    /**
     * Creates a try or try-with-resources operation.
     *
     * @param resources the try body builder of the operation to be built and become its child,
     *                  may be null
     * @param body      the try body builder of the operation to be built and become its child
     * @param catchers  the catch body builders of the operation to be built and become its children
     * @param finalizer the finalizer body builder of the operation to be built and become its child
     * @return the try or try-with-resources operation
     */
    public static TryOp try_(Body.Builder resources,
                             Body.Builder body,
                             List<Body.Builder> catchers,
                             Body.Builder finalizer) {
        return new TryOp(resources, body, catchers, finalizer);
    }

    //
    // Patterns

    /**
     * Creates a pattern match operation.
     *
     * @param target  the target value
     * @param pattern the pattern body builder of the operation to be built and become its child
     * @param match   the match body builder of the operation to be built and become its child
     * @return the pattern match operation
     */
    public static PatternOps.MatchOp match(Value target,
                                           Body.Builder pattern, Body.Builder match) {
        return new PatternOps.MatchOp(target, pattern, match);
    }

    /**
     * Creates a pattern binding operation.
     *
     * @param type        the type of value to be bound
     * @param bindingName the binding name
     * @return the pattern binding operation
     */
    public static PatternOps.TypePatternOp typePattern(TypeElement type, String bindingName) {
        return new PatternOps.TypePatternOp(type, bindingName);
    }

    /**
     * Creates a record pattern operation.
     *
     * @param recordDescriptor the record descriptor
     * @param nestedPatterns   the nested pattern values
     * @return the record pattern operation
     */
    public static PatternOps.RecordPatternOp recordPattern(RecordTypeRef recordDescriptor, Value... nestedPatterns) {
        return recordPattern(recordDescriptor, List.of(nestedPatterns));
    }

    /**
     * Creates a record pattern operation.
     *
     * @param recordDescriptor the record descriptor
     * @param nestedPatterns   the nested pattern values
     * @return the record pattern operation
     */
    public static PatternOps.RecordPatternOp recordPattern(RecordTypeRef recordDescriptor, List<Value> nestedPatterns) {
        return new PatternOps.RecordPatternOp(recordDescriptor, nestedPatterns);
    }

    public static PatternOps.MatchAllPatternOp matchAllPattern() {
        return new PatternOps.MatchAllPatternOp();
    }
}

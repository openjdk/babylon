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
package optkl;

import jdk.incubator.code.Block;
import jdk.incubator.code.Body;
import jdk.incubator.code.CodeElement;
import jdk.incubator.code.Op;
import jdk.incubator.code.Quoted;
import jdk.incubator.code.TypeElement;
import jdk.incubator.code.Value;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.ArrayType;
import jdk.incubator.code.dialect.java.ClassType;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.JavaType;
import jdk.incubator.code.dialect.java.PrimitiveType;
import optkl.ifacemapper.AccessType;
import optkl.ifacemapper.MappableIface;
import optkl.util.Regex;
import optkl.util.carriers.LookupCarrier;
import optkl.util.ops.StatementLikeOp;

import java.lang.annotation.Annotation;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.lang.reflect.Type;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.function.Function;
import java.util.function.Predicate;
import java.util.stream.Stream;


public sealed interface OpHelper<T extends Op> extends LookupCarrier
        permits OpHelper.Binary, OpHelper.Lambda, OpHelper.LoadOrStore, OpHelper.Named, OpHelper.Ternary {
    static <F extends Op, T extends Op> T copyLocation(F from, T to) {
        to.setLocation(from.location());
        return to;
    }

    static Value firstOperandOrNull(Op op) {
        if (!op.operands().isEmpty()) {
            return op.operands().getFirst();
        } else {
            return null;
        }
    }

    static Value firstOperandOrThrow(Op op) {
        if (!op.operands().isEmpty()) {
            return op.operands().getFirst();
        } else {
            throw new RuntimeException("Op has no operands");
        }
    }

    static List<Value> firstOperandAsListOrEmpty(Op op) {
        return op.operands().isEmpty() ? List.of() : List.of(op.operands().getFirst());
    }

    static CoreOp.FuncOp methodModelOrNull(Method method) {
        return CoreOp.FuncOp.ofMethod(method).orElse(null);
    }

    static CoreOp.FuncOp methodModelOrThrow(Method method) {

        if (methodModelOrNull(method) instanceof CoreOp.FuncOp funcOp) {
            return funcOp;
        } else {
            throw new RuntimeException("No funcop/method model for " + method + " did you forget @Reflec");
        }
    }

    T op();

    static Type classTypeToTypeOrThrow(MethodHandles.Lookup lookup, ClassType classType) {
        try {
            return classType.resolve(lookup);
        } catch (ReflectiveOperationException e) {
            throw new RuntimeException(e);
        }
    }

    static boolean isAssignable(MethodHandles.Lookup lookup, TypeElement typeElement, Class<?>... classes) {
        if (typeElement instanceof ClassType classType) {
            Type type = classTypeToTypeOrThrow(lookup, classType);
            return Arrays.stream(classes).anyMatch(clazz -> clazz.isAssignableFrom((Class<?>) type));
        } else if (typeElement instanceof PrimitiveType) {
            return Arrays.stream(classes).anyMatch(clazz ->
                    (typeElement == JavaType.FLOAT && clazz.equals(float.class))
                            || (typeElement == JavaType.DOUBLE && clazz.equals(double.class))
                            || (typeElement == JavaType.INT && clazz.equals(int.class))
                            || (typeElement == JavaType.LONG && clazz.equals(long.class))
                            || (typeElement == JavaType.SHORT && clazz.equals(short.class))
                            || (typeElement == JavaType.CHAR && clazz.equals(char.class))
                            || (typeElement == JavaType.BYTE && clazz.equals(byte.class))
                            || (typeElement == JavaType.BOOLEAN && clazz.equals(boolean.class))
                            || (typeElement == JavaType.VOID && clazz.equals(void.class))
            );
        }
        return false;
    }

    default boolean isAssignable(JavaType javaType, Class<?>... clazzes) {
        return isAssignable(lookup(), javaType, clazzes);
    }

    default int operandCount() {
        return op().operands().size();
    }

    default Op.Result resultFromOperandNOrNull(int i) {
        return resultFromOperandN(op(), i) instanceof Op.Result result ? result : null;
    }

    default Op.Result resultFromFirstOperandOrNull() {
        return resultFromOperandNOrNull(0);
    }


    default Op.Result resultFromOperandNOrThrow(int i) {
        if (resultFromOperandNOrNull(i) instanceof Op.Result result) {
            return result;
        } else {
            throw new IllegalStateException("Expecting operand " + i + " to be a result");
        }
    }

    static Op opFromOperandNOrNull(Op op, int i) {
        return resultFromOperandN(op, i) instanceof Op.Result result && result.op() instanceof Op op2 ? op2 : null;
    }

    default Op opFromOperandNOrNull(int i) {
        return resultFromOperandNOrNull(i) instanceof Op.Result result && result.op() instanceof Op op ? op : null;
    }

    default Op opFromFirstOperandOrNull() {
        return opFromOperandNOrNull(0);
    }

    static Op opFromOperandNOrThrow(Op op, int i) {
        if (opFromOperandNOrNull(op, i) instanceof Op op1) {
            return op1;
        } else {
            throw new IllegalStateException("Expecting operand " + i + " to be a result which yields an Op ");
        }
    }

    default Op opFromOperandNOrThrow(int i) {
        if (opFromOperandNOrNull(i) instanceof Op op) {
            return op;
        } else {
            throw new IllegalStateException("Expecting operand " + i + " to be a result which yields an Op ");
        }
    }

    static Op opFromFirstOperandOrNull(Op op) {
        return opFromOperandNOrNull(op, 0);
    }

    static Op opFromFirstOperandOrThrow(Op op) {
        return opFromOperandNOrThrow(op, 0);
    }

    default Op opFromFirstOperandOrThrow() {
        return opFromOperandNOrThrow(0);
    }

    default CoreOp.VarAccessOp.VarLoadOp varLoadOpFromFirstOperandOrNull() {
        return opFromFirstOperandOrThrow()
                instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp ? varLoadOp : null;
    }

    static Block entryBlockOfBodyN(Op op, int idx) {
        return op.bodies().get(idx).entryBlock();
    }


    static boolean isPrimitiveResult(Value val) {
        return ((val instanceof Op.Result result && result.op().resultType() instanceof PrimitiveType primitiveType) ? primitiveType : null) != null;
    }

    static Op.Result asResultOrThrow(Value value) {
        if (value instanceof Op.Result result) {
            return result;
        } else {
            throw new RuntimeException("Value not a result");
        }
    }


    static Op.Result asResultOrNull(Value operand) {
        return operand instanceof Op.Result result ? result : null;
    }

    static Op asOpFromResultOrNull(Value operand) {
        return asResultOrNull(operand) instanceof Op.Result r && r.op() instanceof Op op ? op : null;
    }

    static Op.Result resultFromOperandN(jdk.incubator.code.CodeElement<?, ?> codeElement, int n) {
        return codeElement instanceof Op op && op.operands().size() > n && op.operands().get(n) instanceof Op.Result result ? result : null;
    }

    static Op.Result resultFromFirstOperandOrNull(jdk.incubator.code.CodeElement<?, ?> codeElement) {
        return resultFromOperandN(codeElement, 0);
    }

    static Op.Result resultFromFirstOperandOrThrow(jdk.incubator.code.CodeElement<?, ?> codeElement) {
        if (resultFromFirstOperandOrNull(codeElement) instanceof Op.Result result) {
            return result;
        } else {
            throw new RuntimeException("Expected result as first operand");
        }
    }

    static Op.Result lhsResult(JavaOp.BinaryOp binaryOp) {
        return (Op.Result) binaryOp.operands().get(0);
    }

    static Op.Result rhsResult(JavaOp.BinaryOp binaryOp) {
        return (Op.Result) binaryOp.operands().get(1);
    }

    static List<Op> lhsOps(JavaOp.JavaConditionalOp javaConditionalOp) {
        return javaConditionalOp.bodies().get(0).entryBlock().ops();
    }

    static List<Op> rhsOps(JavaOp.JavaConditionalOp javaConditionalOp) {
        return javaConditionalOp.bodies().get(1).entryBlock().ops();
    }

    static Op.Result lhsResult(JavaOp.BinaryTestOp binaryTestOp) {
        return (Op.Result) binaryTestOp.operands().get(0);
    }

    static Op.Result rhsResult(JavaOp.BinaryTestOp binaryTestOp) {
        return (Op.Result) binaryTestOp.operands().get(1);
    }

    sealed interface LoadOrStore<T extends Op> extends OpHelper<T> permits VarAccess {
        boolean isLoad();

        boolean isStore();
    }

    sealed interface Named<T extends Op> extends OpHelper<T>
            permits FieldAccess, Func, Invoke, VarAccess, Variable {
        String name();

        default boolean nameMatchesRegex(Regex regex) {
            return regex.matches(name());
        }
        default boolean nameMatchesRegex(String regexStr) {
            return nameMatchesRegex(Regex.of(regexStr));
        }

        default boolean named(String... names) {
            return nameInSet(Set.of(names));
        }

        default boolean named(Predicate<String> predicate) {
            return predicate.test(name());
        }

        default boolean nameInSet(Set<String> set) {
            return set.contains(name());
        }
    }

    sealed interface VarAccess extends Named<CoreOp.VarAccessOp>, LoadOrStore<CoreOp.VarAccessOp> {
        @Override
        default String name() {
            return op().varOp().varName();
        }

        @Override
        default boolean isLoad() {
            return op() instanceof CoreOp.VarAccessOp.VarLoadOp;
        }

        @Override
        default boolean isStore() {
            return op() instanceof CoreOp.VarAccessOp.VarStoreOp;
        }

        default boolean isAssignable(Class<?> classes) {
            return isAssignable((JavaType) op().resultType(), classes);
        }

        record Impl(MethodHandles.Lookup lookup, CoreOp.VarAccessOp op) implements VarAccess {
        }

        static VarAccess varAccess(MethodHandles.Lookup lookup, CodeElement<?, ?> codeElement) {
            return codeElement instanceof CoreOp.VarAccessOp varAccessOp ? new VarAccess.Impl(lookup, varAccessOp) : null;
        }

        static Stream<VarAccess> stream(MethodHandles.Lookup lookup, CoreOp.FuncOp funcOp) {
            return funcOp.elements().filter(ce -> ce instanceof CoreOp.VarAccessOp).map(ce -> varAccess(lookup, ce));
        }
    }

    sealed interface Variable extends Named<CoreOp.VarOp> {
        @Override
        default String name() {
            return op().varName();
        }

        default boolean assignable(Class<?>... clazzes) {
            return isAssignable((JavaType) op().varValueType(), clazzes);
        }

        default TypeElement type() {
            return op().resultType().valueType();
        }

        default Invoke firstOperandAsInvoke() {
            return Invoke.invoke(lookup(), opFromFirstOperandOrNull());
        }

        record Impl(MethodHandles.Lookup lookup, CoreOp.VarOp op) implements Variable {
        }

        static Variable var(MethodHandles.Lookup lookup, CodeElement<?, ?> codeElement) {
            return codeElement instanceof CoreOp.VarOp varOp ? new Variable.Impl(lookup, varOp) : null;
        }

        static Stream<Variable> stream(MethodHandles.Lookup lookup, CoreOp.FuncOp funcOp) {
            return funcOp.elements().filter(ce -> ce instanceof CoreOp.VarOp).map(ce -> var(lookup, ce));
        }
    }


    sealed interface FieldAccess extends Named<JavaOp.FieldAccessOp> permits FieldAccess.Instance, FieldAccess.Static {

        @Override
        default String name() {
            return op().fieldDescriptor().name();
        }

        default boolean isPrimitive() {
            return op().result().type() instanceof PrimitiveType;
        }

        default TypeElement resultType() {
            return op().resultType();
        }

        default TypeElement refType() {
            return op().fieldDescriptor().refType();
        }

        default boolean refType(Class<?>... classes) {
            return OpHelper.isAssignable(lookup(), refType(), classes);
        }

        default Object getStaticFinalPrimitiveValue() {
            if (refType() instanceof ClassType classType) {
                Class<?> clazz = (Class<?>) classTypeToTypeOrThrow(lookup(), classType);
                try {
                    Field field = clazz.getField(name());
                    field.setAccessible(true);
                    return field.get(null);
                } catch (NoSuchFieldException | IllegalAccessException e) {
                    try {
                        Field field = clazz.getDeclaredField(name());
                        field.setAccessible(true);
                        return field.get(null);
                    } catch (NoSuchFieldException | IllegalAccessException e2) {
                        throw new RuntimeException(e2);
                    }
                }
            }
            throw new RuntimeException("Could not find field value" + op());
        }

        default boolean isLoad() {
            return op() instanceof JavaOp.FieldAccessOp.FieldLoadOp;
        }

        default boolean isStore() {
            return op() instanceof JavaOp.FieldAccessOp.FieldStoreOp;
        }

        static <F extends FieldAccess> F fieldAccess(MethodHandles.Lookup lookup, CodeElement<?, ?> codeElement) {
            return codeElement instanceof JavaOp.FieldAccessOp fieldAccessOp
                    ? fieldAccessOp.operands().isEmpty()
                    ?(F)new Static.Impl(lookup, fieldAccessOp)
                    :(F)new Instance.Impl(lookup, fieldAccessOp)
                    : null;
        }

        static Stream<FieldAccess> stream(MethodHandles.Lookup lookup, CoreOp.FuncOp funcOp) {
            return funcOp.elements().filter(ce -> ce instanceof JavaOp.FieldAccessOp).map(ce -> fieldAccess(lookup, ce));
        }
        sealed interface Static extends FieldAccess{
            record Impl(MethodHandles.Lookup lookup, JavaOp.FieldAccessOp op) implements Static {
            }
        }
        sealed interface Instance extends FieldAccess{
            record Impl(MethodHandles.Lookup lookup, JavaOp.FieldAccessOp op) implements Instance {
            }
            default Op.Result instance() {
                    return (Op.Result) op().operands().getFirst();
            }

            default Op instanceOp() {
                return instance() instanceof Op.Result result ? result.op() : null;
            }


            default VarAccess instanceVarAccess() {
                return instanceOp() instanceof CoreOp.VarAccessOp varAccessOp && VarAccess.varAccess(lookup(), varAccessOp) instanceof VarAccess varAccess ? varAccess : null;
            }

        }
    }

    sealed interface Func extends Named<CoreOp.FuncOp> {
        @Override
        default String name() {
            return op().funcName();
        }

        record Impl(MethodHandles.Lookup lookup, CoreOp.FuncOp op) implements Func {
        }

        static Func func(MethodHandles.Lookup lookup, CodeElement<?, ?> codeElement) {
            return codeElement instanceof CoreOp.FuncOp funcOp ? new Func.Impl(lookup, funcOp) : null;
        }

        static Func func(MethodHandles.Lookup lookup, Class<?> clazz, String name, Class<?>... parameterTypes) {
            try {
                var addMethod = Op.ofMethod(clazz.getDeclaredMethod(name, parameterTypes)).orElseThrow();
                return func(lookup, addMethod);
            } catch (NoSuchMethodException nsme) {
                throw new RuntimeException(nsme);
            }
        }
    }

    sealed interface Invoke extends Named<JavaOp.InvokeOp> permits  Invoke.Static, Invoke.Virtual {
        static Stream<Invoke> stream(MethodHandles.Lookup lookup, Op op) {
            return op.elements().filter(ce -> ce instanceof JavaOp.InvokeOp).map(ce -> invoke(lookup, ce));
        }

        static Stream<Invoke> stream(MethodHandles.Lookup lookup, Block block) {
            return block.ops().stream().filter(ce -> ce instanceof JavaOp.InvokeOp).map(ce -> invoke(lookup, ce));
        }



        @Override
        default String name() {
            return op().invokeDescriptor().name();
        }

        default  boolean returns(Class<?> clazz) {
            return isAssignable((JavaType) op().resultType(), clazz);
        }

        default boolean receives(Class<?>... classes) {
            boolean assignable = true;
            int adj = (this instanceof Virtual) ? 1 : 0;// for instance we compare op().operands(1..N) (0..N) for static
            if (classes.length != op().operands().size() - adj) {
                assignable = false;
            } else {
                for (int i = 0; assignable && i < classes.length && i < op().operands().size() - adj; i++) {
                    var operand = op().operands().get(i + adj);
                    TypeElement resultType = operand.type();
                    if (resultType instanceof JavaType javaType) {
                        assignable &= isAssignable(javaType, classes[i]);
                    } else {
                        assignable = false;
                    }
                }
            }
            return assignable;
        }

        default Method resolvedMethodOrNull() {
            try {
                return op().invokeDescriptor().resolveToMethod(lookup()) instanceof Method method ? method : null;
            } catch (ReflectiveOperationException rope) {
                return null;
            }
        }

        default boolean refIs(Class<?>... classes) {
            return OpHelper.isAssignable(lookup(), op().invokeDescriptor().refType(), classes);
        }

        default boolean returnsArray() {
            return op().resultType() instanceof ArrayType;
        }

        default boolean returnsVoid() {
            return op().invokeDescriptor().type().returnType().equals(JavaType.VOID);
        }

        default TypeElement returnType() {
            return op().invokeDescriptor().type().returnType();
        }

        default boolean returnsInt() {
            return returnType().equals(JavaType.INT);
        }


        default boolean returnsClassType() {
            return returnType() instanceof ClassType;
        }


        default TypeElement refType() {
            return op().invokeDescriptor().refType();
        }

        default boolean returnsPrimitive() {
            return returnType() instanceof PrimitiveType;
        }

        default boolean returnsFloat() {
            return returnType() == JavaType.FLOAT;
        }

        default boolean returnsChar() {
            return returnType() == JavaType.CHAR;
        }

        default boolean returnsShort() {
            return returnType() == JavaType.SHORT;
        }

        default boolean returns16BitValue() {
            return returnsChar() || returnsShort();
        }

        default Method resolveMethodOrNull() {
            try {
                return op().invokeDescriptor().resolveToMethod(lookup());
            } catch (ReflectiveOperationException e) {
                return null;
            }
        }

        default Method resolveMethodOrThrow() {
            try {
                return op().invokeDescriptor().resolveToMethod(lookup());
            } catch (ReflectiveOperationException e) {
                throw new RuntimeException(e);
            }
        }

        default Class<?> classOrThrow() {
            if (refType() instanceof ClassType classType) {
                return (Class<?>) classTypeToTypeOrThrow(lookup(), classType);
            } else {
                throw new IllegalStateException(" javaRef class is null");
            }
        }

        default boolean isMappableIface() {
            return refIs(MappableIface.class);
        }

        default List<AccessType.TypeAndAccess> paramaterAccessList() {
            Annotation[][] parameterAnnotations = resolveMethodOrThrow().getParameterAnnotations();
            int firstParam = (this instanceof Virtual) ? 1 : 0; // if virtual
            List<AccessType.TypeAndAccess> typeAndAccesses = new ArrayList<>();
            for (int i = firstParam; i < operandCount(); i++) {
                typeAndAccesses.add(AccessType.TypeAndAccess.of(parameterAnnotations[i - firstParam], op().operands().get(i)));
            }
            return typeAndAccesses;
        }

        default CoreOp.VarOp varOpFromFirstUseOrThrow() {
            var iterator = op().result().uses().iterator();
            if (iterator.hasNext() && iterator.next().op() instanceof CoreOp.VarOp varOp) {
                return varOp;
            } else {
                throw new RuntimeException("Expecting first use of invoke to be VarOp");
            }
        }

        default CoreOp.FuncOp targetMethodModelOrThrow() {
            Method method = resolveMethodOrNull();
            return OpHelper.methodModelOrThrow(method);
        }

        default Op onlyUse() {
            if (op().result().uses().size() == 1) {
                return op().result().uses().iterator().next().op();
            } else {
                return null;
            }
        }
        static <I extends Invoke>I invoke(MethodHandles.Lookup lookup, CodeElement<?, ?> codeElement) {
            return codeElement instanceof JavaOp.InvokeOp invokeOp ?
                    invokeOp.invokeKind().equals(JavaOp.InvokeOp.InvokeKind.STATIC)
                            ? (I)new Static.Impl(lookup,invokeOp)
                            : (I) new Virtual.Impl(lookup, invokeOp)
                    : null;
        }

        default Op.Result returnResult() {
            return op().result();
        }

        static Invoke getTargetInvoke(MethodHandles.Lookup lookup, JavaOp.LambdaOp lambdaOp, Class<?>... classes) {
            return (Invoke) lambdaOp.body().entryBlock().ops().stream()
                    .filter(ce -> ce instanceof JavaOp.InvokeOp)
                    .map(ce -> invoke(lookup, ce))
                    .filter(invoke -> OpHelper.isAssignable(lookup, ((Invoke)invoke).op().operands().getFirst().type(), classes))
                    .findFirst()
                    .orElseThrow();
        }

        sealed interface Virtual extends Invoke{
            default Op.Result instance() {
                    return (Op.Result) op().operands().getFirst();
            }

            default Op instanceOp() {
                return instance() instanceof Op.Result result ? result.op() : null;
            }

            default VarAccess instanceVarAccess() {
                return instanceOp() instanceof CoreOp.VarAccessOp varAccessOp && VarAccess.varAccess(lookup(), varAccessOp) instanceof VarAccess varAccess ? varAccess : null;
            }

            default boolean isInstanceAccessedViaVarAccess() {
                return instanceVarAccess() != null;
            }

            record Impl(MethodHandles.Lookup lookup, JavaOp.InvokeOp op) implements Virtual {
            }
        }
        sealed interface Static extends Invoke{
            record Impl(MethodHandles.Lookup lookup, JavaOp.InvokeOp op) implements Static {
            }
        }
    }

    interface OpSpan {
        List<Op> ops();

        default Op from() {
            return ops().getFirst();
        }

        default Op to() {
            return ops().getLast();
        }

        default boolean firstOrLast(Op op) {
            return isFirst(op) || isLast(op);
        }

        default boolean isFirst(Op op) {
            return to().equals(op);
        }

        default boolean isLast(Op op) {
            return from().equals(op);
        }
    }

    interface Statement {

        static <T extends OpSpan> Map<Op, T> createOpToStatementSpanMap(CoreOp.FuncOp funcOp,
                                                                        Predicate<Op> predicate,
                                                                        Function<List<Op>, T> factory) {
            Map<Op, T> opToStatementSpanMap = new LinkedHashMap<>();
            funcOp.elements()
                    .filter(ce -> ce instanceof Block)
                    .map(ce -> (Block) ce)
                    .forEach(block -> {
                        var statementOps = new LinkedList<Op>();
                        block.ops().forEach(op -> {
                            statementOps.add(op);
                            if (Statement.asStatementOpOrNull(op) != null) {
                                if (statementOps.stream().anyMatch(predicate)) {
                                    T span = factory.apply(new LinkedList<>(statementOps));
                                    statementOps.forEach(opInList -> // we take a snapshot of statementOps
                                            opToStatementSpanMap.put(opInList, span)
                                    );
                                }
                                statementOps.clear(); // and then cleat for the next one
                            }
                        });
                    });
            return opToStatementSpanMap;
        }

        static <T extends OpSpan> Map<Op, T> createOpToStatementSpanMap(CoreOp.FuncOp funcOp, Function<List<Op>, T> factory) {
            return createOpToStatementSpanMap(funcOp, _ -> true, factory);
        }

        static Op asStatementOpOrNull(CodeElement<?, ?> ce) {
            if (ce instanceof Op op) {
                return (
                        (
                                (op instanceof CoreOp.VarAccessOp.VarStoreOp && op.operands().get(1).uses().size() < 2)
                                        || (op instanceof CoreOp.VarOp || (op.result() instanceof Op.Result result && result.uses().isEmpty()))
                                        || (op instanceof StatementLikeOp)
                        )
                                && !(op instanceof CoreOp.VarOp varOp && (!varOp.isUninitialized()
                                && varOp.operands().getFirst() instanceof Block.Parameter parameter
                                && parameter.invokableOperation() instanceof CoreOp.FuncOp)
                        )
                                && !(op instanceof CoreOp.YieldOp)

                )
                        ? op
                        : null;
            } else {
                return null;
            }

        }

        static boolean isStatementOp(CodeElement<?, ?> ce) {
            return Objects.nonNull(asStatementOpOrNull(ce));
        }

        static Stream<Op> statements(Block block) {
            return block.ops().stream().filter(Statement::isStatementOp);
        }

        static Stream<Op> bodyStatements(Body body) {
            var list = new ArrayList<>(statements(body.entryBlock()).toList());
            if (list.getLast() instanceof JavaOp.ContinueOp) {
                list.removeLast();
            }
            return list.stream();
        }

    }

    sealed interface Ternary extends OpHelper<JavaOp.ConditionalExpressionOp> {
        default <T> boolean isAssignable(Class<T> clazz) {
            return isAssignable((JavaType) op().resultType(), clazz);
        }

        default Block condBlock() {
            return OpHelper.entryBlockOfBodyN(op(), 0);
        }

        default Block thenBlock() {
            return OpHelper.entryBlockOfBodyN(op(), 1);
        }

        default Block elseBlock() {
            return OpHelper.entryBlockOfBodyN(op(), 2);
        }

        record Impl(MethodHandles.Lookup lookup, JavaOp.ConditionalExpressionOp op) implements Ternary {
        }

        static Ternary ternary(MethodHandles.Lookup lookup, CodeElement<?, ?> codeElement) {
            return codeElement instanceof JavaOp.ConditionalExpressionOp op ? new Impl(lookup, op) : null;
        }
    }

    sealed interface Lambda extends OpHelper<JavaOp.LambdaOp> {
        default <T> boolean isAssignable(Class<T> clazz) {
            return isAssignable((JavaType) op().resultType(), clazz);
        }

        record Impl(MethodHandles.Lookup lookup, JavaOp.LambdaOp op) implements Lambda {
        }

        static Lambda lambda(MethodHandles.Lookup lookup, CodeElement<?, ?> codeElement) {
            return codeElement instanceof JavaOp.LambdaOp lambdaOp ? new Impl(lookup, lambdaOp) : null;
        }

        default Object[] getQuotedCapturedValues(Quoted quoted, Method method) {
            var block = op().body().entryBlock();
            var ops = block.ops();
            Object[] varLoadNames = ops.stream()
                    .filter(op -> op instanceof CoreOp.VarAccessOp.VarLoadOp)
                    .map(op -> (CoreOp.VarAccessOp.VarLoadOp) op)
                    .map(varLoadOp -> (Op.Result) varLoadOp.operands().getFirst())
                    .map(varLoadOp -> (CoreOp.VarOp) varLoadOp.op())
                    .map(CoreOp.VarOp::varName).toArray();
            Map<String, Object> nameValueMap = new HashMap<>();

            quoted.capturedValues().forEach((k, v) -> {
                if (k instanceof Op.Result result) {
                    if (result.op() instanceof CoreOp.VarOp varOp) {
                        nameValueMap.put(varOp.varName(), v);
                    }
                }
            });
            Object[] args = new Object[method.getParameterCount()];
            if (args.length != varLoadNames.length) {
                throw new IllegalStateException("Why don't we have enough captures.!! ");
            }
            for (int i = 1; i < args.length; i++) {
                args[i] = nameValueMap.get(varLoadNames[i].toString());
                if (args[i] instanceof CoreOp.Var<?> var) {
                    args[i] = var.value();
                }
            }
            return args;

        }

    }


    sealed interface Binary extends OpHelper<JavaOp.BinaryOp> {
        default <T> boolean isAssignable(Class<T> clazz) {
            return isAssignable((JavaType) op().resultType(), clazz);
        }

        record Impl(MethodHandles.Lookup lookup, JavaOp.BinaryOp op) implements Binary {
        }

        static Binary binary(MethodHandles.Lookup lookup, CodeElement<?, ?> codeElement) {
            return codeElement instanceof JavaOp.BinaryOp binaryOp ? new Impl(lookup, binaryOp) : null;
        }
    }

    static Block.Parameter getFuncParamOrNull(Op op, int n) {
        while (op != null && !(op instanceof CoreOp.FuncOp)) {
            op = op.ancestorOp();
        }
        if (op instanceof CoreOp.FuncOp funcOp) {
            return funcOp.bodies().get(0).entryBlock().parameters().get(n);
        } else {
            return null;
        }
    }

    static Block.Parameter getFuncParamOrThrow(Op op, int n) {
        if (getFuncParamOrNull(op, n) instanceof Block.Parameter parameter) {
            return parameter;
        } else {
            throw new IllegalStateException("cant find func parameter parameter " + n);
        }
    }
}

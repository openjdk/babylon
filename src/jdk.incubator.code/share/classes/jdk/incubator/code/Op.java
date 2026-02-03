/*
 * Copyright (c) 2024, 2026, Oracle and/or its affiliates. All rights reserved.
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

import com.sun.tools.javac.api.JavacScope;
import com.sun.tools.javac.api.JavacTrees;
import com.sun.tools.javac.code.Symbol.ClassSymbol;
import com.sun.tools.javac.comp.Attr;
import com.sun.tools.javac.model.JavacElements;
import com.sun.tools.javac.processing.JavacProcessingEnvironment;
import com.sun.tools.javac.tree.JCTree.JCMethodDecl;
import com.sun.tools.javac.tree.TreeMaker;
import com.sun.tools.javac.util.Context;
import jdk.incubator.code.dialect.core.CoreType;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.internal.ReflectMethods;
import jdk.incubator.code.dialect.core.CoreOp.FuncOp;
import jdk.incubator.code.dialect.core.FunctionType;
import jdk.incubator.code.dialect.java.MethodRef;
import jdk.incubator.code.extern.OpWriter;
import jdk.internal.access.SharedSecrets;

import javax.annotation.processing.ProcessingEnvironment;
import javax.lang.model.element.ExecutableElement;
import javax.lang.model.element.Modifier;
import java.lang.reflect.Method;
import java.lang.reflect.Proxy;
import java.util.*;
import java.util.function.BiFunction;

/**
 * An operation modelling a unit of program behaviour.
 * <p>
 * An operation might model the addition of two integers, or a method invocation expression.
 * Alternatively an operation may model something more complex like method declarations, lambda expressions, or
 * try statements. In such cases an operation will contain one or more bodies modelling the nested structure.
 * <p>
 * An instance of an operation when initially constructed is referred to as an unbuilt operation.
 * An unbuilt operation's state and descendants are all immutable except for its {@link #result result} and
 * {@link #parent parent}, which are initially set to {@code null}.
 * <p>
 * An unbuilt operation transitions to a built operation in one of two ways:
 * <ol>
 * <li>
 * {@link #buildAsRoot() building} the unbuilt operation to become a built {@link #isRoot() root} operation. The
 * operation's {@link #result result} and {@link #parent parent} are always {@code null}.
 * </li>
 * <li>
 * {@link Block.Builder#op(Op) appending} the unbuilt operation to a block builder to first become an unbuilt-bound
 * operation that is bound to an operation result and parent block.
 * The unbuilt-bound operation has a non-{@code null} unbuilt {@link #result result} that never changes, an unbuilt
 * value that can be used by subsequent constructed operations.
 * An unbuilt-bound operation transitions to a built bound operation when the block builder it was appended to builds
 * the block, after which the built bound operation has a non-{@code null} {@link #parent parent} that never changes and
 * a built {@link #result}.
 * Before then the unbuilt-bound operation's {@link #parent parent} is inaccessible, as is unbuilt result's
 * {@link Value#declaringBlock() declaring block}) (since both refer to the same block).
 * </li>
 * A built operation is fully immutable either as a root operation, the root of a code model, or as a bound operation
 * within a code model.
 * <p>
 * An operation can only be constructed with unbuilt values as operands (if any) and unbuilt block references as
 * successors (if any), otherwise construction fails with an exception.
 */
public non-sealed abstract class Op implements CodeElement<Op, Body> {

    /**
     * An operation characteristic indicating the operation is pure and has no side effects.
     */
    public interface Pure {
    }

    /**
     * An operation characteristic indicating the operation has one or more bodies.
     */
    public interface Nested {
        /**
         * {@return the bodies of the nested operation.}
         */
        List<Body> bodies();
    }

    /**
     * An operation characteristic indicating the operation represents a loop
     */
    public interface Loop extends Nested {
        /**
         * {@return the body of the loop operation.}
         */
        Body loopBody();
    }

    /**
     * An operation characteristic indicating the operation has one or more bodies,
     * all of which are isolated and capture no values.
     */
    public interface Isolated extends Nested {
    }

    /**
     * An operation characteristic indicating the operation is invokable.
     */
    public interface Invokable extends Nested {
        /**
         * {@return the body of the invokable operation.}
         */
        Body body();

        /**
         * {@return the function type describing the invokable operation's parameter types and return type.}
         */
        FunctionType invokableType();

        /**
         * {@return the entry block parameters of this operation's body}
         */
        default List<Block.Parameter> parameters() {
            return body().entryBlock().parameters();
        }

        /**
         * Computes values captured by this invokable operation's body.
         *
         * @return the captured values.
         * @see Body#capturedValues()
         */
        default List<Value> capturedValues() {
            return List.of();
        }
    }

    /**
     * An operation characteristic indicating the operation can replace itself with a lowered form.
     */
    // @@@ Hide this abstraction within JavaOp?
    public interface Lowerable {

        /**
         * Lowers this operation into the block builder, commonly replacing nested structure
         * with interconnected basic blocks. The previous lowering code transformation
         * is used to compose with a lowering transformation that is applied to bodies
         * of this operation, ensuring lowering is applied consistently to nested content.
         *
         * @param b the block builder
         * @param opT the previous lowering code transformation, may be {@code null}
         * @return the block builder to use for further building
         */
        Block.Builder lower(Block.Builder b, CodeTransformer opT);

        /**
         * Returns a composed code transformer that composes with an operation transformer function adapted to lower
         * operations.
         * <p>
         * This method behaves as if it returns the result of the following expression:
         * {@snippet lang = java:
         * CodeTransformer.andThen(before, lowering(before, f));
         *}
         *
         * @param before the code transformer to apply before
         * @param f the operation transformer function to apply after
         * @return the composed code transformer
         */
        static CodeTransformer andThenLowering(CodeTransformer before, BiFunction<Block.Builder, Op, Block.Builder> f) {
            return CodeTransformer.andThen(before, lowering(before, f));
        }

        /**
         * Returns an adapted operation transformer function that adapts an operation transformer function
         * {@code f} to also transform lowerable operations.
         * <p>
         * The adapted operation transformer function first applies a block builder and operation
         * to the operation transformer function {@code f}.
         * If the result is not {@code null} then the result is returned.
         * Otherwise, if the operation is a lowerable operation then the result of applying the
         * block builder and code transformer {@code before} to {@link Lowerable#lower lower}
         * of the lowerable operation is returned.
         * Otherwise, the operation is copied by applying it to {@link Block.Builder#op op} of the block builder,
         * and the block builder is returned.
         *
         * @param before the code transformer to apply for lowering
         * @param f the operation transformer function to apply after
         * @return the adapted operation transformer function
         */
        static BiFunction<Block.Builder, Op, Block.Builder> lowering(CodeTransformer before, BiFunction<Block.Builder, Op, Block.Builder> f) {
            return (block, op) -> {
                Block.Builder b = f.apply(block, op);
                if (b == null) {
                    if (op instanceof Lowerable lop) {
                        block = lop.lower(block, before);
                    } else {
                        block.op(op);
                    }
                } else {
                    block = b;
                }
                return block;
            };
        }
    }

    /**
     * An operation characteristic indicating the operation is a terminating operation
     * occurring as the last operation in a block.
     * <p>
     * A terminating operation passes control to either another block within the same parent body
     * or to that parent body.
     */
    public interface Terminating {
    }

    /**
     * An operation characteristic indicating the operation is a body terminating operation
     * occurring as the last operation in a block.
     * <p>
     * A body terminating operation passes control back to its nearest ancestor body.
     */
    public interface BodyTerminating extends Terminating {
    }

    /**
     * An operation characteristic indicating the operation is a block terminating operation
     * occurring as the last operation in a block.
     * <p>
     * The operation has one or more successors to other blocks within the same parent body, and passes
     * control to one of those blocks.
     */
    public interface BlockTerminating extends Terminating {
        List<Block.Reference> successors();
    }

    /**
     * A value that is the result of an operation.
     */
    public static final class Result extends Value {

        /**
         * If assigned to an operation result, it indicates the operation is a root operation
        */
        private static final Result ROOT_RESULT = new Result();

        final Op op;

        private Result() {
            // Constructor for instance of ROOT_RESULT
            super(null, null);
            this.op = null;
        }

        Result(Block block, Op op) {
            super(block, op.resultType());

            this.op = op;
        }

        @Override
        public String toString() {
            return "%result@" + Integer.toHexString(hashCode());
        }

        @Override
        public SequencedSet<Value> dependsOn() {
            SequencedSet<Value> depends = new LinkedHashSet<>(op.operands());
            if (op instanceof Terminating) {
                op.successors().stream().flatMap(h -> h.arguments().stream()).forEach(depends::add);
            }

            return Collections.unmodifiableSequencedSet(depends);
        }

        /**
         * {@return the result's declaring operation.}
         */
        public Op op() {
            return op;
        }
    }

    /**
     * Source location information for an operation.
     *
     * @param sourceRef the reference to the source, {@code null} if absent
     * @param line the line in the source
     * @param column the column in the source
     */
    public record Location(String sourceRef, int line, int column) {

        /**
         * The location value, {@code null}, indicating no location information.
         */
        public static final Location NO_LOCATION = null;

        /**
         * Constructions a location with line and column only.
         *
         * @param line the line in the source
         * @param column the column in the source
         */
        public Location(int line, int column) {
            this(null, line, column);
        }
    }

    // Set when op is unbuilt-bound or root, otherwise null when unbuilt
    // @@@ stable value?
    Result result;

    // null if not specified
    // @@@ stable value?
    Location location;

    final List<Value> operands;

    /**
     * Constructs an operation from a given operation.
     * <p>
     * The constructor defers to the {@link Op#Op(List) operands} constructor passing a list of values computed, in
     * order, by mapping the given operation's operands using the code context. The constructor also assigns the new
     * operation's location to the given operation's location, if any.
     *
     * @param that the given operation.
     * @param cc   the code context.
     */
    protected Op(Op that, CodeContext cc) {
        List<Value> outputOperands = cc.getValues(that.operands);
        // Values should be guaranteed to connect to unbuilt blocks since
        // the context only allows such mappings, assert for clarity
        assert outputOperands.stream().noneMatch(Value::isBuilt);
        this.operands = List.copyOf(outputOperands);
        this.location = that.location;
    }

    /**
     * Copies this operation and transforms its bodies, if any.
     * <p>
     * Bodies are {@link Body#transform(CodeContext, CodeTransformer) transformed} with the given code context and
     * code transformer.
     * @apiNote
     * To copy an operation use the {@link CodeTransformer#COPYING_TRANSFORMER copying transformer}.
     *
     * @param cc the code context.
     * @param ot the code transformer.
     * @return the transformed operation.
     * @see CodeTransformer#COPYING_TRANSFORMER
     */
    public abstract Op transform(CodeContext cc, CodeTransformer ot);

    /**
     * Constructs an operation with a list of operands.
     *
     * @param operands the list of operands, a copy of the list is performed if required.
     * @throws IllegalArgumentException if an operand is built because its declaring block is built.
     */
    protected Op(List<? extends Value> operands) {
        for (Value operand : operands) {
            if (operand.isBuilt()) {
                throw new IllegalArgumentException("Operand's declaring block is built: " + operand);
            }
        }
        this.operands = List.copyOf(operands);
    }

    /**
     * Sets the originating source location of this operation, if this operation is not built.
     *
     * @param l the location, the {@link Location#NO_LOCATION} value indicates the location is not specified.
     * @throws IllegalStateException if this operation is built.
     */
    public final void setLocation(Location l) {
        // @@@ Fail if location != null?
        if (isRoot() || (result != null && result.block.isBuilt())) {
            throw new IllegalStateException("Built operation");
        }

        location = l;
    }

    /**
     * {@return the originating source location of this operation, otherwise {@code null} if not specified}
     */
    public final Location location() {
        return location;
    }

    /**
     * Returns this operation's parent block, otherwise {@code null} if this operation is unbuilt or a root.
     * <p>
     * The operation's parent block is the same as the operation result's {@link Value#declaringBlock declaring block}.
     *
     * @return operation's parent block, or {@code null} if this operation is unbuilt or a root.
     * @throws IllegalStateException if this operation is unbuilt-bound.
     * @see Value#declaringBlock()
     */
    @Override
    public final Block parent() {
        if (isRoot() || result == null) {
            return null;
        }

        if (!result.block.isBuilt()) {
            throw new IllegalStateException("Unbuilt-bound operation");
        }

        return result.block;
    }

    @Override
    public final List<Body> children() {
        return bodies();
    }

    /**
     * {@return the operation's bodies, as an unmodifiable list}
     * @implSpec this implementation returns an unmodifiable empty list.
     * @see #children()
     */
    public List<Body> bodies() {
        return List.of();
    }

    /**
     * {@return the operation's result type}
     */
    public abstract TypeElement resultType();


    /**
     * Returns the operation's result, otherwise {@code null} if this operation is unbuilt or a
     * root.
     *
     * @return the operation's result, or {@code null} if this operation is unbuilt or a root.
     */
    public final Result result() {
        return result == Result.ROOT_RESULT ? null : result;
    }

    /**
     * {@return the operation's operands, as an unmodifiable list}
     */
    public final List<Value> operands() {
        return operands;
    }

    /**
     * {@return the operation's successors, as an unmodifiable list}
     * @implSpec this implementation returns an unmodifiable empty list.
     */
    public List<Block.Reference> successors() {
        return List.of();
    }

    /**
     * Returns the operation's function type.
     * <p>
     * The function type's result type is the operation's result type and its parameter types are the
     * operation's operand types, in order.
     *
     * @return the function type
     */
    public final FunctionType opType() {
        List<TypeElement> operandTypes = operands.stream().map(Value::type).toList();
        return CoreType.functionType(resultType(), operandTypes);
    }

    /**
     * Computes values captured by this operation. A captured value is a value that is used
     * but not declared by any descendant operation of this operation.
     * <p>
     * The order of the captured values is first use encountered in depth
     * first search of this operation's descendant operations.
     *
     * @return the list of captured values, modifiable
     * @see Body#capturedValues()
     */
    public final List<Value> capturedValues() {
        Set<Value> cvs = new LinkedHashSet<>();

        Deque<Body> bodyStack = new ArrayDeque<>();
        for (Body childBody : bodies()) {
            Body.capturedValues(cvs, bodyStack, childBody);
        }
        return new ArrayList<>(cvs);
    }

    /**
     * Builds this operation to become a built root operation. After this operation is built its
     * {@link #result result} and {@link #parent parent} will always be {@code null}.
     * <p>
     * This method is idempotent.
     *
     * @throws IllegalStateException if this operation is unbuilt-bound.
     * @see #isRoot()
     */
    public final void buildAsRoot() {
        if (result == Result.ROOT_RESULT) {
            return;
        }
        if (result != null) {
            throw new IllegalStateException("Operation is unbuilt-bound to a parent block");
        }
        result = Result.ROOT_RESULT;
    }

    /**
     * {@return {@code true} if this operation is a root operation.}
     * @see #buildAsRoot()
     * @see #isBound()
     * */
    public final boolean isRoot() {
        return result == Result.ROOT_RESULT;
    }

    /**
     * {@return {@code true} if this operation is a bound to a result and parent block.}
     * @see #buildAsRoot()
     * @see #isRoot()
     * */
    public final boolean isBound() {
        return !isRoot() && result != null;
    }

    /**
     * Externalizes this operation's name as a string.
     * @implSpec this implementation returns the result of the expression {@code this.getClass().getName()}.
     *
     * @return the operation name
     */
    public String externalizeOpName() {
        return this.getClass().getName();
    }

    /**
     * Externalizes this operation's specific state as a map of attributes.
     *
     * <p>A null attribute value is represented by the constant
     * value {@link jdk.incubator.code.extern.ExternalizedOp#NULL_ATTRIBUTE_VALUE}.
     * @implSpec this implementation returns an unmodifiable empty map.
     *
     * @return the operation's externalized state, as an unmodifiable map
     */
    public Map<String, Object> externalize() {
        return Map.of();
    }

    /**
     * Returns the textual form of this operation.
     *
     * @return the textual form of this operation.
     */
    public final String toText() {
        return OpWriter.toText(this);
    }


    /**
     * Returns the code model of a reflectable lambda expression or method reference.
     *
     * @param fiInstance a functional interface instance that is the result of a reflectable lambda expression or
     *                   method reference.
     * @return the code model, or an empty optional if the functional interface instance is not the result of a
     * reflectable lambda expression or method reference.
     * @throws UnsupportedOperationException if the Java version used at compile time to generate and store the code
     * model is not the same as the Java version used at runtime to load the code model.
     * @apiNote if the functional interface instance is a proxy instance, then the code model is unavailable and this
     * method returns an empty optional.
     */
    public static Optional<Quoted<JavaOp.LambdaOp>> ofLambda(Object fiInstance) {
        Object oq = fiInstance;
        if (Proxy.isProxyClass(oq.getClass())) {
            // @@@ The interpreter implements interpretation of
            // lambdas using a proxy whose invocation handler
            // supports the internal protocol to access the quoted instance
            oq = Proxy.getInvocationHandler(oq);
        }

        Method method;
        try {
            method = oq.getClass().getDeclaredMethod("__internal_quoted");
        } catch (NoSuchMethodException e) {
            return Optional.empty();
        }
        method.setAccessible(true);

        Quoted<?> q;
        try {
            q = (Quoted<?>) method.invoke(oq);
        } catch (ReflectiveOperationException e) {
            // op method may throw UOE in case java compile time version doesn't match runtime version
            if (e.getCause() instanceof UnsupportedOperationException uoe) {
                throw uoe;
            }
            throw new RuntimeException(e);
        }
        if (!(q.op() instanceof JavaOp.LambdaOp)) {
            // This can only happen if the stored model is invalid
            throw new RuntimeException("Invalid code model for lambda expression : " + q);
        }
        @SuppressWarnings("unchecked")
        Quoted<JavaOp.LambdaOp> lq = (Quoted<JavaOp.LambdaOp>) q;
        return Optional.of(lq);
    }

    /**
     * Returns the code model of a reflectable method.
     *
     * @param method the method.
     * @return the code model, or an empty optional if the method is not reflectable.
     * @throws UnsupportedOperationException if the Java version used at compile time to generate and store the code
     * model is not the same as the Java version used at runtime to load the code model.
     */
    // @@@ Make caller sensitive with the same access control as invoke
    // and throwing IllegalAccessException
    // @CallerSensitive
    @SuppressWarnings("unchecked")
    public static Optional<FuncOp> ofMethod(Method method) {
        return (Optional<FuncOp>)SharedSecrets.getJavaLangReflectAccess()
                .setCodeModelIfNeeded(method, Op::createCodeModel);
    }

    private static Optional<FuncOp> createCodeModel(Method method) {
        char[] sig = MethodRef.method(method).toString().toCharArray();
        for (int i = 0; i < sig.length; i++) {
            switch (sig[i]) {
                case '.', ';', '[', '/': sig[i] = '$';
            }
        }
        String opMethodName = new String(sig);
        Method opMethod;
        try {
            // @@@ Use method handle with full power mode
            opMethod = method.getDeclaringClass().getDeclaredMethod(opMethodName);
        } catch (NoSuchMethodException e) {
            return Optional.empty();
        }
        opMethod.setAccessible(true);
        try {
            FuncOp funcOp = (FuncOp) opMethod.invoke(null);
            return Optional.of(funcOp);
        } catch (ReflectiveOperationException e) {
            // op method may throw UOE in case java compile time version doesn't match runtime version
            if (e.getCause() instanceof UnsupportedOperationException uoe) {
                throw uoe;
            }
            throw new RuntimeException(e);
        }
    }

    /**
     * Returns the code model of provided executable element (if any).
     * <p>
     * If the executable element has a code model then it will be an instance of
     * {@code java.lang.reflect.code.op.CoreOps.FuncOp}.
     * Note: due to circular dependencies we cannot refer to the type explicitly.
     *
     * @param e the executable element.
     * @return the code model of the provided executable element (if any).
     */
    public static Optional<FuncOp> ofElement(ProcessingEnvironment processingEnvironment, ExecutableElement e) {
        if (e.getModifiers().contains(Modifier.ABSTRACT) ||
                e.getModifiers().contains(Modifier.NATIVE)) {
            return Optional.empty();
        }

        Context context = ((JavacProcessingEnvironment)processingEnvironment).getContext();
        ReflectMethods reflectMethods = ReflectMethods.instance(context);
        Attr attr = Attr.instance(context);
        JavacElements elements = JavacElements.instance(context);
        JavacTrees javacTrees = JavacTrees.instance(context);
        TreeMaker make = TreeMaker.instance(context);
        try {
            JCMethodDecl methodTree = (JCMethodDecl)elements.getTree(e);
            JavacScope scope = javacTrees.getScope(javacTrees.getPath(e));
            ClassSymbol enclosingClass = (ClassSymbol) scope.getEnclosingClass();
            FuncOp op = attr.runWithAttributedMethod(scope.getEnv(), methodTree,
                    attribBlock -> {
                        try {
                            return reflectMethods.getMethodBody(enclosingClass, methodTree, attribBlock, make);
                        } catch (Throwable ex) {
                            // this might happen if the source code contains errors
                            return null;
                        }
                    });
            return Optional.ofNullable(op);
        } catch (RuntimeException ex) {  // ReflectMethods.UnsupportedASTException
            // some other error occurred when attempting to attribute the method
            // @@@ better report of error
            ex.printStackTrace();
            return Optional.empty();
        }
    }
}

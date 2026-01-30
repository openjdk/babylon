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
 * An instance of an operation when initially constructed is referred to as an unbound operation.
 * An unbound operation's state and descendants are all immutable but its {@link #result result} and
 * {@link #parent parent} have yet to be assigned (both methods return {@code null}).
 * Since an unbound operation has no parent it can also be considered an unbound root operation.
 * <p>
 * An unbound operation becomes a bound operation if it is explicitly bound as a root operation, or
 * is bound to a parent block that is being built by {@link Block.Builder#op(Op) appending} it to the block's
 * {@link Block.Builder}. Once an operation is bound it cannot be unbound and so its {@link #result result} and
 * {@link #parent parent} will no longer change. A bound operation is therefore fully immutable either as a bound root
 * operation, the root of a code model, or as some operation within a code model (that is built or in the process of
 * being built).
 * <p>
 * When an operation is bound to a parent block that is being built that block is not accessible as a {@link #parent()}
 * until the block is fully built.
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
         * If assigned to an operation result, it indicates the operation is a bound root operation
        */
        private static final Result BOUND_ROOT_RESULT = new Result();

        final Op op;

        private Result() {
            // Constructor for instance of BOUND_ROOT_RESULT
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
     * @param sourceRef the reference to the source
     * @param line the line in the source
     * @param column the column in the source
     */
    public record Location(String sourceRef, int line, int column) {

        /**
         * The location value, {@code null}, indicating no location information.
         */
        public static final Location NO_LOCATION = null;

        public Location(int line, int column) {
            this(null, line, column);
        }

        @Override
        public String toString() {
            StringBuilder s = new StringBuilder();
            s.append(line).append(":").append(column);
            if (sourceRef != null) {
                s.append(":").append(sourceRef);
            }
            return s.toString();
        }

        public static Location fromString(String s) {
            String[] split = s.split(":", 3);
            if (split.length < 2) {
                throw new IllegalArgumentException();
            }

            int line = Integer.parseInt(split[0]);
            int column = Integer.parseInt(split[1]);
            String sourceRef;
            if (split.length == 3) {
                sourceRef = split[2];
            } else {
                sourceRef = null;
            }
            return new Location(sourceRef, line, column);
        }
    }

    // Set when op is bound to block, otherwise null when unbound
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
        this(cc.getValues(that.operands));
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
     */
    protected Op(List<? extends Value> operands) {
        this.operands = List.copyOf(operands);
    }

    /**
     * Sets the originating source location of this operation, if unbound or the parent block is not yet built.
     *
     * @param l the location, the {@link Location#NO_LOCATION} value indicates the location is not specified.
     * @throws IllegalStateException if this operation is bound.
     */
    public final void setLocation(Location l) {
        // @@@ Fail if location != null?
        if (isBoundAsRoot() || (result != null && result.block.isBound())) {
            throw new IllegalStateException();
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
     * Returns this operation's parent block, otherwise {@code null} if the operation is
     * a root operation (either bound or unbound).
     *
     * @return operation's parent block, or {@code null} if the operation is a root operation.
     * @throws IllegalStateException if an unbuilt block is encountered.
     */
    @Override
    public final Block parent() {
        if (isBoundAsRoot() || result == null) {
            return null;
        }

        if (!result.block.isBound()) {
            throw new IllegalStateException("Parent block is not built");
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
     * Returns the operation's result, otherwise {@code null} if the operation is unbound.
     *
     * @return the operation's result, or {@code null} if unbound.
     */
    public final Result result() {
        return result == Result.BOUND_ROOT_RESULT ? null : result;
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
     * Binds this operation to become a bound root operation. After this operation is bound its
     * {@link #result result} and {@link #parent parent} are guaranteed to always be {@code null}.
     * <p>
     * This method is idempotent.
     *
     * @throws IllegalStateException if this operation a bound to a parent block.
     * @see #isBoundAsRoot()
     */
    public final void bindAsRoot() {
        if (result == Result.BOUND_ROOT_RESULT) {
            return;
        }
        if (result != null) {
            throw new IllegalStateException("Operation is bound to a parent block");
        }
        result = Result.BOUND_ROOT_RESULT;
    }

    /**
     * {@return {@code true} if this operation is a bound root operation.}
     * @see #bindAsRoot()
     * @see #isBound()
     * */
    public final boolean isBoundAsRoot() {
        return result == Result.BOUND_ROOT_RESULT;
    }

    /**
     * {@return {@code true} if this operation is a bound operation.}
     * @see #bindAsRoot()
     * @see #isBoundAsRoot()
     * */
    public final boolean isBound() {
        return result != null;
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

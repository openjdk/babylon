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
import jdk.incubator.code.internal.ReflectMethods;
import jdk.incubator.code.dialect.core.CoreOp.FuncOp;
import jdk.incubator.code.dialect.core.FunctionType;
import jdk.incubator.code.dialect.java.MethodRef;
import jdk.incubator.code.extern.OpWriter;
import jdk.internal.access.SharedSecrets;

import javax.annotation.processing.ProcessingEnvironment;
import javax.lang.model.element.ExecutableElement;
import javax.lang.model.element.Modifier;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.lang.reflect.Proxy;
import java.util.*;
import java.util.function.BiFunction;

/**
 * An operation modelling a unit of functionality.
 * <p>
 * An operation might model the addition of two 32-bit integers, or a Java method call.
 * Alternatively an operation may model something more complex like method bodies, lambda bodies, or
 * try/catch/finally statements. In this case such an operation will contain one or more bodies modelling
 * the nested structure.
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
        List<Body> bodies();
    }

    /**
     * An operation characteristic indicating the operation represents a loop
     */
    public interface Loop extends Nested {
        Body loopBody();
    }

    /**
     * An operation characteristic indicating the operation has one or more bodies,
     * all of which are isolated.
     */
    public interface Isolated extends Nested {
    }

    /**
     * An operation characteristic indicating the operation is invokable, so the operation may be interpreted
     * or compiled.
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
        Block.Builder lower(Block.Builder b, OpTransformer opT);

        /**
         * Returns a composed code transformer that composes with an operation transformer function adapted to lower
         * operations.
         * <p>
         * This method behaves as if it returns the result of the following expression:
         * {@snippet lang = java:
         * OpTransformer.andThen(before, lowering(before, f));
         *}
         *
         * @param before the code transformer to apply before
         * @param f the operation transformer function to apply after
         * @return the composed code transformer
         */
        static OpTransformer andThenLowering(OpTransformer before, BiFunction<Block.Builder, Op, Block.Builder> f) {
            return OpTransformer.andThen(before, lowering(before, f));
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
        static BiFunction<Block.Builder, Op, Block.Builder> lowering(OpTransformer before, BiFunction<Block.Builder, Op, Block.Builder> f) {
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
         * If assigned to an operation result, it indicates the operation is sealed
        */
        private static final Result SEALED_RESULT = new Result();

        final Op op;

        private Result() {
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
        public Set<Value> dependsOn() {
            Set<Value> depends = new LinkedHashSet<>(op.operands());
            if (op instanceof Terminating) {
                op.successors().stream().flatMap(h -> h.arguments().stream()).forEach(depends::add);
            }

            return Collections.unmodifiableSet(depends);
        }

        /**
         * Returns the result's operation.
         *
         * @return the result's operation.
         */
        public Op op() {
            return op;
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
     * Constructs an operation by copying given operation.
     *
     * @param that the operation to copy.
     * @param cc   the copy context.
     * @implSpec The default implementation calls the constructor with the operation's name, result type, and a list
     * values computed, in order, by mapping the operation's operands using the copy context.
     */
    protected Op(Op that, CopyContext cc) {
        this(cc.getValues(that.operands));
        this.location = that.location;
    }

    /**
     * Copies this operation and its bodies, if any.
     * <p>
     * The returned operation is structurally identical to this operation and is otherwise independent
     * of the values declared and used.
     *
     * @return the copied operation.
     */
    public Op copy() {
        return transform(CopyContext.create(), OpTransformer.COPYING_TRANSFORMER);
    }

    /**
     * Copies this operation and its bodies, if any.
     * <p>
     * The returned operation is structurally identical to this operation and is otherwise independent
     * of the values declared and used.
     *
     * @param cc the copy context.
     * @return the copied operation.
     */
    public Op copy(CopyContext cc) {
        return transform(cc, OpTransformer.COPYING_TRANSFORMER);
    }

    /**
     * Copies this operation and transforms its bodies, if any.
     * <p>
     * Bodies are {@link Body#transform(CopyContext, OpTransformer) transformed} with the given copy context and
     * operation transformer.
     *
     * @param cc the copy context.
     * @param ot the operation transformer.
     * @return the transformed operation.
     */
    public abstract Op transform(CopyContext cc, OpTransformer ot);

    /**
     * Constructs an operation with a name and list of operands.
     *
     * @param operands the list of operands, a copy of the list is performed if required.
     */
    protected Op(List<? extends Value> operands) {
        this.operands = List.copyOf(operands);
    }

    /**
     * Sets the originating source location of this operation, if unbound.
     *
     * @param l the location, the {@link Location#NO_LOCATION} value indicates the location is not specified.
     * @throws IllegalStateException if this operation is bound or sealed
     */
    public final void setLocation(Location l) {
        // @@@ Fail if location != null?
        if (isSealed() || (result != null && result.block.isBound())) {
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
     * Returns this operation's parent block, otherwise {@code null} if the operation is unbound or sealed.
     *
     * @return operation's parent block, or {@code null} if the operation is unbound or sealed.
     */
    @Override
    public final Block parent() {
        if (isSealed() || result == null) {
            return null;
        }

        if (!result.block.isBound()) {
            throw new IllegalStateException("Parent block is partially constructed");
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
     */
    public List<Body> bodies() {
        return List.of();
    }

    /**
     * Returns the operation's result, otherwise {@code null} if the operation is unbound or sealed.
     *
     * @return the operation's result, or {@code null} if unbound or sealed.
     */
    public final Result result() {
        return result == Result.SEALED_RESULT ? null : result;
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
     * {@return the operation's result type}
     */
    public abstract TypeElement resultType();

    /**
     * Returns the operation's function type.
     * <p>
     * The function type's result type is the operation's result type and the function type's parameter types are the
     * operation's operand types, in order.
     *
     * @return the function type
     */
    public final FunctionType opType() {
        List<TypeElement> operandTypes = operands.stream().map(Value::type).toList();
        return CoreType.functionType(resultType(), operandTypes);
    }

    /**
     * Externalizes this operation's name as a string.
     * @implSpec this implementation returns the result of the expression {@code this.getClass().getName()}.
     * @return the operation name
     */
    public String opName() {
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
     * Computes values captured by this operation. A captured value is a value that dominates
     * this operation and is used by a descendant operation.
     * <p>
     * The order of the captured values is first use encountered in depth
     * first search of this operation's descendant operations.
     *
     * @return the list of captured values, modifiable
     * @see Body#capturedValues()
     */
    public final List<Value> capturedValues() {
        Set<Value> cvs = new LinkedHashSet<>();

        capturedValues(cvs, new ArrayDeque<>(), this);
        return new ArrayList<>(cvs);
    }

    static void capturedValues(Set<Value> capturedValues, Deque<Body> bodyStack, Op op) {
        for (Body childBody : op.bodies()) {
            Body.capturedValues(capturedValues, bodyStack, childBody);
        }
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
     * Returns the quoted code model of the given quotable reference, if present.
     *
     * @param q the quotable reference.
     * @return the quoted code model or an empty optional if the
     *         quoted code model is unavailable.
     * @apiNote If the quotable reference is a proxy instance, then the
     *          quoted code model is unavailable and this method
     *          returns an empty optional.
     * @since 99
     */
    public static Optional<Quoted> ofQuotable(Quotable q) {
        Object oq = q;
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

        Quoted quoted;
        try {
            quoted = (Quoted) method.invoke(oq);
        } catch (InvocationTargetException | IllegalAccessException e) {
            throw new RuntimeException(e);
        }
        return Optional.of(quoted);
    }

    /**
     * Returns the code model of the given method's body, if present.
     *
     * @param method the method.
     * @return the code model of the method body.
     * @since 99
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
     * @implSpec The default implementation unconditionally returns an empty optional.
     * @param e the executable element.
     * @return the code model of the provided executable element (if any).
     * @since 99
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

    /**
     * Seals this operation. After this operation is sealed its {@link #result result} and {@link #parent parent} are guaranteed to always be {@code null}.
     * <p>
     * If a sealed operation is {@link Block.Builder#op appended} to a {@link Block.Builder} then it is
     * treated as if the operation is bound, and therefore the sealed operation will be transformed.
     * <p>
     * Sealing is idempotent if the operation is already sealed.
     *
     * @throws IllegalStateException if this operation is bound.
     */
    public void seal() {
        if (result == Result.SEALED_RESULT) {
            return;
        }
        if (result != null) {
            throw new IllegalStateException("Operation cannot be sealed since it bound to a parent block");
        }
        result = Result.SEALED_RESULT;
    }

    /**
     * Returns {@code true} if this operation is sealed.
     * @return {@code true} if this operation is sealed.
     * @see #seal()
    * */
    public boolean isSealed() {
        return result == Result.SEALED_RESULT;
    }
}

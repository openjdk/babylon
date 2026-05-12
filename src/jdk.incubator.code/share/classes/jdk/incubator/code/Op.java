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
 * An operation modeling a unit of program behavior.
 * <p>
 * An operation uses zero or more values, exposed as a sequence of {@link #operands()}. A
 * {@link Op.Terminating terminating} operation may have block references, exposed as a sequence of
 * {@link #successors()}. An operation has zero or more bodies, exposed as a sequence of {@link #bodies()}.
 *
 * <h2>Operation construction</h2>
 * <p>
 * Constructing an operation creates an <i>unplaced</i> operation. An unplaced operation is not yet part of a code
 * model. The operation's operands, successors, bodies, and operation-specific state are fixed when construction
 * completes.
 * <p>
 * An operation can only be constructed with operands whose declaring block is being built, otherwise construction fails
 * with an exception.
 *
 * <h2>Operation building</h2>
 * <p>
 * Building an operation places an unplaced operation in a code model in one of two ways:
 * <ol>
 * <li>
 * the operation is <i>placed</i> in a block, which becomes its parent block, by using a block builder to
 * {@link Block.Builder#op(Op) append} the operation to the block. The placed operation has a permanently
 * non-{@code null} {@link #result result} that can be used as an operand of subsequently constructed operations. The
 * block being built is not <a href="Body.Builder.html#body-building-observability">observable</a> through this
 * operation and any attempt to access the block throws {@link IllegalStateException}.
 * <li>
 * the operation is <i>placed</i> as the {@link #isRoot() <i>root operation</i>} of a code model by using
 * {@link #buildAsRoot()}. The root operation's {@link #result result} and {@link #parent parent} are always
 * {@code null}.
 * </ol>
 * <p>
 * Building finishes when the parent body builder of the block in which the operation was placed
 * <a href="Body.Builder.html#body-building-finishing">finishes</a>, after which the block becomes observable, or when
 * the operation is placed as the root of a code model.
 * <p>
 * The {@link #location} may be {@link #setLocation set} while the operation is unplaced or placed in a block whose
 * parent body builder has not finished.
 * <p>
 * An unplaced operation, or an operation placed in a block whose parent body builder has not
 * <a href="Body.Builder.html#body-building-finishing">finished</a>, is not thread-safe.
 *
 * <h2>Operation implementation requirements</h2>
 * <p>
 * A concrete operation class must satisfy the following requirements:
 * <ul>
 * <li>
 * implement {@link #resultType()} to return the result type of operation instances;
 * <li>
 * implement {@link #transform(CodeContext, CodeTransformer)} to return a newly constructed, unplaced copy whose
 * concrete class is the same as the operation's concrete class;
 * <li>
 * call an appropriate {@code Op} superclass constructor from each concrete operation constructor. Constructors
 * for new operations pass the operation's operands to {@link #Op(List)}. Constructors for transformed copies can
 * pass the input operation and code context to {@link #Op(Op, CodeContext)};
 * <li>
 * override {@link #bodies()} if instances may have bodies. If the operation class implements {@link Op.Nested}, then
 * {@code bodies()} must return one or more bodies;
 * <li>
 * override {@link #successors()} if instances may have successors. If the operation class implements
 * {@link Op.BlockTerminating}, then {@code successors()} must return one or more successors;
 * <li>
 * copy mutable constructor arguments that define successors, bodies, and operation-specific state, ensuring
 * they are all fixed when construction completes; and
 * <li>
 * return unmodifiable views or immutable values from accessors that expose successors, bodies, and operation-specific
 * state.
 * </ul>
 * <p>
 * A concrete operation class may additionally:
 * <ul>
 * <li>
 * override {@link #externalizeOpName()} and {@link #externalize()} to define an external form;
 * <li>
 * implement {@link Op.Lowerable} to define a lowering; and
 * <li>
 * provide operation-specific accessors for operation-specific state.
 * </ul>
 *
 * @apiNote
 * An operation might model the {@link JavaOp.AddOp addition} of two integers, or a method
 * {@link JavaOp.InvokeOp invocation} expression. Alternatively an operation may model something more complex like
 * {@link jdk.incubator.code.dialect.core.CoreOp.FuncOp method} declarations, {@link JavaOp.LambdaOp lambda}
 * expressions, or {@link JavaOp.TryOp try} statements. In such cases an operation will contain one or more bodies
 * modeling the nested structure.
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
         * {@return the non-empty list of bodies of this nested operation.}
         */
        List<Body> bodies();
    }

    /**
     * An operation characteristic indicating the operation represents a loop
     */
    public interface Loop extends Nested {
        /**
         * {@return the body of this loop operation.}
         * <p>
         * The returned body is one of this operation's {@link #bodies() bodies}.
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
         * {@return the body of this invokable operation.}
         * <p>
         * The returned body is one of this operation's {@linkplain #bodies() bodies}.
         */
        Body body();

        /**
         * {@return the invokable operation's signature, represented as a function type.}
         * @implSpec
         * The default implementation returns the signature of the invokable operation's body.
         */
        default FunctionType invokableSignature() {
            return body().bodySignature();
        }

        /**
         * {@return the entry block parameters of this operation's body}
         * @implSpec
         * The default implementation returns the entry block's parameters of the invokable operation's body.
         */
        default List<Block.Parameter> parameters() {
            return body().entryBlock().parameters();
        }

        /**
         * Computes values captured by this invokable operation's body.
         * @implSpec
         * The default implementation returns an empty unmodifiable list.
         *
         * @return the captured values.
         * @see Body#capturedValues()
         */
        default List<Value> capturedValues() {
            return List.of();
        }
    }

    /**
     * An operation characteristic indicating the operation can lower itself by replacing itself with blocks and
     * operations that represent the same behavior.
     */
    // @@@ Hide this abstraction within JavaOp?
    public interface Lowerable {

        /**
         * Lowers this operation into the given block builder.
         * <p>
         * A lowering implementation emits the replacement blocks and operations into the given builder, and returns
         * the block builder to use for subsequent operations in an enclosing transformation.
         * <p>
         * If this operation lowers one of its bodies, it should transform that body with a lowering code transformer
         * produced by {@link #loweringTransformer(BiFunction, BiFunction)}. This ensures that lowerable operations
         * encountered in that body are lowered recursively.
         * The {@code inherited} transformer is the operation transformer inherited from an enclosing lowering, if any.
         * A lowering implementation may pass it directly to {@code loweringTransformer}, or compose it with another
         * transformer and pass the composed transformer. The transformer passed to {@code loweringTransformer} is then
         * supplied as the inherited transformer when that lowering code transformer recursively lowers lowerable
         * operations.
         *
         * @param b the block builder into which this operation is lowered
         * @param inherited the inherited operation transformer, may be {@code null}
         * @return the block builder to use for subsequent building
         */
        Block.Builder lower(Block.Builder b, BiFunction<Block.Builder, Op, Block.Builder> inherited);

        /**
         * Returns a lowering code transformer that partially composes the given operation transformers and, if
         * required, lowers lowerable operations and appends non-lowerable operations.
         * <p>
         * The returned code transformer accepts an operation by first applying the partial composition of
         * {@code current} with {@code inherited} in the first argument of {@code current}, as if by the following:
         * {@snippet lang = "java":
         * Block.Builder composedBlock = inherited == null
         *         ? block
         *         : inherited.apply(block, op);
         * Block.Builder currentBlock = current.apply(composedBlock, op);
         * }
         * The returned continuation builder is then selected as if by the following:
         * {@snippet lang = "java":
         * if (currentBlock != null) {
         *     return currentBlock;
         * } else if (op instanceof Op.Lowerable lop) {
         *     return lop.lower(composedBlock, inherited);
         * } else {
         *     composedBlock.op(op);
         *     return composedBlock;
         * }
         * }
         *
         * @param inherited the inherited operation transformer, may be {@code null}
         * @param current the current operation transformer
         * @return the lowering code transformer
         */
        static CodeTransformer loweringTransformer(BiFunction<Block.Builder, Op, Block.Builder> inherited,
                                                   BiFunction<Block.Builder, Op, Block.Builder> current) {
            Objects.requireNonNull(current);
            return (block, op) -> {
                if (inherited != null) {
                    block = inherited.apply(block, op);
                }
                Block.Builder currentBlock = current.apply(block, op);
                if (currentBlock != null) {
                    return currentBlock;
                } else if (op instanceof Op.Lowerable lop) {
                    return lop.lower(block, inherited);
                } else {
                    block.op(op);
                    return block;
                }
            };
        }
    }

    /**
     * An operation characteristic indicating the operation is a terminating operation
     * that occurs as the last operation in a block.
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
        /**
         * {@return the non-empty list of successors of this block terminating operation.}
         */
        List<Block.Reference> successors();
    }

    /**
     * A value that is the result of an operation.
     */
    public static final class Result extends Value {

        /**
         * If assigned to an operation's result field, indicates the operation is a root operation.
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

    // Set when op is placed in a block or as a root operation, otherwise null when unplaced
    // @@@ stable value?
    Result result;

    // null if not specified
    // @@@ stable value?
    Location location;

    final List<Value> operands;

    /**
     * Constructs an operation with operands mapped from, and location copied from, the given operation.
     * <p>
     * The constructed operation's operands are the values computed, in order, by mapping the given operation's operands
     * using the given code context. The new operation's location is the given operation's location, if any.
     *
     * @param that the given operation
     * @param cc   the code context
     */
    protected Op(Op that, CodeContext cc) {
        List<Value> outputOperands = cc.getValues(that.operands);
        // Values should be guaranteed to connect to blocks being built since
        // the context only allows such mappings, assert for clarity
        assert outputOperands.stream().noneMatch(Value::isBuilt);
        this.operands = List.copyOf(outputOperands);
        this.location = that.location;
    }

    /**
     * Transforms this operation, copying the operation and transforming any of its bodies.
     * <p>
     * This method returns a newly constructed, unplaced copy of this operation. The returned operation's concrete
     * class is the same as this operation's concrete class.
     * <p>
     * The returned operation copies this operation's operands, successors, and any operation-specific state. Operands
     * are copied by mapping this operation's operands, in order, with the given code context. Successors are copied as
     * specified by {@link CodeContext#getReferenceOrCreate(Block.Reference)}. Operation-specific state is copied as
     * appropriate for the operation, preserving operation-specific behavior.
     * <p>
     * Bodies are {@link Body#transform(CodeContext, CodeTransformer) transformed} with the given code context and code
     * transformer, and built with the returned operation as their parent.
     *
     * @apiNote
     * To copy an operation use the {@link CodeTransformer#COPYING_TRANSFORMER copying transformer}.
     *
     * @param cc the code context
     * @param ct the code transformer
     * @return the transformed operation
     * @see CodeTransformer#COPYING_TRANSFORMER
     */
    public abstract Op transform(CodeContext cc, CodeTransformer ct);

    /**
     * Constructs an operation with a list of operands.
     *
     * @param operands the list of operands, a copy of the list is performed if required.
     * @throws IllegalArgumentException if an operand's declaring block is built.
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
     * Returns this operation's parent block, otherwise {@code null} if this operation is unplaced or a
     * root operation.
     * <p>
     * The operation's parent block is the same as the operation result's {@link Value#declaringBlock declaring block}.
     *
     * @return operation's parent block, or {@code null} if this operation is unplaced or a root operation.
     * @throws IllegalStateException if this operation is placed in a block that is
     * <a href="Body.Builder.html#body-building-observability">unobservable</a>.
     * @see Value#declaringBlock()
     */
    @Override
    public final Block parent() {
        if (isRoot() || result == null) {
            return null;
        }

        if (!result.block.isBuilt()) {
            throw new IllegalStateException("Parent block is unobservable");
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
    public abstract CodeType resultType();


    /**
     * {@return the operation's result, or {@code null} if this operation is unplaced or a
     * root operation.}
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
     * Returns the operation's signature, represented as a function type.
     * <p>
     * The signature's return type is the operation's result type and its parameter types are the
     * operation's operand types, in order.
     *
     * @return the operation's signature
     */
    public final FunctionType opSignature() {
        List<CodeType> operandTypes = operands.stream().map(Value::type).toList();
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
     * Builds this operation, placing it as the root operation of a code model. After this operation is built its
     * {@link #result result} and {@link #parent parent} will always be {@code null}.
     * <p>
     * This method is idempotent.
     *
     * @throws IllegalStateException if this operation is placed in a block.
     * @see #isRoot()
     */
    public final void buildAsRoot() {
        if (result == Result.ROOT_RESULT) {
            return;
        }
        if (result != null) {
            throw new IllegalStateException("Operation is placed in a block");
        }
        result = Result.ROOT_RESULT;
    }

    /**
     * {@return {@code true} if this operation is a root operation.}
     * @see #buildAsRoot()
     * @see #isAttached()
     * */
    public final boolean isRoot() {
        return result == Result.ROOT_RESULT;
    }

    /**
     * {@return {@code true} if this operation is placed in a block.}
     * @see #buildAsRoot()
     * @see #isRoot()
     * */
    public final boolean isAttached() {
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
     * Returns the code model text for this operation.
     * <p>
     * The format of codel model text is unspecified.
     *
     * @return the code model text for this operation.
     * @apiNote Code model text is designed to be human-readable and is intended for debugging, testing,
     * and comprehension.
     * @see OpWriter#toText(Op, OpWriter.Option...)
     */
    public final String toText() {
        return OpWriter.toText(this);
    }


    /**
     * Returns a quoted instance containing the code model of a reflectable lambda expression or method reference.
     * <p>
     * The quoted instance also contains a mapping from {@link Value values} in the code model that model final, or
     * effectively final, variables used but not declared in the lambda expression to their corresponding run time
     * values. Such run time values are commonly referred to as captured values.
     * <p>
     * Repeated invocations of this method will return a quoted instance containing the same instance of the code model.
     * Therefore, code elements (and more generally code items) contained within the code model can be reliably compared
     * using object identity.
     *
     * @param fiInstance a functional interface instance that is the result of a reflectable lambda expression or
     *                   method reference.
     * @return the quoted instance containing the code model, or an empty optional if the functional interface instance
     * is not the result of a reflectable lambda expression or method reference.
     * @throws UnsupportedOperationException if the Java version used at compile time to generate and store the code
     * model is not the same as the Java version used at runtime to load the code model.
     * @apiNote if the functional interface instance is a proxy instance, then the quoted code model is unavailable and
     * this method returns an empty optional.
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
     * <p>
     * Repeated invocations of this method will return the same instance of the code model. Therefore,
     * code elements (and more generally code items) contained within the code model can be reliably compared using
     * object identity.
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
     * @param processingEnvironment the annotation processing environment
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

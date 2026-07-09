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
import jdk.incubator.code.dialect.core.CoreOp.FuncOp;
import jdk.incubator.code.dialect.core.CoreType;
import jdk.incubator.code.dialect.core.FunctionType;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.MethodRef;
import jdk.incubator.code.extern.OpWriter;
import jdk.incubator.code.internal.ReflectMethods;
import jdk.internal.access.SharedSecrets;

import javax.annotation.processing.ProcessingEnvironment;
import javax.lang.model.element.ExecutableElement;
import javax.lang.model.element.Modifier;
import java.lang.reflect.Method;
import java.lang.reflect.Proxy;
import java.util.*;
import java.util.function.BiFunction;

/**
 * The abstract implementation of an operation. All concrete operations extend this class.
 *
 * <h2>Operation implementation requirements</h2>
 * <p>
 * A concrete operation class must satisfy the following requirements:
 * <ul>
 * <li>
 * implement {@link #resultType()} to return the result type of operation instances;
 * <li>
 * implement {@link #transform(CodeContext, CodeTransformer)} to return a newly constructed, unplaced copy whose
 * concrete class is the concrete operation class;
 * <li>
 * call an appropriate {@code AbstractOp} superclass constructor from each concrete operation constructor. Constructors
 * for new operations pass the operation's operands to {@link #AbstractOp(List)}. Constructors for transformed copies
 * can pass the input operation and code context to {@link #AbstractOp(AbstractOp, CodeContext)};
 * <li>
 * override {@link #bodies()} if instances may have bodies. If the operation class implements {@link Op.Nested}, then
 * {@code bodies()} must return one or more bodies;
 * <li>
 * override {@link #successors()} if instances may have successors. If the operation class implements
 * {@link Op.BlockTerminating}, then {@code successors()} must return one or more successors;
 * <li>
 * copy mutable constructor arguments that define successors, bodies, and operation-specific state, ensuring they are
 * all fixed when construction completes; and
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
 */
public non-sealed abstract class AbstractOp implements Op {

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
    protected AbstractOp(AbstractOp that, CodeContext cc) {
        List<Value> outputOperands = cc.getValues(that.operands);
        // Values should be guaranteed to connect to blocks being built since
        // the context only allows such mappings, assert for clarity
        assert outputOperands.stream().noneMatch(Value::isBuilt);
        this.operands = List.copyOf(outputOperands);
        this.location = that.location;
    }

    /**
     * Constructs an operation with a list of operands.
     *
     * @param operands the list of operands, a copy of the list is performed if required.
     * @throws IllegalArgumentException if an operand's declaring block is built.
     */
    protected AbstractOp(List<? extends Value> operands) {
        for (Value operand : operands) {
            if (operand.isBuilt()) {
                throw new IllegalArgumentException("Operand's declaring block is built: " + operand);
            }
        }
        this.operands = List.copyOf(operands);
    }

    @Override
    public final void setLocation(Location l) {
        // @@@ Fail if location != null?
        if (isRoot() || (result != null && result.block.isBuilt())) {
            throw new IllegalStateException("Built operation");
        }

        location = l;
    }

    @Override
    public final Location location() {
        return location;
    }

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
     * {@inheritDoc}
     * @implSpec this implementation returns an unmodifiable empty list.
     */
    @Override
    public List<Body> bodies() {
        return List.of();
    }

    @Override
    public final Result result() {
        return result == Result.ROOT_RESULT ? null : result;
    }

    @Override
    public final List<Value> operands() {
        return operands;
    }

    /**
     * {@inheritDoc}
     * @implSpec this implementation returns an unmodifiable empty list.
     */
    @Override
    public List<Block.Reference> successors() {
        return List.of();
    }

    @Override
    public final FunctionType opSignature() {
        List<CodeType> operandTypes = operands.stream().map(Value::type).toList();
        return CoreType.functionType(resultType(), operandTypes);
    }

    @Override
    public final List<Value> capturedValues() {
        Set<Value> cvs = new LinkedHashSet<>();

        Deque<Body> bodyStack = new ArrayDeque<>();
        for (Body childBody : bodies()) {
            Body.capturedValues(cvs, bodyStack, childBody);
        }
        return new ArrayList<>(cvs);
    }

    @Override
    public final void buildAsRoot() {
        if (result == Result.ROOT_RESULT) {
            return;
        }
        if (!bodies().stream().allMatch(Body::isIsolated)) {
            throw new IllegalStateException("One of the operation bodies is not isolated");
        }
        if (!operands().isEmpty()) {
            throw new IllegalStateException("Operation has operands");
        }
        if (!successors().isEmpty()) {
            throw new IllegalStateException("Operation has successors");
        }
        if (result != null) {
            throw new IllegalStateException("Operation is placed in a block");
        }
        result = Result.ROOT_RESULT;
    }

    @Override
    public final boolean isRoot() {
        return result == Result.ROOT_RESULT;
    }

    @Override
    public final boolean isPlacedInBlock() {
        return !isRoot() && result != null;
    }

    /**
     * {@inheritDoc}
     * @implSpec this implementation returns the result of the expression {@code this.getClass().getName()}.
     */
    @Override
    public String externalizeOpName() {
        return this.getClass().getName();
    }

    /**
     * {@inheritDoc}
     * @implSpec this implementation returns an unmodifiable empty map.
     */
    @Override
    public Map<String, Object> externalize() {
        return Map.of();
    }

    @Override
    public final String toText() {
        return OpWriter.toText(this);
    }
}

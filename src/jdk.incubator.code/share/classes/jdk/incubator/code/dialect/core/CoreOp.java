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

package jdk.incubator.code.dialect.core;

import jdk.incubator.code.*;
import jdk.incubator.code.dialect.java.*;
import jdk.incubator.code.dialect.java.JavaOp.InvokeOp;
import jdk.incubator.code.dialect.java.JavaOp.LambdaOp;
import jdk.incubator.code.extern.ExternalizedOp;
import jdk.incubator.code.extern.OpFactory;
import jdk.incubator.code.internal.OpDeclaration;

import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import java.util.*;
import java.util.function.Consumer;
import java.util.function.Function;

/**
 * The top-level operation class for core operations.
 * <p>
 * Core operations model the foundational, language-agnostic structure of code, such as functions, modules,
 * variables, tuples, constants, and control flow. Core operations may appear on their own or together with
 * operations expressed in other dialects.
 */
public sealed abstract class CoreOp extends Op {

    CoreOp(Op that, CodeContext cc) {
        super(that, cc);
    }

    CoreOp(List<? extends Value> operands) {
        super(operands);
    }

    @Override
    public String externalizeOpName() {
        OpDeclaration opDecl = this.getClass().getDeclaredAnnotation(OpDeclaration.class);
        assert opDecl != null : this.getClass().getName();
        return opDecl.value();
    }

    /**
     * The function operation, that can model a named function.
     * <p>
     * In code models derived from Java source, function operations can model Java method declarations.
     * <p>
     * A function operation has a {@linkplain #funcName() function name}, which should correspond to an entry in the
     * ancestor module's {@linkplain ModuleOp#functionTable() symbol table}, if any.
     * <p>
     * Function operations feature one body, the {@linkplain #body() function body}. The body accepts the function
     * parameters and yields the function result. That is, the type of the function body corresponds to the signature
     * of the function modeled by the function operation.
     * <p>
     * The result type of a function operation is {@link JavaType#VOID}.
     *
     * @jls 8.4 Method Declarations
     */
    @OpDeclaration(FuncOp.NAME)
    public static final class FuncOp extends CoreOp
            implements Op.Invokable, Op.Isolated, Op.Lowerable {

        /**
         * A builder for constructing a function operation.
         */
        public static class Builder {
            final Body.Builder ancestorBody;
            final MethodRef mref;

            Builder(Body.Builder ancestorBody, MethodRef mref) {
                this.ancestorBody = ancestorBody;
                this.mref = mref;
            }

            Builder(Body.Builder ancestorBody, String funcName, FunctionType functionType) {
                this.ancestorBody = ancestorBody;
                this.mref = MethodRef.method(NO_REF_TYPE, funcName, functionType);
            }

            /**
             * Completes the function operation by adding the function body.
             *
             * @param c a consumer that populates the function body
             * @return the completed function operation
             */
            public FuncOp body(Consumer<Block.Builder> c) {
                Body.Builder body = Body.Builder.of(ancestorBody, mref.type());
                c.accept(body.entryBlock());
                return new FuncOp(mref, body);
            }
        }

        static final String NAME = "func";

        /**
         * The externalized attribute modelling the function name
         */
        static final String ATTRIBUTE_FUNC_NAME = NAME + ".name";
        static final String ATTRIBUTE_FUNC_MREF = NAME + ".mref";
        private static final JavaType NO_REF_TYPE = null;

        final Body body;
        final MethodRef mref;

        FuncOp(ExternalizedOp def) {
            if (!def.operands().isEmpty()) {
                throw new IllegalStateException("Bad op " + def.name());
            }

            MethodRef mref = def.extractAttributeValue(ATTRIBUTE_FUNC_MREF, false,
                    v -> switch (v) {
                        case MethodRef r -> r;
                        case null, default -> {
                            String funcName = def.extractAttributeValue(ATTRIBUTE_FUNC_NAME, true,
                                    u -> switch (u) {
                                        case String s -> s;
                                        case null, default -> throw new UnsupportedOperationException("Unsupported func name value:" + u);
                                    });
                            yield MethodRef.method(NO_REF_TYPE, funcName, def.bodyDefinitions().get(0).bodyType());
                        }
                    });

            this(mref, def.bodyDefinitions().get(0));
        }

        FuncOp(FuncOp that, CodeContext cc, CodeTransformer ot) {
            super(that, cc);

            this.body = that.body.transform(cc, ot).build(this);
            this.mref = that.mref;
        }

        FuncOp(FuncOp that, String funcName, CodeContext cc, CodeTransformer ot) {
            super(that, cc);

            this.body = that.body.transform(cc, ot).build(this);
            this.mref = MethodRef.method(that.mref.refType(), funcName, that.mref.type());
        }

        @Override
        public FuncOp transform(CodeContext cc, CodeTransformer ot) {
            return new FuncOp(this, cc, ot);
        }

        /**
         * Transforms a function operation using the given code transformer and a new context.
         *
         * @param ot code transformer to apply to this function operation
         * @return the transformed function operation
         */
        public FuncOp transform(CodeTransformer ot) {
            return new FuncOp(this, CodeContext.create(), ot);
        }

        /**
         * Transforms a function operation using the given function name, code transformer and a new context.
         *
         * @param funcName the new function name
         * @param ot code transformer to apply to this function operation
         * @return the transformed function operation
         */
        public FuncOp transform(String funcName, CodeTransformer ot) {
            return new FuncOp(this, funcName, CodeContext.create(), ot);
        }

        FuncOp(MethodRef mref, Body.Builder bodyBuilder) {
            super(List.of());

            this.body = bodyBuilder.build(this);
            this.mref = mref;
        }

        FuncOp(String funcName, Body.Builder bodyBuilder) {
            super(List.of());

            this.body = bodyBuilder.build(this);
            this.mref = MethodRef.method(NO_REF_TYPE, funcName, bodyBuilder.bodyType());
        }

        @Override
        public List<Body> bodies() {
            return List.of(body);
        }

        @Override
        public Map<String, Object> externalize() {
            Map<String, Object> m = new HashMap<>();
            if (mref.refType() == NO_REF_TYPE) {
                m.put("", mref.name());
            } else {
                m.put(ATTRIBUTE_FUNC_MREF, mref);
            }
            return m;
        }

        @Override
        public FunctionType invokableType() {
            return body.bodyType();
        }

        /**
         * {@return the function name}
         */
        public String funcName() {
            return mref.name();
        }

        @Override
        public Body body() {
            return body;
        }

        @Override
        public Block.Builder lower(Block.Builder b, CodeTransformer _ignore) {
            // Isolate body with respect to ancestor transformations
            b.rebind(b.context(), CodeTransformer.LOWERING_TRANSFORMER).op(this);
            return b;
        }

        @Override
        public TypeElement resultType() {
            return JavaType.VOID;
        }

        public Optional<MethodRef> mref() {
            return mref.refType() == NO_REF_TYPE ? Optional.empty() : Optional.of(mref);
        }
    }

    /**
     * The function call operation, that can model invocation of a function operation declared in an ancestor module
     * operation.
     * <p>
     * A function call operation accepts zero or more operands, corresponding to the arguments passed to the invoked
     * function operation. The invoked function is identified by its {@linkplain #funcName() function name}, which
     * should correspond to an entry in the ancestor module's {@linkplain ModuleOp#functionTable() symbol table}.
     * <p>
     * The result type of a function call operation is the return type of the invoked function operation.
     */
    // @@@ stack effects equivalent to the call operation as if the function were a Java method?
    @OpDeclaration(FuncCallOp.NAME)
    public static final class FuncCallOp extends CoreOp {
        static final String NAME = "func.call";

        /**
         * The externalized attribute modelling the name of the invoked function
         */
        static final String ATTRIBUTE_FUNC_NAME = NAME + ".name";

        final String funcName;
        final TypeElement resultType;

        FuncCallOp(ExternalizedOp def) {
            String funcName = def.extractAttributeValue(ATTRIBUTE_FUNC_NAME, true,
                    v -> switch (v) {
                        case String s -> s;
                        case null, default -> throw new UnsupportedOperationException("Unsupported func name value:" + v);
                    });

            this(funcName, def.resultType(), def.operands());
        }

        FuncCallOp(FuncCallOp that, CodeContext cc) {
            super(that, cc);

            this.funcName = that.funcName;
            this.resultType = that.resultType;
        }

        @Override
        public FuncCallOp transform(CodeContext cc, CodeTransformer ot) {
            return new FuncCallOp(this, cc);
        }

        FuncCallOp(String funcName, TypeElement resultType, List<Value> args) {
            super(args);

            this.funcName = funcName;
            this.resultType = resultType;
        }

        @Override
        public Map<String, Object> externalize() {
            return Map.of("", funcName);
        }

        /**
         * {@return the function name}
         */
        public String funcName() {
            return funcName;
        }

        @Override
        public TypeElement resultType() {
            return resultType;
        }
    }

    /**
     * The module operation, that can model a collection of function operations.
     * <p>
     * A module operation maintains a symbol table from function name to function operation, referred to as the
     * module operation's <em>module symbol table</em>.
     * <p>
     * Module operations feature one body. The body contains the function operations in the module symbol table and
     * terminates without yielding a value.
     * <p>
     * The result type of a module operation is {@link JavaType#VOID}.
     */
    @OpDeclaration(ModuleOp.NAME)
    public static final class ModuleOp extends CoreOp
            implements Op.Isolated, Op.Lowerable {

        static final String NAME = "module";

        final SequencedMap<String, FuncOp> table;
        final Body body;

        ModuleOp(ExternalizedOp def) {
            if (!def.operands().isEmpty()) {
                throw new IllegalStateException("Bad op " + def.name());
            }

            this(def.bodyDefinitions().get(0));
        }

        ModuleOp(ModuleOp that, CodeContext cc, CodeTransformer ot) {
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
        public ModuleOp transform(CodeContext cc, CodeTransformer ot) {
            return new ModuleOp(this, cc, ot);
        }

        /**
         * Transforms a module operation using the given code transformer and a new context.
         *
         * @param ot code transformer to apply to the module operation
         * @return the transformed module operation
         */
        public ModuleOp transform(CodeTransformer ot) {
            return new ModuleOp(this, CodeContext.create(), ot);
        }

        ModuleOp(Body.Builder bodyBuilder) {
            super(List.of());

            this.body = bodyBuilder.build(this);
            this.table = createTable(body);
        }

        ModuleOp(List<FuncOp> functions) {
            Body.Builder bodyC = Body.Builder.of(null, CoreType.FUNCTION_TYPE_VOID);
            Block.Builder entryBlock = bodyC.entryBlock();
            for (FuncOp f : functions) {
                entryBlock.op(f);
            }
            entryBlock.op(CoreOp.unreachable());

            this(bodyC);
        }

        @Override
        public List<Body> bodies() {
            return List.of(body);
        }

        /**
         * {@return the module symbol table, mapping function name to function operation}
         */
        public SequencedMap<String, FuncOp> functionTable() {
            return table;
        }

        @Override
        public TypeElement resultType() {
            return JavaType.VOID;
        }

        @Override
        public Block.Builder lower(Block.Builder b, CodeTransformer _ignore) {
            b.rebind(b.context(), CodeTransformer.LOWERING_TRANSFORMER).op(this);
            return b;
        }

        static CoreOp.FuncOp invokeToFuncOp(JavaOp.InvokeOp invokeOp, MethodHandles.Lookup l) {
            try {
        Method method = invokeOp.invokeReference().resolveToMethod(l);
                return Op.ofMethod(method).orElse(null);
            } catch (ReflectiveOperationException e) {
                throw new IllegalStateException("Could not resolve invokeOp to method");
            }
        }

        /**
         * Creates a module operation from a root function operation, collecting all reachable function operations.
         * The symbol table of the returned module operation contains the root function operation, followed by
         * reachable function operations in encounter order.
         * <p>
         * More precisely, for a function operation to be included in the symbol table of the returned module
         * operation it has to be:
         * <ul>
         *     <li>the root function operation</li>
         *     <li>a function operation referenced by an
         *     {@linkplain jdk.incubator.code.dialect.java.JavaOp.InvokeOp invoke operation}, where the reference
         *     occurs in the body of a function already included and the {@linkplain InvokeOp#invokeReference() method reference}
         *     associated with the invoke operation resolves to a function operation.</li>
         * </ul>
         *
         * @param root the root function operation
         * @param l    the lookup used to resolve {@linkplain MethodRef method references}
         * @return a module operation containing the root and reachable function operations
         */
        public static CoreOp.ModuleOp ofFuncOp(CoreOp.FuncOp root, MethodHandles.Lookup l) {
            SequencedSet<FuncOp> visited = new LinkedHashSet<>();
            Map<FuncOp, String> funcNames = new HashMap<>(); // holds the original funcOps and their new names
            Deque<CoreOp.FuncOp> stack = new LinkedList<>(); // holds worklist of og funcOps to process
            SequencedSet<FuncOp> transformed = new LinkedHashSet<>();

            stack.push(root);
            funcNames.put(root, root.funcName() + "_" + funcNames.size());
            while (!stack.isEmpty()) {
                CoreOp.FuncOp cur = stack.pop();

                if (!visited.add(cur)) {
                    continue;
                }

                List<CoreOp.FuncOp> calledFuncs = new ArrayList<>();
                // traversing to convert invokeOps -> funcCallOps and gathering invokeOps to be processed later
                transformed.add(cur.transform(funcNames.get(cur), (blockBuilder, op) -> {
                    if (op instanceof JavaOp.InvokeOp iop) {
                        Method invokeOpCalledMethod = null;
                        try {
        invokeOpCalledMethod = iop.invokeReference().resolveToMethod(l);
                        } catch (ReflectiveOperationException e) {
                            throw new RuntimeException("Could not resolve invokeOp to method");
                        }
                        if (invokeOpCalledMethod instanceof Method m &&
                                Op.ofMethod(m).orElse(null) instanceof CoreOp.FuncOp calledFunc) {
                            calledFuncs.add(calledFunc);
                            funcNames.computeIfAbsent(calledFunc,
                                    f -> f.funcName() + "_" + funcNames.size());
                            Op.Result result = blockBuilder.op(CoreOp.funcCall(
                                    funcNames.get(calledFunc),
                                    calledFunc.invokableType(),
                                    blockBuilder.context().getValues(iop.operands())));
                            blockBuilder.context().mapValue(op.result(), result);
                            return blockBuilder;
                        }
                    }
                    blockBuilder.op(op);
                    return blockBuilder;
                }));

                for (FuncOp f : calledFuncs.reversed()) {
                    if (!stack.contains(f)) stack.push(f);
                }
            }
            return CoreOp.module(transformed.stream().toList());
        }

        /**
         * Creates a module operation from a lambda operation, a method handles lookup, and a name for the root
         * lambda.
         * <p>
         * This is equivalent to:
         * {@snippet :
         * ofFuncOp(root)
         * }
         * where {@code root} is derived as follows:
         * <ul>
         *     <li>if {@code lambdaOp} contains a {@linkplain LambdaOp#directInvocation() direct invocation}, and that
         *     direct invocation can be resolved to a function operation, that operation is {@code root}</li>
         *     <li>otherwise, {@code root} is the function operation obtained using
         *     {@code lambdaOp.toFuncOp(lambdaName)}</li>
         * </ul>
         *
         * @param lambdaOp   the lambda operation
         * @param l          the lookup used to resolve {@linkplain MethodRef method references}
         * @param lambdaName the name to use for the root function operation, or {@code null}
         * @return a module operation containing a (derived) root function operation and reachable function operations
         */
        public static CoreOp.ModuleOp ofLambdaOp(JavaOp.LambdaOp lambdaOp, MethodHandles.Lookup l, String lambdaName) {
            if (lambdaName == null) lambdaName = "";
            CoreOp.FuncOp funcOp = lambdaOp.directInvocation().isPresent() ?
                    invokeToFuncOp(lambdaOp.directInvocation().get(), l) :
                    lambdaOp.toFuncOp(lambdaName);
            return ofFuncOp(funcOp, l);
        }
    }

    /**
     * The quoted operation, that can model creation of a {@link Quoted} instance.
     * <p>
     * The created {@link Quoted} instance describes, as data, the operation being quoted.
     * <p>
     * The operation being quoted may depend on values defined outside that operation. Such values become captured
     * values of the created {@link Quoted} instance.
     * <p>
     * Quoted operations feature one body. The body yields the operation being quoted.
     * <p>
     * The result type of a quoted operation is the parameterized class type {@code Quoted<Op>},
     * {@link #QUOTED_OP_TYPE}.
     */
    @OpDeclaration(QuotedOp.NAME)
    public static final class QuotedOp extends CoreOp
            implements Op.Nested, Op.Lowerable, Op.Pure {
        static final String NAME = "quoted";

        /**
         * The Java type element modeling the parameterized type {@code Quoted<Op>}
         * that is the result type of a quoted operation.
         */
        public static final JavaType QUOTED_OP_TYPE = JavaType.parameterized(
                JavaType.type(Quoted.class), JavaType.type(Op.class));

        final Body quotedBody;

        final Op quotedOp;

        QuotedOp(ExternalizedOp def) {
            this(def.bodyDefinitions().get(0));
        }

        QuotedOp(QuotedOp that, CodeContext cc, CodeTransformer ot) {
            super(that, cc);

            this.quotedBody = that.quotedBody.transform(cc, ot).build(this);
            this.quotedOp = that.quotedOp;
        }

        @Override
        public QuotedOp transform(CodeContext cc, CodeTransformer ot) {
            return new QuotedOp(this, cc, ot);
        }

        QuotedOp(Body.Builder bodyC) {
            super(List.of());

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

        /**
         * {@return the operation being quoted}
         */
        public Op quotedOp() {
            return quotedOp;
        }

        @Override
        public Block.Builder lower(Block.Builder b, CodeTransformer _ignore) {
            // Isolate body with respect to ancestor transformations
            // and copy directly without lowering descendant operations
            b.rebind(b.context(), CodeTransformer.COPYING_TRANSFORMER).op(this);
            return b;
        }

        @Override
        public TypeElement resultType() {
            return QUOTED_OP_TYPE;
        }
    }

    /**
     * The return operation, that can model exit from the body of a function operation or a lambda operation.
     * <p>
     * A return operation is a body-terminating operation that accepts zero or one operand, corresponding to the
     * value returned from the function operation or lambda operation.
     * <p>
     * The result type of a return operation is {@link JavaType#VOID}.
     */
    @OpDeclaration(ReturnOp.NAME)
    public static final class ReturnOp extends CoreOp
            implements Op.BodyTerminating, JavaOp.JavaStatement {
        static final String NAME = "return";

        ReturnOp(ExternalizedOp def) {
            if (def.operands().size() > 1) {
                throw new IllegalArgumentException("Operation must have zero or one operand " + def.name());
            }

            this(def.operands().isEmpty() ? null : def.operands().get(0));
        }

        ReturnOp(ReturnOp that, CodeContext cc) {
            super(that, cc);
        }

        @Override
        public ReturnOp transform(CodeContext cc, CodeTransformer ot) {
            return new ReturnOp(this, cc);
        }

        ReturnOp(Value operand) {
            super(operand == null ? List.of() : List.of(operand));
        }

        /**
         * {@return the value returned by this return operation, or null if absent}
         */
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
     * The unreachable operation, that can model exit from a body that cannot complete normally.
     * <p>
     * An unreachable operation is a body-terminating operation.
     * <p>
     * The result type of an unreachable operation is {@link JavaType#VOID}.
     *
     * @jls 14.22 Unreachable Statements
     */
    @OpDeclaration(UnreachableOp.NAME)
    public static final class UnreachableOp extends CoreOp
            implements Op.BodyTerminating {
        static final String NAME = "unreachable";

        UnreachableOp(ExternalizedOp def) {
            if (!def.operands().isEmpty()) {
                throw new IllegalArgumentException("Operation must zero operands " + def.name());
            }

            this();
        }

        UnreachableOp(UnreachableOp that, CodeContext cc) {
            super(that, cc);
        }

        @Override
        public UnreachableOp transform(CodeContext cc, CodeTransformer ot) {
            return new UnreachableOp(this, cc);
        }

        UnreachableOp() {
            super(List.of());
        }

        @Override
        public TypeElement resultType() {
            return JavaType.VOID;
        }
    }

    /**
     * The yield operation, that can model exit from a body.
     * <p>
     * A yield operation is a body-terminating operation that accepts zero or one operand, corresponding to the value
     * yielded from the body to its parent operation.
     * <p>
     * The result type of a yield operation is {@link JavaType#VOID}.
     */
    @OpDeclaration(YieldOp.NAME)
    public static final class YieldOp extends CoreOp
            implements Op.BodyTerminating {
        static final String NAME = "yield";

        YieldOp(ExternalizedOp def) {
            if (def.operands().size() > 1) {
                throw new IllegalArgumentException("Operation must have zero or one operand " + def.name());
            }

            this(def.operands());
        }

        YieldOp(YieldOp that, CodeContext cc) {
            super(that, cc);
        }

        @Override
        public YieldOp transform(CodeContext cc, CodeTransformer ot) {
            return new YieldOp(this, cc);
        }

        YieldOp() {
            super(List.of());
        }

        YieldOp(List<Value> operands) {
            super(operands);
        }

        /**
         * {@return the value yielded by this yield operation, or null if absent}
         */
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
     * The unconditional branch operation, that can model transfer of control from one block to a successor block.
     * <p>
     * A branch operation is a block-terminating operation that accepts no operands and one successor, the next block
     * to branch to. The arguments of the successor are assigned to the parameters to the target block.
     * <p>
     * The result type of a branch operation is {@link JavaType#VOID}.
     */
    @OpDeclaration(BranchOp.NAME)
    public static final class BranchOp extends CoreOp
            implements Op.BlockTerminating {
        static final String NAME = "branch";

        final Block.Reference b;

        BranchOp(ExternalizedOp def) {
            if (!def.operands().isEmpty() || def.successors().size() != 1) {
                throw new IllegalArgumentException("Operation must have zero arguments and one successor" + def.name());
            }

            this(def.successors().get(0));
        }

        BranchOp(BranchOp that, CodeContext cc) {
            super(that, cc);

            this.b = cc.getSuccessorOrCreate(that.b);
        }

        @Override
        public BranchOp transform(CodeContext cc, CodeTransformer ot) {
            return new BranchOp(this, cc);
        }

        BranchOp(Block.Reference successor) {
            super(List.of());

            this.b = successor;
        }

        @Override
        public List<Block.Reference> successors() {
            return List.of(b);
        }

        /**
         * {@return The branch target}
         */
        public Block.Reference branch() {
            return b;
        }

        @Override
        public TypeElement resultType() {
            return JavaType.VOID;
        }
    }

    /**
     * The conditional branch operation, that can model transfer of control from one block to one of two successor
     * blocks.
     * <p>
     * A conditional branch operation is a block-terminating operation that accepts one boolean operand and two
     * successors, the true successor and the false successor. When the operand is true the true successor is
     * selected, otherwise the false successor is selected. The arguments of the selected successor are assigned
     * to the parameters to the target block.
     * <p>
     * The result type of a conditional branch operation is {@link JavaType#VOID}.
     */
    @OpDeclaration(ConditionalBranchOp.NAME)
    public static final class ConditionalBranchOp extends CoreOp
            implements Op.BlockTerminating {
        static final String NAME = "cbranch";

        final Block.Reference t;
        final Block.Reference f;

        ConditionalBranchOp(ExternalizedOp def) {
            if (def.operands().size() != 1 || def.successors().size() != 2) {
                throw new IllegalArgumentException("Operation must one operand and two successors" + def.name());
            }

            this(def.operands().getFirst(), def.successors().get(0), def.successors().get(1));
        }

        ConditionalBranchOp(ConditionalBranchOp that, CodeContext cc) {
            super(that, cc);

            this.t = cc.getSuccessorOrCreate(that.t);
            this.f = cc.getSuccessorOrCreate(that.f);
        }

        @Override
        public ConditionalBranchOp transform(CodeContext cc, CodeTransformer ot) {
            return new ConditionalBranchOp(this, cc);
        }

        ConditionalBranchOp(Value p, Block.Reference t, Block.Reference f) {
            super(List.of(p));

            this.t = t;
            this.f = f;
        }

        @Override
        public List<Block.Reference> successors() {
            return List.of(t, f);
        }

        /**
         * {@return the branch condition}
         */
        public Value predicate() {
            return operands().get(0);
        }

        /**
         * {@return the branch target when the condition is true}
         */
        public Block.Reference trueBranch() {
            return t;
        }

        /**
         * {@return the branch target when the condition is false}
         */
        public Block.Reference falseBranch() {
            return f;
        }

        @Override
        public TypeElement resultType() {
            return JavaType.VOID;
        }
    }

    /**
     * The constant operation, that can model a constant value, such as a Java literal.
     * <p>
     * A constant operation accepts no operands and stores the constant value as an attribute.
     * <p>
     * The result type of a constant operation is the type of the modeled constant value.
     *
     * @jls 15.29 Constant Expressions
     */
    @OpDeclaration(ConstantOp.NAME)
    public static final class ConstantOp extends CoreOp
            implements Op.Pure, JavaOp.JavaExpression {
        static final String NAME = "constant";

        /**
         * The externalized attribute modelling the constant value
         */
        static final String ATTRIBUTE_CONSTANT_VALUE = NAME + ".value";

        final Object value;
        final TypeElement type;

        ConstantOp(ExternalizedOp def) {
            if (!def.operands().isEmpty()) {
                throw new IllegalArgumentException("Operation must have zero operands");
            }

            Object value = def.extractAttributeValue(ATTRIBUTE_CONSTANT_VALUE, true,
                    v -> processConstantValue(def.resultType(), v));

            this(def.resultType(), value);
        }

        static Object processConstantValue(TypeElement t, Object value) {
            if (t.equals(JavaType.BOOLEAN) && value instanceof Boolean) {
                return value;
            } else if (t.equals(JavaType.BYTE) && value instanceof Number n) {
                return n.byteValue();
            } else if (t.equals(JavaType.SHORT) && value instanceof Number n) {
                return n.shortValue();
            } else if (t.equals(JavaType.CHAR) && value instanceof Character) {
                return value;
            } else if (t.equals(JavaType.INT) && value instanceof Number n) {
                return n.intValue();
            } else if (t.equals(JavaType.LONG) && value instanceof Number n) {
                return n.longValue();
            } else if (t.equals(JavaType.FLOAT) && value instanceof Number n) {
                return n.floatValue();
            } else if (t.equals(JavaType.DOUBLE) && value instanceof Number n) {
                return n.doubleValue();
            } else if (t.equals(JavaType.J_L_STRING)) {
                return value == ExternalizedOp.NULL_ATTRIBUTE_VALUE ?
                        null : (String)value;
            } else if (t.equals(JavaType.J_L_CLASS)) {
                return value == ExternalizedOp.NULL_ATTRIBUTE_VALUE ?
                        null : (TypeElement)value;
            } else if (value == ExternalizedOp.NULL_ATTRIBUTE_VALUE) {
                return null; // null constant
            }

            throw new UnsupportedOperationException("Unsupported constant type and value: " + t + " " + value);
        }

        ConstantOp(ConstantOp that, CodeContext cc) {
            super(that, cc);

            this.type = that.type;
            this.value = that.value;
        }

        @Override
        public ConstantOp transform(CodeContext cc, CodeTransformer ot) {
            return new ConstantOp(this, cc);
        }

        ConstantOp(TypeElement type, Object value) {
            super(List.of());

            this.type = type;
            this.value = value;
        }

        @Override
        public Map<String, Object> externalize() {
            return Map.of("", value == null ? ExternalizedOp.NULL_ATTRIBUTE_VALUE : value);
        }

        /**
         * {@return the constant value modeled by this constant operation}
         */
        public Object value() {
            return value;
        }

        @Override
        public TypeElement resultType() {
            return type;
        }
    }

    /**
     * A runtime representation of the storage associated with a variable operation.
     *
     * @param <T> the type of the var's value
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
     * The variable operation, that can model declarations of mutable storage.
     * <p>
     * In code models derived from Java source, variable operations can model Java local variables, method parameters,
     * or lambda parameters.
     * <p>
     * A variable operation accepts zero or one operand, corresponding to the initial value of the variable when
     * present.
     * <p>
     * The result type of a variable operation is the parameterized class type {@code Var<T>},
     * where {@code T} is the type element modeling the variable's type.
     *
     * @jls 14.4 Local Variable Declarations
     * @jls 8.4.1 Formal Parameters
     * @jls 15.27.1 Lambda Parameters
     */
    @OpDeclaration(VarOp.NAME)
    public static final class VarOp extends CoreOp
            implements JavaOp.JavaStatement {
        static final String NAME = "var";

        /**
         * The externalized attribute modelling the variable name
         */
        static final String ATTRIBUTE_NAME = NAME + ".name";

        final String varName;
        final VarType resultType;

        VarOp(ExternalizedOp def) {
            if (def.operands().size() > 1) {
                throw new IllegalStateException("Operation must have zero or one operand");
            }

            String name = def.extractAttributeValue(ATTRIBUTE_NAME, true,
                    v -> switch (v) {
                        case String s -> s;
                        case null -> "";
                        default -> throw new UnsupportedOperationException("Unsupported var name value:" + v);
                    });

            // @@@ Cannot use canonical constructor because type is wrapped
            super(def.operands());

            this.varName = name;
            this.resultType = (VarType) def.resultType();
        }

        VarOp(VarOp that, CodeContext cc) {
            super(that, cc);

            this.varName = that.varName;
            this.resultType = that.isResultTypeOverridable()
                    ? CoreType.varType(initOperand().type()) : that.resultType;
        }

        boolean isResultTypeOverridable() {
            return !isUninitialized() && resultType().valueType().equals(initOperand().type());
        }

        @Override
        public VarOp transform(CodeContext cc, CodeTransformer ot) {
            return new VarOp(this, cc);
        }

        VarOp(String varName, TypeElement type, Value init) {
            super(init == null ? List.of() : List.of(init));

            this.varName =  varName == null ? "" : varName;
            this.resultType = CoreType.varType(type);
        }

        @Override
        public Map<String, Object> externalize() {
            return isUnnamedVariable() ? Map.of() : Map.of("", varName);
        }

        /**
         * {@return the initial value assigned to this variable}
         * @throws IllegalStateException if this variable doesn't have an initial value,
         *                               that is, if it models an uninitialized variable
         */
        public Value initOperand() {
            if (operands().isEmpty()) {
                throw new IllegalStateException("Uninitialized variable");
            }
            return operands().getFirst();
        }

        /**
         * {@return the variable name}
         */
        public String varName() {
            return varName;
        }

        /**
         * {@return the variable type}
         */
        public TypeElement varValueType() {
            return resultType.valueType();
        }

        @Override
        public VarType resultType() {
            return resultType;
        }

        /**
         * {@return true if this variable operation models an unnamed variable}
         */
        public boolean isUnnamedVariable() {
            return varName.isEmpty();
        }

        /**
         * {@return true if this variable operation models an uninitialized variable}
         */
        public boolean isUninitialized() {
            return operands().isEmpty();
        }
    }

    /**
     * A variable access operation, that can model access to mutable storage.
     * <p>
     * In code models derived from Java source, variable access operations can model access to Java local variables,
     * method parameters, or lambda parameters.
     * <p>
     * Variable access operations accept the accessed variable as their first operand.
     *
     * @see JavaOp.FieldAccessOp
     */
    public sealed abstract static class VarAccessOp extends CoreOp
            implements JavaOp.AccessOp {
        VarAccessOp(VarAccessOp that, CodeContext cc) {
            super(that, cc);
        }

        VarAccessOp(List<Value> operands) {
            super(operands);
        }

        /**
         * {@return the accessed variable}
         */
        public Value varOperand() {
            return operands().getFirst();
        }

        /**
         * {@return the type of the accessed variable}
         */
        public VarType varType() {
            return (VarType) varOperand().type();
        }

        /**
         * {@return the variable operation associated with this access operation}
         */
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
         * The var load operation, that can model reading a variable.
         * <p>
         * A var load operation accepts the accessed variable as its operand.
         * <p>
         * The result type of a var load operation is the value type of the accessed variable.
         *
         * @see JavaOp.FieldAccessOp.FieldLoadOp
         * @jls 6.5.6.1 Simple Expression Names
         */
        @OpDeclaration(VarLoadOp.NAME)
        public static final class VarLoadOp extends VarAccessOp
                implements JavaOp.JavaExpression {
            static final String NAME = "var.load";

            VarLoadOp(ExternalizedOp opdef) {
                if (opdef.operands().size() != 1) {
                    throw new IllegalArgumentException("Operation must have one operand");
                }
                checkIsVarOp(opdef.operands().get(0));

                this(opdef.operands().get(0));
            }

            VarLoadOp(VarLoadOp that, CodeContext cc) {
                super(that, cc);
            }

            @Override
            public VarLoadOp transform(CodeContext cc, CodeTransformer ot) {
                return new VarLoadOp(this, cc);
            }

            // (Variable)VarType
            VarLoadOp(Value varValue) {
                super(List.of(varValue));
            }

            @Override
            public TypeElement resultType() {
                return varType().valueType();
            }
        }

        /**
         * The var store operation, that can model assignment to a variable.
         * <p>
         * A var store operation accepts two operands: the accessed variable and the value to store.
         * <p>
         * The result type of a var store operation is {@link JavaType#VOID}.
         *
         * @see JavaOp.FieldAccessOp.FieldStoreOp
         * @jls 15.26 Assignment Operators
         */
        @OpDeclaration(VarStoreOp.NAME)
        public static final class VarStoreOp extends VarAccessOp
                implements JavaOp.JavaExpression, JavaOp.JavaStatement {
            static final String NAME = "var.store";

            VarStoreOp(ExternalizedOp opdef) {
                if (opdef.operands().size() != 2) {
                    throw new IllegalArgumentException("Operation must have two operands");
                }
                checkIsVarOp(opdef.operands().get(0));

                this(opdef.operands().get(0), opdef.operands().get(1));
            }

            VarStoreOp(VarStoreOp that, CodeContext cc) {
                super(that, cc);
            }

            VarStoreOp(List<Value> values) {
                super(values);
            }

            @Override
            public VarStoreOp transform(CodeContext cc, CodeTransformer ot) {
                return new VarStoreOp(this, cc);
            }

            // (Variable, VarType)void
            VarStoreOp(Value varValue, Value v) {
                super(List.of(varValue, v));
            }

            /**
             * {@return the value being stored to the variable}
             */
            public Value storeOperand() {
                return operands().get(1);
            }

            @Override
            public TypeElement resultType() {
                return JavaType.VOID;
            }
        }
    }

    // Tuple operations, for modeling return with multiple values

    /**
     * The tuple operation, that can model constructing a tuple from a fixed set of values.
     * <p>
     * A tuple operation accepts one operand per tuple component, in component order.
     * <p>
     * The result type of a tuple operation is a {@linkplain TupleType tuple type} whose component types are
     * derived from the operand types.
     *
     * @see TupleLoadOp
     * @see TupleWithOp
     */
    @OpDeclaration(TupleOp.NAME)
    public static final class TupleOp extends CoreOp {
        static final String NAME = "tuple";

        TupleOp(ExternalizedOp def) {
            this(def.operands());
        }

        TupleOp(TupleOp that, CodeContext cc) {
            super(that, cc);
        }

        @Override
        public TupleOp transform(CodeContext cc, CodeTransformer ot) {
            return new TupleOp(this, cc);
        }

        TupleOp(List<? extends Value> componentValues) {
            super(componentValues);
        }

        @Override
        public TypeElement resultType() {
            return CoreType.tupleTypeFromValues(operands());
        }
    }

    /**
     * The tuple component load operation, that can model reading a tuple component at a given index.
     * <p>
     * A tuple load operation accepts one operand, the tuple value. The index of the accessed component is
     * identified using an attribute.
     * <p>
     * The result type of a tuple load operation is the type of the selected tuple component.
     *
     * @see TupleOp
     */
    @OpDeclaration(TupleLoadOp.NAME)
    public static final class TupleLoadOp extends CoreOp {
        static final String NAME = "tuple.load";

        /**
         * The externalized attribute modelling the tuple index
         */
        static final String ATTRIBUTE_INDEX = NAME + ".index";

        final int index;

        TupleLoadOp(ExternalizedOp def) {
            if (def.operands().size() != 1) {
                throw new IllegalStateException("Operation must have one operand");
            }

            int index = def.extractAttributeValue(ATTRIBUTE_INDEX, true,
                    v -> switch (v) {
                        case Integer i -> i;
                        case null, default -> throw new UnsupportedOperationException("Unsupported tuple index value:" + v);
                    });

            this(def.operands().get(0), index);
        }

        TupleLoadOp(TupleLoadOp that, CodeContext cc) {
            super(that, cc);

            this.index = that.index;
        }

        @Override
        public TupleLoadOp transform(CodeContext cc, CodeTransformer ot) {
            return new TupleLoadOp(this, cc);
        }

        TupleLoadOp(Value tupleValue, int index) {
            super(List.of(tupleValue));

            this.index = index;
        }

        @Override
        public Map<String, Object> externalize() {
            return Map.of("", index);
        }

        /**
         * {@return the component index of this tuple load operation}
         */
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
     * The tuple with operation, that can model replacing a tuple component at a given index.
     * <p>
     * A tuple with operation accepts two operands: the tuple value and the replacement component value. The index of the
     * component to be replaced is identified using an attribute.
     * <p>
     * The result type of a tuple with operation is a {@linkplain TupleType tuple type} obtained from the
     * tuple value type by replacing the selected component type with the replacement component value type.
     *
     * @see TupleOp
     */
    @OpDeclaration(TupleWithOp.NAME)
    public static final class TupleWithOp extends CoreOp {
        static final String NAME = "tuple.with";

        /**
         * The externalized attribute modelling the tuple index
         */
        static final String ATTRIBUTE_INDEX = NAME + ".index";

        final int index;

        TupleWithOp(ExternalizedOp def) {
            if (def.operands().size() != 2) {
                throw new IllegalStateException("Operation must have two operands");
            }

            int index = def.extractAttributeValue(ATTRIBUTE_INDEX, true,
                    v -> switch (v) {
                        case Integer i -> i;
                        case null, default -> throw new UnsupportedOperationException("Unsupported tuple index value:" + v);
                    });

            this(def.operands().get(0), index, def.operands().get(1));
        }

        TupleWithOp(TupleWithOp that, CodeContext cc) {
            super(that, cc);

            this.index = that.index;
        }

        @Override
        public TupleWithOp transform(CodeContext cc, CodeTransformer ot) {
            return new TupleWithOp(this, cc);
        }

        TupleWithOp(Value tupleValue, int index, Value value) {
            super(List.of(tupleValue, value));

            // @@@ Validate tuple type and index
            this.index = index;
        }

        @Override
        public Map<String, Object> externalize() {
            return Map.of("", index);
        }

        /**
         * {@return the component index of this tuple with operation}
         */
        public int index() {
            return index;
        }

        @Override
        public TypeElement resultType() {
            Value tupleValue = operands().get(0);
            TupleType tupleType = (TupleType) tupleValue.type();
            Value value = operands().get(1);

            List<TypeElement> tupleComponentTypes = new ArrayList<>(tupleType.componentTypes());
            tupleComponentTypes.set(index, value.type());
            return CoreType.tupleType(tupleComponentTypes);
        }
    }

    static Op createOp(ExternalizedOp def) {
        Op op = switch (def.name()) {
            case "branch" -> new BranchOp(def);
            case "cbranch" -> new ConditionalBranchOp(def);
            case "constant" -> new ConstantOp(def);
            case "func" -> new FuncOp(def);
            case "func.call" -> new FuncCallOp(def);
            case "module" -> new ModuleOp(def);
            case "quoted" -> new QuotedOp(def);
            case "return" -> new ReturnOp(def);
            case "tuple" -> new TupleOp(def);
            case "tuple.load" -> new TupleLoadOp(def);
            case "tuple.with" -> new TupleWithOp(def);
            case "unreachable" -> new UnreachableOp(def);
            case "var" -> new VarOp(def);
            case "var.load" -> new VarAccessOp.VarLoadOp(def);
            case "var.store" -> new VarAccessOp.VarStoreOp(def);
            case "yield" -> new YieldOp(def);
            default -> null;
        };
        if (op != null) {
            op.setLocation(def.location());
        }
        return op;
    }

    /**
     * An operation factory for core operations.
     */
    public static final OpFactory CORE_OP_FACTORY = CoreOp::createOp;

    /**
     * Creates a function operation builder.
     *
     * @param funcName the function name
     * @param funcType the function type
     * @return the function operation builder
     */
    public static FuncOp.Builder func(String funcName, FunctionType funcType) {
        return new FuncOp.Builder(null, funcName, funcType);
    }

    /**
     * Creates a function operation builder.
     *
     * @param refType  the function reference type
     * @param funcName the function name
     * @param funcType the function type
     * @return the function operation builder
     */
    public static FuncOp.Builder func(TypeElement refType, String funcName, FunctionType funcType) {
        return new FuncOp.Builder(null, MethodRef.method(refType, funcName, funcType));
    }

    /**
     * Creates a function operation builder.
     *
     * @param mref the function method reference
     * @return the function operation builder
     */
    public static FuncOp.Builder func(MethodRef mref) {
        return new FuncOp.Builder(null, mref);
    }

    /**
     * Creates a function operation.
     *
     * @param funcName the function name
     * @param body     the body builder defining the function body
     * @return the function operation
     */
    public static FuncOp func(String funcName, Body.Builder body) {
        return new FuncOp(funcName, body);
    }

    /**
     * Creates a function operation.
     *
     * @param mref the method reference the function operation models
     * @param body the body builder defining the function body
     * @return the function operation
     */
    public static FuncOp func(MethodRef mref, Body.Builder body) {
        return new FuncOp(mref, body);
    }

    /**
     * Creates a function operation.
     *
     * @param refType the function reference type
     * @param body    the body builder defining the function body
     * @return the function operation
     */
    public static FuncOp func(TypeElement refType, String funcName, Body.Builder body) {
        return new FuncOp(MethodRef.method(refType, funcName, body.bodyType()), body);
    }

    /**
     * Creates a function call operation.
     *
     * @param funcName the name of the target function
     * @param funcType the type of the target function
     * @param args     the function arguments
     * @return the function call operation
     */
    public static FuncCallOp funcCall(String funcName, FunctionType funcType, Value... args) {
        return funcCall(funcName, funcType, List.of(args));
    }

    /**
     * Creates a function call operation.
     *
     * @param funcName the name of the target function
     * @param funcType the type of the target function
     * @param args     the function arguments
     * @return the function call operation
     */
    public static FuncCallOp funcCall(String funcName, FunctionType funcType, List<Value> args) {
        return new FuncCallOp(funcName, funcType.returnType(), args);
    }

    /**
     * Creates a function call operation.
     *
     * @param func the target function
     * @param args the function arguments
     * @return the function call operation
     */
    public static FuncCallOp funcCall(FuncOp func, Value... args) {
        return funcCall(func, List.of(args));
    }

    /**
     * Creates a function call operation.
     *
     * @param func the target function
     * @param args the function arguments
     * @return the function call operation
     */
    public static FuncCallOp funcCall(FuncOp func, List<Value> args) {
        return new FuncCallOp(func.funcName(), func.invokableType().returnType(), args);
    }

    /**
     * Creates a module operation.
     *
     * <p>
     * The function operations are specified in the order in which they will appear in the
     * {@linkplain ModuleOp#functionTable() symbol table} of the returned module operation.
     *
     * @param functions the function operations in the module
     * @return the module operation
     */
    public static ModuleOp module(FuncOp... functions) {
        return module(List.of(functions));
    }

    /**
     * Creates a module operation.
     *
     * <p>
     * The function operations are specified in the order in which they will appear in the
     * {@linkplain ModuleOp#functionTable() symbol table} of the returned module operation.
     *
     * @param functions the function operations in the module
     * @return the module operation
     */
    public static ModuleOp module(List<FuncOp> functions) {
        return new ModuleOp(List.copyOf(functions));
    }

    /**
     * Creates a quoted operation.
     *
     * @param ancestorBody the nearest ancestor body builder from which to construct
     *                     the body builder for this operation
     * @param opFunc       a function that accepts a builder for the quoted operation body and returns the operation to be quoted
     * @return the quoted operation
     */
    public static QuotedOp quoted(Body.Builder ancestorBody,
                                  Function<Block.Builder, Op> opFunc) {
        Body.Builder body = Body.Builder.of(ancestorBody, CoreType.FUNCTION_TYPE_VOID);
        Block.Builder block = body.entryBlock();
        block.op(core_yield(
                block.op(opFunc.apply(block))));
        return new QuotedOp(body);
    }

    /**
     * Creates a quoted operation.
     *
     * @param body the quoted operation body builder, which yields the operation to be quoted
     * @return the quoted operation
     */
    public static QuotedOp quoted(Body.Builder body) {
        return new QuotedOp(body);
    }

    /**
     * Creates a return operation with no returned value.
     *
     * @return the return operation
     */
    public static ReturnOp return_() {
        return return_(null);
    }

    /**
     * Creates a return operation.
     *
     * @param returnValue the return value
     * @return the return operation
     */
    public static ReturnOp return_(Value returnValue) {
        return new ReturnOp(returnValue);
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
     * Creates a yield operation with no yielded value.
     *
     * @return the yield operation
     */
    public static YieldOp core_yield() {
        return new YieldOp();
    }

    /**
     * Creates a yield operation.
     *
     * @param yieldValue the yielded value
     * @return the yield operation
     */
    public static YieldOp core_yield(Value yieldValue) {
        return new YieldOp(List.of(yieldValue));
    }

    /**
     * Creates an unconditional branch operation.
     *
     * @param target the jump target
     * @return the unconditional branch operation
     */
    public static BranchOp branch(Block.Reference target) {
        return new BranchOp(target);
    }

    /**
     * Creates a conditional branch operation.
     *
     * @param condValue   the test value of the conditional branch operation
     * @param trueTarget  the jump target when the test value evaluates to true
     * @param falseTarget the jump target when the test value evaluates to false
     * @return the conditional branch operation
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

    // @@@ Add field load/store overload with explicit fieldType

    /**
     * Creates a variable operation modeling an unnamed and uninitialized variable,
     * either an unnamed local variable or an unnamed parameter.
     *
     * @param type the type of the var's value
     * @return the var operation
     */
    public static VarOp var(TypeElement type) {
        return var(null, type);
    }

    /**
     * Creates a variable operation modeling an uninitialized variable.
     *
     * @param name the name of the variable
     * @param type the variable type
     * @return the var operation
     */
    public static VarOp var(String name, TypeElement type) {
        return var(name, type, null);
    }

    /**
     * Creates a variable operation modeling an unnamed variable.
     *
     * @param init the variable's initial value
     * @return the variable operation
     */
    public static VarOp var(Value init) {
        return var(null, init);
    }

    /**
     * Creates a variable operation.
     * <p>
     * If the given name is {@code null} or an empty string then the variable is an unnamed variable.
     *
     * @param name the variable name
     * @param init the variable's initial value
     * @return the variable operation
     */
    public static VarOp var(String name, Value init) {
        return var(name, init.type(), init);
    }

    /**
     * Creates a variable operation.
     * <p>
     * If the given name is {@code null} or an empty string then the variable is an unnamed variable.
     *
     * @param name the variable name
     * @param type the variable type
     * @param init the variable's initial value
     * @return the var operation
     */
    public static VarOp var(String name, TypeElement type, Value init) {
        return new VarOp(name, type, init);
    }

    /**
     * Creates a variable load operation.
     *
     * @param var the variable to be accessed
     * @return the variable load operation
     */
    public static VarAccessOp.VarLoadOp varLoad(Value var) {
        return new VarAccessOp.VarLoadOp(var);
    }

    /**
     * Creates a variable store operation.
     *
     * @param var the variable to be set
     * @param v   the value to store in {@code var}
     * @return the variable store operation
     */
    public static VarAccessOp.VarStoreOp varStore(Value var, Value v) {
        return new VarAccessOp.VarStoreOp(var, v);
    }

    /**
     * Creates a tuple operation.
     *
     * @param componentValues the values of the tuple components, in order
     * @return the tuple operation
     */
    public static TupleOp tuple(Value... componentValues) {
        return tuple(List.of(componentValues));
    }

    /**
     * Creates a tuple operation.
     *
     * @param componentValues the values of the tuple components, in order
     * @return the tuple operation
     */
    public static TupleOp tuple(List<? extends Value> componentValues) {
        return new TupleOp(componentValues);
    }

    /**
     * Creates a tuple load operation.
     *
     * @param tuple the tuple to be accessed
     * @param index the index of the component to be accessed
     * @return the tuple load operation
     */
    public static TupleLoadOp tupleLoad(Value tuple, int index) {
        return new TupleLoadOp(tuple, index);
    }

    /**
     * Creates a tuple with operation.
     *
     * @param tuple the tuple with the component to be replaced
     * @param index the index of the component to be replaced
     * @param value the replacement component value
     * @return the tuple with operation
     */
    public static TupleWithOp tupleWith(Value tuple, int index, Value value) {
        return new TupleWithOp(tuple, index, value);
    }
}

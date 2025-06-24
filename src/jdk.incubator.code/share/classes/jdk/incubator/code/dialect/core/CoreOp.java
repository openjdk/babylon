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

import java.lang.constant.ClassDesc;
import jdk.incubator.code.*;
import jdk.incubator.code.dialect.java.*;
import jdk.incubator.code.extern.ExternalizableOp;
import jdk.incubator.code.extern.OpFactory;

import java.util.*;
import java.util.function.Consumer;
import java.util.function.Function;

/**
 * The top-level operation class for the set of enclosed core operations.
 * <p>
 * A code model, produced by the Java compiler from Java program source, may consist of core operations and Java
 * operations. Such a model represents the same Java program and preserves the program meaning as defined by the
 * Java Language Specification
 */
public sealed abstract class CoreOp extends ExternalizableOp {

    static final String PACKAGE_NAME = CodeReflection.class.getPackageName();

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
                        case null, default -> throw new UnsupportedOperationException("Unsupported func name value:" + v);
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
                        case null, default -> throw new UnsupportedOperationException("Unsupported func name value:" + v);
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

            Body.Builder bodyC = Body.Builder.of(null, CoreType.FUNCTION_TYPE_VOID);
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
     * The closure operation, that can model a structured Java lambda expression
     * that has no target type (a functional interface).
     */
    @OpFactory.OpDeclaration(ClosureOp.NAME)
    public static final class ClosureOp extends CoreOp
            implements Op.Invokable, Op.Lowerable, JavaOp.JavaExpression {

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
            implements Op.BodyTerminating, JavaOp.JavaStatement {
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
            implements Op.Pure, JavaOp.JavaExpression {
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
                return value == NULL_ATTRIBUTE_VALUE ?
                        null : (String)value;
            } else if (t.equals(JavaType.J_L_CLASS)) {
                return value == NULL_ATTRIBUTE_VALUE ?
                        null : (TypeElement)value;
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
            implements JavaOp.JavaStatement {
        public static final String NAME = "var";
        public static final String ATTRIBUTE_NAME = NAME + ".name";

        final String varName;
        final VarType resultType;

        public static VarOp create(ExternalizedOp def) {
            if (def.operands().size() > 1) {
                throw new IllegalStateException("Operation must have zero or one operand");
            }

            String name = def.extractAttributeValue(ATTRIBUTE_NAME, true,
                    v -> switch (v) {
                        case String s -> s;
                        case null -> "";
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
                    ? CoreType.varType(initOperand().type()) : that.resultType;
        }

        boolean isResultTypeOverridable() {
            return !isUninitialized() && resultType().valueType().equals(initOperand().type());
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

            this.varName =  varName == null ? "" : varName;
            this.resultType = CoreType.varType(type);
        }

        // @@@ This and the above constructor can be merged when
        // statements before super can be used in the jdk.compiler module
        VarOp(String varName, TypeElement type) {
            super(NAME, List.of());

            this.varName =  varName == null ? "" : varName;
            this.resultType = CoreType.varType(type);
        }

        @Override
        public Map<String, Object> attributes() {
            if (isUnnamedVariable()) {
                return super.attributes();
            }

            HashMap<String, Object> m = new HashMap<>(super.attributes());
            m.put("", varName);
            return Collections.unmodifiableMap(m);
        }

        public Value initOperand() {
            if (operands().isEmpty()) {
                throw new IllegalStateException("Uninitialized variable");
            }
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

        public boolean isUnnamedVariable() {
            return varName.isEmpty();
        }

        public boolean isUninitialized() {
            return operands().isEmpty();
        }
    }

    /**
     * The var access operation, that can model access to Java language local variables, method parameters, or
     * lambda parameters.
     */
    public sealed abstract static class VarAccessOp extends CoreOp
            implements JavaOp.AccessOp {
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
                implements JavaOp.JavaExpression {
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
                implements JavaOp.JavaExpression, JavaOp.JavaStatement {
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
            return CoreType.tupleTypeFromValues(operands());
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
                        case Integer i -> i;
                        case null, default -> throw new UnsupportedOperationException("Unsupported tuple index value:" + v);
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
                        case Integer i -> i;
                        case null, default -> throw new UnsupportedOperationException("Unsupported tuple index value:" + v);
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
            return CoreType.tupleType(tupleComponentTypes);
        }
    }


    /**
     * An operation factory for core operations.
     */
    public static final OpFactory CORE_OP_FACTORY = OpFactory.OP_FACTORY.get(CoreOp.class);

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
        Body.Builder body = Body.Builder.of(ancestorBody, CoreType.FUNCTION_TYPE_VOID);
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

    // @@@ Add field load/store overload with explicit fieldType

    /**
     * Creates a var operation modeling an unnamed and uninitialized variable,
     * either an unnamed local variable or an unnamed parameter.
     *
     * @param type the type of the var's value
     * @return the var operation
     */
    public static VarOp var(TypeElement type) {
        return var(null, type);
    }

    /**
     * Creates a var operation modeling an uninitialized variable, either a local variable or a parameter.
     *
     * @param name the name of the var
     * @param type the type of the var's value
     * @return the var operation
     */
    public static VarOp var(String name, TypeElement type) {
        return new VarOp(name, type);
    }

    /**
     * Creates a var operation modeling an unnamed variable, either an unnamed local variable or an unnamed parameter.
     *
     * @param init the initial value of the var
     * @return the var operation
     */
    public static VarOp var(Value init) {
        return var(null, init);
    }

    /**
     * Creates a var operation modeling a variable, either a local variable or a parameter.
     * <p>
     * If the given name is {@code null} or an empty string then the variable is an unnamed variable.
     *
     * @param name the name of the var
     * @param init the initial value of the var
     * @return the var operation
     */
    public static VarOp var(String name, Value init) {
        return new VarOp(name, init);
    }

    /**
     * Creates a var operation modeling a variable, either a local variable or a parameter.
     * <p>
     * If the given name is {@code null} or an empty string then the variable is an unnamed variable.
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
}

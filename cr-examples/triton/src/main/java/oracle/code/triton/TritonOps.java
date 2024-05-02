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

package oracle.code.triton;

import java.lang.reflect.code.*;
import java.lang.reflect.code.op.*;
import java.lang.reflect.code.type.*;
import java.util.*;
import java.util.function.Consumer;

public class TritonOps {

    static abstract class TritonOp extends ExternalizableOp {
        final TypeElement resultType;

        public TritonOp(ExternalizedOp def) {
            super(def);

            this.resultType = def.resultType();
        }

        TritonOp(TritonOp that, CopyContext cc) {
            super(that, cc);

            this.resultType = that.resultType;
        }

        TritonOp(String name, TypeElement resultType, List<? extends Value> operands) {
            super(name, operands);

            this.resultType = resultType;
        }

        @Override
        public TypeElement resultType() {
            return resultType;
        }
    }

    @OpFactory.OpDeclaration(ModuleOp.NAME)
    public static final class ModuleOp extends TritonOp implements Op.Isolated {
        public static final String NAME = "module";

        final Map<String, FuncOp> table;
        final Body body;

        public ModuleOp(ExternalizedOp def) {
            super(def);

            this.body = def.bodyDefinitions().get(0).build(this);
            this.table = createTable(body);
        }

        ModuleOp(ModuleOp that, CopyContext cc, OpTransformer ot) {
            super(that, cc);

            this.body = that.body.transform(cc, ot).build(this);
            this.table = createTable(body);
        }

        static Map<String, FuncOp> createTable(Body body) {
            Map<String, FuncOp> table = new HashMap<>();
            for (var op : body.entryBlock().ops()) {
                if (op instanceof FuncOp fop) {
                    table.put(fop.funcName(), fop);
                } else if (op instanceof CoreOp.UnreachableOp _) {
                    // no operation
                } else {
                    throw new IllegalArgumentException("Bad operation in module: " + op);
                }
            }
            return Collections.unmodifiableMap(table);
        }

        @Override
        public ModuleOp transform(CopyContext cc, OpTransformer ot) {
            return new ModuleOp(this, cc, ot);
        }

        public ModuleOp transform(OpTransformer ot) {
            return new ModuleOp(this, CopyContext.create(), ot);
        }

        ModuleOp(List<FuncOp> functions) {
            super(NAME, JavaType.VOID,
                    List.of());

            Body.Builder bodyC = Body.Builder.of(null, FunctionType.VOID);
            Block.Builder entryBlock = bodyC.entryBlock();
            Map<String, FuncOp> table = new HashMap<>();
            for (FuncOp f : functions) {
                entryBlock.op(f);
                table.put(f.funcName(), f);
            }
            entryBlock.op(CoreOp.unreachable());
            this.table = Collections.unmodifiableMap(table);
            this.body = bodyC.build(this);
        }

        @Override
        public List<Body> bodies() {
            return List.of(body);
        }

        public Map<String, FuncOp> functionTable() {
            return table;
        }
    }

    @OpFactory.OpDeclaration(FuncOp.NAME)
    public static final class FuncOp extends TritonOp implements Op.Invokable, Op.Isolated, Op.Lowerable {

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

        public static final String NAME = "tt.func";
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
            super(NAME, JavaType.VOID,
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
            // and copy directly without lowering descendant operations
            b.op(this, OpTransformer.COPYING_TRANSFORMER);
            return b;
        }
    }

    @OpFactory.OpDeclaration(CallOp.NAME)
    public static final class CallOp extends TritonOp {
        public static final String NAME = "tt.call";
        public static final String ATTRIBUTE_FUNC_NAME = NAME + ".name";

        final String funcName;

        public static CallOp create(ExternalizedOp def) {
            String funcName = def.extractAttributeValue(ATTRIBUTE_FUNC_NAME, true,
                    v -> switch (v) {
                        case String s -> s;
                        default -> throw new UnsupportedOperationException("Unsupported func name value:" + v);
                    });

            return new CallOp(def, funcName);
        }

        CallOp(ExternalizedOp def, String funcName) {
            super(def);

            this.funcName = funcName;
        }

        CallOp(CallOp that, CopyContext cc) {
            super(that, cc);

            this.funcName = that.funcName;
        }

        @Override
        public CallOp transform(CopyContext cc, OpTransformer ot) {
            return new CallOp(this, cc);
        }

        CallOp(String funcName, TypeElement resultType, List<Value> args) {
            super(NAME, resultType, args);

            this.funcName = funcName;
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
    }

    @OpFactory.OpDeclaration(ReduceOp.NAME)
    public static final class ReduceOp extends TritonOp {
        // @@@ SSA transformation does not work with nested ops
        // implements Op.Nested {

        public static class Builder {
            final Body.Builder ancestorBody;
            final int axis;
            final Value v;
            final FunctionType reduceType;

            Builder(Body.Builder ancestorBody, int axis, Value v, FunctionType reduceType) {
                this.ancestorBody = ancestorBody;
                this.axis = axis;
                this.v = v;
                this.reduceType = reduceType;
            }

            public ReduceOp body(Consumer<Block.Builder> c) {
                Body.Builder body = Body.Builder.of(ancestorBody, reduceType);
                c.accept(body.entryBlock());
                return new ReduceOp(axis, v, body);
            }
        }

        public static final String NAME = "tt.reduce";
        public static final String ATTRIBUTE_AXIS = "axis";

        final int axis;
        final Body reducer;

        public static ReduceOp create(ExternalizedOp def) {
            int axis = def.extractAttributeValue(ATTRIBUTE_AXIS, true,
                    v -> switch (v) {
                        case String s -> Integer.valueOf(s);
                        case Integer i -> i;
                        default -> throw new UnsupportedOperationException("Unsupported axis value:" + v);
                    });
            return new ReduceOp(def, axis);
        }

        ReduceOp(ExternalizedOp def, int axis) {
            super(def);

            this.axis = axis;
            this.reducer = def.bodyDefinitions().get(0).build(this);
        }

        ReduceOp(ReduceOp that, CopyContext cc, OpTransformer ot) {
            super(that, cc);

            this.axis = that.axis;
            this.reducer = that.reducer.transform(cc, ot).build(this);
        }

        @Override
        public ReduceOp transform(CopyContext cc, OpTransformer ot) {
            return new ReduceOp(this, cc, ot);
        }

        ReduceOp(int axis, Value tensor, Body.Builder reducerBuilder) {
            super(NAME, reducerBuilder.bodyType().returnType(), List.of(tensor));

            this.axis = axis;
            this.reducer = reducerBuilder.build(this);
        }

        @Override
        public List<Body> bodies() {
            return List.of(reducer);
        }

        @Override
        public Map<String, Object> attributes() {
            HashMap<String, Object> m = new HashMap<>(super.attributes());
            m.put(ATTRIBUTE_AXIS, axis);
            return Collections.unmodifiableMap(m);
        }

        public int axis() {
            return axis;
        }

        public Body reducer() {
            return reducer;
        }
    }

    @OpFactory.OpDeclaration(ReduceReturnOp.NAME)
    public static class ReduceReturnOp extends TritonOp implements Op.Terminating {
        public static final String NAME = "tt.reduce.return";

        public ReduceReturnOp(ExternalizedOp def) {
            super(def);
        }

        ReduceReturnOp(ReduceReturnOp that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public ReduceReturnOp transform(CopyContext cc, OpTransformer ot) {
            return new ReduceReturnOp(this, cc);
        }

        ReduceReturnOp(Value r) {
            super(NAME, JavaType.VOID, List.of(r));
        }
    }

    @OpFactory.OpDeclaration(GetProgramIdOp.NAME)
    public static class GetProgramIdOp extends TritonOp implements Op.Pure {
        public static final String NAME = "tt.get_program_id";
        public static final String ATTRIBUTE_AXIS = "axis";

        final int axis;

        public static GetProgramIdOp create(ExternalizedOp def) {
            int axis = def.extractAttributeValue(ATTRIBUTE_AXIS, true,
                    v -> switch (v) {
                        case String s -> Integer.valueOf(s);
                        case Integer i -> i;
                        default -> throw new UnsupportedOperationException("Unsupported axis value:" + v);
                    });
            return new GetProgramIdOp(def, axis);
        }

        GetProgramIdOp(ExternalizedOp def, int axis) {
            super(def);

            this.axis = axis;
        }

        GetProgramIdOp(GetProgramIdOp that, CopyContext cc) {
            super(that, cc);

            this.axis = that.axis;
        }

        @Override
        public GetProgramIdOp transform(CopyContext cc, OpTransformer ot) {
            return new GetProgramIdOp(this, cc);
        }

        GetProgramIdOp(int axis) {
            super(NAME, JavaType.INT, List.of());

            this.axis = axis;
        }

        @Override
        public Map<String, Object> attributes() {
            HashMap<String, Object> m = new HashMap<>(super.attributes());
            m.put("", axis);
            return Collections.unmodifiableMap(m);
        }

        public int axis() {
            return axis;
        }
    }

    @OpFactory.OpDeclaration(MakeRangeOp.NAME)
    public static class MakeRangeOp extends TritonOp implements Op.Pure {
        public static final String NAME = "tt.make_range";
        public static final String ATTRIBUTE_START = "start";
        public static final String ATTRIBUTE_END = "end";

        final int start;
        final int end;

        public static MakeRangeOp create(ExternalizedOp def) {
            int start = def.extractAttributeValue(ATTRIBUTE_START, false,
                    v -> switch (v) {
                        case String s -> Integer.valueOf(s);
                        case Integer i -> i;
                        default -> throw new UnsupportedOperationException("Unsupported start value:" + v);
                    });
            int end = def.extractAttributeValue(ATTRIBUTE_END, false,
                    v -> switch (v) {
                        case String s -> Integer.valueOf(s);
                        case Integer i -> i;
                        default -> throw new UnsupportedOperationException("Unsupported end value:" + v);
                    });
            return new MakeRangeOp(def, start, end);
        }

        MakeRangeOp(ExternalizedOp def, int start, int end) {
            super(def);

            this.start = start;
            this.end = end;
        }

        MakeRangeOp(MakeRangeOp that, CopyContext cc) {
            super(that, cc);

            this.start = that.start;
            this.end = that.end;
        }

        @Override
        public MakeRangeOp transform(CopyContext cc, OpTransformer ot) {
            return new MakeRangeOp(this, cc);
        }

        MakeRangeOp(int start, int end) {
            super(NAME, tensorType(start, end), List.of());

            this.start = start;
            this.end = end;
        }

        static TensorType tensorType(int start, int end) {
            return new TensorType(JavaType.INT, List.of(end - start));
        }

        @Override
        public Map<String, Object> attributes() {
            HashMap<String, Object> m = new HashMap<>(super.attributes());
            m.put(ATTRIBUTE_START, start);
            m.put(ATTRIBUTE_END, end);
            return Collections.unmodifiableMap(m);
        }
    }

    @OpFactory.OpDeclaration(ExpandOp.NAME)
    public static class ExpandOp extends TritonOp implements Op.Pure {
        public static final String NAME = "tt.expand_dims";
        public static final String ATTRIBUTE_AXIS = "axis";

        final int axis;

        public static ExpandOp create(ExternalizedOp def) {
            int axis = def.extractAttributeValue(ATTRIBUTE_AXIS, true,
                    v -> switch (v) {
                        case String s -> Integer.valueOf(s);
                        case Integer i -> i;
                        default -> throw new UnsupportedOperationException("Unsupported axis value:" + v);
                    });
            return new ExpandOp(def, axis);
        }

        ExpandOp(ExternalizedOp def, int axis) {
            super(def);

            this.axis = axis;
        }

        ExpandOp(ExpandOp that, CopyContext cc) {
            super(that, cc);

            this.axis = that.axis;
        }

        @Override
        public ExpandOp transform(CopyContext cc, OpTransformer ot) {
            return new ExpandOp(this, cc);
        }

        ExpandOp(int axis, TypeElement tensorType, Value v) {
            super(NAME, tensorType, List.of(v));

            this.axis = axis;
        }

        @Override
        public Map<String, Object> attributes() {
            HashMap<String, Object> m = new HashMap<>(super.attributes());
            m.put("", axis);
            return Collections.unmodifiableMap(m);
        }

        public int axis() {
            return axis;
        }
    }

    @OpFactory.OpDeclaration(SplatOp.NAME)
    public static class SplatOp extends TritonOp implements Op.Pure {
        public static final String NAME = "tt.splat";

        public SplatOp(ExternalizedOp def) {
            super(def);
        }

        SplatOp(SplatOp that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public SplatOp transform(CopyContext cc, OpTransformer ot) {
            return new SplatOp(this, cc);
        }

        SplatOp(TypeElement tensorType, Value v) {
            super(NAME, tensorType, List.of(v));
        }
    }

    @OpFactory.OpDeclaration(BroadcastOp.NAME)
    public static class BroadcastOp extends TritonOp implements Op.Pure {
        public static final String NAME = "tt.broadcast";

        public BroadcastOp(ExternalizedOp def) {
            super(def);
        }

        BroadcastOp(BroadcastOp that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public BroadcastOp transform(CopyContext cc, OpTransformer ot) {
            return new BroadcastOp(this, cc);
        }

        BroadcastOp(TypeElement tensorType, Value v) {
            super(NAME, tensorType, List.of(v));
        }
    }

    @OpFactory.OpDeclaration(AddPtrOp.NAME)
    public static class AddPtrOp extends TritonOp implements Op.Pure {
        public static final String NAME = "tt.addptr";

        public AddPtrOp(ExternalizedOp def) {
            super(def);
        }

        AddPtrOp(AddPtrOp that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public AddPtrOp transform(CopyContext cc, OpTransformer ot) {
            return new AddPtrOp(this, cc);
        }

        AddPtrOp(Value ptr, Value offset) {
            super(NAME, ptr.type(), List.of(ptr, offset));
        }
    }

    @OpFactory.OpDeclaration(LoadOp.NAME)
    public static class LoadOp extends TritonOp implements Op.Pure {
        public static final String NAME = "tt.load";

        public LoadOp(ExternalizedOp def) {
            super(def);
        }

        LoadOp(LoadOp that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public LoadOp transform(CopyContext cc, OpTransformer ot) {
            return new LoadOp(this, cc);
        }

        LoadOp(TypeElement tensorType, Value ptr, Value mask) {
            super(NAME, tensorType, List.of(ptr, mask));
        }
    }

    @OpFactory.OpDeclaration(StoreOp.NAME)
    public static class StoreOp extends TritonOp {
        public static final String NAME = "tt.store";

        public StoreOp(ExternalizedOp def) {
            super(def);
        }

        StoreOp(StoreOp that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public StoreOp transform(CopyContext cc, OpTransformer ot) {
            return new StoreOp(this, cc);
        }

        StoreOp(Value ptr, Value v, Value mask) {
            super(NAME, JavaType.VOID, List.of(ptr, v, mask));
        }
    }

    @OpFactory.OpDeclaration(ReturnOp.NAME)
    public static class ReturnOp extends TritonOp implements Op.Terminating {
        public static final String NAME = "tt.return";

        public ReturnOp(ExternalizedOp def) {
            super(def);
        }

        ReturnOp(ReturnOp that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public ReturnOp transform(CopyContext cc, OpTransformer ot) {
            return new ReturnOp(this, cc);
        }

        ReturnOp() {
            super(NAME, JavaType.VOID, List.of());
        }

        ReturnOp(Value v) {
            super(NAME, JavaType.VOID, List.of(v));
        }
    }

    @OpFactory.OpDeclaration(DotOp.NAME)
    public static class DotOp extends TritonOp implements Op.Pure {
        public static final String NAME = "tt.dot";

        public DotOp(ExternalizedOp def) {
            super(def);
        }

        DotOp(DotOp that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public DotOp transform(CopyContext cc, OpTransformer ot) {
            return new DotOp(this, cc);
        }

        DotOp(TypeElement tensorType, Value a, Value b) {
            super(NAME, tensorType, List.of(a, b));
        }
    }


    public static ModuleOp module(FuncOp... functions) {
        return module(List.of(functions));
    }

    public static ModuleOp module(List<FuncOp> functions) {
        return new ModuleOp(List.copyOf(functions));
    }

    public static FuncOp.Builder func(String funcName, FunctionType funcType) {
        return new FuncOp.Builder(null, funcName, funcType);
    }

    public static FuncOp func(String funcName, Body.Builder body) {
        return new FuncOp(funcName, body);
    }

    public static CallOp call(FuncOp func, Value... args) {
        return call(func, List.of(args));
    }

    public static CallOp call(FuncOp func, List<Value> args) {
        return new CallOp(func.funcName(), func.invokableType().returnType(), args);
    }

    public static ReduceOp.Builder reduce(Body.Builder ancestorBody, int axis, Value tensor,
                                          FunctionType reduceType) {
        return new ReduceOp.Builder(ancestorBody, axis, tensor, reduceType);
    }

    public static ReduceOp reduce(int axis, Value tensor, Body.Builder reducerBuilder) {
        return new ReduceOp(axis, tensor, reducerBuilder);
    }

    public static ReduceReturnOp reduceReturn(Value r) {
        return new ReduceReturnOp(r);
    }

    public static GetProgramIdOp getProgramId(int axis) {
        // @@@ 1 <= axis <= 3
        return new GetProgramIdOp(axis);
    }

    public static MakeRangeOp makeRange(int start, int end) {
        // @@@ 0 <= start < end
        return new MakeRangeOp(start, end);
    }

    public static ExpandOp expand(int axis, TypeElement tensorType, Value v) {
        return new ExpandOp(axis, tensorType, v);
    }

    // v is scalar
    public static SplatOp splat(TypeElement tensorType, Value v) {
        return new SplatOp(tensorType, v);
    }

    // v is tensor
    public static BroadcastOp broadcast(TypeElement tensorType, Value v) {
        return new BroadcastOp(tensorType, v);
    }

    public static AddPtrOp addptr(Value ptr, Value offset) {
        return new AddPtrOp(ptr, offset);
    }

    public static LoadOp load(TypeElement tensorType, Value ptr, Value mask) {
        return new LoadOp(tensorType, ptr, mask);
    }

    public static StoreOp store(Value ptr, Value v, Value mask) {
        return new StoreOp(ptr, v, mask);
    }

    public static ReturnOp return_() {
        return new ReturnOp();
    }

    public static ReturnOp return_(Value v) {
        return new ReturnOp(v);
    }

    public static DotOp dot(TypeElement tensorType, Value a, Value b) {
        return new DotOp(tensorType, a, b);
    }


    // Operation and type factories

    public static final OpFactory FACTORY = OpFactory.OP_FACTORY.get(TritonOps.class);

    static final TypeElementFactory TRITON_TYPE_FACTORY = new TypeElementFactory() {
        @Override
        public TypeElement constructType(TypeDefinition tree) {
            return switch (tree.identifier()) {
                case PtrType.NAME -> {
                    if (tree.arguments().size() != 1) {
                        throw new IllegalArgumentException();
                    }

                    TypeElement v = TRITON_JAVA_TYPE_FACTORY.constructType(tree.arguments().getFirst());
                    if (v == null) {
                        throw new IllegalArgumentException("Bad type: " + tree);
                    }
                    if (v instanceof JavaType || v instanceof TritonType) {
                        yield new PtrType(v);
                    } else {
                        throw new IllegalArgumentException("Bad type: " + tree);
                    }
                }
                case TensorType.NAME -> {
                    if (tree.arguments().size() < 2) {
                        throw new IllegalArgumentException("Bad type: " + tree);
                    }

                    List<Integer> shape = new ArrayList<>();
                    for (int i = 0; i < tree.arguments().size() - 1; i++) {
                        TypeDefinition a = tree.arguments().get(i);
                        if (!a.identifier().startsWith("x")) {
                            throw new IllegalArgumentException("Bad type: " + tree);
                        }
                        int d;
                        try {
                            d = Integer.parseInt(a.identifier().substring(1));
                        } catch (NumberFormatException e) {
                            throw new IllegalArgumentException("Bad type: " + tree, e);
                        }
                        shape.add(d);
                    }

                    TypeElement v = TRITON_JAVA_TYPE_FACTORY.constructType(tree.arguments().getLast());
                    if (v == null) {
                        throw new IllegalArgumentException("Bad type: " + tree);
                    }
                    if (v instanceof JavaType || v instanceof TritonType) {
                        yield new TensorType(v, shape);
                    } else {
                        throw new IllegalArgumentException("Bad type: " + tree);
                    }
                }
                default -> null;
            };
        }
    };

    // Triton types then Java types
    static final TypeElementFactory TRITON_JAVA_TYPE_FACTORY =
            TRITON_TYPE_FACTORY.andThen(CoreTypeFactory.JAVA_TYPE_FACTORY);

    // Triton types then Java types, combined with code model types
    public static final TypeElementFactory TYPE_FACTORY =
            CoreTypeFactory.codeModelTypeFactory(TRITON_JAVA_TYPE_FACTORY);

}

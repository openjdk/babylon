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
import java.lang.reflect.code.descriptor.MethodTypeDesc;
import java.lang.reflect.code.descriptor.TypeDesc;
import java.lang.reflect.code.op.*;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;

public class TritonOps {
    static final TypeDesc TYPE_Ptr = TypeDesc.ofString("ptr");
    static final TypeDesc TYPE_Tensor = TypeDesc.ofString("tensor");

    static abstract class TritonOp extends OpWithDefinition {
        final TypeDesc resultType;

        public TritonOp(OpDefinition def) {
            super(def);

            this.resultType = def.resultType();
        }

        TritonOp(TritonOp that, CopyContext cc) {
            super(that, cc);

            this.resultType = that.resultType;
        }

        TritonOp(String name, TypeDesc resultType, List<? extends Value> operands) {
            super(name, operands);

            this.resultType = resultType;
        }

        @Override
        public TypeDesc resultType() {
            return resultType;
        }
    }

    @OpDeclaration(ModuleOp.NAME)
    public static final class ModuleOp extends TritonOp implements Op.Isolated {
        public static final String NAME = "module";

        final Map<String, FuncOp> table;
        final Body body;

        public ModuleOp(OpDefinition def) {
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
                } else if (op instanceof CoreOps.UnreachableOp _) {
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
            super(NAME, TypeDesc.VOID,
                    List.of());

            Body.Builder bodyC = Body.Builder.of(null, MethodTypeDesc.VOID);
            Block.Builder entryBlock = bodyC.entryBlock();
            Map<String, FuncOp> table = new HashMap<>();
            for (FuncOp f : functions) {
                entryBlock.op(f);
                table.put(f.funcName(), f);
            }
            entryBlock.op(CoreOps.unreachable());
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

    @OpDeclaration(FuncOp.NAME)
    public static final class FuncOp extends TritonOp implements Op.Invokable, Op.Isolated, Op.Lowerable {

        public static class Builder {
            final Body.Builder ancestorBody;
            final String funcName;
            final MethodTypeDesc funcDescriptor;

            Builder(Body.Builder ancestorBody, String funcName, MethodTypeDesc funcDescriptor) {
                this.ancestorBody = ancestorBody;
                this.funcName = funcName;
                this.funcDescriptor = funcDescriptor;
            }

            public FuncOp body(Consumer<Block.Builder> c) {
                Body.Builder body = Body.Builder.of(ancestorBody, funcDescriptor);
                c.accept(body.entryBlock());
                return new FuncOp(funcName, body);
            }
        }

        public static final String NAME = "tt.func";
        public static final String ATTRIBUTE_FUNC_NAME = NAME + ".name";

        final String funcName;
        final Body body;

        public static FuncOp create(OpDefinition def) {
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

        FuncOp(OpDefinition def, String funcName) {
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
            super(NAME, TypeDesc.VOID,
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
        public MethodTypeDesc funcDescriptor() {
            return body.descriptor();
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

    @OpDeclaration(CallOp.NAME)
    public static final class CallOp extends TritonOp {
        public static final String NAME = "tt.call";
        public static final String ATTRIBUTE_FUNC_NAME = NAME + ".name";

        final String funcName;

        public static CallOp create(OpDefinition def) {
            String funcName = def.extractAttributeValue(ATTRIBUTE_FUNC_NAME, true,
                    v -> switch (v) {
                        case String s -> s;
                        default -> throw new UnsupportedOperationException("Unsupported func name value:" + v);
                    });

            return new CallOp(def, funcName);
        }

        CallOp(OpDefinition def, String funcName) {
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

        CallOp(String funcName, TypeDesc resultType, List<Value> args) {
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

    @OpDeclaration(ReduceOp.NAME)
    public static final class ReduceOp extends TritonOp {
        // @@@ SSA transformation does not work with nested ops
        // implements Op.Nested {

        public static class Builder {
            final Body.Builder ancestorBody;
            final int axis;
            final Value v;
            final MethodTypeDesc reduceDescriptor;

            Builder(Body.Builder ancestorBody, int axis, Value v, MethodTypeDesc reduceDescriptor) {
                this.ancestorBody = ancestorBody;
                this.axis = axis;
                this.v = v;
                this.reduceDescriptor = reduceDescriptor;
            }

            public ReduceOp body(Consumer<Block.Builder> c) {
                Body.Builder body = Body.Builder.of(ancestorBody, reduceDescriptor);
                c.accept(body.entryBlock());
                return new ReduceOp(axis, v, body);
            }
        }

        public static final String NAME = "tt.reduce";
        public static final String ATTRIBUTE_AXIS = "axis";

        final int axis;
        final Body reducer;

        public static ReduceOp create(OpDefinition def) {
            int axis = def.extractAttributeValue(ATTRIBUTE_AXIS, true,
                    v -> switch (v) {
                        case String s -> Integer.valueOf(s);
                        case Integer i -> i;
                        default -> throw new UnsupportedOperationException("Unsupported axis value:" + v);
                    });
            return new ReduceOp(def, axis);
        }

        ReduceOp(OpDefinition def, int axis) {
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
            super(NAME, reducerBuilder.descriptor().returnType(), List.of(tensor));

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

    @OpDeclaration(ReduceReturnOp.NAME)
    public static class ReduceReturnOp extends TritonOp implements Op.Terminating {
        public static final String NAME = "tt.reduce.return";

        public ReduceReturnOp(OpDefinition def) {
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
            super(NAME, TypeDesc.VOID, List.of(r));
        }
    }

    @OpDeclaration(GetProgramIdOp.NAME)
    public static class GetProgramIdOp extends TritonOp implements Op.Pure {
        public static final String NAME = "tt.get_program_id";
        public static final String ATTRIBUTE_AXIS = "axis";

        final int axis;

        public static GetProgramIdOp create(OpDefinition def) {
            int axis = def.extractAttributeValue(ATTRIBUTE_AXIS, true,
                    v -> switch (v) {
                        case String s -> Integer.valueOf(s);
                        case Integer i -> i;
                        default -> throw new UnsupportedOperationException("Unsupported axis value:" + v);
                    });
            return new GetProgramIdOp(def, axis);
        }

        GetProgramIdOp(OpDefinition def, int axis) {
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
            super(NAME, TypeDesc.INT, List.of());

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

    @OpDeclaration(MakeRangeOp.NAME)
    public static class MakeRangeOp extends TritonOp implements Op.Pure {
        public static final String NAME = "tt.make_range";
        public static final String ATTRIBUTE_START = "start";
        public static final String ATTRIBUTE_END = "end";

        final int start;
        final int end;

        public static MakeRangeOp create(OpDefinition def) {
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

        MakeRangeOp(OpDefinition def, int start, int end) {
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

        static TypeDesc tensorType(int start, int end) {
            return new TensorType(int.class, List.of(end - start)).toDesc();
        }

        @Override
        public Map<String, Object> attributes() {
            HashMap<String, Object> m = new HashMap<>(super.attributes());
            m.put(ATTRIBUTE_START, start);
            m.put(ATTRIBUTE_END, end);
            return Collections.unmodifiableMap(m);
        }
    }

    @OpDeclaration(ExpandOp.NAME)
    public static class ExpandOp extends TritonOp implements Op.Pure {
        public static final String NAME = "tt.expand_dims";
        public static final String ATTRIBUTE_AXIS = "axis";

        final int axis;

        public static ExpandOp create(OpDefinition def) {
            int axis = def.extractAttributeValue(ATTRIBUTE_AXIS, true,
                    v -> switch (v) {
                        case String s -> Integer.valueOf(s);
                        case Integer i -> i;
                        default -> throw new UnsupportedOperationException("Unsupported axis value:" + v);
                    });
            return new ExpandOp(def, axis);
        }

        ExpandOp(OpDefinition def, int axis) {
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

        ExpandOp(int axis, TypeDesc tensorType, Value v) {
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

    @OpDeclaration(SplatOp.NAME)
    public static class SplatOp extends TritonOp implements Op.Pure {
        public static final String NAME = "tt.splat";

        public SplatOp(OpDefinition def) {
            super(def);
        }

        SplatOp(SplatOp that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public SplatOp transform(CopyContext cc, OpTransformer ot) {
            return new SplatOp(this, cc);
        }

        SplatOp(TypeDesc tensorType, Value v) {
            super(NAME, tensorType, List.of(v));
        }
    }

    @OpDeclaration(BroadcastOp.NAME)
    public static class BroadcastOp extends TritonOp implements Op.Pure {
        public static final String NAME = "tt.broadcast";

        public BroadcastOp(OpDefinition def) {
            super(def);
        }

        BroadcastOp(BroadcastOp that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public BroadcastOp transform(CopyContext cc, OpTransformer ot) {
            return new BroadcastOp(this, cc);
        }

        BroadcastOp(TypeDesc tensorType, Value v) {
            super(NAME, tensorType, List.of(v));
        }
    }

    @OpDeclaration(AddPtrOp.NAME)
    public static class AddPtrOp extends TritonOp implements Op.Pure {
        public static final String NAME = "tt.addptr";

        public AddPtrOp(OpDefinition def) {
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

    @OpDeclaration(LoadOp.NAME)
    public static class LoadOp extends TritonOp implements Op.Pure {
        public static final String NAME = "tt.load";

        public LoadOp(OpDefinition def) {
            super(def);
        }

        LoadOp(LoadOp that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public LoadOp transform(CopyContext cc, OpTransformer ot) {
            return new LoadOp(this, cc);
        }

        LoadOp(TypeDesc tensorType, Value ptr, Value mask) {
            super(NAME, tensorType, List.of(ptr, mask));
        }
    }

    @OpDeclaration(StoreOp.NAME)
    public static class StoreOp extends TritonOp {
        public static final String NAME = "tt.store";

        public StoreOp(OpDefinition def) {
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
            super(NAME, TypeDesc.VOID, List.of(ptr, v, mask));
        }
    }

    @OpDeclaration(ReturnOp.NAME)
    public static class ReturnOp extends TritonOp implements Op.Terminating {
        public static final String NAME = "tt.return";

        public ReturnOp(OpDefinition def) {
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
            super(NAME, TypeDesc.VOID, List.of());
        }

        ReturnOp(Value v) {
            super(NAME, TypeDesc.VOID, List.of(v));
        }
    }

    @OpDeclaration(DotOp.NAME)
    public static class DotOp extends TritonOp implements Op.Pure {
        public static final String NAME = "tt.dot";

        public DotOp(OpDefinition def) {
            super(def);
        }

        DotOp(DotOp that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public DotOp transform(CopyContext cc, OpTransformer ot) {
            return new DotOp(this, cc);
        }

        DotOp(TypeDesc tensorType, Value a, Value b) {
            super(NAME, tensorType, List.of(a, b));
        }
    }


    public static ModuleOp module(FuncOp... functions) {
        return module(List.of(functions));
    }

    public static ModuleOp module(List<FuncOp> functions) {
        return new ModuleOp(List.copyOf(functions));
    }

    public static FuncOp.Builder func(String funcName, MethodTypeDesc funcDescriptor) {
        return new FuncOp.Builder(null, funcName, funcDescriptor);
    }

    public static FuncOp func(String funcName, Body.Builder body) {
        return new FuncOp(funcName, body);
    }

    public static CallOp call(FuncOp func, Value... args) {
        return call(func, List.of(args));
    }

    public static CallOp call(FuncOp func, List<Value> args) {
        return new CallOp(func.funcName(), func.funcDescriptor().returnType(), args);
    }

    public static ReduceOp.Builder reduce(Body.Builder ancestorBody, int axis, Value tensor,
                                          MethodTypeDesc reduceDescriptor) {
        return new ReduceOp.Builder(ancestorBody, axis, tensor, reduceDescriptor);
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


    public static final OpFactory FACTORY = OpFactory.OP_FACTORY.get(TritonOps.class);

    public static ExpandOp expand(int axis, TypeDesc tensorType, Value v) {
        return new ExpandOp(axis, tensorType, v);
    }

    // v is scalar
    public static SplatOp splat(TypeDesc tensorType, Value v) {
        return new SplatOp(tensorType, v);
    }

    // v is tensor
    public static BroadcastOp broadcast(TypeDesc tensorType, Value v) {
        return new BroadcastOp(tensorType, v);
    }

    public static AddPtrOp addptr(Value ptr, Value offset) {
        return new AddPtrOp(ptr, offset);
    }

    public static LoadOp load(TypeDesc tensorType, Value ptr, Value mask) {
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

    public static DotOp dot(TypeDesc tensorType, Value a, Value b) {
        return new DotOp(tensorType, a, b);
    }
}

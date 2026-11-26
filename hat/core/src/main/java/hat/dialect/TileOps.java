/*
 * Copyright (c) 2026, Oracle and/or its affiliates. All rights reserved.
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
package hat.dialect;

import jdk.incubator.code.AbstractOp;
import jdk.incubator.code.Block;
import jdk.incubator.code.Body;
import jdk.incubator.code.CodeContext;
import jdk.incubator.code.CodeTransformer;
import jdk.incubator.code.CodeType;
import jdk.incubator.code.Op;
import jdk.incubator.code.Value;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.core.CoreType;
import jdk.incubator.code.dialect.java.JavaType;
import jdk.incubator.code.extern.ExternalizedOp;
import optkl.util.ops.Precedence;

import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Internal class to represent custom Ops to be inserted in the resulting code model that implements a Tile Programming Model.
 */
public class TileOps {

    private TileOps() {
    }

    public static ModuleOp module(List<CoreOp.FuncOp> list) {
        return new ModuleOp(List.copyOf(list));
    }

    public static Op bid(int dimension) {
        return new TilePidOp(dimension);
    }

    public static Op load(CodeType type, Value ptr, Value dimension, Value shape, List<Integer> dims) {
        return new LoadOp(type, ptr, dimension, shape, dims);
    }

    public static Op store(Value ptr, Value id, Value tensor, List<Integer> dims) {
        return new StoreOp(ptr, id, tensor, dims);
    }

    public static Op index(Value... values) {
        return new TileIndexOp(values);
    }

    public static Op indices(Value... values) {
        return new TileIndicesOp(values);
    }

    public static Op shape(Value... values) {
        return new TileShapeOp(values);
    }

    public static Op dim(Value... dims) {
        return new TileDimMapOp(dims);
    }

    public static Op numTiles(CodeType typeResult, Value ptr, Value dimension, Value shape) {
        return new TileNumOp(typeResult, ptr, dimension, shape);
    }

    public static Op full(CodeType type, Value shapeValue, Value initValue) {
        return new TileFullOp(type, shapeValue, initValue);
    }

    public static Op sum(CodeType typeResult, Value tensor, Value dimension) {
        return new TileSumOp(typeResult, tensor, dimension);
    }

    public static Op zeros(CodeType typeResult, Value... tileShape) {
        return new TileZerosOp(typeResult, tileShape);
    }

    public static Op ones(CodeType typeResult, Value... tileShape) {
        return new TileOnesOp(typeResult, tileShape);
    }

    public static Op arange(CodeType typeResult, Value value) {
        return new TileArangeOp(typeResult, JavaType.INT, value);
    }

    public static Op arange(CodeType typeResult, CodeType dType, Value... args) {
        return new TileArangeOp(typeResult, dType, args);
    }

    public static Op toType(CodeType typeResult, Value tensor) {
        return new AsTypeOp(typeResult, tensor);
    }

    public static Op irange(CodeType resultType, Value startValue, Value endValue) {
        return new TileIrangeOp(resultType, startValue, endValue);
    }

    public static Op convert(CodeType resultType, Value tensor) {
        return new TileConvOp(resultType, tensor);
    }

    public static Op gather(CodeType typeResult, Value ptrValueOffset) {
        return new GatherOp(typeResult, ptrValueOffset);
    }

    public static Op scatter(Value ptrValue, Value tile) {
        return new ScatterOp(ptrValue, tile);
    }

    public static Op offsetIndex(CodeType typeResult, Value ptrValue, Value indexTile) {
        return new OffsetOp(typeResult, ptrValue, indexTile);
    }

    public abstract static class TOp extends AbstractOp implements ExternalizedOp.Externalizable {

        final CodeType resultType;
        final String opName;

        protected TOp(ExternalizedOp def) {
            super(def.operands());
            this.opName = def.name();
            this.resultType = def.resultType();
        }

        TOp(TOp that, CodeContext cc) {
            super(that, cc);
            this.opName = that.opName;
            this.resultType = that.resultType;
        }

        TOp(String name, CodeType resultType, List<? extends Value> operands) {
            super(operands);
            this.opName = name;
            this.resultType = resultType;
        }

        @Override
        public CodeType resultType() {
            return resultType;
        }

        @Override
        public String externalizeOpName() {
            return opName;
        }
    }

    // Intermediate class to pack all related TileOps that needs to be processed
    // in a HAT Tile Transformer. If a HAT Tile Transformer is not used, then
    // we can extend directly from TOp
    public abstract static class TileContextOp extends TOp {

        protected TileContextOp(ExternalizedOp def) {
            super(def);
        }

        TileContextOp(TOp that, CodeContext cc) {
            super(that, cc);
        }

        TileContextOp(String name, CodeType resultType, List<? extends Value> operands) {
            super(name, resultType, operands);
        }

    }

    public abstract static class TileCreateOp extends TileContextOp {

        TileCreateOp(TOp that, CodeContext cc) {
            super(that, cc);
        }

        TileCreateOp(String name, CodeType resultType, List<? extends Value> operands) {
            super(name, resultType, operands);
        }
    }

    public static final class ModuleOp extends TOp implements Op.Isolated {

        public static final String NAME = "module";

        final Map<String, CoreOp.FuncOp> table;
        final Body body;

        ModuleOp(ExternalizedOp def) {
            super(def);
            this.body = def.bodyDefinitions().getFirst().build(this);
            this.table = createTable(body);
        }

        ModuleOp(ModuleOp that, CodeContext cc, CodeTransformer ot) {
            super(that, cc);

            this.body = that.body.transform(cc, ot).build(this);
            this.table = createTable(body);
        }

        ModuleOp(List<CoreOp.FuncOp> functions) {

            super(NAME, JavaType.VOID, List.of());

            Body.Builder bodyC = Body.Builder.of(null, CoreType.FUNCTION_TYPE_VOID);
            Block.Builder entryBlock = bodyC.entryBlock();
            Map<String, CoreOp.FuncOp> sTable = new HashMap<>();
            for (CoreOp.FuncOp func : functions) {
                entryBlock.add(func);
                sTable.put(func.funcName(), func);
            }

            entryBlock.add(CoreOp.unreachable());
            this.table = Collections.unmodifiableMap(sTable);
            this.body = bodyC.build(this);
        }

        static Map<String, CoreOp.FuncOp> createTable(Body body) {
            Map<String, CoreOp.FuncOp> table = new LinkedHashMap<>();
            for (var op : body.entryBlock().ops()) {
                if (op instanceof CoreOp.FuncOp funcOp) {
                    table.put(funcOp.funcName(), funcOp);
                } else if (op instanceof CoreOp.UnreachableOp) {
                    // no operation
                } else {
                    throw new IllegalArgumentException("Unknown op: " + op);
                }
            }
            return Collections.unmodifiableMap(table);
        }

        @Override
        public Op transform(CodeContext codeContext, CodeTransformer codeTransformer) {
            return new ModuleOp(this, codeContext, codeTransformer);
        }

        @Override
        public List<Body> bodies() {
            return List.of(body);
        }

        public Map<String, CoreOp.FuncOp> table() {
            return table;
        }
    }

    public static class TilePidOp extends TOp implements Precedence.Invoke {
        public static final String NAME = "tile.pid";
        final int dimension;

        public static TilePidOp create(ExternalizedOp def) {
            // Note: it seems we need to have a contract between the parameter name and the following attribute name
            Object v = getDefaultAttributeValue(def, "dimension");
            int dimension = switch (v) {
                case Integer i -> i;
                case null, default -> throw new IllegalArgumentException("Invalid dimension: " + v);
            };
            return new TilePidOp(dimension);
        }

        protected TilePidOp(int dimensions) {
            super(NAME, JavaType.INT, List.of());
            this.dimension = dimensions;
        }

        protected TilePidOp(TilePidOp that, CodeContext cc, CodeTransformer ot) {
            super(that, cc);
            this.dimension = that.dimension;
        }

        @Override
        public TilePidOp transform(CodeContext codeContext, CodeTransformer codeTransformer) {
            return new TilePidOp(this, codeContext, codeTransformer);
        }

        public int dimension() {
            return dimension;
        }

        @Override
        public Map<String, Object> externalize() {
            return Map.of("dimension", dimension);
        }

    }

    public static class TileIndexOp extends TOp {
        public static final String NAME = "tile.index";
        final int dimension;

        protected TileIndexOp(Value... values) {
            super(NAME, JavaType.INT, Arrays.stream(values).toList());
            this.dimension = values.length;
        }

        protected TileIndexOp(TileIndexOp that, CodeContext cc, CodeTransformer ot) {
            super(that, cc);
            this.dimension = that.dimension;
        }

        @Override
        public TileIndexOp transform(CodeContext codeContext, CodeTransformer codeTransformer) {
            return new TileIndexOp(this, codeContext, codeTransformer);
        }

        public int dimension() {
            return dimension;
        }

        @Override
        public Map<String, Object> externalize() {
            return Map.of("dimension", dimension);
        }
    }

    public static class TileIndicesOp extends TOp {
        public static final String NAME = "tile.indices";

        protected TileIndicesOp(Value... values) {
            super(NAME, JavaType.INT, Arrays.stream(values).toList());
        }

        protected TileIndicesOp(TileIndicesOp that, CodeContext cc, CodeTransformer ot) {
            super(that, cc);
        }

        @Override
        public TileIndicesOp transform(CodeContext codeContext, CodeTransformer codeTransformer) {
            return new TileIndicesOp(this, codeContext, codeTransformer);
        }

        public int dimension() {
            return operands().size();
        }

    }

    public static class TileFullOp extends TileCreateOp implements Precedence.Invoke {
        public static final String NAME = "tile.full";
        protected TileFullOp(CodeType type, Value shapeValue, Value initValue) {
            super(NAME, type, List.of(shapeValue, initValue));
        }

        protected TileFullOp(TileFullOp that, CodeContext cc, CodeTransformer ot) {
            super(that, cc);
        }

        @Override
        public TileFullOp transform(CodeContext codeContext, CodeTransformer codeTransformer) {
            return new TileFullOp(this, codeContext, codeTransformer);
        }
    }

    public static class TileZerosOp extends TileCreateOp implements Precedence.Invoke {
        public static final String NAME = "tile.zeros";
        protected TileZerosOp(CodeType type, Value... shapes) {
            super(NAME, type, Arrays.stream(shapes).toList());
        }

        protected TileZerosOp(TileZerosOp that, CodeContext cc, CodeTransformer ot) {
            super(that, cc);
        }

        @Override
        public TileZerosOp transform(CodeContext codeContext, CodeTransformer codeTransformer) {
            return new TileZerosOp(this, codeContext, codeTransformer);
        }
    }

    public static class TileOnesOp extends TileCreateOp implements Precedence.Invoke {
        public static final String NAME = "tile.ones";
        protected TileOnesOp(CodeType type, Value... shapes) {
            super(NAME, type, Arrays.stream(shapes).toList());
        }

        protected TileOnesOp(TileOnesOp that, CodeContext cc, CodeTransformer ot) {
            super(that, cc);
        }

        @Override
        public TileOnesOp transform(CodeContext codeContext, CodeTransformer codeTransformer) {
            return new TileOnesOp(this, codeContext, codeTransformer);
        }
    }

    public static class TileArangeOp extends TileCreateOp implements Precedence.Invoke {
        public static final String NAME = "tile.arange";
        private final CodeType dType;

        protected TileArangeOp(CodeType type, CodeType dtype, Value... args) {
            super(NAME, type, List.of(args));
            this.dType = dtype;
        }

        protected TileArangeOp(TileArangeOp that, CodeContext cc, CodeTransformer ot) {
            super(that, cc);
            this.dType = that.dType;
        }

        @Override
        public TileArangeOp transform(CodeContext codeContext, CodeTransformer codeTransformer) {
            return new TileArangeOp(this, codeContext, codeTransformer);
        }

        public CodeType dType() {
            return dType;
        }
    }

    public static class TileIrangeOp extends TOp {
        public static final String NAME = "tile.irange";
        protected TileIrangeOp(CodeType type, Value startIndex, Value endIndex) {
            super(NAME, type, List.of(startIndex, endIndex));
        }

        protected TileIrangeOp(TileIrangeOp that, CodeContext cc, CodeTransformer ot) {
            super(that, cc);
        }

        @Override
        public TileIrangeOp transform(CodeContext codeContext, CodeTransformer codeTransformer) {
            return new TileIrangeOp(this, codeContext, codeTransformer);
        }
    }

    public static class TileConvOp extends TileContextOp implements Precedence.Invoke {
        public static final String NAME = "tile.conv";
        protected TileConvOp(CodeType type, Value tensor) {
            super(NAME, type, List.of(tensor));
        }

        protected TileConvOp(TileConvOp that, CodeContext cc, CodeTransformer ot) {
            super(that, cc);
        }

        @Override
        public TileConvOp transform(CodeContext codeContext, CodeTransformer codeTransformer) {
            return new TileConvOp(this, codeContext, codeTransformer);
        }
    }

    public static class AsTypeOp extends TOp {
        public static final String NAME = "tile.astType";
        protected AsTypeOp(CodeType type, Value size) {
            super(NAME, type, List.of(size));
        }

        protected AsTypeOp(AsTypeOp that, CodeContext cc, CodeTransformer ot) {
            super(that, cc);
        }

        @Override
        public AsTypeOp transform(CodeContext codeContext, CodeTransformer codeTransformer) {
            return new AsTypeOp(this, codeContext, codeTransformer);
        }
    }

    public static class TileShapeOp extends TOp {
        public static final String NAME = "tile.shape";
        final int dimension;

        protected TileShapeOp(Value... values) {
            super(NAME, JavaType.INT, Arrays.stream(values).toList());
            this.dimension = values.length;
        }

        protected TileShapeOp(TileShapeOp that, CodeContext cc, CodeTransformer ot) {
            super(that, cc);
            this.dimension = that.dimension;
        }

        @Override
        public TileShapeOp transform(CodeContext codeContext, CodeTransformer codeTransformer) {
            return new TileShapeOp(this, codeContext, codeTransformer);
        }

        public int dimension() {
            return dimension;
        }
    }

    public static class TileDimMapOp extends TOp {
        public static final String NAME = "tile.dimMap";
        final int dimension;

        protected TileDimMapOp(Value... values) {
            super(NAME, JavaType.INT, Arrays.stream(values).toList());
            this.dimension = values.length;
        }

        protected TileDimMapOp(TileDimMapOp that, CodeContext cc, CodeTransformer ot) {
            super(that, cc);
            this.dimension = that.dimension;
        }

        @Override
        public TileDimMapOp transform(CodeContext codeContext, CodeTransformer codeTransformer) {
            return new TileDimMapOp(this, codeContext, codeTransformer);
        }

        public int dimension() {
            return dimension;
        }

    }

    public static class TileNumOp extends TileContextOp implements Precedence.Invoke {
        public static final String NAME = "tile.numTiles";
        TileNumOp(TileNumOp that, CodeContext cc) {
            super(that, cc);
        }

        @Override
        public TileNumOp transform(CodeContext cc, CodeTransformer ot) {
            return new TileNumOp(this, cc);
        }

        TileNumOp(CodeType codeType, Value ptr, Value dimension, Value shape) {
            super(NAME, codeType, List.of(ptr, dimension, shape));
        }
    }

    public static class TileSumOp extends TileContextOp implements Precedence.Invoke {
        public static final String NAME = "tile.sum";

        TileSumOp(TileSumOp that, CodeContext cc) {
            super(that, cc);
        }

        @Override
        public TileSumOp transform(CodeContext cc, CodeTransformer ot) {
            return new TileSumOp(this, cc);
        }

        TileSumOp(CodeType tensorType, Value tensor, Value dimension) {
            super(NAME, tensorType, List.of(tensor, dimension));
        }
    }

    public static class LoadOp extends TileContextOp implements Precedence.LoadOrConv {
        public static final String NAME = "tile.load";

        private final List<Integer> dims;

        LoadOp(LoadOp that, CodeContext cc) {
            super(that, cc);
            this.dims = that.dims;
        }

        @Override
        public LoadOp transform(CodeContext cc, CodeTransformer ot) {
            return new LoadOp(this, cc);
        }

        LoadOp(CodeType tensorType, Value ptr, Value mask, Value other, List<Integer> dims) {
            super(NAME, tensorType, List.of(ptr, mask, other));
            this.dims = dims;
        }

        public List<Integer> dims() {
            return dims;
        }
    }

    public static class StoreOp extends TOp {
        public static final String NAME = "tile.store";

        private final List<Integer> dims;

        StoreOp(StoreOp that, CodeContext cc) {
            super(that, cc);
            this.dims = that.dims;
        }

        @Override
        public StoreOp transform(CodeContext cc, CodeTransformer ot) {
            return new StoreOp(this, cc);
        }

        StoreOp(Value ptr, Value id, Value tensor, List<Integer> dims) {
            super(NAME, JavaType.VOID, List.of(ptr, id, tensor));
            this.dims = dims;
        }

        public List<Integer> dims() {
            return dims;
        }
    }

    public static class GatherOp extends TileContextOp implements Precedence.LoadOrConv {
        public static final String NAME = "tile.gather";

        GatherOp(GatherOp that, CodeContext cc) {
            super(that, cc);
        }

        @Override
        public GatherOp transform(CodeContext cc, CodeTransformer ot) {
            return new GatherOp(this, cc);
        }

        GatherOp(CodeType tensorType, Value ptr) {
            super(NAME, tensorType, List.of(ptr));
        }
    }

    public static class OffsetOp extends TileContextOp implements Precedence.LoadOrConv {
        public static final String NAME = "tile.offset";

        OffsetOp(OffsetOp that, CodeContext cc) {
            super(that, cc);
        }

        @Override
        public OffsetOp transform(CodeContext cc, CodeTransformer ot) {
            return new OffsetOp(this, cc);
        }

        OffsetOp(CodeType tensorType, Value ptr, Value index) {
            super(NAME, tensorType, List.of(ptr, index));
        }
    }

    public static class ScatterOp extends TOp {
        public static final String NAME = "tile.scatter";

        ScatterOp(ScatterOp that, CodeContext cc) {
            super(that, cc);
        }

        @Override
        public ScatterOp transform(CodeContext cc, CodeTransformer ot) {
            return new ScatterOp(this, cc);
        }

        ScatterOp(Value ptr, Value tile) {
            super(NAME, JavaType.VOID, List.of(ptr, tile));
        }
    }


    static Object getDefaultAttributeValue(ExternalizedOp def, String attributeName) {
        var attr = def.attributes();
        return attr.containsKey("") ? attr.get("") : attr.get(attributeName);
    }
}

package hat.dialect;

import hat.codetypes.ConstantType;
import hat.codetypes.TensorType;
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
import jdk.incubator.code.dialect.core.FunctionType;
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

    private TileOps() {}

    public static ModuleOp module(List<CoreOp.FuncOp> list) {
        return new ModuleOp(List.copyOf(list));
    }

    public static Op bid(int dimension) {
        return new TileIDOp(dimension);
    }

    public static Op load(CodeType type, Value ptr, Value dimension, Value shape) {
        return new LoadOp(type, ptr, dimension, shape);
    }

    public static Op store(Value ptr, Value id, Value tensor) {
        return new StoreOp(ptr, id, tensor);
    }

    public static Op index(Value... values) {
        return new TileIndexOp(values);
    }

    public static Op shape(Value index0) {
        return new TileShapeOp(index0);
    }

    public static Op shape(Value index0, Value index1) {
        return new TileShapeOp(index0, index1);
    }

    public static Op shape(Value index0, Value index1, Value index2) {
        return new TileShapeOp(index0, index1, index2);
    }

    public static Op numTiles(Value ptr, Value dimension, Value shape) {
        return new TileNumOp(ptr, dimension, shape);
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

    public static Op arange(CodeType typeResult, Value value) {
        return new TileArangeOp(typeResult, value);
    }

    public static Op arange(CodeType typeResult, Value value, Value start, Value step) {
        return new TileArangeOp(typeResult, value, start, step);
    }

    public static Op toType(CodeType typeResult, Value tensor) {
        return new AsTypeOp(typeResult, tensor);
    }

    public abstract static class TOp extends AbstractOp implements ExternalizedOp.Externalizable {

        final CodeType resultType;

        protected TOp(ExternalizedOp def) {
            super(def.operands());
            this.resultType = def.resultType();
        }

        TOp(TOp that, CodeContext cc) {
            super(that, cc);
            this.resultType = that.resultType;
        }

        TOp(CodeType resultType, List<? extends Value> operands) {
            super(operands);
            this.resultType = resultType;
        }

        @Override
        public CodeType resultType() {
            return resultType;
        }

        @Override
        public String externalizeOpName() {
            return "Tile-Op";
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

            super(JavaType.VOID, List.of());

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

    public static class TileIDOp extends TOp implements Op.Pure, Precedence.Invoke {

        final int dimension;

        public static TileIDOp create(ExternalizedOp def) {
            // Note: it seems we need to have a contract between the parameter name and the following attribute name
            Object v = getDefaultAttributeValue(def, "dimension");
            int dimension = switch (v) {
                case Integer i -> i;
                case null, default -> throw new IllegalArgumentException("Invalid dimension: " + v);
            };
            return new TileIDOp(dimension);
        }

        protected TileIDOp(int dimensions) {
            super(JavaType.INT, List.of());
            this.dimension = dimensions;
        }

        protected TileIDOp(TileIDOp that, CodeContext cc, CodeTransformer ot) {
            super(that, cc);
            this.dimension = that.dimension;
        }

        protected TileIDOp(TileIDOp that, CodeContext cc) {
            super(that, cc);
            this.dimension = that.dimension;
        }

        @Override
        public TileIDOp transform(CodeContext codeContext, CodeTransformer codeTransformer) {
            return new TileIDOp(this, codeContext, codeTransformer);
        }

        public int dimension() {
            return dimension;
        }

        @Override
        public Map<String, Object> externalize() {
            return Map.of("tile.pid ##### ", dimension);
        }
    }

    public static class TileIndexOp extends TOp implements Op.Pure {

        final int dimension;

        protected TileIndexOp(Value... values) {
            super(JavaType.INT, Arrays.stream(values).toList());
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
            return Map.of("tile.index ", dimension);
        }
    }

    public static class TileFullOp extends TOp implements Op.Pure, Precedence.Invoke{

        protected TileFullOp(CodeType type, Value shapeValue, Value initValue) {
            super(type, List.of(shapeValue, initValue));
        }

        protected TileFullOp(TileFullOp that, CodeContext cc, CodeTransformer ot) {
            super(that, cc);
        }

        @Override
        public TileFullOp transform(CodeContext codeContext, CodeTransformer codeTransformer) {
            return new TileFullOp(this, codeContext, codeTransformer);
        }

        @Override
        public Map<String, Object> externalize() {
            return Map.of("tile.full ", operands().get(1));
        }
    }

    public static class TileZerosOp extends TOp implements Op.Pure {

        protected TileZerosOp(CodeType type, Value... shapes) {
            super(type, Arrays.stream(shapes).toList());
        }

        protected TileZerosOp(TileZerosOp that, CodeContext cc, CodeTransformer ot) {
            super(that, cc);
        }

        @Override
        public TileZerosOp transform(CodeContext codeContext, CodeTransformer codeTransformer) {
            return new TileZerosOp(this, codeContext, codeTransformer);
        }

        @Override
        public Map<String, Object> externalize() {
            StringBuilder builder = new StringBuilder();
            if (resultType instanceof ConstantType constantType) {
                Object v = constantType.value();
                // NOTE: The values become the propagate shape, while the codeType represent the element type of
                // the tensor operation (e.g., Tile).
                if (v instanceof TensorType tensorType) {
                    builder.append("shape: ").append(Arrays.toString(tensorType.shape().toArray()));
                }
            }
            return Map.of("tile.zeros ", builder.toString());
        }
    }

    public static class TileArangeOp extends TOp implements Op.Pure {

        protected TileArangeOp(CodeType type, Value size) {
            super(type, List.of(size));
        }

        protected TileArangeOp(CodeType type, Value size, Value start, Value step) {
            super(type, List.of(size, start, step));
        }

        protected TileArangeOp(TileArangeOp that, CodeContext cc, CodeTransformer ot) {
            super(that, cc);
        }

        @Override
        public TileArangeOp transform(CodeContext codeContext, CodeTransformer codeTransformer) {
            return new TileArangeOp(this, codeContext, codeTransformer);
        }

        @Override
        public Map<String, Object> externalize() {
            return Map.of("tile.arange ", operands().get(0));
        }
    }

    public static class AsTypeOp extends TOp implements Op.Pure {

        protected AsTypeOp(CodeType type, Value size) {
            super(type, List.of(size));
        }

        protected AsTypeOp(AsTypeOp that, CodeContext cc, CodeTransformer ot) {
            super(that, cc);
        }

        @Override
        public AsTypeOp transform(CodeContext codeContext, CodeTransformer codeTransformer) {
            return new AsTypeOp(this, codeContext, codeTransformer);
        }

        @Override
        public Map<String, Object> externalize() {
            return Map.of("tile.asType ", operands().get(0));
        }
    }

    public static class TileShapeOp extends TOp implements Op.Pure {

        final int dimension;

        protected TileShapeOp(Value... values) {
            super(JavaType.INT, Arrays.stream(values).toList());
            this.dimension = values.length;
        }

        protected TileShapeOp(TileShapeOp that, CodeContext cc, CodeTransformer ot) {
            super(that, cc);
            this.dimension = that.dimension;
        }

        protected TileShapeOp(TileShapeOp that, CodeContext cc) {
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

        @Override
        public Map<String, Object> externalize() {
            return Map.of("tile.shape ", dimension);
        }
    }

    public static class TileNumOp extends TOp implements Op.Pure, Precedence.Invoke {

        public TileNumOp(ExternalizedOp def) {
            super(def);
        }

        TileNumOp(TileNumOp that, CodeContext cc) {
            super(that, cc);
        }

        @Override
        public TileNumOp transform(CodeContext cc, CodeTransformer ot) {
            return new TileNumOp(this, cc);
        }

        TileNumOp(Value ptr, Value dimension, Value shape) {
            super(JavaType.INT, List.of(ptr, dimension, shape));
        }

        @Override
        public Map<String, Object> externalize() {
            return Map.of("tile.numTile  ", operands().get(1)); // with dimension
        }
    }

    public static class TileSumOp extends TOp implements Op.Pure, Precedence.Invoke {

        public TileSumOp(ExternalizedOp def) {
            super(def);
        }

        TileSumOp(TileSumOp that, CodeContext cc) {
            super(that, cc);
        }

        @Override
        public TileSumOp transform(CodeContext cc, CodeTransformer ot) {
            return new TileSumOp(this, cc);
        }

        TileSumOp(CodeType tensorType, Value tensor, Value dimension) {
            super(tensorType, List.of(tensor, dimension));
        }

        @Override
        public Map<String, Object> externalize() {
            return Map.of("tile.sum  ", resultType);
        }
    }

    public static class LoadOp extends TOp implements Op.Pure, Precedence.LoadOrConv {

        public LoadOp(ExternalizedOp def) {
            super(def);
        }

        LoadOp(LoadOp that, CodeContext cc) {
            super(that, cc);
        }

        @Override
        public LoadOp transform(CodeContext cc, CodeTransformer ot) {
            return new LoadOp(this, cc);
        }

        LoadOp(CodeType tensorType, Value ptr, Value dimension, Value shape) {
            super(tensorType, List.of(ptr, dimension, shape));
        }

        @Override
        public Map<String, Object> externalize() {
            return Map.of("tile.load  ", resultType);
        }
    }

    public static class StoreOp extends TOp implements Precedence.Store {

        public StoreOp(ExternalizedOp def) {
            super(def);
        }

        StoreOp(StoreOp that, CodeContext cc) {
            super(that, cc);
        }

        @Override
        public StoreOp transform(CodeContext cc, CodeTransformer ot) {
            return new StoreOp(this, cc);
        }

        StoreOp(Value ptr, Value id, Value tensor) {
            super(JavaType.VOID, List.of(ptr, id, tensor));
        }

        @Override
        public Map<String, Object> externalize() {
            StringBuilder builder = new StringBuilder();
            Value tensor = operands().getLast();
            if (tensor.type() instanceof ConstantType constantType) {
                Object v = constantType.value();
                // NOTE: The values become the propagate shape, while the codeType represent the element type of
                // the tensor operation (e.g., Tile).
                if (v instanceof TensorType tensorType) {
                    builder.append("shape: ").append(Arrays.toString(tensorType.shape().toArray()));
                }
            }
            builder.append(resultType);
            return Map.of("tile.store  ", builder.toString());
        }
    }

    static Object getDefaultAttributeValue(ExternalizedOp def, String attributeName) {
        var attr = def.attributes();
        return attr.containsKey("") ? attr.get("") : attr.get(attributeName);
    }
}

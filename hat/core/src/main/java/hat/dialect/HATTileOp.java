package hat.dialect;

import jdk.incubator.code.CodeContext;
import jdk.incubator.code.CodeTransformer;
import jdk.incubator.code.CodeType;
import jdk.incubator.code.Op;
import jdk.incubator.code.Value;
import jdk.incubator.code.dialect.java.JavaType;
import optkl.util.ops.Precedence;

import java.util.List;
import java.util.Map;

public abstract sealed class HATTileOp extends HATOp implements Dim, Precedence.LoadOrConv {

    protected HATTileOp(List<Value> operands) {
        super(operands);
    }

    protected HATTileOp(HATTileOp that, CodeContext cc) {
        super(that, cc);
    }

    @Override
    public final CodeType resultType() {
        return JavaType.INT;
    }

    public static HATTileOp create(String name, int dimension) {
        return switch (name) {
            case "bid" -> new HATTileOp.Bid(dimension);
            default -> throw new IllegalStateException("[ERROR] Illegal/unsupported parallel construct: " + name);
        };
    }

    public static final class Bid extends HATTileOp {

        private final int dimension;

        public Bid(Bid op, CodeContext copyContext) {
            super(op, copyContext);
            this.dimension = op.dimension;
        }

        public Bid(int dimension) {
            super(List.of());
            this.dimension = dimension;
        }

        private int getDimension() {
            return dimension;
        }

        @Override
        public Op transform(CodeContext copyContext, CodeTransformer opTransformer) {
            return new Bid(this, copyContext);
        }

        @Override
        public Map<String, Object> externalize() {return Map.of("hat.dialect.Bid#"+dimension, JavaType.INT);
        }
    }

}
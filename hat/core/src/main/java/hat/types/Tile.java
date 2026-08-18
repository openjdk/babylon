package hat.types;

import jdk.incubator.code.CodeType;
import optkl.IfaceValue;

/**
 * Data type to represent views over tensors (tiles) within the compute kernels.
 */
public record Tile() implements IfaceValue {

    /**
     * Transpose an input tile (swap first and second dimensions of the tile shape.
     * @return
     *    Transposed tile
     */
    public Tile transpose() {
        return new Tile();
    }

    /**
     * Transform an input tile to type dType.
     *
     * @param dType
     *    Type to be transformed
     * @return
     *    A new Tile of type dType.
     */
    public Tile asType(CodeType dType) {
        return new Tile();
    }
}

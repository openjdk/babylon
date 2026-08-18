package hat;

import hat.buffer.TensorF32;
import hat.types.Tile;
import jdk.incubator.code.CodeType;


/**
 * Class to represent common tile math operations (add, sub, transpose, mma, etc).
 */
public class TileOp {

    public static Tile add(Tile aTile, Tile bTile) {
        return null;
    }

    public static <T extends Number> Tile add(Tile aTile, T val) {
        return null;
    }

    public static <T extends Number> Tile add(T val, Tile aTile) {
        return null;
    }

    public static Tile sub(Tile aTile, Tile bTile) {
        return null;
    }

    public static <T extends Number> Tile sub(Tile aTile, T val) {
        return null;
    }

    public static <T extends Number> Tile sub(T val, Tile aTile) {
        return null;
    }

    public static Tile mul(Tile aTile, Tile bTile) {
        return null;
    }

    public static <T extends Number> Tile mul(Tile aTile, T val) {
        return null;
    }

    public static <T extends Number> Tile mul(T val, Tile aTile) {
        return null;
    }

    /**
     * true division aTile / bTile
     *
     * @param aTile
     * @param bTile
     * @return
     */
    public static Tile truediv(Tile aTile, Tile bTile) {
        return null;
    }

    public static <T extends Number> Tile truediv(Tile aTile, T val) {
        return null;
    }

    public static <T extends Number> Tile truediv(T val, Tile aTile) {
        return null;
    }

    /**
     * Ceil division (aTile / bTile)
     *
     * @param aTile
     * @param bTile
     * @return
     */
    public static Tile ceildiv(Tile aTile, Tile bTile) {
        return null;
    }

    public static <T extends Number> Tile ceildiv(Tile aTile, T val) {
        return null;
    }

    public static <T extends Number> Tile ceildiv(T val, Tile aTile) {
        return null;
    }

    public static Tile transpose(Tile aTile) {
        return null;
    }

    /**
     * Reshape an input tile to another shape
     *
     * @param aTile
     * @param aShape
     * @return
     */
    public static Tile reshape(Tile aTile, Shape aShape) {
        return null;
    }

    public static Tile sum(Tile aTile, int axis) {
        return null;
    }

    public static Tile min(Tile aTile, int axis) {
        return null;
    }

    public static Tile max(Tile aTile, int axis) {
        return null;
    }

    public static int numTiles(TensorF32 input, int dimension, int tileSize) {
        return 0;
    }

    public static int numTiles(TensorF32 input, int dimension, Shape shape) {
        return 0;
    }

    public static Tile full(Shape shape, float value) {
        return null;
    }

    public static Tile mma(Tile tileA, Tile tileB, Tile accumulator) {
        return null;
    }

    public static Tile zeros(final int axis0) {
        return null;
    }

    public static Tile zeros(final int axis0, final int axis1) {
        return null;
    }

    public static Tile zeros(final int axis0, final int axis1, final int axis2) {
        return null;
    }

    /**
     * Creates a 1D-tile with sequence elements starting with zero.
     * <p>
     * <code>
     * var tile = TileOp.arange(4); // [0, 1, 2, 3]
     * </code>
     * </p>
     * <p>
     * If not code type is passed, then it is assumed F32 (FLOAT) Type
     *
     * @param size
     * @return {@link Tile}
     */
    public static Tile arange(final int size) {
        return null;
    }

    public static Tile arange(final int size, CodeType codeType) {
        return null;
    }

    /**
     * Creates a 1D-tile with sequence of elements starting with "start",
     * in increments of "step".
     * <p>
     * <code>
     * var tile = TileOp.arange(4, 2, 2); // [4, 6, 8, 10]
     * </code>
     * </p>
     *
     * @param size
     * @return {@link Tile}
     */
    public static Tile arange(final int size, final int start, final int step) {
        return null;
    }

    /**
     * Creates a 1D-tile with sequence of elements starting with "start",
     * in increments of "step".
     * <p>
     * <code>
     * var tile = TileOp.arange(4, 2, 2); // [4, 6, 8, 10]
     * </code>
     * </p>
     *
     * @param size
     * @return {@link Tile}
     */
    public static Tile arange(final int size, final int start, final int step, CodeType codeType) {
        return null;
    }

    public static Tile permute(Tile reshape, Shape shape) {
        return null;
    }
}
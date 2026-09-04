
package hat;

import hat.buffer.Tensor2DF16;
import hat.buffer.Tensor2DF32;
import hat.buffer.TensorF32;
import hat.types.Tile;
import optkl.ifacemapper.Buffer;

import java.util.stream.IntStream;

/**
 * Interface to represent common context operations to support the Tile Programming Model.
 * Common operations are:
 * - Obtaining the block index id.
 * - Load tensors into tiles.
 * - Store tensors into tiles.
 * - Represent a shape for a tile.
 */
public interface TileContext {

    static int BIDX() {return 0;}
    static int BIDY() {return 0;}
    static int BIDZ() {return 0;}

    static Tile load(TensorF32 buffer, int pid, int tileSize) {
        return null;
    }

    static Tile load(Tensor2DF32 buffer, TileIndex2D tileIndex2D, Shape tileSize) {
        return null;
    }

    static Tile load(Tensor2DF16 buffer, TileIndex2D tileIndex2D, Shape tileSize) {
        return null;
    }

    static Tile load(TensorF32 buffer, TileIndex2D tileIndex2D, Shape shape) {
        return null;
    }

    static void store(TensorF32 buffer, int pid, Tile result) {

    }

    static void store(TensorF32 buffer, TileIndex2D tileIndex, Tile result) {

    }

    static void store(Tensor2DF32 buffer, TileIndex2D tileIndex, Tile result) {

    }

    static TileIndex2D index(int bidx, int bidy) {
        return new TileIndex2D(bidx, bidy);
    }

    static Shape shape(int tm) {
        return new Shape(tm, 1, 1);
    }

    static Shape shape(int tm, int tn) {
        return new Shape(tm, tn, 1);
    }

    static Shape shape(int tm, int tn, int tk) {
        return new Shape(tm, tn, tk);
    }

    static int[] irange(int startIndex, int endIndex) {
        return IntStream.range(0, (endIndex - startIndex)).map(i -> startIndex + i).toArray();
    }

}

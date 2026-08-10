
package hat;

import hat.buffer.TensorF32;
import optkl.ifacemapper.Buffer;

public interface TileContext {

    static int bid(int tileDim) {
        return 0;
    }

    static Tile load(Buffer buffer, int pid, int tileSize) {
        return null;
    }

    static Tile load(Buffer buffer, int pid, TileShape tileShape) {
        return null;
    }

    static Tile load(Buffer buffer, TileIndex1D pid, TileShape tileShape) {
        return null;
    }

    static Tile load(Buffer buffer, TileIndex2D pid, TileShape tileShape) {
        return null;
    }

    static Tile load(Buffer buffer, TileIndex3D pid, TileShape tileShape) {
        return null;
    }

    static void store(Buffer buffer, int pid, Tile result) {
    }

    static void store(Buffer buffer, TileIndex1D tileIndex1D, Tile result) {}

    static void store(Buffer buffer, TileIndex2D tileIndex2D, Tile result) {}

    static void store(Buffer buffer, TileIndex3D tileIndex3D, Tile result) {}

    static TileIndex2D index(int bidx, int bidy) {
        return new TileIndex2D(bidx, bidy);
    }

    static TileIndex1D index(int bidx) {
        return new TileIndex1D(bidx);
    }

    static TileIndex3D index(int bidx, int bidy, int bidz) {
        return new TileIndex3D(bidx, bidy, bidz);
    }

    static TileShape shape(int tm, int tk) {
        return new TileShape(tm, tk);
    }

    static  TileShape shape(int tm) {
        return new TileShape(tm);
    }

    static int num_tiles(TensorF32 inputA, int i, TileShape shape) {
        return 0;
    }

    static Tile zeros(int tm, int tk) {
        return null;
    }

    static Tile sum(Tile tileA, int index) {
        return null;
    }

    static void sum(Tile tileA, TileIndex1D index) {}

    static void sum(Tile tileA, TileIndex2D index){}

    static void sum(Tile tileA, TileIndex3D index) {}

    static Tile full(TileShape shape, int index) {
        return null;
    }

    static Tile transpose(Tile inputTile) {
        return null;
    }
}

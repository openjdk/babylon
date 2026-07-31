
package hat;

import hat.buffer.TileF32Array;
import optkl.ifacemapper.Buffer;

public interface TileContext {

    int bid(int tileDim);

    Tile load(Buffer buffer, int pid, int tileSize);

    Tile load(Buffer buffer, int pid, TileShape tileShape);

    Tile load(Buffer buffer, TileIndex1D pid, TileShape tileShape);

    Tile load(Buffer buffer, TileIndex2D pid, TileShape tileShape);

    Tile load(Buffer buffer, TileIndex3D pid, TileShape tileShape);

    void store(Buffer buffer, int pid, Tile result);

    void store(Buffer buffer, TileIndex1D tileIndex1D, Tile result);

    void store(Buffer buffer, TileIndex2D tileIndex2D, Tile result);

    void store(Buffer buffer, TileIndex3D tileIndex3D, Tile result);

    default TileIndex2D index(int bidx, int bidy) {
        return new TileIndex2D(bidx, bidy);
    }

    default TileIndex1D index(int bidx) {
        return new TileIndex1D(bidx);
    }

    default TileIndex3D index(int bidx, int bidy, int bidz) {
        return new TileIndex3D(bidx, bidy, bidz);
    }

    default TileShape shape(int tm, int tk) {
        return new TileShape(tm, tk);
    }

    default TileShape shape(int tm) {
        return new TileShape(tm);
    }

    int num_tiles(TileF32Array inputA, int i, TileShape shape);

    Tile zeros(int tm, int tk);

    Tile sum(Tile tileA, int index);

    void sum(Tile tileA, TileIndex1D index);

    void sum(Tile tileA, TileIndex2D index);

    void sum(Tile tileA, TileIndex3D index);

    Tile full(TileShape shape, int index);

    Tile transpose(Tile inputTile);
}

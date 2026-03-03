package hat;

import hat.buffer.F32Array;
import optkl.ifacemapper.Buffer;

public class TileContext {

    public int bid(int tileDim) {
        return 0;
    }

    public TileData load(Buffer buffer, int pid, int tileSize) {
        return null;
    }

    public TileData load(Buffer buffer, TileIndex1D pid, TileShape tileShape) {
        return null;
    }

    public TileData load(Buffer buffer, TileIndex2D pid, TileShape tileShape) {
        return null;
    }

    public TileData load(Buffer buffer, TileIndex3D pid, TileShape tileShape) {
        return null;
    }

    public void store(Buffer buffer, int pid, TileData result) {
    }

    public void store(Buffer buffer, TileIndex1D tileIndex1D, TileData result) {

    }

    public void store(Buffer buffer, TileIndex2D tileIndex2D, TileData result) {

    }

    public void store(Buffer buffer, TileIndex3D tileIndex3D, TileData result) {

    }

    public TileIndex2D index(int bidx, int bidy) {
        return new TileIndex2D(bidx, bidy);
    }

    public TileIndex1D index(int bidx) {
        return new TileIndex1D(bidx);
    }

    public TileIndex3D index(int bidx, int bidy, int bidz) {
        return new TileIndex3D(bidx, bidy, bidz);
    }

    public TileShape shape(int tm, int tk) {
        return new TileShape(tm, tk);
    }

    public TileShape shape(int tm) {
        return new TileShape(tm);
    }

    public int num_tiles(F32Array inputA, int i, TileShape shape) {
        return 0;
    }

    public TileData zeros(int tm, int tk) {
        return null;
    }

    public TileData sum(TileData tileA, int index) {
        return null;
    }

    public void sum(TileData tileA, TileIndex1D index) {

    }

    public void sum(TileData tileA, TileIndex2D index) {

    }

    public void sum(TileData tileA, TileIndex3D index) {

    }

    public TileData full(TileShape shape, int index) {
        return null;
    }

    public TileData transpose(TileData inputTile) {
        return null;
    }
}

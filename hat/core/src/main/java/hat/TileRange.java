package hat;

// Probably we will represent a TileRange as an NDRange with Tile Capabilities
public interface TileRange {

    NDRange.Global global();

    NDRange.Local local();


    record TileRange1D(int size, int tile) implements TileRange {
        @Override
        public NDRange.Global global() {
            return NDRange.Global1D.of(size);
        }

        @Override
        public NDRange.Local local() {
            return NDRange.Local1D.of(tile);
        }
    }

    record TileRange2D(int sizeX, int sizeY, int tileX, int tileY) implements TileRange {
        @Override
        public NDRange.Global global() {
            return NDRange.Global2D.of(sizeX, sizeY);
        }

        @Override
        public NDRange.Local local() {
            return NDRange.Local2D.of(tileX, tileY);
        }
    }

    static TileRange of1D(int length, int tileSize) {
        return new TileRange1D(length, tileSize);
    }

    static TileRange of2D(int m, int n, int tm, int tn) {
        return new TileRange2D(m, n, tm, tn);
    }
}

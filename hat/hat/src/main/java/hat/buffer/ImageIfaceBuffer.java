package hat.buffer;

import java.awt.image.BufferedImage;
import java.awt.image.DataBuffer;
import java.awt.image.DataBufferByte;
import java.awt.image.DataBufferInt;
import java.awt.image.DataBufferUShort;
import java.lang.foreign.MemorySegment;

import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_SHORT;

public interface ImageIfaceBuffer<T extends ImageIfaceBuffer<?>> extends IncompleteBuffer {
    @SuppressWarnings("unchecked")
    default T syncFromRasterDataBuffer(DataBuffer dataBuffer) { // int[], byte[], short[]
        switch (dataBuffer) {
            case DataBufferInt arr ->
                    MemorySegment.copy(arr.getData(), 0, Buffer.getMemorySegment(this), JAVA_INT, 16L, arr.getData().length);
            case DataBufferByte arr ->
                    MemorySegment.copy(arr.getData(), 0, Buffer.getMemorySegment(this), JAVA_BYTE, 16L, arr.getData().length);
            case DataBufferUShort arr ->
                    MemorySegment.copy(arr.getData(), 0, Buffer.getMemorySegment(this), JAVA_SHORT, 16L, arr.getData().length);
            default -> throw new IllegalStateException("Unexpected value: " + dataBuffer);
        }
        return (T)this;
    }

    default T syncFromRaster(BufferedImage bufferedImage) { // int[], byte[], short[]
        return syncFromRasterDataBuffer(bufferedImage.getRaster().getDataBuffer());
    }

    @SuppressWarnings("unchecked")
    default T syncToRasterDataBuffer(DataBuffer dataBuffer) { // int[], byte[], short[]
        switch (dataBuffer) {
            case DataBufferUShort arr ->
                    MemorySegment.copy(Buffer.getMemorySegment(this), JAVA_SHORT, 16L, arr.getData(), 0, arr.getData().length);
            case DataBufferInt arr ->
                    MemorySegment.copy(Buffer.getMemorySegment(this), JAVA_INT, 16L, arr.getData(), 0, arr.getData().length);
            case DataBufferByte arr ->
                    MemorySegment.copy(Buffer.getMemorySegment(this), JAVA_BYTE, 16L, arr.getData(), 0, arr.getData().length);
            default -> throw new IllegalStateException("Unexpected value: " + dataBuffer);
        }
        return (T) this;
    }

    default T syncToRaster(BufferedImage bufferedImage) { // int[], byte[], short[]
        return syncToRasterDataBuffer(bufferedImage.getRaster().getDataBuffer());
    }
}

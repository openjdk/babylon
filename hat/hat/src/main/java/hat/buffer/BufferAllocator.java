package hat.buffer;

import hat.ifacemapper.BoundSchema;
import hat.ifacemapper.SegmentMapper;


public interface BufferAllocator {
    <T extends Buffer>T allocate(SegmentMapper<T> segmentMapper, BoundSchema<T> buffer);
}

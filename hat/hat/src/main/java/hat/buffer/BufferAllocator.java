package hat.buffer;

import hat.ifacemapper.Schema;
import hat.ifacemapper.SegmentMapper;

import java.lang.foreign.MemorySegment;
import java.lang.invoke.MethodHandles;


public interface BufferAllocator {
    <T extends Buffer>T allocate(SegmentMapper<T> segmentMapper, Schema.BoundSchema<T> buffer);
}

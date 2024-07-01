package hat.buffer;

import hat.ifacemapper.SegmentMapper;

import java.lang.foreign.MemorySegment;
@FunctionalInterface
public interface BufferAllocator {
    <T extends Buffer>T allocate(SegmentMapper<T> segmentMapper);
}

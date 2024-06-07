package hat.buffer;

import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;

public interface CompleteBuffer extends Buffer {
    default String schema() {
        MemoryLayout memoryLayout = layout();
        MemorySegment segment = memorySegment();
        StringBuilder sb = new StringBuilder(Long.toString(segment.byteSize())).append("!");
        return buildSchema(sb, memoryLayout, null);
    }
}

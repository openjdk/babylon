package hat.buffer;

import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;

public interface CompleteBuffer extends Buffer {
    default String schema() {
        return new SchemaBuilder()
                .literal(memorySegment().byteSize())
                .hash()
                .layout(layout(),null, false)
                .toString();
    }
}

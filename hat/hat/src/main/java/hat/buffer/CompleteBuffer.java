package hat.buffer;

import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;

public interface CompleteBuffer extends Buffer {
    default String schema() {
        return new SchemaBuilder()
                .literal(Buffer.getMemorySegment(this).byteSize())
                .hash()
                .layout(Buffer.getLayout(this),null, false)
                .toString();
    }
}

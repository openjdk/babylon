package hat.buffer;

import java.lang.foreign.MemoryLayout;
import java.lang.foreign.SequenceLayout;
import java.lang.foreign.StructLayout;

public interface IncompleteBuffer extends Buffer {
    default String schema() {
        MemoryLayout memoryLayout = Buffer.getLayout(this);
        if (memoryLayout instanceof StructLayout structLayout) {
            var memberLayouts = structLayout.memberLayouts();
            if (memberLayouts.getLast() instanceof SequenceLayout tailSequenceLayout) {
                return new SchemaBuilder()
                        .literal(memoryLayout.byteOffset(
                                MemoryLayout.PathElement.groupElement(memberLayouts.size() - 1)))
                        .plus()
                        .layout(Buffer.getLayout(this),tailSequenceLayout,true)
                        .toString();
            } else {
                throw new IllegalStateException("IncompleteBuffer last layout is not SequenceLayout!");
            }
        } else {
            throw new IllegalStateException("IncompleteBuffer must be a StructLayout");
        }

    }
}

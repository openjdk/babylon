package hat.buffer;

import java.lang.foreign.MemoryLayout;
import java.lang.foreign.SequenceLayout;
import java.lang.foreign.StructLayout;

public interface IncompleteBuffer extends Buffer {
    default String schema() {
        MemoryLayout memoryLayout = layout();
        if (memoryLayout instanceof StructLayout structLayout) {
            var memberLayouts = structLayout.memberLayouts();
            if (memberLayouts.getLast() instanceof SequenceLayout tailSequenceLayout) {
                long offsetOfTailSequenceLayout = memoryLayout.byteOffset(MemoryLayout.PathElement.groupElement(memberLayouts.size() - 1));
                //  var offsetOfTailSequenceLayout = tailSequenceLayout.byteOffset();
                StringBuilder sb = new StringBuilder(Long.toString(offsetOfTailSequenceLayout)).append("+");
                return buildSchema(sb, layout(), tailSequenceLayout);
            } else {
                throw new IllegalStateException("IncompleteBuffer last layout is not SequenceLayout!");
            }
        } else {
            throw new IllegalStateException("IncompleteBuffer must be a StructLayout");
        }

    }
}

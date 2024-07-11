package hat.buffer;

import hat.text.CodeBuilder;
import hat.util.StreamCounter;

import java.lang.foreign.MemoryLayout;
import java.lang.foreign.PaddingLayout;
import java.lang.foreign.SequenceLayout;
import java.lang.foreign.StructLayout;
import java.lang.foreign.UnionLayout;
import java.lang.foreign.ValueLayout;

public class SchemaBuilder extends CodeBuilder<SchemaBuilder> {
    SchemaBuilder layout(MemoryLayout layout, SequenceLayout tailSequenceLayout) {
        either(layout.name().isPresent(), (_) -> identifier(layout.name().get()), (_) -> questionMark()).colon();
        switch (layout) {
            case StructLayout structLayout -> {
                brace((_) -> {
                    StreamCounter.of(structLayout.memberLayouts().stream(), (c, l) -> {
                        if (c.isNotFirst()) {
                            comma();
                        }
                        layout(l, tailSequenceLayout);
                    });
                });
            }
            case UnionLayout unionLayout -> {
                chevron((_) -> {
                    StreamCounter.of(unionLayout.memberLayouts().stream(), (c, l) -> {
                        if (c.isNotFirst()) {
                            bar();
                        }
                        layout(l, tailSequenceLayout);
                    });
                });
            }
            case ValueLayout valueLayout -> {
                literal(ArgArray.valueLayoutToSchemaString(valueLayout));
            }
            case PaddingLayout paddingLayout -> {
                literal("x").literal(paddingLayout.byteSize());
            }
            case SequenceLayout sequenceLayout -> {
                sbrace((_) -> {
                   // if (sequenceLayout.equals(tailSequenceLayout) && incomplete) {
                     //   asterisk();
                    //} else {
                        literal(sequenceLayout.elementCount());
                   // }
                    colon();
                    layout(sequenceLayout.elementLayout(), tailSequenceLayout);
                });
            }
        }
        return this;
    }

    public static String schema(Buffer buffer) {
       // if (complete) {
            return new SchemaBuilder()
                    .literal(Buffer.getMemorySegment(buffer).byteSize())
                    .hash()
                    .layout(Buffer.getLayout(buffer), null)
                    .toString();
    /*    }else{
            MemoryLayout memoryLayout = Buffer.getLayout(buffer);
            if (memoryLayout instanceof StructLayout structLayout) {
                var memberLayouts = structLayout.memberLayouts();
                if (memberLayouts.getLast() instanceof SequenceLayout tailSequenceLayout) {
                    return new SchemaBuilder()
                            .literal(memoryLayout.byteOffset(
                                    MemoryLayout.PathElement.groupElement(memberLayouts.size() - 1)))
                            .plus()
                            .layout(Buffer.getLayout(buffer),tailSequenceLayout,true)
                            .toString();
                } else {
                    throw new IllegalStateException("IncompleteBuffer last layout is not SequenceLayout!");
                }
            } else {
                throw new IllegalStateException("IncompleteBuffer must be a StructLayout");
            }
        }*/
    }
}

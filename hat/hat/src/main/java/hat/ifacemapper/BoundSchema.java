package hat.ifacemapper;

import hat.buffer.Buffer;
import hat.buffer.BufferAllocator;

import java.lang.invoke.MethodHandles;
import java.util.List;

public sealed interface BoundSchema<T extends Buffer> permits BoundSchemaNode.BoundSchemaRootNode{

    T allocate(MethodHandles.Lookup lookup, BufferAllocator bufferAllocator);

    Schema<T> schema();

    List<BoundSchemaNode.BoundArrayFieldLayout> boundArrayFields();
}

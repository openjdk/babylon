package hat.ifacemapper;

import hat.buffer.Buffer;
import hat.buffer.BufferAllocator;

import java.lang.foreign.GroupLayout;
import java.lang.foreign.MemoryLayout;
import java.lang.invoke.MethodHandles;
import java.util.ArrayList;
import java.util.List;

public class BoundSchema<T extends Buffer> {
    final private List<BoundArrayFieldLayout> boundArrayFields = new ArrayList<>();
    final private int[] arrayLengths;
    final private Schema<T> schema;
    final private GroupLayout groupLayout;
    BoundSchemaNode<?> rootBoundSchemaNode;

    public static sealed class FieldLayout<T extends Schema.FieldNode> permits ArrayFieldLayout {
        public final T field;
        public MemoryLayout layout;

        FieldLayout(T field, MemoryLayout layout) {
            this.field = field;
            this.layout = layout;
        }
    }

    public sealed static class ArrayFieldLayout extends FieldLayout<Schema.FieldNode>
            permits BoundArrayFieldLayout {
        public final int len;

        ArrayFieldLayout(Schema.FieldNode fieldControlledArray, MemoryLayout layout, int len) {
            super(fieldControlledArray, layout);
            this.len = len;
        }
    }

    public static final class BoundArrayFieldLayout extends ArrayFieldLayout {
        public final int idx;

        BoundArrayFieldLayout(Schema.FieldNode fieldControlledArray, MemoryLayout layout, int len, int idx) {
            super(fieldControlledArray, layout, len);
            this.idx = idx;
        }
    }
    public BoundSchema(Schema<T> schema, int... arrayLengths) {
        this.schema = schema;
        this.arrayLengths = arrayLengths;
        this.rootBoundSchemaNode = new BoundSchemaNode<>((BoundSchema<Buffer>) this, null, this.schema.rootIfaceType);
        this.groupLayout = schema.rootIfaceType.getBoundGroupLayout(rootBoundSchemaNode);
        this.rootBoundSchemaNode.memoryLayouts.add(this.groupLayout);
    }

    int takeArrayLen() {
        return arrayLengths[boundArrayFields.size()];
    }

    public T allocate(MethodHandles.Lookup lookup, BufferAllocator bufferAllocator) {
        return bufferAllocator.allocate(SegmentMapper.of(lookup, schema.iface, groupLayout, this), this);
    }

    public Schema<T> schema() {
        return schema;
    }

    public List<BoundArrayFieldLayout> boundArrayFields() {
        return boundArrayFields;
    }

    public GroupLayout groupLayout() {
        return groupLayout;
    }

    public BoundSchemaNode<?> rootBoundSchemaNode() {
        return rootBoundSchemaNode;
    }


    FieldLayout<?> createFieldBinding(Schema.FieldNode fieldNode, MemoryLayout memoryLayout) {
        if (fieldNode instanceof Schema.FieldNode.IfaceFieldControlledArray
                || fieldNode instanceof Schema.FieldNode.PrimitiveFieldControlledArray) {
            int idx = boundArrayFields.size();
            var arraySizeBinding = new BoundArrayFieldLayout(fieldNode, memoryLayout, arrayLengths[idx], idx);
            boundArrayFields.add(arraySizeBinding);
            return arraySizeBinding;
        } else if (fieldNode instanceof Schema.FieldNode.IfaceFixedArray ifaceMapableFixedArray) {
            return new ArrayFieldLayout(fieldNode, memoryLayout, ifaceMapableFixedArray.len);
        } else if (fieldNode instanceof Schema.FieldNode.PrimitiveFixedArray primitiveFixedArray) {
            return new ArrayFieldLayout(fieldNode, memoryLayout, primitiveFixedArray.len);
        } else {
            return new FieldLayout<>(fieldNode, memoryLayout);
        }
    }

    public static class BoundSchemaNode<T extends MappableIface> {
        final public  BoundSchema<Buffer> boundSchema;
        final public  BoundSchemaNode<T> parent;
        final public List<BoundSchemaNode<?>> children = new ArrayList<>();
        final public List<MemoryLayout> memoryLayouts = new ArrayList<>();
        final public List<FieldLayout<?>> fieldLayouts = new ArrayList<>();
        final public Schema.IfaceType ifaceType;

        BoundSchemaNode(BoundSchema<Buffer> boundSchema, BoundSchemaNode<T> parent, Schema.IfaceType ifaceType) {
            this.boundSchema = boundSchema;
            this.parent = parent;
            this.ifaceType = ifaceType;
        }

        int takeArrayLen() {
            return boundSchema.takeArrayLen();
        }


        FieldLayout<?> createFieldBinding(Schema.FieldNode fieldNode, MemoryLayout memoryLayout) {
            return boundSchema.createFieldBinding(fieldNode, memoryLayout);
        }

        void bind(Schema.FieldNode fieldNode, MemoryLayout memoryLayout) {
            fieldLayouts.add(createFieldBinding(fieldNode, memoryLayout));
            memoryLayouts.add(memoryLayout);
        }

        public MemoryLayout[] memoryLayoutListToArray() {
            return memoryLayouts.toArray(new MemoryLayout[0]);
        }

        public BoundSchemaNode<T> createChild(Schema.IfaceType ifaceType) {
            var boundSchemaChildNode = new BoundSchemaNode<T>(boundSchema, this, ifaceType);
            children.add(boundSchemaChildNode);
            return boundSchemaChildNode;
        }

        public FieldLayout<?> getBoundFieldChild(String fieldName) {
            return fieldLayouts.stream().filter(fieldLayout -> fieldLayout.field.name.equals(fieldName)).findFirst().orElseThrow();
        }
    }
}

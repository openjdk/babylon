package hat.ifacemapper;

import hat.buffer.Buffer;
import hat.buffer.BufferAllocator;

import java.lang.foreign.GroupLayout;
import java.lang.foreign.MemoryLayout;
import java.lang.invoke.MethodHandles;
import java.util.ArrayList;
import java.util.List;

import static hat.ifacemapper.Schema.SchemaNode.IfaceType;

public abstract sealed class BoundSchemaNode
        permits  BoundSchemaNode.BoundSchemaChildNode, BoundSchemaNode.BoundSchemaRootNode {

    static sealed class FieldLayout<T extends Schema.SchemaNode.FieldNode> permits ArrayFieldLayout {
        public final T field;
        public MemoryLayout layout;
        FieldLayout(T field, MemoryLayout layout) {
            this.field = field;
            this.layout = layout;
        }
    }
    public sealed static class ArrayFieldLayout extends FieldLayout<Schema.SchemaNode.FieldNode>
            permits BoundArrayFieldLayout {
        public final int len;
        ArrayFieldLayout(Schema.SchemaNode.FieldNode fieldControlledArray, MemoryLayout layout, int len) {
            super(fieldControlledArray, layout);
            this.len = len;
        }
    }
    public static final class BoundArrayFieldLayout extends ArrayFieldLayout {
        public final int idx;
        BoundArrayFieldLayout(Schema.SchemaNode.FieldNode fieldControlledArray, MemoryLayout layout, int len, int idx) {
            super(fieldControlledArray, layout, len);
            this.idx = idx;
        }
    }

    final protected BoundSchemaNode parent;
    final List<BoundSchemaChildNode> children = new ArrayList<>();
    final List<MemoryLayout> memoryLayouts = new ArrayList<>();
    final List<FieldLayout<?>> fieldLayouts = new ArrayList<>();
    final Schema.SchemaNode.IfaceType ifaceType;

    BoundSchemaNode(BoundSchemaNode parent, Schema.SchemaNode.IfaceType ifaceType) {
        this.parent = parent;
        this.ifaceType = ifaceType;
    }

    abstract int takeArrayLen();

    abstract FieldLayout<?> createFieldBinding(Schema.SchemaNode.FieldNode namedFieldNode, MemoryLayout memoryLayout);

    void bind(Schema.SchemaNode.FieldNode fieldNode, MemoryLayout memoryLayout) {
        fieldLayouts.add(createFieldBinding(fieldNode, memoryLayout));
        memoryLayouts.add(memoryLayout);
    }

    public MemoryLayout[] memoryLayoutListToArray() {
        return memoryLayouts.toArray(new MemoryLayout[0]);
    }

    public BoundSchemaChildNode createChild(Schema.SchemaNode.IfaceType ifaceType) {
        var boundSchemaChildNode = new BoundSchemaChildNode(this, ifaceType);
        children.add(boundSchemaChildNode);
        return boundSchemaChildNode;
    }

    public static final class BoundSchemaRootNode<T extends Buffer> extends BoundSchemaNode implements BoundSchema<T> {
        final private List<BoundArrayFieldLayout> boundArrayFields=new ArrayList<>();
        final private int[] arrayLengths;
        final private Schema<T> schema;
        final private GroupLayout groupLayout;

        public BoundSchemaRootNode(Schema<T> schema, int... arrayLengths) {
            super(null, schema.rootIfaceType);
            this.schema = schema;
            this.arrayLengths = arrayLengths;
            this.groupLayout =ifaceType.getBoundGroupLayout(this);
            memoryLayouts.add(this.groupLayout);
        }
        @Override
        public T allocate(MethodHandles.Lookup lookup, BufferAllocator bufferAllocator) {
            return bufferAllocator.allocate(SegmentMapper.of(lookup, schema.iface, groupLayout, this), this);
        }

        @Override
        public Schema<T> schema(){
            return schema;
        }

        @Override
        public List<BoundArrayFieldLayout> boundArrayFields() {
            return boundArrayFields;
        }

        @Override
        int takeArrayLen() {
            return arrayLengths[boundArrayFields.size()];
        }

        @Override
        FieldLayout<?> createFieldBinding(Schema.SchemaNode.FieldNode fieldNode, MemoryLayout memoryLayout) {
            if (fieldNode instanceof Schema.SchemaNode.IfaceFieldControlledArray
                    || fieldNode instanceof Schema.SchemaNode.PrimitiveFieldControlledArray) {
                int idx = boundArrayFields.size();
                var arraySizeBinding = new BoundArrayFieldLayout(fieldNode, memoryLayout, arrayLengths[idx], idx);
                boundArrayFields.add(arraySizeBinding);
                return arraySizeBinding;
            }else  if (fieldNode instanceof Schema.SchemaNode.IfaceFixedArray ifaceMapableFixedArray){
                return new ArrayFieldLayout(fieldNode, memoryLayout,  ifaceMapableFixedArray.len);
            }else  if (fieldNode instanceof Schema.SchemaNode.PrimitiveFixedArray primitiveFixedArray){
                return new ArrayFieldLayout(fieldNode, memoryLayout, primitiveFixedArray.len);
            }else{
                return new FieldLayout<>(fieldNode,memoryLayout);
            }
        }
    }

    public static final class BoundSchemaChildNode extends BoundSchemaNode {
        BoundSchemaChildNode(BoundSchemaNode parent, Schema.SchemaNode.IfaceType ifaceType) {
            super(parent, ifaceType);
        }
        @Override
        int takeArrayLen() {
            return parent.takeArrayLen();
        }

        @Override
        FieldLayout<?> createFieldBinding(Schema.SchemaNode.FieldNode namedFieldNode, MemoryLayout memoryLayout) {
            return parent.createFieldBinding(namedFieldNode, memoryLayout);
        }
    }
}

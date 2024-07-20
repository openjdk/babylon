package hat.ifacemapper;

import hat.buffer.Buffer;
import hat.buffer.BufferAllocator;

import java.lang.foreign.GroupLayout;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandles;
import java.util.ArrayList;
import java.util.List;

import static hat.ifacemapper.BoundSchema.BoundSchemaNode.getBoundGroupLayout;


public class BoundSchema<T extends Buffer> {
    final private List<BoundArrayFieldLayout> boundArrayFields = new ArrayList<>();
    final private int[] arrayLengths;
    final private Schema<T> schema;
    final private GroupLayout groupLayout;
    BoundSchemaNode<?> rootBoundSchemaNode;


    public static sealed class FieldLayout<T extends Schema.FieldNode> permits ArrayFieldLayout {
        public BoundSchemaNode<?> parent;
        public final T field;
        public MemoryLayout layout;
        MemoryLayout.PathElement pathElement;

        FieldLayout(BoundSchemaNode<?> parent,T field, MemoryLayout layout) {
            this.parent = parent;
            this.field = field;
            this.layout = layout;
            this.pathElement =  MemoryLayout.PathElement.groupElement(field.name);
        }

        public long offset() {
            return parent.memoryLayouts.getLast().byteOffset(pathElement);
        }
        public MemoryLayout layout() {
            return parent.memoryLayouts.getLast().select(pathElement);
        }
    }

    public sealed static class ArrayFieldLayout extends FieldLayout<Schema.FieldNode>
            permits BoundArrayFieldLayout {
        public final int len;

        ArrayFieldLayout(BoundSchemaNode<?> parent, Schema.FieldNode fieldControlledArray, MemoryLayout layout, int len) {
            super(parent, fieldControlledArray, layout);
            this.len = len;
        }

        public long offset(long idx) {
            return 0L;
        }
    }

    public static final class BoundArrayFieldLayout extends ArrayFieldLayout {
        public final int idx;

        BoundArrayFieldLayout(BoundSchemaNode<?> parent,Schema.FieldNode fieldControlledArray, MemoryLayout layout, int len, int idx) {
            super(parent, fieldControlledArray, layout, len);
            this.idx = idx;
        }
    }
    public BoundSchema(Schema<T> schema, int... arrayLengths) {
        this.schema = schema;
        this.arrayLengths = arrayLengths;
        this.rootBoundSchemaNode = new BoundSchemaNode<>((BoundSchema<Buffer>) this, null, this.schema.rootIfaceType);
        this.groupLayout = getBoundGroupLayout(rootBoundSchemaNode,schema.rootIfaceType);
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


    FieldLayout<?> createFieldBinding(BoundSchemaNode<?>parent, Schema.FieldNode fieldNode, MemoryLayout memoryLayout) {
        if (fieldNode instanceof Schema.FieldNode.IfaceFieldControlledArray
                || fieldNode instanceof Schema.FieldNode.PrimitiveFieldControlledArray) {
            int idx = boundArrayFields.size();
            var arraySizeBinding = new BoundArrayFieldLayout(parent,fieldNode, memoryLayout, arrayLengths[idx], idx);
            boundArrayFields.add(arraySizeBinding);
            return arraySizeBinding;
        } else if (fieldNode instanceof Schema.FieldNode.IfaceFixedArray ifaceMapableFixedArray) {
            return new ArrayFieldLayout(parent,fieldNode, memoryLayout, ifaceMapableFixedArray.len);
        } else if (fieldNode instanceof Schema.FieldNode.PrimitiveFixedArray primitiveFixedArray) {
            return new ArrayFieldLayout(parent,fieldNode, memoryLayout, primitiveFixedArray.len);
        } else {
            return new FieldLayout<>(parent,fieldNode, memoryLayout);
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
            return boundSchema.createFieldBinding(this,fieldNode, memoryLayout);
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

        public FieldLayout<?> getName(String fieldName) {
            return fieldLayouts.stream().filter(fieldLayout -> fieldLayout.field.name.equals(fieldName)).findFirst().orElseThrow();


        }

        static GroupLayout getBoundGroupLayout(BoundSchemaNode child, Schema.IfaceType ifaceType) {

            ifaceType.fields.forEach(fieldNode ->
                child.bind(fieldNode,(switch (fieldNode) {
                            case Schema.SchemaNode.Padding field ->
                                    MemoryLayout.paddingLayout(field.len);
                            case Schema.FieldNode.AddressField field ->
                                    ValueLayout.ADDRESS;
                            case Schema.FieldNode.ArrayLen field ->
                                    MapperUtil.primitiveToLayout(field.type);
                            case Schema.FieldNode.AtomicField field ->
                                    MapperUtil.primitiveToLayout(field.type);
                            case Schema.FieldNode.IfaceField field -> {
                                var fieldIfaceType = field.parent.getChild(field.ifaceType.iface);
                                yield getBoundGroupLayout(child.createChild(ifaceType), fieldIfaceType);
                            }
                            case Schema.FieldNode.PrimitiveField field ->
                                    MapperUtil.primitiveToLayout(field.type);
                            case Schema.FieldNode.IfaceFixedArray field -> {
                                var fieldIfaceType = field.parent.getChild(field.ifaceType.iface);
                                var elementLayout = getBoundGroupLayout(child.createChild(ifaceType), fieldIfaceType)
                                        .withName(field.ifaceType.iface.getSimpleName());
                                yield MemoryLayout.sequenceLayout(field.len,elementLayout);
                            }
                            case Schema.FieldNode.PrimitiveFixedArray field -> {
                                var elementLayout = MapperUtil.primitiveToLayout(field.type)
                                        .withName(field.type.getSimpleName());
                                yield MemoryLayout.sequenceLayout(field.len, elementLayout);
                            }
                            case Schema.FieldNode.IfaceFieldControlledArray field -> {
                                // To determine the actual 'array' size we multiply the contributing dims by the stride .
                                int size = field.stride; //usually 1 but developer can define.
                                for (int i = 0; i < field.contributingDims; i++) {
                                    size *= child.takeArrayLen(); // this takes an arraylen and bumps the ptr
                                }
                                var fieldIfaceType = field.parent.getChild(field.ifaceType.iface);
                                var elementLayout = getBoundGroupLayout(child.createChild(ifaceType), fieldIfaceType)
                                        .withName(field.ifaceType.iface.getSimpleName());
                                yield MemoryLayout.sequenceLayout(size,elementLayout);
                            }
                            case Schema.FieldNode.PrimitiveFieldControlledArray field -> {
                                // To determine the actual 'array' size we multiply the contributing dims by the stride .
                                int size = field.stride; //usually 1 but developer can define.
                                for (int i = 0; i < field.contributingDims; i++) {
                                    size *= child.takeArrayLen(); // this takes an arraylen and bumps the ptr
                                }
                                var elementLayout = MapperUtil.primitiveToLayout(field.type)
                                        .withName(field.type.getSimpleName());
                                yield MemoryLayout.sequenceLayout(size,elementLayout);
                            }
                        }).withName(fieldNode.name))
            );
            return (MapperUtil.isUnion(ifaceType.iface)
                    ? MemoryLayout.unionLayout(child.memoryLayoutListToArray())
                    : MemoryLayout.structLayout(child.memoryLayoutListToArray())).withName(ifaceType.iface.getSimpleName());
        }

    }
}

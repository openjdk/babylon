package hat.ifacemapper;

import hat.buffer.Buffer;
import hat.buffer.BufferAllocator;
import hat.ifacemapper.accessor.AccessorInfo;
import hat.ifacemapper.accessor.ValueType;

import java.lang.foreign.GroupLayout;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandles;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.function.Consumer;

public class Schema<T extends Buffer> {
    final public SchemaNode.IfaceType rootIfaceType;
    public Class<T> iface;

    public static abstract sealed class SchemaNode permits SchemaNode.FieldNode, SchemaNode.IfaceType {
        public IfaceType parent;

        SchemaNode(IfaceType parent) {
            this.parent = parent;
        }

        public abstract void toText(String indent, Consumer<String> stringConsumer);

        public static abstract sealed class FieldNode extends SchemaNode
                permits AddressField, AbstractIfaceField, Padding, AbstractPrimitiveField {
            public final AccessorInfo.Key key;
            public final String name;

            FieldNode(IfaceType parent, AccessorInfo.Key key, String name) {
                super(parent);
                this.key = key;
                this.name = name;
            }
            public abstract void toText(String indent, Consumer<String> stringConsumer);
            public abstract void collectLayouts(BoundSchemaNode layoutCollector);
        }

        public static final class Padding extends FieldNode {
            int len;
            Padding(IfaceType parent, int len) {
                super(parent, AccessorInfo.Key.NONE, "pad"+len);
                this.len = len;
            }
            @Override
            public void toText(String indent, Consumer<String> stringConsumer) {
                stringConsumer.accept(indent + "padding " + len + " bytes");
            }

            @Override
            public void collectLayouts(BoundSchemaNode layoutToFieldBindingNode) {
                layoutToFieldBindingNode.bind(this, MemoryLayout.paddingLayout(len));
            }
        }


        public static abstract sealed class AbstractPrimitiveField extends FieldNode
                permits PrimitiveArray, ArrayLen, AtomicField, PrimitiveField {
            public Class<?> type;
            AbstractPrimitiveField(IfaceType parent, AccessorInfo.Key key, Class<?> type, String name) {
                super(parent,key,  name);
                this.type = type;
            }

            @Override
            public void collectLayouts(BoundSchemaNode boundSchemaNode) {
                boundSchemaNode.bind(this, parent.getBoundLayout(this.type, boundSchemaNode).withName(name));
            }
        }
        public static abstract sealed class AbstractIfaceField extends FieldNode
                permits IfaceArray, IfaceField {

            public IfaceType type;
            AbstractIfaceField(IfaceType parent, AccessorInfo.Key key, IfaceType type, String name) {
                super(parent,key, name);
                this.type = type;
            }
            @Override
            public void collectLayouts(BoundSchemaNode boundSchemaNode) {
                boundSchemaNode.bind(this, parent.getBoundLayout(this.type.iface, boundSchemaNode).withName(name));
            }
        }
        public static final class AddressField extends FieldNode {
            Class<MemorySegment> type;
            AddressField(IfaceType parent, AccessorInfo.Key key, Class<MemorySegment> type, String name) {
                super(parent, key, name);
                this.type = type;
            }

            @Override
            public void toText(String indent, Consumer<String> stringConsumer) {
                stringConsumer.accept(indent + "address " + key + ":" + type);
            }
            @Override
            public void collectLayouts(BoundSchemaNode boundSchemaNode) {
                boundSchemaNode.bind(this, parent.getBoundLayout(type, boundSchemaNode).withName(name));
            }
        }
        public static final class ArrayLen extends AbstractPrimitiveField {
            ArrayLen(IfaceType parent, AccessorInfo.Key key, Class<?> type, String name) {
                super(parent, key, type, name);
            }

            @Override
            public void toText(String indent, Consumer<String> stringConsumer) {
                stringConsumer.accept(indent + "arrayLen " + key + ":" + type);
            }

        }

        public static final class AtomicField extends AbstractPrimitiveField {
            AtomicField(IfaceType parent, AccessorInfo.Key key, Class<?> type, String name) {
                super(parent, key, type, name);
            }

            @Override
            public void toText(String indent, Consumer<String> stringConsumer) {
                stringConsumer.accept(indent + "atomic " + key + ":" + type);
            }

        }

        public static final class IfaceField extends AbstractIfaceField {

            IfaceField(IfaceType parent, AccessorInfo.Key key, IfaceType ifaceType, String name) {
                super(parent, key, ifaceType,name);

            }

            @Override
            public void toText(String indent, Consumer<String> stringConsumer) {
                stringConsumer.accept(indent + "mappable field " + key + ":" + type.iface);
            }

        }
        public static final class PrimitiveField extends AbstractPrimitiveField {

            PrimitiveField(IfaceType parent, AccessorInfo.Key key, Class<?> type, String name) {
                super(parent, key, type, name);

            }

            @Override
            public void toText(String indent, Consumer<String> stringConsumer) {
                stringConsumer.accept(indent + "primitive field " + key + ":" + type);
            }
        }

        public static abstract sealed class IfaceType extends SchemaNode
                permits Union, Struct {
            public List<FieldNode> fields = new ArrayList<>();
            public List<IfaceType> ifaceTypes = new ArrayList<>();
            public Class<MappableIface> iface;

            <T extends FieldNode> T addField(T child) {
                fields.add(child);
                return child;
            }

            <T extends IfaceType> T addIfaceTypeNode(T child) {
                ifaceTypes.add(child);
                return child;
            }

            IfaceType(IfaceType parent, Class<MappableIface> iface) {
                super(parent);
                this.iface = iface;
            }

            IfaceType getChild(Class<?> iface){
                Optional<IfaceType> ifaceTypeNodeOptional = ifaceTypes
                        .stream()
                        .filter(n->n.iface.equals(iface))
                        .findFirst();
                if (ifaceTypeNodeOptional.isPresent()){
                    return ifaceTypeNodeOptional.get();
                }else {
                    throw new IllegalStateException("no supported iface type");
                }
            }

            public void visitTypes(int depth, Consumer<IfaceType> ifaceTypeNodeConsumer) {
                ifaceTypes.forEach(t->t.visitTypes(depth+1,ifaceTypeNodeConsumer));
                ifaceTypeNodeConsumer.accept(this);
            }

            static public GroupLayout getBoundGroupLayout(IfaceType ifaceType, BoundSchemaNode child){
                ifaceType.fields.forEach(fieldNode ->
                        fieldNode.collectLayouts(child)
                );
                return (MapperUtil.isUnion(ifaceType.iface)
                        ? MemoryLayout.unionLayout(child.memoryLayoutListToArray())
                        : MemoryLayout.structLayout(child.memoryLayoutListToArray())).withName(ifaceType.iface.getSimpleName());
            }

            /**
             * Get a layout which describes the type.
             * <p>
             * If tyoe holds a primitive (int, float) then just map to JAVA_INT, JAVA_FLOAT value layouts
             * Otherwise we look through the parent's children.  Which should include a type node struct/union matching the type.
             *
             * @param type
             * @param boundSchemaNode
             * @return
             */
            MemoryLayout getBoundLayout(Class<?> type, BoundSchemaNode boundSchemaNode) {
                if (type.isPrimitive()) {
                    return MapperUtil.primitiveToLayout(type);
                }else if (MapperUtil.isMemorySegment(type)) {
                    return ValueLayout.ADDRESS;
                } else {
                    IfaceType ifaceType = ifaceTypes.stream()
                            .filter(i -> i.iface.equals(type))
                            .findFirst().orElseThrow();
                        return getBoundGroupLayout(ifaceType, boundSchemaNode.createChild(ifaceType));
                }
            }

            public IfaceType struct(String name, Consumer<IfaceType> parentSchemaNodeConsumer) {
                parentSchemaNodeConsumer.accept(addIfaceTypeNode(new Struct(this, (Class<MappableIface>)MapperUtil.typeOf(iface, name))));
                return this;
            }

            public IfaceType union(String name, Consumer<IfaceType> parentSchemaNodeConsumer) {
                parentSchemaNodeConsumer.accept(addIfaceTypeNode(new Union(this, (Class<MappableIface>) MapperUtil.typeOf(iface, name))));
                return this;
            }

            public IfaceType field(String name) {
                var key = AccessorInfo.Key.of(iface, name);
                var typeOf = MapperUtil.typeOf(iface, name);
                addField(MapperUtil.isMemorySegment(typeOf)
                        ? new AddressField(this, key, (Class<MemorySegment>)typeOf, name)
                        : MapperUtil.isMappableIface(typeOf)
                           ? new IfaceField(this, key, this.getChild(typeOf), name)
                           : new PrimitiveField(this, key, typeOf, name));
                return this;
            }

            public IfaceType atomic(String name) {
                addField(new AtomicField(this, AccessorInfo.Key.of(iface, name), MapperUtil.typeOf(iface, name), name));
                return this;
            }

            public IfaceType pad(int len) {
                addField(new Padding(this, len));
                return this;
            }

            public IfaceType field(String name, Consumer<IfaceType> parentSchemaNodeConsumer) {
                AccessorInfo.Key fieldKey = AccessorInfo.Key.of(iface, name);
                Class<MappableIface> fieldType = (Class<MappableIface>)MapperUtil.typeOf(iface, name);
                IfaceType structOrUnion= MapperUtil.isStruct(fieldType) ? new Struct(this, fieldType) : new Union(this, fieldType);
                addIfaceTypeNode(structOrUnion);
                addField(new IfaceField(this, fieldKey, structOrUnion,name));
                parentSchemaNodeConsumer.accept(structOrUnion);
                return this;
            }

            public IfaceType fields(String name1, String name2, Consumer<IfaceType> parentSchemaNodeConsumer) {
                AccessorInfo.Key fieldKey1 = AccessorInfo.Key.of(iface, name1);
                AccessorInfo.Key fieldKey2 = AccessorInfo.Key.of(iface, name2);
                if (!fieldKey1.equals(fieldKey2)) {
                    throw new IllegalStateException("fields " + name1 + " and " + name2 + " have different keys");
                }
                Class<MappableIface> structOrUnionType = (Class<MappableIface>) MapperUtil.typeOf(iface, name1);
                Class<?> fieldTypeCheck = MapperUtil.typeOf(iface, name2);
                if (!structOrUnionType.equals(fieldTypeCheck)) {
                    throw new IllegalStateException("fields " + name1 + " and " + name2 + " have different types");
                }
                IfaceType ifaceType = MapperUtil.isStruct(iface)
                        ? new Struct(this, structOrUnionType)
                        : new Union(this, structOrUnionType);
                addIfaceTypeNode(ifaceType);
                addField(new IfaceField(this, fieldKey1, ifaceType,name1));
                addField(new IfaceField(this, fieldKey2, ifaceType,name2));

                parentSchemaNodeConsumer.accept(ifaceType);
                return this;
            }

            public IfaceType fields(String... names) {
                for (var name : names) {
                    field(name);
                }
                return this;
            }

            public IfaceType array(String name, int len) {
                AccessorInfo.Key arrayKey = AccessorInfo.Key.of(iface, name);
                var typeof = MapperUtil.typeOf(iface,name);
                addField(arrayKey.valueType().equals(ValueType.INTERFACE)
                        ?new IfaceFixedArray( this,arrayKey,this.getChild(typeof),name, len)
                        :new PrimitiveFixedArray(this, arrayKey, typeof, name, len));
                return this;
            }

            public IfaceType array(String name, int len, Consumer<IfaceType> parentFieldConsumer) {
                AccessorInfo.Key arrayKey = AccessorInfo.Key.of(iface, name);
                Class<MappableIface> structOrUnionType = (Class<MappableIface>)MapperUtil.typeOf(iface, name);
                IfaceType ifaceType = MapperUtil.isStruct(iface)
                        ? new Struct(this, structOrUnionType)
                        : new Union(this, structOrUnionType);
                parentFieldConsumer.accept(ifaceType);
                addIfaceTypeNode(ifaceType);
                addField(new IfaceFixedArray(this, arrayKey, ifaceType, name, len));
                return this;
            }

            private IfaceType fieldControlledArray(String name, List<ArrayLen> arrayLenFields, int stride) {
                AccessorInfo.Key arrayKey = AccessorInfo.Key.of(iface, name);
                var typeOf = MapperUtil.typeOf(iface, name);
                addField(arrayKey.valueType().equals(ValueType.INTERFACE)
                        ?new IfaceFieldControlledArray(this, arrayKey, this.getChild(typeOf),name,  arrayLenFields, stride)
                        :new PrimitiveFieldControlledArray(this, arrayKey, typeOf, name,  arrayLenFields, stride));
                return this;
            }

            public static class ArrayBuildState {
                IfaceType ifaceType;
                List<ArrayLen> arrayLenFields;
                int padding =0;
                int stride = 1;

                public IfaceType array(String name) {
                    return ifaceType.fieldControlledArray(name, arrayLenFields, stride);
                }

                public ArrayBuildState stride(int stride) {
                    this.stride = stride;
                    return this;
                }
                public ArrayBuildState pad(int padding) {
                    this.padding = padding;
                    var paddingField = new Padding(ifaceType, padding);
                    ifaceType.addField(paddingField);
                    return this;
                }
                public IfaceType array(String name, Consumer<IfaceType> parentFieldConsumer) {
                    Class<MappableIface> arrayType = (Class<MappableIface>) MapperUtil.typeOf(this.ifaceType.iface, name);
                    IfaceType ifaceType = MapperUtil.isStruct(arrayType)
                            ? new Struct(this.ifaceType, arrayType)
                            : new Union(this.ifaceType, arrayType);
                    parentFieldConsumer.accept(ifaceType);
                    this.ifaceType.addIfaceTypeNode(ifaceType);
                    this.ifaceType.fieldControlledArray(name, arrayLenFields, stride);


                    return this.ifaceType;
                }

                ArrayBuildState(IfaceType ifaceType, List<ArrayLen> arrayLenFields) {
                    this.ifaceType = ifaceType;
                    this.arrayLenFields = arrayLenFields;
                }
            }

            public ArrayBuildState arrayLen(String... arrayLenFieldNames) {
                List<ArrayLen> arrayLenFields = new ArrayList<>();
                Arrays.stream(arrayLenFieldNames).forEach(arrayLenFieldName -> {
                    var arrayLenField = new ArrayLen(this, AccessorInfo.Key.of(iface, arrayLenFieldName), MapperUtil.typeOf(iface, arrayLenFieldName), arrayLenFieldName);
                    addField(arrayLenField);
                    arrayLenFields.add(arrayLenField);
                });
                return new ArrayBuildState(this, arrayLenFields);
            }

            @Override
            public void toText(String indent, Consumer<String> stringConsumer) {
                stringConsumer.accept(indent);
                if (MapperUtil.isUnion(iface)) {
                    stringConsumer.accept("union");
                } else if (MapperUtil.isStructOrBuffer(iface)) {
                    stringConsumer.accept("struct");
                } else {
                    throw new IllegalStateException("Oh my ");
                }
                stringConsumer.accept(" " + iface + "{");
                stringConsumer.accept("\n");
                ifaceTypes.forEach(c -> {
                    c.toText(indent + " TYPE: ", stringConsumer);
                    stringConsumer.accept("\n");
                });
                fields.forEach(c -> {
                    c.toText(indent + " FIELD: ", stringConsumer);
                    stringConsumer.accept("\n");
                });

                stringConsumer.accept(indent);
                stringConsumer.accept("}");
            }
        }

        public static final class Struct extends IfaceType {
            Struct(IfaceType parent, Class<MappableIface> type) {
                super(parent, type);
            }
        }

        public static final class Union extends IfaceType {
            Union(IfaceType parent, Class<MappableIface> type) {
                super(parent, type);
            }
        }
        public abstract static sealed class IfaceArray extends AbstractIfaceField permits IfaceFieldControlledArray, IfaceFixedArray {
            IfaceArray(IfaceType parent, AccessorInfo.Key key, IfaceType ifaceType, String name) {
                super(parent, key, ifaceType, name);
            }
        }
        public abstract static sealed class PrimitiveArray extends AbstractPrimitiveField permits PrimitiveFieldControlledArray, PrimitiveFixedArray {
            PrimitiveArray(IfaceType parent, AccessorInfo.Key key, Class<?> type, String name) {
                super(parent, key, type, name);
            }
        }
        public static final class IfaceFixedArray extends IfaceArray {
            public int len;

            IfaceFixedArray(IfaceType parent, AccessorInfo.Key key, IfaceType ifaceType, String name, int len) {
                super(parent, key, ifaceType, name);
                this.len = len;
            }

            @Override
            public void toText(String indent, Consumer<String> stringConsumer) {
                stringConsumer.accept(indent + "array [" + len + "]");
            }

            @Override
            public void collectLayouts(BoundSchemaNode boundSchemaNode) {
                boundSchemaNode.bind(this, MemoryLayout.sequenceLayout(len,
                        parent.getBoundLayout(type.iface, boundSchemaNode).withName(type.iface.getSimpleName())
                ).withName(name));
            }
        }
        public static final class PrimitiveFixedArray extends PrimitiveArray {
            public int len;

            PrimitiveFixedArray(IfaceType parent, AccessorInfo.Key key, Class<?> type, String name, int len) {
                super(parent, key, type, name);
                this.len = len;
            }

            @Override
            public void toText(String indent, Consumer<String> stringConsumer) {
                stringConsumer.accept(indent + "primitive array [" + len + "]");
            }

            @Override
            public void collectLayouts(BoundSchemaNode boundSchemaNode) {
                boundSchemaNode.bind(this, MemoryLayout.sequenceLayout(len,
                        parent.getBoundLayout(type, boundSchemaNode).withName(type.getSimpleName())
                ).withName(name));
            }
        }

        public static final class IfaceFieldControlledArray extends IfaceArray {
            List<ArrayLen> arrayLenFields;
            int stride;
            int contributingDims;

            IfaceFieldControlledArray(IfaceType parent, AccessorInfo.Key key, IfaceType ifaceType, String name, List<ArrayLen> arrayLenFields, int stride) {
                super(parent, key, ifaceType,name);
                this.arrayLenFields = arrayLenFields;
                this.stride = stride;
                this.contributingDims = arrayLenFields.size();
            }

            @Override
            public void toText(String indent, Consumer<String> stringConsumer) {
                stringConsumer.accept(indent + name + "[" + key + ":" + type.iface + "] where len defined by " + arrayLenFields);
            }

            @Override
            public void collectLayouts(BoundSchemaNode boundSchemaNode) {
                // To determine the actual 'array' size we multiply the contributing dims by the stride .
                int size = stride; //usually 1 but developer can define.
                for (int i = 0; i < contributingDims; i++) {
                    size *= boundSchemaNode.takeArrayLen(); // this takes an arraylen and bumps the ptr
                }

                boundSchemaNode.bind(this, MemoryLayout.sequenceLayout(size,
                        parent.getBoundLayout(type.iface, boundSchemaNode).withName(type.iface.getSimpleName())
                ).withName(name));
            }
        }
        public static final class PrimitiveFieldControlledArray extends PrimitiveArray {
            List<ArrayLen> arrayLenFields;
            int stride;
            int contributingDims;

            PrimitiveFieldControlledArray(IfaceType parent, AccessorInfo.Key key, Class<?> type, String name, List<ArrayLen> arrayLenFields, int stride) {
                super(parent, key, type, name);
                this.arrayLenFields = arrayLenFields;
                this.stride = stride;
                this.contributingDims = arrayLenFields.size();
            }

            @Override
            public void toText(String indent, Consumer<String> stringConsumer) {
                stringConsumer.accept(indent + name + "[" + key + ":" + type + "] where len defined by " + arrayLenFields);
            }

            @Override
            public void collectLayouts(BoundSchemaNode boundSchemaNode) {
                // To determine the actual 'array' size we multiply the contributing dims by the stride .
                int size = stride; //usually 1 but developer can define.
                for (int i = 0; i < contributingDims; i++) {
                    size *= boundSchemaNode.takeArrayLen(); // this takes an arraylen and bumps the ptr
                }

                boundSchemaNode.bind(this, MemoryLayout.sequenceLayout(size,
                        parent.getBoundLayout(type, boundSchemaNode).withName(type.getSimpleName())
                ).withName(name));
            }
        }

    }

    Schema(Class<T> iface, SchemaNode.IfaceType rootIfaceType) {
        this.iface = iface;
        this.rootIfaceType = rootIfaceType;
    }


    public T allocate(MethodHandles.Lookup lookup,BufferAllocator bufferAllocator, int... boundLengths) {
        BoundSchema<?> boundSchema = new BoundSchemaNode.BoundSchemaRootNode<>(this, boundLengths);
        return (T) boundSchema.allocate(lookup,bufferAllocator);
    }

    public static <T extends Buffer> Schema<T> of(Class<T> iface,  Consumer<SchemaNode.IfaceType> parentFieldConsumer) {
        var struct = new SchemaNode.Struct(null, (Class<MappableIface>)(Object)iface); // why the need for this?
        parentFieldConsumer.accept(struct);
        return new Schema<>(iface,struct);
    }

    public void toText(Consumer<String> stringConsumer) {
        rootIfaceType.toText("", stringConsumer);
    }
}

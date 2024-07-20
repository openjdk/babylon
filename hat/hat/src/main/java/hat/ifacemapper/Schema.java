package hat.ifacemapper;

import hat.Accelerator;
import hat.buffer.Buffer;
import hat.ifacemapper.accessor.AccessorInfo;
import hat.ifacemapper.accessor.ValueType;

import java.lang.foreign.GroupLayout;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.function.Consumer;

public class Schema<T extends Buffer> {
    final public IfaceType rootIfaceType;
    public Class<T> iface;

    public static abstract class SchemaNode {
        public static final class Padding extends FieldNode {
            int len;
            Padding(IfaceType parent, int len) {
                super(parent, AccessorInfo.Key.NONE, "pad" + len);
                this.len = len;
            }

            @Override
            public void toText(String indent, Consumer<String> stringConsumer) {
                stringConsumer.accept(indent + "padding " + len + " bytes");
            }
        }
    }

    Schema(Class<T> iface, IfaceType rootIfaceType) {
        this.iface = iface;
        this.rootIfaceType = rootIfaceType;
    }

    public T allocate(Accelerator accelerator,  int... boundLengths) {
        BoundSchema<?> boundSchema = new BoundSchema<>(this, boundLengths);
        return (T) boundSchema.allocate(accelerator.lookup, accelerator);
    }

    public static <T extends Buffer> Schema<T> of(Class<T> iface, Consumer<IfaceType> parentFieldConsumer) {
        var struct = new IfaceType.Struct(null, (Class<MappableIface>) (Object) iface); // why the need for this?
        parentFieldConsumer.accept(struct);
        return new Schema<>(iface, struct);
    }

    public void toText(Consumer<String> stringConsumer) {
        rootIfaceType.toText("", stringConsumer);
    }

    public static abstract sealed class IfaceType
            permits IfaceType.Union, IfaceType.Struct {
        public final IfaceType parent;
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
            this.parent = parent;
            this.iface = iface;
        }

        IfaceType getChild(Class<?> iface) {
            Optional<IfaceType> ifaceTypeNodeOptional = ifaceTypes
                    .stream()
                    .filter(n -> n.iface.equals(iface))
                    .findFirst();
            if (ifaceTypeNodeOptional.isPresent()) {
                return ifaceTypeNodeOptional.get();
            } else {
                throw new IllegalStateException("no supported iface type");
            }
        }

        public void visitTypes(int depth, Consumer<IfaceType> ifaceTypeNodeConsumer) {
            ifaceTypes.forEach(t -> t.visitTypes(depth + 1, ifaceTypeNodeConsumer));
            ifaceTypeNodeConsumer.accept(this);
        }


        public IfaceType struct(String name, Consumer<IfaceType> parentSchemaNodeConsumer) {
            parentSchemaNodeConsumer.accept(addIfaceTypeNode(new Struct(this, (Class<MappableIface>) MapperUtil.typeOf(iface, name))));
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
                    ? new FieldNode.AddressField(this, key, (Class<MemorySegment>) typeOf, name)
                    : MapperUtil.isMappableIface(typeOf)
                    ? new FieldNode.IfaceField(this, key, this.getChild(typeOf), name)
                    : new FieldNode.PrimitiveField(this, key, typeOf, name));
            return this;
        }

        public IfaceType atomic(String name) {
            addField(new FieldNode.AtomicField(this, AccessorInfo.Key.of(iface, name), MapperUtil.typeOf(iface, name), name));
            return this;
        }

        public IfaceType pad(int len) {
            addField(new SchemaNode.Padding(this, len));
            return this;
        }

        public IfaceType field(String name, Consumer<IfaceType> parentSchemaNodeConsumer) {
            AccessorInfo.Key fieldKey = AccessorInfo.Key.of(iface, name);
            Class<MappableIface> fieldType = (Class<MappableIface>) MapperUtil.typeOf(iface, name);
            IfaceType structOrUnion = MapperUtil.isStruct(fieldType) ? new Struct(this, fieldType) : new Union(this, fieldType);
            addIfaceTypeNode(structOrUnion);
            addField(new FieldNode.IfaceField(this, fieldKey, structOrUnion, name));
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
            addField(new FieldNode.IfaceField(this, fieldKey1, ifaceType, name1));
            addField(new FieldNode.IfaceField(this, fieldKey2, ifaceType, name2));

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
            var typeof = MapperUtil.typeOf(iface, name);
            addField(arrayKey.valueType().equals(ValueType.INTERFACE)
                    ? new FieldNode.IfaceFixedArray(this, arrayKey, this.getChild(typeof), name, len)
                    : new FieldNode.PrimitiveFixedArray(this, arrayKey, typeof, name, len));
            return this;
        }

        public IfaceType array(String name, int len, Consumer<IfaceType> parentFieldConsumer) {
            AccessorInfo.Key arrayKey = AccessorInfo.Key.of(iface, name);
            Class<MappableIface> structOrUnionType = (Class<MappableIface>) MapperUtil.typeOf(iface, name);
            IfaceType ifaceType = MapperUtil.isStruct(iface)
                    ? new Struct(this, structOrUnionType)
                    : new Union(this, structOrUnionType);
            parentFieldConsumer.accept(ifaceType);
            addIfaceTypeNode(ifaceType);
            addField(new FieldNode.IfaceFixedArray(this, arrayKey, ifaceType, name, len));
            return this;
        }

        private IfaceType fieldControlledArray(String name, List<FieldNode.ArrayLen> arrayLenFields, int stride) {
            AccessorInfo.Key arrayKey = AccessorInfo.Key.of(iface, name);
            var typeOf = MapperUtil.typeOf(iface, name);
            addField(arrayKey.valueType().equals(ValueType.INTERFACE)
                    ? new FieldNode.IfaceFieldControlledArray(this, arrayKey, this.getChild(typeOf), name, arrayLenFields, stride)
                    : new FieldNode.PrimitiveFieldControlledArray(this, arrayKey, typeOf, name, arrayLenFields, stride));
            return this;
        }

        public static class ArrayBuildState {
            IfaceType ifaceType;
            List<FieldNode.ArrayLen> arrayLenFields;
            int padding = 0;
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
                var paddingField = new SchemaNode.Padding(ifaceType, padding);
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

            ArrayBuildState(IfaceType ifaceType, List<FieldNode.ArrayLen> arrayLenFields) {
                this.ifaceType = ifaceType;
                this.arrayLenFields = arrayLenFields;
            }
        }

        public ArrayBuildState arrayLen(String... arrayLenFieldNames) {
            List<FieldNode.ArrayLen> arrayLenFields = new ArrayList<>();
            Arrays.stream(arrayLenFieldNames).forEach(arrayLenFieldName -> {
                var arrayLenField = new FieldNode.ArrayLen(this, AccessorInfo.Key.of(iface, arrayLenFieldName), MapperUtil.typeOf(iface, arrayLenFieldName), arrayLenFieldName);
                addField(arrayLenField);
                arrayLenFields.add(arrayLenField);
            });
            return new ArrayBuildState(this, arrayLenFields);
        }


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
            ifaceTypes.forEach(ifaceType -> {
                ifaceType.toText(indent + " TYPE: ", stringConsumer);
                stringConsumer.accept("\n");
            });
            fields.forEach(field -> {
                field.toText(indent + " FIELD: ", stringConsumer);
                stringConsumer.accept("\n");
            });

            stringConsumer.accept(indent);
            stringConsumer.accept("}");
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
    }

    public static abstract sealed class FieldNode
            permits FieldNode.AddressField, FieldNode.AbstractIfaceField, SchemaNode.Padding, FieldNode.AbstractPrimitiveField {
        public IfaceType parent;
        public final AccessorInfo.Key key;
        public final String name;

        FieldNode(IfaceType parent, AccessorInfo.Key key, String name) {
            this.parent = parent;
            this.key = key;
            this.name = name;
        }

        public abstract void toText(String indent, Consumer<String> stringConsumer);


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
        }

        public static abstract sealed class AbstractPrimitiveField extends FieldNode
                permits ArrayLen, AtomicField, PrimitiveArray, PrimitiveField {
            public Class<?> type;

            AbstractPrimitiveField(IfaceType parent, AccessorInfo.Key key, Class<?> type, String name) {
                super(parent, key, name);
                this.type = type;
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


        public static final class PrimitiveField extends AbstractPrimitiveField {
            PrimitiveField(IfaceType parent, AccessorInfo.Key key, Class<?> type, String name) {
                super(parent, key, type, name);

            }

            @Override
            public void toText(String indent, Consumer<String> stringConsumer) {
                stringConsumer.accept(indent + "primitive field " + key + ":" + type);
            }
        }


        public abstract static sealed class PrimitiveArray extends AbstractPrimitiveField permits PrimitiveFieldControlledArray, PrimitiveFixedArray {
            PrimitiveArray(IfaceType parent, AccessorInfo.Key key, Class<?> type, String name) {
                super(parent, key, type, name);
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
        }

        public static abstract sealed class AbstractIfaceField extends FieldNode
                permits FieldNode.IfaceArray, FieldNode.IfaceField {
            public IfaceType ifaceType;

            AbstractIfaceField(IfaceType parent, AccessorInfo.Key key, IfaceType ifaceType, String name) {
                super(parent, key, name);
                this.ifaceType = ifaceType;
            }
        }


        public static final class IfaceField extends AbstractIfaceField {

            IfaceField(IfaceType parent, AccessorInfo.Key key, IfaceType ifaceType, String name) {
                super(parent, key, ifaceType, name);

            }

            @Override
            public void toText(String indent, Consumer<String> stringConsumer) {
                stringConsumer.accept(indent + "mappable field " + key + ":" + ifaceType.iface);
            }
        }


        public abstract static sealed class IfaceArray extends AbstractIfaceField permits IfaceFieldControlledArray, IfaceFixedArray {
            IfaceArray(IfaceType parent, AccessorInfo.Key key, IfaceType ifaceType, String name) {
                super(parent, key, ifaceType, name);
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
        }

        public static final class IfaceFieldControlledArray extends IfaceArray {
            List<ArrayLen> arrayLenFields;
            int stride;
            int contributingDims;

            IfaceFieldControlledArray(IfaceType parent, AccessorInfo.Key key, IfaceType ifaceType, String name, List<ArrayLen> arrayLenFields, int stride) {
                super(parent, key, ifaceType, name);
                this.arrayLenFields = arrayLenFields;
                this.stride = stride;
                this.contributingDims = arrayLenFields.size();
            }

            @Override
            public void toText(String indent, Consumer<String> stringConsumer) {
                stringConsumer.accept(indent + name + "[" + key + ":" + ifaceType.iface + "] where len defined by " + arrayLenFields);
            }
        }


    }
}

package hat.ifacemapper;

import hat.buffer.Buffer;
import hat.buffer.BufferAllocator;
import hat.ifacemapper.accessor.AccessorInfo;
import hat.util.Result;

import java.lang.foreign.Arena;
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
    SchemaNode.TypeSchemaNode schemaRootField;
    public Class<T> iface;

    static abstract sealed class LayoutToBoundFieldTreeNode permits ChildLayoutToBoundFieldTreeNode, BoundSchema {
        static class FieldToLayoutBinding<T extends SchemaNode> {
            final T field;
            MemoryLayout layout;
            FieldToLayoutBinding(T field, MemoryLayout layout) {
                this.field = field;
                this.layout = layout;
            }
        }

        static class FieldControlledArrayBinding extends FieldToLayoutBinding<SchemaNode.FieldControlledArray> {
            final int idx;
            final int len;

            FieldControlledArrayBinding(int idx, int len, SchemaNode.FieldControlledArray fieldControlledArray) {
                super(fieldControlledArray, null);
                this.idx = idx;
                this.len = len;
            }
        }

        final protected LayoutToBoundFieldTreeNode parent;
        final List<ChildLayoutToBoundFieldTreeNode> children = new ArrayList<>();
        final List<MemoryLayout> memoryLayouts = new ArrayList<>();
        final List<FieldToLayoutBinding<?>> fieldToLayoutBindings = new ArrayList<>();

        LayoutToBoundFieldTreeNode(LayoutToBoundFieldTreeNode parent) {
            this.parent = parent;
        }

        abstract int takeArrayLen();

        abstract FieldControlledArrayBinding createFieldControlledArrayBinding(SchemaNode.FieldControlledArray fieldControlledArray, MemoryLayout memoryLayout);

        void bind(SchemaNode field, MemoryLayout memoryLayout) {
            FieldToLayoutBinding<?> fieldToLayoutBinding = null;
            if (field instanceof SchemaNode.FieldControlledArray fieldControlledArray) {
                fieldToLayoutBinding = createFieldControlledArrayBinding(fieldControlledArray, memoryLayout);
            } else {
                fieldToLayoutBinding = new FieldToLayoutBinding<>(field, memoryLayout);
            }
            fieldToLayoutBindings.add(fieldToLayoutBinding);
            memoryLayouts.add(memoryLayout);
        }

        public MemoryLayout[] memoryLayoutListToArray() {
            return memoryLayouts.toArray(new MemoryLayout[0]);
        }

        public ChildLayoutToBoundFieldTreeNode createChild() {
            var childLayoutCollector = new ChildLayoutToBoundFieldTreeNode(this);
            children.add(childLayoutCollector);
            return childLayoutCollector;
        }
    }

    public static final class BoundSchema<T extends Buffer> extends LayoutToBoundFieldTreeNode {
        final private List<FieldControlledArrayBinding> arraySizeBindings;
        final private int[] arrayLengths;
        final Schema<T> schema;
        final public GroupLayout groupLayout;

        public BoundSchema(Schema<T> schema, int... arrayLengths) {
            super(null);
            this.schema = schema;
            this.arrayLengths = arrayLengths;
            this.arraySizeBindings = new ArrayList<>();
            LayoutToBoundFieldTreeNode scope = createChild();
            schema.schemaRootField.fields.forEach(c -> c.collectLayouts(scope));
            MemoryLayout memoryLayout = isUnion(schema.schemaRootField.type)
                    ? MemoryLayout.unionLayout(scope.memoryLayoutListToArray())
                    : MemoryLayout.structLayout(scope.memoryLayoutListToArray());
            bind(schema.schemaRootField, memoryLayout.withName(schema.iface.getSimpleName()));
            this.groupLayout = (GroupLayout) memoryLayouts.getFirst();
        }

        public T allocate(BufferAllocator bufferAllocator) {
            return bufferAllocator.allocate(SegmentMapper.of(MethodHandles.lookup(), schema.iface, groupLayout));
        }

        @Override
        int takeArrayLen() {
            return arrayLengths[arraySizeBindings.size()];
        }

        FieldControlledArrayBinding createFieldControlledArrayBinding(SchemaNode.FieldControlledArray fieldControlledArray, MemoryLayout memoryLayout) {
            int idx = arraySizeBindings.size();
            var arraySizeBinding = new FieldControlledArrayBinding(idx, arrayLengths[idx], fieldControlledArray);
            arraySizeBindings.add(arraySizeBinding);
            return arraySizeBinding;
        }
    }

    public static final class ChildLayoutToBoundFieldTreeNode extends LayoutToBoundFieldTreeNode {
        ChildLayoutToBoundFieldTreeNode(LayoutToBoundFieldTreeNode parent) {
            super(parent);
        }

        @Override
        int takeArrayLen() {
            return parent.takeArrayLen();
        }

        FieldControlledArrayBinding createFieldControlledArrayBinding(SchemaNode.FieldControlledArray fieldControlledArray, MemoryLayout memoryLayout) {
            return parent.createFieldControlledArrayBinding(fieldControlledArray, memoryLayout);
        }
    }

    /*
     * From the iface mapper
     * T foo()             getter iface|primitive  0 args                  , return T     returnType T
     * T foo(long)    arraygetter iface|primitive  arg[0]==long            , return T     returnType T
     * void foo(T)            setter       primitive  arg[0]==T               , return void  returnType T
     * void foo(long, T) arraysetter       primitive  arg[0]==long, arg[1]==T , return void  returnType T
     */
    static Class<?> typeOf(Class<?> iface, String name) {
        var methods = iface.getDeclaredMethods();
        Result<Class<?>> typeResult = new Result<>();
        Arrays.stream(methods).filter(method -> method.getName().equals(name)).forEach(matchingMethod -> {
            Class<?> returnType = matchingMethod.getReturnType();
            Class<?>[] paramTypes = matchingMethod.getParameterTypes();
            Class<?> thisType = null;
            if (paramTypes.length == 0 && (returnType.isInterface() || returnType.isPrimitive())) {
                thisType = returnType;
            } else if (paramTypes.length == 1 && paramTypes[0].isPrimitive() && returnType == Void.TYPE) {
                thisType = paramTypes[0];
            } else if (paramTypes.length == 1 && MemorySegment.class.isAssignableFrom(paramTypes[0]) && returnType == Void.TYPE) {
                thisType = paramTypes[0];
            } else if (paramTypes.length == 1 && paramTypes[0] == Long.TYPE && (returnType.isInterface() || returnType.isPrimitive())) {
                thisType = returnType;
            } else if (returnType == Void.TYPE && paramTypes.length == 2 &&
                    paramTypes[0] == Long.TYPE && paramTypes[1].isPrimitive()) {
                thisType = paramTypes[1];
            } else {
                throw new IllegalStateException("Can't determine iface mapping type for " + matchingMethod);
            }
            if (!typeResult.isPresent() || typeResult.get().equals(thisType)) {
                typeResult.of(thisType);
            } else {
                throw new IllegalStateException("type mismatch for " + name);
            }
        });
        if (!typeResult.isPresent()) {
            throw new IllegalStateException("No type mapping for " + iface + " " + name);

        }
        return typeResult.get();
    }

    static boolean isBuffer(Class<?> clazz) {
        return clazz.isInterface() && Buffer.class.isAssignableFrom(clazz);
    }

    static boolean isStruct(Class<?> clazz) {
        return clazz.isInterface() && Buffer.Struct.class.isAssignableFrom(clazz);
    }

    static boolean isStructOrBuffer(Class<?> clazz) {
        return clazz.isInterface() && (Buffer.class.isAssignableFrom(clazz) || Buffer.Struct.class.isAssignableFrom(clazz));
    }

    static boolean isUnion(Class<?> clazz) {
        return clazz.isInterface() && Buffer.Union.class.isAssignableFrom(clazz);
    }

    public static abstract class SchemaNode {
        TypeSchemaNode parent;

        SchemaNode(TypeSchemaNode parent) {
            this.parent = parent;
        }

        public abstract void toText(String indent, Consumer<String> stringConsumer);

        public static abstract sealed class FieldSchemaNode extends SchemaNode permits NamedFieldSchemaNode, Padding {
            FieldSchemaNode(TypeSchemaNode parent) {
                super(parent);
            }

            public abstract void toText(String indent, Consumer<String> stringConsumer);

            abstract void collectLayouts(LayoutToBoundFieldTreeNode layoutCollector);
        }

        public static final class Padding extends FieldSchemaNode {
            int len;

            Padding(TypeSchemaNode parent, int len) {
                super(parent);
                this.len = len;
            }

            @Override
            public void toText(String indent, Consumer<String> stringConsumer) {
                stringConsumer.accept(indent + "padding " + len + " bytes");
            }

            @Override
            void collectLayouts(LayoutToBoundFieldTreeNode layoutToFieldBindingNode) {
                layoutToFieldBindingNode.bind(this, MemoryLayout.paddingLayout(len));
            }
        }

        public static abstract sealed class NamedFieldSchemaNode extends FieldSchemaNode permits Array, ArrayLen, AtomicField, Field {
            AccessorInfo.Key key;
            Class<?> type;
            final String name;

            NamedFieldSchemaNode(TypeSchemaNode parent, AccessorInfo.Key key, Class<?> type, String name) {
                super(parent);
                this.key = key;
                this.type = type;
                this.name = name;
            }
        }


        public static final class ArrayLen extends NamedFieldSchemaNode {
            ArrayLen(TypeSchemaNode parent, AccessorInfo.Key key, Class<?> type, String name) {
                super(parent, key, type, name);
            }

            @Override
            public void toText(String indent, Consumer<String> stringConsumer) {
                stringConsumer.accept(indent + "arrayLen " + key + ":" + type);
            }

            @Override
            void collectLayouts(LayoutToBoundFieldTreeNode layoutToFieldBindingNode) {
                layoutToFieldBindingNode.bind(this, parent.getLayout(type, layoutToFieldBindingNode).withName(name));
            }
        }

        public static final class AtomicField extends NamedFieldSchemaNode {


            AtomicField(TypeSchemaNode parent, AccessorInfo.Key key, Class<?> type, String name) {
                super(parent, key, type, name);
            }

            @Override
            public void toText(String indent, Consumer<String> stringConsumer) {
                stringConsumer.accept(indent + "atomic " + key + ":" + type);
            }

            @Override
            void collectLayouts(LayoutToBoundFieldTreeNode layoutToFieldBindingNode) {
                layoutToFieldBindingNode.bind(this, parent.getLayout(type, layoutToFieldBindingNode).withName(name));
            }
        }

        public static final class Field extends NamedFieldSchemaNode {


            Field(TypeSchemaNode parent, AccessorInfo.Key key, Class<?> type, String name) {
                super(parent, key, type, name);

            }

            @Override
            public void toText(String indent, Consumer<String> stringConsumer) {
                stringConsumer.accept(indent + "field " + key + ":" + type);
            }

            @Override
            void collectLayouts(LayoutToBoundFieldTreeNode layoutToFieldBindingNode) {
                layoutToFieldBindingNode.bind(this, parent.getLayout(type, layoutToFieldBindingNode).withName(name));
            }
        }

        public static abstract sealed class TypeSchemaNode extends SchemaNode permits Union, Struct {
            private List<FieldSchemaNode> fields = new ArrayList<>();
            private List<TypeSchemaNode> types = new ArrayList<>();
            Class<?> type;

            <T extends FieldSchemaNode> T addField(T child) {
                fields.add(child);
                return child;
            }

            <T extends TypeSchemaNode> T addType(T child) {
                types.add(child);
                return child;
            }

            TypeSchemaNode(TypeSchemaNode parent, Class<?> type) {
                super(parent);
                this.type = type;
            }

            /**
             * Get a layout which describes the type.
             * <p>
             * If tyoe holds a primitive (int, float) then just map to JAVA_INT, JAVA_FLOAT value layouts
             * Otherwise we look through the parent's children.  Which should include a type node struct/union matching the type.
             *
             * @param type
             * @param layoutToFieldBindingNode
             * @return
             */
            MemoryLayout getLayout(Class<?> type, LayoutToBoundFieldTreeNode layoutToFieldBindingNode) {
                if (type.isPrimitive()) {
                    return MapperUtil.primitiveToLayout(type);
                }else    if (MemorySegment.class.isAssignableFrom(type)) {
                        return ValueLayout.ADDRESS;
                } else {
                    Optional<TypeSchemaNode> optionalTypeSchemaKeyMatchingType = types.stream()
                            .filter(typeSchemaNode -> typeSchemaNode.type.equals(type))
                            .findFirst();
                    if (optionalTypeSchemaKeyMatchingType.isPresent()) {
                        var typeSchemaKeyMatchingType = optionalTypeSchemaKeyMatchingType.get();
                        LayoutToBoundFieldTreeNode scope = layoutToFieldBindingNode.createChild();
                        typeSchemaKeyMatchingType.fields.forEach(fieldSchemaNode ->
                                fieldSchemaNode.collectLayouts(scope)
                        );
                        return isUnion(typeSchemaKeyMatchingType.type)
                                ? MemoryLayout.unionLayout(scope.memoryLayoutListToArray())
                                : MemoryLayout.structLayout(scope.memoryLayoutListToArray());
                    }else{
                        throw new IllegalStateException("Why no type");
                    }
                }
            }

            public TypeSchemaNode struct(String name, Consumer<TypeSchemaNode> parentSchemaNodeConsumer) {
                parentSchemaNodeConsumer.accept(addType(new Struct(this, typeOf(type, name))));
                return this;
            }

            public TypeSchemaNode union(String name, Consumer<TypeSchemaNode> parentSchemaNodeConsumer) {
                parentSchemaNodeConsumer.accept(addType(new Union(this, typeOf(type, name))));
                return this;
            }

            public TypeSchemaNode field(String name) {
                addField(new Field(this, AccessorInfo.Key.of(type, name), typeOf(type, name), name));
                return this;
            }

            public TypeSchemaNode atomic(String name) {
                addField(new AtomicField(this, AccessorInfo.Key.of(type, name), typeOf(type, name), name));
                return this;
            }

            public TypeSchemaNode pad(int len) {
                addField(new Padding(this, len));
                return this;
            }

            public TypeSchemaNode field(String name, Consumer<TypeSchemaNode> parentSchemaNodeConsumer) {
                AccessorInfo.Key fieldKey = AccessorInfo.Key.of(type, name);
                Class<?> fieldType = typeOf(type, name);
                addField(new Field(this, fieldKey, fieldType, name));
                TypeSchemaNode field = isStruct(fieldType) ? new SchemaNode.Struct(this, fieldType) : new SchemaNode.Union(this, fieldType);
                parentSchemaNodeConsumer.accept(addType(field));
                return this;
            }

            public TypeSchemaNode fields(String name1, String name2, Consumer<TypeSchemaNode> parentSchemaNodeConsumer) {
                AccessorInfo.Key fieldKey1 = AccessorInfo.Key.of(type, name1);
                AccessorInfo.Key fieldKey2 = AccessorInfo.Key.of(type, name2);
                if (!fieldKey1.equals(fieldKey2)) {
                    throw new IllegalStateException("fields " + name1 + " and " + name2 + " have different keys");
                }
                Class<?> structOrUnionType = typeOf(type, name1);
                Class<?> fieldTypeCheck = typeOf(type, name2);
                if (!structOrUnionType.equals(fieldTypeCheck)) {
                    throw new IllegalStateException("fields " + name1 + " and " + name2 + " have different types");
                }
                addField(new Field(this, fieldKey1, structOrUnionType, name1));
                addField(new Field(this, fieldKey2, structOrUnionType, name2));
                TypeSchemaNode typeSchemaNode = isStruct(type)
                        ? new SchemaNode.Struct(this, structOrUnionType)
                        : new SchemaNode.Union(this, structOrUnionType);
                parentSchemaNodeConsumer.accept(addType(typeSchemaNode));
                return this;
            }

            public TypeSchemaNode fields(String... names) {
                for (var name : names) {
                    field(name);
                }
                return this;
            }

            public TypeSchemaNode array(String name, int len) {
                addField(new FixedArray(this, AccessorInfo.Key.of(type, name), typeOf(type, name), name, len));
                return this;
            }

            public TypeSchemaNode array(String name, int len, Consumer<TypeSchemaNode> parentFieldConsumer) {
                AccessorInfo.Key arrayKey = AccessorInfo.Key.of(type, name);
                Class<?> structOrUnionType = typeOf(type, name);
                TypeSchemaNode typeSchemaNode = isStruct(type)
                        ? new SchemaNode.Struct(this, structOrUnionType)
                        : new SchemaNode.Union(this, structOrUnionType);
                parentFieldConsumer.accept(typeSchemaNode);
                addType(typeSchemaNode);
                addField(new FixedArray(this, arrayKey, structOrUnionType, name, len));
                return this;
            }

            private TypeSchemaNode fieldControlledArray(String name, List<ArrayLen> arrayLenFields, int stride) {
                addField(new FieldControlledArray(this, AccessorInfo.Key.of(type, name), typeOf(type, name), name,  arrayLenFields, stride));
                return this;
            }

            public static class ArrayBuildState {
                TypeSchemaNode typeSchemaNode;
                List<ArrayLen> arrayLenFields;
                int padding =0;
                int stride = 1;

                public TypeSchemaNode array(String name) {
                    return typeSchemaNode.fieldControlledArray(name, arrayLenFields, stride);
                }

                public ArrayBuildState stride(int stride) {
                    this.stride = stride;
                    return this;
                }
                public ArrayBuildState pad(int padding) {
                    this.padding = padding;
                    var paddingField = new Padding(typeSchemaNode, padding);
                    typeSchemaNode.addField(paddingField);
                    return this;
                }
                public TypeSchemaNode array(String name, Consumer<TypeSchemaNode> parentFieldConsumer) {
                    Class<?> arrayType = typeOf(typeSchemaNode.type, name);
                    this.typeSchemaNode.fieldControlledArray(name, arrayLenFields, stride);
                    TypeSchemaNode typeSchemaNode = isStruct(arrayType)
                            ? new SchemaNode.Struct(this.typeSchemaNode, arrayType)
                            : new SchemaNode.Union(this.typeSchemaNode, arrayType);
                    parentFieldConsumer.accept(typeSchemaNode);
                    this.typeSchemaNode.addType(typeSchemaNode);
                    return this.typeSchemaNode;
                }

                ArrayBuildState(TypeSchemaNode typeSchemaNode, List<ArrayLen> arrayLenFields) {
                    this.typeSchemaNode = typeSchemaNode;
                    this.arrayLenFields = arrayLenFields;
                }
            }

            public ArrayBuildState arrayLen(String... arrayLenFieldNames) {
                List<ArrayLen> arrayLenFields = new ArrayList<>();
                Arrays.stream(arrayLenFieldNames).forEach(arrayLenFieldName -> {
                    var arrayLenField = new ArrayLen(this, AccessorInfo.Key.of(type, arrayLenFieldName), typeOf(type, arrayLenFieldName), arrayLenFieldName);
                    addField(arrayLenField);
                    arrayLenFields.add(arrayLenField);
                });
                return new ArrayBuildState(this, arrayLenFields);
            }

            @Override
            public void toText(String indent, Consumer<String> stringConsumer) {
                stringConsumer.accept(indent);
                if (isUnion(type)) {
                    stringConsumer.accept("union");
                } else if (isStructOrBuffer(type)) {
                    stringConsumer.accept("struct");
                } else {
                    throw new IllegalStateException("Oh my ");
                }
                stringConsumer.accept(" " + type + "{");
                stringConsumer.accept("\n");
                types.forEach(c -> {
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

        public static final class Struct extends TypeSchemaNode {
            Struct(TypeSchemaNode parent, Class<?> type) {
                super(parent, type);
            }
        }

        public static final class Union extends TypeSchemaNode {
            Union(TypeSchemaNode parent, Class<?> type) {
                super(parent, type);
            }
        }

        public abstract static sealed class Array extends NamedFieldSchemaNode permits FieldControlledArray, FixedArray {
            Array(TypeSchemaNode parent, AccessorInfo.Key key, Class<?> type, String name) {
                super(parent, key, type, name);
            }
        }

        public static final class FixedArray extends Array {
            int len;

            FixedArray(TypeSchemaNode parent, AccessorInfo.Key key, Class<?> type, String name, int len) {
                super(parent, key, type, name);
                this.len = len;
            }

            @Override
            public void toText(String indent, Consumer<String> stringConsumer) {
                stringConsumer.accept(indent + "array [" + len + "]");
            }

            @Override
            void collectLayouts(LayoutToBoundFieldTreeNode layoutToFieldBindingNode) {
                layoutToFieldBindingNode.bind(this, MemoryLayout.sequenceLayout(len,
                        parent.getLayout(type, layoutToFieldBindingNode).withName(type.getSimpleName())
                ).withName(name));
            }
        }

        public static final class FieldControlledArray extends Array {
            List<ArrayLen> arrayLenFields;
            int stride;
          //  int padding;
            int contributingDims;

            FieldControlledArray(TypeSchemaNode parent, AccessorInfo.Key key, Class<?> type, String name, List<ArrayLen> arrayLenFields, int stride) {
                super(parent, key, type, name);
               // this.padding = padding;
                this.arrayLenFields = arrayLenFields;
                this.stride = stride;
                this.contributingDims = arrayLenFields.size();
            }

            @Override
            public void toText(String indent, Consumer<String> stringConsumer) {
                stringConsumer.accept(indent + name + "[" + key + ":" + type + "] where len defined by " + arrayLenFields);
            }

            @Override
            void collectLayouts(LayoutToBoundFieldTreeNode layoutToFieldBindingNode) {
                // To determine the actual 'array' size we multiply the contributing dims by the stride .
                int size = stride; //usually 1 but developer can define.
                for (int i = 0; i < contributingDims; i++) {
                    size *= layoutToFieldBindingNode.takeArrayLen(); // this takes an arraylen and bumps the ptr
                }

                layoutToFieldBindingNode.bind(this, MemoryLayout.sequenceLayout(size,
                        parent.getLayout(type, layoutToFieldBindingNode).withName(type.getSimpleName())
                ).withName(name));
            }
        }
    }

    Schema(Class<T> iface, SchemaNode.TypeSchemaNode schemaRootField) {
        this.iface = iface;
        this.schemaRootField = schemaRootField;
    }

    public final static BufferAllocator GlobalArenaAllocator = new BufferAllocator() {
        public <T extends Buffer> T allocate(SegmentMapper<T> s) {
            return s.allocate(Arena.global(), new HatData() {});
        }
    };

    public BoundSchema<T> boundSchema(int... boundLengths) {
        return new BoundSchema<>(this, boundLengths);
    }

    public T allocate(BufferAllocator bufferAllocator, int... boundLengths) {
        return boundSchema(boundLengths).allocate(bufferAllocator);
    }

    public T allocate(int... boundLengths) {
        return allocate(GlobalArenaAllocator, boundLengths);
    }

    public static <T extends Buffer> Schema<T> of(Class<T> iface, Consumer<SchemaNode.TypeSchemaNode> parentFieldConsumer) {
        var struct = new SchemaNode.Struct(null, iface);
        parentFieldConsumer.accept(struct);
        return new Schema<>(iface, struct);
    }

    public void toText(Consumer<String> stringConsumer) {
        schemaRootField.toText("", stringConsumer);
    }
}

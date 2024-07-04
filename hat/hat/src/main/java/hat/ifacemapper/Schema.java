package hat.ifacemapper;

import hat.buffer.Buffer;
import hat.buffer.BufferAllocator;
import hat.util.Result;

import java.lang.foreign.Arena;
import java.lang.foreign.GroupLayout;
import java.lang.foreign.MemoryLayout;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.Consumer;

import static java.lang.foreign.ValueLayout.JAVA_BOOLEAN;
import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_CHAR;
import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;
import static java.lang.foreign.ValueLayout.JAVA_SHORT;

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
                fieldToLayoutBinding = new FieldToLayoutBinding(field, memoryLayout);
            }
            fieldToLayoutBindings.add(fieldToLayoutBinding);
            memoryLayouts.add(memoryLayout);
        }

        public MemoryLayout[] array() {
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
        public BoundSchema(Schema<T> schema, int ...arrayLengths) {
            super(null);
            this.schema = schema;
            this.arrayLengths = arrayLengths;
            this.arraySizeBindings = new ArrayList<>();
            LayoutToBoundFieldTreeNode scope = createChild();
            schema.schemaRootField.fields.forEach(c -> c.collectLayouts(scope));
            MemoryLayout memoryLayout = isUnion(schema.schemaRootField.type)
                    ?MemoryLayout.unionLayout(scope.array())
                    :MemoryLayout.structLayout(scope.array());
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
    enum Mode {
        PRIMITIVE_GETTER_AND_SETTER,
        PRIMITIVE_GETTER,
        PRIMITIVE_SETTER,
        IFACE_GETTER,
        PRIMITIVE_ARRAY_SETTER,
        PRIMITIVE_ARRAY_GETTER,
        PRIMITIVE_ARRAY_GETTER_AND_SETTER,
        IFACE_ARRAY_GETTER;

        /**
         * From the iface mapper we get these mappings
         * <p>
         * T foo()             getter iface|primitive  0 args                  , return T     returnType T
         * T foo(long)    arraygetter iface|primitive  arg[0]==long            , return T     returnType T
         * void foo(T)            setter       primitive  arg[0]==T               , return void  returnType T
         * void foo(long, T) arraysetter       primitive  arg[0]==long, arg[1]==T , return void  returnType T
         *
         * @param m The reflected method
         * @return Class represeting the type this method is mapped to
         */
        static Mode of(Method m) {
            Class<?> returnType = m.getReturnType();
            Class<?>[] paramTypes = m.getParameterTypes();
            if (paramTypes.length == 0 && returnType.isInterface()) {
                return IFACE_GETTER;
            } else if (paramTypes.length == 0 && returnType.isPrimitive()) {
                return PRIMITIVE_GETTER;
            } else if (paramTypes.length == 1 && paramTypes[0].isPrimitive() && returnType == Void.TYPE) {
                return PRIMITIVE_SETTER;
            } else if (paramTypes.length == 1 && paramTypes[0] == Long.TYPE && returnType.isInterface()) {
                return IFACE_ARRAY_GETTER;
            } else if (paramTypes.length == 1 && paramTypes[0] == Long.TYPE && returnType.isPrimitive()) {
                return PRIMITIVE_ARRAY_GETTER;
            } else if (returnType == Void.TYPE && paramTypes.length == 2 &&
                    paramTypes[0] == Long.TYPE && paramTypes[1].isPrimitive()) {
                return PRIMITIVE_ARRAY_SETTER;
            } else {
                throw new IllegalStateException("no possible mode for "+m);
            }
        }
    }

    static Mode modeOf(Class<?> iface, String name) {
        var methods = iface.getDeclaredMethods();
        Result<Mode> modeResult = new Result<>();
        Arrays.stream(methods).filter(method -> method.getName().equals(name)).forEach(matchingMethod -> {
            var thisMode = Mode.of(matchingMethod);
            if (!modeResult.isPresent()){
                modeResult.of(thisMode);
            } else if (( modeResult.get().equals(Mode.PRIMITIVE_ARRAY_GETTER) && thisMode.equals(Mode.PRIMITIVE_ARRAY_SETTER))
                    || ( modeResult.get().equals(Mode.PRIMITIVE_ARRAY_SETTER) && thisMode.equals(Mode.PRIMITIVE_ARRAY_GETTER))) {
                modeResult.of(Mode.PRIMITIVE_ARRAY_GETTER_AND_SETTER);
            } else if (( modeResult.get().equals(Mode.PRIMITIVE_GETTER) && thisMode.equals(Mode.PRIMITIVE_SETTER))
                    || ( modeResult.get().equals(Mode.PRIMITIVE_SETTER) && thisMode.equals(Mode.PRIMITIVE_GETTER))) {
                modeResult.of(Mode.PRIMITIVE_GETTER_AND_SETTER);
            }
        });
        if ( !modeResult.isPresent() ) {
            throw new IllegalStateException("no possible mode for "+iface+" "+name);
           // modeResult.of(Mode.PRIMITIVE_GETTER_AND_SETTER);
        }
        return modeResult.get();
    }
    /**
     * From the iface mapper
     * T foo()             getter iface|primitive  0 args                  , return T     returnType T
     * T foo(long)    arraygetter iface|primitive  arg[0]==long            , return T     returnType T
     * void foo(T)            setter       primitive  arg[0]==T               , return void  returnType T
     * void foo(long, T) arraysetter       primitive  arg[0]==long, arg[1]==T , return void  returnType T
     *
     * @param iface The reflected method
     * @return Class represeting the type this method is mapped to
     */
    static Class<?> typeOf(Class<?> iface, String name) {
        var methods = iface.getDeclaredMethods();
        Result<Class<?>> typeResult = new Result<>();
        Arrays.stream(methods).filter(method -> method.getName().equals(name)).forEach(matchingMethod -> {
            Class<?> returnType = matchingMethod.getReturnType();
            Class<?>[] paramTypes = matchingMethod.getParameterTypes();
            Class<?> thisType = null;
            if (paramTypes.length == 0 && (returnType.isInterface() || returnType.isPrimitive())) {
                thisType= returnType;
            } else if (paramTypes.length == 1 && paramTypes[0].isPrimitive() && returnType == Void.TYPE) {
                thisType=  paramTypes[0];
            } else if (paramTypes.length == 1 && paramTypes[0] == Long.TYPE && (returnType.isInterface() || returnType.isPrimitive())) {
                thisType=  returnType;
            } else if (returnType == Void.TYPE && paramTypes.length == 2 &&
                    paramTypes[0] == Long.TYPE && paramTypes[1].isPrimitive()) {
                thisType=  paramTypes[1];
            } else {
                throw new IllegalStateException("Can't determine iface mapping type for "+matchingMethod);
            }
            if (!typeResult.isPresent() || typeResult.get().equals(thisType)) {
                typeResult.of(thisType);
            } else  {
                throw new IllegalStateException("type mismatch for " + name);
            }
        });
        if (!typeResult.isPresent()) {
            throw new IllegalStateException("No type mapping for "+iface+" "+name);

        }
        return typeResult.get();
    }

    static boolean isBuffer(Class<?> clazz) {
        return clazz.isInterface() && Buffer.class.isAssignableFrom(clazz);
    }

    static boolean isStruct(Class<?> clazz) {
        return clazz.isInterface() && Buffer.StructChild.class.isAssignableFrom(clazz);
    }

    static boolean isStructOrBuffer(Class<?> clazz) {
        return clazz.isInterface() && (Buffer.class.isAssignableFrom(clazz) || Buffer.StructChild.class.isAssignableFrom(clazz));
    }

    static boolean isUnion(Class<?> clazz) {
        return clazz.isInterface() && Buffer.UnionChild.class.isAssignableFrom(clazz);
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
            Mode mode;
            Class<?> type;
            final String name;
            NamedFieldSchemaNode(TypeSchemaNode parent, Mode mode, Class<?> type, String name) {
                super(parent);
                this.mode = mode;
                 this.type = type;
                this.name = name;
            }
        }


        public static final class ArrayLen extends NamedFieldSchemaNode {
            ArrayLen(TypeSchemaNode parent, Mode mode, Class<?> type,  String name) {
                super(parent,mode,type, name);
            }

            @Override
            public void toText(String indent, Consumer<String> stringConsumer) {
                stringConsumer.accept(indent + "arrayLen " + mode+":"+type);
            }

            @Override
            void collectLayouts(LayoutToBoundFieldTreeNode layoutToFieldBindingNode) {
                layoutToFieldBindingNode.bind(this, parent.getLayout(type, layoutToFieldBindingNode).withName(name));
            }
        }

        public static final class AtomicField extends NamedFieldSchemaNode {


            AtomicField(TypeSchemaNode parent,  Mode mode, Class<?> type, String name) {
                super(parent, mode, type,name);
            }

            @Override
            public void toText(String indent, Consumer<String> stringConsumer) {
                stringConsumer.accept(indent + "atomic " + mode+":"+type);
            }

            @Override
            void collectLayouts(LayoutToBoundFieldTreeNode layoutToFieldBindingNode) {
                layoutToFieldBindingNode.bind(this, parent.getLayout(type, layoutToFieldBindingNode).withName(name));
            }
        }

        public static final class Field extends NamedFieldSchemaNode {


            Field(TypeSchemaNode parent,  Mode mode, Class<?> type, String name) {
                super(parent, mode, type,name);

            }

            @Override
            public void toText(String indent, Consumer<String> stringConsumer) {
                stringConsumer.accept(indent + "field " + mode+":"+type);
            }

            @Override
            void collectLayouts(LayoutToBoundFieldTreeNode layoutToFieldBindingNode) {
                layoutToFieldBindingNode.bind(this, parent.getLayout(type, layoutToFieldBindingNode).withName(name));
            }
        }

        public static abstract sealed class TypeSchemaNode extends SchemaNode permits Union,Struct {
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

            TypeSchemaNode(TypeSchemaNode parent,  Class<?> type) {
                super(parent);
                this.type = type;
            }
            /**
             * Get a layout which describes the NameTypeAndMode.
             * <p>
             * If NameTypeAndMode holds a primitive (int, float) then just map to JAVA_INT, JAVA_FLOAT value layouts
             * Otherwise we look through the parent's children.  Which should include a type node struct/union matching the type.
             *
             * @param type
             * @param layoutToFieldBindingNode
             * @return
             */
             MemoryLayout getLayout(Class<?> type, LayoutToBoundFieldTreeNode layoutToFieldBindingNode) {
                MemoryLayout memoryLayout = null;
                if (type == Integer.TYPE) {
                    memoryLayout = JAVA_INT;
                } else if (type == Float.TYPE) {
                    memoryLayout = JAVA_FLOAT;
                } else if (type == Long.TYPE) {
                    memoryLayout = JAVA_LONG;
                } else if (type == Double.TYPE) {
                    memoryLayout = JAVA_DOUBLE;
                } else if (type == Short.TYPE) {
                    memoryLayout = JAVA_SHORT;
                } else if (type == Character.TYPE) {
                    memoryLayout = JAVA_CHAR;
                } else if (type == Byte.TYPE) {
                    memoryLayout = JAVA_BYTE;
                } else if (type == Boolean.TYPE) {
                    memoryLayout = JAVA_BOOLEAN;
                } else {
                    TypeSchemaNode typeSchemaModeMatchingType = types.stream()
                            .filter(typeSchemaNode -> typeSchemaNode.type.equals(type)).findFirst().get();
                    LayoutToBoundFieldTreeNode scope = layoutToFieldBindingNode.createChild();
                    typeSchemaModeMatchingType.fields.stream()
                            .forEach(fieldSchemaNode ->
                                fieldSchemaNode.collectLayouts(scope)
                            );
                    if (isUnion(typeSchemaModeMatchingType.type)) {
                        memoryLayout = MemoryLayout.unionLayout(scope.array());
                    } else if (isStructOrBuffer(typeSchemaModeMatchingType.type)) {
                        memoryLayout = MemoryLayout.structLayout(scope.array());
                    } else {
                        throw new IllegalStateException("Recursing through layout collections and came across  " + typeSchemaModeMatchingType.type);
                    }
                }
                return memoryLayout;
            }


            public TypeSchemaNode struct(String name, Consumer<TypeSchemaNode> parentSchemaNodeConsumer) {
                parentSchemaNodeConsumer.accept(addType(new Struct(this,  typeOf(type,name))));
                return this;
            }

            public TypeSchemaNode union(String name, Consumer<TypeSchemaNode> parentSchemaNodeConsumer) {
                parentSchemaNodeConsumer.accept(addType(new Union(this, typeOf(type,name))));
                return this;
            }

            public TypeSchemaNode field(String name) {
                addField(new Field(this, modeOf(type, name), typeOf(type, name), name));
                return this;
            }

            public TypeSchemaNode atomic(String name) {
                addField(new AtomicField(this, modeOf(type, name), typeOf(type, name), name));
                return this;
            }

            public TypeSchemaNode pad(int len) {
                addField(new Padding(this, len));
                return this;
            }

            public TypeSchemaNode field(String name, Consumer<TypeSchemaNode> parentSchemaNodeConsumer) {
                 Mode fieldMode = modeOf(type, name);
                 Class<?> fieldType = typeOf(type, name);
                addField(new Field(this,fieldMode, fieldType, name));
                TypeSchemaNode field = isStruct(fieldType)?new SchemaNode.Struct(this, fieldType):new SchemaNode.Union(this, fieldType);
                parentSchemaNodeConsumer.accept(addType(field));
                return this;
            }

            public TypeSchemaNode fields(String name1, String name2, Consumer<TypeSchemaNode> parentSchemaNodeConsumer) {
                Mode fieldMode1 = modeOf(type, name1);
                Mode fieldMode2 = modeOf(type, name2);
                if (!fieldMode1.equals(fieldMode2)){
                    throw new IllegalStateException("fields "+name1+" and "+name2+" have different modes");
                }
                Class<?> structOrUnionType = typeOf(type, name1);
                Class<?> fieldTypeCheck = typeOf(type, name2);
                if (!structOrUnionType.equals(fieldTypeCheck)){
                    throw new IllegalStateException("fields "+name1+" and "+name2+" have different types");
                }
                addField(new Field(this,fieldMode1, structOrUnionType, name1));
                addField(new Field(this,fieldMode2,  structOrUnionType, name2));
                TypeSchemaNode typeSchemaNode=isStruct(type)
                        ? new SchemaNode.Struct(this,  structOrUnionType)
                        :new SchemaNode.Union(this,  structOrUnionType);
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
                addField(new FixedArray(this, modeOf(type,name), typeOf(type, name),name, len));
                return this;
            }

            public TypeSchemaNode array(String name, int len, Consumer<TypeSchemaNode> parentFieldConsumer) {
                Mode arrayMode = modeOf(type, name);
                Class<?> structOrUnionType = typeOf(type, name);
               // ModeAndType newAccessStyle = ModeAndType.of(type, name);
                TypeSchemaNode typeSchemaNode = isStruct(type)
                                ?new SchemaNode.Struct(this, structOrUnionType)
                                :new SchemaNode.Union(this, structOrUnionType);
                parentFieldConsumer.accept(typeSchemaNode);
                addType(typeSchemaNode);
                addField(new FixedArray(this, arrayMode,structOrUnionType, name, len));
                return this;
            }

            private TypeSchemaNode fieldControlledArray(String name, ArrayLen arrayLen) {
                addField(new FieldControlledArray(this,  modeOf(type, name), typeOf(type, name),name, arrayLen));
                return this;
            }

            public static class ArrayBuildState {
                TypeSchemaNode typeSchemaNode;
                ArrayLen arrayLenField;

                public TypeSchemaNode array(String name) {
                    return typeSchemaNode.fieldControlledArray(name, arrayLenField);
                }

                public TypeSchemaNode array(String name, Consumer<TypeSchemaNode> parentFieldConsumer) {
                    Class<?> arrayType = typeOf(typeSchemaNode.type, name);
                    this.typeSchemaNode.fieldControlledArray(name, arrayLenField);
                    TypeSchemaNode typeSchemaNode =isStruct(arrayType)
                            ?new SchemaNode.Struct(this.typeSchemaNode, arrayType)
                            :new SchemaNode.Union(this.typeSchemaNode,arrayType);
                    parentFieldConsumer.accept(typeSchemaNode);
                    this.typeSchemaNode.addType(typeSchemaNode);
                    return this.typeSchemaNode;
                }

                ArrayBuildState(TypeSchemaNode typeSchemaNode, ArrayLen arrayLenField) {
                    this.typeSchemaNode = typeSchemaNode;
                    this.arrayLenField = arrayLenField;
                }
            }

            public ArrayBuildState arrayLen(String arrayLenFieldName) {
                var arrayLenField = new ArrayLen(this, modeOf(type, arrayLenFieldName), typeOf(type, arrayLenFieldName),arrayLenFieldName );
                addField(arrayLenField);
                return new ArrayBuildState(this, arrayLenField);
            }

            public void flexArray(String name) {
                 throw new IllegalStateException("flex array");
              //  addField(new FlexArray(this,null, name));
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
            Struct(TypeSchemaNode parent,  Class<?> type) {
                super(parent, type);
            }
        }

        public static final class Union extends TypeSchemaNode {
            Union(TypeSchemaNode parent,  Class<?> type) {
                super(parent,  type);
            }
        }

        public abstract static sealed class Array extends NamedFieldSchemaNode permits FieldControlledArray, FixedArray, FlexArray {
          //  ModeAndType elementAccessStyle;
            Array(TypeSchemaNode parent,  Mode mode, Class<?> type, String name) {
                super(parent, mode,type, name);
            ///    this.elementAccessStyle = elementAccessStyle;
            }
        }

        public static final class FixedArray extends Array {
            int len;

            FixedArray(TypeSchemaNode parent,Mode mode, Class<?> type, String name, int len) {
                super(parent,  mode, type, name);
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

        public static final  class FlexArray extends Array {
            FlexArray(TypeSchemaNode parent,  Mode mode, Class<?> type, String name) {
                super(parent,  mode,type, name);
            }

            @Override
            public void toText(String indent, Consumer<String> stringConsumer) {
                stringConsumer.accept(indent + "array [?] ");
            }

            void collectLayouts(LayoutToBoundFieldTreeNode layoutToFieldBindingNode) {
                layoutToFieldBindingNode.bind(this,
                        MemoryLayout.sequenceLayout(0,
                                parent.getLayout(type, layoutToFieldBindingNode).withName(type.getSimpleName())
                        ).withName(name));
            }
        }

        public static final class FieldControlledArray extends Array {
            ArrayLen arrayLen;

            FieldControlledArray(TypeSchemaNode parent,   Mode mode, Class<?> type,String name, ArrayLen arrayLen) {
                super(parent, mode, type, name);
                this.arrayLen = arrayLen;
            }

            @Override
            public void toText(String indent, Consumer<String> stringConsumer) {
                stringConsumer.accept(indent + name + "[" + mode+":"+type + "] where len defined by " + arrayLen.type);
            }

            @Override
            void collectLayouts(LayoutToBoundFieldTreeNode layoutToFieldBindingNode) {
                layoutToFieldBindingNode.bind(this, MemoryLayout.sequenceLayout(
                        layoutToFieldBindingNode.takeArrayLen(),
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
            return s.allocate(Arena.global());
        }
    };
    public BoundSchema<T> boundSchema(int... boundLengths) {
        return new BoundSchema<>(this, boundLengths);
    }

    public T allocate(BufferAllocator bufferAllocator,int... boundLengths) {
        return boundSchema(boundLengths).allocate(bufferAllocator);
    }
    public T allocate(int... boundLengths) {
        return allocate(GlobalArenaAllocator,boundLengths);
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

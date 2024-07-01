package hat;

import hat.buffer.Buffer;
import hat.buffer.BufferAllocator;
import hat.ifacemapper.SegmentMapper;

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
        static class FieldLayoutBinding<T extends SchemaNode> {
            final T field;
            MemoryLayout layout;
            FieldLayoutBinding(T field, MemoryLayout layout) {
                this.field = field;
                this.layout = layout;
            }
        }

        static class FieldControlledArrayBinding extends FieldLayoutBinding<SchemaNode.FieldControlledArray> {
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
        final List<FieldLayoutBinding> fieldToLayoutBindings = new ArrayList<>();

        LayoutToBoundFieldTreeNode(LayoutToBoundFieldTreeNode parent) {
            this.parent = parent;
        }

        abstract int takeArrayLen();

        abstract FieldControlledArrayBinding createFieldControlledArrayBinding(SchemaNode.FieldControlledArray fieldControlledArray, MemoryLayout memoryLayout);

        void bind(SchemaNode field, MemoryLayout memoryLayout) {
            FieldLayoutBinding fieldLayoutBinding = null;
            if (field instanceof SchemaNode.FieldControlledArray fieldControlledArray) {
                fieldLayoutBinding = createFieldControlledArrayBinding(fieldControlledArray, memoryLayout);
            } else {
                fieldLayoutBinding = new FieldLayoutBinding(field, memoryLayout);
            }
            fieldToLayoutBindings.add(fieldLayoutBinding);
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
            MemoryLayout memoryLayout = isUnion(schema.schemaRootField.nameTypeAndMode.type)
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

    public static class NameTypeAndMode {
        enum Mode {
            ROOT,
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
             *
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
                    System.out.println("skiping " + m);
                    return null;
                }
            }
        }

        Mode mode;
        Class<?> type;
        String name;

        NameTypeAndMode(Mode mode, Class<?> type, String name) {
            this.mode = mode;
            this.type = type;
            this.name = name;
        }

        @Override
        public String toString() {
            return mode.name() + ":" + type.getSimpleName() + ":" + name;
        }

        /**
         * From the iface mapper
         * T foo()             getter iface|primitive  0 args                  , return T     returnType T
         * T foo(long)    arraygetter iface|primitive  arg[0]==long            , return T     returnType T
         * void foo(T)            setter       primitive  arg[0]==T               , return void  returnType T
         * void foo(long, T) arraysetter       primitive  arg[0]==long, arg[1]==T , return void  returnType T
         *
         * @param m The reflected method
         * @return Class represeting the type this method is mapped to
         */
        static Class<?> methodToType(Method m) {
            Class<?> returnType = m.getReturnType();
            Class<?>[] paramTypes = m.getParameterTypes();
            if (paramTypes.length == 0 && (returnType.isInterface() || returnType.isPrimitive())) {
                return returnType;
            } else if (paramTypes.length == 1 && paramTypes[0].isPrimitive() && returnType == Void.TYPE) {
                return paramTypes[0];
            } else if (paramTypes.length == 1 && paramTypes[0] == Long.TYPE && (returnType.isInterface() || returnType.isPrimitive())) {
                return returnType;
            } else if (returnType == Void.TYPE && paramTypes.length == 2 &&
                    paramTypes[0] == Long.TYPE && paramTypes[1].isPrimitive()) {
                return paramTypes[1];
            } else {
                System.out.println("skipping " + m);
                return null;
            }
        }

        static NameTypeAndMode of(Class<?> iface, String name) {
            NameTypeAndMode accessStyle = new NameTypeAndMode(null, null, name);
            var methods = iface.getDeclaredMethods();
            Arrays.stream(methods).filter(method -> method.getName().equals(name)).forEach(matchingMethod -> {
                NameTypeAndMode.Mode mode = NameTypeAndMode.Mode.of(matchingMethod);
                Class<?> type = methodToType(matchingMethod);
                accessStyle.type = type;
                if (accessStyle.type == null) {
                    accessStyle.type = type;
                } else if (!accessStyle.type.equals(type)) {
                    throw new IllegalStateException("type mismatch for " + name);
                }
                if (accessStyle.mode == null){
                    // We don't have one already
                    accessStyle.mode = mode;
                } else if ((accessStyle.mode.equals(Mode.PRIMITIVE_ARRAY_GETTER) && mode.equals(Mode.PRIMITIVE_ARRAY_SETTER))
                        || (accessStyle.mode.equals(Mode.PRIMITIVE_ARRAY_SETTER) && mode.equals(Mode.PRIMITIVE_ARRAY_GETTER))) {
                    // mode was already an array getter or setter and is now a GETTER_AND_SETTER
                    accessStyle.mode = Mode.PRIMITIVE_ARRAY_GETTER_AND_SETTER;
                } else if ((accessStyle.mode.equals(Mode.PRIMITIVE_GETTER) && mode.equals(Mode.PRIMITIVE_SETTER))
                        || (accessStyle.mode.equals(Mode.PRIMITIVE_SETTER) && mode.equals(Mode.PRIMITIVE_GETTER))) {
                    // mode was already a primitive getter or setter and is now a GETTER_AND_SETTER
                    accessStyle.mode= Mode.PRIMITIVE_GETTER_AND_SETTER;
                }

            });
            if (accessStyle.type == null && accessStyle.mode == null) {
                accessStyle.type = iface;
                accessStyle.name = "root";
                accessStyle.mode = Mode.ROOT;
            }
            return accessStyle;
        }
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

        public static abstract sealed class FieldSchemaNode extends SchemaNode permits Array, ArrayLen, AtomicField, Field, Padding {
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

        public static final class ArrayLen extends FieldSchemaNode {
            NameTypeAndMode nameTypeAndMode;

            ArrayLen(TypeSchemaNode parent, NameTypeAndMode nameTypeAndMode) {
                super(parent);
                this.nameTypeAndMode = nameTypeAndMode;
            }

            @Override
            public void toText(String indent, Consumer<String> stringConsumer) {
                stringConsumer.accept(indent + "arrayLen " + nameTypeAndMode);
            }

            @Override
            void collectLayouts(LayoutToBoundFieldTreeNode layoutToFieldBindingNode) {
                layoutToFieldBindingNode.bind(this, parent.getLayout(nameTypeAndMode, layoutToFieldBindingNode).withName(nameTypeAndMode.name));
            }
        }

        public static final class AtomicField extends FieldSchemaNode {
            NameTypeAndMode nameTypeAndMode;

            AtomicField(TypeSchemaNode parent, NameTypeAndMode nameTypeAndMode) {
                super(parent);
                this.nameTypeAndMode = nameTypeAndMode;
            }

            @Override
            public void toText(String indent, Consumer<String> stringConsumer) {
                stringConsumer.accept(indent + "atomic " + nameTypeAndMode);
            }

            @Override
            void collectLayouts(LayoutToBoundFieldTreeNode layoutToFieldBindingNode) {
                layoutToFieldBindingNode.bind(this, parent.getLayout(nameTypeAndMode, layoutToFieldBindingNode).withName(nameTypeAndMode.name));
            }
        }

        public static final class Field extends FieldSchemaNode {
            NameTypeAndMode nameTypeAndMode;

            Field(TypeSchemaNode parent, NameTypeAndMode nameTypeAndMode) {
                super(parent);
                this.nameTypeAndMode = nameTypeAndMode;
            }

            @Override
            public void toText(String indent, Consumer<String> stringConsumer) {
                stringConsumer.accept(indent + "field " + nameTypeAndMode);
            }

            @Override
            void collectLayouts(LayoutToBoundFieldTreeNode layoutToFieldBindingNode) {
                layoutToFieldBindingNode.bind(this, parent.getLayout(nameTypeAndMode, layoutToFieldBindingNode).withName(nameTypeAndMode.name));
            }
        }

        public static abstract sealed class TypeSchemaNode extends SchemaNode permits Union,Struct {
            private List<FieldSchemaNode> fields = new ArrayList<>();
            private List<TypeSchemaNode> types = new ArrayList<>();
            NameTypeAndMode nameTypeAndMode;

            <T extends FieldSchemaNode> T addField(T child) {
                fields.add(child);
                return child;
            }
            <T extends TypeSchemaNode> T addType(T child) {
                types.add(child);
                return child;
            }

            TypeSchemaNode(TypeSchemaNode parent, NameTypeAndMode nameTypeAndMode) {
                super(parent);
                this.nameTypeAndMode = nameTypeAndMode;
            }
            /**
             * Get a layout which describes the NameTypeAndMode.
             * <p>
             * If NameTypeAndMode holds a primitive (int, float) then just map to JAVA_INT, JAVA_FLOAT value layouts
             * Otherwise we look through the parent's children.  Which should include a type node struct/union matching the type.
             *
             * @param nameTypeAndMode
             * @param layoutToFieldBindingNode
             * @return
             */
             MemoryLayout getLayout(NameTypeAndMode nameTypeAndMode, LayoutToBoundFieldTreeNode layoutToFieldBindingNode) {
                MemoryLayout memoryLayout = null;
                if (nameTypeAndMode.type == Integer.TYPE) {
                    memoryLayout = JAVA_INT;
                } else if (nameTypeAndMode.type == Float.TYPE) {
                    memoryLayout = JAVA_FLOAT;
                } else if (nameTypeAndMode.type == Long.TYPE) {
                    memoryLayout = JAVA_LONG;
                } else if (nameTypeAndMode.type == Double.TYPE) {
                    memoryLayout = JAVA_DOUBLE;
                } else if (nameTypeAndMode.type == Short.TYPE) {
                    memoryLayout = JAVA_SHORT;
                } else if (nameTypeAndMode.type == Character.TYPE) {
                    memoryLayout = JAVA_CHAR;
                } else if (nameTypeAndMode.type == Byte.TYPE) {
                    memoryLayout = JAVA_BYTE;
                } else if (nameTypeAndMode.type == Boolean.TYPE) {
                    memoryLayout = JAVA_BOOLEAN;
                } else {
                    TypeSchemaNode o = types.stream()
                            .filter(p -> p.nameTypeAndMode.type.equals(nameTypeAndMode.type)).findFirst().get();
                    LayoutToBoundFieldTreeNode scope = layoutToFieldBindingNode.createChild();
                    o.fields.stream()
                            .forEach(fieldSchemaNode -> {
                                fieldSchemaNode.collectLayouts(scope);
                            });
                    if (isUnion(o.nameTypeAndMode.type)) {
                        memoryLayout = MemoryLayout.unionLayout(scope.array());
                    } else if (isStructOrBuffer(o.nameTypeAndMode.type)) {
                        memoryLayout = MemoryLayout.structLayout(scope.array());
                    } else {
                        throw new IllegalStateException("Recursing through layout collections and came across  " + o.nameTypeAndMode.type);
                    }
                }
                return memoryLayout;
            }


            public TypeSchemaNode struct(String name, Consumer<TypeSchemaNode> parentSchemaNodeConsumer) {
                parentSchemaNodeConsumer.accept(addType(new Struct(this, NameTypeAndMode.of(nameTypeAndMode.type, name))));
                return this;
            }

            public TypeSchemaNode union(String name, Consumer<TypeSchemaNode> parentSchemaNodeConsumer) {
                parentSchemaNodeConsumer.accept(addType(new Union(this, NameTypeAndMode.of(nameTypeAndMode.type, name))));
                return this;
            }

            public TypeSchemaNode field(String name) {
                addField(new Field(this, NameTypeAndMode.of(nameTypeAndMode.type, name)));
                return this;
            }

            public TypeSchemaNode atomic(String name) {
                addField(new AtomicField(this, NameTypeAndMode.of(nameTypeAndMode.type, name)));
                return this;
            }

            public TypeSchemaNode pad(int len) {
                addField(new Padding(this, len));
                return this;
            }

            public TypeSchemaNode field(String name, Consumer<TypeSchemaNode> parentSchemaNodeConsumer) {
                NameTypeAndMode newAccessStyle = NameTypeAndMode.of(nameTypeAndMode.type, name);
                addField(new Field(this, newAccessStyle));
                TypeSchemaNode field = isStruct(newAccessStyle.type)?new SchemaNode.Struct(this, newAccessStyle):new SchemaNode.Union(this, newAccessStyle);
                parentSchemaNodeConsumer.accept(addType(field));
                return this;
            }

            public TypeSchemaNode fields(String name1, String name2, Consumer<TypeSchemaNode> parentSchemaNodeConsumer) {
                NameTypeAndMode newAccessStyle1 = NameTypeAndMode.of(nameTypeAndMode.type, name1);
                NameTypeAndMode newAccessStyle2 = NameTypeAndMode.of(nameTypeAndMode.type, name2);
                addField(new Field(this, newAccessStyle1));
                addField(new Field(this, newAccessStyle2));
                TypeSchemaNode typeSchemaNode=isStruct(newAccessStyle1.type)
                        ? new SchemaNode.Struct(this, newAccessStyle1)
                        :new SchemaNode.Union(this, newAccessStyle2);
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
                addField(new FixedArray(this, name, NameTypeAndMode.of(nameTypeAndMode.type, name), len));
                return this;
            }

            public TypeSchemaNode array(String name, int len, Consumer<TypeSchemaNode> parentFieldConsumer) {
                NameTypeAndMode newAccessStyle = NameTypeAndMode.of(nameTypeAndMode.type, name);
                TypeSchemaNode typeSchemaNode = isStruct(nameTypeAndMode.type)
                                ?new SchemaNode.Struct(this, newAccessStyle)
                                :new SchemaNode.Union(this, newAccessStyle);
                parentFieldConsumer.accept(typeSchemaNode);
                addType(typeSchemaNode);
                addField(new FixedArray(this, name, NameTypeAndMode.of(nameTypeAndMode.type, name), len));
                return this;
            }

            private TypeSchemaNode fieldControlledArray(String name, ArrayLen arrayLen) {
                addField(new FieldControlledArray(this, name, NameTypeAndMode.of(nameTypeAndMode.type, name), arrayLen));
                return this;
            }

            public static class ArrayBuildState {
                TypeSchemaNode typeSchemaNode;
                ArrayLen arrayLenField;

                public TypeSchemaNode array(String name) {
                    return typeSchemaNode.fieldControlledArray(name, arrayLenField);
                }

                public TypeSchemaNode array(String name, Consumer<TypeSchemaNode> parentFieldConsumer) {
                    NameTypeAndMode newAccessStyle = NameTypeAndMode.of(typeSchemaNode.nameTypeAndMode.type, name);
                    this.typeSchemaNode.fieldControlledArray(name, arrayLenField);
                    TypeSchemaNode typeSchemaNode =isStruct(newAccessStyle.type)
                            ?new SchemaNode.Struct(this.typeSchemaNode, newAccessStyle)
                            :new SchemaNode.Union(this.typeSchemaNode, newAccessStyle);
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
                var arrayLenField = new ArrayLen(this, NameTypeAndMode.of(nameTypeAndMode.type, arrayLenFieldName));
                addField(arrayLenField);
                return new ArrayBuildState(this, arrayLenField);
            }

            public void flexArray(String name) {
                addField(new FlexArray(this, name, null));
            }


            @Override
            public void toText(String indent, Consumer<String> stringConsumer) {
                stringConsumer.accept(indent);
                if (isUnion(nameTypeAndMode.type)) {
                    stringConsumer.accept("union");
                } else if (isStructOrBuffer(nameTypeAndMode.type)) {
                    stringConsumer.accept("struct");
                } else {
                    throw new IllegalStateException("Oh my ");
                }
                stringConsumer.accept(" " + nameTypeAndMode + "{");
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
            Struct(TypeSchemaNode parent, NameTypeAndMode nameTypeAndMode) {
                super(parent, nameTypeAndMode);
            }
        }

        public static final class Union extends TypeSchemaNode {
            Union(TypeSchemaNode parent, NameTypeAndMode nameTypeAndMode) {
                super(parent, nameTypeAndMode);
            }
        }

        public abstract static sealed class Array extends FieldSchemaNode permits FieldControlledArray, FixedArray, FlexArray {
            String name;
            NameTypeAndMode elementAccessStyle;
            Array(TypeSchemaNode parent, String name, NameTypeAndMode elementAccessStyle) {
                super(parent);
                this.name = name;
                this.elementAccessStyle = elementAccessStyle;
            }
        }

        public static final class FixedArray extends Array {
            int len;

            FixedArray(TypeSchemaNode parent, String name, NameTypeAndMode elementAccessStyle, int len) {
                super(parent, name, elementAccessStyle);
                this.len = len;
            }

            @Override
            public void toText(String indent, Consumer<String> stringConsumer) {
                stringConsumer.accept(indent + "array [" + len + "]");
            }

            @Override
            void collectLayouts(LayoutToBoundFieldTreeNode layoutToFieldBindingNode) {
                layoutToFieldBindingNode.bind(this, MemoryLayout.sequenceLayout(len,
                        parent.getLayout(elementAccessStyle, layoutToFieldBindingNode).withName(elementAccessStyle.type.getSimpleName())
                ).withName(elementAccessStyle.name));
            }
        }

        public static final  class FlexArray extends Array {
            FlexArray(TypeSchemaNode parent, String name, NameTypeAndMode elementAccessStyle) {
                super(parent, name, elementAccessStyle);
            }

            @Override
            public void toText(String indent, Consumer<String> stringConsumer) {
                stringConsumer.accept(indent + "array [?] ");
            }

            void collectLayouts(LayoutToBoundFieldTreeNode layoutToFieldBindingNode) {
                layoutToFieldBindingNode.bind(this,
                        MemoryLayout.sequenceLayout(0,
                                parent.getLayout(elementAccessStyle, layoutToFieldBindingNode).withName(elementAccessStyle.type.getSimpleName())
                        ).withName(elementAccessStyle.name));
            }
        }

        public static final class FieldControlledArray extends Array {
            ArrayLen arrayLen;

            FieldControlledArray(TypeSchemaNode parent, String name, NameTypeAndMode elementAccessStyle, ArrayLen arrayLen) {
                super(parent, name, elementAccessStyle);
                this.arrayLen = arrayLen;
            }

            @Override
            public void toText(String indent, Consumer<String> stringConsumer) {
                stringConsumer.accept(indent + elementAccessStyle.name + "[" + elementAccessStyle + "] where len defined by " + arrayLen.nameTypeAndMode);
            }

            @Override
            void collectLayouts(LayoutToBoundFieldTreeNode layoutToFieldBindingNode) {
                layoutToFieldBindingNode.bind(this, MemoryLayout.sequenceLayout(
                        layoutToFieldBindingNode.takeArrayLen(),
                        parent.getLayout(elementAccessStyle, layoutToFieldBindingNode).withName(elementAccessStyle.type.getSimpleName())
                ).withName(elementAccessStyle.name));
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
        NameTypeAndMode nameTypeAndMode = NameTypeAndMode.of(iface, iface.getSimpleName());
        var struct = new SchemaNode.Struct(null, nameTypeAndMode);
        parentFieldConsumer.accept(struct);
        return new Schema<>(iface, struct);
    }
    public void toText(Consumer<String> stringConsumer) {
        schemaRootField.toText("", stringConsumer);
    }
}

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
            MemoryLayout memoryLayout = isUnion(schema.schemaRootField.modeAndType.type)
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
    enum Mode {
        UNKNOWN,
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
                System.out.println("skiping " + m);
                return null;
            }
        }
    }
    public static class ModeAndType {

        Mode mode;
        Class<?> type;
        ModeAndType(Mode mode, Class<?> type) {
            this.mode = mode;
            this.type = type;

        }

        @Override
        public String toString() {
            return mode.name() + ":" + type.getSimpleName() ;
        }


        static ModeAndType of(Class<?> iface, String name) {
            ModeAndType accessStyle = new ModeAndType(modeOf(iface, name), typeOf(iface, name));
            return accessStyle;
        }
    }
    static Mode modeOf(Class<?> iface, String name) {
        var methods = iface.getDeclaredMethods();
        Result<Mode> modeResult = new Result<>();
        Arrays.stream(methods).filter(method -> method.getName().equals(name)).forEach(matchingMethod -> {
            var thisMode = Mode.of(matchingMethod);
            if (thisMode == null){
                throw new IllegalStateException("Could not determine the mode of method "+name);
            }
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
        if ( !modeResult.isPresent() || modeResult.get().equals(Mode.UNKNOWN)) {
            modeResult.of(Mode.PRIMITIVE_GETTER_AND_SETTER);
        }
        return modeResult.get();
    }

    static Class<?> typeOf(Class<?> iface, String name) {
        var methods = iface.getDeclaredMethods();
        Result<Class<?>> typeResult = new Result<>();
        Arrays.stream(methods).filter(method -> method.getName().equals(name)).forEach(matchingMethod -> {
            var thisType = methodToType(matchingMethod);
            if (!typeResult.isPresent() || typeResult.get().equals(thisType)) {
                typeResult.of(thisType);
            } else  {
                throw new IllegalStateException("type mismatch for " + name);
            }
        });
        if (!typeResult.isPresent()) {
            typeResult.of(iface);
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
          //  Mode mode;
         //   Class<?> type;
            final ModeAndType modeAndType;
            final String name;
            NamedFieldSchemaNode(TypeSchemaNode parent, ModeAndType  modeAndType, String name) {
                super(parent);
              //  this.mode = mode;
              //  this.type = type;
                this.modeAndType = modeAndType;
                this.name = name;
            }
        }


        public static final class ArrayLen extends NamedFieldSchemaNode {
            ArrayLen(TypeSchemaNode parent, ModeAndType modeAndType,  String name) {
                super(parent,modeAndType, name);
            }

            @Override
            public void toText(String indent, Consumer<String> stringConsumer) {
                stringConsumer.accept(indent + "arrayLen " + modeAndType);
            }

            @Override
            void collectLayouts(LayoutToBoundFieldTreeNode layoutToFieldBindingNode) {
                layoutToFieldBindingNode.bind(this, parent.getLayout(modeAndType, layoutToFieldBindingNode).withName(name));
            }
        }

        public static final class AtomicField extends NamedFieldSchemaNode {


            AtomicField(TypeSchemaNode parent,  ModeAndType modeAndType, String name) {
                super(parent, modeAndType,name);
            }

            @Override
            public void toText(String indent, Consumer<String> stringConsumer) {
                stringConsumer.accept(indent + "atomic " + modeAndType);
            }

            @Override
            void collectLayouts(LayoutToBoundFieldTreeNode layoutToFieldBindingNode) {
                layoutToFieldBindingNode.bind(this, parent.getLayout(modeAndType, layoutToFieldBindingNode).withName(name));
            }
        }

        public static final class Field extends NamedFieldSchemaNode {


            Field(TypeSchemaNode parent,  ModeAndType modeAndType, String name) {
                super(parent, modeAndType,name);

            }

            @Override
            public void toText(String indent, Consumer<String> stringConsumer) {
                stringConsumer.accept(indent + "field " + modeAndType);
            }

            @Override
            void collectLayouts(LayoutToBoundFieldTreeNode layoutToFieldBindingNode) {
                layoutToFieldBindingNode.bind(this, parent.getLayout(modeAndType, layoutToFieldBindingNode).withName(name));
            }
        }

        public static abstract sealed class TypeSchemaNode extends SchemaNode permits Union,Struct {
            private List<FieldSchemaNode> fields = new ArrayList<>();
            private List<TypeSchemaNode> types = new ArrayList<>();
            ModeAndType modeAndType;

            <T extends FieldSchemaNode> T addField(T child) {
                fields.add(child);
                return child;
            }
            <T extends TypeSchemaNode> T addType(T child) {
                types.add(child);
                return child;
            }

            TypeSchemaNode(TypeSchemaNode parent, ModeAndType modeAndType) {
                super(parent);
                this.modeAndType = modeAndType;
            }
            /**
             * Get a layout which describes the NameTypeAndMode.
             * <p>
             * If NameTypeAndMode holds a primitive (int, float) then just map to JAVA_INT, JAVA_FLOAT value layouts
             * Otherwise we look through the parent's children.  Which should include a type node struct/union matching the type.
             *
             * @param modeAndType
             * @param layoutToFieldBindingNode
             * @return
             */
             MemoryLayout getLayout(ModeAndType modeAndType, LayoutToBoundFieldTreeNode layoutToFieldBindingNode) {
                MemoryLayout memoryLayout = null;
                if (modeAndType.type == Integer.TYPE) {
                    memoryLayout = JAVA_INT;
                } else if (modeAndType.type == Float.TYPE) {
                    memoryLayout = JAVA_FLOAT;
                } else if (modeAndType.type == Long.TYPE) {
                    memoryLayout = JAVA_LONG;
                } else if (modeAndType.type == Double.TYPE) {
                    memoryLayout = JAVA_DOUBLE;
                } else if (modeAndType.type == Short.TYPE) {
                    memoryLayout = JAVA_SHORT;
                } else if (modeAndType.type == Character.TYPE) {
                    memoryLayout = JAVA_CHAR;
                } else if (modeAndType.type == Byte.TYPE) {
                    memoryLayout = JAVA_BYTE;
                } else if (modeAndType.type == Boolean.TYPE) {
                    memoryLayout = JAVA_BOOLEAN;
                } else {
                    TypeSchemaNode o = types.stream()
                            .filter(p -> p.modeAndType.type.equals(modeAndType.type)).findFirst().get();
                    LayoutToBoundFieldTreeNode scope = layoutToFieldBindingNode.createChild();
                    o.fields.stream()
                            .forEach(fieldSchemaNode -> {
                                fieldSchemaNode.collectLayouts(scope);
                            });
                    if (isUnion(o.modeAndType.type)) {
                        memoryLayout = MemoryLayout.unionLayout(scope.array());
                    } else if (isStructOrBuffer(o.modeAndType.type)) {
                        memoryLayout = MemoryLayout.structLayout(scope.array());
                    } else {
                        throw new IllegalStateException("Recursing through layout collections and came across  " + o.modeAndType.type);
                    }
                }
                return memoryLayout;
            }


            public TypeSchemaNode struct(String name, Consumer<TypeSchemaNode> parentSchemaNodeConsumer) {
                parentSchemaNodeConsumer.accept(addType(new Struct(this, ModeAndType.of(modeAndType.type, name))));
                return this;
            }

            public TypeSchemaNode union(String name, Consumer<TypeSchemaNode> parentSchemaNodeConsumer) {
                parentSchemaNodeConsumer.accept(addType(new Union(this, ModeAndType.of(modeAndType.type, name))));
                return this;
            }

            public TypeSchemaNode field(String name) {
                addField(new Field(this, ModeAndType.of(modeAndType.type, name), name));
                return this;
            }

            public TypeSchemaNode atomic(String name) {
                addField(new AtomicField(this, ModeAndType.of(modeAndType.type, name), name));
                return this;
            }

            public TypeSchemaNode pad(int len) {
                addField(new Padding(this, len));
                return this;
            }

            public TypeSchemaNode field(String name, Consumer<TypeSchemaNode> parentSchemaNodeConsumer) {
                ModeAndType newAccessStyle = ModeAndType.of(modeAndType.type, name);
                addField(new Field(this, newAccessStyle, name));
                TypeSchemaNode field = isStruct(newAccessStyle.type)?new SchemaNode.Struct(this, newAccessStyle):new SchemaNode.Union(this, newAccessStyle);
                parentSchemaNodeConsumer.accept(addType(field));
                return this;
            }

            public TypeSchemaNode fields(String name1, String name2, Consumer<TypeSchemaNode> parentSchemaNodeConsumer) {
                ModeAndType newAccessStyle1 = ModeAndType.of(modeAndType.type, name1);
                ModeAndType newAccessStyle2 = ModeAndType.of(modeAndType.type, name2);
                addField(new Field(this,  newAccessStyle1, name1));
                addField(new Field(this, newAccessStyle2, name2));
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
                addField(new FixedArray(this, ModeAndType.of(modeAndType.type, name),name, len));
                return this;
            }

            public TypeSchemaNode array(String name, int len, Consumer<TypeSchemaNode> parentFieldConsumer) {
                ModeAndType newAccessStyle = ModeAndType.of(modeAndType.type, name);
                TypeSchemaNode typeSchemaNode = isStruct(modeAndType.type)
                                ?new SchemaNode.Struct(this, newAccessStyle)
                                :new SchemaNode.Union(this, newAccessStyle);
                parentFieldConsumer.accept(typeSchemaNode);
                addType(typeSchemaNode);
                addField(new FixedArray(this,  ModeAndType.of(modeAndType.type, name), name, len));
                return this;
            }

            private TypeSchemaNode fieldControlledArray(String name, ArrayLen arrayLen) {
                addField(new FieldControlledArray(this,  ModeAndType.of(modeAndType.type, name),name, arrayLen));
                return this;
            }

            public static class ArrayBuildState {
                TypeSchemaNode typeSchemaNode;
                ArrayLen arrayLenField;

                public TypeSchemaNode array(String name) {
                    return typeSchemaNode.fieldControlledArray(name, arrayLenField);
                }

                public TypeSchemaNode array(String name, Consumer<TypeSchemaNode> parentFieldConsumer) {
                    ModeAndType newAccessStyle = ModeAndType.of(typeSchemaNode.modeAndType.type, name);
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
                var arrayLenField = new ArrayLen(this,  ModeAndType.of(modeAndType.type, arrayLenFieldName),arrayLenFieldName );
                addField(arrayLenField);
                return new ArrayBuildState(this, arrayLenField);
            }

            public void flexArray(String name) {
                addField(new FlexArray(this,null, name));
            }


            @Override
            public void toText(String indent, Consumer<String> stringConsumer) {
                stringConsumer.accept(indent);
                if (isUnion(modeAndType.type)) {
                    stringConsumer.accept("union");
                } else if (isStructOrBuffer(modeAndType.type)) {
                    stringConsumer.accept("struct");
                } else {
                    throw new IllegalStateException("Oh my ");
                }
                stringConsumer.accept(" " + modeAndType + "{");
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
            Struct(TypeSchemaNode parent, ModeAndType modeAndType) {
                super(parent, modeAndType);
            }
        }

        public static final class Union extends TypeSchemaNode {
            Union(TypeSchemaNode parent, ModeAndType modeAndType) {
                super(parent, modeAndType);
            }
        }

        public abstract static sealed class Array extends NamedFieldSchemaNode permits FieldControlledArray, FixedArray, FlexArray {
            ModeAndType elementAccessStyle;
            Array(TypeSchemaNode parent,  ModeAndType elementAccessStyle, String name) {
                super(parent, elementAccessStyle, name);
                this.elementAccessStyle = elementAccessStyle;
            }
        }

        public static final class FixedArray extends Array {
            int len;

            FixedArray(TypeSchemaNode parent, ModeAndType elementAccessStyle, String name, int len) {
                super(parent,  elementAccessStyle, name);
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
                ).withName(name));
            }
        }

        public static final  class FlexArray extends Array {
            FlexArray(TypeSchemaNode parent,  ModeAndType elementAccessStyle, String name) {
                super(parent,  elementAccessStyle, name);
            }

            @Override
            public void toText(String indent, Consumer<String> stringConsumer) {
                stringConsumer.accept(indent + "array [?] ");
            }

            void collectLayouts(LayoutToBoundFieldTreeNode layoutToFieldBindingNode) {
                layoutToFieldBindingNode.bind(this,
                        MemoryLayout.sequenceLayout(0,
                                parent.getLayout(elementAccessStyle, layoutToFieldBindingNode).withName(elementAccessStyle.type.getSimpleName())
                        ).withName(name));
            }
        }

        public static final class FieldControlledArray extends Array {
            ArrayLen arrayLen;

            FieldControlledArray(TypeSchemaNode parent,  ModeAndType elementAccessStyle,String name, ArrayLen arrayLen) {
                super(parent, elementAccessStyle, name);
                this.arrayLen = arrayLen;
            }

            @Override
            public void toText(String indent, Consumer<String> stringConsumer) {
                stringConsumer.accept(indent + name + "[" + elementAccessStyle + "] where len defined by " + arrayLen.modeAndType);
            }

            @Override
            void collectLayouts(LayoutToBoundFieldTreeNode layoutToFieldBindingNode) {
                layoutToFieldBindingNode.bind(this, MemoryLayout.sequenceLayout(
                        layoutToFieldBindingNode.takeArrayLen(),
                        parent.getLayout(elementAccessStyle, layoutToFieldBindingNode).withName(elementAccessStyle.type.getSimpleName())
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
        ModeAndType modeAndType = ModeAndType.of(iface, iface.getSimpleName());
        var struct = new SchemaNode.Struct(null, modeAndType);
        parentFieldConsumer.accept(struct);
        return new Schema<>(iface, struct);
    }
    public void toText(Consumer<String> stringConsumer) {
        schemaRootField.toText("", stringConsumer);
    }
}

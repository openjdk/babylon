package hat.ifacemapper;

import hat.buffer.Buffer;
import hat.buffer.BufferAllocator;
import hat.ifacemapper.accessor.AccessorInfo;
import hat.ifacemapper.accessor.ValueType;
import hat.util.Result;

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
    final public SchemaNode.IfaceTypeNode rootIfaceTypeNode;
    public Class<T> iface;

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
            } else if (paramTypes.length == 1 && isMemorySegment(paramTypes[0]) && returnType == Void.TYPE) {
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
    static boolean isMappableIface(Class<?> clazz) {
        return  MappableIface.class.isAssignableFrom(clazz);
    }
    static boolean isBuffer(Class<?> clazz) {
        return  Buffer.class.isAssignableFrom(clazz);
    }

    static boolean isStruct(Class<?> clazz) {
        return  Buffer.Struct.class.isAssignableFrom(clazz);
    }
    static boolean isMemorySegment(Class<?> clazz) {
        return  MemorySegment.class.isAssignableFrom(clazz);
    }

    static boolean isStructOrBuffer(Class<?> clazz) {
        return (isBuffer(clazz) || isStruct(clazz));
    }

    static boolean isUnion(Class<?> clazz) {
        return  Buffer.Union.class.isAssignableFrom(clazz);
    }

    public static abstract sealed class SchemaNode permits SchemaNode.FieldNode, SchemaNode.IfaceTypeNode {
        public IfaceTypeNode parent;

        SchemaNode(IfaceTypeNode parent) {
            this.parent = parent;
        }

        public abstract void toText(String indent, Consumer<String> stringConsumer);

        public static abstract sealed class FieldNode extends SchemaNode permits NamedFieldNode, Padding {
            FieldNode(IfaceTypeNode parent) {
                super(parent);
            }
            public abstract void toText(String indent, Consumer<String> stringConsumer);
            public abstract void collectLayouts(BoundSchemaNode layoutCollector);
        }

        public static final class Padding extends FieldNode {
            int len;

            Padding(IfaceTypeNode parent, int len) {
                super(parent);
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

        public static abstract sealed class NamedFieldNode extends FieldNode permits AddressField, MapableIfaceNamedFieldNode, PrimitiveNamedFieldNode {
            AccessorInfo.Key key;

            public final String name;
            NamedFieldNode(IfaceTypeNode parent, AccessorInfo.Key key,  String name) {
                super(parent);
                this.key = key;
                this.name = name;
            }
        }
        
        public static abstract sealed class PrimitiveNamedFieldNode extends NamedFieldNode permits PrimitiveArray, ArrayLen, AtomicField, PrimitiveField {
            public Class<?> type;
            PrimitiveNamedFieldNode(IfaceTypeNode parent, AccessorInfo.Key key, Class<?> type, String name) {
                super(parent,key,  name);
                this.type = type;
            }

            @Override
            public void collectLayouts(BoundSchemaNode boundSchemaNode) {
                boundSchemaNode.bind(this, parent.getLayout(this.type, boundSchemaNode).withName(name));
            }
        }
        public static abstract sealed class MapableIfaceNamedFieldNode extends NamedFieldNode permits IfaceMappableArray,  MappableIfaceField {

            public IfaceTypeNode ifaceTypeNode;
            MapableIfaceNamedFieldNode(IfaceTypeNode parent, AccessorInfo.Key key,  IfaceTypeNode ifaceTypeNode,String name) {
                super(parent,key, name);
                this.ifaceTypeNode = ifaceTypeNode;
            }
            @Override
            public void collectLayouts(BoundSchemaNode boundSchemaNode) {
                boundSchemaNode.bind(this, parent.getLayout(this.ifaceTypeNode.iface, boundSchemaNode).withName(name));
            }
        }
        public static final class AddressField extends NamedFieldNode {

            Class<MemorySegment> type;

            AddressField(IfaceTypeNode parent, AccessorInfo.Key key, Class<MemorySegment> type, String name) {
                super(parent, key, name);
                this.type = type;
            }

            @Override
            public void toText(String indent, Consumer<String> stringConsumer) {
                stringConsumer.accept(indent + "address " + key + ":" + type);
            }
            @Override
            public void collectLayouts(BoundSchemaNode boundSchemaNode) {
                boundSchemaNode.bind(this, parent.getLayout(type, boundSchemaNode).withName(name));
            }
        }
        public static final class ArrayLen extends PrimitiveNamedFieldNode {
            ArrayLen(IfaceTypeNode parent, AccessorInfo.Key key, Class<?> type, String name) {
                super(parent, key, type, name);
            }

            @Override
            public void toText(String indent, Consumer<String> stringConsumer) {
                stringConsumer.accept(indent + "arrayLen " + key + ":" + type);
            }

        }

        public static final class AtomicField extends PrimitiveNamedFieldNode {
            AtomicField(IfaceTypeNode parent, AccessorInfo.Key key, Class<?> type, String name) {
                super(parent, key, type, name);
            }

            @Override
            public void toText(String indent, Consumer<String> stringConsumer) {
                stringConsumer.accept(indent + "atomic " + key + ":" + type);
            }

        }

        public static final class MappableIfaceField extends MapableIfaceNamedFieldNode {

            MappableIfaceField(IfaceTypeNode parent, AccessorInfo.Key key,IfaceTypeNode ifaceTypeNode,String name) {
                super(parent, key, ifaceTypeNode,name);

            }

            @Override
            public void toText(String indent, Consumer<String> stringConsumer) {
                stringConsumer.accept(indent + "mappable field " + key + ":" + ifaceTypeNode.iface);
            }

        }
        public static final class PrimitiveField extends PrimitiveNamedFieldNode {

            PrimitiveField(IfaceTypeNode parent, AccessorInfo.Key key, Class<?> type, String name) {
                super(parent, key, type, name);

            }

            @Override
            public void toText(String indent, Consumer<String> stringConsumer) {
                stringConsumer.accept(indent + "primitive field " + key + ":" + type);
            }
        }

        public static abstract sealed class IfaceTypeNode extends SchemaNode permits Union, Struct {
            public List<FieldNode> fields = new ArrayList<>();
            public List<IfaceTypeNode> ifaceTypeNodes = new ArrayList<>();
            public Class<MappableIface> iface;

            <T extends FieldNode> T addField(T child) {
                fields.add(child);
                return child;
            }

            <T extends IfaceTypeNode> T addIfaceTypeNode(T child) {
                ifaceTypeNodes.add(child);
                return child;
            }

            IfaceTypeNode(IfaceTypeNode parent, Class<MappableIface> iface) {
                super(parent);
                this.iface = iface;
            }

            IfaceTypeNode getChild(Class<?> iface){
                Optional<IfaceTypeNode> ifaceTypeNodeOptional =ifaceTypeNodes
                        .stream()
                        .filter(n->n.iface.equals(iface))
                        .findFirst();
                if (ifaceTypeNodeOptional.isPresent()){
                    return ifaceTypeNodeOptional.get();
                }else {
                    throw new IllegalStateException("no supported iface type");
                }
            }

            public void visitTypes(int depth, Consumer<SchemaNode.IfaceTypeNode> ifaceTypeNodeConsumer) {
                ifaceTypeNodes.forEach(t->t.visitTypes(depth+1,ifaceTypeNodeConsumer));
                ifaceTypeNodeConsumer.accept(this);
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
            MemoryLayout getLayout(Class<?> type, BoundSchemaNode boundSchemaNode) {
                if (type.isPrimitive()) {
                    return MapperUtil.primitiveToLayout(type);
                }else if (isMemorySegment(type)) {
                    return ValueLayout.ADDRESS;
                } else {
                    IfaceTypeNode ifaceTypeNode = ifaceTypeNodes.stream()
                            .filter(i -> i.iface.equals(type))
                            .findFirst().orElseThrow();
                        BoundSchemaNode child = boundSchemaNode.createChild(ifaceTypeNode);
                        ifaceTypeNode.fields.forEach(fieldNode ->
                                fieldNode.collectLayouts(child)
                        );
                        return isUnion(ifaceTypeNode.iface)
                                ? MemoryLayout.unionLayout(child.memoryLayoutListToArray())
                                : MemoryLayout.structLayout(child.memoryLayoutListToArray());
                }
            }

            public IfaceTypeNode struct(String name, Consumer<IfaceTypeNode> parentSchemaNodeConsumer) {
                parentSchemaNodeConsumer.accept(addIfaceTypeNode(new Struct(this, (Class<MappableIface>)typeOf(iface, name))));
                return this;
            }

            public IfaceTypeNode union(String name, Consumer<IfaceTypeNode> parentSchemaNodeConsumer) {
                parentSchemaNodeConsumer.accept(addIfaceTypeNode(new Union(this, (Class<MappableIface>) typeOf(iface, name))));
                return this;
            }

            public IfaceTypeNode field(String name) {
                var key = AccessorInfo.Key.of(iface, name);
                var typeOf = typeOf(iface, name);
                addField(isMemorySegment(typeOf)
                        ? new AddressField(this, key, (Class<MemorySegment>)typeOf, name)
                        : isMappableIface(typeOf)
                           ? new MappableIfaceField(this, key, this.getChild(typeOf), name)
                           : new PrimitiveField(this, key, typeOf, name));
                return this;
            }

            public IfaceTypeNode atomic(String name) {
                addField(new AtomicField(this, AccessorInfo.Key.of(iface, name), typeOf(iface, name), name));
                return this;
            }

            public IfaceTypeNode pad(int len) {
                addField(new Padding(this, len));
                return this;
            }

            public IfaceTypeNode field(String name, Consumer<IfaceTypeNode> parentSchemaNodeConsumer) {
                AccessorInfo.Key fieldKey = AccessorInfo.Key.of(iface, name);
                Class<MappableIface> fieldType = (Class<MappableIface>)typeOf(iface, name);
                IfaceTypeNode structOrUnion= isStruct(fieldType) ? new SchemaNode.Struct(this, fieldType) : new SchemaNode.Union(this, fieldType);
                addIfaceTypeNode(structOrUnion);
                addField(new MappableIfaceField(this, fieldKey, structOrUnion,name));
                parentSchemaNodeConsumer.accept(structOrUnion);
                return this;
            }

            public IfaceTypeNode fields(String name1, String name2, Consumer<IfaceTypeNode> parentSchemaNodeConsumer) {
                AccessorInfo.Key fieldKey1 = AccessorInfo.Key.of(iface, name1);
                AccessorInfo.Key fieldKey2 = AccessorInfo.Key.of(iface, name2);
                if (!fieldKey1.equals(fieldKey2)) {
                    throw new IllegalStateException("fields " + name1 + " and " + name2 + " have different keys");
                }
                Class<MappableIface> structOrUnionType = (Class<MappableIface>) typeOf(iface, name1);
                Class<?> fieldTypeCheck = typeOf(iface, name2);
                if (!structOrUnionType.equals(fieldTypeCheck)) {
                    throw new IllegalStateException("fields " + name1 + " and " + name2 + " have different types");
                }
                IfaceTypeNode ifaceTypeNode = isStruct(iface)
                        ? new SchemaNode.Struct(this, structOrUnionType)
                        : new SchemaNode.Union(this, structOrUnionType);
                addIfaceTypeNode(ifaceTypeNode);
                addField(new MappableIfaceField(this, fieldKey1, ifaceTypeNode,name1));
                addField(new MappableIfaceField(this, fieldKey2,  ifaceTypeNode,name2));

                parentSchemaNodeConsumer.accept(ifaceTypeNode);
                return this;
            }

            public IfaceTypeNode fields(String... names) {
                for (var name : names) {
                    field(name);
                }
                return this;
            }

            public IfaceTypeNode array(String name, int len) {
                AccessorInfo.Key arrayKey = AccessorInfo.Key.of(iface, name);
                var typeof = typeOf(iface,name);
                addField(arrayKey.valueType().equals(ValueType.INTERFACE)
                        ?new IfaceMapableFixedArray( this,arrayKey,this.getChild(typeof),name, len)
                        :new PrimitiveFixedArray(this, arrayKey, typeOf(iface, name), name, len));
                return this;
            }

            public IfaceTypeNode array(String name, int len, Consumer<IfaceTypeNode> parentFieldConsumer) {
                AccessorInfo.Key arrayKey = AccessorInfo.Key.of(iface, name);
                Class<MappableIface> structOrUnionType = (Class<MappableIface>)typeOf(iface, name);
                IfaceTypeNode ifaceTypeNode = isStruct(iface)
                        ? new SchemaNode.Struct(this, structOrUnionType)
                        : new SchemaNode.Union(this, structOrUnionType);
                parentFieldConsumer.accept(ifaceTypeNode);
                addIfaceTypeNode(ifaceTypeNode);
                addField(new IfaceMapableFixedArray(this, arrayKey, ifaceTypeNode, name, len));
                return this;
            }

            private IfaceTypeNode fieldControlledArray(String name, List<ArrayLen> arrayLenFields, int stride) {
                AccessorInfo.Key arrayKey = AccessorInfo.Key.of(iface, name);
                var typeOf = typeOf(iface, name);
                addField(arrayKey.valueType().equals(ValueType.INTERFACE)
                        ?new IfaceMapableFieldControlledArray(this, arrayKey, this.getChild(typeOf),name,  arrayLenFields, stride)
                        :new PrimitiveFieldControlledArray(this, arrayKey, typeOf, name,  arrayLenFields, stride));
                return this;
            }

            public static class ArrayBuildState {
                IfaceTypeNode ifaceTypeNode;
                List<ArrayLen> arrayLenFields;
                int padding =0;
                int stride = 1;

                public IfaceTypeNode array(String name) {
                    return ifaceTypeNode.fieldControlledArray(name, arrayLenFields, stride);
                }

                public ArrayBuildState stride(int stride) {
                    this.stride = stride;
                    return this;
                }
                public ArrayBuildState pad(int padding) {
                    this.padding = padding;
                    var paddingField = new Padding(ifaceTypeNode, padding);
                    ifaceTypeNode.addField(paddingField);
                    return this;
                }
                public IfaceTypeNode array(String name, Consumer<IfaceTypeNode> parentFieldConsumer) {
                    Class<MappableIface> arrayType = (Class<MappableIface>) typeOf(this.ifaceTypeNode.iface, name);
                    IfaceTypeNode ifaceTypeNode = isStruct(arrayType)
                            ? new SchemaNode.Struct(this.ifaceTypeNode, arrayType)
                            : new SchemaNode.Union(this.ifaceTypeNode, arrayType);
                    parentFieldConsumer.accept(ifaceTypeNode);
                    this.ifaceTypeNode.addIfaceTypeNode(ifaceTypeNode);
                    this.ifaceTypeNode.fieldControlledArray(name, arrayLenFields, stride);


                    return this.ifaceTypeNode;
                }

                ArrayBuildState(IfaceTypeNode ifaceTypeNode, List<ArrayLen> arrayLenFields) {
                    this.ifaceTypeNode = ifaceTypeNode;
                    this.arrayLenFields = arrayLenFields;
                }
            }

            public ArrayBuildState arrayLen(String... arrayLenFieldNames) {
                List<ArrayLen> arrayLenFields = new ArrayList<>();
                Arrays.stream(arrayLenFieldNames).forEach(arrayLenFieldName -> {
                    var arrayLenField = new ArrayLen(this, AccessorInfo.Key.of(iface, arrayLenFieldName), typeOf(iface, arrayLenFieldName), arrayLenFieldName);
                    addField(arrayLenField);
                    arrayLenFields.add(arrayLenField);
                });
                return new ArrayBuildState(this, arrayLenFields);
            }

            @Override
            public void toText(String indent, Consumer<String> stringConsumer) {
                stringConsumer.accept(indent);
                if (isUnion(iface)) {
                    stringConsumer.accept("union");
                } else if (isStructOrBuffer(iface)) {
                    stringConsumer.accept("struct");
                } else {
                    throw new IllegalStateException("Oh my ");
                }
                stringConsumer.accept(" " + iface + "{");
                stringConsumer.accept("\n");
                ifaceTypeNodes.forEach(c -> {
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

        public static final class Struct extends IfaceTypeNode {
            Struct(IfaceTypeNode parent, Class<MappableIface> type) {
                super(parent, type);
            }
        }

        public static final class Union extends IfaceTypeNode {
            Union(IfaceTypeNode parent, Class<MappableIface> type) {
                super(parent, type);
            }
        }
        public abstract static sealed class IfaceMappableArray extends MapableIfaceNamedFieldNode permits IfaceMapableFieldControlledArray, IfaceMapableFixedArray {
            IfaceMappableArray(IfaceTypeNode parent, AccessorInfo.Key key,  IfaceTypeNode ifaceTypeNode, String name) {
                super(parent, key, ifaceTypeNode, name);
            }
        }
        public abstract static sealed class PrimitiveArray extends PrimitiveNamedFieldNode permits PrimitiveFieldControlledArray, PrimitiveFixedArray {
            PrimitiveArray(IfaceTypeNode parent, AccessorInfo.Key key, Class<?> type, String name) {
                super(parent, key, type, name);
            }
        }
        public static final class IfaceMapableFixedArray extends IfaceMappableArray {
            public int len;

            IfaceMapableFixedArray(IfaceTypeNode parent, AccessorInfo.Key key,  IfaceTypeNode ifaceTypeNode,String name, int len) {
                super(parent, key, ifaceTypeNode, name);
                this.len = len;
            }

            @Override
            public void toText(String indent, Consumer<String> stringConsumer) {
                stringConsumer.accept(indent + "array [" + len + "]");
            }

            @Override
            public void collectLayouts(BoundSchemaNode boundSchemaNode) {
                boundSchemaNode.bind(this, MemoryLayout.sequenceLayout(len,
                        parent.getLayout(ifaceTypeNode.iface, boundSchemaNode).withName(ifaceTypeNode.iface.getSimpleName())
                ).withName(name));
            }
        }
        public static final class PrimitiveFixedArray extends PrimitiveArray {
            public int len;

            PrimitiveFixedArray(IfaceTypeNode parent, AccessorInfo.Key key, Class<?> type, String name, int len) {
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
                        parent.getLayout(type, boundSchemaNode).withName(type.getSimpleName())
                ).withName(name));
            }
        }

        public static final class IfaceMapableFieldControlledArray extends IfaceMappableArray {
            List<ArrayLen> arrayLenFields;
            int stride;
            int contributingDims;

            IfaceMapableFieldControlledArray(IfaceTypeNode parent, AccessorInfo.Key key, IfaceTypeNode ifaceTypeNode, String name, List<ArrayLen> arrayLenFields, int stride) {
                super(parent, key, ifaceTypeNode,name);
                this.arrayLenFields = arrayLenFields;
                this.stride = stride;
                this.contributingDims = arrayLenFields.size();
            }

            @Override
            public void toText(String indent, Consumer<String> stringConsumer) {
                stringConsumer.accept(indent + name + "[" + key + ":" + ifaceTypeNode.iface + "] where len defined by " + arrayLenFields);
            }

            @Override
            public void collectLayouts(BoundSchemaNode boundSchemaNode) {
                // To determine the actual 'array' size we multiply the contributing dims by the stride .
                int size = stride; //usually 1 but developer can define.
                for (int i = 0; i < contributingDims; i++) {
                    size *= boundSchemaNode.takeArrayLen(); // this takes an arraylen and bumps the ptr
                }

                boundSchemaNode.bind(this, MemoryLayout.sequenceLayout(size,
                        parent.getLayout(ifaceTypeNode.iface, boundSchemaNode).withName(ifaceTypeNode.iface.getSimpleName())
                ).withName(name));
            }
        }
        public static final class PrimitiveFieldControlledArray extends PrimitiveArray {
            List<ArrayLen> arrayLenFields;
            int stride;
            int contributingDims;

            PrimitiveFieldControlledArray(IfaceTypeNode parent, AccessorInfo.Key key, Class<?> type, String name, List<ArrayLen> arrayLenFields, int stride) {
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
                        parent.getLayout(type, boundSchemaNode).withName(type.getSimpleName())
                ).withName(name));
            }
        }

    }

    Schema(Class<T> iface, SchemaNode.IfaceTypeNode rootIfaceTypeNode) {
        this.iface = iface;
        this.rootIfaceTypeNode = rootIfaceTypeNode;
    }


    public T allocate(MethodHandles.Lookup lookup,BufferAllocator bufferAllocator, int... boundLengths) {
        BoundSchema<?> boundSchema = new BoundSchemaNode.BoundSchemaRootNode<>(this, boundLengths);
        return (T) boundSchema.allocate(lookup,bufferAllocator);
    }

    public static <T extends Buffer> Schema<T> of(Class<T> iface,  Consumer<SchemaNode.IfaceTypeNode> parentFieldConsumer) {
        var struct = new SchemaNode.Struct(null, (Class<MappableIface>)(Object)iface); // why the need for this?
        parentFieldConsumer.accept(struct);
        return new Schema<>(iface,struct);
    }

    public void toText(Consumer<String> stringConsumer) {
        rootIfaceTypeNode.toText("", stringConsumer);
    }
}

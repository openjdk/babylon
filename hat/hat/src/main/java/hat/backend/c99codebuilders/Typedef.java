package hat.backend.c99codebuilders;

import hat.buffer.Buffer;
import hat.buffer.IncompleteBuffer;
import hat.util.StreamCounter;

import java.lang.foreign.GroupLayout;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.PaddingLayout;
import java.lang.foreign.SequenceLayout;
import java.lang.foreign.StructLayout;
import java.lang.reflect.Modifier;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

public class Typedef {
    public int rank;
    final boolean isStruct;

    public String name() {
        return iface.getSimpleName();
    }

    public static abstract class AbstractNameAndType {
        public int index;
        public final Typedef containingTypedef;

        public Typedef typeDef;

        public final String name;

        public final Class<?> type;

        public MemoryLayout layout;

        public AbstractNameAndType(Typedef containingTypedef, String name, Class<?> type) {
            this.index = -1;
            this.containingTypedef = containingTypedef;
            this.name = name;
            this.type = type;
        }

        boolean isPrimitive() {
            return this.type.isPrimitive();
        }
    }

    public static class NameAndType extends AbstractNameAndType {
        public NameAndType(Typedef containingTypedef, String name, Class<?> type) {
            super(containingTypedef, name, type);
        }
    }

    public static class NameAndArrayOfType extends AbstractNameAndType {
        public long arraySize;
        public boolean isFlexible;

        public NameAndArrayOfType(Typedef containingTypedef, String name, Class<?> type, long arraySize) {
            super(containingTypedef, name, type);
            this.arraySize = arraySize;
            this.isFlexible = false;
        }
    }


    public final MemoryLayout memoryLayout;
    public final Class<?> iface;

    public List<AbstractNameAndType> nameAndTypes = new ArrayList<>();

    public Typedef(Class<?> iface, MemoryLayout memoryLayout) {
        Map<String, AbstractNameAndType> nameToFieldNameAndType = new LinkedHashMap<>();

        this.iface = iface;
        this.memoryLayout = memoryLayout;
        Arrays.stream(iface.getMethods()).forEach(method -> {
            if (Modifier.isStatic(method.getModifiers())) {
                // forgetaboutit
            } else if (method.isDefault()) {
                // forgetaboutit
            } else {
                String name = method.getName();
                if (name.equals("id")) {
                    name = name;
                }
                if (nameToFieldNameAndType.containsKey(name) || name.equals("equals") || name.equals("toString") || name.equals("hashCode")
                        || name.equals("$_$_$oFfSeT$_$_$") || name.equals("$_$_$sEgMeNt$_$_$")
                        || name.equals("$_$_$lAyOuT$_$_$")
                        || name.equals("notify") || name.equals("notifyAll")
                ) {
                    // forgetaboutit
                } else {
                    var returnType = method.getReturnType();
                    var parameterCount = method.getParameterCount();
                    var parameterTypes = method.getParameterTypes();
                    if (returnType.equals(Void.TYPE)) {
                        if (parameterCount == 0) {
                            throw new IllegalStateException("paramcount ==0 or  >2 arg iface setter with void return ");
                        }
                        if (parameterCount == 1) {
                            nameToFieldNameAndType.put(name, new NameAndType(this, name, parameterTypes[0]));
                        } else {
                            nameToFieldNameAndType.put(name, new NameAndArrayOfType(this, name, parameterTypes[1], -1));
                        }
                    } else {
                        if (parameterCount > 1) {
                            throw new IllegalStateException(" >1 arg iface getter with non void return ");
                        }
                        if (parameterCount == 0) {
                            nameToFieldNameAndType.put(name, new NameAndType(this, name, returnType));
                        } else {
                            nameToFieldNameAndType.put(name, new NameAndArrayOfType(this, name, returnType, -1));
                        }
                    }

                }
            }
        });
        this.isStruct = memoryLayout instanceof StructLayout;

        // We know everything but the order ;)
        // We can get the order from the layout
        if (memoryLayout instanceof GroupLayout groupLayout) {
            if (groupLayout.memberLayouts().size() == 0) {
                throw new IllegalStateException("How");
            }
            StreamCounter.of(groupLayout.memberLayouts().stream().filter(layout -> !(layout instanceof PaddingLayout)), (c, layout) -> {
                Optional<String> optionalLayoutFieldName = layout.name();
                if (optionalLayoutFieldName.isEmpty()) {
                    throw new IllegalStateException("how 2");
                }
                String layoutFieldName = optionalLayoutFieldName.orElseThrow();
                if (nameToFieldNameAndType.containsKey(layoutFieldName)) {
                    var nameAndType = nameToFieldNameAndType.get(layoutFieldName);
                    nameAndType.index = c.value();
                    nameAndType.layout = layout;
                    nameAndTypes.add(nameAndType);
                    if (layout instanceof SequenceLayout sequenceLayout) {
                        if (nameAndType instanceof NameAndArrayOfType nameAndArrayOfType) {
                            nameAndArrayOfType.arraySize = sequenceLayout.elementCount();
                        } else {
                            throw new IllegalStateException("not an array type?");
                        }
                    }
                } else {
                    throw new IllegalStateException("Hmm " + layoutFieldName);
                }
            });
        } else {
            throw new IllegalStateException("a buffer is alwyas a grouplayout!");
        }
        nameAndTypes.sort((lhs, rhs) -> Integer.compare(lhs.index, rhs.index));

        if (isIncomplete()) {
            // Above we captured the actual size of all arrays.  Of course only now do we know which of the fields is last
            // So if we are an incomplete type (i.e. last array is [0]) then tag the last element as flexible
            if (nameAndTypes.getLast() instanceof NameAndArrayOfType nameAndArrayOfType) {
                nameAndArrayOfType.isFlexible = true;
                nameAndArrayOfType.arraySize = 0;
            } else {
                throw new IllegalStateException("last element is not sequence layout!");
            }
        }

        nameAndTypes.forEach(nameAndType -> {
            if (nameAndType.type.isInterface()) {
                if (nameAndType.layout instanceof GroupLayout gl) {
                    nameAndType.typeDef = new Typedef(nameAndType.type, gl);
                } else if (nameAndType.layout instanceof SequenceLayout sl && sl.elementLayout() instanceof GroupLayout slgl) {
                    nameAndType.typeDef = new Typedef(nameAndType.type, slgl);
                }
            }
        });


    }

    Typedef(Buffer instance) {
        this(instance.getClass().getInterfaces()[0], Buffer.getLayout(instance));
    }

    public boolean isIncomplete() {
        return IncompleteBuffer.class.isAssignableFrom(iface);
    }
}

package jdk.incubator.code.type.impl;

import jdk.incubator.code.TypeElement;
import jdk.incubator.code.TypeElement.ExternalizedTypeElement;
import jdk.incubator.code.parser.impl.DescParser;
import jdk.incubator.code.type.ConstructorRef;
import jdk.incubator.code.type.FieldRef;
import jdk.incubator.code.type.JavaRef;
import jdk.incubator.code.type.JavaType;
import jdk.incubator.code.type.MethodRef;
import jdk.incubator.code.type.RecordTypeRef;
import jdk.incubator.code.type.RecordTypeRef.ComponentRef;
import jdk.incubator.code.type.TypeVariableType;
import jdk.incubator.code.type.WildcardType.BoundKind;

import java.lang.constant.ClassDesc;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

public class JavaTypeUtils {

    public static final String JAVA_TYPE_CLASS_NAME = "java.type.class";
    public static final String JAVA_TYPE_ARRAY_NAME = "java.type.array";
    public static final String JAVA_TYPE_WILDCARD_NAME = "java.type.wildcard";
    public static final String JAVA_TYPE_VAR_NAME = "java.type.var";
    public static final String JAVA_TYPE_PRIMITIVE_NAME = "java.type.primitive";

    public static final String JAVA_REF_FIELD_NAME = "java.ref.field";
    public static final String JAVA_REF_METHOD_NAME = "java.ref.method";
    public static final String JAVA_REF_CONSTRUCTOR_NAME = "java.ref.constructor";
    public static final String JAVA_REF_RECORD_NAME = "java.ref.record";

    public static final String JAVA_TYPE_FLAT_NAME_PREFIX = "java.type:";
    public static final String JAVA_REF_FLAT_NAME_PREFIX = "java.ref:";

    public static ExternalizedTypeElement classType(String name, ExternalizedTypeElement encl, List<ExternalizedTypeElement> typeargs) {
        if (encl == null) {
            encl = JavaType.VOID.externalize();
        } else {
            // watch out for names like "1Foo"
            name = escapeInnerClassName(name);
        }
        List<ExternalizedTypeElement> args = Stream.concat(
                Stream.of(nameToType(name), encl),
                typeargs.stream()).toList();
        return new ExternalizedTypeElement(JAVA_TYPE_CLASS_NAME, args);
    }

    public static ExternalizedTypeElement arrayType(ExternalizedTypeElement component) {
        return new ExternalizedTypeElement(JAVA_TYPE_ARRAY_NAME, List.of(component));
    }

    public static ExternalizedTypeElement wildcardType(BoundKind boundKind, ExternalizedTypeElement bound) {
        return new ExternalizedTypeElement(JAVA_TYPE_WILDCARD_NAME,
                List.of(nameToType(boundKind.name()), bound));
    }

    public static ExternalizedTypeElement typeVarType(String name, ExternalizedTypeElement owner, ExternalizedTypeElement bound) {
        return new ExternalizedTypeElement(JAVA_TYPE_VAR_NAME,
                List.of(nameToType(name), owner, bound));
    }

    public static ExternalizedTypeElement primitiveType(String name) {
        return new ExternalizedTypeElement(JAVA_TYPE_PRIMITIVE_NAME,
                List.of(nameToType(name)));
    }

    public static ExternalizedTypeElement fieldRef(String name, ExternalizedTypeElement owner, ExternalizedTypeElement type) {
        return new ExternalizedTypeElement(JAVA_REF_FIELD_NAME,
                List.of(owner, nameToType(name), type));
    }

    public static ExternalizedTypeElement methodRef(String name, ExternalizedTypeElement owner, ExternalizedTypeElement restype, List<ExternalizedTypeElement> paramtypes) {
        return new ExternalizedTypeElement(JAVA_REF_METHOD_NAME,
                List.of(owner, new ExternalizedTypeElement(name, paramtypes), restype));
    }

    public static ExternalizedTypeElement constructorRef(ExternalizedTypeElement owner, List<ExternalizedTypeElement> paramtypes) {
        return new ExternalizedTypeElement(JAVA_REF_CONSTRUCTOR_NAME,
                List.of(owner, new ExternalizedTypeElement("", paramtypes)));
    }

    public static ExternalizedTypeElement recordRef(ExternalizedTypeElement owner, List<String> componentNames, List<ExternalizedTypeElement> componentTypes) {
        return new ExternalizedTypeElement(JAVA_REF_RECORD_NAME,
                Stream.concat(
                        Stream.of(owner),
                        IntStream.range(0, componentNames.size())
                                .mapToObj(i -> new ExternalizedTypeElement(componentNames.get(i), List.of(componentTypes.get(i))))
                ).toList());
    }

    public static JavaType toJavaType(ExternalizedTypeElement tree) {
        return switch (tree.identifier()) {
            case JAVA_TYPE_CLASS_NAME -> {
                String name = unescapeInnerClassName(select(tree, 0, JavaTypeUtils::typeToName));
                JavaType encl = select(tree, 1, JavaTypeUtils::toJavaType);
                List<JavaType> typeargs = selectFrom(tree, 2, JavaTypeUtils::toJavaType);
                JavaType type = !encl.equals(JavaType.VOID) ?
                        JavaType.qualified(encl, name) :
                        JavaType.type(ClassDesc.of(name));
                yield typeargs.isEmpty() ?
                        type :
                        JavaType.parameterized(type, typeargs);
            }
            case JAVA_TYPE_ARRAY_NAME -> {
                JavaType componentType = select(tree, 0, JavaTypeUtils::toJavaType);
                yield JavaType.array(componentType);
            }
            case JAVA_TYPE_WILDCARD_NAME -> {
                BoundKind boundKind = select(tree, 0, t -> BoundKind.valueOf(typeToName(t)));
                JavaType bound = select(tree, 1, JavaTypeUtils::toJavaType);
                yield JavaType.wildcard(boundKind, bound);
            }
            case JAVA_TYPE_VAR_NAME -> {
                String tvarName = select(tree, 0, JavaTypeUtils::typeToName);
                TypeVariableType.Owner owner = (TypeVariableType.Owner)select(tree, 1, t ->
                        t.identifier().startsWith("java.type") ?
                                toJavaType(t) : toJavaRef(t));
                JavaType bound = select(tree, 2, JavaTypeUtils::toJavaType);
                yield JavaType.typeVarRef(tvarName, owner, bound);
            }
            case JAVA_TYPE_PRIMITIVE_NAME -> {
                String primitiveName = select(tree, 0, JavaTypeUtils::typeToName);
                yield PRIMITIVE_TYPES.get(primitiveName);
            }
            default -> throw new UnsupportedOperationException("Unsupported type: " + tree);
        };
    }

    public static JavaRef toJavaRef(ExternalizedTypeElement tree) {
        return switch (tree.identifier()) {
            case JAVA_REF_FIELD_NAME -> {
                JavaType owner = select(tree, 0, JavaTypeUtils::toJavaType);
                String fieldName = select(tree, 1, JavaTypeUtils::typeToName);
                JavaType fieldType = select(tree, 2, JavaTypeUtils::toJavaType);
                yield FieldRef.field(owner, fieldName, fieldType);
            }
            case JAVA_REF_METHOD_NAME -> {
                JavaType owner = select(tree, 0, JavaTypeUtils::toJavaType);
                ExternalizedTypeElement nameAndArgs = select(tree, 1, Function.identity());
                String methodName = nameAndArgs.identifier();
                List<JavaType> paramTypes = selectFrom(nameAndArgs, 0, JavaTypeUtils::toJavaType);
                JavaType restype = select(tree, 2, JavaTypeUtils::toJavaType);
                yield MethodRef.method(owner, methodName, restype, paramTypes);
            }
            case JAVA_REF_CONSTRUCTOR_NAME -> {
                JavaType owner = select(tree, 0, JavaTypeUtils::toJavaType);
                ExternalizedTypeElement nameAndArgs = select(tree, 1, Function.identity());
                List<JavaType> paramTypes = selectFrom(nameAndArgs, 0, JavaTypeUtils::toJavaType);
                yield ConstructorRef.constructor(owner, paramTypes);
            }
            case JAVA_REF_RECORD_NAME -> {
                JavaType owner = select(tree, 0, JavaTypeUtils::toJavaType);
                List<ComponentRef> components = selectFrom(tree, 1, Function.identity()).stream()
                        .map(t -> {
                            String componentName = t.identifier();
                            JavaType componentType = select(t, 0, JavaTypeUtils::toJavaType);
                            return new ComponentRef(componentType, componentName);
                        }).toList();
                yield RecordTypeRef.recordType(owner, components);
            }
            default -> throw new UnsupportedOperationException("Unsupported ref: " + tree);
        };
    }

    public static String toExternalTypeString(ExternalizedTypeElement tree) {
        return switch (tree.identifier()) {
            case JAVA_TYPE_CLASS_NAME -> {
                String className = select(tree, 0, JavaTypeUtils::typeToName);
                ExternalizedTypeElement enclosing = select(tree, 1, Function.identity());
                String typeargs = tree.arguments().size() == 2 ?
                        "" :
                        selectFrom(tree, 2, JavaTypeUtils::toExternalTypeString).stream()
                                .collect(Collectors.joining(", ", "<", ">"));
                if (is(enclosing, JavaType.VOID)) {
                    yield String.format("%s%s", className, typeargs);
                } else {
                    String enclosingString = toExternalTypeString(enclosing);
                    yield String.format("%s::%s%s", enclosingString, className, typeargs);
                }
            }
            case JAVA_TYPE_ARRAY_NAME -> {
                String componentType = select(tree, 0, JavaTypeUtils::toExternalTypeString);
                yield String.format("%s[]", componentType);
            }
            case JAVA_TYPE_WILDCARD_NAME -> {
                BoundKind boundKind = select(tree, 0, t -> BoundKind.valueOf(typeToName(t)));
                ExternalizedTypeElement bound = select(tree, 1, Function.identity());
                yield boundKind == BoundKind.EXTENDS && is(bound, JavaType.J_L_OBJECT) ?
                        "?" :
                        String.format("? %s %s", boundKind.name().toLowerCase(), toExternalTypeString(bound));
            }
            case JAVA_TYPE_VAR_NAME -> {
                String tvarName = select(tree, 0, JavaTypeUtils::typeToName);
                ExternalizedTypeElement owner = select(tree, 1, Function.identity());
                boolean isRef = owner.identifier().startsWith("java.ref");
                String prefix = isRef ?
                        String.format("(%s)", toExternalRefString(owner)) :
                        toExternalTypeString(owner);
                ExternalizedTypeElement bound = select(tree, 2, Function.identity());
                yield is(bound, JavaType.J_L_OBJECT) ?
                        String.format("%s::<%s>", prefix, tvarName) :
                        String.format("%s::<%s extends %s>", prefix, tvarName, toExternalTypeString(bound));
            }
            case JAVA_TYPE_PRIMITIVE_NAME -> select(tree, 0, JavaTypeUtils::typeToName);
            default -> throw new UnsupportedOperationException("Unsupported type: " + tree);
        };
    }

    public static String toExternalRefString(ExternalizedTypeElement tree) {
        return switch (tree.identifier()) {
            case JAVA_REF_FIELD_NAME -> {
                String owner = select(tree, 0, JavaTypeUtils::toExternalTypeString);
                String fieldName = select(tree, 1, JavaTypeUtils::typeToName);
                String fieldType = select(tree, 2, JavaTypeUtils::toExternalTypeString);
                yield String.format("%s::%s:%s", owner, fieldName, fieldType);
            }
            case JAVA_REF_METHOD_NAME -> {
                String owner = select(tree, 0, JavaTypeUtils::toExternalTypeString);
                ExternalizedTypeElement nameAndArgs = select(tree, 1, Function.identity());
                String methodName = nameAndArgs.identifier();
                List<String> paramTypes = selectFrom(nameAndArgs, 0, JavaTypeUtils::toExternalTypeString);
                String restype = select(tree, 2, JavaTypeUtils::toExternalTypeString);
                yield String.format("%s::%s(%s):%s", owner, methodName, String.join(", ", paramTypes), restype);
            }
            case JAVA_REF_CONSTRUCTOR_NAME -> {
                String owner = select(tree, 0, JavaTypeUtils::toExternalTypeString);
                ExternalizedTypeElement nameAndArgs = select(tree, 1, Function.identity());
                List<String> paramTypes = selectFrom(nameAndArgs, 0, JavaTypeUtils::toExternalTypeString);
                yield String.format("%s::(%s)", owner, String.join(", ", paramTypes));
            }
            case JAVA_REF_RECORD_NAME -> {
                String owner = select(tree, 0, JavaTypeUtils::toExternalTypeString);
                List<String> components = selectFrom(tree, 1, Function.identity()).stream()
                        .map(t -> {
                            String componentName = t.identifier();
                            String componentType = select(t, 0, JavaTypeUtils::toExternalTypeString);
                            return String.format("%s %s", componentType, componentName);
                        }).toList();
                yield String.format("(%s)%s", String.join(", ", components), owner);
            }
            default -> throw new UnsupportedOperationException("Unsupported ref: " + tree);
        };
    }

    public static boolean is(ExternalizedTypeElement tree, TypeElement typeElement) {
        return tree.equals(typeElement.externalize());
    }

    public static final boolean isPrimitive(String name) {
        return PRIMITIVE_TYPES.containsKey(name);
    }

    public static ExternalizedTypeElement flatten(ExternalizedTypeElement tree) {
        return switch (tree.identifier()) {
            case JAVA_TYPE_CLASS_NAME, JAVA_TYPE_ARRAY_NAME, JAVA_TYPE_PRIMITIVE_NAME,
                 JAVA_TYPE_WILDCARD_NAME, JAVA_TYPE_VAR_NAME ->
                    nameToType(String.format("%s\"%s\"", JAVA_TYPE_FLAT_NAME_PREFIX, toExternalTypeString(tree)));
            case JAVA_REF_FIELD_NAME, JAVA_REF_METHOD_NAME, JAVA_REF_CONSTRUCTOR_NAME, JAVA_REF_RECORD_NAME ->
                    nameToType(String.format("%s\"%s\"", JAVA_REF_FLAT_NAME_PREFIX, toExternalRefString(tree)));
            default -> new ExternalizedTypeElement(tree.identifier(), tree.arguments().stream().map(JavaTypeUtils::flatten).toList());
        };
    }

    public static ExternalizedTypeElement inflate(ExternalizedTypeElement tree) {
        String id = tree.identifier();
        if (id.startsWith(JAVA_TYPE_FLAT_NAME_PREFIX)) {
            return DescParser.parseJavaType(getDesc(id, JAVA_TYPE_FLAT_NAME_PREFIX));
        } else if (id.startsWith(JAVA_REF_FLAT_NAME_PREFIX)) {
            return DescParser.parseJavaRef(getDesc(id, JAVA_REF_FLAT_NAME_PREFIX));
        } else {
            return new ExternalizedTypeElement(tree.identifier(), tree.arguments().stream().map(JavaTypeUtils::inflate).toList());
        }
    }

    // internal utility methods

    private static final Map<String, JavaType> PRIMITIVE_TYPES = Map.of(
            "boolean", JavaType.BOOLEAN,
            "char", JavaType.CHAR,
            "byte", JavaType.BYTE,
            "short", JavaType.SHORT,
            "int", JavaType.INT,
            "float", JavaType.FLOAT,
            "long", JavaType.LONG,
            "double", JavaType.DOUBLE,
            "void", JavaType.VOID);

    private static String escapeInnerClassName(String s) {
        return (!s.isEmpty() && Character.isDigit(s.charAt(0))) ?
                "$" + s : s;
    }

    private static String unescapeInnerClassName(String s) {
        return (s.length() > 1 && s.charAt(0) == '$' &&
                Character.isDigit(s.charAt(1))) ?
                s.substring(1) : s;
    }

    private static ExternalizedTypeElement nameToType(String name) {
        return new ExternalizedTypeElement(name, List.of());
    }

    private static String typeToName(ExternalizedTypeElement tree) {
        if (!tree.arguments().isEmpty()) {
            throw new IllegalStateException("Unexpected type arguments");
        }
        return tree.identifier();
    }

    private static <T> T select(ExternalizedTypeElement tree, int index, Function<ExternalizedTypeElement, T> valueFunc) {
        if (index >= tree.arguments().size()) {
            throw new IllegalStateException("Invalid selection index");
        }
        return valueFunc.apply(tree.arguments().get(index));
    }

    private static <T> List<T> selectFrom(ExternalizedTypeElement tree, int startIncl, Function<ExternalizedTypeElement, T> valueFunc) {
        if (startIncl >= tree.arguments().size()) {
            return List.of();
        }
        return IntStream.range(startIncl, tree.arguments().size())
                .mapToObj(i -> valueFunc.apply(tree.arguments().get(i)))
                .toList();
    }

    private static String getDesc(String id, String prefix) {
        return id.substring(prefix.length() + 1, id.length() - 1);
    }
}

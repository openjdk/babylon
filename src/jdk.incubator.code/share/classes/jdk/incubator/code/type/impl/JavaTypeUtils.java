package jdk.incubator.code.type.impl;

import jdk.incubator.code.TypeElement;
import jdk.incubator.code.TypeElement.ExternalizedTypeElement;
import jdk.incubator.code.parser.impl.Lexer;
import jdk.incubator.code.parser.impl.Scanner;
import jdk.incubator.code.parser.impl.Tokens;
import jdk.incubator.code.type.*;
import jdk.incubator.code.type.RecordTypeRef.ComponentRef;
import jdk.incubator.code.type.WildcardType.BoundKind;

import java.lang.constant.ClassDesc;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

public class JavaTypeUtils {

    // useful type identifiers

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

    public enum Kind {
        FLATTENED_TYPE,
        FLATTENED_REF,
        INFLATED_TYPE,
        INFLATED_REF,
        OTHER;

        public static Kind of(ExternalizedTypeElement tree) {
            return switch (tree.identifier()) {
                case JAVA_TYPE_CLASS_NAME, JAVA_TYPE_ARRAY_NAME,
                     JAVA_TYPE_PRIMITIVE_NAME, JAVA_TYPE_WILDCARD_NAME,
                     JAVA_TYPE_VAR_NAME -> INFLATED_TYPE;
                case JAVA_REF_FIELD_NAME, JAVA_REF_CONSTRUCTOR_NAME,
                     JAVA_REF_METHOD_NAME, JAVA_REF_RECORD_NAME -> INFLATED_REF;
                case String s when s.startsWith(JAVA_TYPE_FLAT_NAME_PREFIX) -> FLATTENED_TYPE;
                case String s when s.startsWith(JAVA_REF_FLAT_NAME_PREFIX) -> FLATTENED_REF;
                default -> OTHER;
            };
        }
    }

    // Externalized Java type/ref factories

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

    // From externalized Java types/refs into actual Java types/refs

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
                        switch (Kind.of(t)) {
                            case INFLATED_TYPE -> toJavaType(t);
                            case INFLATED_REF -> toJavaRef(t);
                            default -> throw unsupported(t);
                        });
                JavaType bound = select(tree, 2, JavaTypeUtils::toJavaType);
                yield JavaType.typeVarRef(tvarName, owner, bound);
            }
            case JAVA_TYPE_PRIMITIVE_NAME -> {
                String primitiveName = select(tree, 0, JavaTypeUtils::typeToName);
                yield PRIMITIVE_TYPES.get(primitiveName);
            }
            default -> throw unsupported(tree);
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
            default -> throw unsupported(tree);
        };
    }

    // From externalized Java types/refs into external type/refs strings

    public static String toExternalTypeString(ExternalizedTypeElement tree) {
        return switch (tree.identifier()) {
            case JAVA_TYPE_CLASS_NAME -> {
                String className = select(tree, 0, JavaTypeUtils::typeToName);
                ExternalizedTypeElement enclosing = select(tree, 1, Function.identity());
                String typeargs = tree.arguments().size() == 2 ?
                        "" :
                        selectFrom(tree, 2, JavaTypeUtils::toExternalTypeString).stream()
                                .collect(Collectors.joining(", ", "<", ">"));
                if (isSameType(enclosing, JavaType.VOID)) {
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
                yield boundKind == BoundKind.EXTENDS && isSameType(bound, JavaType.J_L_OBJECT) ?
                        "?" :
                        String.format("? %s %s", boundKind.name().toLowerCase(), toExternalTypeString(bound));
            }
            case JAVA_TYPE_VAR_NAME -> {
                String tvarName = select(tree, 0, JavaTypeUtils::typeToName);
                String owner = select(tree, 1, t ->
                        switch (Kind.of(t)) {
                            case INFLATED_REF -> "&" + toExternalRefString(t);
                            case INFLATED_TYPE -> toExternalTypeString(t);
                            default -> throw unsupported(t);
                        });
                ExternalizedTypeElement bound = select(tree, 2, Function.identity());
                yield isSameType(bound, JavaType.J_L_OBJECT) ?
                        String.format("%s::<%s>", owner, tvarName) :
                        String.format("%s::<%s extends %s>", owner, tvarName, toExternalTypeString(bound));
            }
            case JAVA_TYPE_PRIMITIVE_NAME -> select(tree, 0, JavaTypeUtils::typeToName);
            default -> throw unsupported(tree);
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
            default -> throw unsupported(tree);
        };
    }

    // From external type/refs strings to externalized Java types/refs

    /**
     * Parse a type element from its readable textual form.
     * @param desc the textual form of the type element to be parsed
     * @return the type element
     */
    public static ExternalizedTypeElement parseExternalTypeString(String desc) {
        Scanner s = Scanner.factory().newScanner(desc);
        s.nextToken();
        return parseExternalTypeString(s);
    }

    /**
     * Parse a type element from its readable textual form.
     * @param desc the textual form of the type element to be parsed
     * @return the type element
     */
    public static ExternalizedTypeElement parseExternalRefString(String desc) {
        Scanner s = Scanner.factory().newScanner(desc);
        s.nextToken();
        return parseExternalRefString(s);
    }

    // From inflated externalized types/refs to flattened externalized types/refs and back

    public static ExternalizedTypeElement flatten(ExternalizedTypeElement tree) {
        return switch (Kind.of(tree)) {
            case INFLATED_TYPE -> nameToType(String.format("%s\"%s\"", JAVA_TYPE_FLAT_NAME_PREFIX, toExternalTypeString(tree)));
            case INFLATED_REF -> nameToType(String.format("%s\"%s\"", JAVA_REF_FLAT_NAME_PREFIX, toExternalRefString(tree)));
            default -> new ExternalizedTypeElement(tree.identifier(), tree.arguments().stream().map(JavaTypeUtils::flatten).toList());
        };
    }

    public static ExternalizedTypeElement inflate(ExternalizedTypeElement tree) {
        return switch (Kind.of(tree)) {
            case FLATTENED_TYPE -> parseExternalTypeString(getDesc(tree, JAVA_TYPE_FLAT_NAME_PREFIX));
            case FLATTENED_REF -> parseExternalRefString(getDesc(tree, JAVA_REF_FLAT_NAME_PREFIX));
            default -> new ExternalizedTypeElement(tree.identifier(), tree.arguments().stream().map(JavaTypeUtils::inflate).toList());
        };
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
            throw unsupported(tree);
        }
        return tree.identifier();
    }

    private static boolean isSameType(ExternalizedTypeElement tree, TypeElement typeElement) {
        return tree.equals(typeElement.externalize());
    }

    private static boolean isPrimitive(String name) {
        return PRIMITIVE_TYPES.containsKey(name);
    }

    private static <T> T select(ExternalizedTypeElement tree, int index, Function<ExternalizedTypeElement, T> valueFunc) {
        if (index >= tree.arguments().size()) {
            throw unsupported(tree);
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

    private static String getDesc(ExternalizedTypeElement tree, String prefix) {
        String id = tree.identifier();
        return id.substring(prefix.length() + 1, id.length() - 1);
    }

    //    JavaType:
    //        ClassType                                             // class type
    //        PrimitiveType                                         // primitive type
    //        TypeVar                                               // type variable
    //        JavaType '[' ']'                                      // array type
    //
    //    ClassType:
    //        ClassTypeNoPackage
    //        Package '.' ClassTypeNoPackage
    //
    //    Package:
    //        ident
    //        Package '.' ident
    //
    //    ClassTypeNoPackage:
    //        ident                                                 // simple class type
    //        ident '<' TypeArg* '>'                                // parameterized class type
    //        ClassTypeNoPackage '::' ClassTypeNoPackage            // nested class type
    //
    //    PrimitiveType:
    //        'boolean'
    //        'char'
    //        'byte'
    //        'short'
    //        'int'
    //        'float'
    //        'long'
    //        'double'
    //        'void'
    //
    //    TypeVar:
    //        '&' JavaRef TypeVarRest                                   // method/constructor type variable
    //        ClassType TypeVarRest                                 // class type variable
    //
    //    TypeVarRest:
    //        '::' '<' ident '>'
    //        '::' '<' ident 'extends' JavaType '>'
    //
    //    TypeArg:
    //        '?'                                                   // bivariant type argument
    //        '?' 'extends' JavaType                                // covariant type argument
    //        '?' 'super' JavaType                                  // contravariant type argument
    //        JavaType
    private static ExternalizedTypeElement parseExternalTypeString(Lexer l) {
        ExternalizedTypeElement type = null;
        if (l.token().kind == Tokens.TokenKind.AMP) {
            l.nextToken();
            // method or constructor type variable
            ExternalizedTypeElement owner = parseExternalRefString(l);
            l.accept(Tokens.TokenKind.COLCOL);
            type = parseTypeVariableRest(owner, l);
        } else if (l.token().kind == Tokens.TokenKind.IDENTIFIER) {
            if (JavaTypeUtils.isPrimitive(l.token().name())) {
                // primitive type
                type = JavaTypeUtils.primitiveType(l.token().name());
                l.nextToken();
            } else {
                // class type
                while (l.token().kind == Tokens.TokenKind.IDENTIFIER) {
                    StringBuilder className = new StringBuilder();
                    className.append(l.token().name());
                    l.nextToken();
                    while (type == null && l.token().kind == Tokens.TokenKind.DOT) {
                        l.accept(Tokens.TokenKind.DOT);
                        className.append(".");
                        className.append(l.token().name());
                        l.nextToken();
                    }
                    List<ExternalizedTypeElement> typeargs = new ArrayList<>();
                    if (l.acceptIf(Tokens.TokenKind.LT)) {
                        if (l.token().kind != Tokens.TokenKind.GT) {
                            typeargs.add(parseTypeArgument(l));
                            while (l.acceptIf(Tokens.TokenKind.COMMA)) {
                                typeargs.add(parseTypeArgument(l));
                            }
                        }
                        l.accept(Tokens.TokenKind.GT);
                    }
                    type = JavaTypeUtils.classType(className.toString(),
                            type, typeargs);
                    if (l.token(0).kind == Tokens.TokenKind.COLCOL) {
                        if (l.token(1).kind == Tokens.TokenKind.LT) {
                            // class type variable
                            l.nextToken();
                            type = parseTypeVariableRest(type, l);
                            break;
                        } else if (l.token(1).kind == Tokens.TokenKind.IDENTIFIER) {
                            if (l.token(2).kind == Tokens.TokenKind.LPAREN || l.token(2).kind == Tokens.TokenKind.COLON) {
                                // this looks like the middle of a field/method reference -- stop consuming
                                break;
                            }
                            l.nextToken(); // inner type, keep going
                        }
                    } else {
                        // not an inner type
                        break;
                    }
                }
            }
        }
        while (l.token().kind == Tokens.TokenKind.LBRACKET) {
            l.accept(Tokens.TokenKind.LBRACKET);
            l.accept(Tokens.TokenKind.RBRACKET);
            type = JavaTypeUtils.arrayType(type);
        }
        return type;
    }

    private static ExternalizedTypeElement parseTypeVariableRest(ExternalizedTypeElement owner, Lexer l) {
        l.accept(Tokens.TokenKind.LT);
        String name = l.token().name();
        l.nextToken();
        ExternalizedTypeElement bound = JavaType.J_L_OBJECT.externalize();
        if (l.token().kind == Tokens.TokenKind.IDENTIFIER &&
                l.token().name().equals("extends")) {
            l.nextToken();
            bound = parseExternalTypeString(l);
        }
        l.accept(Tokens.TokenKind.GT);
        return JavaTypeUtils.typeVarType(name, owner, bound);
    }

    private static ExternalizedTypeElement parseTypeArgument(Lexer l) {
        if (l.token().kind == Tokens.TokenKind.QUES) {
            // wildcard
            l.nextToken();
            ExternalizedTypeElement bound = JavaType.J_L_OBJECT.externalize();
            WildcardType.BoundKind bk = BoundKind.EXTENDS;
            if (l.token().kind == Tokens.TokenKind.IDENTIFIER) {
                bk = switch (l.token().name()) {
                    case "extends" -> BoundKind.EXTENDS;
                    case "super" -> BoundKind.SUPER;
                    default -> throw l.unexpected();
                };
                l.nextToken();
                bound = parseExternalTypeString(l);
            }
            return JavaTypeUtils.wildcardType(bk, bound);
        } else {
            return parseExternalTypeString(l);
        }
    }

    private static List<ExternalizedTypeElement> parseParameterTypes(Lexer l) {
        List<ExternalizedTypeElement> ptypes = new ArrayList<>();
        l.accept(Tokens.TokenKind.LPAREN);
        if (l.token().kind != Tokens.TokenKind.RPAREN) {
            ptypes.add(parseExternalTypeString(l));
            while (l.acceptIf(Tokens.TokenKind.COMMA)) {
                ptypes.add(parseExternalTypeString(l));
            }
        }
        l.accept(Tokens.TokenKind.RPAREN);
        return ptypes;
    }

    //    JavaRef:
    //        JavaType `::` ident ':' JavaType                      // field reference
    //        JavaType `::` ident '(' JavaType* ')' ':' JavaType    // method reference
    //        JavaType `::` '(' JavaType* ')'                       // constructor reference
    //        '(' RecordComponent* ')' JavaType                     // record reference
    //
    //    RecordComponent:
    //        JavaType ident
    private static ExternalizedTypeElement parseExternalRefString(Lexer l) {
        if (l.acceptIf(Tokens.TokenKind.LPAREN)) {
            // record type reference
            List<String> componentNames = new ArrayList<>();
            List<ExternalizedTypeElement> componentTypes = new ArrayList<>();
            if (l.token().kind != Tokens.TokenKind.RPAREN) {
                do {
                    componentTypes.add(parseExternalTypeString(l));
                    componentNames.add(l.accept(Tokens.TokenKind.IDENTIFIER).name());
                } while(l.acceptIf(Tokens.TokenKind.COMMA));
            }
            l.accept(Tokens.TokenKind.RPAREN);
            ExternalizedTypeElement recordType = parseExternalTypeString(l);
            return JavaTypeUtils.recordRef(recordType, componentNames, componentTypes);
        }
        ExternalizedTypeElement refType = parseExternalTypeString(l);

        l.accept(Tokens.TokenKind.COLCOL);
        if (l.token().kind == Tokens.TokenKind.LPAREN) {
            // constructor ref
            List<ExternalizedTypeElement> ptypes = parseParameterTypes(l);
            return JavaTypeUtils.constructorRef(refType, ptypes);
        }

        // field or method ref
        String memberName = l.accept(Tokens.TokenKind.IDENTIFIER).name();
        if (l.token().kind == Tokens.TokenKind.LPAREN) {
            // method ref
            List<ExternalizedTypeElement> params = parseParameterTypes(l);
            l.accept(Tokens.TokenKind.COLON);
            ExternalizedTypeElement rtype = parseExternalTypeString(l);
            return JavaTypeUtils.methodRef(memberName, refType, rtype, params);
        } else {
            // field ref
            l.accept(Tokens.TokenKind.COLON);
            ExternalizedTypeElement ftype = parseExternalTypeString(l);
            return JavaTypeUtils.fieldRef(memberName, refType, ftype);
        }
    }

    private static UnsupportedOperationException unsupported(ExternalizedTypeElement tree) {
        throw new UnsupportedOperationException("Unsupported type: " + tree);
    }
}

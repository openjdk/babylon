package java.lang.reflect.code.type;

import java.lang.constant.ClassDesc;
import java.lang.reflect.code.TypeElement;
import java.lang.reflect.code.TypeElement.ExternalizedTypeElement;
import java.lang.reflect.code.type.WildcardType.BoundKind;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Predicate;

public final class CoreTypeFactory {

    private CoreTypeFactory() {
    }

    // Code model type factory composed

    /**
     * Create a code model factory combining and composing the construction
     * of code model types with types constructed from the given type factory.
     *
     * @param f the type factory.
     * @return the code model factory.
     */
    public static TypeElementFactory codeModelTypeFactory(TypeElementFactory f) {
        class CodeModelFactory implements TypeElementFactory {
            final TypeElementFactory thisThenF = this.andThen(f);

            @Override
            public TypeElement constructType(TypeElement.ExternalizedTypeElement tree) {
                return switch (tree.identifier()) {
                    case VarType.NAME -> {
                        if (tree.arguments().size() != 1) {
                            throw new IllegalArgumentException();
                        }

                        TypeElement v = thisThenF.constructType(tree.arguments().getFirst());
                        if (v == null) {
                            throw new IllegalArgumentException("Bad type: " + tree);
                        }
                        yield VarType.varType(v);
                    }
                    case TupleType.NAME -> {
                        if (tree.arguments().isEmpty()) {
                            throw new IllegalArgumentException("Bad type: " + tree);
                        }

                        List<TypeElement> cs = new ArrayList<>(tree.arguments().size());
                        for (TypeElement.ExternalizedTypeElement child : tree.arguments()) {
                            TypeElement c = thisThenF.constructType(child);
                            if (c == null) {
                                throw new IllegalArgumentException("Bad type: " + tree);
                            }
                            cs.add(c);
                        }
                        yield TupleType.tupleType(cs);
                    }
                    case FunctionType.NAME -> {
                        if (tree.arguments().isEmpty()) {
                            throw new IllegalArgumentException("Bad type: " + tree);
                        }

                        TypeElement rt = thisThenF.constructType(tree.arguments().getFirst());
                        if (rt == null) {
                            throw new IllegalArgumentException("Bad type: " + tree);
                        }
                        List<TypeElement> pts = new ArrayList<>(tree.arguments().size() - 1);
                        for (TypeElement.ExternalizedTypeElement child : tree.arguments().subList(1, tree.arguments().size())) {
                            TypeElement c = thisThenF.constructType(child);
                            if (c == null) {
                                throw new IllegalArgumentException("Bad type: " + tree);
                            }
                            pts.add(c);
                        }
                        yield FunctionType.functionType(rt, pts);
                    }
                    default -> null;
                };
            }
        }
        if (f instanceof CodeModelFactory) {
            throw new IllegalArgumentException();
        }

        return new CodeModelFactory().thisThenF;
    }

    // Java type factory

    /**
     * The Java type factory.
     */
    public static final TypeElementFactory JAVA_TYPE_FACTORY = new TypeElementFactory() {
        @Override
        public JavaType constructType(TypeElement.ExternalizedTypeElement tree) {
            String identifier = tree.identifier();

            if (identifier.startsWith("[")) {
                // Array types are "flattened". Skip over '[', but keep track of them in 'dimensions'
                if (tree.arguments().size() != 1) {
                    throw badType(tree, "array type");
                }
                for (int i = 1; i < identifier.length(); i++) {
                    if (identifier.charAt(i) != '[') {
                        throw badType(tree, "array type");
                    }
                }
                JavaType elemType = constructType(tree.arguments().getFirst());
                return JavaType.array(elemType, identifier.length());
            } else if (identifier.equals("+") || identifier.equals("-")) {
                // wildcard type
                if (tree.arguments().size() != 1) {
                    throw badType(tree, "wildcard type argument");
                }
                BoundKind kind = identifier.equals("+") ?
                        BoundKind.EXTENDS : BoundKind.SUPER;
                return JavaType.wildcard(kind,
                        constructTypeArgument(tree, 0, NO_WILDCARDS));
            } else if (identifier.startsWith("#")) {
                // type-var
                if (tree.arguments().size() != 1) {
                    throw badType(tree, "type variable");
                }
                String[] parts = identifier.substring(1).split("::");
                if (parts.length == 2) {
                    // class type-var
                    return JavaType.typeVarRef(parts[1],
                            (ClassType)constructType(parseExTypeElem(parts[0])),
                            constructTypeArgument(tree, 0, NO_WILDCARDS));
                } else if (parts.length == 3) {
                    // method type-var
                    return JavaType.typeVarRef(parts[2],
                            parseMethodRef(String.format("%s::%s", parts[0], parts[1])),
                            constructTypeArgument(tree, 0, NO_WILDCARDS));
                } else {
                    throw badType(tree, "type variable");
                }
            } else if (identifier.equals(".")) {
                // qualified type
                if (tree.arguments().size() != 2) {
                    throw badType(tree, "qualified type");
                }
                ClassType enclType = (ClassType)constructTypeArgument(tree, 0, CLASS);
                ClassType innerType = (ClassType)constructTypeArgument(tree, 1, CLASS);
                // the inner class name is obtained by subtracting the name of the enclosing type
                // from the name of the inner type (and also dropping an extra '$')
                String innerName = innerType.toNominalDescriptor().displayName()
                        .substring(enclType.toNominalDescriptor().displayName().length() + 1);
                JavaType qual = JavaType.qualified(enclType, innerName);
                return (innerType.hasTypeArguments()) ?
                    JavaType.parameterized(qual, innerType.typeArguments()) : qual;
            } else {
                // primitive or reference
                JavaType t = switch (identifier) {
                    case "boolean" -> JavaType.BOOLEAN;
                    case "byte" -> JavaType.BYTE;
                    case "char" -> JavaType.CHAR;
                    case "short" -> JavaType.SHORT;
                    case "int" -> JavaType.INT;
                    case "long" -> JavaType.LONG;
                    case "float" -> JavaType.FLOAT;
                    case "double" -> JavaType.DOUBLE;
                    case "void" -> JavaType.VOID;
                    default -> JavaType.type(ClassDesc.of(identifier));
                };
                if (!tree.arguments().isEmpty()) {
                    if (t instanceof PrimitiveType) {
                        throw new IllegalArgumentException("primitive type: " + tree);
                    }
                    return JavaType.parameterized(t,
                            tree.arguments().stream().map(this::constructType).toList());
                } else {
                    return t;
                }
            }
        }

        static IllegalArgumentException badType(ExternalizedTypeElement tree, String str) {
            return new IllegalArgumentException(String.format("Bad %s: %s", str, tree));
        }

        private JavaType constructTypeArgument(ExternalizedTypeElement element, int index, Predicate<JavaType> filter) {
            ExternalizedTypeElement arg = element.arguments().get(index);
            JavaType type = constructType(arg);
            if (!filter.test(type)) {
                throw new IllegalArgumentException(String.format("Unexpected argument %s", element));
            } else {
                return type;
            }
        }

        private static Predicate<JavaType> NO_WILDCARDS = t -> !(t instanceof WildcardType);
        private static Predicate<JavaType> CLASS = t -> t instanceof ClassType;
    };


    /**
     * The core type factory that can construct instance of {@link JavaType}
     * or code model types such as {@link VarType} or {@link TupleType} that
     * may contain instances of those types.
     */
    public static final TypeElementFactory CORE_TYPE_FACTORY = codeModelTypeFactory(JAVA_TYPE_FACTORY);

    // Copied code in jdk.compiler module throws UOE
    static MethodRef parseMethodRef(String desc) {
/*__throw new UnsupportedOperationException();__*/        return java.lang.reflect.code.parser.impl.DescParser.parseMethodRef(desc);
    }

    // Copied code in jdk.compiler module throws UOE
    static TypeElement.ExternalizedTypeElement parseExTypeElem(String desc) {
/*__throw new UnsupportedOperationException();__*/        return java.lang.reflect.code.parser.impl.DescParser.parseExTypeElem(desc);
    }
}

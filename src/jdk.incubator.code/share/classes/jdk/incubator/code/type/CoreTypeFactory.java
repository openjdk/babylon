package jdk.incubator.code.type;

import java.lang.constant.ClassDesc;
import jdk.incubator.code.TypeElement;
import jdk.incubator.code.TypeElement.ExternalizedTypeElement;
import jdk.incubator.code.parser.impl.DescParser;
import jdk.incubator.code.type.WildcardType.BoundKind;
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
        // Returns JavaType or JavaRef
        @Override
        public TypeElement constructType(TypeElement.ExternalizedTypeElement tree) {
            String identifier = tree.identifier();

            if (identifier.startsWith("java.type:")) {
                String typestr = identifier.substring(identifier.indexOf(':') + 1);
                if (!typestr.startsWith("\"") || !typestr.endsWith("\"")) {
                    throw badType(tree, "bad type string in Java type: " + typestr);
                }
                return DescParser.parseJavaType(typestr.substring(1, typestr.length() - 1));
            } else if (identifier.startsWith("java.ref:")) {
                String typestr = identifier.substring(identifier.indexOf(':') + 1);
                if (!typestr.startsWith("\"") || !typestr.endsWith("\"")) {
                    throw badType(tree, "bad type string in Java type: " + typestr);
                }
                return DescParser.parseJavaRef(typestr.substring(1, typestr.length() - 1));
            } else {
                return null;
            }
        }

        static IllegalArgumentException badType(ExternalizedTypeElement tree, String str) {
            return new IllegalArgumentException(String.format("Bad %s: %s", str, tree));
        }

        private JavaType constructTypeArgument(ExternalizedTypeElement element, int index, Predicate<JavaType> filter) {
            ExternalizedTypeElement arg = element.arguments().get(index);
            JavaType type = (JavaType) constructType(arg);
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

    static MethodRef parseMethodRef(String desc) {
        return jdk.incubator.code.parser.impl.DescParser.parseMethodRef(desc);
    }

    static ConstructorRef parseConstructorRef(String desc) {
        return jdk.incubator.code.parser.impl.DescParser.parseConstructorRef(desc);
    }

    static TypeElement.ExternalizedTypeElement parseExTypeElem(String desc) {
        return jdk.incubator.code.parser.impl.DescParser.parseExTypeElem(desc);
    }
}

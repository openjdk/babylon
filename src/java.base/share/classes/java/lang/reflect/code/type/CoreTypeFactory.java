package java.lang.reflect.code.type;

import java.lang.constant.ClassDesc;
import java.lang.reflect.code.CodeType;
import java.lang.reflect.code.type.WildcardType.BoundKind;
import java.util.ArrayList;
import java.util.List;

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
    public static CodeTypeFactory codeModelTypeFactory(CodeTypeFactory f) {
        class CodeModelFactory implements CodeTypeFactory {
            final CodeTypeFactory thisThenF = this.andThen(f);

            @Override
            public CodeType constructType(CodeType.ExternalizedCodeType tree) {
                return switch (tree.identifier()) {
                    case VarType.NAME -> {
                        if (tree.arguments().size() != 1) {
                            throw new IllegalArgumentException();
                        }

                        CodeType v = thisThenF.constructType(tree.arguments().getFirst());
                        if (v == null) {
                            throw new IllegalArgumentException("Bad type: " + tree);
                        }
                        yield VarType.varType(v);
                    }
                    case TupleType.NAME -> {
                        if (tree.arguments().isEmpty()) {
                            throw new IllegalArgumentException("Bad type: " + tree);
                        }

                        List<CodeType> cs = new ArrayList<>(tree.arguments().size());
                        for (CodeType.ExternalizedCodeType child : tree.arguments()) {
                            CodeType c = thisThenF.constructType(child);
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

                        CodeType rt = thisThenF.constructType(tree.arguments().getFirst());
                        if (rt == null) {
                            throw new IllegalArgumentException("Bad type: " + tree);
                        }
                        List<CodeType> pts = new ArrayList<>(tree.arguments().size() - 1);
                        for (CodeType.ExternalizedCodeType child : tree.arguments().subList(1, tree.arguments().size())) {
                            CodeType c = thisThenF.constructType(child);
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
    public static final CodeTypeFactory JAVA_TYPE_FACTORY = new CodeTypeFactory() {
        @Override
        public CodeType constructType(CodeType.ExternalizedCodeType tree) {
            String identifier = tree.identifier();
            int dimensions = 0;
            if (identifier.startsWith("[")) {
                if (tree.arguments().size() != 1) {
                    throw new IllegalArgumentException("Bad type: " + tree);
                }
                for (int i = 1; i < identifier.length(); i++) {
                    if (identifier.charAt(i) != '[') {
                        throw new IllegalArgumentException("Bad type: " + tree);
                    }
                }
                dimensions = identifier.length();
                tree = tree.arguments().getFirst();
                identifier = tree.identifier();
            }

            List<JavaType> typeArguments = new ArrayList<>(tree.arguments().size());
            for (CodeType.ExternalizedCodeType child : tree.arguments()) {
                CodeType t = JAVA_TYPE_FACTORY.constructType(child);
                if (!(t instanceof JavaType a)) {
                    throw new IllegalArgumentException("Bad type: " + tree);
                }
                typeArguments.add(a);
            }
            if (identifier.equals("+") || identifier.equals("-")) {
                // wildcard type
                BoundKind kind = identifier.equals("+") ?
                        BoundKind.EXTENDS : BoundKind.SUPER;
                return JavaType.wildcard(kind, typeArguments.get(0));
            } else if (identifier.startsWith("#")) {
                // type-var
                if (typeArguments.size() != 1) {
                    throw new IllegalArgumentException("Bad type-variable bounds: " + tree);
                }
                String[] parts = identifier.substring(1).split("::");
                if (parts.length == 2) {
                    // class type-var
                    return JavaType.typeVarRef(parts[1],
                            (ClassType)constructType(parseExternalizedCodeType(parts[0])),
                            typeArguments.get(0));
                } else {
                    // method type-var
                    return JavaType.typeVarRef(parts[2],
                            parseMethodRef(String.format("%s::%s", parts[0], parts[1])),
                            typeArguments.get(0));
                }
            }
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
                default -> JavaType.ofNominalDescriptor(ClassDesc.of(identifier));
            };
            if (!typeArguments.isEmpty()) {
                t = JavaType.type(t, typeArguments);
            }
            return dimensions == 0 ?
                    t : JavaType.array(t, dimensions);
        }
    };


    /**
     * The core type factory that can construct instance of {@link JavaType}
     * or code model types such as {@link VarType} or {@link TupleType} that
     * may contain instances of those types.
     */
    public static final CodeTypeFactory CORE_TYPE_FACTORY = codeModelTypeFactory(JAVA_TYPE_FACTORY);

    // Copied code in jdk.compiler module throws UOE
    static MethodRef parseMethodRef(String desc) {
/*__throw new UnsupportedOperationException();__*/        return java.lang.reflect.code.parser.impl.DescParser.parseMethodRef(desc);
    }

    // Copied code in jdk.compiler module throws UOE
    static CodeType.ExternalizedCodeType parseExternalizedCodeType(String desc) {
/*__throw new UnsupportedOperationException();__*/        return java.lang.reflect.code.parser.impl.DescParser.parseExternalizedCodeType(desc);
    }
}

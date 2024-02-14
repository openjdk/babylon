package java.lang.reflect.code.type;

import java.lang.reflect.code.TypeElement;
import java.lang.reflect.code.type.impl.JavaTypeImpl;
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
    public static TypeElementFactory codeModelTypeFactory(TypeElementFactory f) {
        class CodeModelFactory implements TypeElementFactory {
            final TypeElementFactory thisThenF = this.andThen(f);

            @Override
            public TypeElement constructType(TypeDefinition tree) {
                if (tree.isArray()) {
                    return null;
                }
                return switch (tree.name()) {
                    case VarType.NAME -> {
                        if (tree.typeArguments().size() != 1) {
                            throw new IllegalArgumentException();
                        }

                        TypeElement v = thisThenF.constructType(tree.typeArguments().getFirst());
                        if (v == null) {
                            throw new IllegalArgumentException();
                        }
                        yield VarType.varType(v);
                    }
                    case TupleType.NAME -> {
                        if (tree.typeArguments().isEmpty()) {
                            throw new IllegalArgumentException();
                        }

                        List<TypeElement> cs = new ArrayList<>(tree.typeArguments().size());
                        for (TypeDefinition child : tree.typeArguments()) {
                            TypeElement c = thisThenF.constructType(child);
                            if (c == null) {
                                throw new IllegalArgumentException();
                            }
                            cs.add(c);
                        }
                        yield TupleType.tupleType(cs);
                    }
                    case FunctionType.NAME -> {
                        if (tree.typeArguments().isEmpty()) {
                            throw new IllegalArgumentException();
                        }

                        TypeElement rt = thisThenF.constructType(tree.typeArguments().getFirst());
                        if (rt == null) {
                            throw new IllegalArgumentException();
                        }
                        List<TypeElement> pts = new ArrayList<>(tree.typeArguments().size() - 1);
                        for (TypeDefinition child : tree.typeArguments().subList(1, tree.typeArguments().size())) {
                            TypeElement c = thisThenF.constructType(child);
                            if (c == null) {
                                throw new IllegalArgumentException();
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
        public TypeElement constructType(TypeDefinition tree) {
            String name = tree.name();
            int dimensions = tree.dimensions();
            List<JavaType> typeArguments = new ArrayList<>(tree.typeArguments().size());
            for (TypeDefinition child : tree.typeArguments()) {
                TypeElement t = JAVA_TYPE_FACTORY.constructType(child);
                if (!(t instanceof JavaType a)) {
                    throw new IllegalArgumentException();
                }
                typeArguments.add(a);
            }
            return new JavaTypeImpl(name, dimensions, typeArguments);
        }
    };


    /**
     * The core type factory that can construct instance of {@link JavaType}
     * or code model types such as {@link VarType} or {@link TupleType} that
     * may contain instances of those types.
     */
    public static final TypeElementFactory CORE_TYPE_FACTORY = codeModelTypeFactory(JAVA_TYPE_FACTORY);
}

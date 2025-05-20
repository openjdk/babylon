package jdk.incubator.code.type;

import java.lang.constant.ClassDesc;
import jdk.incubator.code.TypeElement;
import jdk.incubator.code.TypeElement.ExternalizedTypeElement;
import jdk.incubator.code.parser.impl.DescParser;
import jdk.incubator.code.type.WildcardType.BoundKind;
import jdk.incubator.code.type.impl.JavaTypeUtils;

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
            return JavaTypeUtils.toJavaType(tree);
        }
    };


    /**
     * The core type factory that can construct instance of {@link JavaType}
     * or code model types such as {@link VarType} or {@link TupleType} that
     * may contain instances of those types.
     */
    public static final TypeElementFactory CORE_TYPE_FACTORY = codeModelTypeFactory(JAVA_TYPE_FACTORY);
}

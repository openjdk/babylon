package jdk.incubator.code.dialect.core;

import jdk.incubator.code.TypeElement;
import jdk.incubator.code.dialect.factory.TypeElementFactory;

import java.util.ArrayList;
import java.util.List;

public sealed interface CoreType extends TypeElement permits
        FunctionType, TupleType, VarType {

    /**
     * Creates a composed type element factory for core type elements and type elements from the given
     * type element factory, where the core type elements can refer to type elements from the
     * given type element factory.
     *
     * @param f the type element factory.
     * @return the composed type element factory.
     */
    static TypeElementFactory coreTypeFactory(TypeElementFactory f) {
        class CodeModelFactory implements TypeElementFactory {
            final TypeElementFactory thisThenF = this.andThen(f);

            @Override
            public TypeElement constructType(ExternalizedTypeElement tree) {
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
                        for (ExternalizedTypeElement child : tree.arguments()) {
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
                        for (ExternalizedTypeElement child : tree.arguments().subList(1, tree.arguments().size())) {
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

    // @@@ move static type factory methods to here
}

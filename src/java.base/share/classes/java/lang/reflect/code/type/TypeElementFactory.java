package java.lang.reflect.code.type;

import java.lang.reflect.code.TypeElement;
import java.util.ArrayList;
import java.util.List;

@FunctionalInterface
public interface TypeElementFactory {
//    record TypeTree(String name, List<TypeTree> children) {}

    // Use TypeDefinition as temporary intermediate type representation
    TypeElement constructType(TypeDefinition tree);

    default TypeElementFactory andThen(TypeElementFactory after) {
        return t -> {
            TypeElement te = constructType(t);
            return te != null ? te : after.constructType(t);
        };
    }

    static TypeElementFactory factory(TypeElementFactory f) {
        class CodeModelFactory implements TypeElementFactory {
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

                        TypeElement v = f.constructType(tree.typeArguments().getFirst());
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
                            TypeElement c = f.constructType(child);
                            if (c == null) {
                                throw new IllegalArgumentException();
                            }
                            cs.add(c);
                        }
                        yield TupleType.tupleType(cs);
                    }
                    default -> null;
                };
            }
        }
        if (f instanceof CodeModelFactory) {
            throw new IllegalArgumentException();
        }

        return new CodeModelFactory().andThen(f);
    }
}
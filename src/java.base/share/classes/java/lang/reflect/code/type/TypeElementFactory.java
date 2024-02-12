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
}
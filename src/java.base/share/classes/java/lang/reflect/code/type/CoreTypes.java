package java.lang.reflect.code.type;

import java.lang.reflect.code.TypeElement;
import java.lang.reflect.code.type.impl.JavaTypeImpl;
import java.util.ArrayList;
import java.util.List;

public class CoreTypes {

    private CoreTypes() {
    }

    // Type factory

    public static final TypeElementFactory FACTORY = new TypeElementFactory() {
        @Override
        public TypeElement constructType(TypeDefinition tree) {
            String name = tree.name();
            int dimensions = tree.dimensions();
            List<JavaType> typeArguments = new ArrayList<>(tree.typeArguments().size());
            for (TypeDefinition child : tree.typeArguments()) {
                TypeElement t = FACTORY.constructType(child);
                if (!(t instanceof JavaType a)) {
                    throw new IllegalArgumentException();
                }
                typeArguments.add(a);
            }
            return new JavaTypeImpl(name, dimensions, typeArguments);
        }
    };

    public static final TypeElementFactory FACTORY2 = TypeElementFactory.factory(FACTORY);
}

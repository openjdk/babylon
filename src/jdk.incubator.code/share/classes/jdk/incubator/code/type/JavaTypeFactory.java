package jdk.incubator.code.type;

import jdk.incubator.code.TypeElement.ExternalizedTypeElement;
import jdk.incubator.code.type.WildcardType.BoundKind;

import java.lang.constant.ClassDesc;
import java.util.List;
import java.util.Map;

class JavaTypeFactory implements TypeElementFactory {
    // Returns JavaType or JavaRef
    @Override
    public JavaType constructType(ExternalizedTypeElement tree) {
        return constructType(null, tree);
    }

    //    JavaType:
    //        ClassType                                             // class type
    //        PrimitiveType                                         // primitive type
    //        TypeVar                                               // type variable
    //        '[' JavaType ']'                                      // array type
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
    //        ClassType '::' ClassType                               // nested class type
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
    //        '(' JavaRef ')' TypeVarRest                           // method/constructor type variable
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
    JavaType constructType(JavaType qualifier, ExternalizedTypeElement tree) {
        return switch (tree.identifier()) {
            case "" -> {
                // array type
                JavaType elem = constructType(tree.arguments().get(0));
                yield JavaType.array(elem);
            }
            case "::" -> {
                if (tree.arguments().get(1).identifier().isEmpty()) {
                    // type-var
                    TypeVariableType.Owner owner = constructTypeVarOwner(qualifier, tree.arguments().get(0));
                    ExternalizedTypeElement tvarDecl = tree.arguments().get(1);
                    boolean hasBounds = false;
                    if (tvarDecl.arguments().get(0).identifier().equals("extends")) {
                        hasBounds = true;
                        tvarDecl = tvarDecl.arguments().get(0);
                    }
                    JavaType bound = hasBounds ?
                            constructType(tvarDecl.arguments().get(1)) :
                            JavaType.J_L_OBJECT;
                    yield JavaType.typeVarRef(tvarDecl.arguments().get(0).identifier(),
                            owner, bound);
                } else {
                    // inner class type
                    qualifier = constructClassType(qualifier, tree.arguments().get(0));
                    yield constructType(qualifier, tree.arguments().get(1));
                }
            }
            default -> {
                JavaType primitiveType = PRIMITIVE_TYPES.get(tree.identifier());
                yield (primitiveType != null) ?
                        primitiveType :
                        constructClassType(qualifier, tree);
            }
        };
    }

    TypeVariableType.Owner constructTypeVarOwner(JavaType qualifier, ExternalizedTypeElement type) {
        return switch (type.identifier()) {
            case "" -> (TypeVariableType.Owner) constructRef(type.arguments().get(0));
            default -> (TypeVariableType.Owner) constructClassType(qualifier, type);
        };
    }

    JavaType constructTypeArguments(ExternalizedTypeElement type) {
        return switch (type.identifier()) {
            case "extends" -> JavaType.wildcard(BoundKind.EXTENDS, constructType(type.arguments().get(1)));
            case "super" -> JavaType.wildcard(BoundKind.SUPER, constructType(type.arguments().get(1)));
            case "?" -> JavaType.wildcard(BoundKind.EXTENDS, JavaType.J_L_OBJECT);
            default -> constructType(type);
        };
    }

    JavaType constructClassType(JavaType qualifier, ExternalizedTypeElement type) {
        JavaType classType = qualifier == null ?
                JavaType.type(ClassDesc.of(type.identifier())) :
                JavaType.qualified(qualifier, type.identifier());
        if (!type.arguments().isEmpty()) {
            classType = JavaType.parameterized(classType,
                    type.arguments().stream().map(this::constructTypeArguments).toList());
        }
        return classType;
    }

    //    JavaRef:
    //        JavaType '::' ident ':' JavaType                      // field reference
    //        JavaType '::' ident '(' JavaType* ')' ':' JavaType    // method reference
    //        JavaType '::' '(' JavaType* ')'                       // constructor reference
    //        ident  '(' RecordComponent* ')'                       // record reference
    //
    //    RecordComponent:
    //        ident ':' JavaType
    JavaRef constructRef(ExternalizedTypeElement type) {
        return constructRef(null, type);
    }

    JavaRef constructRef(JavaType qualifier, ExternalizedTypeElement type) {
        return switch (type.identifier()) {
            case "::" -> {
                if (type.arguments().get(1).identifier().equals("::")) {
                    // owner type is an inner class type
                    qualifier = constructClassType(qualifier, type.arguments().get(0));
                    yield constructRef(qualifier, type.arguments().get(1));
                } else {
                    JavaType ownerType = constructType(qualifier, type.arguments().get(0));
                    if (type.arguments().get(1).identifier().equals(":")) {
                        // field ref or method ref
                        ExternalizedTypeElement nameAndType = type.arguments().get(1);
                        String refName = nameAndType.arguments().get(0).identifier();
                        ExternalizedTypeElement refType = nameAndType.arguments().get(1);
                        if (nameAndType.arguments().get(0).arguments().isEmpty()) {
                            // field ref
                            JavaType fieldType = constructType(refType);
                            yield FieldRef.field(ownerType, refName, fieldType);
                        } else {
                            // method ref
                            List<JavaType> paramtypes = nameAndType.arguments().get(0).arguments().stream()
                                    .map(this::constructType).toList();
                            if (paramtypes.size() == 1 && paramtypes.get(0) == JavaType.VOID) {
                                paramtypes = List.of();
                            }
                            JavaType restype = constructType(refType);
                            yield MethodRef.method(ownerType, refName, restype, paramtypes);
                        }
                    } else {
                        // constructor ref
                        List<JavaType> paramtypes = type.arguments().get(1).arguments().stream()
                                .map(this::constructType).toList();
                        if (paramtypes.size() == 1 && paramtypes.get(0) == JavaType.VOID) {
                            paramtypes = List.of();
                        }
                        yield ConstructorRef.constructor(ownerType, paramtypes);
                    }
                }
            }
            // record ref
            default -> RecordTypeRef.recordType(
                    JavaType.type(ClassDesc.of(type.identifier())),
                    type.arguments().stream().map(c -> {
                        JavaType ctype = constructType(c.arguments().get(1));
                        return new RecordTypeRef.ComponentRef(ctype, c.arguments().get(0).identifier());
                    }).toList());
        };
    }

    // @@@: Validation?
    static IllegalArgumentException badType(ExternalizedTypeElement tree, String str) {
        return new IllegalArgumentException(String.format("Bad %s: %s", str, tree));
    }

    static final Map<String, JavaType> PRIMITIVE_TYPES = Map.of(
            "boolean", JavaType.BOOLEAN,
            "char", JavaType.CHAR,
            "byte", JavaType.BYTE,
            "short", JavaType.SHORT,
            "int", JavaType.INT,
            "float", JavaType.FLOAT,
            "long", JavaType.LONG,
            "double", JavaType.DOUBLE,
            "void", JavaType.VOID);
}

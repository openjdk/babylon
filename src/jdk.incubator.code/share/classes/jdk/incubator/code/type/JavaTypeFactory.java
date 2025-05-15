package jdk.incubator.code.type;

import jdk.incubator.code.TypeElement.ExternalizedTypeElement;
import jdk.incubator.code.type.WildcardType.BoundKind;

import java.lang.constant.ClassDesc;
import java.util.List;
import java.util.Map;
import java.util.function.Predicate;

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
                // array type:
                // < type >
                checkArity(tree, 1);
                JavaType elem = constructType(tree.arguments().get(0));
                yield JavaType.array(elem);
            }
            case "::" -> {
                checkArity(tree, 2);
                if (tree.arguments().get(1).identifier().isEmpty()) {
                    // type-var:
                    // ::<tvarowner, <ident>>
                    // ::<tvarowner, extends<ident, type>>
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
                    // inner class type:
                    // ::<classtype, type>
                    qualifier = constructClassType(qualifier, tree.arguments().get(0));
                    yield constructType(qualifier, tree.arguments().get(1));
                }
            }
            default -> {
                // primitive or class type:
                // ident
                // ident < typargs >
                JavaType primitiveType = PRIMITIVE_TYPES.get(tree.identifier());
                yield (primitiveType != null) ?
                        primitiveType :
                        constructClassType(qualifier, tree);
            }
        };
    }

    TypeVariableType.Owner constructTypeVarOwner(JavaType qualifier, ExternalizedTypeElement tree) {
        // tvar owner:
        // < ref >
        // classtype
        return switch (tree.identifier()) {
            case "" -> {
                checkArity(tree, 1);
                yield (TypeVariableType.Owner) constructRef(tree.arguments().get(0));
            }
            default -> (TypeVariableType.Owner) constructClassType(qualifier, tree);
        };
    }

    JavaType constructTypeArguments(ExternalizedTypeElement tree) {
        // typargs:
        // ?
        // extends<?, type>
        // super<?, type>
        // type
        return switch (tree.identifier()) {
            case "extends" -> {
                checkArity(tree, 2);
                yield JavaType.wildcard(BoundKind.EXTENDS, constructType(tree.arguments().get(1)));
            }
            case "super" -> {
                checkArity(tree, 2);
                yield JavaType.wildcard(BoundKind.SUPER, constructType(tree.arguments().get(1)));
            }
            case "?" -> {
                checkArity(tree, 0);
                yield JavaType.wildcard(BoundKind.EXTENDS, JavaType.J_L_OBJECT);
            }
            default -> constructType(tree);
        };
    }

    JavaType constructClassType(JavaType qualifier, ExternalizedTypeElement tree) {
        JavaType classType = qualifier == null ?
                JavaType.type(ClassDesc.of(tree.identifier())) :
                JavaType.qualified(qualifier, tree.identifier());
        if (!tree.arguments().isEmpty()) {
            classType = JavaType.parameterized(classType,
                    tree.arguments().stream().map(this::constructTypeArguments).toList());
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
    JavaRef constructRef(ExternalizedTypeElement tree) {
        return constructRef(null, tree);
    }

    JavaRef constructRef(JavaType qualifier, ExternalizedTypeElement tree) {
        return switch (tree.identifier()) {
            case "::" -> {
                checkArity(tree, 2);
                if (tree.arguments().get(1).identifier().equals("::")) {
                    // owner type is an inner class type
                    qualifier = constructClassType(qualifier, tree.arguments().get(0));
                    yield constructRef(qualifier, tree.arguments().get(1));
                } else {
                    JavaType ownerType = constructType(qualifier, tree.arguments().get(0));
                    if (tree.arguments().get(1).identifier().equals(":")) {
                        // field ref or method ref
                        ExternalizedTypeElement nameAndType = tree.arguments().get(1);
                        checkArity(nameAndType, 2);
                        String refName = nameAndType.arguments().get(0).identifier();
                        ExternalizedTypeElement refType = nameAndType.arguments().get(1);
                        if (nameAndType.arguments().get(0).arguments().isEmpty()) {
                            // field ref:
                            // ::<type, :<ident, type>>
                            JavaType fieldType = constructType(refType);
                            yield FieldRef.field(ownerType, refName, fieldType);
                        } else {
                            // method ref:
                            // ::<type, :<ident< type, type, ...>, type>>
                            // ::<type, :<ident< 'void' >, type>>
                            List<JavaType> paramtypes = constructRefParameters(nameAndType.arguments().get(0));
                            JavaType restype = constructType(refType);
                            yield MethodRef.method(ownerType, refName, restype, paramtypes);
                        }
                    } else {
                        // constructor ref:
                        // ::<type, < type, type, ...>>
                        // ::<type, < 'void' >>
                        List<JavaType> paramtypes = constructRefParameters(tree.arguments().get(1));
                        yield ConstructorRef.constructor(ownerType, paramtypes);
                    }
                }
            }
            // record ref:
            // ident<<:<ident, type>,<:<ident, type>, ...>
            default -> RecordTypeRef.recordType(
                    JavaType.type(ClassDesc.of(tree.identifier())),
                    tree.arguments().stream().map(c -> {
                        JavaType ctype = constructType(c.arguments().get(1));
                        return new RecordTypeRef.ComponentRef(ctype, c.arguments().get(0).identifier());
                    }).toList());
        };
    }

    List<JavaType> constructRefParameters(ExternalizedTypeElement tree) {
        List<JavaType> paramtypes = tree.arguments().stream()
                .map(this::constructType).toList();
        if (paramtypes.size() == 1 && paramtypes.get(0) == JavaType.VOID) {
            paramtypes = List.of();
        }
        return paramtypes;
    }

    static void checkArity(ExternalizedTypeElement tree, int expectedArity) {
        if (tree.arguments().size() != expectedArity) {
            throw badType(tree, "type arity; expected: \"" + expectedArity + "\"");
        }
    }

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

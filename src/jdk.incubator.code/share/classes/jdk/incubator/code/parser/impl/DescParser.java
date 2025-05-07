/*
 * Copyright (c) 2024, Oracle and/or its affiliates. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.  Oracle designates this
 * particular file as subject to the "Classpath" exception as provided
 * by Oracle in the LICENSE file that accompanied this code.
 *
 * This code is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
 * version 2 for more details (a copy is included in the LICENSE file that
 * accompanied this code).
 *
 * You should have received a copy of the GNU General Public License version
 * 2 along with this work; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
 * or visit www.oracle.com if you need additional information or have any
 * questions.
 */

package jdk.incubator.code.parser.impl;

import jdk.incubator.code.parser.impl.Tokens.TokenKind;
import jdk.incubator.code.type.*;
import jdk.incubator.code.TypeElement;
import jdk.incubator.code.type.RecordTypeRef;
import jdk.incubator.code.type.WildcardType.BoundKind;
import jdk.incubator.code.type.impl.ConstructorRefImpl;
import jdk.incubator.code.type.impl.FieldRefImpl;
import jdk.incubator.code.type.impl.MethodRefImpl;
import jdk.incubator.code.type.impl.RecordTypeRefImpl;

import java.lang.constant.ClassDesc;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public final class DescParser {
    private DescParser() {}

    /**
     * Parse an externalized type element from its serialized textual form.
     * @param desc the serialized externalized type element
     * @return the externalized type element
     */
    public static TypeElement.ExternalizedTypeElement parseExTypeElem(String desc) {
        Scanner s = Scanner.factory().newScanner(desc);
        s.nextToken();
        return parseExTypeElem(s);
    }

    /**
     * Parse a type element from its readable textual form.
     * @param desc the textual form of the type element to be parsed
     * @return the type element
     */
    public static TypeElement parseJavaType(String desc) {
        Scanner s = Scanner.factory().newScanner(desc);
        s.nextToken();
        return parseJavaType(s);
    }

    /**
     * Parse a type element from its readable textual form.
     * @param desc the textual form of the type element to be parsed
     * @return the type element
     */
    public static TypeElement parseJavaRef(String desc) {
        Scanner s = Scanner.factory().newScanner(desc);
        s.nextToken();
        return parseJavaRef(s);
    }

    /**
     * Parse a method reference from its serialized textual form.
     *
     * @param desc the serialized method reference
     * @return the method reference
     */
    public static MethodRef parseMethodRef(String desc) {
        Scanner s = Scanner.factory().newScanner(desc);
        s.nextToken();
        return parseMethodRef(s);
    }

    /**
     * Parse a constructor reference from its serialized textual form.
     *
     * @param desc the serialized constructor reference
     * @return the constructor reference
     */
    public static ConstructorRef parseConstructorRef(String desc) {
        Scanner s = Scanner.factory().newScanner(desc);
        s.nextToken();
        return parseConstructorRef(s);
    }

    /**
     * Parse a field reference from its serialized textual form.
     *
     * @param desc the serialized field reference
     * @return the field reference
     */
    public static FieldRef parseFieldRef(String desc) {
        Scanner s = Scanner.factory().newScanner(desc);
        s.nextToken();
        return parseFieldRef(s);
    }

    /**
     * Parse a record type reference from its serialized textual form.
     *
     * @param desc the serialized record type reference
     * @return the record type reference
     */
    public static RecordTypeRef parseRecordTypeRef(String desc) {
        Scanner s = Scanner.factory().newScanner(desc);
        s.nextToken();
        return parseRecordTypeRef(s);
    }

    public static TypeElement.ExternalizedTypeElement parseExTypeElem(Lexer l) {
        StringBuilder identifier = new StringBuilder();
        identloop: while (true) {
            switch (l.token().kind) {
                case COLON, DOT -> identifier.append(l.token().kind.name);
                case IDENTIFIER -> identifier.append(l.token().name());
                case STRINGLITERAL -> {
                    identifier.append('"');
                    identifier.append(l.token().stringVal());
                    identifier.append('"');
                }
                default -> {
                    break identloop;
                }
            }
            l.nextToken();
        }
        List<TypeElement.ExternalizedTypeElement> args = new ArrayList<>();
        if (l.token().kind == TokenKind.LT) {
            args.add(parseExTypeElem(l));
            while (l.token().kind == TokenKind.COMMA) {
                l.accept(TokenKind.COMMA);
                args.add(parseExTypeElem(l));
            }
            l.accept(TokenKind.GT);
        }

        return new TypeElement.ExternalizedTypeElement(identifier.toString(), args);
    }

//    Type:
//    TypeNoPackage
//    Package TypeNoPackage
//
//    Package:
//    ident ('.' ident)
//
//    TypeNoPackage:
//    ident						// class type
//    ident '<' TypeArg* '>'				// parameterized class type
//    TypeNoPackage '[]'				// array type
//    TypeNoPackage '$' TypeNoPackage		// nested type
//
//    TypeArg:
//            '?' 						// bivariant type argument
//            '?' 'extends' Type				// covariant type argument
//            '?' 'super' Type				// contravariant type argument
//            Type						// invariant type argument
//
//    Ref:
//    Type `::` ident '(' Type* ')' Type 		// method/field
//    Type `::` '<new>' '(' Type* ')' Type 		// constructor

    static final Map<String, JavaType> PRIMITIVE_TYPES = Map.of(
            "boolean", JavaType.BOOLEAN,
            "char", JavaType.CHAR,
            "short", JavaType.SHORT,
            "int", JavaType.INT,
            "float", JavaType.FLOAT,
            "long", JavaType.LONG,
            "double", JavaType.DOUBLE,
            "void", JavaType.VOID);

    public static JavaType parseJavaType(Lexer l) {
        JavaType type = null;
        if (l.token().kind == TokenKind.IDENTIFIER) {
            type = PRIMITIVE_TYPES.get(l.token().name());
            if (type != null) {
                // primitive type
                l.nextToken();
            } else {
                // class type
                do {
                    StringBuilder className = new StringBuilder();
                    className.append(type == null ?
                            l.token().name() :
                            l.token().name().substring(1));
                    l.nextToken();
                    while (type == null && l.token().kind == TokenKind.DOT) {
                        l.accept(TokenKind.DOT);
                        className.append(".");
                        className.append(l.token().name());
                        l.nextToken();
                    }
                    type = (type == null) ?
                            JavaType.type(ClassDesc.of(className.toString())) :
                            JavaType.qualified(type, className.toString());
                    if (l.acceptIf(TokenKind.LT)) {
                        List<JavaType> typeargs = new ArrayList<>();
                        if (l.token().kind != TokenKind.GT) {
                            typeargs.add(parseTypeArgument(l));
                            while (l.acceptIf(TokenKind.COMMA)) {
                                typeargs.add(parseTypeArgument(l));
                            }
                        }
                        l.accept(TokenKind.GT);
                        type = JavaType.parameterized(type, typeargs);
                    }
                } while (l.token().kind == TokenKind.IDENTIFIER && l.token().name().startsWith("$"));
                //        if (l.token().kind == TokenKind.COLCOL) {
                //            // this is a type-variable reference
                //            l.nextToken();
                //            String name = l.token().name();
                //            l.nextToken();
                //            TypeVariableType.Owner owner = (TypeVariableType.Owner)type;
                //            if (l.token().kind == TokenKind.LPAREN) {
                //                // constructor or method reference
                //                List<TypeElement> params = parseParameterTypes(l);
                //                if (name.equals("<new>")) {
                //                    owner = ConstructorRef.constructor(type, params);
                //                } else {
                //                    JavaType restype = parseTypeElem(l);
                //                    owner = MethodRef.method(type, name, restype, params);
                //                }
                //                l.accept(TokenKind.COLCOL);
                //                name = l.token().name();
                //                l.nextToken();
                //            }
                //            if (!l.token().name().equals("extends")) {
                //                throw new IllegalArgumentException();
                //            }
                //            l.nextToken();
                //            JavaType bound = parseTypeElem(l);
                //            return JavaType.typeVarRef(name, owner, bound);
                //        }
            }
        }
        while (l.token().kind == TokenKind.LBRACKET) {
            l.accept(TokenKind.LBRACKET);
            l.accept(TokenKind.RBRACKET);
            type = JavaType.array(type);
        }
        return type;
    }

    public static JavaType parseTypeArgument(Lexer l) {
        if (l.token().kind == TokenKind.QUES) {
            // wildcard
            l.nextToken();
            if (l.token().kind == TokenKind.IDENTIFIER) {
                WildcardType.BoundKind bk = switch (l.token().name()) {
                    case "extends" -> BoundKind.EXTENDS;
                    case "super" -> BoundKind.SUPER;
                    default -> throw new IllegalArgumentException("Bad wildcard bound");
                };
                l.nextToken();
                JavaType bound = parseJavaType(l);
                return JavaType.wildcard(bk, bound);
            } else {
                return JavaType.wildcard(BoundKind.EXTENDS, JavaType.J_L_OBJECT);
            }
        } else {
            return parseJavaType(l);
        }
    }

    static List<TypeElement> parseParameterTypes(Lexer l) {
        List<TypeElement> ptypes = new ArrayList<>();
        l.accept(Tokens.TokenKind.LPAREN);
        if (l.token().kind != Tokens.TokenKind.RPAREN) {
            ptypes.add(parseJavaType(l));
            while (l.acceptIf(Tokens.TokenKind.COMMA)) {
                ptypes.add(parseJavaType(l));
            }
        }
        l.accept(Tokens.TokenKind.RPAREN);
        return ptypes;
    }

    // (T, T, T, T)R
    static FunctionType parseMethodType(Lexer l) {
        List<TypeElement> ptypes = parseParameterTypes(l);
        TypeElement rtype = parseJavaType(l);
        return FunctionType.functionType(rtype, ptypes);
    }

    static JavaRef parseJavaRef(Lexer l) {
        TypeElement refType = parseJavaType(l);

        l.accept(Tokens.TokenKind.COLCOL);

        if (l.acceptIf(TokenKind.LPAREN)) {
            // constructor ref
            List<TypeElement> ptypes = parseParameterTypes(l);
            return new ConstructorRefImpl(FunctionType.functionType(refType, ptypes));
        }

        // field or method ref
        String memberName = l.accept(Tokens.TokenKind.IDENTIFIER).name();
        l.accept(TokenKind.COLON);
        if (l.token().kind == TokenKind.LPAREN) {
            // method ref
            return new MethodRefImpl(refType, memberName, parseMethodType(l));
        } else {
            // field ref
            return new FieldRefImpl(refType, memberName, parseJavaType(l));
        }
    }

    static MethodRef parseMethodRef(Lexer l) {
        if (parseJavaRef(l) instanceof MethodRef mr) {
            return mr;
        } else {
            throw new IllegalArgumentException("Not a method ref");
        }
    }

    static FieldRef parseFieldRef(Lexer l) {
        if (parseJavaRef(l) instanceof FieldRef fr) {
            return fr;
        } else {
            throw new IllegalArgumentException("Not a field ref");
        }
    }

    static ConstructorRef parseConstructorRef(Lexer l) {
        if (parseJavaRef(l) instanceof ConstructorRef cr) {
            return cr;
        } else {
            throw new IllegalArgumentException("Not a constructor ref");
        }
    }

    static RecordTypeRef parseRecordTypeRef(Lexer l) {
        List<RecordTypeRef.ComponentRef> components = new ArrayList<>();
        l.accept(Tokens.TokenKind.LPAREN);
        if (l.token().kind != Tokens.TokenKind.RPAREN) {
            do {
                TypeElement componentType = parseJavaType(l);
                String componentName = l.accept(Tokens.TokenKind.IDENTIFIER).name();

                components.add(new RecordTypeRef.ComponentRef(componentType, componentName));
            } while(l.acceptIf(Tokens.TokenKind.COMMA));
        }
        l.accept(Tokens.TokenKind.RPAREN);
        TypeElement recordType = parseJavaType(l);
        return new RecordTypeRefImpl(recordType, components);
    }
}

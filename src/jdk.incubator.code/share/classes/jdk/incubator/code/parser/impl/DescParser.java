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

import jdk.incubator.code.TypeElement.ExternalizedTypeElement;
import jdk.incubator.code.parser.impl.Tokens.TokenKind;
import jdk.incubator.code.type.*;
import jdk.incubator.code.TypeElement;
import jdk.incubator.code.type.WildcardType.BoundKind;
import jdk.incubator.code.type.impl.JavaTypeUtils;

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
    public static ExternalizedTypeElement parseJavaType(String desc) {
        Scanner s = Scanner.factory().newScanner(desc);
        s.nextToken();
        return parseJavaType(s);
    }

    /**
     * Parse a type element from its readable textual form.
     * @param desc the textual form of the type element to be parsed
     * @return the type element
     */
    public static ExternalizedTypeElement parseJavaRef(String desc) {
        Scanner s = Scanner.factory().newScanner(desc);
        s.nextToken();
        return parseJavaRef(s);
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
            l.accept(TokenKind.LT);
            args.add(parseExTypeElem(l));
            while (l.token().kind == TokenKind.COMMA) {
                l.accept(TokenKind.COMMA);
                args.add(parseExTypeElem(l));
            }
            l.accept(TokenKind.GT);
        }

        return new TypeElement.ExternalizedTypeElement(identifier.toString(), args);
    }

    //    JavaType:
    //        ClassType                                             // class type
    //        PrimitiveType                                         // primitive type
    //        TypeVar                                               // type variable
    //        JavaType '[' ']'                                      // array type
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
    //        ClassType '$' ClassType                               // nested class type
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
    public static ExternalizedTypeElement parseJavaType(Lexer l) {
        ExternalizedTypeElement type = null;
        if (l.token().kind == TokenKind.LPAREN) {
            l.nextToken();
            // method or constructor type variable
            ExternalizedTypeElement owner = parseJavaRef(l);
            l.accept(TokenKind.RPAREN);
            l.accept(TokenKind.COLCOL);
            type = parseTypeVariableRest(owner, l);
        } else if (l.token().kind == TokenKind.IDENTIFIER) {
            if (JavaTypeUtils.isPrimitive(l.token().name())) {
                // primitive type
                type = JavaTypeUtils.primitiveType(l.token().name());
                l.nextToken();
            } else {
                // class type
                while (l.token().kind == TokenKind.IDENTIFIER) {
                    StringBuilder className = new StringBuilder();
                    className.append(l.token().name());
                    l.nextToken();
                    while (type == null && l.token().kind == TokenKind.DOT) {
                        l.accept(TokenKind.DOT);
                        className.append(".");
                        className.append(l.token().name());
                        l.nextToken();
                    }
                    List<ExternalizedTypeElement> typeargs = new ArrayList<>();
                    if (l.acceptIf(TokenKind.LT)) {
                        if (l.token().kind != TokenKind.GT) {
                            typeargs.add(parseTypeArgument(l));
                            while (l.acceptIf(TokenKind.COMMA)) {
                                typeargs.add(parseTypeArgument(l));
                            }
                        }
                        l.accept(TokenKind.GT);
                    }
                    type = JavaTypeUtils.classType(className.toString(),
                            type, typeargs);
                    if (l.token(0).kind == TokenKind.COLCOL) {
                        if (l.token(1).kind == TokenKind.LT) {
                            // class type variable
                            l.nextToken();
                            type = parseTypeVariableRest(type, l);
                            break;
                        } else if (l.token(1).kind == TokenKind.IDENTIFIER) {
                            if (l.token(2).kind == TokenKind.LPAREN || l.token(2).kind == TokenKind.COLON) {
                                // this looks like the middle of a field/method reference -- stop consuming
                                break;
                            }
                            l.nextToken(); // inner type, keep going
                        }
                    } else {
                        // not an inner type
                        break;
                    }
                }
            }
        }
        while (l.token().kind == TokenKind.LBRACKET) {
            l.accept(TokenKind.LBRACKET);
            l.accept(TokenKind.RBRACKET);
            type = JavaTypeUtils.arrayType(type);
        }
        return type;
    }

    public static ExternalizedTypeElement parseTypeVariableRest(ExternalizedTypeElement owner, Lexer l) {
        l.accept(TokenKind.LT);
        String name = l.token().name();
        l.nextToken();
        ExternalizedTypeElement bound = JavaType.J_L_OBJECT.externalize();
        if (l.token().kind == TokenKind.IDENTIFIER &&
                l.token().name().equals("extends")) {
            l.nextToken();
            bound = parseJavaType(l);
        }
        l.accept(TokenKind.GT);
        return JavaTypeUtils.typeVarType(name, owner, bound);
    }

    public static ExternalizedTypeElement parseTypeArgument(Lexer l) {
        if (l.token().kind == TokenKind.QUES) {
            // wildcard
            l.nextToken();
            ExternalizedTypeElement bound = JavaType.J_L_OBJECT.externalize();
            WildcardType.BoundKind bk = BoundKind.EXTENDS;
            if (l.token().kind == TokenKind.IDENTIFIER) {
                bk = switch (l.token().name()) {
                    case "extends" -> BoundKind.EXTENDS;
                    case "super" -> BoundKind.SUPER;
                    default -> throw new IllegalArgumentException("Bad wildcard bound");
                };
                l.nextToken();
                bound = parseJavaType(l);
            }
            return JavaTypeUtils.wildcardType(bk, bound);
        } else {
            return parseJavaType(l);
        }
    }

    static List<ExternalizedTypeElement> parseParameterTypes(Lexer l) {
        List<ExternalizedTypeElement> ptypes = new ArrayList<>();
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

    //    JavaRef:
    //        JavaType `::` ident ':' JavaType                      // field reference
    //        JavaType `::` ident '(' JavaType* ')' ':' JavaType    // method reference
    //        JavaType `::` '(' JavaType* ')'                       // constructor reference
    //        '(' RecordComponent* ')' JavaType                     // record reference
    //
    //    RecordComponent:
    //        JavaType ident
    static ExternalizedTypeElement parseJavaRef(Lexer l) {
        if (l.acceptIf(TokenKind.LPAREN)) {
            // record type reference
            List<String> componentNames = new ArrayList<>();
            List<ExternalizedTypeElement> componentTypes = new ArrayList<>();
            if (l.token().kind != Tokens.TokenKind.RPAREN) {
                do {
                    componentTypes.add(parseJavaType(l));
                    componentNames.add(l.accept(Tokens.TokenKind.IDENTIFIER).name());
                } while(l.acceptIf(Tokens.TokenKind.COMMA));
            }
            l.accept(Tokens.TokenKind.RPAREN);
            ExternalizedTypeElement recordType = parseJavaType(l);
            return JavaTypeUtils.recordRef(recordType, componentNames, componentTypes);
        }
        ExternalizedTypeElement refType = parseJavaType(l);

        l.accept(Tokens.TokenKind.COLCOL);
        if (l.token().kind == TokenKind.LPAREN) {
            // constructor ref
            List<ExternalizedTypeElement> ptypes = parseParameterTypes(l);
            return JavaTypeUtils.constructorRef(refType, ptypes);
        }

        // field or method ref
        String memberName = l.accept(Tokens.TokenKind.IDENTIFIER).name();
        if (l.token().kind == TokenKind.LPAREN) {
            // method ref
            List<ExternalizedTypeElement> params = parseParameterTypes(l);
            l.accept(TokenKind.COLON);
            ExternalizedTypeElement rtype = parseJavaType(l);
            return JavaTypeUtils.methodRef(memberName, refType, rtype, params);
        } else {
            // field ref
            l.accept(TokenKind.COLON);
            ExternalizedTypeElement ftype = parseJavaType(l);
            return JavaTypeUtils.fieldRef(memberName, refType, ftype);
        }
    }
}
